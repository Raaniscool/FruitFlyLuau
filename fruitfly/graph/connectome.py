"""Directed weighted connectome in Compressed Sparse Row form.

Design notes (performance)
--------------------------
* One ``Connectome`` = ``n`` consecutive integer neuron indices ``0..n-1`` plus a
  CSR adjacency built from NumPy arrays. No per-neuron Python objects, ever.
* Synaptic *weights* live in the CSR ``data`` array. Plasticity writes into the
  same array (see ``PlasticityRule``), so learning never needs a dict of edges.
* ``pre_ids`` / ``post_ids`` keep the original FlyWire root ids (int64), so any
  spike or weight can be traced back to a real neuron id in the published data.
* A CSC view is built lazily for ``in_neurons``/upstream queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class Connectome:
    """Sparse directed weighted graph over a selected neuron population.

    Attributes
    ----------
    root_ids:
        int64 array, position ``i`` is the FlyWire root id of simulated neuron ``i``.
    labels:
        Optional display names aligned with ``root_ids`` (never used for computation).
    matrix:
        CSR matrix ``W`` with ``W[i, j] > 0`` meaning "neuron i sends weight w to
        neuron j" (row = presynaptic source, column = postsynaptic target), so the
        per-step input current is ``W.T @ spikes``.
    """

    root_ids: np.ndarray
    matrix: sparse.csr_matrix
    labels: np.ndarray | None = None
    synapse_counts: np.ndarray | None = None  # raw syn_count per edge, pre-plasticity
    nt_types: np.ndarray | None = None  # per-neuron transmitter label (string)
    ei_sign: np.ndarray | None = None  # per-neuron sign applied to outgoing edges
    coords: np.ndarray | None = None  # (n, 3) float32 nm anchor positions, NaN if unknown
    neuropil_per_edge: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- basics
    @property
    def n_neurons(self) -> int:
        """Number of neurons ACTUALLY simulated. Reported everywhere; never inflated."""
        return int(self.root_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.matrix.nnz)

    @property
    def weights(self) -> np.ndarray:
        """Direct handle on the CSR data array (mutable; this is the learning state)."""
        return self.matrix.data

    def index_of(self, root_id: int) -> int | None:
        """Local index of a FlyWire root id, or ``None``."""
        hit = np.flatnonzero(self.root_ids == int(root_id))
        return int(hit[0]) if hit.size else None

    def indices_of(self, root_ids: np.ndarray) -> np.ndarray:
        """Vectorised root-id -> local-index mapping; missing ids become -1."""
        pos = np.searchsorted(self.root_ids, root_ids)
        pos = np.clip(pos, 0, self.n_neurons - 1)
        ok = self.root_ids[pos] == root_ids
        out = np.where(ok, pos, -1).astype(np.int64)
        return out

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """(in_degree, out_degree) in number of edges."""
        out_deg = np.diff(self.matrix.indptr).astype(np.int64)
        in_deg = np.asarray((self.matrix != 0).sum(axis=0)).ravel().astype(np.int64)
        return in_deg, out_deg

    @property
    def csc(self) -> sparse.csc_matrix:
        cached = getattr(self, "_csc", None)
        if cached is None or cached.nnz != self.matrix.nnz:
            cached = self.matrix.tocsc()
            object.__setattr__(self, "_csc", cached)
        return cached

    def row_slice(self, i: int) -> np.ndarray:
        """Postsynaptic targets of neuron ``i``."""
        return self.matrix.indices[self.matrix.indptr[i] : self.matrix.indptr[i + 1]]

    def col_slice(self, j: int) -> np.ndarray:
        """Presynaptic sources onto neuron ``j``."""
        c = self.csc
        return c.indices[c.indptr[j] : c.indptr[j + 1]]

    # -------------------------------------------------------------- operators
    def subgraph(self, keep: np.ndarray, *, renumber: bool = True) -> "Connectome":
        """Induced subgraph on ``keep`` (array of local indices, any order)."""
        keep = np.asarray(keep, dtype=np.int64)
        if keep.size == 0:
            raise ValueError("subgraph needs at least one neuron")
        if keep.min() < 0 or keep.max() >= self.n_neurons:
            raise IndexError("subgraph index outside population range")
        sub = self.matrix[keep][:, keep].tocsr()
        labels = self.labels[keep] if self.labels is not None else None
        nt = self.nt_types[keep] if self.nt_types is not None else None
        sign = self.ei_sign[keep] if self.ei_sign is not None else None
        coords = self.coords[keep] if self.coords is not None else None
        return Connectome(
            root_ids=self.root_ids[keep].copy(),
            matrix=sub,
            labels=labels,
            synapse_counts=np.asarray(sub.data, dtype=np.float64).copy(),
            nt_types=nt,
            ei_sign=sign,
            coords=coords,
            provenance={**self.provenance, "derived_from": "subgraph", "n_kept": int(keep.size)},
        )

    def copy_weights(self) -> np.ndarray:
        return self.matrix.data.copy()

    def restore_weights(self, data: np.ndarray) -> None:
        if data.shape != self.matrix.data.shape:
            raise ValueError(f"weight array shape {data.shape} != CSR data {self.matrix.data.shape}")
        self.matrix.data[:] = data
        object.__setattr__(self, "_csc", None)

    def set_neuron_current_indices(self, idx: np.ndarray) -> None:  # pragma: no cover
        """Convenience used by encoders: validate a neuron index array."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_neurons):
            raise IndexError("neuron index outside population")
        return idx

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Persist structure + weights + ids as one ``.npz`` (no dataset needed to resume)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray | str] = {
            "root_ids": self.root_ids,
            "indptr": self.matrix.indptr,
            "indices": self.matrix.indices,
            "data": self.matrix.data,
            "meta_json": np.array([json.dumps(self.provenance, sort_keys=True, default=str)]),
        }
        if self.labels is not None:
            payload["labels"] = self.labels.astype(str)
        if self.synapse_counts is not None:
            payload["synapse_counts"] = self.synapse_counts
        if self.nt_types is not None:
            payload["nt_types"] = self.nt_types.astype(str)
        if self.ei_sign is not None:
            payload["ei_sign"] = self.ei_sign
        if self.coords is not None:
            payload["coords"] = self.coords
        np.savez_compressed(path, **payload)
        return path

    @staticmethod
    def load(path: str | Path) -> "Connectome":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            root_ids = np.asarray(z["root_ids"], dtype=np.int64)
            n = int(root_ids.size)
            matrix = sparse.csr_matrix((np.asarray(z["data"]), np.asarray(z["indices"]), np.asarray(z["indptr"])), shape=(n, n))
            meta: dict = {}
            if "meta_json" in z.files:
                meta = json.loads(str(z["meta_json"][0]))
            return Connectome(
                root_ids=root_ids,
                matrix=matrix,
                labels=z["labels"] if "labels" in z.files else None,
                synapse_counts=z["synapse_counts"] if "synapse_counts" in z.files else None,
                nt_types=z["nt_types"] if "nt_types" in z.files else None,
                ei_sign=z["ei_sign"] if "ei_sign" in z.files else None,
                coords=z["coords"] if "coords" in z.files else None,
                provenance=meta,
            )

    # ---------------------------------------------------------------- summary
    def stats(self) -> dict:
        in_deg, out_deg = self.degrees()
        w = self.matrix.data
        return {
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "density": float(self.n_edges / max(1, self.n_neurons * self.n_neurons)),
            "mean_in_degree": float(in_deg.mean()),
            "median_in_degree": float(np.median(in_deg)),
            "max_in_degree": int(in_deg.max()) if in_deg.size else 0,
            "mean_out_degree": float(out_deg.mean()),
            "median_out_degree": float(np.median(out_deg)),
            "max_out_degree": int(out_deg.max()) if out_deg.size else 0,
            "isolated_neurons": int(((in_deg == 0) & (out_deg == 0)).sum()),
            "weight_min": float((w.min() if w.size else 0.0)),
            "weight_mean": float((float(w.mean()) if w.size else 0.0)),
            "weight_max": float((w.max() if w.size else 0.0)),
            "n_self_connections": int((self.matrix.diagonal() != 0).sum()),
            "reciprocal_pair_fraction": float(self._reciprocity()),
            **self.excitation_balance(),
        }

    def excitation_balance(self) -> dict:
        """How much of this subgraph is actually inhibitory.

        Asked because the first real mushroom-body run was bistable between seizure
        (100% active at ~430 Hz) and silence (6% active), which is what a recurrent graph
        does when almost every edge is excitatory. FAFB is ACh-dominated, so the answer
        is expected to be low -- but expected is not measured, so it is reported.

        ``inhibitory_edge_fraction`` is the fraction of edges with negative weight;
        ``mean_signed_weight`` is the average signed edge weight, i.e. the net drive one
        spike delivers. Both are properties of OUR sign mapping applied to the data, not
        facts about the fly.
        """
        w = np.asarray(self.matrix.data)
        if w.size == 0:
            return {"inhibitory_edge_fraction": 0.0, "excitatory_edge_fraction": 0.0,
                    "silent_edge_fraction": 0.0, "mean_signed_weight": 0.0,
                    "ei_balance_note": "no edges"}
        neg = int((w < 0).sum())
        pos = int((w > 0).sum())
        zero = int((w == 0).sum())
        out = {
            "inhibitory_edge_fraction": float(neg / w.size),
            "excitatory_edge_fraction": float(pos / w.size),
            "silent_edge_fraction": float(zero / w.size),
            "mean_signed_weight": float(w.mean()),
        }
        if self.nt_types is not None:
            labels, counts = np.unique(np.asarray(self.nt_types, dtype=str), return_counts=True)
            out["neurons_by_transmitter"] = {str(k): int(v) for k, v in zip(labels, counts)}
        if out["inhibitory_edge_fraction"] < 0.05:
            out["ei_balance_note"] = (
                f"only {100 * out['inhibitory_edge_fraction']:.1f}% of edges are inhibitory: "
                "a recurrent graph this excitatory has no stable middle regime, and the "
                "simulation will tend to sit at either silence or saturation. Consider "
                "lif.global_inhibition."
            )
        else:
            out["ei_balance_note"] = "inhibitory fraction is within a range that can stabilise recurrence"
        return out

    def _reciprocity(self) -> float:
        """Fraction of edges whose reverse edge also exists (network property, not a model)."""
        if self.n_edges == 0:
            return 0.0
        m = (self.matrix != 0).astype(np.int8)
        both = m.multiply(m.T).nnz
        return float(both / (2 * self.n_edges)) if self.n_edges else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Connectome(neurons={self.n_neurons:,}, edges={self.n_edges:,}, "
            f"density={self.stats()['density']:.2e})"
        )


def build_connectome(
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    weights: np.ndarray,
    root_ids: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    nt_types: np.ndarray | None = None,
    ei_sign: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    neuropil_per_edge: np.ndarray | None = None,
    drop_self: bool = True,
    provenance: dict | None = None,
    merge_duplicates: str = "sum",
) -> Connectome:
    """Assemble a CSR connectome from parallel edge arrays.

    Parameters
    ----------
    pre_idx, post_idx:
        int64 local indices of equal length.
    weights:
        float64 (or int) edge weights; NaN/inf rows are dropped and counted.
    root_ids:
        int64 array defining the population; ``n = len(root_ids)``.
    merge_duplicates:
        "sum" aggregates repeated (pre, post) pairs across neuropils (the
        filtered connections file legitimately has several rows per pair).
    """
    pre_idx = np.asarray(pre_idx, dtype=np.int64)
    post_idx = np.asarray(post_idx, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    n = int(root_ids.size)
    if pre_idx.shape != post_idx.shape or pre_idx.shape != w.shape:
        raise ValueError(f"edge array length mismatch: {pre_idx.shape} {post_idx.shape} {w.shape}")

    n_before = pre_idx.size
    valid = (
        np.isfinite(w)
        & (pre_idx >= 0)
        & (post_idx >= 0)
        & (pre_idx < n)
        & (post_idx < n)
    )
    pre_idx, post_idx, w = pre_idx[valid], post_idx[valid], w[valid]
    if neuropil_per_edge is not None:
        neuropil_per_edge = np.asarray(neuropil_per_edge, dtype=object)[valid]
    n_dropped_invalid = n_before - pre_idx.size

    if drop_self:
        keep = pre_idx != post_idx
        n_self = int((~keep).sum())
        pre_idx, post_idx, w = pre_idx[keep], post_idx[keep], w[keep]
        if neuropil_per_edge is not None:
            neuropil_per_edge = neuropil_per_edge[keep]
    else:  # pragma: no cover - autapses already absent in the filtered file
        n_self = 0

    if pre_idx.size == 0:
        raise ValueError("no valid edges after filtering; check the neuron population selection")

    matrix = sparse.coo_matrix((w, (pre_idx, post_idx)), shape=(n, n)).tocsr()
    if merge_duplicates == "sum":
        matrix.sum_duplicates()
    else:
        raise ValueError(f"unsupported merge_duplicates={merge_duplicates!r}")
    # coo->csr sum_duplicates() leaves explicit zeros if input had them; prune.
    matrix.eliminate_zeros()

    conn = Connectome(
        root_ids=np.asarray(root_ids, dtype=np.int64).copy(),
        matrix=matrix,
        labels=labels,
        synapse_counts=np.asarray(matrix.data, dtype=np.float64).copy(),
        nt_types=nt_types,
        ei_sign=ei_sign,
        coords=coords,
        neuropil_per_edge=neuropil_per_edge,
        provenance={
            "n_edges_input": int(n_before),
            "n_edges_dropped_invalid": int(n_dropped_invalid),
            "n_self_connections_dropped": int(n_self),
            "n_edges_built": int(matrix.nnz),
            **(provenance or {}),
        },
    )
    log.debug("built %r", conn)
    return conn
