"""Turn a config into a simulated neuron population (the connectome we run on)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from ..config import AppConfig
from ..dataset.discover import DatasetInventory, discover

# NOTE: fruitfly.dataset and fruitfly.graph import each other, so dataset
# modules that depend on graph.connectome are imported lazily inside functions.
from ..utils import get_logger
from .connectome import Connectome, build_connectome

log = get_logger(__name__)


@dataclass
class Population:
    """A connectome plus the bookkeeping needed to report honestly what it is."""

    connectome: Connectome
    mode: str
    scale_label: str
    requested_neurons: int
    simulated_neurons: int
    source_description: str
    annotations_loaded: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    connection_table_stats: dict = field(default_factory=dict)

    def describe(self) -> str:
        c = self.connectome
        s = c.stats()
        lines = [
            f"scale label      : {self.scale_label}",
            f"requested        : {self.requested_neurons:,} neurons",
            f"SIMULATED        : {self.simulated_neurons:,} neurons   <-- what actually runs",
            f"edges            : {c.n_edges:,}",
            f"density          : {s['density']:.3e}",
            f"mean/median in   : {s['mean_in_degree']:.1f} / {s['median_in_degree']:.1f}",
            f"mean/median out  : {s['mean_out_degree']:.1f} / {s['median_out_degree']:.1f}",
            f"isolated neurons : {s['isolated_neurons']:,}",
            f"reciprocal pairs : {s['reciprocal_pair_fraction']:.3f}",
            f"weight min/mean/max: {s['weight_min']:.3f} / {s['weight_mean']:.3f} / {s['weight_max']:.3f}",
            f"source           : {self.source_description}",
        ]
        if self.simulated_neurons < 100_000:
            lines.append(
                f"NOTE: this is a {self.simulated_neurons:,}-neuron SUBSET of FAFB v783 "
                "(139,255 neurons), not 'the fly brain'."
            )
        return "\n".join(lines)


#: Documented scale labels. "full" only ever applies to the whole loaded graph.
SCALES = {"tiny": 100, "small": 1_000, "medium": 10_000, "large": 40_000}


def resolve_scale(cfg: AppConfig) -> tuple[str, int]:
    mode = cfg.graph.mode.lower().strip()
    if mode == "full":
        return "full", 0
    if mode in {"sample", "synthetic"}:
        return "sample", cfg.graph.n_neurons or SCALES["tiny"]
    if mode in SCALES:
        return mode, cfg.graph.n_neurons or SCALES[mode]
    # explicit numeric mode, e.g. mode="1000"
    try:
        n = int(mode)
        return f"custom:{n}", n
    except ValueError:
        raise ValueError(f"unknown graph.mode {cfg.graph.mode!r} (tiny|small|medium|large|full|<int>)") from None


def _load_table(cfg: AppConfig, data_dir: Path | None, inventory: DatasetInventory | None) -> ConnectionTable:
    from ..dataset.loader import load_connections

    if data_dir is None:
        raise FileNotFoundError("no data directory resolved; see fruitfly/paths.py error message")
    cache = None
    if cfg.data.use_cache:
        cache = Path(cfg.data.cache_dir)
        if not cache.is_absolute():
            cache = Path(__file__).resolve().parents[2] / cache
        cache = cache / f"{cfg.data.source}_{cfg.data.max_rows or 'all'}_min{cfg.data.min_synapses_per_pair}.npz"
    return load_connections(
        data_dir,
        source=cfg.data.source,
        chunksize=cfg.data.read_chunksize,
        max_rows=cfg.data.max_rows,
        min_synapses_per_pair=cfg.data.min_synapses_per_pair,
        cache_path=cache,
        inventory=inventory,
    )


def _adjacency_over_all(table: ConnectionTable) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Full loaded graph as CSR over sorted unique ids (needed by subgraph selectors)."""
    ids = np.union1d(table.pre, table.post)
    pre = np.searchsorted(ids, table.pre)
    post = np.searchsorted(ids, table.post)
    adj = sparse.csr_matrix(
        (np.ones(table.pre.size, np.float32), (pre, post)), shape=(ids.size, ids.size)
    )
    adj.sum_duplicates()
    return ids, adj


def build_population(cfg: AppConfig, *, seed: int | None = None) -> Population:
    """Build the :class:`Connectome` for one experiment run.

    ``graph.mode="sample"`` (or ``data.source="sample"``) skips FAFB entirely and
    returns a synthetic small-world graph, so the whole pipeline is testable
    without the 68 MB download. The result is always labelled as synthetic.
    """
    t_all = time.perf_counter()
    rng = np.random.default_rng(cfg.train.seed if seed is None else seed)
    scale, n_request = resolve_scale(cfg)
    use_sample = cfg.graph.mode.lower() in {"sample", "synthetic"} or cfg.data.source.lower() == "sample"

    if use_sample:
        from ..dataset.build_sample import sample_connectome

        conn, meta = sample_connectome(
            n=n_request or SCALES["tiny"], rng=rng, ei_mode=cfg.graph.ei_mode,
            weight_transform=cfg.graph.weight_transform, weight_scale=cfg.graph.weight_scale,
            selection=cfg.graph.selection, **cfg.graph.selection_kwargs,
        )
        return Population(
            connectome=conn, mode=cfg.graph.mode, scale_label=f"{scale} (synthetic sample)",
            requested_neurons=n_request or SCALES["tiny"], simulated_neurons=conn.n_neurons,
            source_description="SYNTHETIC sample graph (not FlyWire data) - generated by fruitfly.dataset.build_sample",
            annotations_loaded=[], timings={"total": time.perf_counter() - t_all},
            connection_table_stats={"synthetic": True, "n_edges": conn.n_edges},
        )

    from ..paths import require_data_dir

    data_dir = require_data_dir(cfg.data.fafb_data_path or None)
    inventory = discover(data_dir)
    t0 = time.perf_counter()
    table = _load_table(cfg, data_dir, inventory)
    t_load = time.perf_counter() - t0

    ids, adj = _adjacency_over_all(table)
    metadata: dict[str, np.ndarray] = {}
    if table.nt_type is not None:
        # per-neuron transmitter from the edge table (dominant label), as a
        # fallback when neurons.csv.gz is absent
        from ..dataset.loader import annotate_neuron_metadata  # noqa: F401

        nt_edge = np.asarray(table.nt_type, dtype=object)
        per_neuron = np.empty(ids.size, dtype=object)
        per_neuron[:] = "unknown"
        pre_pos = np.searchsorted(ids, table.pre)
        np.put(per_neuron, pre_pos, nt_edge)  # type: ignore[arg-type]
        metadata["__edge_derived_nt"] = per_neuron
    try:
        from ..dataset.loader import annotate_neuron_metadata

        metadata.update(annotate_neuron_metadata(data_dir, ids, inventory=inventory))
    except Exception as exc:  # annotations are optional by design
        log.warning("annotation loading skipped: %s", exc)
    if "__edge_derived_nt" in metadata and "nt_predictions.nt_type" not in metadata:
        metadata["nt_predictions.nt_type"] = metadata.pop("__edge_derived_nt")
    else:
        metadata.pop("__edge_derived_nt", None)

    t0 = time.perf_counter()
    from ..graph.select import select_population
    from ..dataset.loader import ei_signs, transmitter_labels

    if scale == "full":
        keep = np.arange(ids.size, dtype=np.int64)
    else:
        keep = select_population(
            cfg.graph.selection, ids=ids, adjacency=adj, metadata=metadata, rng=rng,
            kwargs={**cfg.graph.selection_kwargs, "n": n_request},
        )
    t_select = time.perf_counter() - t0

    keep = np.sort(keep)
    pop_ids = ids[keep]
    # extract the induced subgraph weights from the *table* (so pair merging and
    # thresholds already applied are respected)
    code = -np.ones(ids.size, dtype=np.int64)
    code[keep] = np.arange(keep.size, dtype=np.int64)
    pre_c = code[np.searchsorted(ids, table.pre)]
    post_c = code[np.searchsorted(ids, table.post)]
    inside = (pre_c >= 0) & (post_c >= 0)
    w = table.weight[inside]
    if cfg.graph.weight_transform == "log1p":
        w = np.log1p(np.maximum(w, 0.0))
    elif cfg.graph.weight_transform == "sqrt":
        w = np.sqrt(np.maximum(w, 0.0))
    elif cfg.graph.weight_transform == "zscore":
        mu, sd = float(w.mean()), float(w.std())
        w = (w - mu) / (sd or 1.0)
    elif cfg.graph.weight_transform not in {"none", ""}:
        raise ValueError(f"unknown weight_transform {cfg.graph.weight_transform!r}")
    w = w * float(cfg.graph.weight_scale)

    nt_labels = transmitter_labels(metadata.get("nt_predictions.nt_type")[keep] if metadata.get("nt_predictions.nt_type") is not None else None, keep.size)
    signs = ei_signs(nt_labels) if cfg.graph.ei_mode == "nt" else np.ones(keep.size, np.float32)
    coords = metadata.get("coordinates.xyz")
    coords_pop = coords[keep] if coords is not None and np.ndim(coords) == 2 else None

    labels = None
    name_col = metadata.get("names_groups.name")
    if name_col is not None:
        labels = np.asarray([("" if v is None else str(v)) for v in name_col[keep]], dtype=object)

    t0 = time.perf_counter()
    conn = build_connectome(
        pre_c[inside], post_c[inside], w, pop_ids,
        labels=labels, nt_types=nt_labels, ei_sign=signs, coords=coords_pop,
        provenance={
            "source_asset": cfg.data.source, "source_file": table.source_file,
            "data_dir": str(data_dir), "scale": scale,
            "min_synapses_per_pair": cfg.data.min_synapses_per_pair,
            "weight_transform": cfg.graph.weight_transform,
            "weight_scale": cfg.graph.weight_scale, "ei_mode": cfg.graph.ei_mode,
            "selection": cfg.graph.selection, "selection_kwargs": cfg.graph.selection_kwargs,
            "n_pairs_in_loaded_table": int(table.pre.size),
            "edge_fraction_kept_in_subset": float(inside.mean()),
        },
    )
    t_build = time.perf_counter() - t0

    return Population(
        connectome=conn, mode=cfg.graph.mode, scale_label=scale,
        requested_neurons=n_request or int(ids.size), simulated_neurons=conn.n_neurons,
        source_description=f"FAFB v783 {cfg.data.source} ({table.source_file}) via selection={cfg.graph.selection}",
        annotations_loaded=sorted({k.split('.')[0] for k in metadata}),
        timings={"load_connections": t_load, "select": t_select, "build": t_build, "total": time.perf_counter() - t_all},
        connection_table_stats=table.to_dict(),
    )
