"""Synthetic stand-ins for FAFB data.

Two purposes, both clearly labelled as non-biological:

1. :func:`sample_connectome` builds a *fly-inspired random graph* (heavy-tailed
   degrees, log-normal synaptic weights, E/I mixture, local clustering) so the
   simulator and experiment pipeline run with zero external data.
2. :func:`write_sample_fafb_files` writes tiny files that use the *real FAFB
   v783 column names and formats* (``connections_princeton.csv.gz``,
   ``neurons.csv.gz``, ...) so loader/discovery code can be tested against the
   documented schema without shipping 68 MB of biology in Git.

The synthetic graph is NOT a substitute for real connectivity in any scientific
claim: results on it demonstrate that the machinery works, nothing more.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import get_logger
from ..graph.connectome import build_connectome

log = get_logger(__name__)

#: Root ids in FAFB v783 share the prefix 720575940 followed by 9 digits.
ROOT_ID_PREFIX = 720575940

NT_TOKENS = np.array(["glut", "cholin", "gaba", "glut/gaba", "da", "ser", "oct"], dtype=object)
NT_PROBS = np.array([0.36, 0.28, 0.19, 0.08, 0.03, 0.03, 0.03])
NEUROPIILS = ["MB_CA_L", "MB_CA_R", "LOP_R", "LOP_L", "ME_R", "ME_L", "BU_L", "BU_R", "ELP", "PB", "FB", "GNG"]


def _fly_inspired_edges(n: int, rng: np.random.Generator, *, mean_out_degree: float = 27.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heavy-tailed, locally-clustered directed edges.

    Degree distribution: geometric with the published mean out-degree of FAFB
    v783 (~3.73M pairs / 139,255 neurons ~ 26.8). Targets are biased towards
    index-proximity to create community structure (index here plays the role of
    spatial position; there is no anatomical claim).
    """
    k = 1.0 / max(mean_out_degree - 1.0, 1e-3)
    out_deg = rng.geometric(k, size=n).astype(np.int64)
    out_deg = np.clip(out_deg, 1, max(1, n - 1))
    total = int(out_deg.sum())

    src = np.repeat(np.arange(n, dtype=np.int64), out_deg)
    # Laplace-distributed offsets => many short edges, a few long ones
    offsets = (np.sign(rng.random(total) - 0.5) * np.abs(rng.laplace(0.0, max(2.0, n / 25.0), size=total))).astype(np.int64)
    dst = (src + offsets) % n
    src = src.astype(np.int64); dst = dst.astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]

    # log-normal synapse counts, integer, clipped to the documented 1..2633 range
    counts = np.exp(rng.normal(1.1, 0.9, size=src.size))
    counts = np.clip(np.rint(counts), 1, 2633).astype(np.int64)
    return src, dst, counts


def sample_connectome(
    n: int = 100,
    *,
    rng: np.random.Generator | None = None,
    ei_mode: str = "nt",
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
    selection: str = "connected_subgraph",
    **_: object,
) -> tuple[object, dict]:
    """Return ``(Connectome, meta)`` for a synthetic fly-inspired population."""
    rng = rng or np.random.default_rng(0)
    n = int(max(4, n))
    src, dst, counts = _fly_inspired_edges(n, rng)

    # Connect the largest component to a ring so 'connected_subgraph' has work to do
    ring = np.arange(n, dtype=np.int64)
    src = np.concatenate([src, ring])
    dst = np.concatenate([dst, (ring + 1) % n])
    counts = np.concatenate([counts, np.ones(n, np.int64)])

    if weight_transform == "log1p":
        w = np.log1p(counts.astype(np.float64))
    elif weight_transform == "sqrt":
        w = np.sqrt(counts.astype(np.float64))
    else:
        w = counts.astype(np.float64)
    w = w * float(weight_scale)

    nt = rng.choice(NT_TOKENS, size=n, p=NT_PROBS)
    from ..dataset.loader import ei_signs

    signs = ei_signs(nt) if ei_mode == "nt" else np.ones(n, np.float32)

    # pseudo root ids with the real prefix, so id-format handling is exercised
    # unique pseudo root ids with the real 720575940_XXXXXXXXX shape
    ids = (ROOT_ID_PREFIX * 10**9 + np.sort(rng.choice(10**8, size=n, replace=False))).astype(np.int64)

    conn = build_connectome(
        src, dst, w, ids,
        nt_types=np.asarray(nt, dtype=object), ei_sign=signs,
        labels=np.array([f"synthetic_{i:04d}" for i in range(n)], dtype=object),
        coords=np.column_stack([
            rng.normal(5e5, 1.2e5, n), rng.normal(3e5, 1.2e5, n), rng.normal(1.4e5, 8e4, n)
        ]).astype(np.float32),
        provenance={
            "synthetic": True,
            "generator": "fruitfly.dataset.build_sample.sample_connectome",
            "note": (
                "Random graph with fly-inspired degree/weight statistics. NOT FlyWire data. "
                "Use only to exercise the simulator, plasticity and experiment plumbing."
            ),
            "weight_transform": weight_transform, "weight_scale": weight_scale, "ei_mode": ei_mode,
        },
    )
    meta = {"n": n, "mean_out_degree": float(conn.n_edges / max(n, 1))}
    return conn, meta


# ------------------------------------------------------------- sample FAFB files
def write_sample_fafb_files(
    out_dir: str | Path,
    *,
    n_neurons: int = 640,
    mean_out_degree: float = 12.0,
    seed: int = 20260923,
    compress: bool = True,
) -> dict[str, int]:
    """Write tiny files with *real* FAFB v783 asset names/columns (synthetic content).

    Returns filename -> row count. Used by the test-suite and by
    ``fruitfly sample --write-fafb`` to give users a working data directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}

    ids = (ROOT_ID_PREFIX * 10**9 + np.sort(rng.choice(10**8, size=n_neurons, replace=False))).astype(np.int64)
    src, dst, syn = _fly_inspired_edges(n_neurons, rng, mean_out_degree=mean_out_degree)
    # spread each pair over 1-3 "neuropils" so the per-neuropil row structure exists
    n_rows = src.size
    reps = rng.integers(1, 4, size=n_rows)
    pre = np.repeat(ids[src], reps)
    post = np.repeat(ids[dst], reps)
    neuropil = rng.choice(NEUROPIILS, size=pre.size)
    share = np.maximum(1, syn // np.maximum(reps, 1))
    per_row = np.repeat(share, reps)
    nt_edge = rng.choice(np.array(["glut", "cholin", "gaba", "glut/gaba", "da"]), size=pre.size, p=[0.4, 0.3, 0.2, 0.07, 0.03])
    conn_df = pd.DataFrame(
        {"pre_root_id": pre, "post_root_id": post, "neuropil": neuropil, "syn_count": per_row, "nt_type": nt_edge}
    )
    # keep only pairs totalling >=5 synapses, mirroring the filtered asset's rule
    tot = conn_df.groupby(["pre_root_id", "post_root_id"])["syn_count"].transform("sum")
    conn_df = conn_df[tot >= 5].reset_index(drop=True)
    _write(conn_df, out / "connections_princeton.csv.gz", compress, counts)

    # unfiltered twin (superset), for source-switching tests
    conn_unf = conn_df.copy()
    extra = conn_df.head(max(1, len(conn_df) // 10)).copy()
    extra["syn_count"] = 1
    conn_unf = pd.concat([conn_unf, extra], ignore_index=True)
    _write(conn_unf, out / "connections_princeton_no_threshold.csv.gz", compress, counts)

    nt_neuron = pd.DataFrame({
        "root_id": ids,
        "group": rng.choice([f"grp{i:03d}" for i in range(24)], size=n_neurons),
        "nt_type": np.asarray(rng.choice(np.array(["glut", "cholin", "gaba", "glut/gaba", "da", "ser", None], dtype=object), size=n_neurons, p=[.34, .28, .19, .08, .04, .04, .03]), dtype=object),
        "nt_type_score": np.round(rng.uniform(0.3, 1.0, n_neurons), 3),
        "da_avg": np.round(rng.uniform(0, 0.1, n_neurons), 3),
        "ser_avg": np.round(rng.uniform(0, 0.1, n_neurons), 3),
        "gaba_avg": np.round(rng.uniform(0, 1, n_neurons), 3),
        "glut_avg": np.round(rng.uniform(0, 1, n_neurons), 3),
        "ach_avg": np.round(rng.uniform(0, 1, n_neurons), 3),
        "oct_avg": np.round(rng.uniform(0, 0.05, n_neurons), 3),
    })
    _write(nt_neuron, out / "neurons.csv.gz", compress, counts)

    classes = ["Kenyon", "MBON", "TmNeuron", "LCNeuron", "DN", "ShNeuron", "Dopaminergic", "Projection"]
    cls_df = pd.DataFrame({
        "root_id": ids,
        "flow": rng.choice(["ascending", "local", "projection", "descending"], size=n_neurons),
        "super_class": rng.choice(["Central neurons", "Sensory neurons", "Neuromodulatory", "Efferents"], size=n_neurons),
        "class": rng.choice(classes, size=n_neurons),
        "sub_class": rng.choice(["wide-field", "columnar", "layer-specific"], size=n_neurons),
        "hemilineage": rng.choice([f"BLA{ i:03d}" for i in range(16)], size=n_neurons),
        "side": rng.choice(["left", "right", "bilaterial"], size=n_neurons, p=[0.45, 0.45, 0.1]),
        "nerve": rng.choice([None, "NVC", "ION"], size=n_neurons, p=[0.85, 0.1, 0.05]),
    })
    _write(cls_df, out / "classification.csv.gz", compress, counts)

    _write(
        pd.DataFrame({
            "root_id": ids,
            "primary_type": rng.choice(classes, size=n_neurons),
            "additional_types": rng.choice([None, "T1", "T2a"], size=n_neurons, p=[0.7, 0.2, 0.1]),
        }),
        out / "consolidated_cell_types.csv.gz", compress, counts,
    )
    _write(
        pd.DataFrame({
            "root_id": ids,
            "length_nm": rng.integers(4_930, 103_733_576, n_neurons),
            "area_nm": rng.integers(15_141_888, 258_113_806_848, n_neurons),
            "size_nm": rng.integers(427_591_680, 16_416_917_166_080, n_neurons),
        }),
        out / "cell_stats.csv.gz", compress, counts,
    )
    _write(
        pd.DataFrame({
            "root_id": ids,
            "name": [f"synthetic{i:04d}_grp{rng.integers(0, 24)}" for i in range(n_neurons)],
            "group": rng.choice([f"grp{i:03d}" for i in range(24)], size=n_neurons),
        }),
        out / "names.csv.gz", compress, counts,
    )
    # Marked Neuron Coordinates: position encoded as "x,y,z" string, >1 row per neuron
    n_coord = int(n_neurons * 1.7)
    coord_ids = rng.choice(ids, size=n_coord, replace=True)
    _write(
        pd.DataFrame({
            "root_id": coord_ids,
            "position": [f"{x},{y},{z}" for x, y, z in rng.integers(90_000, 900_000, size=(n_coord, 3))],
            "supervoxel_id": rng.integers(1, 10**9, n_coord),
        }),
        out / "coordinates.csv.gz", compress, counts,
    )
    _write(
        pd.DataFrame({"root_id": rng.choice(ids, size=n_neurons // 2), "connectivity_tag": rng.choice(["broadcaster", "integrator", "hub", "relay"], size=n_neurons // 2)}),
        out / "connectivity_tags.csv.gz", compress, counts,
    )
    vis = rng.choice(n_neurons, size=n_neurons // 3, replace=False)
    _write(
        pd.DataFrame({
            "root_id": ids[vis],
            "type": rng.choice(["Mi1", "T4", "L1", "R8", "Car", "Dm8"], size=vis.size),
            "family": rng.choice(["Motion", "Color", "Lamina", "Medulla"], size=vis.size),
            "subsystem": rng.choice(["Motion", "Color"], size=vis.size),
            "category": rng.choice(["input", "output"], size=vis.size),
            "side": rng.choice(["R", "L"], size=vis.size),
        }),
        out / "visual_neuron_types.csv.gz", compress, counts,
    )
    _write(
        pd.DataFrame({"root_id": rng.choice(ids, size=n_neurons // 4), "processed_labels": rng.choice(["ell body", "fan-shaped body", "nodulus", "fan wedge"], size=n_neurons // 4)}),
        out / "processed_labels.csv.gz", compress, counts,
    )
    n_lbl = n_neurons // 2
    _write(
        pd.DataFrame({
            "root_id": rng.choice(ids, size=n_lbl), "label": rng.choice(["blob", "columnar", "ring neuron"], size=n_lbl),
            "user_id": rng.integers(1, 60, n_lbl), "position": [f"{x},{y},{z}" for x, y, z in rng.integers(1, 10**6, (n_lbl, 3))],
            "supervoxel_id": rng.integers(1, 10**9, n_lbl), "label_id": np.arange(n_lbl),
            "date_created": pd.date_range("2024-01-01", periods=n_lbl, freq="D").strftime("%Y-%m-%d"),
            "user_name": rng.choice(["alice", "bob", "carol"], size=n_lbl),
            "user_affiliation": rng.choice(["Princeton", "Janelia", "Cambridge"], size=n_lbl),
        }),
        out / "labels.csv.gz", compress, counts,
    )
    _write(
        pd.DataFrame({
            "root_id": rng.choice(ids, size=n_neurons // 5), "hemisphere": rng.choice(["L", "R"], size=n_neurons // 5),
            "type": rng.choice(["R16", "L1", "T4"], size=n_neurons // 5), "column_id": rng.integers(1, 800, n_neurons // 5),
            "x": rng.integers(-9, 9, n_neurons // 5), "y": rng.integers(-29, 31, n_neurons // 5),
            "p": rng.integers(-19, 19, n_neurons // 5), "q": rng.integers(-17, 18, n_neurons // 5),
        }),
        out / "column_assignment.csv.gz", compress, counts,
    )
    # tiny synapse-table twin (documents the split-root-id prefix convention)
    n_syn = 400
    _write(
        pd.DataFrame({
            "pre_x": rng.integers(90_000, 900_000, n_syn), "pre_y": rng.integers(60_000, 540_000, n_syn), "pre_z": rng.integers(3_000, 277_000, n_syn),
            "ctr_x": rng.integers(90_000, 900_000, n_syn), "ctr_y": rng.integers(60_000, 540_000, n_syn), "ctr_z": rng.integers(3_000, 277_000, n_syn),
            "post_x": rng.integers(90_000, 900_000, n_syn), "post_y": rng.integers(60_000, 540_000, n_syn), "post_z": rng.integers(3_000, 277_000, n_syn),
            "size": rng.integers(1, 300, n_syn),
            f"pre_root_id_{ROOT_ID_PREFIX}": ids[rng.integers(0, n_neurons, n_syn)] % 10**9,
            f"post_root_id_{ROOT_ID_PREFIX}": ids[rng.integers(0, n_neurons, n_syn)] % 10**9,
            "neuropil": rng.choice(NEUROPIILS, n_syn),
        }),
        out / "fafb_v783_princeton_synapse_table.csv.gz", compress, counts,
    )

    (out / "README.txt").write_text(
        "SYNTHETIC sample files. They reproduce the FAFB v783 asset NAMES and COLUMN\n"
        "NAMES documented for the Codex 'Download Data' page, but the contents are random\n"
        "numbers generated by fruitfly.dataset.build_sample. Do not use them for results.\n"
        f"Generated with seed={seed}, n_neurons={n_neurons}.\n",
        encoding="utf-8",
    )
    return counts


def _write(df: pd.DataFrame, path: Path, compress: bool, counts: dict[str, int]) -> None:
    counts[path.name] = int(len(df))
    if compress:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False, na_rep="")
    else:
        df.to_csv(path.with_suffix(""), index=False, na_rep="")
