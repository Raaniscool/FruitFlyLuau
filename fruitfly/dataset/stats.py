"""Dataset statistics for the loaded FAFB data (measured, not quoted)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


def connectome_stats(conn: Connectome) -> dict:
    return conn.stats()


def table_stats(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, *, nt=None, neuropil=None) -> dict:
    ids = np.union1d(pre, post)
    out: dict = {
        "n_rows": int(pre.size),
        "n_unique_neuron_ids": int(ids.size),
        "n_directed_pairs": int(np.unique(pre.astype(np.int64) * np.int64(ids.size) + post.astype(np.int64)).size)
        if ids.size**2 < 2**62
        else int(pre.size),
        "id_min": int(ids.min()) if ids.size else 0,
        "id_max": int(ids.max()) if ids.size else 0,
        "id_digits": sorted({int(np.log10(v)) + 1 for v in ids[:1000]} | {int(np.log10(v)) + 1 for v in ids[-1000:]}),
        "id_prefix9": sorted({str(v)[:9] for v in ids[:200]}),
        "weight_min": (float(weight.min()) if weight.size else 0.0),
        "weight_median": float(np.median(weight)) if weight.size else 0.0,
        "weight_mean": float(weight.mean()) if weight.size else 0.0,
        "weight_max": (float(weight.max()) if weight.size else 0.0),
        "autapses": int((pre == post).sum(initial=0)),
    }
    if nt is not None:
        c = Counter(str(v) for v in nt)
        out["nt_type_values"] = dict(sorted(c.items(), key=lambda kv: -kv[1])[:12])
        out["nt_type_n_unique"] = len(c)
    if neuropil is not None:
        c = Counter(str(v) for v in neuropil)
        out["neuropil_values"] = dict(sorted(c.items(), key=lambda kv: -kv[1])[:20])
        out["neuropil_n_unique"] = len(c)
    deg_out = np.bincount(np.searchsorted(ids, pre), minlength=ids.size)
    deg_in = np.bincount(np.searchsorted(ids, post), minlength=ids.size)
    out["out_degree"] = {"mean": float(deg_out.mean()), "median": float(np.median(deg_out)), "max": int(deg_out.max()) if deg_out.size else 0}
    out["in_degree"] = {"mean": float(deg_in.mean()), "median": float(np.median(deg_in)), "max": int(deg_in.max()) if deg_in.size else 0}
    out["zero_out_degree"] = int((deg_out == 0).sum())
    out["zero_in_degree"] = int((deg_in == 0).sum())
    return out


def format_report(payload: dict) -> str:
    """Human-readable text for `fruitfly stats`."""
    lines: list[str] = []

    def emit(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, dict) and len(v) <= 24:
                lines.append(f"{prefix}{k}:")
                emit(v, prefix + "  ")
            elif isinstance(v, (list, tuple)) and len(v) > 12:
                lines.append(f"{prefix}{k}: [{v[0]}, {v[1]}, ... {len(v)} items]")
            else:
                if isinstance(v, (int, np.integer)):
                    lines.append(f"{prefix}{k}: {int(v):,}")
                elif isinstance(v, float):
                    lines.append(f"{prefix}{k}: {v:.6g}")
                else:
                    lines.append(f"{prefix}{k}: {v}")

    emit(payload)
    return "\n".join(lines)
