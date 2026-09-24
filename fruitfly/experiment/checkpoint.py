"""Checkpoints: everything needed to resume training exactly where it stopped.

Layout (one directory per run)::

    runs/<name>/
      config.resolved.yaml       the full config that produced this run
      dataset_stats.json         what was loaded, and how big it really was
      metrics.jsonl              one line per episode (append-only)
      checkpoint/
        latest.npz               synaptic weights + plasticity traces + RNG state
        latest.json              episode counter, reward ledger, metrics, hashes
        step000050.npz/.json     periodic snapshots (``train.keep_last`` prunes)

The weight array is the CSR data of the connectome, so a checkpoint is a snapshot
of the *only* learned state in the system. Resume is verified by a test that
trains in one shot versus in two halves with a checkpoint between.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class Checkpoint:
    """In-memory description of a checkpoint on disk."""

    path: Path
    episode: int
    config_hash: str
    seed: int
    metadata: dict[str, Any]

    @property
    def weights(self) -> np.ndarray:
        with np.load(self.path, allow_pickle=False) as z:
            return np.asarray(z["weights"], dtype=np.float64)


def checkpoint_dir(run_dir: str | Path) -> Path:
    d = Path(run_dir) / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rng_state(rs: object) -> tuple:
    """Normalise ``np.random.get_state()`` and a ``RandomState`` to the same tuple."""
    if isinstance(rs, tuple):
        return rs
    return tuple(rs.get_state())  # type: ignore[attr-defined]


def save_checkpoint(
    run_dir: str | Path,
    *,
    conn: Connectome,
    rule_state: dict[str, Any],
    reward_state: dict[str, Any],
    modulator_state: dict[str, Any],
    encoder_state: dict[str, Any],
    decoder_state: dict[str, Any],
    episode: int,
    seed: int,
    config_hash: str,
    rng_state: "np.random.RandomState | tuple",
    extra: dict[str, Any] | None = None,
    tag: str = "latest",
) -> Path:
    """Persist all learnable + bookkeeping state. Returns the ``.npz`` path."""
    d = checkpoint_dir(run_dir)
    npz = d / f"{tag}.npz"
    js = d / f"{tag}.json"
    m = conn.matrix
    arrays: dict[str, np.ndarray] = {
        "weights": np.asarray(m.data, dtype=np.float64),
        "indices": np.asarray(m.indices, dtype=np.int32),
        "indptr": np.asarray(m.indptr, dtype=np.int64),
        "root_ids": np.asarray(conn.root_ids, dtype=np.int64),
        "eligibility": np.asarray(rule_state.get("eligibility", []), dtype=np.float32),
        "updates": np.asarray(rule_state.get("updates", []), dtype=np.int64),
        "trace_pre": np.asarray(rule_state.get("trace_pre", []), dtype=np.float32),
        "trace_post": np.asarray(rule_state.get("trace_post", []), dtype=np.float32),
    }
    np.savez_compressed(npz, **arrays)
    payload = {
        "episode": int(episode),
        "seed": int(seed),
        "config_hash": config_hash,
        "n_neurons": conn.n_neurons,
        "n_edges": conn.n_edges,
        "rule": {k: v for k, v in rule_state.items() if not isinstance(v, np.ndarray)},
        "reward": reward_state,
        "modulator": modulator_state,
        "encoder": encoder_state,
        "decoder": decoder_state,
        "rng_state": np.asarray(_rng_state(rng_state)[1], dtype=np.int64).tolist(),
        "rng_normal": str(_rng_state(rng_state)[0]),
        "extra": extra or {},
    }
    js.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")
    log.info("checkpoint written: %s (%.1f MB)", npz.name, npz.stat().st_size / 1e6)
    return npz


def prune(run_dir: str | Path, keep_last: int) -> list[str]:
    """Keep ``latest`` plus the newest ``keep_last`` numbered snapshots."""
    d = checkpoint_dir(run_dir)
    tagged = sorted(p.stem for p in d.glob("step*.npz"))
    removed: list[str] = []
    for stem in tagged[:-keep_last] if keep_last > 0 else tagged:
        for suffix in (".npz", ".json"):
            f = d / f"{stem}{suffix}"
            if f.exists():
                f.unlink()
                removed.append(f.name)
    return removed


def load_checkpoint(run_dir: str | Path, tag: str = "latest") -> Checkpoint:
    d = checkpoint_dir(run_dir)
    npz = d / f"{tag}.npz"
    js = d / f"{tag}.json"
    if not npz.exists():
        raise FileNotFoundError(f"no checkpoint {npz}")
    meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    return Checkpoint(
        path=npz,
        episode=int(meta.get("episode", 0)),
        config_hash=str(meta.get("config_hash", "")),
        seed=int(meta.get("seed", 0)),
        metadata=meta,
    )


def restore_connectome(cp: Checkpoint, conn: Connectome, *, strict: bool = True) -> dict[str, Any]:
    """Copy checkpointed weights into ``conn`` in place. Returns rule state dict."""
    with np.load(cp.path, allow_pickle=False) as z:
        w = np.asarray(z["weights"], dtype=np.float64)
        if w.shape != conn.matrix.data.shape:
            raise ValueError(
                f"checkpoint weight array {w.shape} does not match connectome {conn.matrix.data.shape}; "
                "resuming requires the same population (same mode/selection/seed)"
            )
        conn.matrix.data[:] = w
        state = {
            "eligibility": np.asarray(z["eligibility"]) if "eligibility" in z.files and z["eligibility"].size else np.zeros_like(conn.matrix.data, np.float32),
            "updates": np.asarray(z["updates"]) if "updates" in z.files and z["updates"].size else np.zeros(conn.matrix.data.shape, np.int64),
            "trace_pre": np.asarray(z["trace_pre"]) if "trace_pre" in z.files and z["trace_pre"].size else None,
            "trace_post": np.asarray(z["trace_post"]) if "trace_post" in z.files and z["trace_post"].size else None,
        }
    if strict and cp.metadata.get("n_neurons") not in (None, conn.n_neurons):
        raise ValueError(f"checkpoint population {cp.metadata.get('n_neurons')} != current {conn.n_neurons}")
    return state


def copy_run(src: str | Path, dst: str | Path) -> Path:  # pragma: no cover - convenience
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return Path(dst)
