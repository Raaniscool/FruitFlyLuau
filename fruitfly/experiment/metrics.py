"""Metrics logging: append-only JSONL plus windowed summaries and statistics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)


class MetricsLogger:
    """Append-only ``metrics.jsonl`` writer with an in-memory mirror.

    JSONL rather than a DataFrame because a run may be interrupted and resumed;
    the file stays readable either way.
    """

    def __init__(self, path: str | Path, *, buffer_every: int = 20) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            # a resumed run appends to the same file; the windowed curves and the
            # learning criterion must see the earlier episodes too, or "continue
            # training" would silently reset the accuracy history.
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("ignoring truncated metrics line in %s (interrupted run?)", self.path)
        self.buffer_every = int(buffer_every)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._since_flush = 0

    def log(self, **record: Any) -> None:
        record.setdefault("wall_time_s", time.time())
        self.records.append(record)
        self._fh.write(json.dumps(record, default=float, sort_keys=True) + "\n")
        self._since_flush += 1
        if self._since_flush >= self.buffer_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ series
    def series(self, key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in self.records], dtype=np.float64)

    def windowed(self, key: str, window: int) -> np.ndarray:
        """Trailing moving average (NaN-tolerant); used for learning curves."""
        import pandas as pd

        vals = self.series(key)
        if vals.size == 0:
            return vals
        return pd.Series(vals).rolling(max(1, int(window)), min_periods=1).mean().to_numpy()

@dataclass
class RunningStats:
    """Streaming mean/std for a single scalar series."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    last: float = 0.0
    max: float = -np.inf
    min: float = np.inf
    total: float = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.last = x
        self.total += x
        self.max = max(self.max, x)
        self.min = min(self.min, x)
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / (self.n - 1))) if self.n > 1 else 0.0

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "std": self.std, "min": None if not np.isfinite(self.min) else self.min,
                "max": None if not np.isfinite(self.max) else self.max, "total": self.total, "last": self.last}


def chance_level(n_classes: int) -> float:
    return 1.0 / max(1, int(n_classes))


def binom_p_above_chance(k_successes: int, n_trials: int, p_chance: float) -> float:
    """One-sided exact binomial p-value (success probability > chance).

    Returns ``nan`` when scipy is unavailable, so reporting degrades visibly
    rather than silently. An empty test returns 1.0: no trials is no evidence, but
    it is *defined*, and it keeps ``learning_detected`` False instead of poisoning
    the report with a NaN.
    """
    if n_trials <= 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(k_successes), int(n_trials), float(p_chance), alternative="greater").pvalue)
    except Exception:  # pragma: no cover
        return float("nan")


def summarize_curve(values: Iterable[float], *, window: int = 25) -> dict:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    tail = v[-min(v.size, max(1, window)):]
    return {
        "n": int(v.size),
        "first": float(v[0]),
        "last": float(v[-1]),
        "best": float(v.max()),
        "mean": float(v.mean()),
        "window_mean": float(tail.mean()),
        "slope_per_100": float(np.polyfit(np.arange(v.size), v, 1)[0] * 100) if v.size > 2 else 0.0,
    }
