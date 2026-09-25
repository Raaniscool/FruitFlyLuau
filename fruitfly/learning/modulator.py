"""Dopamine-like modulatory signal.

Terminology discipline: this is a **modulatory reinforcement signal** inspired by
phasic/tonic dopamine accounts of reinforcement learning. It is a scalar trace
variable. It is *not* dopamine, and nothing here should be described as a
neurochemical claim.

    DA[t] = decay * DA[t-1] + (reward[t] - baseline[t])

``baseline`` is an exponential moving average of delivered reward, so a fully
expected reward stops driving plasticity (a reward-prediction-error flavour).
Set ``baseline_subtract=False`` for the raw trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RewardConfig
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class DopamineLikeModulator:
    """Discrete-time modulatory trace driven by the reward ledger."""

    cfg: RewardConfig
    dt_ms: float = 0.5
    value: float = 0.0
    tonic: float = 0.0
    baseline: float = 0.0
    n_steps: int = 0
    n_deliveries: int = 0
    trace: list[float] = field(default_factory=list)
    max_trace: int = 4000

    def __post_init__(self) -> None:
        tau = max(float(self.cfg.trace_tau_ms), self.dt_ms)
        self._decay = float(np.exp(-self.dt_ms / tau))
        self._alpha = max(float(self.cfg.trace_decay), 1e-6)  # baseline EMA rate

    def deliver(self, reward: float) -> None:
        """Called once per simulation step with the reward value due now (0 if none)."""
        r = float(reward)
        if r:
            self.n_deliveries += 1
        signal = r - self.baseline if self.cfg.baseline_subtract else r
        self.tonic = self._alpha * self.tonic + (1 - self._alpha) * r
        cap = float(getattr(self.cfg, "value_abs_max", 0.0) or 0.0)
        v = self._decay * self.value + signal
        if cap > 0:
            v = max(-cap, min(cap, v))
        self.value = v
        self.baseline = self._alpha * self.baseline + (1 - self._alpha) * r
        self.n_steps += 1
        if len(self.trace) < self.max_trace:
            self.trace.append(self.value)

    @property
    def current(self) -> float:
        return float(self.value)

    def reset_trial(self) -> None:
        """Optional per-trial reset; by default the trace persists across trials."""
        self.value = 0.0

    def stats(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_deliveries": self.n_deliveries,
            "mean_abs_value": float(np.mean(np.abs(self.trace))) if self.trace else 0.0,
            "max_abs_value": float(np.max(np.abs(self.trace))) if self.trace else 0.0,
            "final_value": self.current,
            "tonic": float(self.tonic),
            "baseline": float(self.baseline),
            "decay_per_step": self._decay,
            "prediction_error_mode": bool(self.cfg.baseline_subtract),
        }

    def state_dict(self) -> dict:
        return {
            "value": self.value, "tonic": self.tonic, "baseline": self.baseline,
            "n_steps": self.n_steps, "n_deliveries": self.n_deliveries, "trace": list(self.trace),
        }

    def load_state(self, d: dict) -> None:
        self.value = float(d.get("value", 0.0))
        self.tonic = float(d.get("tonic", 0.0))
        self.baseline = float(d.get("baseline", 0.0))
        self.n_steps = int(d.get("n_steps", 0))
        self.n_deliveries = int(d.get("n_deliveries", 0))
        self.trace = list(d.get("trace", []))
