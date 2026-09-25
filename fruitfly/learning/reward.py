"""Configurable reinforcement ledger.

The reward system does two things and nothing else: it decides what value a
trial outcome deserves, and it makes that value available at a configurable
*delay* so that the timing of reinforcement (not just its sign) can be studied.
It keeps full history for learning curves. It never stores task answers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import RewardConfig
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class RewardEvent:
    """One delivered reinforcement signal."""

    episode: int
    step: int
    value: float
    correct: bool | None = None
    kind: str = "outcome"  # outcome | punishment | shaping | exploration

    def to_dict(self) -> dict:
        return {"episode": self.episode, "step": self.step, "value": float(self.value),
                "correct": None if self.correct is None else bool(self.correct), "kind": self.kind}


@dataclass
class RewardSystem:
    """Reward delivery with configurable magnitude and delay.

    ``delay_steps`` shifts when a value becomes visible to the modulator, measured
    in simulation steps of the *current* trial (a delayed-reward experiment).
    """

    cfg: RewardConfig
    history: list[RewardEvent] = field(default_factory=list)
    outcomes: list[bool | None] = field(default_factory=list)
    _pending: dict[int, float] = field(default_factory=dict)
    _episode: int = 0
    _errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cfg.positive <= 0 and self.cfg.positive != 0:
            self._errors.append("positive reward <= 0 was configured")
        if int(self.cfg.delay_steps) < 0:
            raise ValueError("reward.delay_steps must be >= 0")
        self._recent: deque[float] = deque(maxlen=max(8, int(self.cfg.max_history) // 10))
        self._recent_correct: deque[float] = deque(maxlen=max(8, int(self.cfg.max_history) // 10))

    # ------------------------------------------------------------------ config
    @property
    def positive(self) -> float:
        return float(self.cfg.positive)

    @property
    def negative(self) -> float:
        return float(self.cfg.negative)

    def value_for(self, correct: bool, *, magnitude: float = 1.0) -> float:
        """Map a trial outcome onto a signed reinforcement value."""
        if correct:
            return self.positive * float(magnitude)
        if self.cfg.penalize_no_response:
            return self.negative * float(magnitude)
        return float(self.cfg.neutral)

    # ----------------------------------------------------------------- delivery
    def deliver(self, value: float, *, step: int = 0, correct: bool | None = None, kind: str = "outcome") -> RewardEvent:
        """Queue a reward value; it appears to the modulator ``delay_steps`` later."""
        ev = RewardEvent(episode=self._episode, step=int(step) + int(self.cfg.delay_steps), value=float(value), correct=correct, kind=kind)
        self.history.append(ev)
        if len(self.history) > self.cfg.max_history * 2:
            del self.history[: len(self.history) - self.cfg.max_history]
        due = ev.step
        self._pending[due] = self._pending.get(due, 0.0) + ev.value
        return ev

    def deliver_outcome(self, correct: bool, *, step: int = 0, magnitude: float = 1.0) -> RewardEvent:
        return self.deliver(self.value_for(correct, magnitude=magnitude), step=step, correct=correct)

    def pop(self, step: int) -> float:
        """Reward value due at ``step`` (0 if none). Called once per sim step."""
        v = self._pending.pop(int(step), 0.0)
        if v:
            self._recent.append(v)
        return v

    def end_trial(self, *, correct: bool | None = None) -> float:
        """Close a trial. Returns any reward value still queued but not yet due.

        A reward delivered with ``delay_steps`` past the end of the trial would
        otherwise sit in ``_pending`` forever -- the next trial's ``pop(step)``
        never asks for that step again. The runner adds the returned value to the
        modulator instead of dropping it, and it is also what
        :func:`fruitfly.experiment.readout.probe_accuracy` would need to know about.
        """
        if correct is not None:
            self.outcomes.append(bool(correct))
            self._recent_correct.append(1.0 if correct else 0.0)
        self._episode += 1
        leftover = 0.0
        if self._pending:
            leftover = float(sum(self._pending.values()))
            self._pending.clear()
            log.debug("flushing %.4g undelivered reward at trial end", leftover)
        return leftover

    # ----------------------------------------------------------------- reports
    def window_reward(self, n: int = 50) -> float:
        vals = [e.value for e in self.history[-n:]]
        return float(np.mean(vals)) if vals else 0.0

    def window_accuracy(self, n: int = 50) -> float | None:
        known = [o for o in self.outcomes[-n:] if o is not None]
        return float(np.mean(known)) if known else None

    def cumulative(self) -> dict:
        vals = np.array([e.value for e in self.history], dtype=np.float64)
        return {
            "n_events": int(vals.size),
            "total_reward": float(vals.sum(initial=0.0)),
            "mean_reward": float((float(vals.mean()) if vals.size else 0.0)),
            "n_positive": int((vals > 0).sum(initial=0)),
            "n_negative": int((vals < 0).sum(initial=0)),
            "accuracy_overall": self.window_accuracy(len(self.outcomes) or 1),
            "errors": list(self._errors),
        }

    def curve(self, every: int = 1) -> dict[str, list[float]]:
        """Episode-indexed reward/accuracy series for plotting and reporting."""
        per_ep: dict[int, float] = {}
        for e in self.history:
            per_ep[e.episode] = per_ep.get(e.episode, 0.0) + e.value
        eps = sorted(per_ep)
        return {
            "episode": [float(e) for e in eps],
            "reward": [per_ep[e] for e in eps],
            "accuracy": [
                (1.0 if (e < len(self.outcomes) and self.outcomes[e]) else 0.0) if e < len(self.outcomes) and self.outcomes[e] is not None else np.nan
                for e in eps
            ],
        }

    def state_dict(self) -> dict:
        return {
            "history": [e.to_dict() for e in self.history][-self.cfg.max_history:],
            "outcomes": [None if o is None else bool(o) for o in self.outcomes],
            "episode": self._episode,
        }

    def load_state(self, d: dict) -> None:
        self.history = [RewardEvent(**{k: v for k, v in e.items()}) for e in d.get("history", [])]
        self.outcomes = list(d.get("outcomes", []))
        self._episode = int(d.get("episode", len(self.outcomes)))
