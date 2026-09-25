"""Experiment framework: an environment + encoder + decoder + population + rules.

An experiment is fully described by its config (seed included), so re-running
with the same config must reproduce the same numbers -- enforced by a test
(``tests/test_reproducibility.py``), not by assertion in prose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class Trial:
    """One interaction: what the system is shown, and what counts as right."""

    observation: Any
    target: Any
    key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Task source. Must provide disjoint train/eval item sets (no leakage)."""

    name: str = "base"
    #: classes the readout can emit; order matters (it defines group boundaries)
    classes: Sequence[Any] = (0, 1)
    #: how the observation should be encoded, by name (e.g. "binary", "bitvector")
    encoder_kind: str = "binary"
    decoder_kind: str = "group_rate"
    #: number of simulation steps per trial (input window)
    trial_steps: int = 12

    def __init__(self, *, seed: int = 0, n_train: int = 240, n_eval: int = 120, **_: Any) -> None:
        self.seed = int(seed)
        self.n_train = int(n_train)
        self.n_eval = int(n_eval)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ data
    @abstractmethod
    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        """Produce ``n`` trials. ``offset`` keeps train/eval streams disjoint."""

    def train_items(self) -> list[Trial]:
        if not hasattr(self, "_train"):
            self._train = self.build_items(self.n_train, rng=np.random.default_rng(self.seed), offset=0)
        return self._train

    #: True for tasks whose stimulus space is fully enumerated (e.g. binary, xor):
    #: for those, train and eval *must* share observations, so accuracy measures
    #: associative learning and can never be described as generalisation.
    exhaustive_input_space: bool = False

    def eval_items(self) -> list[Trial]:
        """The held-out set, kept disjoint from training whenever the task allows it.

        For a non-exhaustive task (patterns with fresh corruption, symbolic
        arguments drawn from a large modulus) an observation that already appeared
        in training is replaced by a fresh draw, so ``overlap_check`` can honestly
        report zero leakage. For an exhaustive task there is nothing to replace it
        with, and :attr:`exhaustive_input_space` says so out loud instead of
        pretending.
        """
        if not hasattr(self, "_eval"):
            rng = np.random.default_rng(self.seed + 7919)
            items = self.build_items(self.n_eval, rng=rng, offset=self.n_train)
            if not self.exhaustive_input_space:
                tr = {t.key or repr(t.observation) for t in self.train_items()}
                for attempt in range(12):
                    bad = [i for i, t in enumerate(items) if (t.key or repr(t.observation)) in tr]
                    if not bad:
                        break
                    repl = self.build_items(self.n_eval, rng=np.random.default_rng(self.seed + 7919 + attempt + 1),
                                             offset=self.n_train + (attempt + 1) * self.n_eval)
                    for i in bad:
                        cand = repl[i]
                        if (cand.key or repr(cand.observation)) not in tr:
                            items[i] = cand
                            tr.add(cand.key or repr(cand.observation))
                else:  # pragma: no cover - only if the sampler cannot produce fresh items
                    log.warning("could not construct a leakage-free eval split for %s after 12 attempts", self.name)
            self._eval = items
        return self._eval

    def overlap_check(self) -> dict:
        """Leakage audit: do any eval observations appear in training data?"""
        tr = {t.key or repr(t.observation) for t in self.train_items()}
        ev = [t for t in self.eval_items() if (t.key or repr(t.observation)) in tr]
        return {
            "n_train": len(self.train_items()),
            "n_eval": len(self.eval_items()),
            "n_eval_obs_in_train": len(ev),
            "leakage_free": len(ev) == 0,
        }

    def epochs(self, n_episodes: int) -> Iterator[Trial]:
        """Infinite-ish training stream: shuffled repeats of the train set."""
        items = list(self.train_items())
        rng = np.random.default_rng(self.seed + 1)
        count = 0
        while count < n_episodes:
            order = rng.permutation(len(items))
            for j in order:
                if count >= n_episodes:
                    return
                count += 1
                yield items[int(j)]

    # ----------------------------------------------------------------- reward
    def is_correct(self, prediction: Any, target: Any) -> bool:
        return prediction is not None and prediction == target

    def describe(self) -> dict:
        return {
            "name": self.name,
            "classes": list(map(str, self.classes)),
            "encoder": self.encoder_kind,
            "decoder": self.decoder_kind,
            "trial_steps": self.trial_steps,
            "exhaustive_input_space": bool(self.exhaustive_input_space),
            "n_classes": len(self.classes),
            **self.overlap_check(),
        }


# ---------------------------------------------------------------- registry
Envs = dict[str, type[Environment]]
_REGISTRY: Envs = {}


def register_env(name: str) -> Callable[[type[Environment]], type[Environment]]:
    def deco(cls: type[Environment]) -> type[Environment]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def get_environment(name: str) -> type[Environment]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_environment(name: str, **kwargs: Any) -> Environment:
    cls = get_environment(name)
    allowed = {k: v for k, v in kwargs.items() if k not in {"kind", "_"}}
    return cls(**allowed)


def available_environments() -> list[str]:
    return sorted(_REGISTRY)
