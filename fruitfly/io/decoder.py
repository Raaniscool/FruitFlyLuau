"""Output decoding: neural activity -> a label, a symbol, or a sequence of tokens.

Decoding is a readout of measured activity only: it stores no answers and does not
adapt to specific questions. Where a decoder has parameters (e.g. a learned
readout vector), those are explicitly listed and included in checkpoints so a
reviewer can see exactly what state exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class Decision:
    """One readout result, with the evidence that produced it."""

    label: Any
    scores: np.ndarray
    margin: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def correct(self, target: Any) -> bool:
        return self.label == target


class OutputDecoder(ABC):
    """Base class for readouts over a designated output population."""

    kind: str = "base"

    def __init__(self, *, n_output_neurons: int = 8, min_spikes: int = 1, **_: Any) -> None:
        self.n_output_neurons = int(n_output_neurons)
        self.min_spikes = int(min_spikes)
        self.output_neurons: np.ndarray | None = None
        self.classes: Sequence[Any] = (0, 1)

    def configure(self, conn: Connectome, rng: np.random.Generator, *, classes: Sequence[Any] | None = None) -> None:
        """Claim output neurons. Default: the ``k`` most strongly-innervated cells."""
        n = conn.n_neurons
        k = min(self.n_output_neurons, max(1, n))
        if classes is not None:
            self.classes = list(classes)
        if conn.n_edges == 0:
            self.output_neurons = np.linspace(0, n - 1, k).round().astype(np.int64)
            return
        in_deg = np.asarray((conn.matrix != 0).sum(axis=0)).ravel()
        pick = np.argsort(-in_deg, kind="stable")[:k]
        self.output_neurons = np.sort(pick).astype(np.int64)
        log.debug("readout neurons: %s", self.output_neurons)

    @abstractmethod
    def decode(self, spike_counts: np.ndarray, *, per_step: np.ndarray | None = None) -> Decision:  # pragma: no cover
        """Map per-neuron spike counts (or an optional (steps, n) raster) to a label."""

    def groups_for_classes(self) -> list[np.ndarray]:
        """Split the readout population into one contiguous group per class."""
        k = int(self.output_neurons.size)
        c = max(1, len(self.classes))
        edges = np.linspace(0, k, c + 1).round().astype(int)
        return [self.output_neurons[edges[i] : edges[i + 1]] for i in range(c)]

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "output_neurons": None if self.output_neurons is None else self.output_neurons.tolist(),
            # class labels keep their native type: the experiment's ``is_correct``
            # compares them against the environment's targets, so stringifying
            # them here would silently break every resumed run.
            "classes": [c.item() if hasattr(c, "item") else c for c in self.classes],
        }

    def load_state(self, d: Mapping[str, Any]) -> None:
        if d.get("output_neurons"):
            self.output_neurons = np.asarray(d["output_neurons"], dtype=np.int64)
        if d.get("classes"):
            self.classes = list(d["classes"])


class GroupRateDecoder(OutputDecoder):
    """Class = argmax over spike counts summed within each class's neuron group.

    This keeps the readout *inside* the neural system: nothing outside the
    connectome is trained here, and the only statistic used is how much each
    designated output population fired.
    """

    kind = "group_rate"

    def decode(self, spike_counts: np.ndarray, *, per_step: np.ndarray | None = None) -> Decision:
        counts = np.asarray(spike_counts, dtype=np.float64).ravel()
        groups = self.groups_for_classes()
        scores = np.array([counts[g].sum() if g.size else 0.0 for g in groups], dtype=np.float64)
        if scores.size == 0:
            raise ValueError("decoder has no classes")
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        margin = float(scores[best] - scores[order[1]]) if scores.size > 1 else float(scores[best])
        total = float(scores.sum())
        conf = float(scores[best] / total) if total > 0 else 0.0
        below = int((scores < self.min_spikes).sum())
        label = self.classes[best] if self.min_spikes <= 0 or scores[best] >= self.min_spikes else None
        return Decision(
            label=label, scores=scores, margin=margin, confidence=conf,
            evidence={
                "group_spikes": scores.tolist(), "silence_fallbacks": below,
                "total_spikes_in_readout": total,
            },
        )


class PopulationRateDecoder(OutputDecoder):
    """Binary decision from a rate comparison between two readout halves."""

    kind = "population_rate"

    def decode(self, spike_counts: np.ndarray, *, per_step: np.ndarray | None = None) -> Decision:
        counts = np.asarray(spike_counts, dtype=np.float64).ravel()[self.output_neurons]
        half = counts.size // 2
        lo, hi = float(counts[:half].sum()), float(counts[half:].sum())
        scores = np.array([lo, hi])
        best = int(np.argmax(scores))
        total = float(scores.sum())
        return Decision(
            label=self.classes[best] if total > 0 else None, scores=scores, margin=abs(lo - hi),
            confidence=(scores[best] / total) if total else 0.0,
            evidence={"first_half": lo, "second_half": hi},
        )


class ThresholdDecoder(OutputDecoder):
    """0/1 from whether any 'true'-class neuron fired at all (a spike/no-spike rule)."""

    kind = "threshold"

    def decode(self, spike_counts: np.ndarray, *, per_step: np.ndarray | None = None) -> Decision:
        counts = np.asarray(spike_counts, dtype=np.float64).ravel()
        groups = self.groups_for_classes()
        scores = np.array([counts[g].sum() if g.size else 0.0 for g in groups])
        firing = scores >= max(1, self.min_spikes)
        label = self.classes[int(np.argmax(scores))] if firing.any() else self.classes[0]
        return Decision(label=label, scores=scores, margin=float((scores.max() if scores.size else 0.0) - (scores.min() if scores.size else 0.0)),
                        confidence=float(firing.mean()), evidence={"n_firing_groups": int(firing.sum())})


class TokenDecoder(OutputDecoder):
    """Sequence generation: decode one token per window, in order.

    ``decode_stream(rasters)`` is what phase 7 uses to emit Luau token sequences;
    phase 3 already uses it for the sequence-prediction experiment.
    """

    kind = "token"

    def __init__(self, *, window_steps: int = 10, vocabulary: Sequence[Any] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.window_steps = int(window_steps)
        self.vocabulary = list(vocabulary) if vocabulary else None

    def configure(self, conn: Connectome, rng: np.random.Generator, *, classes: Sequence[Any] | None = None) -> None:
        if classes is not None:
            self.classes = list(classes)
        if self.vocabulary is not None and len(self.vocabulary) == len(self.classes):
            self.classes = list(self.vocabulary)
        # one neuron per class is the clearest readout for token emission
        self.n_output_neurons = max(self.n_output_neurons, len(self.classes))
        super().configure(conn, rng, classes=self.classes)

    def decode(self, spike_counts: np.ndarray, *, per_step: np.ndarray | None = None) -> Decision:
        d = GroupRateDecoder(n_output_neurons=self.n_output_neurons)
        d.output_neurons = self.output_neurons
        d.classes = self.classes
        d.min_spikes = self.min_spikes
        dec = d.decode(spike_counts, per_step=per_step)
        return dec

    def decode_stream(self, raster: np.ndarray) -> list[Decision]:
        """Split an (steps, n) spike raster into windows and decode each."""
        r = np.asarray(raster)
        if r.ndim != 2:
            raise ValueError("decode_stream needs a (steps, n_neurons) raster")
        out = []
        for start in range(0, r.shape[0], self.window_steps):
            block = r[start : start + self.window_steps].sum(axis=0)
            out.append(self.decode(block))
        return out

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["window_steps"] = self.window_steps
        d["vocabulary"] = None if self.vocabulary is None else [str(v) for v in self.vocabulary]
        return d


_DECODERS: dict[str, type[OutputDecoder]] = {
    cls.kind: cls for cls in (GroupRateDecoder, PopulationRateDecoder, ThresholdDecoder, TokenDecoder)
}


def register_decoder(cls: type[OutputDecoder]) -> type[OutputDecoder]:
    _DECODERS[cls.kind] = cls
    return cls


def build_decoder(spec: str | Mapping[str, Any] | None) -> OutputDecoder:
    if spec is None:
        return GroupRateDecoder()
    if isinstance(spec, OutputDecoder):
        return spec
    if isinstance(spec, Mapping):
        spec = dict(spec)
        kind = str(spec.pop("kind", "group_rate"))
        kwargs = dict(spec.pop("kwargs", {}))
        kwargs.update(spec)
    else:
        kind, kwargs = str(spec), {}
    if kind not in _DECODERS:
        raise ValueError(f"unknown decoder {kind!r}; available: {sorted(_DECODERS)}")
    return _DECODERS[kind](**kwargs)


def available_decoders() -> list[str]:
    return sorted(_DECODERS)
