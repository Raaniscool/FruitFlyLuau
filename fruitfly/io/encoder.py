"""Input encoding: external information -> per-step injected current.

Everything is replaceable: an encoder is just ``encode(observation) -> InjectionPlan``
plus a ``configure(...)`` hook that claims neurons from the population. The
Luau-phase encoders (``fruitfly/luau/encoder.py``) plug into the same registry,
so nothing in the simulator knows or cares what the input "means".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class InjectionPlan:
    """Sparse external-drive schedule for one trial.

    ``per_step[t] = (neuron_indices, amplitudes)`` keeps memory O(active neurons)
    instead of O(steps x n_neurons), which matters at full-brain scale.
    """

    n_steps: int
    per_step: list[tuple[np.ndarray, np.ndarray]]
    detail: dict[str, Any] = field(default_factory=dict)

    def total_charge(self) -> float:
        return float(sum(v.sum() for _, v in self.per_step)) if self.per_step else 0.0


class InputEncoder(ABC):
    """Base class. Subclasses must be deterministic given (observation, seed)."""

    kind: str = "base"

    def __init__(self, *, n_input_neurons: int = 12, amplitude: float = 22.0, duration_steps: int = 24, **_: Any) -> None:
        #: ``amplitude`` is in the same units as the LIF current input; 22 is the
        #: value selected by scripts/calibrate_dynamics.py, not a round number.

        self.n_input_neurons = int(n_input_neurons)
        self.amplitude = float(amplitude)
        self.duration_steps = int(duration_steps)
        self.input_neurons: np.ndarray | None = None
        self.gain: np.ndarray | None = None  # per (pattern, input neuron) gain, learnable option
        self._vocab: dict[Any, int] = {}

    # ------------------------------------------------------------------ hooks
    def configure(self, conn: Connectome, rng: np.random.Generator, *, vocab: Sequence[Any] | None = None) -> None:
        """Claim input neurons. Default: spread evenly over the population."""
        n = conn.n_neurons
        k = min(self.n_input_neurons, max(1, n))
        idx = np.linspace(0, n - 1, k).round().astype(np.int64)
        self.input_neurons = np.unique(idx)
        self.vocab = list(vocab) if vocab is not None else None
        self.rng = rng

    @property
    @abstractmethod
    def patterns(self) -> int:  # pragma: no cover - interface
        """Number of distinct input patterns this encoder can emit."""

    @abstractmethod
    def pattern_for(self, observation: Any) -> np.ndarray:  # pragma: no cover - interface
        """Boolean/int mask over ``input_neurons`` selecting which cells are driven."""

    def encode(self, observation: Any, *, n_steps: int | None = None) -> InjectionPlan:
        steps = int(n_steps or self.duration_steps)
        pat = np.asarray(self.pattern_for(observation), dtype=bool)
        idx = self.input_neurons[pat]
        amp = np.full(idx.size, self.amplitude, dtype=np.float64)
        if self.gain is not None and pat.any():
            code = self.code_for(observation)
            amp = amp * self.gain[code, pat]
        per_step = [(idx, amp.copy()) for _ in range(steps)]
        return InjectionPlan(n_steps=steps, per_step=per_step, detail={"observation": repr(observation), "n_driven": int(idx.size)})

    def code_for(self, observation: Any) -> int:
        key = self.canonical(observation)
        if key not in self._vocab:
            if len(self._vocab) >= 4096:
                raise RuntimeError("encoder vocab overflow")
            self._vocab[key] = len(self._vocab)
        return self._vocab[key]

    def canonical(self, observation: Any) -> Any:
        return observation

    # -------------------------------------------------------------- learnable drive
    def enable_learning(self, patterns: int) -> None:
        """Opt-in per-(pattern, input-neuron) gain, initialised to 1.

        This abstracts "the sensory drive onto first-layer cells gets stronger".
        It is off by default: the headline experiments learn only through connectome
        weights, so a result cannot be explained away by a tuned input stage.
        """
        k = int(self.input_neurons.size) if self.input_neurons is not None else self.n_input_neurons
        self.gain = np.ones((int(patterns), k), dtype=np.float64)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "input_neurons": None if self.input_neurons is None else self.input_neurons.tolist(),
                "gain": None if self.gain is None else self.gain.tolist(), "vocab": dict(sorted(self._vocab.items(), key=lambda kv: kv[1]))}

    def load_state(self, d: Mapping[str, Any]) -> None:
        if d.get("input_neurons"):
            self.input_neurons = np.asarray(d["input_neurons"], dtype=np.int64)
        if d.get("gain"):
            self.gain = np.asarray(d["gain"], dtype=np.float64)
        self._vocab = dict(d.get("vocab", {}))


class BinaryEncoder(InputEncoder):
    """0/1 -> two disjoint input-neuron subsets."""

    kind = "binary"

    @property
    def patterns(self) -> int:
        return 2

    def canonical(self, observation: Any) -> Any:
        return int(bool(observation))

    def pattern_for(self, observation: Any) -> np.ndarray:
        k = int(self.input_neurons.size)
        half = max(1, k // 2)
        out = np.zeros(k, dtype=bool)
        if int(bool(observation)) == 0:
            out[:half] = True
        else:
            out[half:] = True
        return out


class BitVectorEncoder(InputEncoder):
    """Fixed-width integer / bit vector -> one neuron group per (bit, position)."""

    kind = "bitvector"

    def __init__(self, *, width: int = 4, **kw: Any) -> None:
        super().__init__(**kw)
        self.width = int(width)

    def configure(self, conn: Connectome, rng: np.random.Generator, *, vocab: Sequence[Any] | None = None) -> None:
        n = conn.n_neurons
        k = min(max(2 * self.width, self.n_input_neurons), max(2, n))
        self.input_neurons = np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))
        self.vocab = None
        self.rng = rng

    @property
    def patterns(self) -> int:
        return 2 ** self.width

    def canonical(self, observation: Any) -> Any:
        if isinstance(observation, (tuple, list, np.ndarray)):
            bits = np.asarray(observation, dtype=np.int64)
            val = int(sum(int(b) << (self.width - 1 - i) for i, b in enumerate(bits)))
        else:
            val = int(observation)
        return val % (2 ** self.width)

    def pattern_for(self, observation: Any) -> np.ndarray:
        val = self.canonical(observation)
        k = int(self.input_neurons.size)
        out = np.zeros(k, dtype=bool)
        per = max(1, k // self.width)
        for b in range(self.width):
            bit = (val >> (self.width - 1 - b)) & 1
            lo, hi = b * per, (b + 1) * per if b < self.width - 1 else k
            group = out[lo:hi]
            mid = len(group) // 2
            if bit:
                group[mid:] = True
            else:
                group[:max(1, mid)] = True
            out[lo:hi] = group
        return out


class SymbolEncoder(InputEncoder):
    """Discrete symbol -> deterministic hash subset of the input population.

    ``symbols`` (if given) fixes the vocabulary; otherwise it grows on first use,
    which is what lets the same encoder serve tokens in later phases.
    """

    kind = "symbol"

    def __init__(self, *, symbols: Sequence[Any] | None = None, sparsity: float = 0.4, **kw: Any) -> None:
        super().__init__(**kw)
        self.symbols = list(symbols) if symbols is not None else None
        self.sparsity = float(sparsity)
        self._masks: dict[Any, np.ndarray] = {}

    @property
    def patterns(self) -> int:
        return len(self.symbols) if self.symbols is not None else max(2, len(self._vocab) + 1)

    def canonical(self, observation: Any) -> Any:
        return observation if not isinstance(observation, (list, tuple, np.ndarray)) else tuple(observation)

    def _mask_for(self, symbol: Any) -> np.ndarray:
        if symbol in self._masks:
            return self._masks[symbol]
        k = int(self.input_neurons.size)
        rng = np.random.default_rng(abs(hash((self.kind, str(symbol)))) % (2**32))
        n_on = max(1, int(round(k * self.sparsity)))
        m = np.zeros(k, dtype=bool)
        m[rng.choice(k, size=n_on, replace=False)] = True
        self._masks[symbol] = m
        return m

    def pattern_for(self, observation: Any) -> np.ndarray:
        return self._mask_for(self.canonical(observation))


class SequenceEncoder(InputEncoder):
    """Sequence of items -> one item per timestep (patterns, tokens, Luau code).

    The per-item encoder is injected rather than hardcoded, so phase 7 can supply
    a Luau-token encoder and phase 3 can supply a symbol encoder with no changes
    to this class.
    """

    kind = "sequence"

    def __init__(
        self,
        *,
        item_kind: str = "symbol",
        item_encoder: InputEncoder | None = None,
        symbols: Sequence[Any] | None = None,
        hold_last: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        if item_encoder is None:
            item_encoder = SymbolEncoder(symbols=symbols) if item_kind == "symbol" else build_encoder(item_kind)
        self.item_encoder = item_encoder
        self.hold_last = bool(hold_last)

    def configure(self, conn: Connectome, rng: np.random.Generator, *, vocab: Sequence[Any] | None = None) -> None:
        super().configure(conn, rng, vocab=vocab)
        self.item_encoder.configure(conn, rng, vocab=vocab)
        self.item_encoder.input_neurons = self.input_neurons
        self.item_encoder.amplitude = self.amplitude

    @property
    def patterns(self) -> int:
        return int(self.item_encoder.patterns)

    def pattern_for(self, observation: Sequence[Any]) -> np.ndarray:
        """Union of the per-item masks, for callers that need "which cells at all".

        The temporal detail lives in :meth:`encode`; this exists so the class is a
        complete :class:`InputEncoder` (it is an abstract method on the base) and so
        diagnostics can report the total driven footprint of a sequence.
        """
        k = int(self.input_neurons.size)
        m = np.zeros(k, dtype=bool)
        for item in list(observation):
            m |= np.asarray(self.item_encoder.pattern_for(item), dtype=bool)
        return m

    def encode(self, observation: Sequence[Any], *, n_steps: int | None = None) -> InjectionPlan:
        items = list(observation)
        steps = int(n_steps or max(1, len(items)))
        per_step: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(steps):
            if t >= len(items):
                # tail: hold the final symbol so the readout window always sees
                # something, or go silent -- configurable, never implicit
                if not self.hold_last or not items:
                    per_step.append((np.empty(0, np.int64), np.empty(0, np.float64)))
                    continue
                j = len(items) - 1
            else:
                j = t
            pat = np.asarray(self.item_encoder.pattern_for(items[j]), dtype=bool)
            idx = self.input_neurons[pat]
            per_step.append((idx, np.full(idx.size, self.amplitude, dtype=np.float64)))
        return InjectionPlan(n_steps=steps, per_step=per_step, detail={"length": len(items), "steps": steps})

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["item_encoder"] = self.item_encoder.state_dict()
        return d

    def load_state(self, d: Mapping[str, Any]) -> None:
        super().load_state(d)
        if d.get("item_encoder"):
            self.item_encoder.load_state(d["item_encoder"])


class ConstantCurrentEncoder(InputEncoder):
    """Raw analog drive: any array of amplitudes over the input neurons."""

    kind = "current"

    def pattern_for(self, observation: Any) -> np.ndarray:
        """All assigned cells, always: this encoder carries intensity, not identity.

        ``observation`` is read by :meth:`encode` as a per-cell amplitude vector
        (padded/truncated to the input population), which is the point of having a
        second encoder kind for the ablation "does symbolic targeting matter?".
        """
        k = int(self.input_neurons.size) if self.input_neurons is not None else self.n_input_neurons
        return np.ones(k, dtype=bool)

    @property
    def patterns(self) -> int:
        return 0

    def encode(self, observation: Any, *, n_steps: int | None = None) -> InjectionPlan:
        steps = int(n_steps or self.duration_steps)
        arr = np.asarray(observation, dtype=np.float64).ravel()
        k = int(self.input_neurons.size)
        if arr.size != k:
            tmp = np.zeros(k)
            tmp[: min(k, arr.size)] = arr[: min(k, arr.size)]
            arr = tmp
        per_step = [(self.input_neurons, arr.copy()) for _ in range(steps)]
        return InjectionPlan(n_steps=steps, per_step=per_step)


_ENCODERS: dict[str, type[InputEncoder]] = {
    cls.kind: cls for cls in (BinaryEncoder, BitVectorEncoder, SymbolEncoder, SequenceEncoder, ConstantCurrentEncoder)
}


def register_encoder(cls: type[InputEncoder]) -> type[InputEncoder]:
    _ENCODERS[cls.kind] = cls
    return cls


def build_encoder(spec: str | Mapping[str, Any] | None) -> InputEncoder:
    if spec is None:
        return BinaryEncoder()
    if isinstance(spec, InputEncoder):
        return spec
    if isinstance(spec, Mapping):
        spec = dict(spec)
        kind = str(spec.pop("kind", "binary"))
        kwargs = dict(spec.pop("kwargs", {}))
        kwargs.update(spec)
    else:
        kind, kwargs = str(spec), {}
    if kind not in _ENCODERS:
        raise ValueError(f"unknown encoder {kind!r}; available: {sorted(_ENCODERS)}")
    return _ENCODERS[kind](**kwargs)


def available_encoders() -> list[str]:
    return sorted(_ENCODERS)
