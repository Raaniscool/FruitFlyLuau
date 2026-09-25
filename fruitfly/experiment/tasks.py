"""Experiments 001-005: the deliberately simple tasks that gate everything later.

Each task states what generalisation means for it, because that distinction
matters: XOR/binary have a tiny exhaustive input space, so "learning" there is
associative; the pattern, sequence and symbolic tasks carry genuinely held-out
items. The runner reports both, and documentation never blurs them.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..utils import get_logger
from .base import Environment, Trial, register_env

log = get_logger(__name__)


@register_env("exp001_binary")
class BinaryClassification(Environment):
    """Input 0/1 -> output 0/1. Reward +1 correct / -1 wrong (magnitudes configurable).

    ``mapping="invert"`` flips the required answer, which is how we check the
    system learns the association rather than a fixed motor bias towards one
    output group.
    """

    classes: Sequence[Any] = (0, 1)
    #: the binary task has exactly two stimuli, so eval items must overlap train
    exhaustive_input_space = True
    encoder_kind = "binary"
    decoder_kind = "group_rate"
    trial_steps = 48
    #: 4 input pairs, fully enumerated: eval items necessarily reuse train items
    exhaustive_input_space = True  # only 2 distinct inputs exist: generalisation is not testable here

    def __init__(self, *, mapping: str = "identity", **kw: Any) -> None:
        super().__init__(**kw)
        self.mapping = mapping

    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        items: list[Trial] = []
        for i in range(int(n)):
            x = int((i + offset) % 2)
            y = x if self.mapping == "identity" else 1 - x
            items.append(Trial(observation=x, target=y, key=f"bin:{x}"))
        return items

    def describe(self) -> dict:
        return {**super().describe(), "mapping": self.mapping, "exhaustive_input_space": True,
                "generalisation": "not applicable (2 stimuli); associative learning only"}


@register_env("exp002_xor")
class XOREnvironment(Environment):
    """(a, b) -> a XOR b over the four possible inputs.

    The classic non-linearly-separable mapping, used here as a plasticity test on
    a recurrent connectome rather than as a feed-forward function-approximation
    test.
    """

    classes: Sequence[Any] = (0, 1)
    encoder_kind = "bitvector"
    decoder_kind = "group_rate"
    trial_steps = 48
    #: 4 input pairs, fully enumerated: eval items necessarily reuse train items
    exhaustive_input_space = True

    def __init__(self, *, operation: str = "xor", **kw: Any) -> None:
        super().__init__(**kw)
        self.operation = operation
        self._ops = {
            "xor": lambda a, b: int(a) ^ int(b),
            "and": lambda a, b: int(a) & int(b),
            "or": lambda a, b: int(a) | int(b),
            "xnor": lambda a, b: 1 - (int(a) ^ int(b)),
        }
        if operation not in self._ops:
            raise ValueError(f"unknown operation {operation!r}; have {sorted(self._ops)}")

    @property
    def encoder_kwargs(self) -> dict:
        return {"width": 2}

    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        f = self._ops[self.operation]
        combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        items: list[Trial] = []
        for i in range(int(n)):
            a, b = combos[(i + offset) % 4]
            items.append(Trial(observation=(a, b), target=f(a, b), key=f"xor:{a}{b}"))
        return items

    def describe(self) -> dict:
        return {**super().describe(), "operation": self.operation, "exhaustive_input_space": True,
                "generalisation": "not applicable (4 stimuli); associative learning only"}


@register_env("exp003_pattern")
class PatternRecognition(Environment):
    """Distinguish K binary patterns, optionally corrupted by bit-flip noise.

    Held-out items are *different noise realisations* of the same prototypes, so
    accuracy here measures robustness, not memorisation of exact strings.
    """

    encoder_kind = "bitvector"
    decoder_kind = "group_rate"
    trial_steps = 16

    def __init__(
        self,
        *,
        n_patterns: int = 3,
        pattern_length: int = 8,
        noise: float = 0.15,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.n_patterns = int(n_patterns)
        self.pattern_length = int(pattern_length)
        self.noise = float(noise)
        self.classes = tuple(range(self.n_patterns))
        # prototypes fixed by seed, identical for train and eval streams
        prng = np.random.default_rng(self.seed + 31)
        self.prototypes = (prng.random((self.n_patterns, self.pattern_length)) > 0.5).astype(np.int64)

    @property
    def encoder_kwargs(self) -> dict:
        return {"width": self.pattern_length}

    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        items: list[Trial] = []
        for i in range(int(n)):
            cls = int((i + offset) % self.n_patterns)
            pat = self.prototypes[cls].copy()
            flips = rng.random(self.pattern_length) < self.noise
            pat = np.where(flips, 1 - pat, pat).astype(np.int64)
            items.append(Trial(observation=tuple(pat.tolist()), target=cls, key=f"pat{cls}:{pat.tobytes().hex()}"))
        return items

    def describe(self) -> dict:
        return {
            **super().describe(),
            "n_patterns": self.n_patterns,
            "pattern_length": self.pattern_length,
            "noise": self.noise,
            "generalisation": "held-out noise realisations of the same prototypes",
        }


@register_env("exp004_sequence")
class SequencePrediction(Environment):
    """Predict the next symbol of a deterministic cyclic sequence.

    The rule is ``next = (last + step) mod K`` with ``step`` fixed per run, so the
    target is a function of the tail of the input: the network must retain and
    transform activity, not just detect a stimulus.
    """

    encoder_kind = "sequence"
    decoder_kind = "group_rate"
    trial_steps = 48
    #: 4 starts x one fixed step: the input space is 4 sequences, fully enumerated
    exhaustive_input_space = True

    def __init__(self, *, alphabet: int = 4, seq_len: int = 4, step: int = 1, **kw: Any) -> None:
        super().__init__(**kw)
        self.alphabet = int(alphabet)
        self.seq_len = int(seq_len)
        self.step = int(step)
        self.classes = tuple(range(self.alphabet))

    @property
    def encoder_kwargs(self) -> dict:
        return {"item_kind": "symbol", "symbols": list(range(self.alphabet))}

    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        items: list[Trial] = []
        for i in range(int(n)):
            start = int((i + offset) % self.alphabet)
            seq = [(start + j * self.step) % self.alphabet for j in range(self.seq_len)]
            nxt = (seq[-1] + self.step) % self.alphabet
            items.append(Trial(observation=seq, target=nxt, key="seq:" + ",".join(map(str, seq))))
        return items

    def describe(self) -> dict:
        return {
            **super().describe(),
            "alphabet": self.alphabet,
            "seq_len": self.seq_len,
            "rule": f"next = (last + {self.step}) mod {self.alphabet}",
            "generalisation": "all cyclic rotations seen; target determined by input tail",
        }


@register_env("exp005_symbolic")
class SymbolicOperations(Environment):
    """Apply a symbolic rule to arguments: ``op a b -> result`` over Z_mod.

    Operators: ``add``/``sub`` (modular arithmetic) and ``swap`` (returns ``b``).
    This is the on-ramp to code: a Luau token sequence is exactly "apply
    syntactic rules to symbols".
    """

    encoder_kind = "sequence"
    decoder_kind = "group_rate"
    trial_steps = 48
    #: 4 starts x one fixed step: the input space is 4 sequences, fully enumerated
    exhaustive_input_space = True
    OPS = ("add", "sub", "copy")

    @property
    def exhaustive_input_space(self) -> bool:  # type: ignore[override]
        """True when ``ops x modulus x modulus`` cannot supply a novel eval item.

        At the default ``modulus=3`` there are 27 possible inputs, so a 40-trial
        training set already contains all of them and *no* held-out split exists.
        Rather than let a run quietly claim generalisation, the environment reports
        itself as exhaustive; ``modulus=16`` or larger gives a real held-out set.
        """
        return len(self.ops) * self.modulus * self.modulus <= (self.n_train + self.n_eval)

    def __init__(self, *, modulus: int = 3, ops: Sequence[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.modulus = int(modulus)
        self.ops = tuple(ops or self.OPS)
        bad = set(self.ops) - set(self.OPS)
        if bad:
            raise ValueError(f"unknown symbolic ops {sorted(bad)}; available {list(self.OPS)}")
        # vocabulary: operator tokens then value tokens
        self.symbols: list[Any] = list(self.ops) + list(range(self.modulus))
        self.classes = tuple(range(self.modulus))
        self._f = {
            "add": lambda a, b: (a + b) % self.modulus,
            "sub": lambda a, b: (a - b) % self.modulus,
            "copy": lambda a, b: b % self.modulus,
        }

    @property
    def encoder_kwargs(self) -> dict:
        return {"item_kind": "symbol", "symbols": list(self.symbols)}

    def build_items(self, n: int, *, rng: np.random.Generator, offset: int = 0) -> list[Trial]:
        items: list[Trial] = []
        for i in range(int(n)):
            op = self.ops[(i + offset) % len(self.ops)]
            a = int(rng.integers(0, self.modulus))
            b = int(rng.integers(0, self.modulus))
            res = self._f[op](a, b)
            items.append(Trial(observation=[op, a, b], target=res, key=f"sym:{op}:{a}:{b}"))
        return items

    def describe(self) -> dict:
        return {
            **super().describe(),
            "modulus": self.modulus,
            "ops": list(self.ops),
            "vocabulary": list(map(str, self.symbols)),
            "generalisation": (
                "held-out (a, b) combinations; operator must generalise across arguments"
                if not self.exhaustive_input_space
                else f"NOT held out: {len(self.ops) * self.modulus ** 2} possible inputs are "
                     f"all present in training at modulus={self.modulus}"
            ),
            "exhaustive_input_space": self.exhaustive_input_space,
        }
