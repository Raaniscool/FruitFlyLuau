"""Luau support -- phase 7 scaffolding, deliberately NOT a training target yet.

What exists here and why:

* :mod:`fruitfly.luau.tokens` -- a real, tested Luau tokeniser. It is needed
  anyway by experiment 005-style symbolic tasks (it is just another symbol
  sequence), so building and testing it now costs nothing and de-risks phase 7.
* :mod:`fruitfly.luau.tasks` -- task *specification* objects (instruction,
  context, expected behaviour, tests, reward). No training loop consumes them yet.
* :mod:`fruitfly.luau.evaluator` -- the sandbox contract. It refuses to run
  anything unless an explicit sandbox backend is configured, and it does not
  claim Roblox coverage: a mock API is required for that and is not built.

Nothing in ``phases 1-6`` imports this package. If you are reading this to find
out whether the system can write Luau: it cannot yet, and ``EXPERIMENTS.md``
records where the foundation actually stands.
"""

from .tokens import Token, tokenize, VOCAB_PRELUDE, LUAU_KEYWORDS, ROBLOX_GLOBALS

__all__ = ["Token", "tokenize", "VOCAB_PRELUDE", "LUAU_KEYWORDS", "ROBLOX_GLOBALS", "PHASE_STATUS"]

PHASE_STATUS = ("scaffold only: tokeniser + task specs + sandbox contract; this build "
               "cannot execute Luau and does not train on it yet")
