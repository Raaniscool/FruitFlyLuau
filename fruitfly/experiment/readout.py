"""Readout mapping: making sure the task is not already solved at initialisation.

Motivation, measured rather than assumed: during development exp001 scored 1.00
*before* any training, because the sink neurons of a random connected subgraph
happen to separate the two input patterns by chance. A task the untrained network
already solves cannot demonstrate learning.

So we probe every group->class assignment (all ``K!`` permutations for K classes,
K <= 5) and pick the one with the **lowest** initial accuracy. The network then
has to overcome its own pre-existing bias through plasticity. Everything about
this choice is recorded in ``setup.json`` (the full accuracy table, the chosen
permutation) so a reviewer can see that we hardened the benchmark rather than
softened it, and it can be disabled with ``graph.counterbalance_init: false``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)

MAX_PERMUTATIONS = 120  # 5! -- beyond this we stop enumerating and use the identity map


@dataclass
class MappingSearch:
    """Outcome of a readout-mapping search."""

    performed: bool
    chosen: tuple[int, ...]
    table: list[dict[str, Any]] = field(default_factory=list)
    best_accuracy: float | None = None
    identity_accuracy: float | None = None
    reason: str = ""

    def describe(self) -> str:
        if not self.performed:
            return f"readout mapping: identity ({self.reason})"
        return (
            f"readout mapping: permutation {list(self.chosen)} chosen from {len(self.table)} "
            f"candidates; initial accuracy {self.best_accuracy:.3f} (identity would give "
            f"{self.identity_accuracy:.3f})"
        )


def probe_accuracy(runner: Any, items: Sequence[Any], perm: Sequence[int]) -> float:
    """Fraction of ``items`` the CURRENT network gets wrong, given a group->class perm.

    Uses the runner's own encode/simulate path so the probe measures exactly what
    training will later optimise against.
    """
    dec = runner.decoder
    original = tuple(dec.classes)
    try:
        dec.classes = tuple(original[i] for i in perm) if len(perm) == len(original) else original
        n_correct = 0
        for trial in items:
            rec = runner.run_trial(trial, learn=False, episode=-1)
            n_correct += int(rec.correct)
        return n_correct / max(1, len(items))
    finally:
        dec.classes = original


def search_mapping(
    runner: Any,
    items: Sequence[Any],
    *,
    enabled: bool = True,
    maximize: bool = False,
) -> MappingSearch:
    """Pick the group->class assignment (default: the hardest one).

    ``maximize=True`` would instead pick the easiest assignment; it exists so the
    "did we accidentally make this impossible?" question can be answered with a
    number rather than a reassurance.
    """
    k = len(runner.env.classes)
    identity = tuple(range(k))
    if not enabled:
        return MappingSearch(False, identity, reason="disabled by config")
    if k < 2:
        return MappingSearch(False, identity, reason="single class")
    perms = list(itertools.permutations(range(k)))
    if len(perms) > MAX_PERMUTATIONS:
        return MappingSearch(False, identity, reason=f"{len(perms)} permutations > {MAX_PERMUTATIONS}")

    table = []
    for perm in perms:
        acc = probe_accuracy(runner, items, perm)
        table.append({"perm": list(perm), "initial_accuracy": round(acc, 4)})
    ranked = sorted(table, key=lambda r: r["initial_accuracy"], reverse=maximize)
    chosen = tuple(ranked[0]["perm"])
    ident = next(r["initial_accuracy"] for r in table if tuple(r["perm"]) == identity)
    log.info(
        "readout mapping search over %d assignments: chosen %s (initial accuracy %.3f; identity %.3f)",
        len(table), list(chosen), ranked[0]["initial_accuracy"], ident,
    )
    return MappingSearch(
        performed=True, chosen=chosen, table=table,
        best_accuracy=ranked[0]["initial_accuracy"], identity_accuracy=ident,
        reason="hardest assignment selected" if not maximize else "easiest assignment selected",
    )
