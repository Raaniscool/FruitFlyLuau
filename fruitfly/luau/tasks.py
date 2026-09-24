"""Luau task specification objects (phase 7/8). Not wired into training yet.

A task is data + a reward function, never a stored answer key: ``expected`` is
used only to compute reward, and the evaluator judges *behaviour* (does the code
run and produce the required effect) rather than string equality.

Difficulty ladder, per the project roadmap:
    variables -> conditionals -> loops -> tables -> functions -> events
    -> Instances/properties -> services -> RemoteEvent/Function -> ModuleScripts
    -> client/server -> CFrame/Vector3/Humanoid/GUI -> complete small games
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

#: Which Roblox-side surface a task touches. Used to report coverage honestly.
CATEGORIES = (
    "syntax",
    "variables",
    "conditionals",
    "loops",
    "tables",
    "functions",
    "events",
    "instances",
    "properties",
    "services",
    "remotes",
    "modulescript",
    "cframe",
    "vector3",
    "gui",
    "client_server",
    "correction",
    "completion",
)


@dataclass
class TestCase:
    """One behavioural check run against the generated code in the sandbox."""

    #: pytest must not try to collect this as a test class of its own
    __test__ = False

    name: str
    call: str
    expect: Any = None
    kind: str = "value"  # value | error | no_error | property | contains


@dataclass
class LuauTask:
    """Instruction, context, expected behaviour, tests and reward function."""

    id: str
    instruction: str
    category: str
    difficulty: int = 1
    context: str = ""
    expected_snippet: str = ""
    tests: tuple[TestCase, ...] = ()
    #: reward knobs are configurable everywhere in this project
    reward_pass: float = 1.0
    reward_fail: float = -1.0
    reward_partial: bool = True
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {self.category!r}")

    def reward(self, *, n_passed: int, n_total: int, executed: bool = True) -> float:
        """Reward from measured test outcomes only.

        With ``reward_partial`` the value scales with the fraction passed, which
        gives a gradient on multi-test tasks; without it the task is strictly
        pass/fail. Both are reported, never mixed silently.
        """
        if not executed or n_total <= 0:
            return float(self.reward_fail)
        if n_passed >= n_total:
            return float(self.reward_pass)
        if self.reward_partial:
            frac = n_passed / n_total
            return float(self.reward_fail + (self.reward_pass - self.reward_fail) * frac)
        return float(self.reward_fail)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "n_tests": len(self.tests),
            "instruction": self.instruction,
            "tags": list(self.tags),
        }


# ------------------------------------------------------------------ seed tasks
#: A handful of phase-7 tasks so the schema is exercised by tests. The corpus for
#: real training is a separate deliverable (see DATA.md, "Luau dataset").
SEED_TASKS: tuple[LuauTask, ...] = (
    LuauTask(
        id="luau000_variable",
        instruction="Create a variable named coins and set it to 10.",
        category="variables",
        difficulty=1,
        expected_snippet="local coins = 10",
        tests=(TestCase("reads_back", "get('coins')", 10),),
        tags=("luau", "beginner"),
    ),
    LuauTask(
        id="luau001_conditional",
        instruction="Write a function isEnough(price, money) that returns true when money >= price.",
        category="conditionals",
        difficulty=2,
        tests=(
            TestCase("enough", "isEnough(4, 10)", True),
            TestCase("exact", "isEnough(10, 10)", True),
            TestCase("short", "isEnough(11, 10)", False),
        ),
    ),
    LuauTask(
        id="luau002_loop_table",
        instruction="Return a table of the squares of 1..n using a for loop.",
        category="loops",
        difficulty=3,
        tests=(TestCase("squares", "squares(4)", [1, 4, 9, 16]),),
    ),
    LuauTask(
        id="luau010_service",
        instruction="Get the Players service and store it in a local named Players.",
        category="services",
        difficulty=2,
        expected_snippet='local Players = game:GetService("Players")',
        tests=(TestCase("service_retrieved", "getService('Players')", "Players"),),
        meta={"needs_roblox_mock": True},
    ),
    LuauTask(
        id="luau011_remote",
        instruction="Fire a RemoteEvent named GiveCoins to the server with the player's name.",
        category="remotes",
        difficulty=4,
        tests=(TestCase("fired", "lastRemoteFire()", ("GiveCoins", "Steve")),),
        meta={"needs_roblox_mock": True},
    ),
)


def corpus(*, categories: Sequence[str] | None = None, max_difficulty: int = 99) -> list[LuauTask]:
    return [
        t
        for t in SEED_TASKS
        if (not categories or t.category in set(categories)) and t.difficulty <= max_difficulty
    ]


def split_corpus(seed: int = 0, train_fraction: float = 0.8) -> tuple[list[LuauTask], list[LuauTask]]:
    """Deterministic train/eval split -- evaluation tasks never enter training."""
    import random

    items = list(SEED_TASKS)
    random.Random(seed).shuffle(items)
    k = int(round(len(items) * train_fraction))
    return items[:k], items[k:]
