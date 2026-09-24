"""Sandbox contract for evaluating generated Luau (phase 8).

This module intentionally cannot execute arbitrary code. It defines the safety
interface the project will use later, and the pieces that are safe to build now:

* a static checker (tokenizer-based) that flags forbidden constructs before
  anything is ever handed to a runtime;
* a hard refusal to run unless an explicit backend is configured, with the
  resource limits that must be present for that backend to be accepted;
* a ``MockRoblox`` *specification*: the surface a controlled Roblox mock must
  expose, so reward can be computed from behaviour instead of string equality.

Threat model, stated plainly: running model-generated code is remote-code
execution. ``subprocess`` alone is **not** a sandbox on Windows; the accepted
backends below are the ones that actually constrain it. Until such a backend
exists, ``execute`` raises and evaluation stays static.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .tokens import tokenize

#: Anything that can leave the sandbox, hang it, or read the host is rejected.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "require": "module import (can reach arbitrary code)",
    "loadstring": "runtime compiler",
    "load": "runtime compiler",
    "dofile": "file execution",
    "pcall": "error masking (allowed only in generated user code, not harness)",
    "os": "host OS access",
    "io": "filesystem access",
    "debug": "interpreter introspection",
    "string.dump": "bytecode exfiltration",
    "spawn": "process creation",
    "execute": "process creation",
}

#: Time/memory ceilings a backend must enforce to be considered sandboxed.
REQUIRED_LIMITS = {"timeout_s": (0.1, 10.0), "memory_mb": (16, 512), "max_output_bytes": (0, 1 << 20)}

#: The API surface a Roblox mock must provide for behaviour-based grading.
MOCK_ROBLOX_API: dict[str, Sequence[str]] = {
    "game": ("GetService", "Players", "Workspace", "ReplicatedStorage"),
    "workspace": ("CurrentCamera", "FindPartOnRay", "Children", "WaitForChild"),
    "Instance": ("new", "Parent", "Name", "Destroy", "GetPropertyChangedSignal", "FindFirstChildOfClass"),
    "Part": ("Position", "Size", "Color", "Anchored", "CanCollide", "Touched", "AssemblyLinearVelocity"),
    "Humanoid": ("Health", "MaxHealth", "MoveTo", "Jump", "Died", "TakeDamage"),
    "Vector3": ("new", "X", "Y", "Z", "Magnitude", "Unit", "Dot", "Cross"),
    "CFrame": ("new", "lookVector", "position", "ToWorld", "PointToWorldSpace"),
    "RemoteEvent": ("FireServer", "FireClient", "OnServerEvent", "OnClientEvent"),
    "RemoteFunction": ("InvokeServer", "OnServerInvoke"),
    "TaskDelay": ("task.delay", "task.wait", "task.spawn", "task.loop"),
    "GuiObject": ("Text", "Visible", "Position", "Size", "Activated"),
}


@dataclass
class StaticReport:
    """Result of the pre-execution safety scan."""

    ok: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    n_tokens: int = 0
    n_error_tokens: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "findings": self.findings,
            "n_tokens": self.n_tokens,
            "n_error_tokens": self.n_error_tokens,
            "metrics": self.metrics,
        }


@dataclass
class ExecResult:
    """Outcome of a sandboxed execution attempt."""

    ran: bool
    status: str  # ok | timeout | error | refused | unavailable
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    tests: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def n_passed(self) -> int:
        return sum(1 for t in self.tests if t.get("passed"))

    @property
    def n_total(self) -> int:
        return len(self.tests)

    def to_dict(self) -> dict:
        return {"ran": self.ran, "status": self.status, "stdout": self.stdout[-4000:],
                "stderr": self.stderr[-4000:], "exit_code": self.exit_code,
                "tests": self.tests, "error": self.error}


def scan_static(source: str, *, allow: Sequence[str] = ()) -> StaticReport:
    """Token-level safety scan. Never executes ``source``."""
    toks = tokenize(source)
    findings: list[dict[str, Any]] = []
    allowed = set(allow)
    joined = " ".join(t.text for t in toks if t.type in {"name", "operator", "keyword"})
    for name, why in FORBIDDEN_PATTERNS.items():
        if name in allowed:
            continue
        if name == "pcall":
            continue  # only the harness must not mask errors; user code may use it
        pat = rf"(?<![A-Za-z0-9_]){re.escape(name)}\b"
        if re.search(pat, joined):
            findings.append({"pattern": name, "why": why})
    return StaticReport(
        ok=not findings,
        findings=findings,
        n_tokens=len(toks),
        n_error_tokens=sum(1 for t in toks if t.type == "error"),
        metrics={
            "n_newlines": sum(1 for t in toks if t.type == "newline"),
            "max_line_len": max([len(ln) for ln in source.splitlines()] or [0]),
            "n_chars": len(source),
        },
    )


def backend_available(name: str) -> tuple[bool, str]:
    """Can this machine actually honour the requested sandbox backend?"""
    if name in {"", "none", None}:
        return False, (
            "no sandbox backend configured. Generated-code execution is disabled by "
            "default: this project will not run model output on your machine until a "
            "real sandbox is set up (luau.eval.sandbox = 'firejail' | 'docker' | 'windows_job')."
        )
    if name == "firejail":
        return (shutil.which("firejail") is not None), "firejail binary present" if shutil.which("firejail") else "firejail not on PATH"
    if name == "docker":
        return (shutil.which("docker") is not None), "docker binary present" if shutil.which("docker") else "docker not on PATH"
    if name == "windows_job":
        return (sys.platform.startswith("win")), "windows Job Object + restricted token required (not implemented in v0.1)"
    return False, f"unknown backend {name!r}"


def validate_limits(timeout_s: float, memory_mb: int, max_output_bytes: int) -> list[str]:
    """Return a list of reasons the requested limits are not safe enough."""
    problems = []
    lo, hi = REQUIRED_LIMITS["timeout_s"]
    if not (lo <= float(timeout_s) <= hi):
        problems.append(f"timeout_s={timeout_s} outside safe range [{lo}, {hi}]")
    lo, hi = REQUIRED_LIMITS["memory_mb"]
    if not (lo <= float(memory_mb) <= hi):
        problems.append(f"memory_mb={memory_mb} outside safe range [{lo}, {hi}]")
    if int(max_output_bytes) > REQUIRED_LIMITS["max_output_bytes"][1]:
        problems.append("max_output_bytes too large (output flooding)")
    return problems


def execute(
    source: str,
    *,
    backend: str = "",
    timeout_s: float = 2.0,
    memory_mb: int = 256,
    max_output_bytes: int = 65536,
    interpreter: str = "luau-analyze",
    env: Mapping[str, str] | None = None,
) -> ExecResult:
    """Refuse-or-run under a verified sandbox. v0.1 always refuses.

    The refusal is the point: an evaluator that "tries its best" to be safe is how
    arbitrary code execution bugs ship. When a backend is implemented, it must (a)
    pass :func:`validate_limits`, (b) expose no filesystem/network/job objects, and
    (c) drive the mock Roblox surface in :data:`MOCK_ROBLOX_API` so tests can assert
    behaviour rather than text.
    """
    ok, why = backend_available(backend)
    if not ok:
        return ExecResult(ran=False, status="refused", error=why)
    problems = validate_limits(timeout_s, memory_mb, max_output_bytes)
    if problems:
        return ExecResult(ran=False, status="refused", error="; ".join(problems))
    static = scan_static(source)
    if not static.ok:
        return ExecResult(ran=False, status="refused", error=f"static scan blocked: {static.findings}")
    return ExecResult(
        ran=False,
        status="unavailable",
        error=(
            f"backend {backend!r} passed configuration checks but no runner is implemented in "
            "v0.1. Implement SandboxRunner.run() (subprocess + rlimits + network down + mock Roblox "
            "harness) before enabling generated-code execution. See DEVELOPMENT.md 'phase 8'."
        ),
    )


#: The harness contract a real runner must satisfy, kept as data so tests can assert
#: on it and phase 8 can be verified against a checklist instead of prose.
HARNESS_REQUIREMENTS: tuple[str, ...] = (
    "isolated process, killed on timeout, no shell=True",
    "no filesystem access (read-only empty cwd or tmpfs, nothing outside it)",
    "no network namespace access",
    "memory cap enforced by the OS (rlimit / Job Object), not by the guest",
    "wall-clock and CPU-time caps",
    "stdout/stderr size capped",
    "Roblox API provided only via the mock in MOCK_ROBLOX_API",
    "tests graded by observable mock state (calls, property writes), never by string equality",
    "deterministic: fixed seed, no host time, no environment leakage",
)


def self_check() -> dict:
    """Report what the evaluator can and cannot do on this machine right now."""
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "static_scan_available": True,
        "backends": {b: backend_available(b)[1] for b in ("", "firejail", "docker", "windows_job")},
        "luau_on_path": bool(shutil.which("luau") or shutil.which("luau-stdio")),
        "requirements_not_yet_met": list(HARNESS_REQUIREMENTS),
        "status": "static-only; execution disabled",
    }
