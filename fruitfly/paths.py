"""Locating the local FAFB v783 data directory.

The large FlyWire/FAFB assets are deliberately NOT stored in Git (see DATA.md).
They live on the user's machine. This module resolves, in priority order:

1. an explicit ``data_path`` argument
2. the ``FAFB_DATA_PATH`` environment variable
3. ``data.fafb_data_path`` in ``config/local.yaml`` (git-ignored, machine-local)
4. ``data.fafb_data_path`` in ``config/default.yaml``
5. ``.fafb_path`` file at the repo root (git-ignored)
6. OS-specific conventional locations, including the user's Downloads folder

Every resolution attempt is recorded in ``SearchTrail`` so that when data is
missing we can tell the user *exactly* which paths were tried, rather than
failing with a bare exception.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ENV_VAR = "FAFB_DATA_PATH"

#: Names that make a directory recognisable as a FAFB data directory.
_RECOGNITION_HINTS = (
    "connections_princeton.csv.gz",
    "connections_princeton_no_threshold.csv.gz",
    "neurons.csv.gz",
    "classification.csv.gz",
    "consolidated_cell_types.csv.gz",
    "coordinates.csv.gz",
    "connections_princeton.csv",
    "proofread_connections_783.feather",
)


def repo_root() -> Path:
    """Repository root (directory containing this package's parent)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class SearchTrail:
    """Diagnostic record of how the data directory was (or was not) found."""

    source: str | None = None
    path: Path | None = None
    tried: list[tuple[str, Path]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        if self.path is not None and self.source is not None:
            lines.append(f"FAFB data directory resolved from {self.source}: {self.path}")
        else:
            lines.append("FAFB data directory NOT found. Locations tried:")
        for label, p in self.tried:
            if p is None:
                continue
            exists = p.exists()
            kind = "dir " if p.is_dir() else ("file" if exists else "absent")
            lines.append(f"  [{kind}] via {label}: {p}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _candidate_roots() -> Iterable[tuple[str, Path]]:
    """OS-appropriate conventional places a user may have unpacked FAFB data."""
    home = Path.home()
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA")
        yield ("Downloads", home / "Downloads" / "FAFB_v783")
        yield ("Downloads", home / "Downloads" / "fafb_v783")
        yield ("Downloads", home / "Downloads")
        yield ("USERPROFILE", home / "Downloads" / "FlyWire")
        if local:
            # conditional: LOCALAPPDATA is normally set on Windows but is absent in
            # a sanitised environment. The old form put a bare None in the list,
            # which blew up in the caller's `for label, p in _candidate_roots()`.
            yield ("LOCALAPPDATA", Path(local) / "FruitFly" / "data" / "FAFB_v783")
        yield ("drive", Path("D:/FAFB_v783"))
        yield ("drive", Path("D:/data/FAFB_v783"))
    else:
        yield from [
            ("XDG_DATA_HOME", Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "fafb" / "FAFB_v783"),
            ("data", home / "data" / "FAFB_v783"),
            ("data", home / "Data" / "FAFB_v783"),
            ("Downloads", home / "Downloads" / "FAFB_v783"),
            ("repo", repo_root() / "data" / "fafb"),
        ]
        if sys.platform == "darwin":
            yield ("Downloads", home / "Downloads")


def _looks_like_fafb_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    try:
        names = {q.name for q in p.iterdir()}
    except OSError:
        return False
    if any(h in names for h in _RECOGNITION_HINTS):
        return True
    # nested single folder (e.g. Downloads/FAFB_v783/fafb_783_v1/)
    return False


def _read_config_values() -> list[tuple[str, str]]:
    """Pull ``data.fafb_data_path`` from YAML configs if present."""
    out: list[tuple[str, str]] = []
    try:
        import yaml
    except Exception:  # pyyaml not installed -> configs simply skipped
        return out
    for label, cfg in (
        ("config/local.yaml", repo_root() / "config" / "local.yaml"),
        ("config/default.yaml", repo_root() / "config" / "default.yaml"),
    ):
        if not cfg.is_file():
            continue
        try:
            data: Any = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # malformed config should not mask other sources
            out.append((label, ""))
            continue
        section = data.get("data") if isinstance(data, dict) else None
        if isinstance(section, dict):
            val = section.get("fafb_data_path")
            if val:
                out.append((label, str(val)))
    return out


def _read_tramp_file() -> str | None:
    """``.fafb_path`` at repo root: a one-line pointer, handy on Windows."""
    p = repo_root() / ".fafb_path"
    if p.is_file():
        try:
            txt = p.read_text(encoding="utf-8").strip()
            return txt or None
        except OSError:
            return None
    return None


def find_data_dir(explicit: str | os.PathLike[str] | None = None) -> tuple[Path | None, SearchTrail]:
    """Resolve the FAFB data directory.

    Returns ``(path, trail)``. ``path`` is ``None`` when nothing usable was
    found; ``trail`` then explains exactly which locations were tried.
    """
    trail = SearchTrail()
    attempts: list[tuple[str, Path]] = []

    if explicit:
        attempts.append(("explicit argument", Path(str(explicit)).expanduser()))
    env = os.environ.get(ENV_VAR)
    if env:
        attempts.append((f"${ENV_VAR}", Path(env).expanduser()))
    for label, raw in _read_config_values():
        if raw:
            attempts.append((label, Path(raw).expanduser()))
    tramp = _read_tramp_file()
    if tramp:
        attempts.append((".fafb_path", Path(tramp).expanduser()))
    for label, p in _candidate_roots():
        if p is not None:
            attempts.append((label, p))

    for label, p in attempts:
        if p is None:
            continue
        if _looks_like_fafb_dir(p):
            trail.source, trail.path = label, p
            trail.tried.append((label, p))
            return p, trail
        trail.tried.append((label, p))

    # A directory may exist but not contain a recognised file yet.
    for label, p in trail.tried:
        if p.is_dir():
            trail.notes.append(
                f"directory from {label} exists but contains no recognised FAFB file: {p}"
            )
    return None, trail


def require_data_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the data dir or raise with an actionable, cross-platform message."""
    p, trail = find_data_dir(explicit)
    if p is None:
        raise FileNotFoundError(
            "Could not locate the FAFB v783 data directory.\n\n"
            + trail.describe()
            + "\n\nFix it by either:\n"
            f"  1. setting the environment variable, e.g. (PowerShell)\n"
            f"       Set-Item -Path Env:{ENV_VAR} -Value \"$env:USERPROFILE\\Downloads\\FAFB_v783\"\n"
            "     (bash/zsh)\n"
            f"       export {ENV_VAR}=$HOME/data/FAFB_v783\n"
            f"  2. writing the path into config/local.yaml under `data.fafb_data_path`, or\n"
            f"  3. writing it into a one-line file named `.fafb_path` at the repo root.\n\n"
            "The directory should contain the Codex 'Download Data' files for FAFB v783\n"
            "(connections_princeton.csv.gz, neurons.csv.gz, classification.csv.gz, ...).\n"
            "Run `python scripts/inspect_fafb.py --dir <your folder>` to see what is actually there.\n"
            "No FAFB data is needed for the tiny/sample modes: use `fruitfly sample --write`."
        )
    return p
