"""Small shared helpers: logging, determinism, formatting."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

_LOG_NAME = "fruitfly"


def get_logger(name: str | None = None, level: int | None = None) -> logging.Logger:
    """Project-namespaced logger with a single shared handler."""
    root = logging.getLogger(_LOG_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(handler)
        root.setLevel(int(os.environ.get("FRUITFLY_LOG_LEVEL", logging.INFO)))
    return root.getChild(name) if name else root


def set_verbosity(verbose: bool = True, quiet: bool = False) -> None:
    lvl = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.getLogger(_LOG_NAME).setLevel(lvl)


def seed_all(seed: int) -> None:
    """Seed Python and NumPy RNGs.

    Torch/CUDA seeding is added opportunistically if those packages happen to be
    installed; the project does not depend on them.
    """
    import random

    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed % (2**32))
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)
    except Exception:
        pass


def stable_hash(obj: Any) -> str:
    """Content hash of a JSON-serialisable config, for run identity/reproducibility."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:,.1f} TB"


def human_int(n: float) -> str:
    return f"{int(n):,}"


def table(rows: Iterable[Iterable[Any]], header: Iterable[str]) -> str:
    """Tiny fixed-width text table for CLI reports (no tabulate dependency)."""
    rows = [[str(c) for c in r] for r in rows]
    head = [str(h) for h in header]
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) if rows else len(head[i]) for i in range(len(head))]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(head)), "  ".join("-" * w for w in widths)]
    out += ["  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)) for r in rows]
    return "\n".join(out)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
