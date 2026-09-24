"""Discover what is actually inside a FAFB data directory.

Nothing here trusts file *names* alone: each candidate is opened, its real
header row sniffed, and its columns matched against the logical schema. That
matters because the same logical asset arrives as ``.csv.gz``, plain ``.csv``,
or a ``.feather``/``.parquet`` export from unofficial mirrors, and because a
portal label ("Connections (Filtered)") is not a file name.
"""

from __future__ import annotations

import gzip
import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import get_logger, human_bytes
from .assets import ASSETS, AssetSpec
from .schema import (
    CONNECTION_FIELDS,
    COORD_FIELDS,
    SYNAPSE_FIELDS,
    FieldSpec,
    NEURON_ID_FIELDS,
    SchemaMatch,
    match_schema,
)

log = get_logger(__name__)

_MAGIC = {
    b"\x1f\x8b": "gzip",
    b"PAR1": "parquet",
    b"ARROW1": "feather",
    b"PK\x03\x04": "zip",
}


@dataclass
class FileProbe:
    """What we learned by opening one file."""

    path: Path
    size_bytes: int
    container: str
    header: tuple[str, ...] = ()
    n_rows_estimate: str = ""
    sample_row: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def is_tabular(self) -> bool:
        return bool(self.header) and self.container in {"csv", "gzip", "csv.gz", "parquet", "feather"}


@dataclass
class AssetMatch:
    """One asset key matched to a concrete file on disk."""

    key: str
    display: str
    spec: AssetSpec
    probe: FileProbe | None = None
    schema: SchemaMatch | None = None
    score: float = 0.0
    why: str = ""

    @property
    def path(self) -> Path | None:
        return self.probe.path if self.probe else None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "display": self.display,
            "file": str(self.path.name) if self.path else None,
            "size_bytes": self.probe.size_bytes if self.probe else None,
            "container": self.probe.container if self.probe else None,
            "header": list(self.probe.header) if self.probe else [],
            "resolved_columns": dict(self.schema.resolved) if self.schema else {},
            "missing": list(self.schema.missing) if self.schema else [],
            "score": round(self.score, 3),
            "why": self.why,
        }


@dataclass
class DatasetInventory:
    """Full picture of a data directory: matched assets plus unmatched extras."""

    data_dir: Path
    matches: dict[str, AssetMatch] = field(default_factory=dict)
    unmatched: list[FileProbe] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def connections(self) -> AssetMatch | None:
        for key in ("connections_filtered", "connections_unfiltered", "connections_legacy"):
            m = self.matches.get(key)
            if m is not None and m.probe is not None:
                return m
        return None

    def get(self, key: str) -> AssetMatch | None:
        return self.matches.get(key)

    def require(self, key: str) -> Path:
        m = self.matches.get(key)
        if m is None or m.path is None:
            spec = ASSETS[key]
            raise FileNotFoundError(
                f"asset {key!r} ({spec.display}) not found in {self.data_dir}.\n"
                f"Expected one of: {', '.join((spec.file,) + tuple(spec.alternates))}\n"
                f"Portal name to look for when downloading: '{spec.display}'."
            )
        return m.path

    def summary(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "matched": {k: v.to_dict() for k, v in self.matches.items()},
            "unmatched_files": [{"file": p.path.name, "size": p.size_bytes, "container": p.container} for p in self.unmatched],
            "notes": self.notes,
        }

    def report_text(self) -> str:
        lines = [f"FAFB data directory: {self.data_dir}", ""]
        lines.append(f"{'asset':<28} {'portal label':<34} {'file':<38} {'size':>10}  status")
        lines.append("-" * 132)
        for key, spec in ASSETS.items():
            m = self.matches.get(key)
            if m and m.probe:
                status = "matched" + ("" if (m.schema and m.schema.ok) else " (schema gap!)")
                lines.append(
                    f"{key:<28} {spec.display[:33]:<34} {m.probe.path.name[:37]:<38} "
                    f"{human_bytes(m.probe.size_bytes):>10}  {status}"
                )
            else:
                lines.append(
                    f"{key:<28} {spec.display[:33]:<34} {'-':<38} {'-':>10}  absent"
                    + ("  (optional)" if not spec.required_for else "  (REQUIRED)")
                )
        for p in self.unmatched:
            lines.append(f"{'-':<28} {'-':<34} {p.path.name[:37]:<38} {human_bytes(p.size_bytes):>10}  unrecognised")
        if self.notes:
            lines.append("")
            lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------- io
def _sniff_container(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(4)
    for magic, name in _MAGIC.items():
        if head.startswith(magic):
            return "csv.gz" if name == "gzip" else name
    return "csv" if path.suffix.lower() in {".csv", ".tsv"} else "unknown"


def _read_text_head(path: Path, n_bytes: int = 65536) -> str:
    if path.suffix.lower() in {".gz", ".gzip"} or _sniff_container(path) == "csv.gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
            return fh.read(n_bytes)
    with open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read(n_bytes)


def _delimiter(sample: str) -> str:
    first = sample.split("\n", 1)[0]
    counts = {d: first.count(d) for d in (",", "\t", ";")}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else ","


def probe_file(path: Path, *, count_rows: bool = False) -> FileProbe:
    """Open one file and learn its real format/schema."""
    try:
        size = path.stat().st_size
    except OSError as exc:  # pragma: no cover
        return FileProbe(path=path, size_bytes=0, container="error", error=str(exc))
    container = _sniff_container(path)
    probe = FileProbe(path=path, size_bytes=size, container=container)
    try:
        if container in {"csv", "csv.gz"}:
            sample = _read_text_head(path)
            lines = [ln for ln in sample.splitlines() if ln.strip()][:2]
            if not lines:
                probe.error = "file appears empty"
                return probe
            delim = _delimiter(lines[0])
            import csv

            reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
            rows = list(reader)
            probe.header = tuple(h.strip() for h in rows[0])
            if len(rows) > 1:
                probe.sample_row = dict(zip(probe.header, rows[1]))
            probe.n_rows_estimate = f">{len(lines) - 1}" if len(lines) == 2 else str(max(0, len(lines) - 1))
            if count_rows:
                n = -1  # header
                with (gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") if container == "csv.gz" else open(path, "rt", encoding="utf-8", errors="replace", newline="")) as fh:
                    for _ in fh:
                        n += 1
                probe.n_rows_estimate = f"{n:,}"
        elif container in {"parquet", "feather"}:
            try:
                import pyarrow.parquet as pq  # type: ignore
                import pyarrow.feather as feather  # type: ignore

                if container == "parquet":
                    pf = pq.ParquetFile(path)
                    probe.header = tuple(pf.schema_arrow.names)
                    probe.n_rows_estimate = f"{pf.metadata.num_rows:,}"
                else:
                    tbl = feather.read_table(path, columns=None)
                    probe.header = tuple(tbl.schema.names)
                    probe.n_rows_estimate = f"{tbl.num_rows:,}"
            except Exception as exc:
                probe.error = f"{exc.__class__.__name__}: {exc}"
        elif container == "zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()
                probe.header = tuple(names[:5])
                probe.n_rows_estimate = f"{len(names):,} members"
            except Exception as exc:  # pragma: no cover
                probe.error = str(exc)
    except Exception as exc:
        probe.error = f"{exc.__class__.__name__}: {exc}"
    return probe


def _score_match(spec: AssetSpec, probe: FileProbe) -> tuple[float, str]:
    """Confidence that ``probe`` is the content ``spec`` describes."""
    score, reasons = 0.0, []
    name = probe.path.name.lower()
    if name == spec.file.lower():
        score += 1.0
        reasons.append("exact filename")
    elif name in (a.lower() for a in spec.alternates):
        score += 0.85
        reasons.append("known alternate filename")
    elif any(tok in name for tok in Path(spec.file).stem.lower().replace("_", " ").split() if len(tok) > 3):
        score += 0.35
        reasons.append("filename tokens overlap")

    cols = {_norm(c): c for c in probe.header}
    if spec.columns:
        overlap = sum(1 for c in spec.columns if _norm(c) in cols)
        frac = overlap / len(spec.columns)
        score += 0.6 * frac
        reasons.append(f"columns {overlap}/{len(spec.columns)}")
    return score, "; ".join(reasons)


def _norm(col: str) -> str:
    return "".join(ch for ch in col.strip().lower() if ch.isalnum() or ch == "_")


def _fields_for(spec: AssetSpec) -> tuple[FieldSpec, ...]:
    """Which logical schema a file of this asset should satisfy."""
    if "connections" in spec.key:
        return CONNECTION_FIELDS
    if spec.key == "synapse_table":
        return SYNAPSE_FIELDS
    if spec.key == "coordinates":
        return COORD_FIELDS
    return NEURON_ID_FIELDS


def discover(data_dir: str | os.PathLike[str], *, count_rows: bool = False) -> DatasetInventory:
    """Scan ``data_dir`` (and one level down) and match files to FAFB assets."""
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    inv = DatasetInventory(data_dir=root)

    candidates: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        # don't wander into 13 GB of skeleton extraction output
        if p.suffix.lower() == ".swc":
            continue
        if len(p.parts) - len(root.parts) > 2:
            continue
        candidates.append(p)
    if not candidates:
        inv.notes.append("directory contains no files")
        return inv

    probes = [probe_file(p, count_rows=count_rows) for p in candidates]
    by_key: dict[str, AssetMatch] = {}
    claimed: set[Path] = set()

    for spec in ASSETS.values():
        best: AssetMatch | None = None
        for probe in probes:
            if probe.path in claimed or not (probe.is_tabular or spec.key == "skeletons"):
                continue
            score, why = _score_match(spec, probe)
            if score < 0.5:
                continue
            schema = None
            if probe.is_tabular and probe.header:
                schema = match_schema(probe.header, _fields_for(spec))
                if not schema.ok:
                    score -= 0.5  # wrong shape: prefer another file
            if score < 0.5:
                continue
            if best is None or score > best.score:
                best = AssetMatch(key=spec.key, display=spec.display, spec=spec, probe=probe, schema=schema, score=score, why=why)
        if best is not None:
            by_key[spec.key] = best
            claimed.add(best.probe.path)  # type: ignore[union-attr]

    inv.matches = by_key
    inv.unmatched = [p for p in probes if p.path not in claimed]
    if "connections_filtered" not in by_key:
        inv.notes.append(
            "no 'Connections (Filtered)' file matched; the loader can still work from "
            "connections_unfiltered/legacy or a cached .npz built by inspect_fafb.py"
        )
    if by_key.get("connections_filtered") and by_key["connections_filtered"].schema and not by_key["connections_filtered"].schema.ok:
        inv.notes.append(
            "matched connections file is missing expected columns -> loader will report the real header"
        )
    return inv
