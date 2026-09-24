"""Column-name resolution for FAFB assets.

Real-world exports of the same logical table use slightly different spellings
depending on provenance (Codex live portal vs Zenodo static snapshot vs an
unofficial mirror). This module never guesses by position: it looks for the
*documented* name, then a list of known aliases, and reports which spelling it
matched so the choice is auditable. If nothing matches for a required field, it
raises with the actual header line so the user can see what is really there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    """Logical field -> accepted spellings, most preferred first."""

    name: str
    aliases: tuple[str, ...]
    required: bool = True
    dtype: str = "int64"  # int64 | float | str | id

    def __post_init__(self) -> None:
        if not self.aliases:
            raise ValueError(f"FieldSpec {self.name} needs at least one alias")


def _norm(col: str) -> str:
    return "".join(ch for ch in col.strip().lower() if ch.isalnum() or ch == "_").strip("_")


def _loose(col: str) -> str:
    return "".join(ch for ch in col.strip().lower() if ch.isalnum())


@dataclass
class SchemaMatch:
    """Result of matching logical fields against an actual file header."""

    header: tuple[str, ...]
    resolved: dict[str, str] = field(default_factory=dict)  # logical -> real column
    missing: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def has(self, name: str) -> bool:
        return name in self.resolved

    def column(self, name: str) -> str:
        try:
            return self.resolved[name]
        except KeyError:
            raise KeyError(
                f"logical field {name!r} not present; resolved={self.resolved}; header={self.header}"
            ) from None

    @property
    def ok(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        lines = [f"header ({len(self.header)} cols): {', '.join(self.header)}"]
        for logical, real in self.resolved.items():
            lines.append(f"  {logical:<12s} -> {real!r}")
        if self.missing:
            lines.append(f"  MISSING required: {', '.join(self.missing)}")
        if self.unknown:
            lines.append(f"  unused columns:   {', '.join(self.unknown)}")
        return "\n".join(lines)


def match_schema(header: Sequence[str], specs: Iterable[FieldSpec]) -> SchemaMatch:
    """Resolve ``specs`` against an actual header row.

    Matching is exact-normalised first (case/separator-insensitive), then a
    loose alphanumeric comparison, and finally a startswith fallback for
    prefix-encoded columns such as ``pre_root_id_720575940`` (the Codex synapse
    table stores the shared root-id prefix in the header).
    """
    specs = list(specs)
    norm_map: dict[str, str] = {}
    loose_map: dict[str, str] = {}
    for col in header:
        norm_map.setdefault(_norm(col), col)
        loose_map.setdefault(_loose(col), col)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for spec in specs:
        found: str | None = None
        for alias in spec.aliases:
            found = norm_map.get(_norm(alias))
            if found is not None:
                break
        if found is None:
            for alias in spec.aliases:
                cand = [c for c in header if _loose(c).startswith(_loose(alias))]
                if len(cand) == 1:
                    found = cand[0]
                    break
        if found is None:
            if spec.required:
                missing.append(spec.name)
            continue
        resolved[spec.name] = found

    used = set(resolved.values())
    unknown = tuple(c for c in header if c not in used)
    return SchemaMatch(header=tuple(header), resolved=resolved, missing=tuple(missing), unknown=unknown)


# --------------------------------------------------------------------------
# Machine-local profiles
# --------------------------------------------------------------------------

PROFILE_ENV = "FAFB_PROFILE"
PROFILE_RELPATH = os.path.join("config", "fafb_profile.yaml")


def load_profile(path: "str | os.PathLike[str] | None" = None) -> dict:
    """Column spellings observed on *this* machine, by scripts/inspect_fafb.py.

    ``python scripts/inspect_fafb.py --write-profile`` writes
    ``config/fafb_profile.yaml`` (git-ignored; it contains local paths) recording,
    per asset, the filename and the real header. When present, the loader puts each
    profiled spelling first in the alias list, so an export that uses an undocumented
    column name works without editing this file. Returns ``{}`` when there is no
    profile -- the built-in aliases then decide everything, as they do in CI.
    """
    import json

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    elif os.environ.get(PROFILE_ENV):
        candidates.append(Path(os.environ[PROFILE_ENV]).expanduser())
    else:
        try:
            from ..paths import repo_root

            candidates.append(repo_root() / PROFILE_RELPATH)
        except Exception:  # pragma: no cover - paths must always resolve
            pass
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            text = cand.read_text(encoding="utf-8")
            if cand.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml

                    data = yaml.safe_load(text) or {}
                except ImportError:  # profile is readable without pyyaml
                    data = json.loads(text)
            else:
                data = json.loads(text)
        except Exception as exc:  # a broken profile must not break a run
            log.warning("ignoring unreadable data profile %s: %s", cand, exc)
            return {}
        if not isinstance(data, dict):
            return {}
        prof = data.get("assets") if isinstance(data.get("assets"), dict) else data
        return {k: v for k, v in prof.items() if isinstance(v, dict)}
    return {}


def specs_for(profile: Mapping[str, Any], asset_key: str, base: tuple[FieldSpec, ...]) -> tuple[FieldSpec, ...]:
    """Return ``base`` with this machine's spelling promoted to first alias.

    A profile entry may be keyed either by *logical* field (``pre``, ``weight``) or
    by one of the documented aliases (``pre_root_id``, ``syn_count``); both map to
    the real column name in your file. Keying by logical field is how you teach the
    loader a spelling nobody has seen before, without touching this source file.
    """
    entry = (profile or {}).get(asset_key) or {}
    columns = entry.get("columns") or {}
    if not isinstance(columns, Mapping) or not columns:
        return base
    by_norm = {_norm(str(k)): str(v) for k, v in columns.items()}
    by_loose = {_loose(str(k)): str(v) for k, v in columns.items()}
    out = []
    for spec in base:
        promoted: list[str] = []
        for key in (spec.name, *spec.aliases):
            real = by_norm.get(_norm(key)) or by_loose.get(_loose(key))
            if real and real not in promoted:
                promoted.append(real)
        aliases = tuple(dict.fromkeys([*promoted, *spec.aliases]))
        out.append(spec if aliases == spec.aliases else replace(spec, aliases=aliases))
    return tuple(out)


# --------------------------------------------------------------------------
# Logical field definitions per asset family
# --------------------------------------------------------------------------

#: Connections tables (filtered / unfiltered / legacy) all share this shape.
CONNECTION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("pre", ("pre_root_id", "pre_pt_root_id", "pre_rootid", "pre_tnode_id", "pre"), dtype="id"),
    FieldSpec("post", ("post_root_id", "post_pt_root_id", "post_rootid", "post_tnode_id", "post"), dtype="id"),
    FieldSpec("neuropil", ("neuropil", "brain_region", "region"), required=False, dtype="str"),
    FieldSpec("weight", ("syn_count", "count", "syncount", "num_syn", "weight", "count_pre", "edge_weight"), dtype="int64"),
    FieldSpec("nt_type", ("nt_type", "nt", "neurotransmitter", "transmitter"), required=False, dtype="str"),
    FieldSpec("pre_stem", ("pre_stem_name", "stem_name_pre"), required=False, dtype="str"),
    FieldSpec("post_stem", ("post_stem_name", "stem_name_post"), required=False, dtype="str"),
    FieldSpec("confidence", ("conf", "confidence", "conf_edge"), required=False, dtype="float"),
)

#: Per-neuron annotation tables (neurons.csv.gz, classification, cell types...).
NEURON_ID_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("root_id", ("root_id", "rootid", "pt_root_id", "segment_id", "cell_id", "id"), dtype="id"),
)

#: Marked neuron coordinates.
COORD_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("root_id", ("root_id", "pt_root_id"), dtype="id"),
    FieldSpec("position", ("position", "pos", "coordinates", "xyz"), required=False, dtype="str"),
    FieldSpec("x", ("x", "coord_x", "post_pt_position_x"), required=False, dtype="float"),
    FieldSpec("y", ("y", "coord_y", "post_pt_position_y"), required=False, dtype="float"),
    FieldSpec("z", ("z", "coord_z", "post_pt_position_z"), required=False, dtype="float"),
    FieldSpec("supervoxel_id", ("supervoxel_id", "sv"), required=False, dtype="id"),
)

#: Synapse table (root ids may be split as prefix-encoded suffix columns).
SYNAPSE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("pre", ("pre_root_id", "pre_pt_root_id"), dtype="id"),
    FieldSpec("post", ("post_root_id", "post_pt_root_id"), dtype="id"),
    FieldSpec("cx", ("ctr_x", "center_x", "x"), required=False, dtype="float"),
    FieldSpec("cy", ("ctr_y", "center_y", "y"), required=False, dtype="float"),
    FieldSpec("cz", ("ctr_z", "center_z", "z"), required=False, dtype="float"),
    FieldSpec("size", ("size", "syn_size", "n_voxels"), required=False, dtype="float"),
    FieldSpec("neuropil", ("neuropil",), required=False, dtype="str"),
)
