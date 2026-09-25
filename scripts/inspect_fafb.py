#!/usr/bin/env python3
r"""Inspect your local FAFB v783 download and report its REAL schema.

Why this script exists: the project must not assume the shape of your files. This
is a standalone diagnostic (Python 3.10+, **stdlib only** -- no numpy/pandas
needed) designed to be run on your machine against the folder you downloaded,
e.g. your Downloads directory. It writes a machine-readable JSON report and a
human-readable markdown report that you can commit to this repository so the
loader can be validated against the actual bytes rather than against documentation.

It measures; it does not guess. Anything it cannot establish from the files is
written as `UNKNOWN -- requires further investigation`.

Usage (PowerShell):

    cd <path\\to>\\FruitFlyLuau
    python scripts/inspect_fafb.py --dir "$env:USERPROFILE\\Downloads" --count-rows --write-profile
    python scripts\\inspect_fafb.py --dir D:\\data\\FAFB_v783 --deep

Options
-------
--dir PATH        folder to scan (also searches one level of subfolders)
--out PREFIX      write <PREFIX>.json and <PREFIX>.md (default: fafb_schema_report)
--sample-rows N   rows per file to profile for types/missing values (default 250000)
--count-rows      also stream every row of each CSV to get exact row counts
--deep            full-file profiling + duplicate-pair and reciprocity analysis
                  (one extra pass over the connections table; slower, more RAM)
--include-large   also probe the ~2.7 GB synapse table and the ~13 GB skeleton zip
--write-profile   also emit config/fafb_profile.yaml that the loader will trust
                  over its built-in name/column guesses

Exit codes: 0 = found at least a connections table, 2 = no connections table
found (you may have pointed at the wrong folder), 1 = usage error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

UNKNOWN = "UNKNOWN -- requires further investigation"

ASSET_BY_FILENAME = {
    "connections_princeton.csv.gz": "connections_filtered",
    "connections_princeton.csv": "connections_filtered",
    "connections.csv.gz": "connections_filtered",
    "connections_princeton_no_threshold.csv.gz": "connections_unfiltered",
    "connections_princeton_no_threshold.csv": "connections_unfiltered",
    "connections_no_threshold.csv.gz": "connections_unfiltered",
    "connections_buhmann_no_threshold.csv.gz": "connections_legacy",
    "neurons.csv.gz": "nt_predictions",
    "neurons.csv": "nt_predictions",
    "classification.csv.gz": "classification",
    "consolidated_cell_types.csv.gz": "cell_types",
    "cell_stats.csv.gz": "cell_stats",
    "names.csv.gz": "names_groups",
    "visual_neuron_types.csv.gz": "visual_types",
    "column_assignment.csv.gz": "visual_columns",
    "labels.csv.gz": "labels_raw",
    "processed_labels.csv.gz": "labels_refined",
    "connectivity_tags.csv.gz": "connectivity_tags",
    "coordinates.csv.gz": "coordinates",
    "morphology_clusters.csv.gz": "morphology_clusters",
    "connectivity_clusters.csv.gz": "connectivity_clusters",
    "fafb_v783_princeton_synapse_table.csv.gz": "synapse_table",
    "synapse_table.csv.gz": "synapse_table",
    "sk_lod1_783_healed.zip": "skeletons",
    "synapse_coordinates.csv.gz": "synapse_coordinates_legacy",
    "synapse_attachment_rates.csv.gz": "attachment_rates_legacy",
}

# Assets that are *not* part of the modern v783 pipeline. Presence is fine; use is not.
LEGACY_ASSETS = {
    "connections_legacy": "Buhmann-era unthresholded connections; superseded by the Princeton tables.",
    "synapse_coordinates_legacy": "legacy per-synapse coordinate dump; v783 ships the Princeton synapse table.",
    "attachment_rates_legacy": "legacy QC product, not used for connectivity.",
}

LARGE_ASSETS = {"synapse_table", "skeletons"}

EXPECTED_COLUMNS = {
    "connections_filtered": ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
    "connections_unfiltered": ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
    "connections_legacy": ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
    "nt_predictions": ["root_id", "group", "nt_type", "nt_type_score", "da_avg", "ser_avg",
                       "gaba_avg", "glut_avg", "ach_avg", "oct_avg"],
    "classification": ["root_id", "flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"],
    "cell_types": ["root_id", "primary_type", "additional_types"],
    "cell_stats": ["root_id", "length_nm", "area_nm", "size_nm"],
    "names_groups": ["root_id", "name", "group"],
    "visual_types": ["root_id", "type", "family", "subsystem", "category", "side"],
    "visual_columns": ["root_id", "hemisphere", "type", "column_id", "x", "y", "p", "q"],
    "labels_raw": ["root_id", "label", "user_id", "position", "supervoxel_id", "label_id",
                   "date_created", "user_name", "user_affiliation"],
    "labels_refined": ["root_id", "processed_labels"],
    "connectivity_tags": ["root_id", "connectivity_tag"],
    "coordinates": ["root_id", "position", "supervoxel_id"],
    "synapse_table": ["pre_x", "pre_y", "pre_z", "ctr_x", "ctr_y", "ctr_z", "post_x", "post_y",
                      "post_z", "size", "pre_root_id_720575940", "post_root_id_720575940", "neuropil"],
}

# Which logical role each asset plays, used for the "canonical schema" section.
ROLE = {
    "connections_filtered": "edges (weighted, directed, per-neuropil)",
    "connections_unfiltered": "edges without the >=5 synapse threshold",
    "nt_predictions": "neuron table + neurotransmitter prediction",
    "classification": "neuron classification (flow/super_class/class/side/nerve)",
    "cell_types": "consolidated cell type per neuron",
    "visual_types": "visual-system neuron annotation",
    "visual_columns": "visual column assignment",
    "coordinates": "representative coordinate per neuron",
    "labels_raw": "free-text community labels",
    "labels_refined": "cleaned community labels",
    "cell_stats": "per-neuron morphometrics",
    "names_groups": "human-readable names/groups",
    "connectivity_tags": "derived connectivity tags",
    "synapse_table": "per-synapse table (very large)",
    "skeletons": "per-neuron skeletons (very large)",
}

REFERENCE = {
    "cells": 139255,
    "connections_rows_filtered": 5342446,
    "unique_pairs": 3732460,
    "synapses": 50666648,
    "neuropils": 79,
}

#: Only these are ever opened and parsed. A Downloads folder is full of .exe,
#: .msi and .zip files; reading them as CSV would be slow, useless and (with
#: --count-rows) would stream gigabytes of binary through the csv module.
TABULAR_SUFFIXES = {".csv", ".tsv", ".gz", ".gzip", ".feather", ".parquet"}

MAX_DISTINCT = 2000          # per-column distinct values tracked before giving up
MAX_IDS = 600_000            # per-file neuron ids retained for cross-file comparison
MAX_PAIRS = 8_000_000        # (pre, post) keys retained for duplicate/reciprocity analysis
SAMPLE_VALUES = 5


# --------------------------------------------------------------------------- utils
def _norm(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_").strip("_")


def _match_column(header: list[str], want: str) -> str | None:
    """The real header that corresponds to a documented column name, if any."""
    target = _norm(want)
    for col in header:
        if _norm(col) == target:
            return col
    loose = "".join(ch for ch in target if ch.isalnum())
    cands = [c for c in header if "".join(ch for ch in c.lower() if ch.isalnum()).startswith(loose)]
    return cands[0] if len(cands) == 1 else None


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "rt", encoding="utf-8", errors="replace", newline="")


def _kind(value: str) -> str:
    v = value.strip()
    if not v:
        return "missing"
    if v.lstrip("+-").isdigit():
        return "int"
    try:
        float(v)
        return "float"
    except ValueError:
        pass
    if v.lower() in {"true", "false"}:
        return "bool"
    return "str"


class ColumnProfile:
    """Streaming per-column statistics. Bounded memory by construction."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.kinds: Counter[str] = Counter()
        self.n = 0
        self.min_num: float | None = None
        self.max_num: float | None = None
        self.max_len = 0
        self.distinct: set[str] | None = set()
        self.examples: list[str] = []
        self.digit_lengths: Counter[int] = Counter()

    def add(self, value: str) -> None:
        self.n += 1
        k = _kind(value)
        self.kinds[k] += 1
        if k == "missing":
            return
        v = value.strip()
        self.max_len = max(self.max_len, len(v))
        if k in ("int", "float"):
            f = float(v)
            self.min_num = f if self.min_num is None else min(self.min_num, f)
            self.max_num = f if self.max_num is None else max(self.max_num, f)
            if k == "int":
                self.digit_lengths[len(v.lstrip("+-"))] += 1
        if self.distinct is not None:
            self.distinct.add(v)
            if len(self.distinct) > MAX_DISTINCT:
                self.distinct = None
        if len(self.examples) < SAMPLE_VALUES and v not in self.examples:
            self.examples.append(v)

    def as_dict(self) -> dict:
        non_missing = self.n - self.kinds.get("missing", 0)
        order = ["str", "float", "int", "bool"]
        inferred = "empty"
        for k in order:
            if self.kinds.get(k):
                inferred = k
                break
        if self.kinds.get("int") and self.kinds.get("float") and not self.kinds.get("str"):
            inferred = "float"
        big_int = bool(self.digit_lengths) and max(self.digit_lengths) >= 16
        return {
            "column": self.name,
            "inferred_type": inferred,
            "n_values": self.n,
            "n_missing": self.kinds.get("missing", 0),
            "pct_missing": round(100.0 * self.kinds.get("missing", 0) / self.n, 4) if self.n else 0.0,
            "kind_counts": dict(self.kinds),
            "min": self.min_num,
            "max": self.max_num,
            "max_str_len": self.max_len,
            "n_distinct": (len(self.distinct) if self.distinct is not None else f">{MAX_DISTINCT}"),
            "distinct_values": (sorted(self.distinct)[:40] if self.distinct is not None and len(self.distinct) <= 40 else None),
            "examples": self.examples,
            "int_digit_lengths": dict(sorted(self.digit_lengths.items())) or None,
            "exceeds_float64_exact_int": big_int and (self.max_num or 0) > 2 ** 53,
            "non_missing": non_missing,
        }


# --------------------------------------------------------------------------- file scan
def profile_file(path: Path, asset: str | None, *, sample_rows: int, count_rows: bool,
                 deep: bool) -> dict:
    info: dict = {
        "file": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "container": "csv.gz" if path.suffix.lower() == ".gz" else (path.suffix.lstrip(".").lower() or "unknown"),
        "asset": asset,
        "role": ROLE.get(asset or "", UNKNOWN),
        "header": [],
        "columns": [],
        "example_rows": [],
        "n_rows_profiled": 0,
        "n_rows_including_header": None,
        "row_count_is_exact": False,
        "error": "",
    }
    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            info["container"] = "zip"
            info["n_members"] = len(names)
            info["first_members"] = names[:5]
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    limit = None if (deep or count_rows) else sample_rows
    profiles: dict[str, ColumnProfile] = {}
    ids: set[int] = set()
    id_dups = 0
    id_col = None
    pair_keys: set[int] | None = set() if (deep and (asset or "").startswith("connections")) else None
    pair_dups = 0
    self_loops = 0
    pre_ids: set[int] = set()
    post_ids: set[int] = set()
    weight_sum = 0
    weight_col = None
    neuropils: set[str] = set()
    n = 0
    try:
        with _open_text(path) as fh:
            head = fh.read(1 << 16)
            if not head.strip():
                info["error"] = "file appears empty"
                return info
            if "\x00" in head[:4096]:
                info["error"] = "binary content -- not a CSV; not parsed"
                info["role"] = "not a table -- ignored"
                return info
            first_line = head.splitlines()[0]
            delim = max([",", "\t", ";"], key=first_line.count)
            info["delimiter"] = delim
        with _open_text(path) as fh:
            reader = csv.reader(fh, delimiter=delim)
            header = [h.strip() for h in next(reader)]
            info["header"] = header
            profiles = {c: ColumnProfile(c) for c in header}
            id_col = _match_column(header, "root_id")
            pre_col = _match_column(header, "pre_root_id") or _match_column(header, "pre_root_id_720575940")
            post_col = _match_column(header, "post_root_id") or _match_column(header, "post_root_id_720575940")
            weight_col = _match_column(header, "syn_count")
            np_col = _match_column(header, "neuropil")
            idx = {c: i for i, c in enumerate(header)}
            for row in reader:
                n += 1
                if limit is None or n <= limit:
                    for c in header:
                        i = idx[c]
                        profiles[c].add(row[i] if i < len(row) else "")
                    if len(info["example_rows"]) < 3:
                        info["example_rows"].append(dict(zip(header, row)))
                elif not count_rows:
                    break
                if id_col and n <= MAX_IDS:
                    v = row[idx[id_col]].strip() if idx[id_col] < len(row) else ""
                    if v.isdigit():
                        if int(v) in ids:
                            id_dups += 1
                        ids.add(int(v))
                if pre_col and post_col and idx[pre_col] < len(row) and idx[post_col] < len(row):
                    a, b = row[idx[pre_col]].strip(), row[idx[post_col]].strip()
                    if a.isdigit() and b.isdigit():
                        ai, bi = int(a), int(b)
                        if len(pre_ids) < MAX_IDS:
                            pre_ids.add(ai)
                        if len(post_ids) < MAX_IDS:
                            post_ids.add(bi)
                        if ai == bi:
                            self_loops += 1
                        if pair_keys is not None and len(pair_keys) < MAX_PAIRS:
                            key = ai * (1 << 64) + bi
                            if key in pair_keys:
                                pair_dups += 1
                            pair_keys.add(key)
                if weight_col and idx[weight_col] < len(row):
                    w = row[idx[weight_col]].strip()
                    if w.isdigit():
                        weight_sum += int(w)
                if np_col and len(neuropils) < 5000 and idx[np_col] < len(row):
                    s = row[idx[np_col]].strip()
                    if s:
                        neuropils.add(s)
        info["n_rows_profiled"] = min(n, limit) if limit else n
        if count_rows or deep or (limit and n <= limit):
            info["n_rows_including_header"] = n
            info["row_count_is_exact"] = True
        else:
            info["n_rows_including_header"] = None
        info["columns"] = [p.as_dict() for p in profiles.values()]
        if id_col:
            info["id_column"] = id_col
            info["n_unique_ids"] = len(ids)
            info["n_duplicate_id_rows"] = id_dups
            info["ids_truncated_at"] = MAX_IDS if n > MAX_IDS else None
        if pre_ids or post_ids:
            info["n_unique_pre"] = len(pre_ids)
            info["n_unique_post"] = len(post_ids)
            info["n_unique_nodes"] = len(pre_ids | post_ids)
            info["n_self_loops"] = self_loops
            info["sum_syn_count"] = weight_sum if weight_col else None
            info["n_neuropils_seen"] = len(neuropils) if neuropils else None
            info["neuropils_sample"] = sorted(neuropils)[:10] if neuropils else None
            if pair_keys is not None:
                info["n_unique_pairs"] = len(pair_keys)
                info["n_duplicate_pair_rows"] = pair_dups
                recip = sum(1 for k in pair_keys
                            if ((k % (1 << 64)) * (1 << 64) + (k >> 64)) in pair_keys)
                info["n_reciprocal_pairs"] = recip
                info["reciprocity_fraction"] = round(recip / len(pair_keys), 6) if pair_keys else None
                info["directed"] = ("yes -- (pre,post) and (post,pre) are distinct rows; "
                                    f"{recip:,}/{len(pair_keys):,} pairs have a reverse counterpart")
            else:
                info["n_unique_pairs"] = UNKNOWN + " (run with --deep)"
                info["directed"] = ("column names pre_root_id/post_root_id imply direction; "
                                    "reciprocity not measured (run with --deep)")
        # id sets kept out of the JSON, handed back for cross-file work
        info["_ids"] = ids or (pre_ids | post_ids)
    except Exception as exc:  # never die on one odd file
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def scan(root: Path, *, sample_rows: int, count_rows: bool, deep: bool, include_large: bool) -> dict:
    files: list[dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() == ".swc":
            continue
        if len(p.relative_to(root).parts) > 2:
            continue
        asset = ASSET_BY_FILENAME.get(p.name.lower())
        if asset is None and p.suffix.lower() not in TABULAR_SUFFIXES:
            # Inventory it, do not open it. This is what keeps the script usable
            # when --dir is a real Downloads folder rather than a clean FAFB folder.
            files.append({"file": p.name, "path": str(p), "size_bytes": p.stat().st_size,
                          "container": p.suffix.lstrip(".").lower() or "unknown", "asset": None,
                          "role": "not a table -- ignored", "header": [], "columns": [],
                          "note": "non-tabular file, not opened"})
            continue
        if asset is None and p.stat().st_size > 2_000_000_000:
            files.append({"file": p.name, "path": str(p), "size_bytes": p.stat().st_size,
                          "container": "skipped", "asset": None, "role": "unrecognised",
                          "header": [], "columns": [],
                          "note": "unrecognised file larger than 2 GB, not opened"})
            continue
        if not include_large and asset in LARGE_ASSETS:
            files.append({"file": p.name, "path": str(p), "size_bytes": p.stat().st_size,
                          "container": "skipped", "asset": asset, "role": ROLE.get(asset, ""),
                          "header": [], "columns": [],
                          "note": "large asset; re-run with --include-large to probe it"})
            continue
        info = profile_file(p, asset, sample_rows=sample_rows, count_rows=count_rows, deep=deep)
        if asset:
            exp = EXPECTED_COLUMNS.get(asset, [])
            got = {_norm(c) for c in info["header"]}
            info["expected_columns"] = exp
            info["columns_missing"] = [c for c in exp if _norm(c) not in got]
            info["columns_extra"] = [c for c in info["header"] if _norm(c) not in {_norm(e) for e in exp}]
            info["matches_documented_schema"] = not info["columns_missing"]
            if asset in LEGACY_ASSETS:
                info["legacy"] = LEGACY_ASSETS[asset]
        files.append(info)

    # cross-file id compatibility, computed before the id sets are dropped
    id_sets = {f.get("asset") or f["file"]: f.pop("_ids", set()) for f in files}
    id_sets = {k: v for k, v in id_sets.items() if v}
    ref_key = "connections_filtered" if "connections_filtered" in id_sets else (
        "nt_predictions" if "nt_predictions" in id_sets else next(iter(id_sets), None))
    cross = []
    if ref_key:
        ref = id_sets[ref_key]
        for key, s in sorted(id_sets.items()):
            inter = len(ref & s)
            cross.append({
                "asset": key, "n_ids": len(s), "n_shared_with_reference": inter,
                "pct_of_this_asset_in_reference": round(100.0 * inter / len(s), 3) if s else 0.0,
                "pct_of_reference_covered": round(100.0 * inter / len(ref), 3) if ref else 0.0,
            })
    for f in files:
        f.pop("_ids", None)

    rep = {
        "scanned_dir": str(root.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "options": {"sample_rows": sample_rows, "count_rows": count_rows, "deep": deep,
                    "include_large": include_large},
        "reference_counts": REFERENCE,
        "files": files,
        "n_files": len(files),
        "cross_file_ids": cross,
        "cross_file_reference": ref_key or UNKNOWN,
    }
    rep["hazards"] = find_hazards(rep)
    return rep


# --------------------------------------------------------------------------- hazards
def find_hazards(rep: dict) -> list[str]:
    out: list[str] = []
    big_int_cols: list[str] = []
    by_asset = {f.get("asset"): f for f in rep["files"] if f.get("asset")}
    for f in rep["files"]:
        if f.get("error"):
            out.append(f"`{f['file']}`: could not be read -- {f['error']}")
        if f.get("columns_missing"):
            out.append(f"`{f['file']}`: documented columns absent: {', '.join(f['columns_missing'])} "
                       "-> add the real name as an alias in fruitfly/dataset/schema.py or use --write-profile.")
        if f.get("legacy"):
            out.append(f"`{f['file']}`: LEGACY asset -- {f['legacy']} Do not feed it to the modern pipeline.")
        for c in f.get("columns", []):
            if c.get("exceeds_float64_exact_int"):
                big_int_cols.append(f"{f['file']}.{c['column']}")
            if c["n_values"] and c["pct_missing"] >= 50.0:
                out.append(f"`{f['file']}`.{c['column']}: {c['pct_missing']:.1f}% missing in the profiled rows.")
        if f.get("n_duplicate_id_rows"):
            out.append(f"`{f['file']}`: {f['n_duplicate_id_rows']:,} repeated root_id rows -- "
                       "this file is not one-row-per-neuron; aggregate before joining.")
        if f.get("n_self_loops"):
            out.append(f"`{f['file']}`: {f['n_self_loops']:,} self-connections (pre == post).")
    conn = by_asset.get("connections_filtered")
    if conn and conn.get("n_duplicate_pair_rows"):
        out.append(f"`{conn['file']}`: {conn['n_duplicate_pair_rows']:,} repeated (pre,post) rows -- the table is "
                   "one row per (pre, post, neuropil); sum syn_count per pair before building a graph.")
    if conn and conn.get("row_count_is_exact"):
        n = conn["n_rows_including_header"]
        d = n - REFERENCE["connections_rows_filtered"]
        if abs(d) > 0:
            out.append(f"`{conn['file']}`: {n:,} rows vs documented {REFERENCE['connections_rows_filtered']:,} "
                       f"({d:+,}) -- the download may be a different snapshot than v783. Verify before citing counts.")
    if big_int_cols:
        out.append("Neuron ids exceed 2^53 and therefore lose precision if any reader converts them to "
                   "float64 (the pandas default when a column has a missing value). Read every id column as "
                   "int64 or str. Affected columns: " + ", ".join(f"`{c}`" for c in big_int_cols) + ".")
    if not conn:
        out.append("No filtered connections table found -- no graph can be built from this directory.")
    if not out:
        out.append("None detected in the profiled rows.")
    return out


# --------------------------------------------------------------------------- report
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer() and abs(v) >= 1e12:
            return f"{int(v):,}"
        return f"{v:,.4g}"
    return str(v)


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def write_markdown(rep: dict, out: Path) -> None:
    by_asset = {f.get("asset"): f for f in rep["files"] if f.get("asset")}
    opts = rep["options"]
    L: list[str] = [
        "# FAFB v783 local data -- real-schema inspection report",
        "",
        f"- scanned directory: `{rep['scanned_dir']}`",
        f"- generated: {rep['generated_utc']} | python {rep['python']} | platform {rep['platform']}",
        f"- options: sample_rows={opts['sample_rows']} count_rows={opts['count_rows']} "
        f"deep={opts['deep']} include_large={opts['include_large']}",
        f"- files examined: {rep['n_files']}",
        "",
        "Every number below was measured from the files in that directory by "
        "`scripts/inspect_fafb.py`. Where a fact cannot be established from the bytes, the report "
        f"says `{UNKNOWN}` instead of guessing.",
        "",
        "## 1. Dataset inventory",
        "",
    ]
    rows = []
    ignored = 0
    for f in rep["files"]:
        if f.get("asset") is None and not f.get("header"):
            ignored += 1
            continue
        rows.append([f"`{f['file']}`", f.get("asset") or "unrecognised", f.get("role") or "-",
                     "LEGACY" if f.get("legacy") else ("skipped (large)" if f.get("container") == "skipped" else "use")])
    L += _table(["file", "asset key", "role", "status"], rows)
    if ignored:
        L += ["", f"{ignored} further files in this directory are not tables (installers, images, archives, "
              "source files). They were inventoried by name and size only -- never opened. If this is your "
              "Downloads folder rather than a dedicated FAFB folder, that is expected."]

    L += ["", "## 2. File sizes", ""]
    relevant = [f for f in rep["files"] if f.get("asset") or f.get("header")]
    total = sum(f.get("size_bytes", 0) for f in relevant)
    L += _table(["file", "bytes", "MiB"],
                [[f"`{f['file']}`", f"{f.get('size_bytes', 0):,}", f"{f.get('size_bytes', 0) / 1048576:.1f}"]
                 for f in relevant])
    L += ["", f"Total of the tables above: **{total:,} bytes ({total / 1073741824:.2f} GiB)**."]

    L += ["", "## 3. File formats", ""]
    L += _table(["file", "container", "delimiter", "members"],
                [[f"`{f['file']}`", f.get("container", "?"),
                  repr(f.get("delimiter")) if f.get("delimiter") else "-",
                  _fmt(f.get("n_members"))] for f in relevant])

    L += ["", "## 4. Row counts", ""]
    rows = []
    skipped_non_tables = 0
    for f in rep["files"]:
        if f.get("container") in ("zip", "skipped"):
            continue
        if not f.get("columns"):
            # never parsed (installer, image, archive): it has no rows to count, and
            # listing 200 of them as "UNKNOWN row count" buries the six files that matter
            skipped_non_tables += 1
            continue
        exact = f.get("row_count_is_exact")
        n = f.get("n_rows_including_header")
        rows.append([f"`{f['file']}`",
                     _fmt(n) if n is not None else f"{UNKNOWN} (re-run with --count-rows)",
                     "exact" if exact else f"profiled first {f.get('n_rows_profiled', 0):,} rows",
                     _fmt(f.get("n_rows_profiled"))])
    L += _table(["file", "data rows (header excluded)", "basis", "rows profiled"], rows)
    if skipped_non_tables:
        L += ["", f"{skipped_non_tables} non-table files are omitted from this table: they were never "
                  "opened, so they have no row count to report."]
    L += ["", "Documented reference counts for comparison (FlyWire/Codex portal):", "",
          "```", json.dumps(rep["reference_counts"], indent=2, sort_keys=True), "```"]

    L += ["", "## 5. Column names and inferred types", ""]
    for f in rep["files"]:
        if not f.get("columns"):
            continue
        L += [f"### `{f['file']}`", ""]
        rows = []
        for c in f["columns"]:
            dv = c.get("distinct_values")
            vals = ", ".join(f"`{x}`" for x in dv[:12]) if dv else ", ".join(f"`{x}`" for x in c["examples"][:3])
            rows.append([f"`{c['column']}`", c["inferred_type"], _fmt(c["n_distinct"]),
                         f"{c['pct_missing']:.2f}%", _fmt(c["min"]), _fmt(c["max"]), vals or "-"])
        L += _table(["column", "type", "distinct", "missing", "min", "max", "values / examples"], rows)
        L += [""]

    L += ["## 6. Example rows", ""]
    for f in rep["files"]:
        if not f.get("example_rows"):
            continue
        L += [f"`{f['file']}`", "", "```json", json.dumps(f["example_rows"][:2], indent=2), "```", ""]

    L += ["## 7. Unique-ID statistics", ""]
    rows = []
    for f in rep["files"]:
        if f.get("n_unique_ids") is not None:
            rows.append([f"`{f['file']}`", f.get("id_column", "-"), _fmt(f["n_unique_ids"]),
                         _fmt(f.get("n_duplicate_id_rows")),
                         "capped" if f.get("ids_truncated_at") else "complete"])
        elif f.get("n_unique_nodes") is not None:
            rows.append([f"`{f['file']}`", "pre/post", _fmt(f["n_unique_nodes"]), "-", "complete"])
    L += _table(["file", "id column", "unique ids", "duplicate id rows", "coverage"], rows) if rows else [UNKNOWN]
    idcols = [(f["file"], c) for f in rep["files"] for c in f.get("columns", [])
              if c.get("int_digit_lengths") and max(c["int_digit_lengths"]) >= 16]
    if idcols:
        L += ["", "Neuron-ID format observed (digit-length histogram per column):", ""]
        L += _table(["file", "column", "digit lengths", "max value", "> 2^53?"],
                    [[f"`{fn}`", f"`{c['column']}`", str(c["int_digit_lengths"]), _fmt(c["max"]),
                      "YES -- int64/str only" if c["exceeds_float64_exact_int"] else "no"] for fn, c in idcols])

    L += ["", "## 8. Missing-value statistics", ""]
    rows = [[f"`{f['file']}`", f"`{c['column']}`", _fmt(c["n_missing"]), f"{c['pct_missing']:.2f}%"]
            for f in rep["files"] for c in f.get("columns", []) if c["n_missing"]]
    L += _table(["file", "column", "missing (profiled rows)", "%"], rows) if rows else \
        ["No empty cells in the profiled rows."]

    L += ["", "## 9. Cross-file ID compatibility", ""]
    L += [f"Reference id set: **{rep['cross_file_reference']}**.", ""]
    if rep["cross_file_ids"]:
        L += _table(["asset", "ids seen", "shared with reference", "% of asset in reference", "% of reference covered"],
                    [[r["asset"], _fmt(r["n_ids"]), _fmt(r["n_shared_with_reference"]),
                      f"{r['pct_of_this_asset_in_reference']:.2f}%", f"{r['pct_of_reference_covered']:.2f}%"]
                     for r in rep["cross_file_ids"]])
        L += ["", "An asset whose ids are ~100% contained in the reference shares the FlyWire root-id namespace. "
              "Anything well below that is either a different namespace (e.g. supervoxel ids) or a different "
              "snapshot, and must not be joined without investigation."]
    else:
        L += [UNKNOWN]

    conn = by_asset.get("connections_filtered") or by_asset.get("connections_unfiltered")
    L += ["", "## 10. Connection-table structure", ""]
    if not conn:
        L += ["**No connections file found.** The loader expects `connections_princeton.csv.gz` "
              "(portal label: *Connections (Filtered)*). If your download names it differently, add the name to "
              "`ASSET_BY_FILENAME` here and to `fruitfly/dataset/assets.py`."]
    else:
        L += [f"- file: `{conn['file']}` ({conn.get('size_bytes', 0):,} bytes)",
              f"- columns: {', '.join('`' + c + '`' for c in conn['header'])}",
              f"- rows: {_fmt(conn.get('n_rows_including_header')) if conn.get('row_count_is_exact') else UNKNOWN}",
              f"- unique presynaptic ids: {_fmt(conn.get('n_unique_pre'))}",
              f"- unique postsynaptic ids: {_fmt(conn.get('n_unique_post'))}",
              f"- unique nodes (union): {_fmt(conn.get('n_unique_nodes'))}",
              f"- unique (pre, post) pairs: {_fmt(conn.get('n_unique_pairs'))}",
              f"- repeated (pre, post) rows: {_fmt(conn.get('n_duplicate_pair_rows'))} "
              "(expected: the table is one row per pre/post/neuropil)",
              f"- self-connections: {_fmt(conn.get('n_self_loops'))}",
              f"- distinct neuropils seen: {_fmt(conn.get('n_neuropils_seen'))} "
              f"{conn.get('neuropils_sample') or ''}",
              f"- summed syn_count over profiled rows: {_fmt(conn.get('sum_syn_count'))}",
              f"- directed? {conn.get('directed', UNKNOWN)}",
              "",
              "Weight semantics: `syn_count` is a synapse count per (pre, post, neuropil) row -- an integer "
              "count, not a normalised strength. Any scalar 'weight' used by the simulator is a modelling "
              "choice made downstream and is documented in DATA.md."]

    syn = by_asset.get("synapse_table")
    L += ["", "## 11. Synapse-table structure", ""]
    if not syn:
        L += ["Not present in this directory."]
    elif syn.get("container") == "skipped":
        L += [f"Present as `{syn['file']}` ({syn['size_bytes']:,} bytes) but not profiled. "
              "Re-run with `--include-large` to read its header. It is not needed for the first experiments: "
              "the connection table already aggregates synapse counts per pair."]
    else:
        L += [f"- columns: {', '.join('`' + c + '`' for c in syn['header'])}",
              f"- rows: {_fmt(syn.get('n_rows_including_header')) if syn.get('row_count_is_exact') else UNKNOWN}",
              "- join safety: the id columns must be compared against the connection table's ids before any "
              "join; see section 9. Coordinates in this table are nanometres while skeletons are microns -- "
              "never mix them without converting."]

    L += ["", "## 12. Cell-type / classification structure", ""]
    for key in ("classification", "cell_types", "names_groups", "cell_stats"):
        f = by_asset.get(key)
        if not f:
            L += [f"- `{key}`: not present."]
            continue
        cats = [c for c in f.get("columns", []) if c.get("distinct_values")]
        L += [f"- `{f['file']}` ({key}): columns {', '.join(f['header'])}"]
        for c in cats:
            L += [f"    - `{c['column']}`: {_fmt(c['n_distinct'])} distinct, e.g. " +
                  ", ".join(f"`{v}`" for v in (c["distinct_values"] or [])[:8])]
        for c in f.get("columns", []):
            if not c.get("distinct_values"):
                L += [f"    - `{c['column']}`: {_fmt(c['n_distinct'])} distinct values "
                      f"(examples: {', '.join('`' + e + '`' for e in c['examples'][:3])})"]

    L += ["", "## 13. Visual-neuron annotation structure", ""]
    for key in ("visual_types", "visual_columns"):
        f = by_asset.get(key)
        if not f:
            L += [f"- `{key}`: not present."]
            continue
        L += [f"- `{f['file']}` ({key}): columns {', '.join(f['header'])}; "
              f"rows profiled {f.get('n_rows_profiled', 0):,}; unique ids {_fmt(f.get('n_unique_ids'))}"]

    L += ["", "## 14. Neurotransmitter structure", ""]
    nt = by_asset.get("nt_predictions")
    if not nt:
        L += ["- `neurons.csv.gz` not present -- neurotransmitter predictions unavailable."]
    else:
        L += [f"- `{nt['file']}`: columns {', '.join(nt['header'])}"]
        for c in nt.get("columns", []):
            if _norm(c["column"]) == "nt_type":
                L += [f"- `nt_type`: {_fmt(c['n_distinct'])} distinct "
                      f"({', '.join('`' + v + '`' for v in (c['distinct_values'] or [])[:12])}); "
                      f"missing in {c['pct_missing']:.2f}% of profiled rows"]
        L += ["- The per-transmitter `*_avg` columns are classifier scores in [0, 1], not measurements. "
              "Treating a predicted transmitter as ground truth is a modelling assumption, not a fact of the data."]
    conn_nt = [c for c in (conn or {}).get("columns", []) if _norm(c["column"]) == "nt_type"] if conn else []
    if conn_nt:
        c = conn_nt[0]
        L += [f"- the connection table also carries `nt_type` ({_fmt(c['n_distinct'])} distinct, "
              f"{c['pct_missing']:.2f}% missing) -- per-edge, derived from the presynaptic neuron."]

    L += ["", "## 15. Potential schema hazards", ""]
    L += [f"{i + 1}. {h}" for i, h in enumerate(rep["hazards"])]

    L += ["", "## 16. Recommended canonical internal schema", "",
          "Derived from the columns actually present above. Everything not listed stays in the raw files "
          "and is loaded lazily.", "",
          "```text",
          "Neuron",
          "    id            : int64   <- root_id (18-digit FlyWire root id; never float)",
          "    nt_type       : str|''  <- neurons.csv.gz nt_type (prediction)",
          "    super_class   : str|''  <- classification.csv.gz super_class",
          "    cell_class    : str|''  <- classification.csv.gz class",
          "    side          : str|''  <- classification.csv.gz side",
          "    primary_type  : str|''  <- consolidated_cell_types.csv.gz primary_type",
          "    visual_type   : str|''  <- visual_neuron_types.csv.gz type",
          "    metadata      : dict    <- anything else, lazily attached",
          "",
          "Connection (one row per pre/post/neuropil in the raw file)",
          "    source_id     : int64   <- pre_root_id",
          "    target_id     : int64   <- post_root_id",
          "    syn_count     : int32   <- syn_count (raw synapse count)",
          "    neuropil      : str     <- neuropil",
          "    nt_type       : str|''  <- nt_type",
          "",
          "Edge (after aggregation, what the simulator consumes)",
          "    source_id, target_id, weight = sum(syn_count over neuropils), sign = f(nt_type)",
          "",
          "ConnectomeGraph",
          "    root_ids      : int64[n]           dense index <-> root_id map",
          "    adjacency     : scipy CSR [n x n]  directed, weights from syn_count",
          "    annotations   : per-neuron columns, joined on root_id, left-join semantics",
          "```", "",
          "Transformations to document in DATA.md: (a) row aggregation over neuropils; (b) the "
          "syn_count -> weight scaling; (c) the nt_type -> sign mapping, which is a modelling assumption; "
          "(d) any subsetting/sampling, which must be seeded."]

    L += ["", "## 17. Files to use for the first real experiment", ""]
    order = [("connections_filtered", "required -- the graph itself"),
             ("nt_predictions", "required -- neuron list + transmitter prediction"),
             ("classification", "recommended -- lets populations be selected by class/side"),
             ("cell_types", "optional -- named cell types for readable populations"),
             ("visual_types", "optional -- needed only for visual-pathway experiments"),
             ("coordinates", "optional -- only for plots")]
    rows = []
    for key, why in order:
        f = by_asset.get(key)
        rows.append([key, f"`{f['file']}`" if f else "MISSING",
                     f"{f.get('size_bytes', 0) / 1048576:.1f} MiB" if f else "-", why])
    L += _table(["asset", "file", "size", "role in experiment 1"], rows)
    L += ["", "Smallest scientifically valid subset: the filtered connection table plus the neuron table. "
          "Everything else is annotation that improves population selection but is not needed to build a graph."]

    L += ["", "## 18. Files to keep optional / lazy-loaded", ""]
    rows = []
    for f in rep["files"]:
        # ONLY recognised dataset assets. A 1.5 GB installer in the same folder is not
        # a FAFB file to "lazy-load"; saying so would be advice about the wrong thing.
        if not f.get("asset"):
            continue
        if f.get("asset") in LARGE_ASSETS or f.get("size_bytes", 0) > 200 * 1048576 or f.get("legacy"):
            rows.append([f"`{f['file']}`", f"{f.get('size_bytes', 0) / 1048576:.1f} MiB",
                         f.get("legacy") or "size -- stream or skip; never load whole into RAM"])
    if rows:
        L += _table(["file", "size", "reason"], rows)
    else:
        L += ["No FAFB asset in this directory is large enough to require lazy loading. "
              "The two that would be -- the synapse table (~2.7 GB) and the skeletons (~13 GB) -- "
              "are not present here."]

    L += ["", "---", "",
          "A connectome is a wiring map. Everything this project builds on top of it -- neuron dynamics, "
          "plasticity, reward -- is an artificial computational model, not a biological claim.", "",
          "Produced by `scripts/inspect_fafb.py`. Commit this file so the loader is validated against real "
          "bytes rather than documentation.", ""]
    out.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- profile
def write_profile(rep: dict, out: Path) -> None:
    """Emit a YAML profile the loader prefers over its built-in guesses."""
    prof = {"generated_utc": rep["generated_utc"], "scanned_dir": rep["scanned_dir"], "assets": {}}
    for f in rep["files"]:
        key = f.get("asset")
        if not key:
            continue
        header = list(f.get("header", []))
        # columns maps the names the loader already knows to the names your file
        # actually uses, so a variant spelling (or a prefix-encoded id column) can be
        # bridged without editing Python. A documented name with no match is left out
        # -- hand-add  `pre: <your column>`  if the file uses a name nobody has seen.
        cols: dict[str, str] = {}
        for want in EXPECTED_COLUMNS.get(key, []):
            real = _match_column(header, want)
            if real is not None:
                cols[want] = real
        prof["assets"][key] = {
            "filename": f["file"],
            "path": f["path"],
            "header": header,
            "columns": cols,
            "n_rows": f.get("n_rows_including_header"),
        }
    try:
        import yaml

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# Generated by scripts/inspect_fafb.py -- describes the REAL files on this machine.\n"
            + yaml.safe_dump(prof, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prof, indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)[:50]
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="", help="FAFB data directory (default: $FAFB_DATA_PATH, then ~/Downloads)")
    ap.add_argument("--out", default="fafb_schema_report", help="report path prefix")
    ap.add_argument("--sample-rows", type=int, default=250_000, help="rows per file to profile (default 250000)")
    ap.add_argument("--count-rows", action="store_true", help="stream every row for exact row counts")
    ap.add_argument("--deep", action="store_true", help="full profiling + duplicate/reciprocity analysis")
    ap.add_argument("--include-large", action="store_true", help="also probe skeleton zip / synapse table")
    ap.add_argument("--write-profile", action="store_true", help="also write config/fafb_profile.yaml")
    args = ap.parse_args(argv)

    root = Path(args.dir or os.environ.get("FAFB_DATA_PATH", "") or (Path.home() / "Downloads")).expanduser()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    rep = scan(root, sample_rows=args.sample_rows, count_rows=args.count_rows,
               deep=args.deep, include_large=args.include_large)
    prefix = Path(args.out)
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)
    j = prefix.with_suffix(".json")
    md = prefix.with_suffix(".md")
    j.write_text(json.dumps(_json_safe(rep), indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_markdown(rep, md)
    if args.write_profile:
        write_profile(rep, Path("config/fafb_profile.yaml"))
        print("wrote config/fafb_profile.yaml")

    found = [f for f in rep["files"] if (f.get("asset") or "").startswith("connections")]
    print(f"scanned {root}  ({rep['n_files']} files)")
    if not rep["files"]:
        print("  no files found in that directory (is the path right? is the download still a .zip?)")
    n_ignored = 0
    for f in rep["files"]:
        if f.get("asset") is None and not f.get("header"):
            n_ignored += 1
            continue
        mark = f.get("asset") or ("skipped" if f.get("container") == "skipped" else "unrecognised")
        print(f"  [{mark:>26}] {f['file']:<44} {f.get('size_bytes', 0):>14,} bytes  cols={len(f.get('header', []))}")
        if f.get("columns_missing"):
            print(f"  {'':>26}   !! missing documented columns: {', '.join(f['columns_missing'])}")
        if f.get("error"):
            print(f"  {'':>26}   !! {f['error']}")
    if n_ignored:
        print(f"  ({n_ignored} non-table files inventoried by name/size only -- never opened)")
    print(f"\nwrote {j}\nwrote {md}")
    print(f"hazards flagged: {len(rep['hazards'])}")
    if not found:
        print("\nNo connections table found -> the simulator cannot build a graph from this directory yet.")
        return 2
    print('Connections table present -> try: python -m fruitfly stats --data-dir "%s"' % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
