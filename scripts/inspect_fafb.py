#!/usr/bin/env python3
"""Inspect your local FAFB v783 download and report its REAL schema.

Why this script exists: the project must not assume the shape of your files. This
is a standalone diagnostic (Python 3.10+, **stdlib only** -- no numpy/pandas
needed) designed to be run on your machine against the folder you downloaded,
e.g. your Downloads directory. It writes a machine-readable JSON report and a
human-readable markdown report that you can commit to this repository so the
loader can be validated against the actual bytes rather than against documentation.

Usage (PowerShell):

    cd <path\to>\FruitFlyLuau
    python scripts/inspect_fafb.py --dir "$env:USERPROFILE\Downloads"
    python scripts\inspect_fafb.py --dir D:\data\FAFB_v783 --count-rows

Options
-------
--dir PATH        folder to scan (also searches one level of subfolders)
--out PREFIX      write <PREFIX>.json and <PREFIX>.md (default: fafb_schema_report)
--count-rows      fully decompress each CSV to count rows (slower; the 2.7 GB
                  synapse table is skipped unless --include-large is given)
--include-large   also probe the 13 GB skeleton zip and 2.7 GB synapse table
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
from datetime import datetime, timezone
from pathlib import Path

ASSET_BY_FILENAME = {
    "connections_princeton.csv.gz": "connections_filtered",
    "connections_princeton.csv": "connections_filtered",
    "connections_princeton_no_threshold.csv.gz": "connections_unfiltered",
    "connections_princeton_no_threshold.csv": "connections_unfiltered",
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
    "fafb_v783_princeton_synapse_table.csv.gz": "synapse_table",
    "synapse_table.csv.gz": "synapse_table",
    "sk_lod1_783_healed.zip": "skeletons",
    "synapse_coordinates.csv.gz": "synapse_coordinates_legacy",
    "synapse_attachment_rates.csv.gz": "attachment_rates_legacy",
}

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

REFERENCE = {
    "cells": 139255,
    "connections_rows_filtered": 5342446,
    "unique_pairs": 3732460,
    "synapses": 50666648,
    "neuropils": 79,
}


def sniff(path: Path, *, count_rows: bool = False) -> dict:
    """Read a file's real header (+ optional row count and dtypes)."""
    info: dict = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "container": "csv.gz" if path.suffix.lower() == ".gz" else path.suffix.lstrip(".").lower() or "unknown",
        "header": [],
        "sample_row": {},
        "numeric_like": {},
        "id_digits": {},
        "error": "",
    }
    try:
        if path.suffix.lower() == ".zip":
            import zipfile

            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            info["container"] = "zip"
            info["n_members"] = len(names)
            info["first_members"] = names[:5]
            return info
        opener = gzip.open if info["container"] == "csv.gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
            head = fh.read(1 << 16)
        lines = [ln for ln in head.splitlines() if ln.strip()][:6]
        if not lines:
            info["error"] = "file appears empty"
            return info
        delim = max([",", "\t", ";"], key=lambda d: lines[0].count(d))
        rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
        info["delimiter"] = delim
        info["header"] = [h.strip() for h in rows[0]]
        if len(rows) > 1:
            info["sample_row"] = dict(zip(info["header"], rows[1]))
            for col, val in info["sample_row"].items():
                v = (val or "").strip()
                if v and v.lstrip("-").isdigit():
                    info["numeric_like"][col] = True
                    if len(v) >= 12:
                        info["id_digits"][col] = {"digits": len(v), "prefix9": v[:9], "value": int(v)}
                elif v:
                    try:
                        float(v)
                        info["numeric_like"][col] = True
                    except ValueError:
                        pass
        if count_rows:
            n = 0
            with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
                for _ in fh:
                    n += 1
            info["n_rows_including_header"] = n - 1
    except Exception as exc:  # never die on one odd file
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def scan(root: Path, *, count_rows: bool, include_large: bool) -> dict:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".swc":
            continue
        if len(p.relative_to(root).parts) > 2:
            continue
        if not include_large and p.name in {"sk_lod1_783_healed.zip", "fafb_v783_princeton_synapse_table.csv.gz",
                                           "synapse_table.csv.gz"}:
            files.append({"file": p.name, "size_bytes": p.stat().st_size, "container": "skipped",
                          "note": "large asset; re-run with --include-large to probe it"})
            continue
        info = sniff(p, count_rows=count_rows and p.stat().st_size < 500_000_000)
        info["path"] = str(p)
        key = ASSET_BY_FILENAME.get(p.name.lower())
        if key:
            info["asset"] = key
            exp = EXPECTED_COLUMNS.get(key, [])
            got = {c.lower() for c in info["header"]}
            info["expected_columns"] = exp
            info["columns_missing"] = [c for c in exp if c.lower() not in got]
            info["columns_extra"] = [c for c in info["header"] if c.lower() not in {e.lower() for e in exp}]
            info["matches_documented_schema"] = not info["columns_missing"]
        files.append(info)
    return {"scanned_dir": str(root.resolve()), "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": sys.platform, "reference_counts": REFERENCE,
            "files": files, "n_files": len(files)}


def write_markdown(rep: dict, out: Path) -> None:
    lines = [
        "# FAFB v783 local data -- schema inspection report",
        "",
        f"- scanned directory: `{rep['scanned_dir']}`",
        f"- generated: {rep['generated_utc']}  |  python {rep['python']}  |  platform {rep['platform']}",
        f"- files examined: {rep['n_files']}",
        "",
        "Documented reference counts (FlyWire/Codex portal, for completeness checks only):",
        "",
        "```",
        json.dumps(rep["reference_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Files found",
        "",
        "| file | asset | size | columns | missing vs documented | rows | notes |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for f in rep["files"]:
        cols = ", ".join(f.get("header", [])[:8]) + (" ..." if len(f.get("header", [])) > 8 else "")
        missing = ", ".join(f.get("columns_missing", [])) or "-"
        rows = f.get("n_rows_including_header", "")
        rows = f"{rows:,}" if isinstance(rows, int) else "-"
        note = f.get("error") or f.get("note") or ""
        if f.get("id_digits"):
            col = next(iter(f["id_digits"]))
            note += f" | id col `{col}` has {f['id_digits'][col]['digits']} digits, prefix {f['id_digits'][col]['prefix9']}"
        lines.append(
            f"| `{f['file']}` | {f.get('asset', '-')} | {f.get('size_bytes', 0):,} "
            f"| {cols or '-'} | {missing} | {rows} | {note.strip(' |')} |"
        )
    conn = [f for f in rep["files"] if f.get("asset", "").startswith("connections")]
    lines += ["", "## Connections table", ""]
    if not conn:
        lines.append(
            "**No connections file found.** The loader expects "
            "`connections_princeton.csv.gz` (portal label: *Connections (Filtered)*). "
            "If your download uses a different file name, either rename it, or add the name to "
            "`fruitfly/dataset/assets.py` (alternates) and `scripts/inspect_fafb.py` (ASSET_BY_FILENAME)."
        )
    else:
        c = conn[0]
        lines.append(f"- using `{c['file']}` with columns: {', '.join(c['header'])}")
        if c.get("columns_missing"):
            lines.append(f"- **missing documented columns:** {', '.join(c['columns_missing'])} -- tell the loader "
                         "the real names via `fruitfly/dataset/schema.py:CONNECTION_FIELDS` aliases.")
        if c.get("n_rows_including_header"):
            n = c["n_rows_including_header"]
            delta = n - REFERENCE["connections_rows_filtered"]
            lines.append(f"- rows: {n:,} (documented 5,342,446 -> {delta:+,}, {100 * delta / REFERENCE['connections_rows_filtered']:+.2f}%)")
        if c.get("id_digits"):
            lines.append("- neuron id format observed: " + json.dumps(c["id_digits"], indent=2))
    lines += ["", "---", "", "Produced by `scripts/inspect_fafb.py`. Commit this file to the repository "
              "so the loader is validated against real bytes, not documentation.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="", help="FAFB data directory (default: $FAFB_DATA_PATH, then ~/Downloads)")
    ap.add_argument("--out", default="fafb_schema_report", help="report path prefix")
    ap.add_argument("--count-rows", action="store_true", help="decompress files to count rows")
    ap.add_argument("--include-large", action="store_true", help="also probe skeleton zip / synapse table")
    ap.add_argument("--write-profile", action="store_true", help="also write config/fafb_profile.yaml")
    args = ap.parse_args(argv)

    root = Path(args.dir or os.environ.get("FAFB_DATA_PATH", "") or (Path.home() / "Downloads")).expanduser()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    rep = scan(root, count_rows=args.count_rows, include_large=args.include_large)
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    j = prefix.with_suffix(".json")
    md = prefix.with_suffix(".md")
    j.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(rep, md)
    if args.write_profile:
        write_profile(rep, Path("config/fafb_profile.yaml"))
        print("wrote config/fafb_profile.yaml")

    found = [f for f in rep["files"] if f.get("asset", "").startswith("connections")]
    print(f"scanned {root}  ({rep['n_files']} files)")
    for f in rep["files"]:
        mark = f.get("asset") or ("skipped" if f.get("container") == "skipped" else "unrecognised")
        print(f"  [{mark:>26}] {f['file']:<44} {f.get('size_bytes', 0):>14,} bytes  cols={len(f.get('header', []))}")
        if f.get("columns_missing"):
            print(f"  {'':>26}   !! missing documented columns: {', '.join(f['columns_missing'])}")
        if f.get("error"):
            print(f"  {'':>26}   !! {f['error']}")
    print(f"\nwrote {j}\nwrote {md}")
    if not found:
        print("\nNo connections table found -> the simulator cannot build a graph from this directory yet.")
        return 2
    print("Connections table present -> try: python -m fruitfly stats --data-dir \"%s\"" % root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
