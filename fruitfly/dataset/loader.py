"""Streaming loader for FAFB connection tables.

Reading a 68 MB gzip CSV of 5.3M rows on a laptop is the whole job here, so the
loader is deliberately boring and vectorised:

* columns are selected by *name*, resolved from the file's real header;
* the file is read in chunks (``pandas`` with explicit dtypes, or pyarrow for
  feather/parquet) so peak memory stays bounded;
* per-neuropil rows for the same (pre, post) pair are summed by a sparse
  COO->CSR conversion, never by Python-level dict of edge objects;
* the result is cached as a ``.npz`` so a re-run pays the 68 MB parse once.

Every number the loader reports comes from what it actually read.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, human_bytes
from .assets import NT_CONVENTION
from .discover import DatasetInventory, discover
from .schema import CONNECTION_FIELDS, NEURON_ID_FIELDS, load_profile, match_schema, specs_for

log = get_logger(__name__)


@dataclass
class ConnectionTable:
    """Raw (pre, post, weight) edge lists + per-neuron metadata, before subsetting."""

    pre: np.ndarray  # int64 FlyWire root ids
    post: np.ndarray
    weight: np.ndarray  # float64, summed syn_count
    neuropil: np.ndarray | None  # dominant neuropil string per aggregated edge
    nt_type: np.ndarray | None  # dominant predicted transmitter per aggregated edge
    n_rows: int
    n_unique_ids: int
    source_file: str
    container: str
    columns_used: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def neuron_ids(self) -> np.ndarray:
        """All neuron ids appearing as either side of an edge, sorted."""
        return _all_ids(self.pre, self.post)

    def population_size(self) -> int:
        return int(self.neuron_ids.size)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_edges_after_pair_merge": int(self.pre.size),
            "n_unique_ids": self.n_unique_ids,
            "source_file": self.source_file,
            "container": self.container,
            "columns_used": self.columns_used,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "notes": self.notes,
        }


def _all_ids(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    return np.union1d(pre, post)


# --------------------------------------------------------------------- reading
def _resolve_path(inventory: DatasetInventory, source: str) -> tuple[Path, str]:
    if source in {"connections_filtered", "connections_unfiltered", "connections_legacy"}:
        m = inventory.matches.get(source)
        if m is None or m.path is None:
            m = inventory.connections
            if m is None:
                raise FileNotFoundError(
                    f"no connection table found in {inventory.data_dir}. Expected\n"
                    f"  {json.dumps([a for a in ('connections_princeton.csv.gz',)], indent=0)}\n"
                    "Run `python scripts/inspect_fafb.py --dir <path>` for a full report."
                )
            if source != m.key:
                m.spec.__class__  # keep type checkers quiet
                log.warning("requested %s but only %s is present -> using it", source, m.key)
            return m.path, m.key
        return m.path, source
    raise ValueError(f"unknown connection source {source!r}")


def _read_chunks(path: Path, columns: dict[str, str], chunksize: int) -> tuple[pd.DataFrame, float]:
    """Read only the needed columns, in chunks, concatenating once at the end."""
    t0 = time.perf_counter()
    suffix = path.suffix.lower()
    name_map = {v: k for k, v in columns.items()}  # real column -> logical
    parts: list[pd.DataFrame] = []
    if suffix == ".feather" or (suffix == ".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore

            tbl = pq.read_table(path, columns=list(name_map)) if suffix == ".parquet" else None
            if tbl is None:
                import pyarrow.feather as feather  # type: ignore

                tbl = feather.read_table(path, columns=list(name_map))
            df = tbl.to_pandas()
            df.columns = [name_map[c] for c in df.columns]
            parts = [df]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"{path.name} needs pyarrow: pip install 'fruitflyluau[feather]'"
            ) from exc
    else:
        header = pd.read_csv(path, nrows=0, compression="infer")
        sep = ","
        if len(header.columns) == 1 and "\t" in header.columns[0]:
            sep, header = "\t", pd.read_csv(path, nrows=0, sep="\t", compression="infer")
        usecols = [c for c in header.columns if c in name_map]
        # Only force dtypes on the numeric logical fields; label columns (neuropil,
        # nt_type) must stay strings, otherwise pandas tries to parse 'PB' as a float.
        dtype: dict[str, type] = {}
        for c in usecols:
            if name_map[c] in {"pre", "post"}:
                dtype[c] = np.int64
            elif name_map[c] == "weight":
                dtype[c] = np.float64
        reader = pd.read_csv(
            path, usecols=usecols, dtype=dtype, chunksize=chunksize, sep=sep,
            compression="infer", low_memory=False, na_filter=False,
        )
        for chunk in reader:
            # real file column -> logical name, so the rest of the loader never
            # needs to know which alias this particular export used.
            chunk = chunk.rename(columns={c: name_map[c] for c in chunk.columns if c in name_map})
            parts.append(chunk)
    if not parts:
        raise ValueError(f"no rows read from {path}")
    df = pd.concat(parts, ignore_index=True)
    return df, time.perf_counter() - t0


def load_connections(
    data_dir: str | Path,
    *,
    source: str = "connections_filtered",
    chunksize: int = 1_000_000,
    max_rows: int = 0,
    min_synapses_per_pair: int = 1,
    cache_path: str | Path | None = None,
    inventory: DatasetInventory | None = None,
) -> ConnectionTable:
    """Load a FAFB connection table and merge per-neuropil rows into pair weights.

    Parameters
    ----------
    data_dir:
        Directory holding the downloaded FAFB assets.
    source:
        ``connections_filtered`` (default), ``connections_unfiltered`` or
        ``connections_legacy``.
    max_rows:
        Cap rows read (0 = all). Handy for a quick laptop smoke test.
    min_synapses_per_pair:
        Drop pairs whose *total* synapse count is below this. The filtered asset
        needs 1 (already 5+ filtered upstream); unfiltered wants 5.
    cache_path:
        Where to store/read a ``.npz`` derived cache of this table.
    """
    inv = inventory or discover(data_dir)
    if inv.connections is None and source not in inv.matches:
        raise FileNotFoundError(
            f"no connection table in {inv.data_dir}.\n\n{inv.report_text()}"
        )
    path, used_key = _resolve_path(inv, source)
    match = inv.matches.get(used_key)

    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.is_file() and cache.stat().st_mtime > path.stat().st_mtime:
        t0 = time.perf_counter()
        with np.load(cache, allow_pickle=False) as z:
            table = ConnectionTable(
                pre=z["pre"],
                post=z["post"],
                weight=z["weight"],
                neuropil=z["neuropil"].astype(str) if "neuropil" in z.files else None,
                nt_type=z["nt_type"].astype(str) if "nt_type" in z.files else None,
                n_rows=int(z["n_rows"]) if "n_rows" in z.files else int(z["pre"].size),
                n_unique_ids=int(z["n_unique_ids"]) if "n_unique_ids" in z.files else 0,
                source_file=str(z["source_file"][0]),
                container=str(z["container"][0]),
                timings={"npz_cache": time.perf_counter() - t0},
                notes=[f"loaded from derived cache {cache.name}"],
            )
        log.info(
            "connection table from cache: %s edges (%.1f s)",
            f"{table.pre.size:,}",
            table.timings["npz_cache"],
        )
        return table

    if match is not None and match.probe is not None and match.probe.error:
        log.warning("probe of %s reported: %s", path.name, match.probe.error)
    header = match.probe.header if (match and match.probe) else tuple(pd.read_csv(path, nrows=0, compression="infer").columns)
    profile = load_profile()
    specs = specs_for(profile, source, CONNECTION_FIELDS)
    if specs is not CONNECTION_FIELDS:
        log.info("using machine data profile column names for %s", source)
    schema = match_schema(header, specs)
    if not schema.ok:
        raise ValueError(
            f"cannot find required columns in {path.name}.\n{schema.describe()}\n"
            "Tell the loader the real names by adding them to "
            "fruitfly/dataset/schema.py:CONNECTION_FIELDS aliases, or run "
            "`python scripts/inspect_fafb.py --dir <dir> --write-profile` to record the "
            "spellings your files actually use."
        )
    log.info("reading %s (%s)", path.name, human_bytes(path.stat().st_size))
    log.debug("column mapping: %s", schema.describe())

    needed = {"pre": schema.column("pre"), "post": schema.column("post")}
    if schema.has("weight"):
        needed["weight"] = schema.column("weight")
    if schema.has("neuropil"):
        needed["neuropil"] = schema.column("neuropil")
    if schema.has("nt_type"):
        needed["nt_type"] = schema.column("nt_type")

    df, t_read = _read_chunks(path, needed, chunksize)
    timings = {"read_parse": t_read}

    if "weight" not in df.columns:
        # Some exports carry no count column: each row is then one unit of weight.
        df["weight"] = np.int64(1)
        schema_missing_weight = True
    else:
        schema_missing_weight = False

    pre = np.ascontiguousarray(df["pre"].to_numpy(np.int64))
    post = np.ascontiguousarray(df["post"].to_numpy(np.int64))
    n_rows_raw = int(pre.size)

    # Drop malformed ids and autapses (autapses are already absent in the
    # filtered asset; keeping the check makes the loader safe for other sources).
    t0 = time.perf_counter()
    ok = (pre > 0) & (post > 0) & (pre != post)
    if max_rows and ok.sum() > max_rows:
        keep = np.flatnonzero(ok)[:max_rows]
        mask = np.zeros_like(ok)
        mask[keep] = True
        ok = mask
    pre, post = pre[ok], post[ok]
    aux = {}
    for col in ("neuropil", "nt_type"):
        if col in df.columns:
            aux[col] = df[col].to_numpy()[ok]
    weight = df["weight"].to_numpy(np.float64)[ok] if "weight" in df.columns else np.ones(pre.size, np.float64)
    n_valid = int(pre.size)
    timings["filter"] = time.perf_counter() - t0

    # ---- merge (pre, post) rows that repeat across neuropils.
    # Labels (neuropil, predicted transmitter) are taken from the largest-weight
    # row of each merged group; ties keep file order. Vectorised: no per-edge
    # Python objects.
    t0 = time.perf_counter()
    ids = _all_ids(pre, post)
    pre_code = np.searchsorted(ids, pre).astype(np.int64)
    post_code = np.searchsorted(ids, post).astype(np.int64)
    n = ids.size
    pair = pre_code * n + post_code  # unique int64 key per ordered pair
    order = np.argsort(pair, kind="stable")
    pair_sorted = pair[order]
    uniq, starts = np.unique(pair_sorted, return_index=True)
    counts = np.diff(np.append(starts, pair_sorted.size))
    w_sorted = weight[order]
    sums = np.add.reduceat(w_sorted, starts)
    pre_out = pre_code[order][starts]
    post_out = post_code[order][starts]
    aux_out: dict[str, np.ndarray] = {}
    for col, arr in aux.items():
        arr_s = arr[order]
        group_max = np.maximum.reduceat(w_sorted, starts)
        is_max = w_sorted == np.repeat(group_max, counts)
        cand = np.flatnonzero(is_max)
        pick = np.searchsorted(cand, starts)
        pick = np.clip(pick, 0, cand.size - 1)
        aux_out[col] = np.asarray(arr_s)[pick].astype(object)
    timings["merge_pairs"] = time.perf_counter() - t0

    if min_synapses_per_pair > 1:
        keep = sums >= min_synapses_per_pair
        sums, pre_out, post_out = sums[keep], pre_out[keep], post_out[keep]
        aux_out = {k: v[keep] for k, v in aux_out.items()}

    pre_ids = ids[pre_out]
    post_ids = ids[post_out]
    notes: list[str] = []
    if schema_missing_weight:
        notes.append("file had no synapse-count column; every row counted as weight 1")
    if n_valid < n_rows_raw:
        notes.append(f"dropped {n_rows_raw - n_valid:,} rows (non-positive id or autapse)")
    if pre_out.size < n_valid:
        notes.append(f"merged {n_valid - pre_out.size:,} duplicate rows into pair-level weights")
    notes.append(f"min_synapses_per_pair={min_synapses_per_pair}")

    table = ConnectionTable(
        pre=pre_ids, post=post_ids, weight=np.asarray(sums, dtype=np.float64),
        neuropil=aux_out.get("neuropil"),
        nt_type=aux_out.get("nt_type"),
        n_rows=n_rows_raw,
        n_unique_ids=int(ids.size),
        source_file=path.name,
        container=path.suffix.lstrip(".") or "csv",
        columns_used={k: v for k, v in needed.items()},
        timings=timings,
        notes=notes,
    )
    log.info(
        "connections: %s rows -> %s merged pair edges over %s neurons (%.1f s)",
        f"{n_rows_raw:,}", f"{table.pre.size:,}", f"{table.n_unique_ids:,}", sum(timings.values()),
    )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pre": table.pre, "post": table.post, "weight": table.weight}
        if table.neuropil is not None:
            payload["neuropil"] = np.asarray(table.neuropil, dtype="U16")
        if table.nt_type is not None:
            payload["nt_type"] = np.asarray(table.nt_type, dtype="U12")
        payload["n_rows"] = np.asarray(table.n_rows)
        payload["n_unique_ids"] = np.asarray(table.n_unique_ids)
        payload["source_file"] = np.asarray([table.source_file])
        payload["container"] = np.asarray([table.container])
        np.savez(cache, **payload)
        log.info("wrote derived cache %s (%s)", cache.name, human_bytes(cache.stat().st_size))
    return table


# ------------------------------------------------------------------ annotation
def load_annotation_table(path: Path, key: str) -> pd.DataFrame:
    """Read one per-neuron annotation asset, resolving its id column by sniffing."""
    df = pd.read_csv(path, compression="infer", low_memory=False)
    if df.columns.size == 1:  # likely TSV
        df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    schema = match_schema(tuple(df.columns), NEURON_ID_FIELDS)
    if not schema.ok:
        log.warning("annotation %s has no recognisable neuron id column: %s", key, list(df.columns)[:8])
        return df
    df = df.rename(columns={schema.column("root_id"): "root_id"})
    df["root_id"] = df["root_id"].astype("int64")
    return df


def annotate_neuron_metadata(
    data_dir: str | Path,
    ids: np.ndarray,
    *,
    inventory: DatasetInventory | None = None,
    keys: tuple[str, ...] = ("nt_predictions", "classification", "cell_types", "names_groups", "connectivity_tags", "visual_types", "coordinates"),
) -> dict[str, np.ndarray]:
    """Attach annotation columns to a population of ``ids`` (aligned, NaN-safe).

    Returns a dict of ``<asset>.<column>`` -> array aligned with ``ids``. Missing
    assets are skipped with a note rather than raising: annotations are optional.
    """
    inv = inventory or discover(data_dir)
    ids = np.asarray(ids, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for key in keys:
        m = inv.matches.get(key)
        if m is None or m.path is None:
            continue
        try:
            df = load_annotation_table(m.path, key)
        except Exception as exc:  # pragma: no cover - defensive against odd files
            log.warning("could not read %s: %s", key, exc)
            continue
        if "root_id" not in df.columns:
            continue
        # multiple rows per neuron (labels, coordinates): keep first per id, but
        # for coordinates we average, since they are spatial anchors.
        if key == "coordinates" and "position" in df.columns:
            pos = df["position"].astype("string").str.extract(r"^([-\d.]+)[,; ]+([-\d.]+)[,; ]+([-\d.]+)").astype(float)
            pos.columns = ["x", "y", "z"]
            joined = df.assign(**pos.to_dict("list"))
            agg = joined.groupby("root_id")[["x", "y", "z"]].mean()
            pos_arr = np.full((ids.size, 3), np.nan, dtype=np.float32)
            idx = np.searchsorted(ids, agg.index.to_numpy())
            good = (idx < ids.size) & (ids[np.clip(idx, 0, ids.size - 1)] == agg.index.to_numpy())
            pos_arr[idx[good]] = agg.to_numpy()[good].astype(np.float32)
            out["coordinates.xyz"] = pos_arr
            continue
        first = df.drop_duplicates("root_id", keep="first").set_index("root_id")
        for col in first.columns:
            if col == "root_id":
                continue
            idx = np.searchsorted(ids, first.index.to_numpy())
            inb = idx < ids.size
            vals = np.empty(ids.size, dtype=object)
            vals[:] = None
            src = first[col].to_numpy(dtype=object)
            vals[idx[inb]] = src[inb]
            out[f"{key}.{col}"] = vals
    return out


def transmitter_labels(nt_column: np.ndarray | None, n: int) -> np.ndarray:
    """Normalise a per-neuron transmitter string column to canonical tokens."""
    out = np.empty(n, dtype=object)
    out[:] = "unknown"
    if nt_column is None:
        return out
    for i, raw in enumerate(nt_column):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        tok = str(raw).strip().lower()
        out[i] = tok if tok in NT_CONVENTION else ("unknown" if tok in {"", "nan", "none"} else tok)
    return out


def ei_signs(labels: np.ndarray) -> np.ndarray:
    """+1 excitatory, -1 inhibitory, +1 mixed, 0 modulatory, +1 unknown (documented choice)."""
    sign = np.ones(labels.size, dtype=np.float32)
    for i, lab in enumerate(labels):
        kind, modulatory = NT_CONVENTION.get(str(lab), ("unknown", False))
        if kind == "inhibitory":
            sign[i] = -1.0
        elif modulatory:
            sign[i] = 0.0
    return sign
