"""FAFB discovery + connection loading against files that use the REAL schema."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from fruitfly.dataset.assets import ASSETS, REFERENCE_COUNTS
from fruitfly.dataset.discover import discover, probe_file
from fruitfly.dataset.loader import load_connections
from fruitfly.dataset.schema import CONNECTION_FIELDS, match_schema
from fruitfly.dataset.stats import table_stats


def test_documented_asset_names_are_registered():
    # Portal label -> file name mapping, from the Codex 'Download Data' page.
    assert ASSETS["connections_filtered"].file == "connections_princeton.csv.gz"
    assert ASSETS["connections_filtered"].display == "Connections (Filtered)"
    assert ASSETS["nt_predictions"].file == "neurons.csv.gz"
    assert ASSETS["nt_predictions"].display == "Neurotransmitter Type Predictions"
    assert ASSETS["connections_filtered"].columns == (
        "pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type")
    assert REFERENCE_COUNTS["cells"] == 139_255
    assert REFERENCE_COUNTS["connections_rows_filtered"] == 5_342_446


def test_schema_matcher_handles_aliasing_and_prefix_encoded_ids():
    real = ("pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", "ach_avg")
    m = match_schema(real, CONNECTION_FIELDS)
    assert m.ok, m.describe()
    assert m.column("pre") == "pre_pt_root_id"
    # unused columns are reported, never silently dropped
    assert "ach_avg" in m.unknown
    # the synapse table encodes the shared root-id prefix in the header name
    syn = ("pre_root_id_720575940", "post_root_id_720575940", "ctr_x")
    m2 = match_schema(syn, CONNECTION_FIELDS)
    assert m2.has("pre") and m2.column("pre").startswith("pre_root_id")


def test_schema_matcher_fails_loudly_on_unknown_shape():
    m = match_schema(("foo", "bar"), CONNECTION_FIELDS)
    assert not m.ok
    assert "pre" in m.missing and "post" in m.missing
    assert "header (2 cols)" in m.describe()


def test_probe_reads_gzip_header_and_sample(tmp_path):
    p = tmp_path / "connections_princeton.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("pre_root_id,post_root_id,neuropil,syn_count,nt_type\n")
        fh.write("720575940111111111,720575940222222222,BU_L,12,glut\n")
    probe = probe_file(p)
    assert probe.container == "csv.gz"
    assert probe.header == ("pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type")
    assert probe.sample_row["nt_type"] == "glut"
    assert probe.size_bytes > 0


def test_discovery_matches_sample_files_and_finds_nothing_else(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    inv = discover(empty)
    assert inv.connections is None
    assert "no files" in " ".join(inv.notes)


def test_discovery_of_sample_fafb_dir(sample_dir):
    inv = discover(sample_dir)
    assert inv.connections is not None
    assert inv.require("connections_filtered").name == "connections_princeton.csv.gz"
    for key in ("nt_predictions", "classification", "cell_types", "coordinates", "connectivity_tags"):
        assert inv.get(key) is not None, f"{key} should be discoverable in the sample dir"
    txt = inv.report_text()
    assert "Connections (Filtered)" in txt


def test_load_connections_merges_per_neuropil_rows(sample_dir):
    table = load_connections(sample_dir, source="connections_filtered")
    assert table.pre.size > 0
    # the sample writer deliberately repeats pairs across neuropils; loading must
    # collapse them to one edge per ordered pair
    pairs = set(zip(table.pre.tolist(), table.post.tolist()))
    assert len(pairs) == table.pre.size, "duplicate (pre,post) pairs survived merging"
    assert (table.weight >= 5).all(), "filtered asset keeps pairs with >=5 total synapses"
    assert table.source_file == "connections_princeton.csv.gz"
    d = table.to_dict()
    assert d["n_rows"] >= table.pre.size
    json.dumps(d)  # must be serialisable for run reports


def test_load_connections_respects_max_rows_and_cache(sample_dir, tmp_path):
    cache = tmp_path / "cache" / "t.npz"
    a = load_connections(sample_dir, max_rows=500, cache_path=cache)
    assert a.pre.size <= 500
    assert cache.is_file()
    b = load_connections(sample_dir, max_rows=500, cache_path=cache)
    assert b.pre.size == a.pre.size
    assert "cache" in " ".join(b.notes).lower()


def test_unfiltered_source_is_a_superset(sample_dir):
    filt = load_connections(sample_dir, source="connections_filtered")
    unf = load_connections(sample_dir, source="connections_unfiltered", min_synapses_per_pair=1)
    assert unf.pre.size >= filt.pre.size


def test_missing_required_columns_raise_with_header(tmp_path):
    p = tmp_path / "connections_princeton.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("a,b,c\n1,2,3\n")
    (tmp_path / "neurons.csv.gz").write_bytes(gzip.compress(b"root_id\n1\n"))
    inv = discover(tmp_path)
    with pytest.raises(ValueError) as exc:
        load_connections(tmp_path, inventory=inv)
    assert "cannot find required columns" in str(exc.value)


def test_table_stats_reports_id_format(sample_dir):
    table = load_connections(sample_dir)
    # ConnectionTable keeps the real FlyWire root ids; compaction to 0..n-1
    # happens later, in build_connectome (see tests/test_graph_connectome.py).
    st = table_stats(table.pre, table.post, table.weight,
                     nt=table.nt_type, neuropil=table.neuropil)
    assert st["id_digits"]  # 18-digit root ids, matching FlyWire's format
    assert st["id_prefix9"] == ["720575940"]
    assert st["autapses"] == 0, "the filtered asset excludes autapses; so should we"
    assert st["weight_min"] >= 5
    assert st["nt_type_n_unique"] >= 1


def test_neuron_index_mapping_is_exact(sample_dir):
    """Integer indexing must round-trip to real FlyWire root ids."""
    from fruitfly.graph.build import build_population
    from fruitfly.config import AppConfig

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.fafb_data_path = str(sample_dir)
    cfg.graph.mode = "tiny"
    cfg.graph.n_neurons = 60
    cfg.graph.selection = "random"
    cfg.data.use_cache = False
    pop = build_population(cfg)
    assert pop.simulated_neurons == 60
    assert pop.connectome.n_neurons == 60
    for idx in (0, 7, pop.connectome.n_neurons - 1):
        rid = int(pop.connectome.root_ids[idx])
        assert pop.connectome.index_of(rid) == idx
    missing = pop.connectome.index_of(-12345)
    assert missing is None


def test_machine_profile_bridges_an_undocumented_column_name(tmp_path, monkeypatch):
    """`inspect_fafb.py --write-profile` can fix a header the aliases do not know.

    Without the profile the loader must fail (the spelling is genuinely unknown);
    with it, the same file loads. This is the escape hatch for real-world export
    variants, and it must stay opt-in: absent a profile, nothing changes.
    """
    import pandas as pd
    import yaml

    from fruitfly.dataset.schema import load_profile, specs_for

    data_dir = tmp_path / "fafb"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "source_neuron": [101, 101, 102],
            "target_neuron": [202, 203, 202],
            "n_syn": [4, 2, 9],
            "brain_region": ["MB", "MB", "PB"],
        }
    )
    conn = data_dir / "connections_princeton.csv.gz"
    df.to_csv(conn, index=False, compression="gzip")

    with pytest.raises(ValueError, match="cannot find required columns"):
        load_connections(data_dir, source="connections_filtered")

    profile = tmp_path / "fafb_profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {"assets": {"connections_filtered": {"filename": conn.name,
                                                   "columns": {"pre": "source_neuron",
                                                               "post": "target_neuron",
                                                               "weight": "n_syn",
                                                               "neuropil": "brain_region"}}}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FAFB_PROFILE", str(profile))
    monkeypatch.chdir(tmp_path)  # a repo-root profile must not leak into this test

    table = load_connections(data_dir, source="connections_filtered")
    assert table.pre.size == 3
    assert table.columns_used["pre"] == "source_neuron"

    # unit-level: the promotion keeps the built-in aliases as fallbacks
    specs = specs_for(load_profile(profile), "connections_filtered", CONNECTION_FIELDS)
    pre = next(sp for sp in specs if sp.name == "pre")
    assert pre.aliases[0] == "source_neuron" and "pre_root_id" in pre.aliases
    assert specs_for({}, "connections_filtered", CONNECTION_FIELDS) is CONNECTION_FIELDS

    # the shape inspect_fafb.py --write-profile emits: documented name -> real name,
    # including prefix-encoded id columns that only match after normalisation.
    alt = tmp_path / "alt.csv.gz"
    pd.DataFrame({"pre_root_id_720575940": [1], "post_root_id_720575940": [2], "syn_count": [7]}).to_csv(
        alt, index=False, compression="gzip"
    )
    prof2 = {"connections_filtered": {"columns": {"pre_root_id": "pre_root_id_720575940"}}}
    specs2 = specs_for(prof2, "connections_filtered", CONNECTION_FIELDS)
    assert next(sp for sp in specs2 if sp.name == "pre").aliases[0] == "pre_root_id_720575940"


def test_specs_for_ignores_an_unrelated_asset_key():
    from fruitfly.dataset.schema import specs_for

    prof = {"neurons": {"columns": {"source_neuron": "source_neuron"}}}
    assert specs_for(prof, "connections_filtered", CONNECTION_FIELDS) is CONNECTION_FIELDS


# --------------------------------------------------------------- measured schema
# The constants below were MEASURED from a real FAFB v783 download on 2026-09-25
# (Windows, 242-file Downloads folder) by scripts/inspect_fafb.py --deep. They are
# not copied from the portal documentation -- they happen to agree with it exactly,
# which is itself the finding. See docs/fafb_schema_report_summary.md.
MEASURED_V783 = {
    "connection_rows": 5_342_446,
    "unique_pairs": 3_732_460,
    "repeated_pair_rows": 1_609_986,
    "summed_syn_count": 50_666_648,
    "neuropils": 79,
    "unique_pre": 137_518,
    "unique_post": 130_183,
    "unique_nodes": 138_584,
    "self_connections": 0,
    "reciprocal_pairs": 620_180,
    "neurons_rows": 139_255,
    "nt_type_missing": 19_658,
    "max_root_id": 720_575_940_661_339_776,
}


def test_reference_counts_match_what_was_measured_on_the_real_download():
    """Documentation and reality agree. If they ever diverge, the measurement wins."""
    from fruitfly.dataset.assets import REFERENCE_COUNTS

    assert REFERENCE_COUNTS["cells"] == MEASURED_V783["neurons_rows"]
    assert REFERENCE_COUNTS["connections_rows_filtered"] == MEASURED_V783["connection_rows"]
    assert REFERENCE_COUNTS["unique_pairs"] == MEASURED_V783["unique_pairs"]
    assert REFERENCE_COUNTS["synapses"] == MEASURED_V783["summed_syn_count"]
    assert REFERENCE_COUNTS["neuropils"] == MEASURED_V783["neuropils"]


def test_the_connection_schema_the_loader_expects_is_the_one_the_real_file_has():
    """Column names and order, verbatim from the real header."""
    from fruitfly.dataset.schema import CONNECTION_FIELDS

    real_header = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
    names = {f.name for f in CONNECTION_FIELDS}
    for col in real_header:
        assert any(col in ({f.name} | set(f.aliases)) for f in CONNECTION_FIELDS), (
            f"the real file has a {col!r} column that no FieldSpec accepts: {sorted(names)}"
        )


def test_root_ids_must_survive_the_real_magnitude():
    """Every observed id is 18 digits and exceeds 2^53: float64 would corrupt them."""
    big = MEASURED_V783["max_root_id"]
    assert big > 2 ** 53
    # at this magnitude float64 has a spacing of 128, so ids are silently snapped to a
    # multiple of 128. The max id happens to be one; its neighbours are not.
    assert int(float(big + 1)) != big + 1, "float64 must be shown to corrupt a real id"
    assert np.int64(big) == big and np.int64(big + 1) == big + 1


def test_duplicate_pair_merging_is_required_by_the_real_file():
    """5,342,446 rows collapse to 3,732,460 pairs -- 30% of rows are extra neuropils."""
    m = MEASURED_V783
    assert m["connection_rows"] - m["unique_pairs"] == m["repeated_pair_rows"]
    fraction = m["repeated_pair_rows"] / m["connection_rows"]
    assert 0.29 < fraction < 0.31, (
        "a loader that skipped merge_duplicates='sum' would silently keep only one "
        f"neuropil's synapses for {fraction:.0%} of rows"
    )


def test_mushroom_body_selector_is_registered_and_documented():
    from fruitfly.graph import list_selectors
    from fruitfly.graph.select import _mushroom_body

    assert "mushroom_body" in list_selectors()
    doc = _mushroom_body.__doc__ or ""
    assert "Kenyon_Cell" in doc
    assert "our inventions" in doc, "the selector must not imply biological equivalence"


def test_mushroom_body_selection_fails_loudly_without_the_classification_file(caplog, tiny_connectome):
    """Without classification.csv.gz there is no class column. Say which file is missing."""
    import logging

    from fruitfly.graph.select import select_population

    with caplog.at_level(logging.WARNING, logger="fruitfly.fruitfly.graph.select"):
        with pytest.raises(ValueError) as exc:
            select_population(
                "mushroom_body",
                ids=tiny_connectome.root_ids,
                adjacency=tiny_connectome.matrix,
                metadata=None,
                rng=np.random.default_rng(0),
            )
    # the exception says the population is empty; the log says WHY
    assert "mushroom_body" in str(exc.value)
    assert any("classification" in r.message for r in caplog.records), (
        "an empty selection must name the file that would have made it non-empty"
    )


def test_cache_from_one_data_dir_is_never_served_for_another(tmp_path, sample_dir):
    """Reproduces a real failure: `stats --data-dir <real FAFB>` returned 529 edges.

    The cache filename was keyed only on loader options, so a table derived from
    data/sample/fafb_v783 was served for a completely different directory that happened
    to contain a file of the same name. The user's 3.7-million-edge download was
    silently replaced by a 529-edge synthetic graph -- and nothing in the output said so
    except the edge count.
    """
    from fruitfly.dataset.loader import load_connections

    # two directories, same filename, different contents
    dir_a = tmp_path / "sample"
    dir_b = tmp_path / "real"
    for d in (dir_a, dir_b):
        d.mkdir()
    name = "connections_princeton.csv.gz"
    (dir_a / name).write_bytes((sample_dir / name).read_bytes())

    rows = ["pre_root_id,post_root_id,neuropil,syn_count,nt_type"]
    for i in range(50):
        rows.append(f"{720575940600000000 + i},{720575940600000100 + i},MB_R,{i + 5},ACH")
    with gzip.open(dir_b / name, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")

    cache = tmp_path / "cache" / "shared.npz"
    a = load_connections(dir_a, cache_path=cache)
    b = load_connections(dir_b, cache_path=cache)

    assert b.pre.size == 50, (
        f"got {b.pre.size} edges for dir_b but it only has 50 rows -- the cache from "
        f"dir_a ({a.pre.size} edges) was served for a different directory"
    )
    assert "720575940600000000" in str(b.pre[0]) or b.pre[0] > 0


def test_a_cache_without_provenance_is_rejected_not_trusted(tmp_path, sample_dir):
    """Caches written before the provenance fields existed must not be believed."""
    import numpy as np

    from fruitfly.dataset.loader import _cache_is_stale

    src = sample_dir / "connections_princeton.csv.gz"
    old_style = tmp_path / "old.npz"
    np.savez(old_style, pre=np.array([1, 2]), post=np.array([3, 4]))
    assert "no source provenance" in _cache_is_stale(old_style, src)

    good = tmp_path / "good.npz"
    np.savez(good, source_path=np.asarray([str(src.resolve())]),
             source_size=np.asarray([src.stat().st_size]))
    assert _cache_is_stale(good, src) == ""

    wrong_size = tmp_path / "wrong.npz"
    np.savez(wrong_size, source_path=np.asarray([str(src.resolve())]),
             source_size=np.asarray([src.stat().st_size + 1]))
    assert "changed size" in _cache_is_stale(wrong_size, src)


def test_cache_filenames_differ_per_data_directory(tmp_path):
    """The key must include the directory, so two downloads cannot collide."""
    from fruitfly.config import AppConfig
    from fruitfly.graph.build import _load_table

    seen = []

    def fake_loader(data_dir, **kw):
        seen.append(kw["cache_path"])
        raise RuntimeError("stop here; we only want the cache path")

    import fruitfly.dataset.loader as loader_mod

    real = loader_mod.load_connections
    loader_mod.load_connections = fake_loader
    try:
        cfg = AppConfig.load()
        cfg.data.use_cache = True
        for d in (tmp_path / "one", tmp_path / "two"):
            d.mkdir()
            try:
                _load_table(cfg, d, None)
            except RuntimeError:
                pass
    finally:
        loader_mod.load_connections = real

    assert len(seen) == 2 and seen[0] != seen[1], f"cache paths collided: {seen}"
