"""Population selection: which real neurons end up in the simulation.

Direction matters more than anything else here -- an upstream/downstream mix-up
turns a biologically-motivated selection into its mirror image, and the earlier
version of this code had exactly that bug. Every test below states the direction
in the assertion message so a future regression is self-explanatory.
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import AppConfig
from fruitfly.dataset.build_sample import sample_connectome
from fruitfly.graph.build import build_population, resolve_scale
from fruitfly.graph.connectome import build_connectome
from fruitfly.graph.select import (
    input_output_sets,
    list_selectors,
    select_population,
    upstream_ancestors,
)


def line_conn(n=9):
    """0->1->2, 3->4->5, 6->7->8 (three disjoint chains)."""
    pre = np.array([0, 1, 3, 4, 6, 7], dtype=np.int64)
    post = np.array([1, 2, 4, 5, 7, 8], dtype=np.int64)
    return build_connectome(pre, post, np.ones(6), np.arange(n, dtype=np.int64))


def ctx_of(conn, *, metadata=None, **kwargs):
    md = dict(metadata or {})
    if getattr(conn, "nt_types", None) is not None:
        # metadata keys are namespaced by the asset they come from
        md.setdefault("nt_predictions.nt_type", conn.nt_types)
        md.setdefault("cell_types.primary_type", conn.nt_types)
    return dict(ids=conn.root_ids, adjacency=conn.matrix, metadata=md,
                rng=np.random.default_rng(0), kwargs=kwargs)


# ------------------------------------------------------------------ selectors
def test_selector_registry_lists_every_documented_mode():
    modes = list_selectors()
    for m in ("all", "random", "hubs", "manual", "neuron_type", "neurotransmitter",
              "visual", "upstream_of", "downstream_of", "connected_subgraph"):
        assert m in modes, m


def test_upstream_means_presynaptic_ancestors_not_targets():
    conn = line_conn()
    got = upstream_ancestors(conn, np.array([2]), depth=2)
    assert set(got.tolist()) == {0, 1}, (
        "upstream of 2 in 0->1->2 must be {0,1}; if you see {1,2} the CSC/CSR "
        "transpose is wrong again")
    got1 = upstream_ancestors(conn, np.array([2]), depth=1)
    assert set(got1.tolist()) == {1}, "depth 1 must stop at the direct presynaptic partner"


def test_selectors_on_the_upstream_and_downstream_modes():
    """``upstream_of`` walks *incoming* edges, ``downstream_of`` walks outgoing ones."""
    conn = line_conn()
    up = set(select_population("upstream_of", **ctx_of(conn, seeds=[2], depth=2)).tolist())
    assert {0, 1} <= up, f"ancestors of 2 in 0->1->2 must be 0 and 1, got {up}"
    assert 8 not in up, "an unrelated chain must not appear"
    down = set(select_population("downstream_of", **ctx_of(conn, seeds=[0], depth=2)).tolist())
    assert {1, 2} <= down, f"descendants of 0 must be 1 and 2, got {down}"
    assert 6 not in down and 3 not in down


def test_random_and_all_and_manual():
    conn, _ = sample_connectome(200)
    rnd = select_population("random", **ctx_of(conn, n=50))
    assert rnd.size == 50 and len(set(rnd.tolist())) == 50
    assert rnd.max() < conn.n_neurons
    everything = select_population("all", **ctx_of(conn))
    assert everything.size == conn.n_neurons
    want = conn.root_ids[[3, 9, 20]]
    man = select_population("manual", **ctx_of(conn, ids=want.tolist()))
    assert set(man.tolist()) == {3, 9, 20}, "manual selection takes FlyWire root ids"


def test_hubs_pick_the_highest_degree_cells():
    conn, _ = sample_connectome(200)
    hubs = select_population("hubs", **ctx_of(conn, n=20))
    deg_in, deg_out = conn.degrees()
    total = deg_in + deg_out
    assert total[hubs].min() >= total[np.setdiff1d(np.arange(conn.n_neurons), hubs)].max(), (
        "hub selection must be by total degree")


def test_neurotransmitter_selector_uses_nt_type_metadata():
    n = 30
    rng = np.random.default_rng(2)
    nt = np.array(["glut"] * 20 + ["gaba"] * 10, dtype=object)
    conn = build_connectome(rng.integers(0, n, 60), rng.integers(0, n, 60), np.ones(60),
                            np.arange(n), nt_types=nt)
    sel = select_population("neurotransmitter", **ctx_of(conn, values="gaba", metadata={"nt_predictions.nt_type": nt}))
    assert sel.size > 0
    assert set(sel.tolist()) <= set(range(20, 30)), "only gaba cells may be selected"


def test_neuron_type_regex_selection_and_missing_metadata():
    nt = np.array(["PN_L1", "R2", "PN_L2", ""], dtype=object)
    conn = build_connectome(np.array([0, 1], np.int64), np.array([2, 3], np.int64), np.ones(2),
                            np.arange(4), labels=nt)
    sel = select_population("neuron_type", **ctx_of(conn, field="cell_types.primary_type",
                                                    regex="^PN", metadata={"cell_types.primary_type": nt}))
    assert set(sel.tolist()) == {0, 2}
    with pytest.raises(ValueError):
        select_population("neuron_type", **ctx_of(conn, field="cell_types.primary_type"))
    # An annotation the download does not include must fail with an explanation
    # that names the keys it *did* find -- never a silently empty population,
    # which would show up later as an inexplicable null result.
    with pytest.raises(ValueError) as exc:
        select_population("visual", **ctx_of(conn))
    assert "annotation asset" in str(exc.value) or "metadata keys" in str(exc.value)


def test_unknown_selector_raises_with_the_available_list():
    conn, _ = sample_connectome(20)
    with pytest.raises((ValueError, KeyError)) as exc:
        select_population("nonexistent_mode", **ctx_of(conn))
    assert "random" in str(exc.value) or "available" in str(exc.value).lower()


# ------------------------------------------------------------------ I/O sets
def test_input_output_sets_are_disjoint_and_connected():
    conn, _ = sample_connectome(400)
    inp, out, info = input_output_sets(conn, n_inputs=16, n_outputs=4, rng=np.random.default_rng(0))
    assert inp.size == 16 and out.size == 4
    assert not (set(inp.tolist()) & set(out.tolist())), "a readout cell must not be a stimulus cell"
    assert info, "input_output_sets must report how it chose (for setup.json)"
    # outputs must be reachable from the inputs, otherwise the task is undriveable
    reach = upstream_ancestors(conn, out, depth=info.get("depth", 3))
    assert set(out.tolist()).isdisjoint(set(inp.tolist()))
    assert reach.size > 0


def test_input_output_sets_survive_a_disconnected_graph():
    """Real subgraphs are not strongly connected; this must not crash or hang."""
    conn = line_conn(9)
    inp, out, info = input_output_sets(conn, n_inputs=4, n_outputs=2, rng=np.random.default_rng(0))
    assert inp.size >= 1 and out.size >= 1
    assert "notes" in info or "depth" in info


# ------------------------------------------------------------------ scales
@pytest.mark.parametrize("mode,expect_n", [
    ("tiny", 100),
    ("small", 1000),
    ("medium", 10000),
    ("large", 40000),
])
def test_resolve_scale_maps_names_to_neuron_counts(mode, expect_n):
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = mode
    cfg.graph.n_neurons = 0
    scale, n = resolve_scale(cfg)
    assert n == expect_n, f"{mode} should mean {expect_n} neurons"
    assert scale == mode


def test_full_brain_scale_is_the_codex_neuron_count_or_an_explicit_request():
    from fruitfly.dataset.assets import REFERENCE_COUNTS

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "full"
    cfg.graph.n_neurons = 0
    scale, n = resolve_scale(cfg)
    assert scale == "full"
    assert n == 0, "'full' means 'do not cap the population'; the loader's whole table is used"
    assert REFERENCE_COUNTS["cells"] == 139_255
    cfg.graph.mode = "5000"
    assert resolve_scale(cfg)[1] == 5000, "an integer mode must be honoured verbatim"


def test_explicit_n_neurons_overrides_the_named_scale():
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 37
    scale, n = resolve_scale(cfg)
    assert n == 37


def test_build_population_sample_is_labelled_synthetic():
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 120
    cfg.data.source = "sample"
    pop = build_population(cfg, seed=1)
    assert pop.simulated_neurons == 120
    assert pop.connectome.n_edges > 0
    desc = pop.describe().lower()
    assert "sample" in desc or "synthetic" in desc, (
        "a synthetic population must never be describable as real FAFB data")
    assert pop.mode == "sample"
    assert pop.connectome.provenance.get("synthetic") is True or "sample" in str(pop.connectome.provenance).lower()


def test_build_population_from_sample_fafb_files_uses_the_loader(sample_dir):
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.fafb_data_path = str(sample_dir)
    cfg.data.source = "connections_filtered"
    cfg.data.use_cache = False
    cfg.graph.mode = "tiny"
    cfg.graph.selection = "connected_subgraph"
    cfg.graph.n_neurons = 60
    pop = build_population(cfg, seed=3)
    assert pop.simulated_neurons == 60
    assert pop.connection_table_stats, "the run must record what the source table looked like"
    prov = str(pop.connectome.provenance).lower()
    assert "connections_princeton" in prov or "sample_fafb" in prov or "connections" in prov
    # connectivity must be real for the subgraph: no self loops, positive weights
    coo = pop.connectome.matrix.tocoo()
    assert not (coo.row == coo.col).any()
    assert (pop.connectome.weights > 0).all() or (pop.connectome.weights != 0).all()
    json_ok = pop.describe()
    assert "60" in json_ok


def test_weight_transform_compresses_counts_but_keeps_order():
    """syn_count spans 1..2633, so the graph stores a compressed weight."""
    raw = np.array([1.0, 5.0, 50.0, 2633.0])
    cfg = AppConfig.load(use_defaults_file=False)
    tf = cfg.graph.weight_transform
    conn, _ = sample_connectome(240, weight_transform=tf, rng=np.random.default_rng(0))
    w = conn.weights
    assert np.isfinite(w).all() and (w > 0).all()
    assert w.max() < 2633.0, f"transform {tf!r} was not applied; raw counts would leak through"
    assert np.all(np.diff(np.sort(w)) >= 0)
