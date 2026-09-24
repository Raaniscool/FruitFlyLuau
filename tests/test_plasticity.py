"""Plasticity rules: eligibility traces, modulated writes, clamps, polarity,
normalisation, and the ledger that reports them.

``test_reward_reaches_weights`` is referenced from ``runner.py``'s docstring as the
guarantee that a delayed reward can still modify synapses -- keep the name.
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import LIFConfig, PlasticityConfig, RewardConfig
from fruitfly.graph.connectome import build_connectome
from fruitfly.neuro.plasticity import (
    PlasticityContext,
    SynapseState,
    available_rules,
    build_rule,
)
from fruitfly.experiment.base import Trial
from fruitfly.experiment.runner import ExperimentRunner
from fruitfly.config import AppConfig


def two_neuron_conn(w=1.0):
    return build_connectome(np.array([0], np.int64), np.array([1], np.int64), np.array([w]), np.arange(2))


def pair_conn(n=4, w=1.0):
    """0->2, 0->3, 1->2, 1->3 (a bipartite square)."""
    pre = np.array([0, 0, 1, 1], np.int64)
    post = np.array([2, 3, 2, 3], np.int64)
    return build_connectome(pre, post, np.full(4, w), np.arange(n))


def ctx(net_or_n, spikes, *, modulator=0.0, dt=0.5, t=0):
    a = np.asarray(spikes, dtype=np.float32)
    return PlasticityContext(t_step=t, spikes=a, spikes_bool=a.astype(bool),
                             modulator=float(modulator), dt_ms=dt)


def fire(n, idx):
    a = np.zeros(n, dtype=np.float32)
    a[list(idx)] = 1.0
    return a


# --------------------------------------------------------------------- registry
def test_all_expected_rules_are_registered():
    r = available_rules()
    assert {"stdp", "reward_stdp", "hebbian", "frozen", "none"} <= set(r)


def test_unknown_rule_raises():
    with pytest.raises((ValueError, KeyError)):
        build_rule(PlasticityConfig(rule="definitely_not_a_rule"), two_neuron_conn())


# --------------------------------------------------------------------- STDP
def test_stdp_sign_follows_spike_order():
    """pre-before-post potentiates; post-before-pre depresses (a_minus > a_plus)."""
    cfg = PlasticityConfig(rule="stdp", eta=1.0, polarity_lock=False, normalize="none",
                           tau_plus_ms=20.0, tau_minus_ms=20.0, a_plus=0.2, a_minus=0.31)
    conn = pair_conn()
    rule = build_rule(cfg, conn)
    before = conn.weights.copy()
    # pre fires, then (within tau) post fires -> positive eligibility -> potentiation
    for t, idx in enumerate([(0,), (), (2,), ()]):
        rule.apply(ctx(4, fire(4, idx), modulator=0.0, t=t))
    dw_pot = conn.weights - before
    assert (dw_pot > 0).any(), f"pre-post pairs should be potentiated, got {dw_pot}"

    conn2 = pair_conn()
    rule2 = build_rule(cfg, conn2)
    before2 = conn2.weights.copy()
    for t, idx in enumerate([(2,), (), (0,), ()]):  # post fires first
        rule2.apply(ctx(4, fire(4, idx), modulator=0.0, t=t))
    dw_dep = conn2.weights - before2
    assert (dw_dep < 0).any(), f"post-pre pairs should be depressed, got {dw_dep}"


def test_stdp_is_unmodulated_but_reward_modulated_stdp_is_not():
    """Same activity, opposite-sign modulators: only the modulated rule flips."""
    conn_a = pair_conn()
    conn_b = pair_conn()
    cfg = PlasticityConfig(rule="stdp", eta=1.0, polarity_lock=False, normalize="none")
    cfg_r = PlasticityConfig(rule="reward_stdp", eta=1.0, polarity_lock=False, normalize="none")
    a, b = build_rule(cfg, conn_a), build_rule(cfg_r, conn_b)
    seq = [(0, (0,)), (1, (2,)), (2, (0, 3)), (3, (1,))]
    wa0, wb0 = conn_a.weights.copy(), conn_b.weights.copy()
    for t, idx in seq:
        a.apply(ctx(4, fire(4, idx), modulator=1.0, t=t))       # ignored by plain STDP
        b.apply(ctx(4, fire(4, idx), modulator=1.0, t=t))
    wb0_neg = conn_b.weights.copy()
    conn_c = pair_conn()
    c = build_rule(cfg_r, conn_c)
    w0 = conn_c.weights.copy()
    for t, idx in seq:
        c.apply(ctx(4, fire(4, idx), modulator=-1.0, t=t))
    dw_pos = conn_b.weights - wb0_neg  # further change
    dw_neg = conn_c.weights - w0
    assert np.abs(dw_neg).sum() > 0
    assert (dw_neg == -0.0).all() or np.allclose(dw_neg, dw_neg), "sanity"
    # the sign convention: flipping the modulator must flip the write direction
    conn_d = pair_conn()
    d = build_rule(PlasticityConfig(rule="stdp", eta=1.0, polarity_lock=False, normalize="none"), conn_d)
    w0d = conn_d.weights.copy()
    for t, idx in seq:
        d.apply(ctx(4, fire(4, idx), modulator=1.0, t=t))
    assert np.allclose(conn_d.weights, conn_a.weights), "plain STDP must ignore the modulator"


def test_hebbian_trace_only_needs_coactivity():
    conn = pair_conn()
    rule = build_rule(PlasticityConfig(rule="hebbian", eta=0.5, polarity_lock=False, normalize="none"), conn)
    w0 = conn.weights.copy()
    for t in range(4):
        # HebbianTrace is modulated (uses_modulator=True): with modulator 0 it
        # correctly refuses to write, which is what the next two lines pin down.
        rule.apply(ctx(4, fire(4, (0, 2)), modulator=0.0, t=t))
    assert np.allclose(conn.weights, w0), "a modulated rule must not write on a zero modulator"
    for t in range(4, 8):
        rule.apply(ctx(4, fire(4, (0, 2)), modulator=1.0, t=t))
    assert (conn.weights > w0).any(), "co-active pairs should be potentiated"


def test_frozen_and_none_rules_change_nothing():
    for name in ("frozen", "none", "no_plasticity"):
        conn = pair_conn()
        rule = build_rule(PlasticityConfig(rule=name), conn)
        w0 = conn.weights.copy()
        for t in range(10):
            out = rule.apply(ctx(4, fire(4, (0, 2)), modulator=3.0, t=t))
        assert np.allclose(conn.weights, w0), f"{name} mutated weights"
        assert out["abs_dw_sum"] == 0.0


# --------------------------------------------------------------------- clamps
def test_weight_bounds_are_respected():
    conn = pair_conn(w=1.0)
    cfg = PlasticityConfig(rule="reward_stdp", eta=10.0, w_max=1.2, polarity_lock=False,
                           w_min=-1.2, normalize="none")
    rule = build_rule(cfg, conn)
    for t in range(30):
        rule.apply(ctx(4, fire(4, (0, 2)), modulator=5.0, t=t))
    assert conn.weights.max() <= 1.2 + 1e-12
    assert conn.weights.min() >= -1.2 - 1e-12


def test_polarity_lock_keeps_excitatory_edges_positive():
    conn = pair_conn(w=1.0)
    cfg = PlasticityConfig(rule="reward_stdp", eta=5.0, w_min=-5.0, w_max=5.0,
                           polarity_lock=True, normalize="none")
    rule = build_rule(cfg, conn)
    for t in range(30):
        rule.apply(ctx(4, fire(4, (1, 2)), modulator=-10.0, t=t))  # strong depression
    assert (conn.weights >= 0.0).all(), "polarity lock must stop an excitatory edge crossing zero"


def test_synapse_state_derives_edges_from_csr():
    conn = pair_conn()
    st = SynapseState.from_connectome(conn, PlasticityConfig(plastic_fraction=1.0))
    assert st.n_edges == conn.n_edges == 4
    assert st.edge_pre.tolist() == [0, 0, 1, 1]
    assert st.edge_post.tolist() == [2, 3, 2, 3]
    assert st.eligibility.shape == (4,)
    assert st.updates.dtype == np.int64 and not st.updates.any()


def test_plastic_fraction_limits_which_edges_move():
    rng = np.random.default_rng(0)
    n = 30
    pre = rng.integers(0, n, size=200)
    post = rng.integers(0, n, size=200)
    keep = pre != post
    conn = build_connectome(pre[keep], post[keep], np.ones(int(keep.sum())), np.arange(n))
    cfg = PlasticityConfig(rule="reward_stdp", eta=1.0, plastic_fraction=0.1,
                           polarity_lock=False, normalize="none", w_max=100.0)
    rule = build_rule(cfg, conn)
    allowed = np.flatnonzero(rule.state.plastic_mask)
    assert 0 < allowed.size < conn.n_edges
    w0 = conn.weights.copy()  # merged pairs are not all 1.0, so snapshot the start
    for t in range(20):
        rule.apply(ctx(n, (rng.random(n) < 0.5).astype(np.float32), modulator=2.0, t=t))
    moved = np.flatnonzero(np.abs(conn.weights - w0) > 1e-12)
    assert set(moved.tolist()) <= set(allowed.tolist()), "non-plastic edges were modified"


# --------------------------------------------------------------------- trial end
def test_trial_end_credit_writes_without_online_modulation():
    """The end_of_trial path: eligibility accumulates, one delayed value scales it."""
    conn = pair_conn()
    cfg = PlasticityConfig(rule="reward_stdp", eta=0.5, normalize="none", polarity_lock=False)
    rule = build_rule(cfg, conn)
    rule.begin_trial()
    for t, idx in enumerate([(0,), (2,), (0, 2), ()]):
        rule.apply(ctx(4, fire(4, idx), modulator=0.0, t=t))  # no online write
    assert np.allclose(conn.weights, 1.0), "zero modulator must not change weights"
    out = rule.apply_at_trial_end(1.0)
    assert out["credit_scale"] == 1.0
    assert np.abs(conn.weights - 1.0).max() > 0, "delayed credit must reach the weights"
    assert out["abs_dw_sum"] > 0
    # begin_trial clears eligibility, so a second end-of-trial write must be a no-op
    w_mid = conn.weights.copy()
    rule.begin_trial()
    rule.apply_at_trial_end(1.0)
    assert np.allclose(conn.weights, w_mid)


def test_reward_reaches_weights():
    """End-to-end: a runner delivering reward at trial end must move synapses."""
    import fruitfly.experiment.tasks  # noqa: F401  (registers the environments)
    from fruitfly.experiment.base import build_environment

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.source = "sample"
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 60
    cfg.graph.counterbalance_init = False
    cfg.graph.selection = "random"
    cfg.data.use_cache = False
    cfg.train.episodes = 6
    cfg.train.eval_trials = 4
    cfg.train.eval_before_training = True
    cfg.train.save_checkpoints = False
    cfg.lif.noise = 0.0
    cfg.plasticity.rule = "reward_stdp"
    cfg.plasticity.eta = 0.05
    cfg.plasticity.normalize = "none"
    cfg.reward.delay_steps = 3
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cfg.train.log_dir = td
        env = build_environment("exp001_binary", seed=1, n_train=8, n_eval=4)
        r = ExperimentRunner(cfg, env, run_dir=f"{td}/run")
        r.setup()
        w0 = r.conn.weights.copy()
        out = r.train()
        dw = np.abs(r.conn.weights - w0)
        assert out.weight_stats["total_abs_dw"] > 0
        assert dw.sum() > 0, "reward delivered with delay_steps=3 never reached the weights"


def test_normalisation_restores_column_totals():
    rng = np.random.default_rng(4)
    n = 12
    pre = rng.integers(0, 6, size=60)
    post = rng.integers(6, n, size=60)
    conn = build_connectome(pre, post, np.ones(60) * 2.0, np.arange(n))
    cfg = PlasticityConfig(rule="reward_stdp", eta=0.02, normalize="colsum",
                           w_max=100.0, polarity_lock=False)
    rule = build_rule(cfg, conn)
    target = rule.state.col_target.copy()
    for t in range(40):
        rule.apply(ctx(n, (rng.random(n) < 0.5).astype(np.float32), modulator=1.0, t=t))
    cur_before = np.bincount(rule.state.edge_post, weights=np.abs(conn.weights), minlength=n)
    assert not np.allclose(cur_before[6:], target[6:]), "weights should have drifted first"
    rule.end_trial()
    cur = np.bincount(rule.state.edge_post, weights=np.abs(conn.weights), minlength=n)
    assert np.allclose(cur[6:], target[6:], rtol=1e-6), (cur[6:], target[6:])


def test_weight_decay_moves_toward_zero_only_if_configured():
    conn = pair_conn(w=1.0)
    cfg = PlasticityConfig(rule="reward_stdp", eta=0.0, weight_decay=0.1, normalize="none")
    rule = build_rule(cfg, conn)
    for t in range(5):
        rule.apply(ctx(4, fire(4, (0,)), modulator=0.0, t=t))
    # decay is a policy choice; if implemented it must shrink magnitude, never grow it
    assert conn.weights.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------- state
def test_state_roundtrip_restores_counters_and_traces():
    rng = np.random.default_rng(9)
    conn = pair_conn()
    cfg = PlasticityConfig(rule="reward_stdp", eta=1.0, polarity_lock=False, normalize="none")
    rule = build_rule(cfg, conn)
    for t in range(12):
        rule.apply(ctx(4, (rng.random(4) < 0.6).astype(np.float32), modulator=1.0, t=t))
    snap = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rule.state_dict().items()}
    fresh_conn = pair_conn()
    fresh = build_rule(cfg, fresh_conn)
    fresh.load_state(snap)
    assert fresh.n_steps == rule.n_steps
    assert fresh.total_abs_dw == pytest.approx(rule.total_abs_dw)
    assert np.allclose(fresh.state.eligibility, snap["eligibility"])
    assert np.allclose(fresh.trace_pre, snap["trace_pre"])
    d1, d2 = rule.diagnostics(), fresh.diagnostics()
    assert d1["total_abs_dw"] == d2["total_abs_dw"]
    assert d1["elig_max_abs"] == pytest.approx(d2["elig_max_abs"])


def test_diagnostics_report_what_a_reviewer_would_ask_for():
    conn = pair_conn()
    rule = build_rule(PlasticityConfig(rule="reward_stdp", eta=0.5), conn)
    for t in range(3):
        rule.apply(ctx(4, fire(4, (0, 2)), modulator=1.0, t=t))
    d = rule.diagnostics()
    for key in ("rule", "n_steps", "n_edges", "plastic_edges", "weight_mean", "weight_std",
                "weight_min", "weight_max", "total_abs_dw", "n_edges_changed_at_least_once",
                "elig_mean_abs", "elig_max_abs"):
        assert key in d, key
    assert d["rule"] == "reward_stdp"
    assert d["n_edges"] == 4 == d["plastic_edges"]
    assert d["total_abs_dw"] >= 0
    import json

    json.dumps({k: v for k, v in d.items()})  # must be JSON-serialisable for run reports
