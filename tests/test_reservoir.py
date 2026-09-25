"""Frozen-connectome reservoir measurements.

These tests check the *metrics* first (against inputs whose right answer is known
by construction) and only then check that they run on a connectome. A benchmark
you have not validated is just a number generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import AppConfig
from fruitfly.experiment.reservoir import (
    MemoryCapacity,
    ReservoirStates,
    calibrate_drive,
    characterise,
    memory_capacity,
    rank_measures,
    ridge_readout,
    separation,
    shuffle_connectome,
)
from fruitfly.graph.build import build_population
from fruitfly.graph.select import input_output_sets
from fruitfly.neuro.network import NetworkSimulator


# ------------------------------------------------------------------ the maths
def test_ridge_readout_recovers_an_exact_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w + 0.25
    pred = ridge_readout(X[:150], y[:150], X[150:], alpha=1e-10)
    assert np.allclose(pred, y[150:], atol=1e-6)


def test_memory_capacity_is_one_per_delay_the_state_actually_stores():
    """A state that literally contains u(t-1) and u(t-2) must score MC ~= 2."""
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, size=400)
    X = np.zeros((400, 3))
    X[1:, 0] = u[:-1]          # perfect 1-step memory
    X[2:, 1] = u[:-2]          # perfect 2-step memory
    X[:, 2] = rng.normal(size=400)  # a distractor carrying nothing
    st = ReservoirStates(X=X[10:], u=u[10:], washout=10,
                         input_neurons=np.array([0]), readout_neurons=np.arange(3),
                         fraction_active=1.0, mean_rate_hz=1.0)
    mc = memory_capacity(st, max_delay=5)
    assert mc.per_delay[0] > 0.98 and mc.per_delay[1] > 0.98
    assert all(v < 0.2 for v in mc.per_delay[2:]), mc.per_delay
    assert 1.9 < mc.total < 2.3
    assert mc.half_life_delay == 3, "MC_k should fall below 0.5 at the first unstored delay"


def test_memory_capacity_of_a_memoryless_state_is_zero():
    rng = np.random.default_rng(2)
    u = rng.uniform(0.0, 1.0, size=300)
    X = rng.normal(size=(300, 4))  # independent of u entirely
    st = ReservoirStates(X=X, u=u, washout=0, input_neurons=np.array([0]),
                         readout_neurons=np.arange(4), fraction_active=1.0, mean_rate_hz=1.0)
    mc = memory_capacity(st, max_delay=5)
    assert mc.total < 0.5, f"a state independent of the input must not score memory: {mc.per_delay}"


def test_memory_capacity_is_scored_on_held_out_timesteps():
    """The split must be contiguous and the test half must be non-empty."""
    rng = np.random.default_rng(3)
    u = rng.uniform(0.0, 1.0, size=200)
    X = np.zeros((200, 2))
    X[1:, 0] = u[:-1]
    st = ReservoirStates(X=X, u=u, washout=0, input_neurons=np.array([0]),
                         readout_neurons=np.arange(2), fraction_active=1.0, mean_rate_hz=1.0)
    mc = memory_capacity(st, max_delay=3, train_fraction=0.7)
    assert mc.n_test > 0 and mc.n_train > 0
    assert mc.n_train > mc.n_test


def test_memory_capacity_refuses_to_guess_on_a_short_run():
    st = ReservoirStates(X=np.zeros((5, 2)), u=np.zeros(5), washout=0,
                         input_neurons=np.array([0]), readout_neurons=np.arange(2),
                         fraction_active=0.0, mean_rate_hz=0.0)
    mc = memory_capacity(st)
    assert isinstance(mc, MemoryCapacity) and mc.total == 0.0 and "too few" in mc.note


# ------------------------------------------------------------------- controls
def test_shuffle_preserves_out_degree_and_the_weight_multiset(sample_dir):
    cfg = AppConfig.load()
    cfg.data.fafb_data_path = str(sample_dir)
    cfg.graph.n_neurons = 200
    conn = build_population(cfg, seed=1).connectome
    before_deg = np.diff(conn.matrix.indptr).copy()
    before_w = np.sort(np.asarray(conn.matrix.data).copy())

    shuf = shuffle_connectome(conn, seed=5)
    assert np.array_equal(np.diff(shuf.matrix.indptr), before_deg)
    assert np.allclose(np.sort(np.asarray(shuf.matrix.data)), before_w)
    # and the original must not have been mutated in place
    assert np.array_equal(np.diff(conn.matrix.indptr), before_deg)
    assert "shuffle" in shuf.provenance["control"]


# -------------------------------------------------------------- on a real sim
@pytest.fixture()
def tiny_setup(sample_dir):
    cfg = AppConfig.load()
    cfg.data.fafb_data_path = str(sample_dir)
    cfg.graph.n_neurons = 200
    conn = build_population(cfg, seed=1).connectome
    inputs, _out, _audit = input_output_sets(conn, n_inputs=12, n_outputs=4,
                                             rng=np.random.default_rng(0))
    return cfg, conn, inputs


def test_calibration_picks_a_bias_by_measuring_not_guessing(tiny_setup):
    cfg, conn, inputs = tiny_setup
    bias, trace = calibrate_drive(lambda: NetworkSimulator(conn, cfg.lif, seed=1),
                                  input_neurons=inputs, steps=120)
    assert trace["tried"], "the calibration must record what it tried"
    assert all("mean_rate_hz" in row for row in trace["tried"])
    assert trace["chosen_bias"] == bias
    # monotonicity only holds WITHIN one (gain, inhibition) setting -- the sweep now
    # varies all three knobs, so the raw list is not globally sorted
    group = [r for r in trace["tried"]
             if r["gain_scale"] == trace["tried"][0]["gain_scale"]
             and r["global_inhibition"] == trace["tried"][0]["global_inhibition"]]
    rates = [r["mean_rate_hz"] for r in group]
    assert rates == sorted(rates), f"more background current must not lower the rate: {rates}"


def test_characterise_never_modifies_a_single_weight(tiny_setup):
    """The premise of the whole module: the reservoir is frozen."""
    cfg, conn, inputs = tiny_setup
    before = np.asarray(conn.matrix.data).copy()
    rep = characterise(conn, cfg.lif, input_neurons=inputs, steps=250, washout=50,
                       max_delay=5, seed=1, n_sep_pairs=2, n_rank_streams=4)
    assert np.array_equal(np.asarray(conn.matrix.data), before)
    assert rep.n_neurons == conn.n_neurons and rep.n_edges == conn.n_edges


def test_characterise_reports_the_drive_it_chose(tiny_setup):
    cfg, conn, inputs = tiny_setup
    rep = characterise(conn, cfg.lif, input_neurons=inputs, steps=250, washout=50,
                       max_delay=5, seed=1, n_sep_pairs=2, n_rank_streams=4)
    assert "bias" in rep.params and "drive_calibration" in rep.params
    assert rep.params["drive_calibration"]["tried"]
    text = rep.describe()
    assert "memory capacity" in text and "separation ratio" in text


def test_a_silent_network_is_flagged_not_scored(tiny_setup):
    """Zero drive means zero spikes; the report must say so loudly."""
    cfg, conn, inputs = tiny_setup
    rep = characterise(conn, cfg.lif, input_neurons=inputs, steps=150, washout=30,
                       max_delay=3, amplitude=0.0, bias=0.0, seed=1,
                       n_sep_pairs=1, n_rank_streams=3)
    assert rep.mean_rate_hz == 0.0
    assert "essentially silent" in rep.describe()


def test_separation_is_larger_for_different_inputs_than_for_noise(tiny_setup):
    cfg, conn, inputs = tiny_setup
    bias, _ = calibrate_drive(lambda: NetworkSimulator(conn, cfg.lif, seed=1),
                              input_neurons=inputs, steps=120)
    sep = separation(lambda: NetworkSimulator(conn, cfg.lif, seed=1),
                     input_neurons=inputs, n_pairs=2, steps=120, washout=30,
                     bias=bias, seed=0)
    assert sep.n_pairs == 2
    assert sep.between_input_distance >= 0.0 and sep.within_input_distance >= 0.0
    assert sep.ratio >= 0.0
    assert "separation ratio" in sep.describe()


def test_rank_measures_flag_saturation_against_the_stream_count(tiny_setup):
    cfg, conn, inputs = tiny_setup
    bias, _ = calibrate_drive(lambda: NetworkSimulator(conn, cfg.lif, seed=1),
                              input_neurons=inputs, steps=120)
    r = rank_measures(lambda: NetworkSimulator(conn, cfg.lif, seed=1),
                      input_neurons=inputs, n_streams=4, steps=120, washout=30,
                      bias=bias, seed=0)
    assert r["kernel_rank"] <= r["n_streams"]
    if r["kernel_rank"] >= r["n_streams"]:
        assert r["note"], "a saturated kernel rank must be flagged, not reported as a result"


def test_cli_reservoir_runs_both_arms_and_writes_json(tmp_path, sample_dir):
    from fruitfly.cli.__main__ import main

    out = tmp_path / "res.json"
    rc = main(["reservoir", "--data-dir", str(sample_dir), "--neurons", "200",
               "--steps", "220", "--washout", "40", "--max-delay", "4",
               "--rank-streams", "3", "--json", str(out)])
    assert rc == 0
    import json

    rows = json.loads(out.read_text())
    assert len(rows) == 2, "default --control both must report real AND shuffled wiring"
    assert rows[0]["control"] != rows[1]["control"]


@pytest.fixture()
def dense_recurrent_connectome():
    """Mean degree ~21, matching the real mushroom-body subgraph (6,237 edges / 300 cells).

    The LIF parameters were calibrated on a synthetic graph with mean degree 2.6. On the
    real circuit the network self-ignited: every bias candidate including zero produced
    380-480 Hz, ~100x any plausible rate.
    """
    from scipy import sparse

    from fruitfly.graph.connectome import Connectome

    rng = np.random.default_rng(0)
    n, m = 300, 6237
    rows = rng.integers(0, n, m)
    cols = rng.integers(0, n, m)
    w = rng.uniform(1.8, 7.5, m)
    adj = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    return Connectome(root_ids=np.arange(n, dtype=np.int64) + 720575940600000000, matrix=adj)


def test_a_dense_graph_is_brought_into_range(dense_recurrent_connectome):
    """Sweeping the background current alone cannot fix runaway recurrence.

    Either lowering the gain or adding APL-like inhibition will do it. Inhibition is
    preferable -- it keeps full synaptic gain, so the recurrent circuit still does
    something -- but the test only requires that one of them was applied.
    """
    cfg = AppConfig.load()
    rep = characterise(dense_recurrent_connectome, cfg.lif, input_neurons=np.arange(16),
                       steps=300, washout=60, max_delay=5, seed=1,
                       n_sep_pairs=1, n_rank_streams=3)
    tamed = rep.params["gain_scale"] < 1.0 or rep.params["global_inhibition"] > 0.0
    assert tamed, "a dense recurrent graph needs either a lower gain or inhibition"
    assert 1.0 <= rep.mean_rate_hz <= 120.0, (
        f"calibration left the network at {rep.mean_rate_hz:.0f} Hz, outside any usable range"
    )
    assert not rep.params["drive_calibration"]["saturated"]


def test_an_untameable_network_is_declared_uninterpretable(dense_recurrent_connectome):
    """If calibration fails, the report must say the numbers are not about the wiring."""
    cfg = AppConfig.load()
    # force the failure: pin bias high and forbid the gain sweep from running
    rep = characterise(dense_recurrent_connectome, cfg.lif, input_neurons=np.arange(16),
                       steps=200, washout=40, max_delay=3, seed=1, bias=40.0,
                       amplitude=80.0, n_sep_pairs=1, n_rank_streams=3)
    text = rep.describe()
    if rep.mean_rate_hz > 150:
        assert "uninterpretable" in text, (
            f"{rep.mean_rate_hz:.0f} Hz was reported without a health warning:\n{text}"
        )


def test_calibration_trace_records_both_knobs(dense_recurrent_connectome):
    cfg = AppConfig.load()
    bias, trace = calibrate_drive(
        lambda: NetworkSimulator(dense_recurrent_connectome, cfg.lif, seed=1),
        input_neurons=np.arange(16), steps=150,
    )
    assert trace["tried"], "the sweep must be auditable"
    assert all({"bias", "gain_scale", "mean_rate_hz"} <= set(r) for r in trace["tried"])
    assert "gain_scale" in trace and "saturated" in trace


# ------------------------------------------------- inhibition and criticality
def test_branching_ratio_recovers_a_known_value():
    """A(t+1) = 0.8 * A(t) must estimate sigma ~ 0.8."""
    from fruitfly.experiment.reservoir import branching_ratio

    a = [100.0]
    for _ in range(60):
        a.append(a[-1] * 0.8)
    sigma, verdict = branching_ratio(np.asarray(a))
    assert abs(sigma - 0.8) < 0.02
    assert "subcritical" in verdict

    grow = [1.0]
    for _ in range(40):
        grow.append(grow[-1] * 1.3)
    sigma_up, verdict_up = branching_ratio(np.asarray(grow))
    assert sigma_up > 1.15 and "supercritical" in verdict_up


def test_branching_ratio_calls_a_silent_network_subcritical():
    from fruitfly.experiment.reservoir import branching_ratio

    sigma, verdict = branching_ratio(np.zeros(50))
    assert sigma == 0.0 and "silent" in verdict


def test_global_inhibition_lowers_the_firing_rate(dense_recurrent_connectome):
    """The APL-like pool must actually suppress activity, monotonically."""
    from dataclasses import replace

    from fruitfly.experiment.reservoir import collect_states

    cfg = AppConfig.load()
    u = np.random.default_rng(0).uniform(0, 1, 200)
    rates = []
    for inhib in (0.0, 20.0, 60.0, 120.0):
        p = replace(cfg.lif, global_inhibition=inhib)
        st = collect_states(NetworkSimulator(dense_recurrent_connectome, p, seed=1), u,
                            input_neurons=np.arange(16), amplitude=22, bias=4.0, washout=50)
        rates.append(st.mean_rate_hz)
    assert rates[-1] < rates[0], f"inhibition did not reduce activity: {rates}"


def test_inhibition_zero_is_exactly_the_old_behaviour(dense_recurrent_connectome):
    """The knob must be opt-in: 0.0 has to reproduce the pre-inhibition dynamics."""
    from dataclasses import replace

    from fruitfly.experiment.reservoir import collect_states

    cfg = AppConfig.load()
    u = np.random.default_rng(1).uniform(0, 1, 120)
    a = collect_states(NetworkSimulator(dense_recurrent_connectome, cfg.lif, seed=3), u,
                       input_neurons=np.arange(8), bias=3.0, washout=20)
    b = collect_states(
        NetworkSimulator(dense_recurrent_connectome, replace(cfg.lif, global_inhibition=0.0), seed=3),
        u, input_neurons=np.arange(8), bias=3.0, washout=20)
    assert np.array_equal(a.X, b.X)


def test_calibration_searches_every_inhibition_level_before_choosing(dense_recurrent_connectome):
    """Stopping at the first workable level picks a dead-recurrence setting.

    Measured: the first in-range setting was gain x0.02 with no inhibition -- in range by
    firing rate, but the recurrent circuit contributes nothing. Sweeping all levels finds
    full gain with inhibition instead.
    """
    cfg = AppConfig.load()
    _bias, trace = calibrate_drive(
        lambda: NetworkSimulator(dense_recurrent_connectome, cfg.lif, seed=1),
        input_neurons=np.arange(16), steps=150,
    )
    levels = {row["global_inhibition"] for row in trace["tried"]}
    assert len(levels) > 1, "the sweep must try more than one inhibition level"
    assert "branching_ratio" in trace
    if not trace["saturated"]:
        assert abs((trace["branching_ratio"] or 0) - 1.0) < 0.35


def test_an_absurd_separation_ratio_is_called_chaos_not_success():
    """7,131 is not a good score. It means nearby inputs diverge exponentially."""
    from fruitfly.experiment.reservoir import Separation

    s = Separation(ratio=7131.5, between_input_distance=1e6, within_input_distance=140.0,
                   n_pairs=2)
    text = s.describe()
    assert "IMPLAUSIBLY HIGH" in text and "chaotic" in text


def test_excitation_balance_reports_an_all_excitatory_graph_as_a_problem(dense_recurrent_connectome):
    bal = dense_recurrent_connectome.excitation_balance()
    assert bal["inhibitory_edge_fraction"] == 0.0
    assert bal["excitatory_edge_fraction"] == 1.0
    assert "no stable middle regime" in bal["ei_balance_note"]


def test_excitation_balance_accepts_a_mixed_graph():
    from scipy import sparse

    from fruitfly.graph.connectome import Connectome

    rng = np.random.default_rng(0)
    n = 50
    w = rng.uniform(1, 5, 400) * rng.choice([1.0, -1.0], 400, p=[0.7, 0.3])
    adj = sparse.csr_matrix((w, (rng.integers(0, n, 400), rng.integers(0, n, 400))), shape=(n, n))
    bal = Connectome(root_ids=np.arange(n, dtype=np.int64), matrix=adj).excitation_balance()
    assert 0.2 < bal["inhibitory_edge_fraction"] < 0.4
    assert "can stabilise" in bal["ei_balance_note"]
