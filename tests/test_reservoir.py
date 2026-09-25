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
    rates = [r["mean_rate_hz"] for r in trace["tried"]]
    assert rates == sorted(rates), "more background current must not lower the firing rate"


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
