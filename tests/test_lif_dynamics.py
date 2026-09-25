"""LIF dynamics on the sparse connectome: rest, threshold, refractoriness,
direction of propagation, and the live-weight contract plasticity depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import LIFConfig
from fruitfly.graph.connectome import build_connectome
from fruitfly.neuro.network import NetworkSimulator


STRONG = 30.0  # one spike must drive the next cell: gain*W (2*30) >> 10 mV threshold


def chain(n=6, *, weights=None, **kw):
    ids = np.arange(n, dtype=np.int64)
    pre = np.arange(n - 1, dtype=np.int64)
    post = pre + 1
    w = np.full(n - 1, STRONG) if weights is None else np.asarray(weights, float)
    conn = build_connectome(pre, post, w, ids)
    return NetworkSimulator(conn, params=LIFConfig(**kw), seed=1)


def drive(net, steps, neuron=0, amps=20.0):
    sched = np.zeros((steps, net.n), dtype=np.float64)
    sched[:, neuron] = amps
    return net.run(steps, current_schedule=sched)


def test_no_input_means_no_spikes_and_rest_potential():
    net = chain(noise=0.0)
    counts, spikes, rates = net.run(60)
    assert counts.sum() == 0
    assert rates.sum() == 0.0
    assert np.allclose(net.v, net.p.v_rest)
    assert not spikes.any()


def test_suprathreshold_current_makes_the_neuron_fire():
    net = chain()
    counts, spikes, rates = drive(net, 40)
    assert counts[0] >= 1, "20 mV of sustained current must cross a 10 mV threshold"
    first = int(np.flatnonzero(spikes[:, 0])[0])
    # v relaxes toward v_rest + I with tau=6ms, dt=0.5ms: crossing takes a few steps
    assert 1 <= first <= 12, f"first spike at step {first}"
    assert net.v[0] == pytest.approx(net.p.v_reset), "spike must be followed by reset"


def test_subthreshold_current_never_spikes():
    net = chain()
    counts, _, _ = drive(net, 200, amps=5.0)  # 5 mV below the 10 mV threshold
    assert counts.sum() == 0


def test_refractory_period_sets_minimum_isi():
    net = chain(refractory_ms=4.0, dt_ms=0.5)
    _, spikes, _ = drive(net, 60)
    times = np.flatnonzero(spikes[:, 0])
    assert times.size >= 2, "sustained drive should produce several spikes"
    assert np.diff(times).min() >= 8, f"violated 4 ms refractory at dt 0.5: {times}"


def test_activity_propagates_downstream_in_the_right_order():
    net = chain(n=5)
    _, spikes, _ = drive(net, 120)
    first = []
    for i in range(5):
        t = np.flatnonzero(spikes[:, i])
        first.append(int(t[0]) if t.size else -1)
    assert all(f >= 0 for f in first), f"chain must carry activity end to end: {first}"
    assert first == sorted(first), f"activation order should follow the chain: {first}"
    assert all(b > a for a, b in zip(first, first[1:])), "one-step axonal delay per edge"


def test_no_backwards_propagation():
    """Stimulating the *last* neuron must not light up the chain upstream."""
    net = chain(n=5)
    _, spikes, _ = drive(net, 120, neuron=4)
    assert spikes[:, 4].any()
    assert not spikes[:, 0:3].any(), "edges are directed: 3->4 does not mean 4->3"


def test_inhibitory_edge_subtracts_drive():
    ids = np.arange(3, dtype=np.int64)
    exc = build_connectome(np.array([0, 1], np.int64), np.array([2, 2], np.int64),
                           np.array([STRONG, STRONG]), ids)
    conn = build_connectome(np.array([0, 1], np.int64), np.array([2, 2], np.int64),
                            np.array([STRONG, -STRONG]), ids)
    cfg = LIFConfig(noise=0.0)
    a = NetworkSimulator(exc, params=cfg, seed=2)
    b = NetworkSimulator(conn, params=cfg, seed=2)
    sched_a = np.zeros((40, 3)); sched_a[:, :2] = 20.0
    sched_b = sched_a.copy()
    ca, _, _ = a.run(40, current_schedule=sched_a)
    cb, _, _ = b.run(40, current_schedule=sched_b)
    assert ca[2] > 0
    assert cb[2] <= ca[2], "a negative weight cannot increase downstream firing"


def test_weight_matrix_is_the_live_csr_data():
    net = chain()
    assert net.weights is net.conn.matrix.data
    net.weights[:] = 0.0
    counts, _, _ = drive(net, 60)
    assert counts[1:].sum() == 0, "zeroed weights must stop propagation immediately"


def test_diagnostics_match_the_spikes_actually_produced():
    net = chain()
    steps = 100
    counts, _, rates = drive(net, steps)
    d = net.diagnostics(steps)
    assert d.n_steps == steps
    assert d.n_spikes == int(counts.sum())
    total_sec = steps * net.p.dt_ms / 1000.0
    assert d.mean_rate_hz == pytest.approx(d.n_spikes / net.n / total_sec, rel=1e-9)
    assert 0.0 < d.fraction_neurons_ever_active <= 1.0
    assert d.max_single_neuron_rate_hz >= d.mean_rate_hz


def test_set_weights_validates_shape():
    net = chain()
    net.set_weights(np.full(net.conn.n_edges, 0.5))
    assert np.allclose(net.weights, 0.5)
    with pytest.raises(ValueError):
        net.set_weights(np.zeros(3))


def test_invalid_parameters_are_rejected():
    conn = build_connectome(np.array([0], np.int64), np.array([1], np.int64), np.ones(1), np.arange(2))
    with pytest.raises(ValueError):
        NetworkSimulator(conn, params=LIFConfig(dt_ms=0.0))
    with pytest.raises(ValueError):
        NetworkSimulator(conn, params=LIFConfig(v_thresh=-70.0, v_reset=-62.0))


def test_run_sparse_matches_dense_run():
    """The memory-friendly path must be numerically identical, not approximate."""
    n = 8
    ids = np.arange(n, dtype=np.int64)
    pre = np.arange(n - 1, dtype=np.int64)
    conn = build_connectome(pre, pre + 1, np.full(n - 1, 1.0), ids)
    cfg = LIFConfig(noise=0.0)
    dense = NetworkSimulator(conn, params=cfg, seed=5)
    sparse = NetworkSimulator(conn, params=cfg, seed=5)
    sched = np.zeros((50, n)); sched[:10, 0] = 25.0
    cd, _, rd = dense.run(50, current_schedule=sched)
    cs, _, rs = sparse.run_sparse(50, [(np.array([0]), np.array([25.0]))] * 10)
    assert np.array_equal(cd, cs)
    assert np.allclose(rd, rs)
