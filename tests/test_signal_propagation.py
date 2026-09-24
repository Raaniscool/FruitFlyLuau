"""Cross-check the simulator against an independent dense reference model, and
verify the promise made in ``network.py``'s docstring: plasticity writes are
visible to propagation with no rebuild (``test_plasticity_is_visible_to_propagation``).
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.config import LIFConfig, PlasticityConfig
from fruitfly.graph.connectome import build_connectome
from fruitfly.neuro.network import NetworkSimulator
from fruitfly.neuro.plasticity import PlasticityContext, build_rule


def rand_conn(n=40, seed=3, wmin=0.2, wmax=2.0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=n * 3)
    dst = rng.integers(0, n, size=n * 3)
    keep = src != dst
    return build_connectome(src[keep], dst[keep], rng.uniform(wmin, wmax, size=int(keep.sum())), np.arange(n))


def dense_reference(conn, W, p: LIFConfig, steps: int, current: np.ndarray):
    """Straightforward dense re-implementation of the equations in network.py.

    Deliberately naive (full matrices, explicit loops) so that a bug in either
    implementation shows up as a mismatch rather than being shared.
    """
    n = conn.n_neurons
    Wm = np.zeros((n, n), dtype=np.float64)
    coo = conn.matrix.tocoo()
    Wm[coo.row, coo.col] = W  # rows = presynaptic, cols = postsynaptic
    v = np.full(n, p.v_rest)
    g = np.zeros(n)
    ref = np.zeros(n, dtype=np.int32)
    refr = np.zeros(n, dtype=np.int32)
    spikes_out = np.zeros((steps, n), dtype=bool)
    for t in range(steps):
        g = g * np.exp(-p.dt_ms / p.tau_syn_ms) + Wm.T.dot(ref.astype(np.float64))
        i_syn = p.synaptic_gain * g
        i_ext = current[t] if current is not None else 0.0
        dv = (p.dt_ms / p.tau_ms) * (-p.leak * (v - p.v_rest) + i_syn + i_ext)
        v = v + dv
        if p.clip_v:
            v = np.clip(v, p.v_rest - p.clip_v, p.v_rest + p.clip_v)
        fired = (refr <= 0) & (v > p.v_thresh)
        if fired.any():
            v = np.where(fired, p.v_reset, v)
            refr = np.where(fired, max(1, int(round(p.refractory_ms / p.dt_ms))), refr)
        refr = np.maximum(refr - 1, 0)
        ref = fired.astype(np.int32)
        spikes_out[t] = fired
    return spikes_out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_dense_reference_model(seed):
    conn = rand_conn(n=30, seed=seed)
    p = LIFConfig(dt_ms=0.5, tau_ms=6.0, tau_syn_ms=2.0, synaptic_gain=2.0, noise=0.0,
                  background_current=0.0, refractory_ms=2.0, clip_v=40.0)
    net = NetworkSimulator(conn, params=p, seed=seed)
    rng = np.random.default_rng(100 + seed)
    cur = np.zeros((80, conn.n_neurons))
    cur[:20] = rng.choice([0.0, 0.0, 15.0, 30.0], size=(20, conn.n_neurons))
    expected = dense_reference(conn, conn.weights.copy(), p, 80, cur)
    counts, spikes, _ = net.run(80, current_schedule=cur)
    assert np.array_equal(spikes, expected), (
        f"spike count {spikes.sum()} vs reference {expected.sum()} "
        f"(net n={conn.n_neurons}, edges={conn.n_edges})"
    )


def test_drive_is_the_transpose_matvec_of_previous_spikes():
    """``g`` must equal the one-step-delayed W.T @ spikes recurrence, exactly."""
    conn = rand_conn()
    net = NetworkSimulator(conn, params=LIFConfig(dt_ms=0.5, tau_syn_ms=2.0, noise=0.0), seed=0)
    expected_g = np.zeros(net.n)
    for t in range(40):
        prev = net._spikes.copy()  # spikes the integrator will consume on this step
        cur = np.zeros(net.n)
        cur[0] = 25.0 if t < 15 else 0.0
        net.step(cur)
        expected_g = expected_g * np.exp(-net.p.dt_ms / net.p.tau_syn_ms) + np.asarray(conn.matrix.T.dot(prev)).ravel()
        assert np.allclose(net.g, expected_g), f"mismatch at step {t}: {(net.g - expected_g).max()}"
    assert expected_g.max() > 0, "the drive must have been non-trivial for this to mean anything"


def test_plasticity_is_visible_to_propagation():
    """The single most important structural guarantee of the whole simulator.

    ``NetworkSimulator._WT`` is a *view* of the CSR data array. If a learning rule
    changed its own copy of the weights, the dynamics would silently keep running
    on the initial graph and every "learning" result would be fake.
    """
    conn = rand_conn(n=30, seed=5, wmin=4.0, wmax=6.0)
    cfg = LIFConfig(dt_ms=0.5, noise=0.0, synaptic_gain=2.0)
    net = NetworkSimulator(conn, params=cfg, seed=0)
    driven = np.flatnonzero(np.diff(conn.matrix.indptr) > 0)[:3]
    pattern = np.zeros(conn.n_neurons)
    pattern[driven] = 30.0
    sched = np.tile(pattern, (30, 1))

    net.reset()

    net.reset()
    net.weights[:] = 0.0
    counts_zeroed = net.run(30, current_schedule=sched)[0]
    downstream_zeroed = set(np.flatnonzero(counts_zeroed > 0).tolist()) - set(driven.tolist())
    assert downstream_zeroed == set(), "zeroed weights must disconnect the network entirely"

    net.reset()
    net.weights[:] = np.full(conn.n_edges, 6.0)
    counts_strong = net.run(30, current_schedule=sched)[0]
    downstream_strong = set(np.flatnonzero(counts_strong > 0).tolist()) - set(driven.tolist())
    assert downstream_strong, f"restored weights must drive downstream cells ({targets} expected)"
    assert counts_strong.sum() > counts_zeroed.sum()


def test_learning_rule_writes_reach_the_simulator():
    """Same guarantee, exercised through the plasticity rule rather than by hand."""
    conn = rand_conn(n=24, seed=7)
    pcfg = PlasticityConfig(rule="stdp", eta=0.5, normalize="none", w_max=50.0, polarity_lock=False)
    rule = build_rule(pcfg, conn)
    net = NetworkSimulator(conn, params=LIFConfig(noise=0.0), seed=0)
    assert net.weights is conn.weights, "the simulator must read the rule's array"
    before = net.weights.copy()
    rng = np.random.default_rng(1)
    for _ in range(200):
        spikes = (rng.random(conn.n_neurons) < 0.25).astype(np.float32)
        rule.apply(PlasticityContext(t_step=net.t_step, spikes=spikes, spikes_bool=spikes.astype(bool),
                                    modulator=1.0, dt_ms=net.p.dt_ms))
    dw = np.abs(net.weights - before)
    assert dw.max() > 0, "the rule wrote to its own array but not to the live weights"
    assert (dw > 0).sum() > 1
    assert np.shares_memory(net.weights, conn.weights)


def test_accumulate_builds_traces_without_writing():
    conn = rand_conn(n=20, seed=8)
    pcfg = PlasticityConfig(rule="reward_stdp", eta=0.5, normalize="none", polarity_lock=False)
    rule = build_rule(pcfg, conn)
    net = NetworkSimulator(conn, params=LIFConfig(noise=0.0), seed=0)
    w0 = net.weights.copy()
    rng = np.random.default_rng(3)
    rule.begin_trial()
    for t in range(30):
        spikes = (rng.random(conn.n_neurons) < 0.3).astype(np.float32)
        m = rule.accumulate(PlasticityContext(t_step=t, spikes=spikes, spikes_bool=spikes.astype(bool),
                                             modulator=0.0, dt_ms=net.p.dt_ms))
    assert m["edges_updated"] == 0
    assert np.allclose(net.weights, w0), "accumulate() must never write weights"
    assert np.abs(rule.state.eligibility).max() > 0, "accumulate() must build the eligibility trace"
    out = rule.apply_at_trial_end(1.0)
    assert out["abs_dw_sum"] > 0, "a trace built by accumulate() must be cashable at trial end"
