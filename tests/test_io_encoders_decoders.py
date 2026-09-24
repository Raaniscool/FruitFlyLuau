"""Sensory drive in, behavioural readout out.

The encoder is the only place an observation touches the network, and the decoder
is the only place a decision leaves it, so both are tested for determinism,
injectivity and honest state round-tripping (a resumed run must read out the same
classes it was trained on).
"""

from __future__ import annotations

import numpy as np
import pytest

from fruitfly.graph.connectome import build_connectome
from fruitfly.io.decoder import (Decision, GroupRateDecoder, PopulationRateDecoder,
                                 ThresholdDecoder, TokenDecoder, available_decoders,
                                 build_decoder)
from fruitfly.io.encoder import (BinaryEncoder, BitVectorEncoder, ConstantCurrentEncoder,
                                 InjectionPlan, SequenceEncoder, SymbolEncoder,
                                 available_encoders, build_encoder)


@pytest.fixture(scope="module")
def conn():
    rng = np.random.default_rng(0)
    n = 60
    pre = rng.integers(0, n, size=400)
    post = rng.integers(0, n, size=400)
    keep = pre != post
    return build_connectome(pre[keep], post[keep], np.ones(int(keep.sum())), np.arange(n))


def configured(enc, conn, seed=0, vocab=None):
    enc.configure(conn, np.random.default_rng(seed), vocab=vocab)
    return enc


# ------------------------------------------------------------------ encoders
def test_encoder_registry_is_complete():
    assert {"binary", "bitvector", "symbol", "sequence", "current"} <= set(available_encoders())
    with pytest.raises((ValueError, KeyError)):
        build_encoder("not_an_encoder")


def test_plan_shape_and_targeting(conn):
    enc = configured(BinaryEncoder(n_input_neurons=8, amplitude=22.0, duration_steps=10), conn)
    plan = enc.encode(1, n_steps=10)
    assert isinstance(plan, InjectionPlan)
    assert plan.n_steps == 10
    assert len(plan.per_step) == 10
    for idx, amp in plan.per_step:
        assert idx.size <= enc.input_neurons.size
        assert set(idx.tolist()) <= set(enc.input_neurons.tolist()), "stimulus must stay on assigned cells"
        assert (np.asarray(amp) > 0).all()
    assert plan.total_charge() > 0


def test_different_observations_drive_different_cells(conn):
    enc = configured(BinaryEncoder(n_input_neurons=8), conn)
    a, b = enc.pattern_for(0), enc.pattern_for(1)
    assert not np.array_equal(a, b), "0 and 1 must not be encoded identically"
    assert enc.code_for(0) != enc.code_for(1)
    # amplitude/phase are identical, only the cell selection differs
    pa, pb = enc.encode(0, n_steps=8), enc.encode(1, n_steps=8)
    assert pa.per_step[0][0].tolist() != pb.per_step[0][0].tolist()


def test_encoding_is_deterministic_for_a_given_configuration(conn):
    def once():
        e = configured(SymbolEncoder(symbols=["a", "b", "c"], n_input_neurons=12), conn, seed=5)
        return np.concatenate([e.pattern_for(s).astype(np.int64) for s in ("a", "b", "c")]), e.input_neurons

    x, xs = once()
    y, ys = once()
    assert np.array_equal(x, y) and np.array_equal(xs, ys)


def test_symbol_encoder_grows_its_vocabulary_instead_of_raising(conn):
    """Documented design: unknown symbols get a stable hashed mask, not an error.

    That is what lets one encoder serve Luau tokens in phase 7, but it also means a
    typo in a task definition silently becomes a new pattern -- hence the
    explicit-vocabulary check below.
    """
    enc = configured(SymbolEncoder(symbols=["x", "y"]), conn)
    a = enc.pattern_for("zzz")
    b = enc.pattern_for("zzz")
    assert np.array_equal(a, b), "masking must be deterministic for a repeated symbol"
    assert not np.array_equal(a, enc.pattern_for("x")), "a new symbol must not alias an existing one"
    assert enc.pattern_for([1, 2]).dtype == bool


def test_bitvector_encoder_uses_one_cell_per_bit(conn):
    enc = configured(BitVectorEncoder(width=4, n_input_neurons=16), conn)
    assert enc.input_neurons.size >= 8, "need at least 2 cells per bit (on/off)"
    p0, p15 = enc.pattern_for(0), enc.pattern_for(15)
    assert not np.array_equal(p0, p15)
    assert p0.sum() == 0 or p0.sum() < p15.sum() + 8  # more bits set -> more cells driven


def test_sequence_encoder_offsets_items_over_time(conn):
    enc = configured(SequenceEncoder(symbols=["a", "b"], steps_per_item=4, n_input_neurons=8), conn)
    plan = enc.encode(["a", "b"], n_steps=8)
    assert plan.n_steps == 8
    active_first = np.concatenate([plan.per_step[t][0] for t in range(4)])
    active_second = np.concatenate([plan.per_step[t][0] for t in range(4, 8)])
    assert active_first.size > 0 and active_second.size > 0
    # the two items are separated in time; if they used the same cells the
    # sequence would be indistinguishable from its multiset
    assert set(active_first.tolist()) != set(active_second.tolist()) or enc.allow_reuse


def test_constant_current_encoder_is_tonic(conn):
    enc = configured(ConstantCurrentEncoder(amplitude=15.0, duration_steps=6), conn)
    plan = enc.encode(1, n_steps=6)
    amps = [np.asarray(a).sum() for _, a in plan.per_step]
    assert all(x > 0 for x in amps[:6]), "tonic drive must persist for the whole window"


def test_encoder_state_roundtrip(conn):
    enc = configured(BinaryEncoder(n_input_neurons=8), conn, seed=3)
    snap = enc.state_dict()
    fresh = build_encoder(snap["kind"])
    fresh.load_state(snap)
    assert np.array_equal(fresh.input_neurons, enc.input_neurons)
    assert np.array_equal(fresh.pattern_for(1), enc.pattern_for(1))


# ------------------------------------------------------------------ decoders
def test_decoder_registry():
    assert {"group_rate", "population_rate", "threshold", "token"} <= set(available_decoders())
    with pytest.raises((ValueError, KeyError)):
        build_decoder("nope")


def make_dec(cls, conn, classes=(0, 1), seed=0, **kw):
    d = cls(n_output_neurons=8, **kw)
    d.configure(conn, np.random.default_rng(seed), classes=list(classes))
    return d


def test_group_rate_reads_the_louder_population(conn):
    d = make_dec(GroupRateDecoder, conn)
    counts = np.zeros(conn.n_neurons)
    g0, g1 = d.groups_for_classes()
    counts[g1] = 7  # class 1's group is the loud one
    dec = d.decode(counts)
    assert dec.label == 1
    assert dec.scores[1] > dec.scores[0]
    assert dec.margin == pytest.approx(dec.scores[1] - dec.scores[0])
    assert dec.confidence >= 0.5
    assert set(np.concatenate(d.groups_for_classes()).tolist()) == set(d.output_neurons.tolist())


def test_silence_is_reported_as_no_decision_not_a_guess(conn):
    d = make_dec(GroupRateDecoder, conn)
    dec = d.decode(np.zeros(conn.n_neurons))
    assert dec.label is None, "a silent readout must not be scored as a correct 0"
    assert dec.scores.sum() == 0
    assert d.decode(np.zeros(conn.n_neurons), per_step=np.zeros((10, conn.n_neurons))).evidence is not None


def test_class_permutation_relabels_the_decision(conn):
    """This is the lever the readout-mapping counterbalance pulls."""
    counts = np.zeros(conn.n_neurons)
    d = make_dec(GroupRateDecoder, conn, classes=(0, 1))
    g0, _ = d.groups_for_classes()
    counts[g0] = 5
    assert d.decode(counts).label == 0
    d.classes = (1, 0)
    assert d.decode(counts).label == 1, "swapping the mapping must swap the label"
    d2 = make_dec(GroupRateDecoder, conn, classes=(0, 1, 2, 3))
    groups = d2.groups_for_classes()
    assert len(groups) == 4
    counts2 = np.zeros(conn.n_neurons)
    counts2[groups[3]] = 9
    assert d2.decode(counts2).label == 3


def test_population_rate_is_a_single_signed_axis(conn):
    d = make_dec(PopulationRateDecoder, conn)
    counts = np.zeros(conn.n_neurons)
    counts[d.output_neurons] = 1.0
    dec = d.decode(counts)
    assert dec.label in list(d.classes)
    assert dec.scores.size == 2


def test_threshold_decoder_is_spike_or_no_spike(conn):
    d = make_dec(ThresholdDecoder, conn)
    off = d.decode(np.zeros(conn.n_neurons))
    assert off.label in (0, None), "no spikes must not be reported as class 1"
    g0, g1 = d.groups_for_classes()
    counts = np.zeros(conn.n_neurons)
    counts[g1] = 3
    on = d.decode(counts)
    assert on.label == 1
    counts2 = np.zeros(conn.n_neurons)
    counts2[g0] = 3
    assert d.decode(counts2).label == 0


def test_token_decoder_streams_one_label_per_step(conn):
    d = TokenDecoder(n_output_neurons=9, window_steps=3)
    d.configure(conn, np.random.default_rng(0), classes=[0, 1, 2])
    raster = np.zeros((12, conn.n_neurons), dtype=bool)
    groups = d.groups_for_classes()
    for t in range(0, 9, 3):
        raster[t, groups[(t // 3) % 3]] = True
    out = d.decode_stream(raster)
    assert len(out) == 4, "12 steps / 3-step windows"
    assert all(isinstance(x, Decision) for x in out)
    labels = [x.label for x in out[:3]]
    assert labels == [0, 1, 2], f"each window should read the group driven in it, got {labels}"


def test_decoder_state_roundtrip_preserves_class_types(conn):
    """Resumed runs must read out the *same* labels, not their string reprs."""
    d = make_dec(GroupRateDecoder, conn, classes=(0, 1))
    snap = d.state_dict()
    fresh = build_decoder(snap["kind"])
    fresh.configure(conn, np.random.default_rng(0))
    fresh.load_state(snap)
    assert fresh.classes == [0, 1]
    assert all(isinstance(c, int) for c in fresh.classes)
    assert np.array_equal(fresh.output_neurons, d.output_neurons)
    counts = np.zeros(conn.n_neurons)
    counts[d.groups_for_classes()[1]] = 4
    assert fresh.decode(counts).label == d.decode(counts).label == 1


def test_min_spikes_gate_is_configurable(conn):
    strict = make_dec(GroupRateDecoder, conn, min_spikes=5)
    loose = make_dec(GroupRateDecoder, conn, min_spikes=0)
    counts = np.zeros(conn.n_neurons)
    counts[strict.groups_for_classes()[0]] = 1.0  # one spike total, below the gate
    assert strict.decode(counts).label is None
    assert loose.decode(counts).label == 0
