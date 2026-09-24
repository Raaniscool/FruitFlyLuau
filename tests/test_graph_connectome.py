"""Sparse connectome representation: correctness + persistence."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from fruitfly.graph.connectome import Connectome, build_connectome


def test_csr_orientation_is_pre_rows_post_cols(tiny_connectome):
    c = tiny_connectome
    # neuron 0 -> 1 and 0 -> 2, so row 0 holds targets {1,2}
    assert set(c.row_slice(0).tolist()) == {1, 2}
    # 1 -> 3 and 2 -> 4, so the only presynaptic cell onto 3 is 1
    assert set(c.col_slice(3).tolist()) == {1}
    assert set(c.col_slice(4).tolist()) == {2}
    assert c.col_slice(0).size == 0  # nothing drives the source neuron


def test_weights_are_the_live_csr_data_array(tiny_connectome):
    c = tiny_connectome
    assert c.weights is c.matrix.data
    c.weights[0] = 42.0
    assert c.matrix.data[0] == 42.0


def test_duplicate_pairs_are_summed():
    ids = np.arange(4, dtype=np.int64)
    pre = np.array([0, 0, 1], dtype=np.int64)   # two rows for 0->1, like two neuropils
    post = np.array([1, 1, 2], dtype=np.int64)
    w = np.array([3.0, 4.0, 5.0])
    c = build_connectome(pre, post, w, ids)
    assert c.n_edges == 2
    assert c.matrix[0, 1] == 7.0
    assert c.matrix[1, 2] == 5.0


def test_autapses_dropped_and_counted():
    ids = np.arange(3, dtype=np.int64)
    c = build_connectome(np.array([0, 1, 1]), np.array([0, 1, 2]), np.array([5.0, 9, 3]), ids)
    assert c.provenance["n_self_connections_dropped"] == 2
    assert c.n_edges == 1
    assert c.matrix[1, 2] == 3.0


def test_invalid_indices_are_dropped_not_silently_kept():
    ids = np.arange(3, dtype=np.int64)
    c = build_connectome(np.array([0, 5, -1]), np.array([1, 1, 2]), np.array([1.0, 2, 3]), ids)
    assert c.n_edges == 1
    assert c.provenance["n_edges_dropped_invalid"] == 2


def test_empty_edge_list_raises():
    with pytest.raises(ValueError):
        build_connectome(np.array([], np.int64), np.array([], np.int64), np.array([]), np.arange(3, dtype=np.int64))


def test_degrees_and_stats(tiny_connectome):
    in_deg, out_deg = tiny_connectome.degrees()
    s = tiny_connectome.stats()
    assert in_deg.sum() == out_deg.sum() == tiny_connectome.n_edges
    assert s["n_neurons"] == tiny_connectome.n_neurons
    assert s["n_edges"] == tiny_connectome.n_edges
    assert s["isolated_neurons"] == tiny_connectome.n_neurons - int(((in_deg + out_deg) > 0).sum())


def test_root_id_lookup_roundtrip(tiny_connectome):
    c = tiny_connectome
    ids = c.root_ids[[3, 3, 0]]
    got = c.indices_of(ids)
    assert got.tolist() == [3, 3, 0]
    assert c.indices_of(np.array([-7]))[0] == -1


def test_subgraph_preserves_only_internal_edges(tiny_connectome):
    keep = np.array([0, 1, 2])
    sub = tiny_connectome.subgraph(keep)
    assert sub.n_neurons == 3
    assert set(sub.root_ids.tolist()) == set(tiny_connectome.root_ids[keep].tolist())
    # 0->1 and 0->2 internal, 1->3 excluded
    assert sub.n_edges == 2


def test_save_load_roundtrip_preserves_learning_state(tmp_path, tiny_connectome):
    tiny_connectome.weights[:] = np.arange(1, tiny_connectome.n_edges + 1, dtype=np.float64)
    p = tiny_connectome.save(tmp_path / "c.npz")
    back = Connectome.load(p)
    assert back.n_edges == tiny_connectome.n_edges
    assert np.array_equal(back.weights, tiny_connectome.weights)
    assert np.array_equal(back.root_ids, tiny_connectome.root_ids)
    assert back.provenance == tiny_connectome.provenance


def test_restore_weights_rejects_shape_mismatch(tiny_connectome):
    with pytest.raises(ValueError):
        tiny_connectome.restore_weights(np.zeros(3))


def test_representation_stays_array_based(tiny_connectome):
    """No per-edge Python objects: the graph is CSR arrays, so 3.7M edges cost RAM, not 3.7M dicts."""
    assert isinstance(tiny_connectome.matrix, sparse.spmatrix)
    assert tiny_connectome.matrix.format == "csr"
    assert isinstance(tiny_connectome.weights, np.ndarray) and tiny_connectome.weights.dtype == np.float64
    assert tiny_connectome.root_ids.dtype == np.int64
    # every edge is described by three array slots and nothing else
    import sys
    per_edge = sys.getsizeof(tiny_connectome.matrix.data) + sys.getsizeof(tiny_connectome.matrix.indices)
    assert per_edge < 1e6, "array storage must be compact"
