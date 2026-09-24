"""Shared fixtures. Everything here is SYNTHETIC -- no FAFB data is required
to run the suite, by design, so CI on a fresh clone works without the 68 MB
download (see DATA.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fruitfly.config import AppConfig  # noqa: E402
from fruitfly.dataset.build_sample import sample_connectome, write_sample_fafb_files  # noqa: E402
from fruitfly.graph.connectome import build_connectome  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """A tiny directory using the real FAFB v783 asset/column names."""
    out = tmp_path_factory.mktemp("fafb_sample")
    counts = write_sample_fafb_files(out, n_neurons=180, seed=1234)
    assert counts, "sample writer produced nothing"
    return out


@pytest.fixture
def tiny_connectome():
    rng = np.random.default_rng(0)
    n = 24
    pre = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    post = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    w = np.ones(pre.size, dtype=np.float64)
    ids = 720575940_000000000 + np.arange(n, dtype=np.int64)
    return build_connectome(pre, post, w, ids, labels=np.array([f"n{i}" for i in range(n)], dtype=object))


@pytest.fixture
def chain_connectome():
    """A strictly linear chain 0 -> 1 -> 2 -> ... -> 9 with weight 1."""
    n = 10
    pre = np.arange(n - 1, dtype=np.int64)
    post = pre + 1
    ids = 720575940_000000000 + np.arange(n, dtype=np.int64)
    return build_connectome(pre, post, np.full(n - 1, 1.0), ids)


@pytest.fixture
def sample_pop():
    conn, meta = sample_connectome(120, rng=np.random.default_rng(7))
    return conn


@pytest.fixture
def base_cfg(tmp_path) -> AppConfig:
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 80
    cfg.data.source = "sample"
    cfg.data.use_cache = False
    cfg.train.episodes = 8
    cfg.train.eval_trials = 8
    cfg.train.eval_before_training = True
    cfg.train.log_dir = str(tmp_path / "runs")
    cfg.train.checkpoint_every = 4
    cfg.graph.counterbalance_probe_items = 8
    cfg.lif.noise = 0.0
    cfg.lif.background_current = 0.0
    return cfg
