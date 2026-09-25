"""Shared fixtures. Everything here is SYNTHETIC -- no FAFB data is required
to run the suite, by design, so CI on a fresh clone works without the 68 MB
download (see DATA.md).
"""

from __future__ import annotations

import pathlib
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


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at an empty directory.

    Several path-resolution tests assert "nothing was found". On a real Windows
    machine ``~/Downloads`` exists and often *is* the FAFB folder, so without this
    the tests assert a property of the developer's disk rather than of the code.
    """
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOMEDRIVE", str(home.drive or ""))
    monkeypatch.setenv("HOMEPATH", str(home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture(autouse=True)
def _no_test_may_write_reports_into_the_checkout():
    """Guard: the test suite must never create artefacts in the repository root.

    A test that ran `scripts/inspect_fafb.py` without pinning --out once overwrote a
    user's real `fafb_schema_report.md` with a report of an empty tmp directory --
    silently, because the script writes relative to the working directory. Any test
    that produces one of these names in the checkout now fails loudly instead.
    """
    root = Path(__file__).resolve().parents[1]
    watched = ("fafb_schema_report.md", "fafb_schema_report.json",
               "config/fafb_profile.yaml", "config/fafb_profile.json")
    before = {name: (root / name).exists() for name in watched}
    yield
    created = [name for name in watched if (root / name).exists() and not before[name]]
    if created:
        for name in created:
            (root / name).unlink()
        raise AssertionError(
            "a test wrote " + ", ".join(created) + " into the repository root. "
            "Pass --out (and cwd=tmp_path) when invoking scripts/inspect_fafb.py: "
            "these files belong to the user and overwriting them destroys real data."
        )
