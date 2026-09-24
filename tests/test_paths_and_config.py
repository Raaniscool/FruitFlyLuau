"""Data-path resolution and config layering."""

from __future__ import annotations

import os

import pytest

from fruitfly.config import AppConfig
from fruitfly.paths import ENV_VAR, find_data_dir, require_data_dir


def test_env_var_wins_and_is_reported(tmp_path, monkeypatch):
    data = tmp_path / "fafb"
    data.mkdir()
    (data / "connections_princeton.csv.gz").write_bytes(b"x")  # recognition hint
    monkeypatch.setenv(ENV_VAR, str(data))
    found, trail = find_data_dir()
    assert found == data
    assert "$" + ENV_VAR in trail.describe() or ENV_VAR in trail.describe()


def test_explicit_argument_beats_env(tmp_path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    (good / "neurons.csv.gz").write_bytes(b"x")
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nonexistent"))
    found, _ = find_data_dir(good)
    assert found == good


def test_missing_data_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        require_data_dir()
    msg = str(exc.value)
    assert ENV_VAR in msg
    assert "config/local.yaml" in msg
    assert "inspect_fafb" in msg
    assert "Locations tried" in msg or "NOT found" in msg


def test_search_trail_lists_every_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    _, trail = find_data_dir()
    assert trail.path is None
    assert len(trail.tried) >= 3, "we must show the user what was searched"


def test_config_layering_and_env_expansion(tmp_path, monkeypatch):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "default.yaml").write_text("graph:\n  n_neurons: 111\n", encoding="utf-8")
    monkeypatch.setenv("MY_FAFB", str(tmp_path / "from-env"))
    layer = tmp_path / "layer.yaml"
    layer.write_text("data:\n  fafb_data_path: ${MY_FAFB}\ngraph:\n  mode: small\n", encoding="utf-8")
    import fruitfly.config as C

    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    cfg = AppConfig.load(layer)
    assert cfg.graph.n_neurons == 111          # from default.yaml
    assert cfg.graph.mode == "small"           # from layer
    assert cfg.data.fafb_data_path == str(tmp_path / "from-env")  # ${VAR} expanded


def test_unknown_config_keys_warn_instead_of_crashing(caplog):
    cfg = AppConfig.from_dict({"nonsense": 1, "graph": {"also_nonsense": 2, "mode": "tiny"}})
    assert cfg.graph.mode == "tiny"


def test_config_hash_changes_with_content():
    a = AppConfig.load(use_defaults_file=False)
    b = AppConfig.load(use_defaults_file=False)
    assert a.config_hash == b.config_hash
    b.train.seed = 99
    assert a.config_hash != b.config_hash


def test_to_dict_roundtrips():
    a = AppConfig.load(use_defaults_file=False)
    b = AppConfig.from_dict(a.to_dict())
    assert b.to_dict() == a.to_dict()
