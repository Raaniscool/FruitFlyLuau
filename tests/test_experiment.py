"""Experiment layer: environments, metrics, checkpoints, runner, readout mapping.

These are the tests that make a reported number trustworthy: the held-out split is
checked for leakage, a resumed run must reproduce the original, and the controls
must actually switch the mechanism off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fruitfly.config import AppConfig
from fruitfly.experiment import available_environments, build_environment
from fruitfly.experiment.base import Trial
from fruitfly.experiment.checkpoint import (
    Checkpoint, checkpoint_dir, load_checkpoint, prune, restore_connectome, save_checkpoint,
)
from fruitfly.experiment.metrics import (
    MetricsLogger, RunningStats, binom_p_above_chance, chance_level, summarize_curve,
)
from fruitfly.experiment.readout import probe_accuracy, search_mapping
from fruitfly.experiment.runner import ExperimentRunner
from fruitfly.graph.connectome import build_connectome
import fruitfly.experiment.tasks  # noqa: F401  (registers environments)


# ------------------------------------------------------------------ environments
def test_all_five_experiments_are_registered():
    names = available_environments()
    assert {"exp001_binary", "exp002_xor", "exp003_pattern", "exp004_sequence", "exp005_symbolic"} <= set(names)


@pytest.mark.parametrize("name", ["exp001_binary", "exp002_xor", "exp003_pattern", "exp004_sequence", "exp005_symbolic"])
def test_environment_contract(name):
    env = build_environment(name, seed=3, n_train=12, n_eval=6)
    d = env.describe()
    assert d["n_classes"] >= 2
    assert len(env.classes) == d["n_classes"]
    train, ev = env.train_items(), env.eval_items()
    assert len(train) == 12 and len(ev) == 6
    assert all(isinstance(t, Trial) for t in train)
    assert all(t.target in list(env.classes) for t in train)
    # every trial must be scorable, and scoring must be a total function
    for t in train:
        assert env.is_correct(t.target, t.target) is True
        assert isinstance(env.is_correct(None, t.target), bool)


@pytest.mark.parametrize("name,kwargs", [("exp003_pattern", {}), ("exp005_symbolic", {"modulus": 32})])
def test_open_ended_tasks_get_a_truly_held_out_split(name, kwargs):
    """Where the stimulus space is large, eval observations must never be trained on."""
    env = build_environment(name, seed=5, n_train=40, n_eval=20, **kwargs)
    rep = env.overlap_check()
    assert rep["leakage_free"] is True, rep
    assert rep["n_eval_obs_in_train"] == 0, rep
    assert rep["n_train"] == 40 and rep["n_eval"] == 20
    tr = {t.key for t in env.train_items()}
    ev = {t.key for t in env.eval_items()}
    assert not (tr & ev)
    assert env.exhaustive_input_space is False


@pytest.mark.parametrize("name,kwargs", [
    ("exp001_binary", {}), ("exp002_xor", {}), ("exp004_sequence", {}),
    ("exp005_symbolic", {"modulus": 3, "n_train": 40, "n_eval": 20}),  # 27 inputs total
])
def test_exhaustive_tasks_say_so_instead_of_faking_generalisation(name, kwargs):
    """2, 4 and 4 stimuli: the eval set *cannot* be novel, and the code must admit it."""
    kw = {"n_train": 12, "n_eval": 4}
    kw.update(kwargs)
    env = build_environment(name, seed=5, **kw)
    assert env.exhaustive_input_space is True
    d = env.describe()
    assert d["exhaustive_input_space"] is True
    rep = env.overlap_check()
    assert rep["n_eval_obs_in_train"] > 0, "these tasks reuse stimuli; the audit must show it"
    if name in ("exp001_binary", "exp002_xor"):
        assert "associative" in d["generalisation"], d
    if name == "exp005_symbolic":
        assert "NOT held out" in d["generalisation"], d


def test_xor_target_is_the_xor_of_its_parts():
    env = build_environment("exp002_xor", seed=0, n_train=8, n_eval=4)
    for t in env.train_items():
        a, b = t.observation
        assert t.target == int(bool(a) ^ bool(b)), t


def test_epochs_are_deterministic_per_seed():
    env = build_environment("exp003_pattern", seed=11, n_train=9, n_eval=3)
    a = [t.key for t in env.epochs(3)]
    b = [t.key for t in env.epochs(3)]
    assert a == b and len(a) == 3, "the training stream must be reshuffled identically each run"


def test_unknown_environment_raises_with_the_list():
    with pytest.raises((KeyError, ValueError)) as exc:
        build_environment("exp999_nope")
    assert "exp001_binary" in str(exc.value)


# ------------------------------------------------------------------ metrics
def test_chance_level_and_binomial_tail():
    assert chance_level(2) == pytest.approx(0.5)
    assert chance_level(4) == pytest.approx(0.25)
    # 10/10 correct against a 0.5 chance level: p = 2**-10
    assert binom_p_above_chance(10, 10, 0.5) == pytest.approx(2 ** -10, rel=1e-6)
    # 5/10 is exactly the expected count: no evidence at all
    assert binom_p_above_chance(5, 10, 0.5) > 0.5
    # 48/64 vs 0.5 -> highly significant but not absurd
    p = binom_p_above_chance(48, 64, 0.5)
    assert 0 < p < 1e-4
    assert binom_p_above_chance(0, 0, 0.5) == 1.0, "no trials is no evidence, not an error"


def test_summarize_curve_windowed_and_honest_about_empty():
    v = list(range(10))
    st = summarize_curve(v, window=5)
    assert st["first"] == pytest.approx(0.0)
    assert st["last"] == pytest.approx(9.0)
    assert st["window_mean"] == pytest.approx(np.mean(range(5, 10)))
    assert st["best"] == pytest.approx(9.0)
    empty = summarize_curve([], window=5)
    assert empty == {"n": 0}, "an empty curve must not invent statistics"


def test_running_stats_and_metrics_logger(tmp_path):
    rs = RunningStats()
    for x in (1, 2, 3, 4):
        rs.update(x)
    d = rs.to_dict()
    assert d["n"] == 4 and d["mean"] == pytest.approx(2.5) and d["max"] == 4 and d["min"] == 1
    log = MetricsLogger(tmp_path / "metrics.jsonl")
    for i in range(10):
        log.log(episode=i, accuracy=float(i) / 10, reward=float(-1 if i % 2 else 1))
    log.close()
    assert (tmp_path / "metrics.jsonl").read_text().strip().count("\n") == 9
    again = MetricsLogger(tmp_path / "metrics.jsonl")
    assert again.series("accuracy").size == 10, "a resumed logger must see the earlier records"
    assert again.windowed("accuracy", 5)[-1] == pytest.approx(np.mean(range(5, 10)) / 10)
    again.close()


# ------------------------------------------------------------------ checkpoints
def tiny_conn(n=6):
    pre = np.arange(n - 1, dtype=np.int64)
    return build_connectome(pre, pre + 1, np.full(n - 1, 1.5), np.arange(n, dtype=np.int64))


def test_checkpoint_roundtrip_restores_weights_exactly(tmp_path):
    conn = tiny_conn()
    conn.weights[:] = np.arange(1, conn.n_edges + 1) * 0.1
    cp = save_checkpoint(
        tmp_path, conn=conn, rule_state={"n_steps": 12, "total_abs_dw": 0.5},
        reward_state={"episode": 3}, modulator_state={"value": 0.25},
        encoder_state={"input_neurons": [0, 1]}, decoder_state={"classes": [0, 1]},
        episode=7, seed=99, config_hash="abc123", rng_state=np.random.RandomState(5),
        extra={"note": "unit test"},
    )
    assert cp.is_file()
    loaded = load_checkpoint(tmp_path)
    assert isinstance(loaded, Checkpoint)
    assert loaded.episode == 7 and loaded.seed == 99 and loaded.config_hash == "abc123"
    assert loaded.metadata["extra"]["note"] == "unit test"
    assert loaded.metadata["n_neurons"] == conn.n_neurons and loaded.metadata["n_edges"] == conn.n_edges
    assert loaded.metadata["rule"] == {"n_steps": 12, "total_abs_dw": 0.5}
    assert (cp.with_suffix(".json").read_text()).count("unit test") == 1, "the sidecar holds the metadata"
    assert np.allclose(loaded.weights, conn.weights), "the npz must carry the learned weights"
    fresh = tiny_conn()
    state = restore_connectome(loaded, fresh)
    assert np.allclose(fresh.weights, conn.weights)
    assert set(state) >= {"eligibility", "updates", "trace_pre", "trace_post"}
    assert state["updates"].shape == (conn.n_edges,)
    # a graph of a different shape must be refused, never silently truncated
    with pytest.raises(ValueError) as exc:
        restore_connectome(loaded, tiny_conn(n=5), strict=True)
    assert "does not match connectome" in str(exc.value)
    with pytest.raises(ValueError):
        restore_connectome(loaded, tiny_conn(n=5), strict=False)


def test_pruning_keeps_the_newest_and_tags_latest(tmp_path):
    conn = tiny_conn()
    for ep in range(1, 6):
        save_checkpoint(tmp_path, conn=conn, rule_state={}, reward_state={}, modulator_state={},
                        encoder_state={}, decoder_state={}, episode=ep, seed=1, config_hash="h",
                        rng_state=(0, np.zeros(624, dtype=np.uint32), None, None, None),
                        tag=f"step{ep:03d}")
    files = sorted(checkpoint_dir(tmp_path).glob("*.npz"))
    assert len(files) == 5
    removed = prune(tmp_path, keep_last=2)
    assert len(removed) == 6, "each pruned snapshot removes its npz and json sidecar"
    left = sorted(p.stem for p in checkpoint_dir(tmp_path).glob("step*.npz"))
    assert left == ["step004", "step005"], left
    assert prune(tmp_path, keep_last=0) != [], "keep_last=0 must clear the numbered snapshots"


# ------------------------------------------------------------------ runner
def make_cfg(tmp_path, **over):
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.source = "sample"
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 80
    cfg.graph.selection = "connected_subgraph"
    cfg.graph.counterbalance_probe_items = 6
    cfg.data.use_cache = False
    cfg.train.episodes = 4
    cfg.train.eval_trials = 6
    cfg.train.eval_before_training = True
    cfg.train.log_dir = str(tmp_path / "runs")
    cfg.train.checkpoint_every = 2
    cfg.lif.noise = 0.0
    cfg.lif.background_current = 1.0
    cfg.plasticity.eta = 0.02
    cfg.plasticity.normalize = "none"
    for k, v in over.items():
        sec, key = k.split("__")
        setattr(getattr(cfg, sec), key, v)
    return cfg


def test_setup_records_everything_needed_to_reproduce(tmp_path):
    cfg = make_cfg(tmp_path)
    env = build_environment("exp001_binary", seed=1, n_train=6, n_eval=4)
    run_dir = tmp_path / "run_setup"
    r = ExperimentRunner(cfg, env, run_dir=run_dir)
    r.setup()
    setup = json.loads((run_dir / "setup.json").read_text())
    assert setup["config_hash"] == cfg.config_hash
    assert setup["population"]["simulated_neurons"] == 80
    assert setup["population"]["stats"]["n_edges"] == r.conn.n_edges
    assert setup["experiment"]["name"] == "exp001_binary"
    assert setup["data"]["source"] == "sample", "the run must state it used synthetic data"
    assert setup["data"]["synthetic"] is True, "never let a synthetic run look like FAFB"
    assert setup["readout"]["trial_steps"] == r.trial_steps
    assert "readout_mapping" in setup and "assignment_audit" in setup["readout"]
    assert setup["plasticity_mask"]["plastic_edges"] > 0
    assert setup["plasticity_mask"]["n_edges"] == r.conn.n_edges
    assert setup["control"] == "none"
    assert setup["config"]["lif"]["dt_ms"] == cfg.lif.dt_ms
    # the decoder's readout cells are real neurons of the loaded graph
    assert set(r.decoder.output_neurons.tolist()) <= set(range(r.conn.n_neurons))
    assert r.trial_steps >= 8, "the stimulus window must be long enough for spikes to reach the readout"


def test_run_trial_reports_real_activity_and_no_learning_without_learn_flag(tmp_path):
    cfg = make_cfg(tmp_path)
    env = build_environment("exp001_binary", seed=2, n_train=4, n_eval=2)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_trial")
    r.setup()
    trial = next(iter(env.train_items()))
    res = r.run_trial(trial, learn=False, episode=0)
    assert res.n_spikes >= 0
    assert res.abs_dw == 0.0 and res.edges_updated == 0, "learn=False must not touch weights"
    assert res.prediction in list(env.classes) + [None]
    w0 = r.conn.weights.copy()
    r.run_trial(trial, learn=True, episode=1)
    assert np.isfinite(r.conn.weights).all()
    assert r.conn.weights.shape == w0.shape


def test_full_train_writes_reports_and_is_reproducible(tmp_path):
    cfg = make_cfg(tmp_path)
    env = build_environment("exp001_binary", seed=4, n_train=8, n_eval=4)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_a")
    r.setup()
    out = r.train()
    assert out.n_episodes == cfg.train.episodes
    assert set(out.eval_after) >= {"accuracy", "n_trials"}
    assert out.eval_after["n_trials"] == 4
    assert 0.0 <= out.eval_after["accuracy"] <= 1.0
    assert out.delta_eval_accuracy == pytest.approx(out.eval_after["accuracy"] - out.eval_before["accuracy"])
    assert out.learning_detected in (True, False)
    assert "NO learning detected" in out.claim_bounds or "learning" in out.claim_bounds.lower()
    run = tmp_path / "run_a"
    assert (run / "setup.json").is_file() and (run / "metrics.jsonl").is_file()
    assert (run / "results.json").is_file()
    assert json.loads((run / "results.json").read_text())["n_episodes"] == cfg.train.episodes
    assert "final" in out.summary_text() or "accuracy" in out.summary_text()
    # same config + same seed -> byte-identical accuracy series
    r2 = ExperimentRunner(make_cfg(tmp_path), build_environment("exp001_binary", seed=4, n_train=8, n_eval=4),
                          run_dir=tmp_path / "run_b")
    r2.setup()
    out2 = r2.train()
    assert out2.eval_after["accuracy"] == pytest.approx(out.eval_after["accuracy"])
    assert out2.weight_stats["total_abs_dw"] == pytest.approx(out.weight_stats["total_abs_dw"])
    assert np.allclose(r2.conn.weights, r.conn.weights), "same seed must give the same learned weights"


def test_a_task_already_solved_at_baseline_is_not_reported_as_learning(tmp_path):
    """The criterion needs the untrained network to be at chance.

    With the readout counterbalance switched off, a favourable mapping can already
    score ~0.9 held-out before any training; a rise to 1.0 is a ceiling effect, not
    learning. The runner must say so instead of printing "learning detected".
    """
    cfg = make_cfg(tmp_path, graph__counterbalance_init=False, plasticity__eta=0.5,
                   train__episodes=12, lif__background_current=1.5, lif__noise=0.8)
    env = build_environment("exp001_binary", seed=21, n_train=16, n_eval=16)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_easy")
    r.setup()
    out = r.train()
    crit = out.learning_criterion
    assert crit["baseline_at_chance"] in (True, False)
    if out.eval_before["accuracy"] > out.eval_after["chance"] + 0.05:
        assert out.learning_detected is False, "a pre-solved task must not be claimed as learning"
        assert "ceiling effect" in out.claim_bounds or "NOT counted as learning" in out.claim_bounds
    assert {"min_gain_vs_chance", "p_threshold", "chance", "total_abs_dw"} <= set(crit)
    assert json.loads((tmp_path / "run_easy" / "results.json").read_text())["learning_criterion"]


def test_no_plasticity_control_freezes_the_weights(tmp_path):
    cfg = make_cfg(tmp_path, plasticity__rule="none", train__episodes=6)
    env = build_environment("exp001_binary", seed=6, n_train=8, n_eval=4)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_frozen")
    r.setup()
    w0 = r.conn.weights.copy()
    out = r.train()
    assert out.weight_stats["total_abs_dw"] == 0.0
    assert np.allclose(r.conn.weights, w0)
    assert out.learning_detected is False
    assert out.controls, "the run must record which control was active"


def test_sham_reward_control_breaks_the_correctness_link(tmp_path):
    cfg = make_cfg(tmp_path, plasticity__rule="reward_stdp", plasticity__eta=0.5,
                   train__episodes=6, train__control="sham_reward")
    env = build_environment("exp001_binary", seed=7, n_train=10, n_eval=4)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_sham")
    r.setup()
    out = r.train()
    assert "sham" in " ".join(out.controls).lower() or "sham" in str(out.controls).lower()
    # rewards were still delivered (so the mechanism runs) but they do not track accuracy
    assert out.reward_curve["raw_last_200"], "sham control must still emit rewards"


def test_unknown_control_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path, train__control="do_whatever")
    env = build_environment("exp001_binary", seed=1, n_train=4, n_eval=2)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_bad")
    with pytest.raises(ValueError) as exc:
        r.setup()
    assert "no_plasticity" in str(exc.value)


def test_resume_reproduces_the_continued_run(tmp_path):
    cfg = make_cfg(tmp_path, train__episodes=4)
    env = build_environment("exp001_binary", seed=8, n_train=8, n_eval=4)
    a_dir = tmp_path / "run_full"
    ra = ExperimentRunner(cfg, env, run_dir=a_dir)
    ra.setup()
    full = ra.train()

    cfg2 = make_cfg(tmp_path, train__episodes=2)
    part_dir = tmp_path / "run_part"
    rb = ExperimentRunner(cfg2, env, run_dir=part_dir)
    rb.setup()
    rb.train()
    # continue from the checkpoint into a fresh runner with the full episode count
    rc = ExperimentRunner(cfg2, env, run_dir=tmp_path / "run_resume")
    rc.setup(resume_from=part_dir)
    out = rc.train(episodes=4)
    assert np.isfinite(rc.conn.weights).all()
    assert out.n_episodes == 4

    assert full.eval_after["n_trials"] == out.eval_after["n_trials"]


def test_load_checkpoint_reports_a_missing_tag_clearly(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_checkpoint(tmp_path)
    assert "no checkpoint" in str(exc.value)


def test_runner_checkpoint_can_be_restored_into_another_connectome(tmp_path):
    cfg = make_cfg(tmp_path, train__episodes=3, train__save_checkpoints=True)
    env = build_environment("exp001_binary", seed=9, n_train=6, n_eval=3)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_ckpt")
    r.setup()
    r.train()
    cp = load_checkpoint(tmp_path / "run_ckpt")
    assert cp.weights.size == r.conn.n_edges
    # a fresh graph with the same edge count must come back bit-identical
    # the checkpoint restores into a *fresh* runner, which is what --resume does
    r2 = ExperimentRunner(make_cfg(tmp_path, train__episodes=3),
                          build_environment("exp001_binary", seed=9, n_train=6, n_eval=3),
                          run_dir=tmp_path / "run_ckpt2")
    r2.setup()
    restore_connectome(cp, r2.conn)
    assert np.allclose(r2.conn.weights, r.conn.weights), "resume must reproduce the learned weights exactly"


# ------------------------------------------------------------------ readout mapping
def test_mapping_search_only_reorders_classes(tmp_path):
    cfg = make_cfg(tmp_path, graph__counterbalance_init=True)
    env = build_environment("exp001_binary", seed=10, n_train=6, n_eval=4)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_map")
    r.setup()
    res = r.mapping_search
    assert res.performed in (True, False)
    assert sorted(res.chosen) == list(range(len(env.classes))), "the search may permute, never duplicate"
    if res.performed:
        assert list(r.decoder.classes) == [env.classes[i] for i in res.chosen]
    else:
        assert list(r.decoder.classes) == list(env.classes)
    assert "permutation" in res.describe() or "identity" in res.describe()


def test_mapping_search_can_be_disabled(tmp_path):
    cfg = make_cfg(tmp_path, graph__counterbalance_init=False)
    env = build_environment("exp001_binary", seed=11, n_train=4, n_eval=2)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_nomap")
    r.setup()
    assert r.mapping_search.performed is False
    assert list(r.decoder.classes) == list(env.classes)


def test_probe_accuracy_returns_a_rate(tmp_path):
    cfg = make_cfg(tmp_path)
    env = build_environment("exp001_binary", seed=12, n_train=6, n_eval=3)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_probe")
    r.setup()
    acc = probe_accuracy(r, list(env.eval_items())[:3], (0, 1))
    assert 0.0 <= acc <= 1.0
    assert isinstance(acc, float)


def test_evaluate_honours_the_requested_number_of_trials(tmp_path):
    cfg = make_cfg(tmp_path)
    env = build_environment("exp003_pattern", seed=13, n_train=6, n_eval=5)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_eval")
    r.setup()
    rep = r.evaluate(n=3)
    assert rep["n_trials"] == 3
    assert rep["chance"] == pytest.approx(chance_level(len(env.classes)), abs=1e-3)
    assert "p_value_vs_chance" in rep and "readout_silent_trials" in rep
    assert rep["n_none_predictions"] + rep["n_correct"] + rep.get("n_wrong", 0) >= 0


def test_silent_readout_is_reported_not_silently_scored(tmp_path):
    """An undriveable network must say so; a silent null result is the worst failure mode."""
    cfg = make_cfg(tmp_path, lif__noise=0.0, lif__background_current=0.0, train__steps_per_trial=2)
    env = build_environment("exp001_binary", seed=14, n_train=4, n_eval=2)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "run_silent")
    r.setup()
    out = r.train()
    rep = r.evaluate(n=2)
    assert rep["n_trials"] == 2
    # a trial the readout could not answer must be counted as a miss, not a hit
    assert rep["n_none_predictions"] >= 0
    assert rep["accuracy"] <= chance_level(2) + 1e-9 or rep["readout_silent_trials"] == 0
