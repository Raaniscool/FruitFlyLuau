"""CLI smoke tests and figure generation.

Everything runs through ``main(argv)`` in-process (fast, no shell quoting
problems) except ``scripts/inspect_fafb.py``, which is deliberately executed as a
subprocess because it must work with nothing but the standard library on a plain
Windows machine -- the user runs it against their own FAFB download.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fruitfly.cli.__main__ import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*argv: str) -> int:
    return main(list(argv))


@pytest.fixture(scope="module")
def sample_fafb(tmp_path_factory):
    from fruitfly.dataset.build_sample import write_sample_fafb_files

    out = tmp_path_factory.mktemp("cli_sample")
    write_sample_fafb_files(out, n_neurons=160, seed=4242)
    return out


# ------------------------------------------------------------------ parser
def test_parser_exposes_every_documented_command():
    ap = build_parser()
    subs = ap._subparsers._group_actions[0].choices  # noqa: SLF001 - argparse has no public API
    for cmd in ("doctor", "sample", "inspect", "stats", "sim", "run", "rules"):
        assert cmd in subs, cmd


def test_shared_flags_are_not_duplicated():
    """``--neurons`` lives on the shared parent parser exactly once per subcommand."""
    ap = build_parser()
    for name, sp in ap._subparsers._group_actions[0].choices.items():  # noqa: SLF001
        opts = [a.option_strings for a in sp._actions]  # noqa: SLF001
        flat = [o for group in opts for o in group]
        assert len(flat) == len(set(flat)), f"{name} defines a duplicate flag: {flat}"


def test_set_override_is_parsed_as_json_when_possible():
    from fruitfly.cli.__main__ import _cfg
    from fruitfly.config import AppConfig

    ns = build_parser().parse_args(["stats", "--set", "plasticity.eta=0.025",
                                    "--set", "graph.counterbalance_init=false",
                                    "--set", "experiment=hello",
                                    "--set", "train.episodes=3"])
    cfg = _cfg(ns)
    assert cfg.plasticity.eta == pytest.approx(0.025)
    assert cfg.graph.counterbalance_init is False
    assert cfg.experiment == "hello", "top-level --set keys must work too"
    assert cfg.train.episodes == 3
    # other keys in a touched section keep their defaults
    assert cfg.plasticity.rule == AppConfig.load(use_defaults_file=False).plasticity.rule


def test_bad_set_override_is_a_clean_error():
    ns = build_parser().parse_args(["stats", "--set", "no_equals_sign"])
    with pytest.raises(SystemExit) as exc:
        _cfg_call(ns)
    assert "section.key=value" in str(exc.value)


def _cfg_call(ns):
    from fruitfly.cli.__main__ import _cfg

    return _cfg(ns)


# ------------------------------------------------------------------ commands
def test_rules_lists_every_registry(capsys):
    assert run_cli("rules") == 0
    out = capsys.readouterr().out
    for token in ("reward_stdp", "stdp", "frozen", "group_rate", "binary", "upstream_of", "connected_subgraph"):
        assert token in out, token


def test_doctor_reports_missing_data_without_crashing(capsys, tmp_path, monkeypatch, isolated_home):
    monkeypatch.delenv("FAFB_DATA_PATH", raising=False)
    monkeypatch.setattr("fruitfly.paths.repo_root", lambda: tmp_path)
    assert run_cli("doctor", "--data-dir", str(tmp_path / "nope_missing")) == 0
    out = capsys.readouterr().out
    assert "FAFB_DATA_PATH" in out
    assert "python -m fruitfly sample" in out, "the error must point at the offline path"
    assert "numpy" in out


def test_sample_writes_a_fafb_shaped_directory(capsys, tmp_path, sample_fafb):
    out_dir = tmp_path / "cli_samples"
    assert run_cli("sample", "--out", str(out_dir), "--neurons", "120") == 0
    names = {p.name for p in out_dir.iterdir()}
    assert "connections_princeton.csv.gz" in names
    assert "neurons.csv.gz" in names
    assert "rows" in capsys.readouterr().out


def test_stats_reads_the_sample_and_labels_its_provenance(capsys, sample_fafb, tmp_path):
    j = tmp_path / "stats.json"
    rc = run_cli("stats", "--data-dir", str(sample_fafb), "--mode", "tiny", "--neurons", "60",
                 "--set", "data.use_cache=false", "--json", str(j))
    assert rc == 0
    cap = capsys.readouterr().out
    assert "SIMULATED" in cap and "60 neurons" in cap
    assert "SUBSET" in cap, "a sampled subgraph must never be described as the whole brain"
    payload = json.loads(j.read_text())
    assert payload["dataset"]["n_rows"] > 0
    assert payload["population"]["n_edges"] > 0


def test_inspect_matches_the_standalone_script(capsys, sample_fafb, tmp_path):
    j = tmp_path / "inv.json"
    rc = run_cli("inspect", "--data-dir", str(sample_fafb), "--json", str(j))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Connections (Filtered)" in out
    payload = json.loads(j.read_text())
    assert payload["matched"], "the JSON export must carry the asset matches"
    assert "connections_filtered" in payload["matched"]
    assert payload["data_dir"] == str(sample_fafb)


def test_inspect_without_data_explains_itself(capsys, tmp_path):
    missing = tmp_path / "does_not_exist"
    rc = run_cli("inspect", "--data-dir", str(missing))
    assert rc == 3, "a missing directory is a clean exit code, not a traceback"
    err = capsys.readouterr().err
    assert str(missing) in err, err


def test_sim_reports_measured_activity(capsys, sample_fafb):
    rc = run_cli("sim", "--data-dir", str(sample_fafb), "--mode", "tiny", "--neurons", "60",
                 "--steps", "30", "--set", "data.use_cache=false")
    assert rc == 0
    out = capsys.readouterr().out
    assert "total spikes" in out
    assert "CV of ISI" in out


def test_run_executes_an_experiment_and_writes_reports(capsys, sample_fafb, tmp_path):
    run_dir = tmp_path / "cli_run"
    rc = run_cli(
        "run", "-e", "exp001_binary", "--data-dir", str(sample_fafb), "--mode", "tiny",
        "--neurons", "70", "--episodes", "4", "--n-train", "6", "--n-eval", "4",
        "--set", "data.use_cache=false", "--set", "train.save_checkpoints=false",
        "--run-dir", str(run_dir),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXPERIMENT exp001_binary" in out
    assert "learning detected by our criterion" in out
    assert "claim bounds" in out
    assert (run_dir / "setup.json").is_file() and (run_dir / "results.json").is_file()
    assert "total |dw|" in out


def test_run_with_unknown_experiment_fails_cleanly(capsys):
    rc = run_cli("run", "-e", "exp404_missing", "--mode", "sample", "--neurons", "40")
    assert rc == 1
    assert "exp001_binary" in capsys.readouterr().err


# ------------------------------------------------------------------ viz
def test_figures_are_written(tmp_path, sample_fafb):
    pytest.importorskip("matplotlib")
    import numpy as np

    from fruitfly.viz.plots import plot_activity, plot_connectivity_sample, plot_raster

    rates = np.abs(np.sin(np.arange(60))) + 0.1
    counts = np.arange(30, dtype=np.float64)
    a = plot_activity(rates, counts, out=str(tmp_path / "a.png"))
    assert a is not None and Path(a).stat().st_size > 1000
    b = plot_raster(np.random.default_rng(0).random((40, 24)) > 0.85, out=str(tmp_path / "r.png"))
    assert b is not None
    from fruitfly.graph.build import build_population
    from fruitfly.config import AppConfig

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.fafb_data_path = str(sample_fafb)
    cfg.graph.mode = "tiny"
    cfg.graph.n_neurons = 60
    cfg.data.use_cache = False
    pop = build_population(cfg)
    c = plot_connectivity_sample(pop.connectome, out=str(tmp_path / "c.png"))
    assert c is not None


def test_plot_learning_handles_a_real_runner(tmp_path, sample_fafb):
    pytest.importorskip("matplotlib")
    from fruitfly.config import AppConfig
    from fruitfly.experiment import ExperimentRunner, build_environment
    from fruitfly.viz.plots import plot_learning

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.data.fafb_data_path = str(sample_fafb)
    cfg.graph.mode = "tiny"
    cfg.graph.n_neurons = 60
    cfg.data.use_cache = False
    cfg.train.episodes = 6
    cfg.train.eval_trials = 4
    cfg.train.save_checkpoints = False
    cfg.lif.background_current = 1.0
    env = build_environment("exp001_binary", seed=1, n_train=8, n_eval=4)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "viz_run")
    r.setup()
    r.train()
    paths = plot_learning(r, out_dir=str(tmp_path / "figs"))
    assert paths, "plot_learning must return the files it wrote"
    assert (tmp_path / "figs" / "learning.png").stat().st_size > 5000
    for p in paths:
        assert Path(p).is_file(), p


def test_plot_learning_survives_no_data(tmp_path):
    """A figure with nothing logged must be produced anyway -- reports depend on it."""
    pytest.importorskip("matplotlib")
    from fruitfly.config import AppConfig
    from fruitfly.experiment import ExperimentRunner, build_environment
    from fruitfly.viz.plots import plot_learning

    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "sample"
    cfg.graph.n_neurons = 40
    cfg.data.source = "sample"
    cfg.data.use_cache = False
    cfg.train.episodes = 2
    cfg.train.eval_trials = 2
    cfg.train.save_checkpoints = False
    env = build_environment("exp001_binary", seed=0, n_train=4, n_eval=2)
    r = ExperimentRunner(cfg, env, run_dir=tmp_path / "tiny")
    r.setup()
    paths = plot_learning(r, out_dir=str(tmp_path / "f2"))  # no training yet: empty series
    assert isinstance(paths, list)


# --------------------------------------------------- standalone inspect script
def test_standalone_inspection_script_runs_on_a_plain_python(tmp_path, sample_fafb):
    """The user runs this on Windows with only the standard library installed."""
    script = ROOT / "scripts" / "inspect_fafb.py"
    assert script.is_file()
    # Only *module-level* imports matter: yaml is imported lazily inside an
    # optional feature, which a plain-python Windows user never triggers.
    import ast

    tree = ast.parse(script.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    third_party = {"numpy", "pandas", "scipy", "yaml", "matplotlib", "networkx", "pyarrow", "fitz"}
    assert not (top & third_party), f"the standalone script must import only the stdlib: {top & third_party}"
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [sys.executable, str(script), "--dir", str(sample_fafb), "--out", str(out), "--count-rows"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    text = out.read_text()
    assert "connections_princeton.csv.gz" in text
    assert "pre_root_id" in text, "the report must name the columns it actually found"
    assert "SYNTHETIC" in text.upper() or "sample" in text.lower()


def test_standalone_inspection_script_explains_an_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    # --out AND cwd are both pinned into tmp_path on purpose: with the default
    # --out the script writes fafb_schema_report.{md,json} relative to the working
    # directory, so an unpinned test run inside a checkout DESTROYS the user's real
    # report. That happened once; see the guard fixture in conftest.py.
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_fafb.py"),
                           "--dir", str(empty), "--out", str(tmp_path / "empty_report")],
                          capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert proc.returncode in (0, 2)
    combined = (proc.stdout + proc.stderr).lower()
    assert "no files" in combined or "not found" in combined or "empty" in combined


def test_write_profile_round_trips_into_the_loader(tmp_path, sample_fafb, monkeypatch):
    """``inspect_fafb.py --write-profile`` must produce a profile the loader honours.

    This is the whole point of the profile: a scan of the user's real bytes becomes a
    column-name map that ``fruitfly/dataset`` reads, with no code edit. So the test
    closes the loop -- run the script, load the emitted file, check the mapping.
    """
    cwd = tmp_path / "proj"
    (cwd / "config").mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_fafb.py"),
         "--dir", str(sample_fafb), "--out", str(tmp_path / "rep"), "--write-profile"],
        capture_output=True, text=True, timeout=300, cwd=str(cwd),
    )
    assert proc.returncode == 0, proc.stderr
    prof = cwd / "config" / "fafb_profile.yaml"
    assert prof.is_file(), "profile should be written relative to the working directory"

    from fruitfly.dataset.loader import load_connections
    from fruitfly.dataset.schema import CONNECTION_FIELDS, load_profile, specs_for

    monkeypatch.setenv("FAFB_PROFILE", str(prof))
    profile = load_profile()
    entry = profile["connections_filtered"]
    assert entry["filename"] == "connections_princeton.csv.gz"
    assert entry["columns"]["pre_root_id"] == "pre_root_id"
    specs = specs_for(profile, "connections_filtered", CONNECTION_FIELDS)
    assert next(sp for sp in specs if sp.name == "weight").aliases[0] == "syn_count"
    # a profile describing a schema-conformant file must not change the outcome
    table = load_connections(sample_fafb)
    assert table.pre.size > 0


def test_inspection_report_has_every_required_section(tmp_path, sample_fafb):
    """The audit report must answer all 18 questions, or say it cannot.

    The real FAFB download lives on the user's machine, not in CI, so the contract we
    can actually test is the *shape* of the report: the section headings exist, the
    measured facts are present for a schema-correct directory, and nothing is invented.
    """
    out = tmp_path / "rep"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_fafb.py"), "--dir", str(sample_fafb),
         "--out", str(out), "--deep", "--count-rows"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.with_suffix(".md").read_text()
    for n, title in enumerate([
        "Dataset inventory", "File sizes", "File formats", "Row counts",
        "Column names and inferred types", "Example rows", "Unique-ID statistics",
        "Missing-value statistics", "Cross-file ID compatibility", "Connection-table structure",
        "Synapse-table structure", "Cell-type / classification structure",
        "Visual-neuron annotation structure", "Neurotransmitter structure",
        "Potential schema hazards", "Recommended canonical internal schema",
        "Files to use for the first real experiment", "Files to keep optional / lazy-loaded",
    ], start=1):
        assert f"## {n}. {title}" in text, f"section {n} ({title}) missing from the report"

    report = json.loads(out.with_suffix(".json").read_text())
    conn = next(f for f in report["files"] if f.get("asset") == "connections_filtered")
    assert conn["row_count_is_exact"] and conn["n_rows_including_header"] > 0
    assert conn["n_unique_pairs"] <= conn["n_rows_including_header"]
    assert conn["n_unique_pre"] > 0 and conn["n_unique_post"] > 0
    # the >2^53 id hazard is a real property of FlyWire root ids and must be flagged
    assert any("2^53" in h for h in report["hazards"])
    # cross-file id compatibility was actually computed, not asserted
    assert any(r["asset"] == "nt_predictions" and r["n_shared_with_reference"] > 0
               for r in report["cross_file_ids"])


def test_inspection_report_says_unknown_rather_than_guessing(tmp_path, sample_fafb):
    """Without --count-rows the row count is unmeasured, so it must be reported as unknown."""
    out = tmp_path / "shallow"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_fafb.py"), "--dir", str(sample_fafb),
         "--out", str(out), "--sample-rows", "5"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    text = out.with_suffix(".md").read_text()
    assert "UNKNOWN -- requires further investigation" in text
    report = json.loads(out.with_suffix(".json").read_text())
    conn = next(f for f in report["files"] if f.get("asset") == "connections_filtered")
    assert conn["row_count_is_exact"] is False
    assert conn["n_rows_profiled"] == 5


def test_inspection_script_never_opens_non_table_files(tmp_path, sample_fafb):
    """A real Downloads folder is mostly installers. The audit must not parse them.

    Reproduces what happened on the user's machine: `--dir ~/Downloads` where the FAFB
    files sit next to a 1.5 GB .exe. Opening those as CSV would be slow and useless,
    and with --count-rows would stream the whole binary through the csv module.
    """
    junk = tmp_path / "downloads"
    junk.mkdir()
    for f in sample_fafb.glob("*.gz"):
        (junk / f.name).write_bytes(f.read_bytes())
    (junk / "BigInstaller.exe").write_bytes(b"MZ\x00\x00" + b"\x00\xff" * 500_000)
    (junk / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 1000)
    (junk / "script.lua").write_text("print('hi')\n", encoding="utf-8")

    out = tmp_path / "rep"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_fafb.py"), "--dir", str(junk),
         "--out", str(out), "--count-rows"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.with_suffix(".json").read_text())
    for name in ("BigInstaller.exe", "photo.jpg", "script.lua"):
        entry = next(f for f in report["files"] if f["file"] == name)
        assert not entry["header"], f"{name} was parsed as a table"
        assert not entry["columns"]
    conn = next(f for f in report["files"] if f.get("asset") == "connections_filtered")
    assert conn["header"], "the real assets must still be profiled alongside the junk"
    assert "non-table files" in proc.stdout


def test_doctor_only_calls_the_two_core_assets_required(capsys, sample_fafb, monkeypatch):
    """`visual_types`/`coordinates` are enrichment, not requirements.

    The user's download has connections + neurons and can build a graph; the doctor
    used to print REQUIRED next to assets the pipeline runs without.
    """
    from fruitfly.dataset.assets import CORE_ASSETS
    from fruitfly.dataset.discover import discover

    assert CORE_ASSETS == ("connections_filtered", "nt_predictions")
    text = discover(sample_fafb).report_text()
    for line in text.splitlines():
        if "(REQUIRED)" in line:
            assert line.split()[0] in CORE_ASSETS, line
