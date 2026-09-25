"""Command-line interface: ``python -m fruitfly <command>`` / ``fruitfly <command>``.

Commands
--------
doctor    what the project can see: data dir, assets, python deps, scales
sample    write a tiny synthetic FAFB-shaped sample directory
inspect   deep schema probe of a data directory (same engine as scripts/inspect_fafb.py)
stats     dataset + derived-graph statistics for a chosen scale
sim       run a short LIF simulation and report measured activity
run       run an experiment (train + evaluate) with controls
rules     list plasticity rules / encoders / decoders / selections
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..paths import ENV_VAR, find_data_dir
from ..utils import get_logger, human_bytes, table

log = get_logger("cli")


def _cfg(args: argparse.Namespace) -> AppConfig:
    overrides: dict = {}
    if getattr(args, "mode", None):
        overrides.setdefault("graph", {})["mode"] = args.mode
    if getattr(args, "neurons", None):
        overrides.setdefault("graph", {})["n_neurons"] = int(args.neurons)
    if getattr(args, "selection", None):
        overrides.setdefault("graph", {})["selection"] = args.selection
    if getattr(args, "data_dir", None):
        overrides.setdefault("data", {})["fafb_data_path"] = str(args.data_dir)
    if getattr(args, "episodes", None):
        overrides.setdefault("train", {})["episodes"] = int(args.episodes)
    if getattr(args, "seed", None) is not None:
        overrides.setdefault("train", {})["seed"] = int(args.seed)
    if getattr(args, "rule", None):
        overrides.setdefault("plasticity", {})["rule"] = args.rule
    for pair in getattr(args, "set", []) or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects section.key=value, got {pair!r}")
        path, value = pair.split("=", 1)
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = value
        if "." in path:
            section, key = path.split(".", 1)
            overrides.setdefault(section, {})[key] = parsed
        else:
            # top-level scalars (experiment, name, notes) have no section
            overrides[path] = parsed
    return AppConfig.load(getattr(args, "config", None), overrides=overrides)


# ----------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    p, trail = find_data_dir(args.data_dir if hasattr(args, "data_dir") else None)
    print(f"python {sys.version.split()[0]}  on {sys.platform}")
    for mod in ("numpy", "scipy", "pandas", "yaml", "matplotlib", "pyarrow"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12} {getattr(m, '__version__', 'ok')}")
        except Exception:
            print(f"  {mod:<12} MISSING  (pip install 'fruitflyluau[dev]')")
    print()
    print(trail.describe())
    if p is None:
        print()
        print(f"Set {ENV_VAR} to your folder, e.g. (PowerShell)")
        print(f'  Set-Item -Path Env:{ENV_VAR} -Value "$env:USERPROFILE\\Downloads"')
        print("then re-run:  python scripts/inspect_fafb.py --dir $env:" + ENV_VAR)
        print("The 'sample' mode works without any data:  python -m fruitfly sample")
    else:
        from ..dataset.discover import discover

        inv = discover(p)
        print()
        print(inv.report_text())
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from ..dataset.build_sample import write_sample_fafb_files

    out = Path(args.out)
    counts = write_sample_fafb_files(out, n_neurons=int(args.neurons or 640),
                                     seed=int(args.seed) if args.seed is not None else 20260923)
    total = sum(Path(out / f).stat().st_size for f in counts)
    print(f"wrote SYNTHETIC sample FAFB directory: {out} ({human_bytes(total)})")
    for name, n in sorted(counts.items()):
        print(f"  {name:<44} {n:>8,} rows")
    print("\nThese files use the real FAFB v783 column names but random content.")
    print("Use them for:  fruitfly stats --data-dir %s --mode tiny" % out)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from ..dataset.discover import discover

    root = Path(args.data_dir).expanduser() if args.data_dir else find_data_dir()[0]
    if root is None:
        print("no data directory found; pass --data-dir or set " + ENV_VAR, file=sys.stderr)
        return 2
    inv = discover(root, count_rows=args.count_rows)
    print(inv.report_text())
    if args.json:
        Path(args.json).write_text(json.dumps(inv.summary(), indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if inv.connections is not None else 2


def _load_population(cfg: AppConfig):
    from ..graph.build import build_population

    return build_population(cfg)


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    pop = _load_population(cfg)
    print("=== dataset (as read) ===")
    print(json.dumps(pop.connection_table_stats, indent=2, default=str))
    print("\n=== simulated population ===")
    print(pop.describe())
    if args.json:
        Path(args.json).write_text(
            json.dumps({"dataset": pop.connection_table_stats, "population": pop.connectome.stats(),
                        "describe": pop.describe(), "provenance": pop.connectome.provenance}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    from ..neuro.network import NetworkSimulator

    cfg = _cfg(args)
    pop = _load_population(cfg)
    conn = pop.connectome
    sim = NetworkSimulator(conn, cfg.lif, seed=cfg.train.seed)
    n = conn.n_neurons
    rng = np_rng(cfg.train.seed)
    drive = np.zeros(n)
    inputs = np.linspace(0, n - 1, min(16, max(2, n // 8))).round().astype(int)
    drive[inputs] = 22.0
    steps = int(args.steps)
    rates = np.zeros(steps)
    counts = np.zeros(n, dtype=np.int64)
    for t in range(steps):
        s = sim.step(drive)
        counts += s
        rates[t] = s.sum()
    diag = sim.diagnostics(steps)
    print(pop.describe())
    print(
        "\n=== simulation ===\n"
        f"steps {steps}  dt {cfg.lif.dt_ms} ms  ({steps * cfg.lif.dt_ms / 1000:.2f} s simulated)\n"
        f"total spikes        : {diag.n_spikes:,}\n"
        f"neurons ever active : {diag.fraction_neurons_ever_active:.3f} of {n:,}\n"
        f"mean population rate: {diag.mean_rate_hz:.2f} Hz\n"
        f"max single neuron   : {diag.max_single_neuron_rate_hz:.2f} Hz\n"
        f"CV of ISI           : {diag.cv_isi:.3f}\n"
        f"wall time           : see --verbose"
    )
    if args.plot:
        from ..viz.plots import plot_activity

        out = _figure_path(args.plot, "activity.png")
        written = plot_activity(rates, counts, out=str(out), dpi=cfg.viz.dpi)
        print(f"figure -> {written}" if written else "plotting skipped (matplotlib not installed)")
    return 0


def _figure_path(spec: str, default_name: str) -> Path:
    """Accept either a directory or a file path for --plot."""
    pth = Path(spec).expanduser()
    if pth.suffix.lower() in {".png", ".pdf", ".svg", ".jpg"}:
        return pth
    return pth / default_name


def np_rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def cmd_run(args: argparse.Namespace) -> int:
    from ..experiment import ExperimentRunner, available_environments, build_environment

    cfg = _cfg(args)
    cfg.experiment = args.experiment
    env_kwargs = {"seed": cfg.train.seed}
    if args.n_train:
        env_kwargs["n_train"] = args.n_train
    if args.n_eval:
        env_kwargs["n_eval"] = args.n_eval
    if args.env_opt:
        for pair in args.env_opt:
            k, v = pair.split("=", 1)
            try:
                env_kwargs[k] = json.loads(v)
            except Exception:
                env_kwargs[k] = v
    if args.experiment not in available_environments():
        print(f"unknown experiment {args.experiment!r}; available: {', '.join(available_environments())}", file=sys.stderr)
        return 1
    env = build_environment(args.experiment, **env_kwargs)
    runner = ExperimentRunner(cfg, env, run_dir=args.run_dir or None)
    runner.setup(resume_from=args.resume)
    print(f"run directory: {runner.run_dir}")
    outcome = runner.train()
    print(outcome.summary_text())
    if args.controls:
        for control in ("no_plasticity", "shuffled_wiring", "sham_reward"):
            cfg2 = AppConfig.from_dict(cfg.to_dict())
            cfg2.train.control = control
            cfg2.train.log_dir = str(Path(runner.run_dir).parent / f"{Path(runner.run_dir).name}__{control}")
            env2 = build_environment(args.experiment, **env_kwargs)
            r2 = ExperimentRunner(cfg2, env2)
            r2.setup()
            o2 = r2.train()
            print(f"[control {control}] eval after = {o2.eval_after['accuracy']} (treatment {outcome.eval_after['accuracy']})")
    if args.plot:
        from ..viz.plots import plot_learning

        plot_dir = str(runner.run_dir) if args.plot == "auto" else args.plot
        written = plot_learning(runner, out_dir=plot_dir, dpi=cfg.viz.dpi)
        if written:
            print("figures -> " + ", ".join(str(w) for w in written))
        else:
            print("plotting skipped (matplotlib not installed)")
    return 0


def cmd_reservoir(args: argparse.Namespace) -> int:
    """Characterise the FROZEN connectome as a reservoir. Trains no weights."""
    from ..experiment.reservoir import characterise
    from ..graph.select import input_output_sets

    cfg = _cfg(args)
    pop = _load_population(cfg)
    conn = pop.connectome
    n_in = int(cfg.graph.selection_kwargs.get("n_inputs", 16))
    n_out = int(cfg.graph.selection_kwargs.get("n_outputs", 4))
    inputs, _outputs, audit = input_output_sets(
        conn, n_inputs=n_in, n_outputs=n_out, rng=np_rng(cfg.train.seed))
    print(pop.describe())
    print(f"\ninput drive neurons: {inputs.size} (source: {audit.get('input_source', '-')}, "
          f"{audit.get('n_ancestors_within_depth', 0):,} ancestors within depth {audit.get('depth', '-')})")
    rows = []
    controls = ["none", "shuffled"] if args.control == "both" else [args.control]
    for ctrl in controls:
        rep = characterise(
            conn, cfg.lif,
            input_neurons=np.asarray(inputs),
            steps=int(args.steps), washout=int(args.washout),
            max_delay=int(args.max_delay), amplitude=float(args.amplitude),
            bias=(None if args.bias is None else float(args.bias)),
            seed=cfg.train.seed, n_rank_streams=int(args.rank_streams),
            control=ctrl,
        )
        print()
        print(rep.describe())
        rows.append(rep.to_dict())
    print("\nNo weight was modified: this measures what the wiring makes available to a "
          "linear decoder, not what the network can learn.")
    if len(rows) == 2:
        a, b = rows[0]["memory"]["total"], rows[1]["memory"]["total"]
        print(f"real vs shuffled memory capacity: {a:.2f} vs {b:.2f} "
              f"({'real wiring ahead' if a > b else 'no advantage for the real wiring'})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Serve the desk-fly demo. The UI states on screen that it is a placeholder."""
    from ..demo.server import run

    print(f"desk-fly demo on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    print("NOTE: the default backend is a snippet library, not the simulated connectome.")
    return run(args.host, int(args.port))


def cmd_rules(args: argparse.Namespace) -> int:
    from ..io.decoder import available_decoders
    from ..io.encoder import available_encoders
    from ..neuro.plasticity import available_rules
    from ..graph.select import list_selectors

    print("plasticity rules :", ", ".join(available_rules()))
    print("encoders         :", ", ".join(available_encoders()))
    print("decoders         :", ", ".join(available_decoders()))
    print("selection modes  :", ", ".join(list_selectors()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fruitfly", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML config file to layer over config/default.yaml")
    common.add_argument("--set", action="append", help="override, e.g. --set plasticity.eta=0.01")
    common.add_argument("--mode", default=None, help="tiny|small|medium|large|full|sample")
    common.add_argument("--neurons", type=int, default=None)
    common.add_argument("--selection", default=None)
    common.add_argument("--data-dir", default=None)
    common.add_argument("--seed", type=int, default=None)

    sub.add_parser("doctor", parents=[common], help="report what the project can see").set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("sample", parents=[common], help="write a tiny synthetic FAFB-shaped directory")
    sp.add_argument("--out", default="data/sample_fafb")
    # --neurons comes from `common`; the sample default lives here so the shared
    # flag stays None-meaningless. (Re-adding it here would collide with common.)
    sp.set_defaults(fn=cmd_sample)

    ip = sub.add_parser("inspect", parents=[common], help="probe a data directory's real schema")
    ip.add_argument("--count-rows", action="store_true")
    ip.add_argument("--json", default="")
    ip.set_defaults(fn=cmd_inspect)

    st = sub.add_parser("stats", parents=[common], help="dataset + population statistics")
    st.add_argument("--json", default="")
    st.set_defaults(fn=cmd_stats)

    sm = sub.add_parser("sim", parents=[common], help="short LIF run, measured activity report")
    sm.add_argument("--steps", type=int, default=200)
    sm.add_argument("--plot", nargs="?", const=".", default="",
                    help="write activity.png; bare --plot uses the current directory")
    sm.set_defaults(fn=cmd_sim)

    rn = sub.add_parser("run", parents=[common], help="train an experiment")
    rn.add_argument("--experiment", "-e", default="exp001_binary")
    rn.add_argument("--episodes", type=int, default=None)
    rn.add_argument("--n-train", type=int, default=0)
    rn.add_argument("--n-eval", type=int, default=0)
    rn.add_argument("--rule", default=None)
    rn.add_argument("--run-dir", default=None)
    rn.add_argument("--resume", default=None)
    rn.add_argument("--controls", action="store_true", help="also run the three negative controls")
    rn.add_argument("--env-opt", action="append", help="environment kwarg, e.g. --env-opt noise=0.2")
    rn.add_argument("--plot", nargs="?", const="auto", default="",
                    help="write the figures; bare --plot uses the run directory")
    rn.set_defaults(fn=cmd_run)

    rv = sub.add_parser("reservoir", parents=[common],
                        help="frozen-connectome reservoir measurements (no training)")
    rv.add_argument("--steps", type=int, default=600)
    rv.add_argument("--washout", type=int, default=100)
    rv.add_argument("--max-delay", type=int, default=20)
    rv.add_argument("--amplitude", type=float, default=22.0)
    rv.add_argument("--bias", type=float, default=None,
                    help="background current; omit to calibrate it by measurement")
    rv.add_argument("--rank-streams", type=int, default=8)
    rv.add_argument("--control", default="both", choices=["none", "shuffled", "both"])
    rv.add_argument("--json", default="")
    rv.set_defaults(fn=cmd_reservoir)

    dm = sub.add_parser("demo", help="serve the desk-fly web demo (placeholder backend)")
    dm.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on the network")
    dm.add_argument("--port", type=int, default=8000)
    dm.set_defaults(fn=cmd_demo)

    sub.add_parser("rules", help="list registered rules/encoders/decoders/selectors").set_defaults(fn=cmd_rules)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..utils import set_verbosity

    set_verbosity(verbose=args.verbose, quiet=args.quiet)
    try:
        return int(args.fn(args) or 0)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
