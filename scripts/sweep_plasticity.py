#!/usr/bin/env python3
"""Reproduce the plasticity sweep behind the table in EXPERIMENTS.md.

Nine configurations x two experiments, each a full ``ExperimentRunner`` training run
with pre/post held-out evaluation, printing one measured line per run. No result here
crossed the learning criterion; the point of the script is that the *numbers* are
auditable (weight magnitude moves by three orders of magnitude while accuracy does
not), not that the sweep is a leaderboard.

    python scripts/sweep_plasticity.py                      # 600-neuron synthetic pop, ~6 min
    python scripts/sweep_plasticity.py --neurons 200 --episodes 60 --quick
    python scripts/sweep_plasticity.py --json sweep.json --data-dir data/sample/fafb_v783

Uses the synthetic sample population by default so a fresh clone can run it without
downloading FAFB. Pass --data-dir pointing at real FAFB assets to sweep on a real
subgraph (selection then follows graph.mode/scale).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitfly.config import AppConfig  # noqa: E402
from fruitfly.experiment import ExperimentRunner, build_environment  # noqa: E402
import fruitfly.experiment.tasks  # noqa: F401,E402  (registers the environments)

#: (eta, episodes, extra overrides as "section__key" -> value)
CONFIGS: list[tuple[float, int, dict]] = [
    (0.001, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5}),
    (0.003, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5}),
    (0.001, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "plasticity__center_eligibility": False,
                  "lif__noise": 0.8, "lif__background_current": 1.5}),
    (0.003, 800, {"plasticity__plastic_targets": "output", "plasticity__normalize": "none",
                  "plasticity__w_max": 3.0, "lif__noise": 0.8, "lif__background_current": 1.5}),
    (0.003, 800, {"plasticity__plastic_targets": "output", "plasticity__normalize": "none",
                  "plasticity__w_max": 3.0, "lif__noise": 0.8, "lif__background_current": 1.5,
                  "graph__counterbalance_init": False}),
    (0.003, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5,
                  "graph__counterbalance_maximize": True}),
    (0.003, 400, {"plasticity__plastic_targets": "all", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5}),
    (0.003, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5, "plasticity__rule": "stdp"}),
    # negative control: with the rule off, |dW| must be exactly 0. If this row ever
    # reports a non-zero value, the plasticity plumbing is lying somewhere.
    (0.003, 400, {"plasticity__plastic_targets": "output", "plasticity__normalize": "colsum",
                  "lif__noise": 0.8, "lif__background_current": 1.5, "plasticity__rule": "none"}),
]


def run_one(eta: float, episodes: int, extra: dict, *, exp: str, neurons: int, seed: int,
            data_dir: str, out_root: Path) -> dict:
    cfg = AppConfig.load(use_defaults_file=False)
    cfg.graph.mode = "sample" if not data_dir else "tiny"
    if data_dir:
        cfg.data.fafb_data_path = data_dir
        cfg.data.use_cache = False
    else:
        cfg.data.source = "sample"
    cfg.graph.n_neurons = neurons
    cfg.train.episodes = episodes
    cfg.train.eval_trials = 48
    cfg.train.window = 25
    cfg.train.save_checkpoints = False
    cfg.train.eval_before_training = True
    cfg.plasticity.eta = eta
    for key, value in extra.items():
        section, name = key.split("__")
        setattr(getattr(cfg, section), name, value)
    env = build_environment(exp, seed=seed, n_train=max(32, episodes // 8), n_eval=48)
    run_dir = out_root / f"{exp}_{abs(hash(json.dumps([eta, episodes, sorted(extra.items())], default=str))):012x}"
    runner = ExperimentRunner(cfg, env, run_dir=run_dir)
    runner.setup()
    out = runner.train()
    return {
        "experiment": exp, "eta": eta, "episodes": episodes, "neurons": neurons,
        "seed": seed, "overrides": {k: v for k, v in extra.items()},
        "accuracy_before": out.eval_before["accuracy"], "accuracy_after": out.eval_after["accuracy"],
        "train_window_accuracy": out.train_accuracy_window,
        "total_abs_dw": out.weight_stats["total_abs_dw"],
        "p_value_after": out.eval_after.get("p_value_vs_chance"),
        "spikes_per_trial": out.activity["spikes_per_trial_mean"],
        "active_fraction": out.activity["active_fraction_mean"],
        "learning_detected": bool(out.learning_detected),
        "claim_bounds": out.claim_bounds,
        "run_dir": str(run_dir),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neurons", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=0, help="scale every config's episodes to this")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--experiments", default="exp001_binary,exp002_xor")
    ap.add_argument("--data-dir", default="", help="use real FAFB assets instead of the synthetic sample")
    ap.add_argument("--out-root", default="runs/sweep")
    ap.add_argument("--json", default="", help="also write a machine-readable result file")
    ap.add_argument("--quick", action="store_true", help="200 neurons x 60 episodes, for a smoke test")
    args = ap.parse_args(argv)

    neurons = 200 if args.quick else args.neurons
    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
    out_root = Path(args.out_root)
    rows: list[dict] = []
    for eta, episodes, extra in CONFIGS:
        if args.quick:
            episodes = 60
        elif args.episodes:
            episodes = args.episodes
        for exp in exps:
            row = run_one(eta, episodes, extra, exp=exp, neurons=neurons, seed=args.seed,
                          data_dir=args.data_dir, out_root=out_root)
            rows.append(row)
            knobs = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in extra.items())
            print(
                f"{exp:<14} eta={eta} eps={episodes} pop={neurons} {knobs} "
                f":: before={row['accuracy_before']:.3f} after={row['accuracy_after']:.3f} "
                f"trainwin={row['train_window_accuracy']} |dw|={row['total_abs_dw']:,.0f} "
                f"spikes/trial={row['spikes_per_trial']:.0f} learned={row['learning_detected']}",
                flush=True,
            )
    n_learned = sum(1 for r in rows if r["learning_detected"])
    print(f"\n{n_learned}/{len(rows)} configurations crossed the learning criterion "
          f"(+5% held-out above chance and above pre-training, p<0.05, |dW|>0, "
          f"and an untrained baseline at chance).")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
