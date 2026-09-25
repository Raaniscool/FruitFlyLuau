#!/usr/bin/env python3
"""Empirically calibrate the LIF + encoding parameters, then report the sweep.

Why this exists: the simulation's time constant, gains and input amplitude are
*not* given by the connectome -- only the wiring and synapse counts are. Choosing
them by hand produces either a silent network or a saturated one (both of which we
hit during development). This script sweeps the free parameters and scores each
setting on measurements that matter for the learning experiments:

  active_fraction   share of neurons that ever spike            (want > 0.05)
  saturation        share of neurons firing on >50% of steps    (want < 0.20)
  separation        normalised readout difference between the two
                    stimulus patterns                           (want > 0)
  spread            separation's variability across seeds      (reported)

Run:
    python scripts/calibrate_dynamics.py --neurons 1000 --episodes 8
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitfly.config import LIFConfig  # noqa: E402
from fruitfly.dataset.build_sample import sample_connectome  # noqa: E402
from fruitfly.io.encoder import BinaryEncoder  # noqa: E402
from fruitfly.neuro.network import NetworkSimulator  # noqa: E402


def measure(conn, lif: LIFConfig, *, amp: float, n_input: int, n_out: int, steps: int, seeds: range) -> dict:
    """Drive two distinct input patterns and measure the resulting activity."""
    from fruitfly.graph.select import input_output_sets

    n = conn.n_neurons
    in_idx, out_idx, audit = input_output_sets(conn, n_inputs=n_input, n_outputs=n_out, rng=np.random.default_rng(0))
    half_in = max(1, in_idx.size // 2)
    groups = np.array_split(out_idx, 2)
    seps, active, sats, tot, readout_tot, balance = [], [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sim = NetworkSimulator(conn, lif, seed=seed)
        scores = []
        for which in (0, 1):
            sim.reset()
            counts = np.zeros(n, dtype=np.int64)
            driven = in_idx[:half_in] if which == 0 else in_idx[half_in:]
            for t in range(steps):
                cur = np.zeros(n)
                cur[driven] = amp * (1.0 + 0.15 * rng.standard_normal(driven.size))
                counts += sim.step(cur)
            g = np.array([counts[grp].sum() for grp in groups], dtype=np.float64)
            scores.append(g)
            tot.append(float(counts.sum()))
            active.append(float((counts > 0).mean()))
            sats.append(float((counts > steps * 0.5).mean()))
        s0, s1 = np.array(scores[0]), np.array(scores[1])
        denom = s0.sum() + s1.sum()
        # signed difference per group: how much each pattern pushes each readout group
        delta = s1 - s0
        seps.append(float(np.abs(delta).sum() / denom) if denom > 0 else 0.0)
        readout_tot.append(float(s0.sum()))
        readout_tot.append(float(s1.sum()))
        balance.append(min(s0.sum(), s1.sum()) / max(s0.sum(), s1.sum(), 1.0))
    return {
        "active_fraction": float(np.mean(active)),
        "saturation": float(np.mean(sats)),
        "separation": float(np.mean(seps)),
        "separation_spread": float(np.std(seps)),
        "spikes_per_trial": float(np.mean(tot)),
        "readout_spikes_min": float(np.min(readout_tot)) if readout_tot else 0.0,
        "readout_balance": float(np.mean(balance)) if balance else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=4, help="seeds to average over")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--n-input", type=int, default=24)
    ap.add_argument("--n-output", type=int, default=16)
    ap.add_argument("--amp", default="8,12,16,22,30")
    ap.add_argument("--gain", default="1,2,4")
    ap.add_argument("--tau", default="3,4,6,10")
    ap.add_argument("--adapt", default="0,1.5,3")
    ap.add_argument("--out", default="", help="optional CSV path to write the sweep")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    conn, _ = sample_connectome(args.neurons, rng=rng)
    print(f"graph: {conn.n_neurons:,} neurons, {conn.n_edges:,} edges (synthetic sample)")
    print(f"spikes/trial cap for viability: {args_cap:,.0f}" if (args_cap := 0.30 * args.neurons * args.steps) else "")
    seeds = range(args.episodes)

    rows = []
    args_cap = 0.30 * args.neurons * args.steps  # no more than 30% of a fully saturated run
    for amp, gain, tau, adapt in itertools.product(
        [float(x) for x in args.amp.split(",")],
        [float(x) for x in args.gain.split(",")],
        [float(x) for x in args.tau.split(",")],
        [float(x) for x in args.adapt.split(",")],
    ):
        lif = LIFConfig(
            dt_ms=0.5, tau_ms=tau, v_rest=-62.0, v_thresh=-52.0, v_reset=-62.0,
            refractory_ms=2.0, tau_syn_ms=2.0, synaptic_gain=gain, adaptation=adapt, clip_v=40.0,
        )
        m = measure(conn, lif, amp=amp, n_input=args.n_input, n_out=args.n_output, steps=args.steps, seeds=seeds)
        m.update({"amp": amp, "gain": gain, "tau": tau, "adapt": adapt, "steps": args.steps})
        # Both patterns must actually reach the readout, otherwise "separation"
        # is trivially maximal because one condition is simply silent.
        active_ok = m["active_fraction"] >= 0.02
        not_saturated = m["saturation"] < 0.20
        readout_ok = m["readout_spikes_min"] >= 5.0 and m["readout_balance"] >= 0.25
        too_loud = m["spikes_per_trial"] > args_cap
        viable = bool(active_ok and not_saturated and readout_ok and not too_loud)
        m["viable"] = viable
        # reward discriminability, balance, and enough absolute readout spikes that
        # the signal is not one Poisson blip: log-damped activity term
        m["score"] = (
            m["separation"] * m["readout_balance"] * np.log1p(m["readout_spikes_min"]) if viable else -1.0
        )
        rows.append(m)

    rows.sort(key=lambda r: -r["score"])
    hdr = (f"{'amp':>5} {'gain':>5} {'tau':>5} {'adapt':>6} {'active':>7} {'satur':>7} "
           f"{'separ':>7} {'balanc':>7} {'rdout_min':>9} {'spikes':>9}  viable")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:18]:
        print(
            f"{r['amp']:>5.1f} {r['gain']:>5.1f} {r['tau']:>5.1f} {r['adapt']:>6.1f} "
            f"{r['active_fraction']:>7.3f} {r['saturation']:>7.3f} {r['separation']:>7.3f} "
            f"{r['readout_balance']:>7.3f} {r['readout_spikes_min']:>9.1f} "
            f"{r['spikes_per_trial']:>9.1f}  {r['viable']}  score={r['score']:.3f}"
        )
    n_viable = sum(r["viable"] for r in rows)
    print(f"\n{len(rows)} settings swept; {n_viable} viable "
          f"(active>=2%, saturation<20%, both patterns reach readout with >=5 spikes, "
          f"balance>=0.25, spikes<=30% of saturation)")
    if rows and rows[0]["score"] > 0:
        b = rows[0]
        print(
            "BEST:", ", ".join(f"{k}={b[k]}" for k in ("amp", "gain", "tau", "adapt")),
            f"separation={b['separation']:.3f} active={b['active_fraction']:.3f} "
            f"saturation={b['saturation']:.3f} readout_min={b['readout_spikes_min']:.1f} score={b['score']:.3f}",
        )
    else:
        print("BEST: no setting produced a non-saturated, discriminative readout at this scale")
    if args.out:
        import csv

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
