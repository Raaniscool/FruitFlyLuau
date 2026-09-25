# EXPERIMENTS.md

Five experiments, one learning criterion, and the numbers as measured. Nothing here
reports success; the point of the file is that the null results are reproducible and
explained.

## How to run one

```bash
# synthetic sample data, ~2 s, no download
python -m fruitfly run --data-dir data/sample/fafb_v783 -e exp001_binary \
    --mode tiny --neurons 400 --episodes 200 --plot runs/exp001/figures

# the three negative controls in one go
python -m fruitfly run --data-dir data/sample/fafb_v783 -e exp003_pattern \
    --mode small --episodes 400 --controls

# any parameter, from the command line (config is the single source of truth)
python -m fruitfly run -e exp005_symbolic --set plasticity.eta=0.01 \
    --set train.steps_per_trial=0 --set lif.noise=0.8 --set lif.background_current=1.5
```

Layered config: `config/default.yaml` ← `config/local.yaml` (git-ignored) ←
`--config config/experiments/<name>.yaml` ← `--set` ← `$FAFB_DATA_PATH`.
`config.resolved.yaml` in the run directory records what was actually used, and its
`config_hash` is echoed in `setup.json`.

## The criterion

A run counts as **learning detected** only if all five hold:

1. held-out accuracy after training > chance + 5 points,
2. held-out accuracy after training > held-out accuracy **before** training by > 5 points,
3. one-sided exact binomial p < 0.05 against chance (`scipy.stats.binomtest`),
4. the weight vector actually moved: `total |dW| > 0`, and
5. **the untrained network was at or near chance to begin with** (`baseline_at_chance`).

Condition 5 was added after measuring a run where the readout mapping was already
0.875 accurate before training and reached 1.000 afterwards: four of the original
five conditions passed and the harness reported "learning". That is a ceiling effect
of a favourable mapping, not learning, and the runner now refuses to claim it (see
`tests/test_experiment.py::test_a_task_already_solved_at_baseline_is_not_reported_as_learning`).
The full breakdown of which condition failed is written to `results.json` under
`learning_criterion`.

Held-out means `env.eval_items()`, measured by `runner.evaluate()` both before and
after training. Requirement 4 exists because a silent network produces an accurate
looking 0.5/0.5 null that is really a plumbing failure — and because the runner logs a
warning when the readout population receives no spikes during the stimulus window.

## The experiments

| id | task | encoder | readout | classes | steps/trial | held-out status |
|---|---|---|---|---|---|---|
| `exp001_binary` | stimulus 0/1 → target (identity or inverted) | `binary` | `group_rate` | 2 | 48 | **not held out** — 2 stimuli exist; `exhaustive_input_space=True`, measures associative learning only |
| `exp002_xor` | (a,b) → a XOR b | `bitvector` | `group_rate` | 2 | 48 | not held out — 4 stimuli |
| `exp003_pattern` | classify one of K binary patterns, corrupted by bit-flip noise | `bitvector` | `group_rate` | 3 | 16 | **genuinely held out**: eval items are fresh noise realisations (verified: 0/20 eval observations appear in training) |
| `exp004_sequence` | next symbol of a cyclic sequence | `sequence` | `group_rate` | 4 | 48 | not held out — the alphabet enumerates the input space |
| `exp005_symbolic` | `op a b` over Z_mod | `sequence` | `group_rate` | 3 | 48 | held out **only** for a large modulus; at the default `modulus=3` there are 27 possible inputs and a 40-trial training set contains all of them, so the environment reports `exhaustive_input_space=True` and the report says "NOT held out" |

`Environment.eval_items()` resamples until an eval item's key is absent from the
training keys, for tasks that have room to do so. `overlap_check()` is written into
every `results.json` as `controls.leakage_audit`, so a leaked split cannot be hidden
by omission. Two of the audits above (exp005's default modulus, and exp001/002/004)
are *reported as unfixable by construction* rather than papered over.

## Calibration (appendix: how the LIF parameters were chosen)

`scripts/calibrate_dynamics.py` sweeps 180 combinations of `dt_ms, tau_ms, tau_syn_ms,
v_thresh, v_reset, refractory_ms, synaptic_gain, adaptation, clip_v, encoder.amplitude,
train.steps_per_trial` on a 1,000-neuron synthetic subgraph, and scores each on
measured activity rather than on a target number:

- spikes per trial and per neuron,
- fraction of neurons ever active,
- **saturation** (fraction of steps at the `clip_v` bound) — the failure mode where
  everything fires all the time,
- whether spikes reach the *readout* population at all,
- separability of the two stimulus conditions (mean rate difference).

55 of 180 settings were declared viable (not saturated, readout driven, both
conditions distinguishable); the adopted defaults in `fruitfly/config.py:LIFConfig`
are the smallest-noise member of that set:

```
dt_ms 0.5 · tau_ms 6 · tau_syn_ms 2 · v_rest -62 · v_thresh -52 · v_reset -62
refractory_ms 2 · synaptic_gain 2 · adaptation 0 · clip_v 40 · encoder.amplitude 22
train.steps_per_trial 0 (= environment's trial_steps, 48)
```

Two lessons are recorded here because they cost hours and are not obvious from the
code: score calibration by *saturation and readout drive*, never by condition
separation alone (separation looks perfect when one condition is simply silent);
and `active_fraction == 1.0` is a failure, not a success.

## Results as measured

**Set A — 600 neurons, 400–800 episodes** (`scripts/sweep_plasticity.py` without `--quick`;
synthetic sample population, `noise=0.8`, `background_current=1.5`, seed 1, 48 held-out
trials, counterbalanced readout unless the row says otherwise). Every row measured
`learned=False`:

| config (η, episodes, overrides) | exp001 before → after | exp002 before → after | \|dW\| (exp001 / exp002) | spikes/trial |
|---|---|---|---|---|
| η=1e-3, 400, `normalize=colsum` | 0.000 → 0.000 | 0.479 → 0.417 | 2,938 / 795 | 2,511 / 2,250 |
| η=3e-3, 400, `normalize=colsum` | 0.000 → 0.000 | 0.479 → 0.500 | 8,471 / 2,312 | 2,513 / 2,250 |
| η=1e-3, 400, `center_eligibility=false` | 0.000 → 0.000 | 0.479 → 0.458 | 3,497 / 1,021 | 2,488 / 2,250 |
| η=3e-3, 800, `normalize=none, w_max=3` | 0.000 → **0.500** | 0.479 → 0.500 | 3,994 / 3,675 | 2,471 / 2,245 |
| … same, `counterbalance_init=false` | 1.000 → 1.000 | 0.479 → 0.479 | 342 / 3,660 | 2,517 / 2,245 |
| … same, `counterbalance_maximize=true` | 1.000 → 1.000 | 0.521 → 0.500 | 367 / 3,150 | 2,513 / 2,281 |
| η=3e-3, 400, `plastic_targets=all` | 0.000 → 0.000 | 0.479 → 0.458 | **1,558,854** / 262,868 | 4,805 / 1,771 |
| η=3e-3, 400, `rule=stdp` (unmodulated) | 0.000 → 0.000 | 0.479 → 0.375 | 2,696 / 2,399 | 2,474 / 2,249 |
| η=3e-3, 400, `rule=none` (frozen control) | 0.000 → 0.000 | 0.479 → 0.500 | **0** / **0** | 2,476 / 2,252 |

**Set B — 200 neurons, 60 episodes** (`scripts/sweep_plasticity.py --quick`, same
overrides; this is the fast reproducible run). 0/18 configurations crossed the
criterion. The rows that moved, and how far:

| config | exp001 before → after | note |
|---|---|---|
| η=1e-3, `colsum` | 0.021 → 0.188 | below chance at baseline (counterbalanced to the worst mapping) |
| η=3e-3, `colsum` | 0.021 → 0.500 | rises to exactly chance, no further |
| η=1e-3, `colsum`, `center_eligibility=false` | 0.021 → 0.500 | as above |
| η=3e-3, `normalize=none, w_max=3` | 0.021 → 0.500 | train-window 0.433 |
| η=3e-3, `plastic_targets=all` | 0.021 → 0.500, \|dW\|=26,085 | three orders of magnitude more weight change, same 0.500 |
| η=3e-3, `colsum`, `rule=stdp` | 0.021 → 0.021 | unmodulated STDP does nothing here — the control works |
| η=3e-3, `colsum`, `rule=none` | 0.021 → 0.104, \|dW\|=0 | frozen control works (accuracy moves only via noise, not weights) |
| η=3e-3, `normalize=none`, `counterbalance_init=false` | 0.875 → 1.000, \|dW\|=880 | **excluded by condition 5**: already solved at baseline |
| η=3e-3, `colsum`, `counterbalance_maximize=true` | 0.979 → 1.000 | best-mapping case: no headroom, correctly not claimed |

Reading of both tables:

- **No configuration crossed the criterion in either set.**
- The controls behave correctly: `rule=none` gives exactly |dW| = 0 in both sets, and
  with the readout counterbalance off exp001 is already at 0.875–1.000 held-out
  *before* training — which is precisely why the counterbalance (and now condition 5) exists.
- Weight magnitudes respond strongly to the plasticity knobs (|dW| spans 342 → 1,558,854,
  a factor of ~4,500) while accuracy does not move past chance. That dissociation is the
  key measurement: **the failure is representational, not a missing mechanism.** The
  synapses change; the decision does not.
- The one repeatable effect is **0.021 → 0.500**: the network escapes the deliberately
  worst readout mapping and lands on chance. That is plasticity doing something
  measurable on held-out data (a 48-point improvement, significant by binomial test),
  but it stops exactly where the information in the mapping runs out, so it is reported
  as "unlearned the anti-mapping, did not acquire the task" rather than as learning.
- exp003/exp004/exp005 run through the same pipeline (`python -m fruitfly run -e …`)
  are not tabulated: at ≤400 episodes with a 16–48 step window they stay in the same
  near-silent-or-saturated regime, so they would add no information beyond the
  diagnosis below.

## Diagnosis (what the null result is evidence *for*)

1. **The readout cannot represent a controllable signal in a small homogeneous
   subgraph.** `group_rate` compares spike counts of two cell groups chosen from a
   random ~600-neuron sample. The input stimulus reaches both groups through similar
   local wiring, so any monotone weight change lifts or lowers them together and the
   argmax is unchanged.
2. **The "hardest" readout mapping is not learnable in this architecture.** With a
   constant-sign modulator, co-active synapses onto *both* class groups are depressed
   equally; weights pile up at `w_min`; the ranking cannot flip. Discovered by
   measuring the counterbalance target, not by intuition — it is why
   `counterbalance_maximize` and `normalize=colsum`/`center_eligibility` were added
   (both make the modulation competitive; neither, so far, moves accuracy).
3. **Activity regime is a trade-off, measured not guessed.** With defaults
   (`noise=0`, `background_current=0`) at 60–300 neurons the readout receives no
   spikes: |dW| = 0 and the run is vacuous. Adding noise+background (0.8/1.5) drives
   the readout but saturates the population (~2,500 spikes/trial) and destroys
   selectivity. The only setting that moved exp001 off 0.000 was the compromise
   (`normalize=none, w_max=3`) and it stopped at chance.
4. **The modulator must not be baseline-subtracted by default.** With `positive=negative=±1`
   delivered every trial and a constant outcome, an EMA "reward prediction error"
   collapses to ~0 and the mechanism contributes nothing — `reward.baseline_subtract`
   is `False`, and `tests/test_learning.py::test_prediction_error_mode_collapses_for_a_constant_outcome`
   pins that behaviour so the default can't be changed back by accident.
5. **Delayed credit was silently a no-op until the tests existed.** In
   `credit_mode="end_of_trial"` the rule used to accumulate nothing (only `apply()`
   fills traces, and it was skipped), so |dW| was identically zero. Fixed by
   `PlasticityRule.accumulate()` + `apply_at_trial_end()`; `test_reward_reaches_weights`
   now covers it. This is the kind of bug that survives forever behind a "0.5 vs 0.5"
   null result.

## Next step that follows from the data

Do not tune further on this readout. The two candidate fixes implied by the diagnosis,
in order of expected value:

1. **Initialise the readout mapping at chance instead of at minimum accuracy** — the
   0.021 → 0.500 rows above are what the minimum-accuracy target buys, and it is
   unlearnable past chance in this architecture. `graph.counterbalance_maximize: true`
   (already wired) is the other side of that coin: measured at 0.979 → 1.000, i.e. no
   headroom, so neither extreme produces an interpretable learning signal today.
2. **Make the readout cell groups functionally distinct before training** — select
   inputs/outputs with `graph.selection=upstream_of`/`downstream_of` around a target
   neuropil (e.g. the olfactory → MBON path via `neuron_type`/`visual` selection on a
   real download), so the two groups differ in their anatomy rather than in a random
   draw. The `input_output_sets()` plumbing and the audit it reports already exist.

Until one of those (or something better) produces a held-out number above chance with
|dW| > 0, the correct summary of this project is: **framework verified end-to-end,
learning not yet demonstrated.**

---

## Reservoir characterisation (frozen connectome, no training)

Added after the null results above. Its purpose is to separate two explanations
that the learning experiments could not distinguish:

1. the plasticity rule is wrong, or
2. the network's state never contains the information a readout would need — in
   which case no learning rule could succeed.

**Nothing is trained here except a closed-form ridge readout; the connectome is
frozen and the code asserts it** (`characterise()` raises if a single weight
changes). Method follows Jaeger (2001) memory capacity, Legenstein & Maass (2007)
separation property, and kernel/generalisation rank.

### What is measured

| metric | question it answers | how to read it |
|---|---|---|
| memory capacity (MC) | how many past timesteps of the input can a linear decoder recover from the current state? | in units of timesteps; scored on **held-out** timesteps with a contiguous split |
| separation ratio | do different input streams leave distinguishable traces? | > 1 means distinguishable; ≈ 1 means the network smears inputs together |
| kernel rank − generalisation rank | useful dimensionality minus noise-driven dimensionality | higher is better; kernel rank is capped by `--rank-streams` and saturation is flagged |

The background drive is **calibrated, not assumed**: `calibrate_drive()` sweeps
candidate bias currents and picks the smallest one whose measured mean rate lands
in 5–80 Hz, recording every candidate it tried. Without it a sparse subgraph sits
below threshold and every metric describes a silent network — which the report
flags explicitly rather than scoring.

### Reproduce

```bash
python -m fruitfly reservoir --data-dir data/sample/fafb_v783 --neurons 300 \
    --steps 600 --washout 100 --max-delay 20 --json runs/reservoir.json
```

### Measured on the synthetic sample (300 requested → 178 in the connected subgraph)

| arm | mean rate | MC (delays 1–20) | separation | kernel − generalisation rank |
|---|---:|---:|---:|---:|
| real wiring | 7.77 Hz | 0.51 | 1.59 | 4 − 4 = 0 |
| degree-preserving shuffle | 7.94 Hz | 0.66 | 0.53 | 8 − 8 = 0 |

**Read this as a validation of the instrument, not as a result about the fly.**
The sample data is synthetic random wiring, so the real arm *should* score about
the same as its shuffle — and it does (MC 0.51 vs 0.66, i.e. no advantage, with
the shuffle nominally ahead). A tool that reported the real wiring winning here
would be hallucinating structure that does not exist in the file.

The number that matters is the same table computed on the actual FAFB download,
where the two arms differ in something real. That run is pending the schema audit.

