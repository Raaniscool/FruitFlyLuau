# ARCHITECTURE.md

How a real synapse becomes a decision, and the invariants that keep the result
interpretable. Each section names the module that owns the concern.

## Data flow

```
FAFB v783 files                     fruitfly/dataset/
  discover.py  -> match assets by name + header
  schema.py    -> alias columns to logical names (pre, post, weight, neuropil, nt_type)
  loader.py    -> chunked read, per-neuropil rows merged to one edge per ordered pair
                  -> ConnectionTable(pre, post, weight, neuropil, nt_type, root ids, timings)
        |
        v                       fruitfly/graph/
  select.py      -> which neurons: random / hubs / manual / neurotransmitter /
                    neuron_type / visual / upstream_of / downstream_of / connected_subgraph
  build.py       -> scale resolution, weight transform (log1p), E/I signs, counterbalanced init
  connectome.py  -> Connectome: CSR (rows = presynaptic), edge arrays, root ids, npz persistence
        |
        v                       fruitfly/neuro/ + fruitfly/io/
  network.py     -> NetworkSimulator: g = Wᵀ·s_prev;  v += (dt/τ)(-L(v-v_rest) + gain·g + I + noise)
                   spike if v > thresh + adapt and not refractory;  reset, adapt, count
  encoder.py     -> observation -> (neuron indices, amplitude) per step  [sensory drive]
  decoder.py     -> spike counts on designated readout cells -> Decision(label, scores, margin)
        |
        v                       fruitfly/learning/ + fruitfly/experiment/
  reward.py      -> ledger: deliver(value, step) becomes due at step + delay_steps
  modulator.py   -> DopamineLikeModulator: exponentially-decaying trace, optional RPE baseline, |value| cap
  plasticity.py  -> eligibility traces per edge, × modulator, per-edge clamps, row/col normalisation
  runner.py      -> trial = stimulus → readout → outcome → (delayed) reinforcement → report
  checkpoint.py  -> weights + rule/reward/modulator/encoder/decoder state per N episodes
  readout.py     -> readout-mapping counterbalance (see below)
```

Everything is NumPy/SciPy array arithmetic. A connectome of 3.7 M pairs is three
arrays, not 3.7 M Python objects, and one simulation step is one sparse mat-vec plus
a handful of O(n) vector operations.

## Invariants (each one has a test that would fail if it broke)

1. **CSR orientation.** Rows are presynaptic, columns postsynaptic, so input current
   is `W.T @ spikes_prev`. `upstream_of` walks *incoming* edges. An earlier revision
   had this mirrored, which made "upstream of MBON" select the wrong cells and turned
   every direction-dependent result into its own opposite
   (`tests/test_graph_selection.py`, `tests/test_signal_propagation.py`).
2. **The live-weight contract.** `NetworkSimulator._WT` is a transposed *view* of the
   CSR data array, and `PlasticityRule` writes into that same array. A learning rule
   that mutated a copy would silently produce a frozen-weights experiment. Enforced by
   `tests/test_signal_propagation.py::test_plasticity_is_visible_to_propagation`.
3. **Root ids survive compaction.** Neurons are indexed 0..n-1 for speed while the
   18-digit FlyWire root ids are kept alongside, so every reported spike, weight and
   figure can be mapped back to a real cell (`test_neuron_index_mapping_is_exact`).
4. **Per-neuropil rows are merged.** `connections_princeton.csv.gz` carries one row
   per (pair, neuropil); the loader sums them into one edge. Not merging inflates edge
   count and makes weight statistics meaningless.
5. **Deterministic seeding.** `seed_all(cfg.train.seed)` seeds Python, NumPy and the
   simulator's own generator; the same config + seed reproduces the accuracy series
   and the learned weight vector bit-for-bit
   (`test_full_train_writes_reports_and_is_reproducible`).
6. **Nothing is claimed without a held-out measurement.** `evaluate()` runs on
   `env.eval_items()`, and `Environment.eval_items()` actively resamples to avoid
   training keys where the task has enough stimuli. Where it cannot (binary/XOR have
   2 and 4 stimuli), the environment sets `exhaustive_input_space = True` and the
   report says "associative learning only, generalisation not applicable" instead of
   quietly reporting a leak-free number.
7. **A silent network says so.** If the readout population receives no spikes during
   the stimulus window, the runner logs a warning naming the fix (longer window,
   different selection, gains). An undriveable network otherwise looks exactly like a
   failed learning experiment.

## Credit assignment: two modes, one rule

`reward.credit_mode: auto` picks between:

- **`online`** (delay_steps = 0): the rule's `apply()` runs every step, writing
  `η · modulator · eligibility` as the trial proceeds;
- **`end_of_trial`** (delay_steps > 0): the rule's `accumulate()` runs every step —
  building eligibility traces with **no** weight write — and one
  `apply_at_trial_end(value)` call cashes them in. This is the biologically-shaped
  path (coincidence detection at the synapse, delayed reinforcement from a modulatory
  neuron) and the one an earlier version of this repo silently never ran: traces were
  never filled, so |dW| was identically zero.
  `tests/test_plasticity.py::test_reward_reaches_weights` and
  `test_accumulate_builds_traces_without_writing` pin it.

Eligibility uses the standard asymmetric STDP window (`a_plus=0.2`, `a_minus=0.31`,
τ±=20 ms) with an eligibility decay of `tau_trace_ms`; the modulator is clamped by
`reward.value_abs_max` because an unclamped trace with τ=60 ms over a 48-step trial
grows without bound and produced |dW| 100× larger with no accuracy change.

Per-edge state (`SynapseState`) carries `w_lo`/`w_hi` clamps derived from each edge's
starting sign (`plasticity.polarity_lock`), a `plastic_mask` (used by
`plasticity.plastic_fraction` for speed and by `plastic_targets="output"` to restrict
learning to synapses onto readout cells), and `row_target`/`col_target` sums for
homeostatic renormalisation (`normalize: rowsum|colsum|both|none`).

## The readout-mapping counterbalance

`experiment/readout.py` enumerates the permutations of `decoder.classes`, measures
the *untrained* accuracy of each, and adopts the **worst** one (or the best with
`graph.counterbalance_maximize: true`). Rationale: with a group-rate readout on a
homogeneous random subgraph, the identity mapping is frequently correct by accident,
and a "learning" result on top of it measures nothing. The choice, the alternatives
and the resulting initial accuracy go into `setup.json`.

**Measured caveat, recorded in EXPERIMENTS.md:** the minimum-accuracy permutation
turned out to be structurally hard to escape with a constant-sign modulator — both
class groups get depressed together, weights hit `w_min`, and the argmax ranking never
changes. That is a finding about the mechanism, not a tuning failure, and it is why
`normalize: colsum` + `center_eligibility` exist (they make the signal competitive
rather than uniformly depressory). They increase |dW| by 10–100× and, so far, still
change no accuracy.

## Experiments, controls, honesty strings

`ExperimentRunner.train()` writes `setup.json` (config, config hash, population
provenance, readout assignment + audit, plasticity mask, leakage audit),
`metrics.jsonl` (one row per trial), `results.json` (pre/post held-out accuracy,
chance, exact binomial p, weight statistics, activity, timings), `config.resolved.yaml`
and `console.log`. `TrainOutcome.summary_text()` ends with an explicit bound on what
the numbers support — "NO learning detected … Reported as a null result rather than
tuned until it passes" — because the report, not the researcher, should be the place a
claim gets limited.

Three controls are first-class: `--controls` runs `no_plasticity` (frozen weights:
proves the task was not already solved), `shuffled_wiring` (same plasticity on a
rewired graph: proves the *structure* matters) and `sham_reward` (reward sign
randomised: proves reinforcement, not self-organising STDP alone, moves behaviour).

## Deliberate absences in v0.1

- No GPU path: `train.device: cpu` and no torch dependency. The bottleneck is the
  sparse mat-vec, so the same structure works later.
- No `networkx` in the runtime path (selection is CSR BFS); `scipy.sparse.csgraph` is
  used where a component/shortest-path query is needed.
- Luau (`fruitfly/luau/`) is **not** wired into training: a tokenizer, a task schema
  and a sandbox contract only. `evaluator.execute()` refuses on every input by design,
  and `tests/test_luau.py` asserts that refusal (it checks no marker file appears).
- Skeletons/morphology are not read, so there is no distance-dependent conduction
  delay or axonal arboration model; delays are one fixed simulation step per edge.
