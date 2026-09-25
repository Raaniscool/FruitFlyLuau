# Development

For anyone editing this repo, including future-us. Read `ARCHITECTURE.md` first for the
invariants; this file is about how to work here.

## Environment

```bash
python -m venv .venv && . .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # numpy scipy pandas pyyaml pytest matplotlib
# optional: pip install -e ".[feather]" for .feather/.parquet snapshots (pyarrow)
```

Python >= 3.10 (the code uses `X | Y` annotations and `match`-free but union-heavy typing).
Everything works on Windows, macOS, Linux; the only platform-flavoured code is
`fruitfly/luau/evaluator.py` resource limits (`resource` is POSIX-only and degrades to a
wall-clock-only limit elsewhere).

```bash
python -m pytest                       # 200 tests, ~15 s, no network, no FAFB download
python -m pytest tests/test_fafb_loading.py -x -q
python -m fruitfly doctor              # what the project can see on this machine
python -m fruitfly rules               # registered rules / encoders / decoders / selectors
```

There is no lint/format tool wired into the repo. Conventions, in the order they matter:
**config-driven over hardcoded**, **fail loudly over defaulting silently**, **one source of
truth per fact** (the asset registry, the scale table, the calibration block), and a docstring
that says *why* when a number or a guard looks arbitrary.

## The phase plan

Phases are the order the system was built in, and each one is a working end state. They are not
a marketing roadmap: 7-9 are unimplemented, and the docs say so wherever they are relevant.

| phase | scope | state |
|---|---|---|
| 1 | locate + load FAFB v783 connectivity (`dataset/assets.py`, `loader.py`), schema-tolerant, cached | done, validated against synthetic files |
| 2 | graph layer: `Connectome` CSR, weight transforms, 10 selection modes, scales (`graph/`) | done |
| 3 | spiking simulation + plasticity: LIF, STDP/R-STDP/Hebbian, credit assignment, encoders/decoders (`neuro/`, `io/`, `learning/`) | done, calibrated |
| 4 | experiment harness: environments exp001-exp005, train/eval splits, metrics, checkpoints, controls, resume (`experiment/`) | done |
| 5 | reporting: CLI (`fruitfly doctor/sample/inspect/stats/sim/run/rules`), figures, `viz/` | done |
| 6 | documentation of data, architecture, experiments, limitations | this file + 4 others |
| 7 | Luau corpus → token stream → sequence task (`luau/tokens.py`, `tasks.py` exist; not wired into training) | scaffolded |
| 8 | safe execution of generated Luau (`luau/evaluator.py`) | contract only, see below |
| 9 | scale-up to the full connectome with real annotations, and a learning result worth reporting | blocked on phase 1 running against real files, and on the diagnosis in `EXPERIMENTS.md` |

### phase 8 — the execution backend, and the checklist it must pass

`fruitfly/luau/evaluator.py` today does two things and refuses to do a third: it scans a
program statically (`FORBIDDEN_PATTERNS`, `REQUIRED_LIMITS`, `MOCK_ROBLOX_API`) and it returns
a structured "no backend" result from `execute()`. **It never runs code.** That is intentional
and must not be removed casually — the functions that look like stubs are a security boundary
with tests (`tests/test_luau.py`) asserting they reject the things they reject.

Before any backend is enabled, all of the following must be true. This is the checklist the
docstring at `evaluator.py:202` points here for.

1. **Isolation.** Execution happens in a sandbox the host cannot escape: a container
   (`luau.sandbox: docker`) or a `subprocess` with dropped privileges, no network namespace,
   read-only rootfs, and a memory cap enforced by the OS (`RLIMIT_AS` / job object), not by
   trusting Luau. `luau.sandbox: none` must keep raising/refusing exactly as it does now.
2. **Hard limits, enforced twice.** Wall clock (`eval_timeout_s`, default 2 s) via `kill` on a
   timer as well as `RLIMIT_CPU`; output byte cap; no file handles other than pipes. A `while
   true do end` loop must cost the harness seconds, not a reboot.
3. **No Roblox surface.** The evaluator speaks to a mock API only (`MOCK_ROBLOX_API`); no
   `game`, no `script_context`, no instances. Generated code is evaluated against pure-Luau
   semantics; anything requiring an engine is a test failure, not a skip.
4. **Static gate first.** `scan_static()` runs before execution and `validate_limits()` runs on
   the request; forbidden constructs never reach the interpreter.
5. **Untrusted-input framing.** Generated code is adversarial by construction (it comes from a
   stochastic sampler). Treat the source buffer as hostile: fixed-size, length-checked, no
   temp paths derived from the code text.
6. **Results are recorded, not trusted.** Every execution writes its return code, wall time,
   peak RSS and the violated limit name to the run directory, so a hang or an OOM is visible in
   `metrics.jsonl` rather than looking like a wrong answer.
7. **`self_check()` stays honest.** It reports which of the above are actually implemented on
   this machine; a backend that satisfies 1-6 must also make `backend_available()` return True
   with a named backend, so a misconfiguration can never read as "0/100 tests passed".

The cheap way to start: `luau-bin` or the `lua5.4` interpreter (Luau is a Lua superset for the
subset in `tokens.py`) invoked through `subprocess` with a prebuilt args array, then run
`TestCase`s and compare pass/fail against expectation. Reward shaping on top of that is
`experiment/runner.py`'s existing `deliver_outcome` path — no new reward machinery needed.

## How to add things

**A FAFB asset.** Add one `AssetSpec` row to `fruitfly/dataset/assets.py` (key, display name,
filename, role, columns, `alternates` for known filename variants, `required_for`, approximate
size). Nothing else: `discover()`, `doctor`, `stats`, `inspect_fafb.py` and `DATA.md`'s table
all read from that registry. If the file's columns differ from the registry, the registry is
wrong or you found a new Codex export variant — update it and say which one you saw.

**An experiment.** Subclass `Environment` in `fruitfly/experiment/tasks.py` and register it
with `@register_env("expNNN_name")`. Set `encoder_kind`, `decoder_kind`, `trial_steps`
(must be >= ~48 in the calibrated regime; shorter windows silently read out noise) and
`exhaustive_input_space` if the eval split contains every possible stimulus — that flag is
what keeps the docs honest about generalisation. `overlap_check()` must return
`leakage_free=True` for any claim about unseen data, otherwise use a fresh-noise split like
`exp003_pattern`.

**A plasticity rule.** `fruitfly/neuro/plasticity.py`: subclass `PlasticityRule`, implement
`begin_trial / accumulate / end_trial / apply`, register in `available_rules()`. Respect
`state.plastic_mask` in `_write_weights` (that is how `plastic_targets: output` works) and
honour `polarity_lock`.

**A CLI flag.** Add it to the `common` parent parser *or* the subcommand, never both — argparse
aborts the whole program on a duplicate option string, and the failure is a crash on every
command. `rules` is the one subcommand with no `common` flags, so `--set` there is an error by
design.

## Testing rules

- Tests must not need the network or a FAFB download. Synthetic fixtures only
  (`write_sample_fafb_files`, `data/sample/fafb_v783/`).
- Anything that touches the criterion must have a test proving a **negative**: `rule=none`
  yields exactly |dW|=0, `sham_reward` changes behaviour but not weights, `shuffled_wiring`
  permutes the graph, and a pre-solved task is *not* reported as learning. The last one exists
  because we measured a run that passed four of five criteria while demonstrating nothing.
- `pyproject.toml` sets `addopts = -q`, which hides the summary line; check the exit code in
  scripts, not the word "passed".
- `scripts/inspect_fafb.py` is stdlib-only **on purpose** (it runs on a laptop with the
  downloads and nothing installed). `tests/test_cli_and_viz.py` enforces that with an AST scan of
  module-level imports. Do not import numpy in it.

## Reproducing the numbers in the docs

```bash
python -m fruitfly run --config config/experiments/exp001_binary_sample.yaml --plot   # ~8 s, null
python scripts/sweep_plasticity.py --quick    # 18 runs, ~13 s, prints 0/18 crossed
python scripts/sweep_plasticity.py            # 600-neuron Set A, ~6 min
python scripts/calibrate_dynamics.py --neurons 600   # the LIF/activity sweep behind the defaults
python -m fruitfly sample --out data/sample/fafb_v783 --neurons 300  # regenerate the sample
```

Config hashes are recorded in `runs/*/setup.json` (`config_hash`), and every run directory holds
`config.resolved.yaml`, so a number in a doc can be traced to the exact dictionary that made it.
If you cannot reproduce one, that is a bug.

## Data on your machine

`$FAFB_DATA_PATH` → `config/local.yaml` (git-ignored) → `.fafb_path` → a handful of common
download directories (`fruitfly/paths.py` prints the whole `SearchTrail` in `doctor`). Point it
at the folder holding `connections_princeton.csv.gz` etc. — the 68 MB one, not the 277 MB or
212 MB alternates — and start with

```powershell
python scripts/inspect_fafb.py --dir "$env:USERPROFILE\Downloads" --count-rows
```

which writes `fafb_schema_report.md` / `.json` in the current directory (`--out PREFIX` moves
them) and tells you whether the loader's assumed schema matches reality. Expect it to take a
couple of minutes on the real tables; `--count-rows` is the slow part and the only part that can
validate row counts. Add `--include-large` to probe the 2.7 GB synapse table and the 13 GB
skeleton archive.

If a header differs from the documented one, `--write-profile` emits
`config/fafb_profile.yaml` (git-ignored, since it stores local paths) mapping the names the
loader knows to the names your files use; `fruitfly/dataset/schema.py:specs_for` reads it and
promotes those spellings ahead of the built-in aliases. For a spelling nobody has seen, hand-add
the logical field: `columns: {pre: your_column_name}`. That is the supported way to absorb an
export variant without editing library code — and if the variant is real, please also add it to
`CONNECTION_FIELDS` so the next person does not need a profile.
