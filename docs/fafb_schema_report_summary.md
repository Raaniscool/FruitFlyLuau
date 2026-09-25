# What the real FAFB v783 download actually contains

Measured on **2026-09-25** by `scripts/inspect_fafb.py --deep --write-profile` against a real
Codex/FlyWire v783 download (Windows 11, Python 3.14.6, files sitting in a 242-item Downloads
folder). This is a transcription of the machine-generated `fafb_schema_report.md`; the full
report stays out of Git because it embeds local paths and the user's unrelated filenames.

**Headline: every documented reference count is confirmed exactly.** Nothing in the loader's
assumptions had to change. That is a real result — it means `fruitfly/dataset/assets.py`,
written from documentation, describes the bytes correctly.

## Files present (6 of 19 catalogued assets, 71.6 MiB total)

| asset | file | bytes | rows |
|---|---|---|---|
| `connections_filtered` | `connections_princeton.csv.gz` | 68,456,801 | 5,342,446 |
| `nt_predictions` | `neurons.csv.gz` | 1,679,884 | 139,255 |
| `classification` | `classification.csv.gz` | 934,402 | 139,255 |
| `cell_types` | `consolidated_cell_types.csv.gz` | 901,707 | 138,327 |
| `cell_stats` | `cell_stats.csv.gz` | 2,526,548 | 139,246 |
| `connectivity_tags` | `connectivity_tags.csv.gz` | 637,719 | 134,437 |

Absent: `visual_types`, `coordinates`, `names_groups`, the synapse table, skeletons, and all
legacy assets. None of them is needed to build a graph.

## Connection table, measured

| property | measured | documented | agrees |
|---|---:|---:|:--:|
| rows | 5,342,446 | 5,342,446 | yes |
| unique (pre, post) pairs | 3,732,460 | 3,732,460 | yes |
| summed `syn_count` | 50,666,648 | 50,666,648 | yes |
| distinct neuropils | 79 | 79 | yes |
| neurons in `neurons.csv.gz` | 139,255 | 139,255 | yes |

Columns, verbatim: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.

Further measurements with no documented counterpart:

- unique presynaptic ids **137,518**; postsynaptic **130,183**; union **138,584**
- **1,609,986 repeated (pre, post) rows** (30.1% of all rows) — the table is one row per
  (pre, post, **neuropil**). A loader that does not sum over neuropils silently keeps one
  neuropil's synapses for nearly a third of its edges. `merge_duplicates="sum"` is mandatory,
  not an option.
- **0 self-connections**
- **620,180 / 3,732,460 pairs (16.6%) have a reverse counterpart** — the graph is strongly
  directed, not a symmetric matrix in disguise
- `syn_count` ranges 1–2,633 across 628 distinct values

## Hazards that affect the code

1. **Root ids exceed 2^53.** Every id is exactly 18 digits, max `720575940661339776`. At that
   magnitude float64 spacing is 128, so any reader that converts an id column to float — which
   is pandas' default the moment a column has a missing value — silently snaps ids to multiples
   of 128. Read as int64 or str, always. Affects the `root_id` column of all six files.
2. **`nt_type` is missing for 19,658 neurons (14.12%).** The excitatory/inhibitory sign for
   those cells is undetermined by the data, and whatever the model does there is an assumption.
3. **Annotation sparsity**: `hemilineage` 73.0% missing, `nerve` 93.1%, `additional_type(s)`
   89.9%, `class` 22.7%. Only `flow`, `super_class` and `side` are near-complete.
4. `cell_stats` covers 139,246 of 139,255 neurons — nine have no morphometrics.

## Cross-file joins are safe

Taking the connection table's 138,584 nodes as the reference id set:

| asset | % of its ids found in the reference | % of reference it covers |
|---|---:|---:|
| classification | 99.52% | 100.00% |
| nt_predictions | 99.52% | 100.00% |
| cell_stats | 99.52% | 99.99% |
| cell_types | 99.59% | 99.41% |
| connectivity_tags | 100.00% | 97.01% |

All six share one FlyWire root-id namespace. Left-joining on `root_id` is sound. The ~0.5%
of annotated neurons absent from the connection table are cells with no surviving edge after
the ≥5-synapse filter — expected, not an error.

## The find that changes the plan

`classification.class` contains, as literal values, **`Kenyon_Cell`, `MBON`, `MBIN`, `DAN`**
(29 classes total, alongside `CX`, `ALPN`, `LHCENT` and others).

That means the mushroom-body associative-learning circuit — items 1 and 2 of the task shortlist,
and the best candidate for a first genuine learning result — **can be selected with the files
already downloaded**. No extra asset, no hand-curated id list. The `mushroom_body` selector
added in `fruitfly/graph/select.py` does exactly this.

Selecting the circuit the fly uses for learning does **not** make the model biologically
faithful. The dynamics, the plasticity rule and the reward signal remain our constructions.

## Reproduce

```powershell
python scripts/inspect_fafb.py --dir "<your FAFB folder>" --deep --write-profile --out fafb_report_real
```

Runtime on the measured machine: a few minutes, well under 1 GB peak RSS, streaming throughout.
`--deep` is what produces the pair-level facts (duplicate pairs, reciprocity); without it those
lines read `UNKNOWN -- requires further investigation` rather than being guessed.
