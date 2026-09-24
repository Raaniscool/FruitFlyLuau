# DATA.md — the FAFB v783 assets this project reads

Everything below is what `fruitfly/dataset/assets.py` encodes as data. The module is
the authority; this file is the human-readable version, including the numbers the
loader checks a local download against.

## What "the connectome" means here

FAFB v783 (FlyWire) is a **structural wiring diagram** of one *Drosophila*
melanogaster brain: proofread neuron skeletons, synapse predictions, and
connection counts per neuron pair. It contains **no dynamics and no learning
rules**. FruitFlyLuau reads the graph, samples a subgraph, and *adds* an LIF
model + plasticity on top. Those additions are this repository's contribution and
its approximations — they are not present in the data. See [LIMITATIONS.md](LIMITATIONS.md).

## Asset registry

`required` means the loader needs it for the features described in the *role*
column; `optional` means the pipeline degrades gracefully and says so in the logs.

| key | portal label | filename | columns (first 5) | loader role |
|---|---|---|---|---|
| `connections_filtered` | Connections (Filtered) | `connections_princeton.csv.gz` | `pre_root_id, post_root_id, neuropil, syn_count, nt_type` | **the graph**: one row per (pair × neuropil), ≥5 synapses |
| `connections_unfiltered` | Connections (Unfiltered) | `connections_princeton_no_threshold.csv.gz` | same | every predicted pair, including singletons |
| `connections_legacy` | Connections … Buhmann Et. Al. [Original …] | `connections_buhmann_no_threshold.csv.gz` | same | previous segmentation vintage |
| `nt_predictions` | Neurotransmitter Type Predictions | `neurons.csv.gz` | `root_id, group, nt_type, nt_score_avg, n_pairs` | E/I sign per neuron |
| `classification` | Classification / Hierarchical Annotations | `classification.csv.gz` | `root_id, flow, super_class, class, sub_class` | annotation-driven selection |
| `cell_types` | Cell Types | `consolidated_cell_types.csv.gz` | `root_id, primary_type, additional_types` | `graph.selection=neuron_type` |
| `cell_stats` | Cell Size Measurements | `cell_stats.csv.gz` | `root_id, length_nm, area_nm, size_nm` | reporting only |
| `names_groups` | Proofread Cell Names And Groups | `names.csv.gz` | `root_id, name, group` | human-readable labels in figures |
| `visual_types` | Visual Neuron Annotations | `visual_neuron_types.csv.gz` | `root_id, type, family, subsystem, category` | `graph.selection=visual` |
| `visual_columns` | Visual Neuron Columns | `column_assignment.csv.gz` | `root_id, hemisphere, type, column_id, x` | column-restricted selections |
| `labels_raw` | Community Labels (Raw) | `labels.csv.gz` | `root_id, label, user_id, position, supervoxel_id` | free-text annotation search |
| `labels_refined` | Community Labels (Refined) | `processed_labels.csv.gz` | `root_id, processed_labels` | as above |
| `connectivity_tags` | Connectivity Tags | `connectivity_tags.csv.gz` | `root_id, connectivity_tag` | cell-type grouping |
| `coordinates` | Marked Neuron Coordinates | `coordinates.csv.gz` | `root_id, position, supervoxel_id` | spatial figures only |
| `synapse_table` | Synapse Table | `fafb_v783_princeton_synapse_table.csv.gz` | `pre_x, pre_y, pre_z, ctr_x, ctr_y, …` | inter-distances, per-synapse detail |
| `skeletons` | Neuron Skeletons | `sk_lod1_783_healed.zip` | — | morphology; **not read by v0.1** |
| `synapse_coordinates_legacy` | Synapse Coordinates [Original …] | `synapse_coordinates.csv.gz` | `pre_root_id, post_root_id, x, y, z` | superseded by `synapse_table` |
| `attachment_rates_legacy` | Synapse Attachment Rates [Original …] | `synapse_attachment_rates.csv.gz` | — | legacy |
| `neuropil_counts_legacy` | Per Neuropil Connection And Synapse Counts [Original …] | `per_neuropil_counts.csv.gz` | — | legacy |

## Sizes and row counts to validate against

These are the numbers a correct download should produce. The loader reports what it
actually found (`fruitfly.dataset.stats.table_stats`), and
`python scripts/inspect_fafb.py --count-rows` writes them to a report you can commit.

| asset | size | rows | notes |
|---|---|---|---|
| `connections_princeton.csv.gz` | 68,456,801 B | 5,342,446 | 137,518 unique `pre_root_id`, 130,183 unique `post_root_id`; `syn_count` spans 1–2,633; pairs totalling <5 synapses are excluded from this file |
| `connections_princeton_no_threshold.csv.gz` | 277 MB | 22,285,323 | every pair, including singletons |
| `connections_buhmann_no_threshold.csv.gz` | 212 MB | 16,847,997 | previous vintage |
| `neurons.csv.gz` | ~7 MB | 139,255 | `nt_type` is empty for 19,658 of them |
| `labels.csv.gz` | — | 160,045 | **110,038 unique root ids**: multiple labels per neuron, so this is not a per-neuron table |
| `coordinates.csv.gz` | — | 238,909 | a *marked* subset, not all 139,255 |
| `fafb_v783_princeton_synapse_table.csv.gz` | 2,695 MB | 80,215,790 | columns are `pre_root_id_720575940` / `post_root_id_720575940` — the shared prefix is baked into the header |
| `sk_lod1_783_healed.zip` | 13 GB | — | skeleton coordinates are in **microns**; synapse coordinates in **nanometres** |

Whole-graph reference counts (`fruitfly.dataset.assets.REFERENCE_COUNTS`):

```
cells 139,255 · unique connected pairs 3,732,460 · synapses 50,666,648
annotations 1,168,054 · neuropils 79 · primary cell types 8,772
```

## Where to get it

- **Codex portal (live service):** `https://codex.flywire.ai/api/download?dataset=fafb`,
  per-asset via `…/api/download?data_product=<id>&dataset=fafb`. Exports support a
  minimum-synapse threshold, top-connection caps and inside/cross-column filters.
  **The portal is not a frozen snapshot** — files can be regenerated, so a changed
  checksum is not proof of corruption. For a reproducible paper-style setup, pin the
  static archive instead.
- **Static archive (preferred for reproducibility):** Zenodo `10676866` (v783.0) holds
  `proofread_connections_783.feather` (`pre_pt_root_id, post_pt_root_id, neuropil,
  syn_count, gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg`) and the 9.5 GB
  `flywire_synapses_783.feather` with `pre/post_pt_position_{x,y,z}` in nm. Zenodo
  `10877326` holds skeletons; `flywire_annotations` holds annotation snapshots.
- **Kaggle mirrors** exist. A mirror reproduces the 5.34 M-row figure only *after*
  aggregating per-pair synapse counts and keeping pairs with ≥5 — the published count
  is a pair count, not a row count for every mirror layout.

## How the loader handles reality

Real downloads vary in container, spelling and dtype, so `fruitfly/dataset/`:

1. **discovers** files by name *and* by header (`discover.py`), one level down, and
   reports `matched / absent / unrecognised` instead of assuming a layout;
2. **aliases** column names (`schema.py`): `pre_pt_root_id`, `pre_root_id`,
   `pre_root_id_720575940` all map to the logical `pre`. Unused columns are reported,
   never silently dropped, and a table with no recognisable id column **fails loudly**
   with the header printed, telling you exactly which alias to add;
3. **sniffs containers**: `.csv`, `.csv.gz`, `.tsv(.gz)`, `.feather`, `.parquet`
   (feather/parquet need `pip install 'fruitflyluau[feather]'`);
4. **reads in chunks** (`data.read_chunksize`, default 1 M rows) and **merges the
   per-neuropil rows into one edge per ordered pair** by summing `syn_count` —
   skipping this step makes the graph ~40 % too dense and double-counts every pair
   that spans two neuropils;
5. **keeps FlyWire root ids** (18 digits, prefix `720575940`) alongside the compacted
   0-based indices used by the simulator, so every edge in every figure is
   traceable back to a real neuron;
6. **caches** the merged result as `data/derived/*.npz` keyed by the source file's
   mtime (`data.use_cache: true`). Delete the cache if you re-download a file.

`data.min_synapses_per_pair` (default 1 here, 5 for the filtered asset) and
`data.max_rows` (0 = all) let you trade fidelity for a fast iteration loop; both are
recorded in every run's `setup.json`.

## Sample data, and what it is not

`data/sample/fafb_v783/` (86 KB) is generated by
`fruitfly.dataset.build_sample.write_sample_fafb_files`. It uses the real asset
filenames, the real column names, real 18-digit root-id shape, real neuropil names
and real transmitter labels, with **random graph content** (a small-world toy brain
of ~300 neurons). It exists for CI, offline development and for eyeballing what the
loader expects. `build_population()` labels any population built from it
`synthetic`, and `setup.json` in every run records `data.synthetic: true`, so a
synthetic result can never be misread as a connectome result.

## Privacy / size rules for this repository

`.gitignore` blocks `*.csv`, `*.csv.gz`, `*.feather`, `*.parquet`, `*.npz`, `*.zip`,
`data/raw/`, `data/fafb/`, `data/derived/`. Never commit a FAFB download, a derived
cache or a run's checkpoints: they are reproducible from the asset list above plus a
seed. `config/local.yaml` and `.fafb_path` (machine-local absolute paths) are
ignored as well.
