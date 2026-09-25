"""Registry of the FlyWire FAFB v783 (Codex) downloadable assets.

Schemas below are transcribed from the Codex "Download Data" portal
documentation for dataset FAFB v783 (mirrored publicly, retrieved 2026-09-23)
and from the FlyWire Zenodo connectivity release record 10676866. They are the
*documented* expectations, NOT a guarantee about the files on this machine: the
loader always sniffs the real header row and adapts (see
``fruitfly.dataset.schema``). Where documentation and reality disagree, reality
wins and ``scripts/inspect_fafb.py`` is what records it.

Portal labels (``display``) differ from file names (``file``); both are kept so
a download can be matched to what the user clicked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class AssetSpec:
    """One downloadable asset of the FAFB v783 release.

    Attributes
    ----------
    key:
        Stable internal identifier.
    display:
        Label shown by the Codex download portal.
    file:
        Canonical file name inside the data directory.
    alternates:
        Other file names observed in the wild for the same content (unofficial
        mirrors sometimes re-export without ``.gz``, or under the Zenodo static
        snapshot names). Tried in order, never guessed silently.
    role:
        What the project uses it for.
    columns:
        Documented column order. Used for validation and for headerless-file
        recovery, and as the *preferred* spelling when several aliases exist.
    required_for:
        Which capabilities stop working without it.
    approx_size:
        Portal-reported size, used by discovery to sanity-check a match.
    large:
        True for assets too big to belong in Git and usually too big to load
        eagerly on a laptop (13 GB skeletons, 2.7 GB synapse table).
    """

    key: str
    display: str
    file: str
    role: str
    columns: tuple[str, ...] = ()
    alternates: tuple[str, ...] = ()
    required_for: tuple[str, ...] = ()
    approx_size: str = ""
    large: bool = False


ASSETS: dict[str, AssetSpec] = {
    # ------------------------------------------------------------------ edges
    "connections_filtered": AssetSpec(
        key="connections_filtered",
        display="Connections (Filtered)",
        file="connections_princeton.csv.gz",
        alternates=(
            "connections_princeton.csv",
            "connections_princeton_filtered.csv.gz",
            "connections_princeton_filtered.csv",
            "proofread_connections_783.feather",
            "proofread_connections_783.csv.gz",
            "connections.csv.gz",
            "connections.csv",
        ),
        role=(
            "Directed weighted graph: one row per (pre, post, neuropil); pairs with "
            "<5 total synapses and autapses are already excluded upstream. Primary "
            "connectivity source for this project."
        ),
        columns=("pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"),
        required_for=("connectome graph", "simulation", "experiments"),
        approx_size="68 MB (68,456,801 bytes, 5,342,446 rows + header)",
    ),
    "connections_unfiltered": AssetSpec(
        key="connections_unfiltered",
        display="Connections (Unfiltered)",
        file="connections_princeton_no_threshold.csv.gz",
        alternates=(
            "connections_princeton_no_threshold.csv",
            "connections_princeton.csv",
        ),
        role="Superset with 1-synapse rows; noisier. Optional alternative source.",
        columns=("pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"),
        approx_size="277 MB (22,285,323 rows)",
        large=True,
    ),
    "connections_legacy": AssetSpec(
        key="connections_legacy",
        display="Connections Predicted With Buhmann Et. Al. [Original Version Used Prior To July 2025]",
        file="connections_buhmann_no_threshold.csv.gz",
        alternates=("connections_buhmann.csv.gz",),
        role="Pre-July-2025 connectivity (Buhmann synapse model). Not used by default.",
        columns=("pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"),
        approx_size="212 MB (16,847,997 rows)",
        large=True,
    ),
    # ------------------------------------------------------------ annotations
    "nt_predictions": AssetSpec(
        key="nt_predictions",
        display="Neurotransmitter Type Predictions",
        file="neurons.csv.gz",
        alternates=("neurons.csv",),
        role=(
            "Per-neuron predicted transmitter + confidence and auto-group. Drives "
            "E/I sign convention and neurotransmitter-based neuron selection."
        ),
        columns=(
            "root_id",
            "group",
            "nt_type",
            "nt_type_score",
            "da_avg",
            "ser_avg",
            "gaba_avg",
            "glut_avg",
            "ach_avg",
            "oct_avg",
        ),
        required_for=("E/I signs", "neurotransmitter selection", "dopamine-like priors"),
        approx_size="1,680 KB (139,255 rows)",
    ),
    "classification": AssetSpec(
        key="classification",
        display="Classification / Hierarchical Annotations",
        file="classification.csv.gz",
        alternates=("classification.csv",),
        role="flow / super_class / class / sub_class / hemilineage / side / nerve.",
        columns=("root_id", "flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"),
        required_for=("classification-based selection",),
        approx_size="934 KB (139,255 rows)",
    ),
    "cell_types": AssetSpec(
        key="cell_types",
        display="Cell Types",
        file="consolidated_cell_types.csv.gz",
        alternates=("consolidated_cell_types.csv", "cell_types.csv.gz"),
        role="Consolidated primary (+additional) cell type per neuron.",
        columns=("root_id", "primary_type", "additional_types"),
        required_for=("neuron-type selection",),
        approx_size="902 KB (138,327 rows)",
    ),
    "cell_stats": AssetSpec(
        key="cell_stats",
        display="Cell Size Measurements",
        file="cell_stats.csv.gz",
        alternates=("cell_stats.csv",),
        role="Cable length / surface area / volume in nanometre units.",
        columns=("root_id", "length_nm", "area_nm", "size_nm"),
        approx_size="2,527 KB (139,246 rows)",
    ),
    "names_groups": AssetSpec(
        key="names_groups",
        display="Proofread Cell Names And Groups",
        file="names.csv.gz",
        alternates=("names.csv",),
        role="Autogenerated per-cell name and 629-way brain-region group.",
        columns=("root_id", "name", "group"),
        approx_size="1,182 KB (139,255 rows)",
    ),
    "visual_types": AssetSpec(
        key="visual_types",
        display="Visual Neuron Annotations",
        file="visual_neuron_types.csv.gz",
        alternates=("visual_neuron_types.csv",),
        role="Optic-lobe visual neuron type/family/subsystem/category/side.",
        columns=("root_id", "type", "family", "subsystem", "category", "side"),
        required_for=("visual-neuron selection",),
        approx_size="632 KB (95,079 rows)",
    ),
    "visual_columns": AssetSpec(
        key="visual_columns",
        display="Visual Neuron Columns",
        file="column_assignment.csv.gz",
        alternates=("column_assignment.csv",),
        role="Retinotopic column id and x/y/p/q coordinates for columnar types.",
        columns=("root_id", "hemisphere", "type", "column_id", "x", "y", "p", "q"),
        approx_size="463 KB (45,528 rows)",
    ),
    "labels_raw": AssetSpec(
        key="labels_raw",
        display="Community Labels (Raw)",
        file="labels.csv.gz",
        alternates=("labels.csv",),
        role="Unprocessed community labels with contributor attribution.",
        columns=(
            "root_id",
            "label",
            "user_id",
            "position",
            "supervoxel_id",
            "label_id",
            "date_created",
            "user_name",
            "user_affiliation",
        ),
        approx_size="4,771 KB (160,045 rows)",
    ),
    "labels_refined": AssetSpec(
        key="labels_refined",
        display="Community Labels (Refined)",
        file="processed_labels.csv.gz",
        alternates=("processed_labels.csv",),
        role="Deduplicated/cleaned community labels; good for text-based selection.",
        columns=("root_id", "processed_labels"),
        approx_size="1,018 KB (100,091 rows)",
    ),
    "connectivity_tags": AssetSpec(
        key="connectivity_tags",
        display="Connectivity Tags",
        file="connectivity_tags.csv.gz",
        alternates=("connectivity_tags.csv",),
        role="Network-analysis descriptors (e.g. broadcaster, integrator).",
        columns=("root_id", "connectivity_tag"),
        approx_size="638 KB (134,437 rows)",
    ),
    "coordinates": AssetSpec(
        key="coordinates",
        display="Marked Neuron Coordinates",
        file="coordinates.csv.gz",
        alternates=("coordinates.csv", "marked_neuron_coordinates.csv.gz"),
        role=(
            "Proofreading anchor positions (nanometres) as an x,y,z string per "
            "row; multiple rows per neuron. Enables spatial visualisation."
        ),
        columns=("root_id", "position", "supervoxel_id"),
        required_for=("spatial visualisation",),
        approx_size="5,315 KB (238,909 rows)",
    ),
    # --------------------------------------------------- deferred heavyweight
    "synapse_table": AssetSpec(
        key="synapse_table",
        display="Synapse Table",
        file="fafb_v783_princeton_synapse_table.csv.gz",
        alternates=("synapse_table.csv.gz", "synapse_table.csv"),
        role=(
            "Per-synapse positions/sizes. NOTE the root-id columns carry a common "
            "prefix in their header (pre_root_id_720575940) and store only the "
            "trailing digits; full id = int(prefix + zero-padded suffix). Not loaded "
            "by default (2.7 GB)."
        ),
        columns=(
            "pre_x", "pre_y", "pre_z",
            "ctr_x", "ctr_y", "ctr_z",
            "post_x", "post_y", "post_z",
            "size", "pre_root_id_720575940", "post_root_id_720575940", "neuropil",
        ),
        approx_size="2,695 MB (80,215,790 rows)",
        large=True,
    ),
    "skeletons": AssetSpec(
        key="skeletons",
        display="Neuron Skeletons",
        file="sk_lod1_783_healed.zip",
        alternates=("skeleton_swc_files.zip",),
        role="SWC skeletons in microns. Not used by the simulator; out of scope for v0.1.",
        approx_size="13 GB",
        large=True,
    ),
    "synapse_coordinates_legacy": AssetSpec(
        key="synapse_coordinates_legacy",
        display="Synapse Coordinates [Original Version Used Prior To July 2025]",
        file="synapse_coordinates.csv.gz",
        alternates=("synapse_coordinates.csv",),
        role=(
            "Legacy single-point synapse coordinates with forward-fill semantics for "
            "empty pre/post id columns. Not loaded by default."
        ),
        columns=("pre_root_id", "post_root_id", "x", "y", "z"),
        approx_size="317 MB (34,156,320 rows)",
        large=True,
    ),
    "attachment_rates_legacy": AssetSpec(
        key="attachment_rates_legacy",
        display="Synapse Attachment Rates [Original Version Used Prior To July 2025]",
        file="synapse_attachment_rates.csv.gz",
        role="Pre/post synapse attachment rates to proofread neurons, by neuropil.",
        approx_size="3 KB",
    ),
    "neuropil_counts_legacy": AssetSpec(
        key="neuropil_counts_legacy",
        display="Per Neuropil Connection And Synapse Counts [Original Version Used Prior To July 2025]",
        file="per_neuropil_counts.csv.gz",
        alternates=("in_out_counts_by_neuropil.csv.gz",),
        role="Wide table: root_id then input/output x synapses/partners x neuropil (321 columns).",
        approx_size="4,675 KB (134,181 rows, 321 cols)",
        large=True,
    ),
}


#: Documented scale of the v783 release, for honest "is my file complete?" checks.
#: These are portal/documentation figures, not measurements of this machine's copy.
#: The only assets without which nothing can run. Everything else enriches the
#: graph (selection by class, visual annotation, coordinates for plots) but the
#: pipeline builds and simulates without it -- measured on a download that has
#: only these two plus classification/cell_types/cell_stats/connectivity_tags.
CORE_ASSETS: tuple[str, ...] = ("connections_filtered", "nt_predictions")

REFERENCE_COUNTS: dict[str, int] = {
    "cells": 139_255,
    "connections_rows_filtered": 5_342_446,
    "unique_pairs": 3_732_460,
    "synapses": 50_666_648,
    "annotations": 1_168_054,
    "neuropils": 79,
    "primary_cell_types": 8_772,
}

#: Neurotransmitter label -> (sign convention, is_modulatory).
#:
#: The sign is an engineering choice that makes a recurrent spiking network
#: tractable, NOT a claim about receptor physiology: glutamatergic/cholinergic
#: rows are treated as net excitation, GABAergic as net inhibition, and
#: bi-directional predictions as mixed (positive). Modulatory transmitters do not
#: get a fixed sign. The simulator's ``ei_mode`` can disable this entirely.
NT_CONVENTION: Mapping[str, tuple[str, bool]] = {
    "glut": ("excitatory", False),
    "glutamatergic": ("excitatory", False),
    "cholin": ("excitatory", False),
    "cholinergic": ("excitatory", False),
    "ach": ("excitatory", False),
    "gaba": ("inhibitory", False),
    "gabaergic": ("inhibitory", False),
    "glut/gaba": ("mixed", False),
    "glut+gaba": ("mixed", False),
    "cholin/gaba": ("mixed", False),
    "da": ("modulatory", True),
    "dopaminergic": ("modulatory", True),
    "ser": ("modulatory", True),
    "serotonergic": ("modulatory", True),
    "oct": ("modulatory", True),
    "octopaminergic": ("modulatory", True),
    "tyr": ("modulatory", True),
    "none": ("unknown", False),
    "nan": ("unknown", False),
    "": ("unknown", False),
}

#: Transmitter columns in ``neurons.csv.gz`` worth caching per neuron.
NT_SCORE_COLUMNS: tuple[str, ...] = ("da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg")

#: Categorical annotation columns available for neuron selection, asset -> columns.
SELECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "classification": ("flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"),
    "cell_types": ("primary_type",),
    "names_groups": ("group", "name"),
    "visual_types": ("type", "family", "subsystem", "category", "side"),
    "connectivity_tags": ("connectivity_tag",),
    "nt_predictions": ("nt_type", "group"),
}


def asset(key: str) -> AssetSpec:
    try:
        return ASSETS[key]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(f"unknown FAFB asset {key!r}; known: {sorted(ASSETS)}") from None


def with_file(key: str, filename: str) -> AssetSpec:
    """Return a copy of an asset spec bound to an actually-found filename."""
    return replace(asset(key), file=filename)
