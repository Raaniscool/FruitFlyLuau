"""Layered configuration: defaults <- YAML file <- kwargs <- environment.

Everything the simulator can be tuned by lives here, so an experiment is fully
described by one serialisable dict (which is exactly what gets saved next to a
run's results for reproducibility).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .utils import get_logger, stable_hash

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    fafb_data_path: str = ""  # empty -> auto-discovery (see fruitfly.paths)
    source: str = "connections_filtered"  # asset key, or "sample" for the tiny fixture
    min_synapses_per_pair: int = 1
    """Aggregate threshold applied to the (pre,post) pair sum.

    The Codex 'Connections (Filtered)' file already excludes pairs with <5 total
    synapses, so 1 keeps every row of that file. Use 5+ when loading the
    unfiltered/legacy tables.
    """
    max_rows: int = 0
    """Read at most this many connection rows (0 = no cap). Useful on a laptop."""
    read_chunksize: int = 1_000_000
    use_cache: bool = True
    cache_dir: str = "data/derived"


@dataclass
class GraphConfig:
    mode: str = "tiny"  # tiny | small | medium | full | sample
    n_neurons: int = 100
    selection: str = "connected_subgraph"
    selection_kwargs: dict[str, Any] = field(default_factory=dict)
    weight_transform: str = "log1p"  # none | log1p | sqrt | zscore
    weight_scale: float = 1.0
    ei_mode: str = "nt"  # "nt" -> sign from predicted transmitter, "all_exc" -> ignore signs
    allow_reciprocal: bool = True
    counterbalance_init: bool = True
    """Search the readout group->class mapping and adopt the one the UNTRAINED
    network performs worst on (see fruitfly/experiment/readout.py). Prevents
    'learning' that is really pre-existing wiring luck."""
    counterbalance_probe_items: int = 24
    counterbalance_maximize: bool = False  # True = pick the EASIEST mapping (control)


@dataclass
class LIFConfig:
    """Default dynamics, chosen by the parameter sweep in scripts/calibrate_dynamics.py.

    These values were NOT picked for biological realism: the sweep searched
    input amplitude x synaptic gain x membrane tau x adaptation and scored each
    setting on measured active fraction, saturation and readout separability on a
    1,000-neuron population. This row was the best compromise
    (separation 0.62, saturation 0.00, 44% of neurons active, ~19-55 readout
    spikes/trial). Re-run the script to re-derive them after changing anything
    structural.
    """

    dt_ms: float = 0.5
    tau_ms: float = 6.0
    v_rest: float = -62.0
    v_thresh: float = -52.0
    v_reset: float = -62.0
    refractory_ms: float = 2.0
    leak: float = 1.0
    #: mV of drive per unit of (signed, log1p-scaled) synaptic weight
    synaptic_gain: float = 2.0
    tau_syn_ms: float = 2.0
    reversal_mode: str = "current"  # current | conductance
    e_rev_exc: float = 0.0
    e_rev_inh: float = -75.0
    noise: float = 0.0  # std of Gaussian background current (mV per step)
    background_current: float = 0.0
    adaptation: float = 0.0  # per-spike threshold adaptation (mV), decays with tau 100ms
    max_rate_guard: float = 0.0  # reserved: rate-based refractory extension (not implemented)
    clip_v: float = 40.0  # hard clamp on |V - v_rest| to keep the integrator stable


@dataclass
class PlasticityConfig:
    rule: str = "reward_stdp"  # stdp | reward_stdp | hebbian | none
    eta: float = 1e-3
    a_plus: float = 0.2
    a_minus: float = 0.31
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    tau_trace_ms: float = 40.0  # eligibility trace decay
    w_min: float | None = None
    """Lower bound on weight. ``None`` = auto: ``-w_max`` when the graph has
    inhibitory (negative) edges, else ``0``."""
    w_max: float = 8.0
    polarity_lock: bool = True
    """Keep each edge's sign fixed during learning (a Dale-like engineering
    constraint, not a biological claim). Set False to allow sign flips."""
    weight_decay: float = 0.0
    normalize: str = "colsum"
    """Weight renormalisation applied at the end of each trial:
    ``none`` | ``rowsum`` (constant outgoing strength per source neuron) |
    ``colsum`` (constant incoming strength per target neuron) | ``both``.

    ``colsum`` matters: without post-synaptic competition, a run of punishments
    simply depresses every active edge until the readout goes silent, and the
    decision can never flip. With it, weakening the winning pathway necessarily
    strengthens the rival's share -- zero-sum competition inside the connectome.
    """
    center_eligibility: bool = True
    """Subtract the mean eligibility over plastic edges before applying it, so
    credit is *relative* causality rather than co-activity (a covariance-style
    three-factor rule). Without centering, both readout groups receive the same
    signed push and a biased initialisation cannot be escaped (measured)."""
    plastic_targets: str = "all"
    """Which edges may change: ``all`` (whole population) or ``output`` (only
    edges synapsing onto the readout population). ``output`` localises learning to
    the decision stage and is far cheaper on large graphs; both are reportable."""
    plastic_fraction: float = 1.0
    """Plasticise this fraction of edges (deterministic stride). Keeps the full
    connectome runnable: 3.7M edges x per-step updates is the hot loop."""
    seed: int = 1234


@dataclass
class RewardConfig:
    positive: float = 1.0
    negative: float = -1.0
    neutral: float = 0.0
    delay_steps: int = 0
    """Credit delay, in simulation steps. Models delayed reinforcement; 0 = immediate."""
    trace_decay: float = 0.9
    """Dopamine-like trace decay per step (modulatory state, not dopamine itself)."""
    trace_tau_ms: float = 60.0
    """Modulator time constant. Must stay comparable to a few trials: a long tau
    makes the trace accumulate across trials and silently ramps the effective
    learning rate (measured during development: tau=200 ms -> |DA| ~ 6 at eta 0.01)."""
    value_abs_max: float = 4.0
    """Numerical clamp on the modulatory trace magnitude."""
    penalize_no_response: bool = True
    baseline_subtract: bool = False
    """Subtract an exponential baseline (reward-prediction-error flavour).

    Measured caveat, kept in the repo rather than hidden: with a constant-outcome
    regime (e.g. a counterbalanced exp001 where the untrained network is wrong on
    every trial) the RPE baseline converges onto the reward itself, the modulatory
    signal collapses to ~0, and plasticity stops entirely. The default is therefore
    the raw trace; prediction-error centring is an ablation to run deliberately.
    """
    credit_mode: str = "auto"
    """How a trial's outcome reaches plasticity:
    ``online`` -> delivered during the post-trial window and applied step-by-step;
    ``end_of_trial`` -> accumulated eligibility multiplied by the modulator once at
    the end of the trial; ``auto`` -> end_of_trial when delay_steps > 0 else online."""
    max_history: int = 10_000


@dataclass
class TrainConfig:
    episodes: int = 200
    steps_per_trial: int = 0
    """Stimulus-window length in simulation steps. 0 = use the environment's
    ``trial_steps`` (48 by default, from scripts/calibrate_dynamics.py). Shorter
    windows leave the readout starved: activity needs several hops of synaptic
    delay to reach the sink neurons, which silently looks like "no learning"."""
    batch_trials: int = 1
    eval_every: int = 20
    eval_trials: int = 40
    seed: int = 7
    shuffle_each_epoch: bool = True
    log_dir: str = "runs"
    save_checkpoints: bool = True
    checkpoint_every: int = 50
    keep_last: int = 3
    early_stop_accuracy: float = 0.99
    post_steps: int = -1
    """Unstimulated steps run after each trial so a *delayed* reward can land on
    still-decaying eligibility traces. -1 = auto (reward delay + 5)."""
    control: str = ""
    """Experiment controls: "" (none) | no_plasticity | shuffled_wiring | sham_reward.
    A positive result without a control run is not a result; see EXPERIMENTS.md."""
    eval_before_training: bool = True
    window: int = 25
    """Sliding-window size for reported accuracy/reward curves."""
    max_wallclock_minutes: float = 0.0
    """Soft stop: finish the current episode, then stop and checkpoint. 0 = no limit."""
    device: str = "cpu"  # "cpu" | "cuda" (sparse GPU support is opt-in, see ARCHITECTURE.md)


@dataclass
class VizConfig:
    enabled: bool = True
    dpi: int = 110
    max_edges_drawn: int = 4000
    activity_raster_neurons: int = 40


@dataclass
class LuauConfig:
    """Phase 7+ only. Nothing in phases 1-6 reads these fields."""

    tokenizer: str = "luau-v0"
    vocab_max: int = 4000
    eval_timeout_s: float = 2.0
    eval_mem_mb: int = 256
    sandbox: str = "none"  # none | subprocess | docker -- see fruitfly/luau/evaluator.py
    enabled: bool = False


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    lif: LIFConfig = field(default_factory=LIFConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    luau: LuauConfig = field(default_factory=LuauConfig)
    experiment: str = "exp001_binary"
    encoder: dict[str, Any] = field(default_factory=dict)
    """Input encoder override, e.g. {kind: sequence, kwargs: {amplitude: 4}}.
    Empty = take the environment's own encoder_kind."""
    decoder: dict[str, Any] = field(default_factory=dict)
    """Output decoder override, e.g. {kind: token, kwargs: {window_steps: 5}}."""
    name: str = ""
    notes: str = ""

    # ------------------------------------------------------------- plumbing
    def to_dict(self) -> dict[str, Any]:
        def conv(obj: Any) -> Any:
            if is_dataclass(obj):
                return {f.name: conv(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(v) for v in obj]
            return obj

        return conv(self)

    @property
    def config_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AppConfig":
        cfg = cls()
        for key, value in d.items():
            if key == "_comment":
                continue
            if not hasattr(cfg, key):
                log.warning("ignoring unknown config key %r", key)
                continue
            section = getattr(cfg, key)
            if is_dataclass(section) and isinstance(value, Mapping):
                for k2, v2 in value.items():
                    if hasattr(section, k2):
                        setattr(section, k2, v2)
                    else:
                        log.warning("ignoring unknown config key %s.%s", key, k2)
            else:
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "AppConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(_expand_env(raw))

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        use_defaults_file: bool = True,
    ) -> "AppConfig":
        """``config/default.yaml`` <- ``config/local.yaml`` <- ``path`` <- overrides <- env."""
        merged: dict[str, Any] = {}
        for candidate in (REPO_ROOT / "config" / "default.yaml", REPO_ROOT / "config" / "local.yaml"):
            if use_defaults_file and candidate.is_file():
                merged = _deep_merge(merged, _load_yaml(candidate))
        if path is not None:
            p = Path(path)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.is_file():
                merged = _deep_merge(merged, _load_yaml(p))
            elif not p.name.startswith("<"):
                raise FileNotFoundError(f"config file not found: {p}")
        merged = _deep_merge(merged, dict(overrides or {}))
        env_path = os.environ.get("FAFB_DATA_PATH")
        if env_path:
            merged.setdefault("data", {})["fafb_data_path"] = env_path
        cfg = cls.from_dict(_expand_env(merged))
        if cfg.data.fafb_data_path == "":
            from .paths import find_data_dir

            found, _ = find_data_dir()
            if found is not None:
                cfg.data.fafb_data_path = str(found)
        return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_ENV_RE = None


def _expand_env(obj: Any) -> Any:
    """Expand ``${VAR}`` / ``$VAR`` inside string config values."""
    global _ENV_RE
    if _ENV_RE is None:
        import re

        _ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

    def sub(value: str) -> str:
        def repl(m: re.Match[str]) -> str:  # type: ignore[name-defined]
            return os.environ.get(m.group(1) or m.group(2), "")

        return _ENV_RE.sub(repl, value)

    if isinstance(obj, str):
        return sub(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
