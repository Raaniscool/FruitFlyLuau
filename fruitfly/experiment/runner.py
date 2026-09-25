"""Experiment runner: the only place where all subsystems meet.

Per trial:

1. encode the observation into a sparse current schedule and run the network for
   ``trial_steps`` steps, applying the plasticity rule every step with the current
   dopamine-like modulator value;
2. decode the readout population's spike counts into a prediction;
3. score it against the target, deliver the (possibly delayed) reward, then run a
   short unstimulated post-window so a delayed signal can still land on
   decaying eligibility traces;
4. log reward, error, activity and weight-change statistics.

No answer keys, no per-question storage: the only state that persists between
trials is the synaptic weight array (plus eligibility traces, modulator and
reward history, all of which are reported).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import AppConfig
from ..graph.build import build_population
from ..graph.connectome import Connectome
from ..io.decoder import Decision, build_decoder
from ..io.encoder import build_encoder
from ..learning.modulator import DopamineLikeModulator
from ..learning.reward import RewardSystem
from ..neuro.network import NetworkSimulator
from ..neuro.plasticity import PlasticityContext, build_rule
from ..utils import ensure_dir, get_logger, seed_all, stable_hash, write_json
from .base import Environment, Trial
from .checkpoint import load_checkpoint, restore_connectome, save_checkpoint
from .metrics import MetricsLogger, RunningStats, binom_p_above_chance, chance_level, summarize_curve

log = get_logger(__name__)


@dataclass
class TrialResult:
    episode: int
    prediction: Any
    target: Any
    correct: bool
    reward: float
    n_spikes: int
    mean_rate_hz: float
    margin: float
    n_active_output: int
    edges_updated: int
    abs_dw: float
    key: str = ""


@dataclass
class TrainOutcome:
    """Everything a report needs, all measured rather than declared."""

    run_dir: Path
    config: dict
    n_neurons: int
    n_edges: int
    n_episodes: int
    train_accuracy_window: float | None
    eval_before: dict
    eval_after: dict
    delta_eval_accuracy: float | None
    reward_curve: dict
    accuracy_curve: dict
    weight_stats: dict
    plasticity_diagnostics: dict
    activity: dict
    timings: dict
    controls: dict
    learning_detected: bool
    claim_bounds: str
    learning_criterion: dict = field(default_factory=dict)
    checkpoint_path: str | None = None

    def summary_text(self) -> str:
        eb, ea = self.eval_before, self.eval_after
        lines = [
            "=" * 74,
            f"EXPERIMENT {self.config.get('experiment')}  ({self.config.get('scale_label')})",
            f"population      : {self.n_neurons:,} neurons, {self.n_edges:,} edges",
            f"episodes        : {self.n_episodes:,}",
            f"eval BEFORE     : acc={eb.get('accuracy')!r} chance={eb.get('chance')!r} p={eb.get('p_value_vs_chance')!r}",
            f"eval AFTER      : acc={ea.get('accuracy')!r} chance={ea.get('chance')!r} p={ea.get('p_value_vs_chance')!r}",
            f"delta accuracy  : {self.delta_eval_accuracy}",
            f"train window acc: {self.train_accuracy_window}",
            f"weights changed : {self.weight_stats.get('n_edges_changed_at_least_once'):,} edges, "
            f"total |dw|={self.weight_stats.get('total_abs_dw'):.4g}",
            f"activity        : {self.activity.get('spikes_per_trial_mean'):.1f} spikes/trial mean, "
            f"{self.activity.get('active_fraction_mean'):.2f} of neurons ever active",
            f"learning detected by our criterion: {self.learning_detected}",
            f"claim bounds    : {self.claim_bounds}",
            "=" * 74,
        ]
        return "\n".join(str(x) for x in lines)


class ExperimentRunner:
    """Owns one run: population, simulator, rule, reward ledger, logging."""

    def __init__(
        self,
        cfg: AppConfig,
        env: Environment,
        *,
        population: Any | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.population = population
        self.rng = np.random.default_rng(cfg.train.seed)
        self.run_dir = Path(run_dir) if run_dir else self._default_run_dir()
        self.started = time.perf_counter()
        self._episodes_done = 0
        self.weight_history: list[tuple[int, float, float]] = []
        self.activity_records: list[dict] = []

    # ------------------------------------------------------------------ setup
    @property
    def _tag(self) -> str:
        c = self.cfg
        return f"{c.experiment}_{c.graph.mode}_n{c.graph.n_neurons}_seed{c.train.seed}_{c.config_hash[:8]}"

    def _default_run_dir(self) -> Path:
        base = Path(self.cfg.train.log_dir)
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[2] / base
        return base / self._tag

    def setup(self, *, resume_from: str | Path | None = None) -> None:
        ensure_dir(self.run_dir)
        seed_all(self.cfg.train.seed)
        if self.population is None:
            self._warn_if_expensive()
            self.population = build_population(self.cfg, seed=self.cfg.train.seed)
        self.conn: Connectome = self.population.connectome
        self._apply_controls()

        from ..graph.select import input_output_sets

        n_in = int(self.cfg.graph.selection_kwargs.get("n_inputs", 16))
        n_out = int(self.cfg.graph.selection_kwargs.get("n_outputs", 2 * len(self.env.classes)))
        inputs, outputs, self.readout_audit = input_output_sets(
            self.conn, n_inputs=n_in, n_outputs=n_out, rng=self.rng,
            depth=int(self.cfg.graph.selection_kwargs.get("drive_depth", 3)),
        )
        log.info(
            "readout assignment: %d input cells, %d output cells over %d neurons (%s; %s ancestors within depth %d)",
            inputs.size, outputs.size, self.conn.n_neurons, self.readout_audit["input_source"],
            f"{self.readout_audit['n_ancestors_within_depth']:,}", self.readout_audit["depth"],
        )

        enc_spec: dict[str, Any] = dict(self.cfg.encoder or {"kind": self.env.encoder_kind})
        enc_kwargs = dict(getattr(self.env, "encoder_kwargs", {}) or {})
        enc_kwargs.update(dict(enc_spec.pop("kwargs", {})))
        enc_kwargs.setdefault("duration_steps", self.trial_steps)
        enc_kwargs.setdefault("n_input_neurons", int(inputs.size))
        enc = build_encoder({**enc_spec, **enc_kwargs})
        enc.configure(self.conn, self.rng, vocab=getattr(self.env, "symbols", None))
        enc.input_neurons = inputs
        self.encoder = enc

        dec_spec: dict[str, Any] = dict(self.cfg.decoder or {"kind": self.env.decoder_kind})
        dec_kwargs = dict(dec_spec.pop("kwargs", {}))
        dec = build_decoder({**dec_spec, **dec_kwargs})
        dec.configure(self.conn, self.rng, classes=list(self.env.classes))
        dec.output_neurons = outputs
        self.decoder = dec

        self.sim = NetworkSimulator(self.conn, self.cfg.lif, seed=self.cfg.train.seed + 1)
        self.rule = build_rule(self.cfg.plasticity, self.conn)
        self._restrict_plastic_targets()
        self.reward = RewardSystem(self.cfg.reward)
        self.mod = DopamineLikeModulator(self.cfg.reward, dt_ms=self.cfg.lif.dt_ms)
        self.w0 = self.conn.copy_weights()
        self.mapping_search = self._counterbalance_readout()
        self.metrics = MetricsLogger(self.run_dir / "metrics.jsonl")
        self._write_setup(resume_from)
        if resume_from is not None:
            self.resume(resume_from)

    def _counterbalance_readout(self):
        """Adopt the readout mapping the untrained network is worst at (see module docs)."""
        from .readout import search_mapping

        n = int(self.cfg.graph.counterbalance_probe_items)
        items = list(self.env.train_items())[: max(4, n)]
        res = search_mapping(
            self, items,
            enabled=bool(self.cfg.graph.counterbalance_init),
            maximize=bool(self.cfg.graph.counterbalance_maximize),
        )
        if res.performed:
            self.decoder.classes = tuple(self.env.classes[i] for i in res.chosen)
            log.info("%s", res.describe())
        return res

    def _restrict_plastic_targets(self) -> None:
        """Apply ``plasticity.plastic_targets`` by masking the rule's edge set.

        ``"output"`` keeps learning local to the decision stage (edges synapsing
        onto readout neurons). It is not a cheat: those are real synapses of real
        neurons in the loaded connectome, and the mask is recorded in the config so
        any reviewer can see which edges were allowed to change.
        """
        mode = (self.cfg.plasticity.plastic_targets or "all").strip().lower()
        st = self.rule.state
        if mode in {"all", ""}:
            st.plastic_targets = "all"
            return
        if mode != "output":
            raise ValueError(f"plasticity.plastic_targets must be 'all' or 'output', got {mode!r}")
        targets = np.zeros(self.conn.n_neurons, dtype=bool)
        targets[self.decoder.output_neurons] = True
        st.plastic_mask = st.plastic_mask & targets[st.edge_post]
        st.plastic_targets = "output"
        log.info(
            "plasticity restricted to edges onto the %d readout neurons: %s of %s edges",
            self.decoder.output_neurons.size, f"{int(st.plastic_mask.sum()):,}", f"{st.n_edges:,}",
        )
        if not st.plastic_mask.any():
            raise RuntimeError(
                "plastic_targets='output' selected no edges: the readout population has no "
                "incoming synapses in this subgraph. Use selection mode 'connected_subgraph' "
                "or plastic_targets='all'."
            )

    @property
    def trial_steps(self) -> int:
        return int(self.cfg.train.steps_per_trial or self.env.trial_steps)

    def _warn_if_expensive(self) -> None:
        c = self.cfg
        n = {"tiny": 100, "small": 1000, "medium": 10000, "large": 40000}.get(c.graph.mode.lower(), c.graph.n_neurons)
        est_steps = c.train.episodes * (c.train.steps_per_trial or 16)
        cost = est_steps * max(1, n) * 30  # ~30 O(n) vector ops per step
        if n >= 10_000 or est_steps >= 20_000:
            log.warning(
                "this run looks expensive: ~%s neurons, %s steps total (est. %.1f s of "
                "vector work on one CPU thread, more on a laptop). Consider "
                "graph.mode=tiny/small, fewer episodes, or plasticity.plastic_fraction<1.",
                f"{n:,}", f"{est_steps:,}", cost / 1e6,
            )
        if c.graph.mode.lower() == "full" and c.plasticity.rule not in {"none", "frozen"}:
            log.warning(
                "full connectome + plasticity: eligibility arrays are ~%.1f MB each and the "
                "hot loop is over %s edges per step.",
                3 * 1.4e6 * 4 / 1e6, "3.7M",
            )

    def _apply_controls(self) -> None:
        """Negative controls, applied to the object graph, not just logged."""
        ctrl = (self.cfg.train.control or "").strip().lower()
        self.control_note = "none"
        if ctrl in {"", "none"}:
            return
        if ctrl in {"no_plasticity", "no-learning", "frozen"}:
            self.cfg.plasticity.rule = "none"
            self.control_note = "no_plasticity: rule forced to 'none' (weights frozen)"
        elif ctrl in {"shuffled", "shuffled_wiring", "shuffle"}:
            # destroy WHICH targets each neuron talks to, keeping each neuron's
            # out-degree and the global weight multiset: the standard
            # shuffled-connectivity control.
            m = self.conn.matrix
            new_idx = np.asarray(m.indices).copy()
            n = m.shape[1]
            indptr = m.indptr
            for i in range(m.shape[0]):
                lo, hi = int(indptr[i]), int(indptr[i + 1])
                if hi - lo > 1:
                    new_idx[lo:hi] = self.rng.choice(n, size=hi - lo, replace=False)
            m.indices[:] = new_idx
            m.sort_indices()
            self.control_note = "shuffled_wiring: CSR target indices re-randomised per source neuron"
        elif ctrl in {"sham_reward", "sham"}:
            self.control_note = "sham_reward: reward sign randomised, independent of correctness"
        else:
            raise ValueError(f"unknown control {ctrl!r} (use no_plasticity|shuffled_wiring|sham_reward)")
        log.warning("CONTROL ACTIVE -> %s", self.control_note)

    def _write_setup(self, resume_from: Any) -> None:
        payload = {
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash,
            "experiment": self.env.describe(),
            "population": {
                "describe": self.population.describe(),
                "simulated_neurons": self.population.simulated_neurons,
                "requested_neurons": self.population.requested_neurons,
                "scale_label": self.population.scale_label,
                "source": self.population.source_description,
                "connection_table": self.population.connection_table_stats,
                "stats": self.conn.stats(),
                "provenance": self.conn.provenance,
                "timings_s": {k: round(v, 3) for k, v in self.population.timings.items()},
            },
            "readout": {
                "input_neurons": self.encoder.input_neurons.tolist() if self.encoder.input_neurons is not None else [],
                "output_neurons": self.decoder.output_neurons.tolist() if self.decoder.output_neurons is not None else [],
                "trial_steps": self.trial_steps,
                "assignment_audit": getattr(self, "readout_audit", {}),
            },
            "data": {
                "source": self.cfg.data.source,
                "fafb_data_path": self.cfg.data.fafb_data_path,
                "synthetic": bool(self.cfg.data.source == "sample" or self.cfg.graph.mode in {"sample", "synthetic"}),
            },
            "control": getattr(self, "control_note", "none"),
            "readout_mapping": {
                "performed": getattr(self, "mapping_search", None) is not None and self.mapping_search.performed,
                "chosen_permutation": list(getattr(self, "mapping_search", None).chosen or []) if getattr(self, "mapping_search", None) else [],
                "initial_accuracy": getattr(self, "mapping_search", None).best_accuracy if getattr(self, "mapping_search", None) else None,
                "identity_accuracy": getattr(self, "mapping_search", None).identity_accuracy if getattr(self, "mapping_search", None) else None,
                "reason": getattr(self, "mapping_search", None).reason if getattr(self, "mapping_search", None) else "",
                "table": getattr(self, "mapping_search", None).table[:24] if getattr(self, "mapping_search", None) else [],
                "decoder_classes_after": [str(c) for c in self.decoder.classes],
            },
            "plasticity_mask": {
                "plastic_targets": getattr(self.rule.state, "plastic_targets", "all"),
                "plastic_edges": int(self.rule.state.plastic_mask.sum()),
                "n_edges": int(self.rule.state.n_edges),
            },
            "resumed_from": str(resume_from) if resume_from else None,
        }
        write_json(self.run_dir / "setup.json", payload)
        try:
            import yaml

            (self.run_dir / "config.resolved.yaml").write_text(
                yaml.safe_dump(self.cfg.to_dict(), sort_keys=True), encoding="utf-8"
            )
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------- trial
    def _inject(self, plan: Any, t: int) -> np.ndarray | None:
        idx, amp = plan.per_step[t] if t < len(plan.per_step) else (np.empty(0, np.int64), np.empty(0, np.float64))
        if idx.size == 0:
            return None
        cur = np.zeros(self.conn.n_neurons, dtype=np.float64)
        cur[idx] = amp
        return cur

    def run_trial(self, trial: Trial, *, learn: bool, episode: int = -1) -> TrialResult:
        """One trial: drive -> measure -> score -> (delayed) reinforcement.

        The input phase and a short post-window are separated deliberately: the
        outcome of a trial can only be known after its readout, so reinforcement
        arrives at the start of the post-window (or ``delay_steps`` later) and is
        applied to eligibility traces that are still decaying. That ordering is the
        whole point of the eligibility mechanism, and the earlier single-loop
        version silently never modulated anything (|dW| = 0) -- caught by
        tests/test_plasticity.py::test_reward_reaches_weights.
        """
        cfg = self.cfg
        p = cfg.lif
        plan = self.encoder.encode(trial.observation, n_steps=self.trial_steps)
        self.sim.reset()
        if learn:
            self.rule.begin_trial()
        n = self.conn.n_neurons
        counts = np.zeros(n, dtype=np.int64)
        step_metrics = {"edges_updated": 0, "abs_dw_sum": 0.0}
        credit_mode = self._credit_mode() if learn else "none"

        def one_step(with_input: bool, t: int) -> np.ndarray:
            spikes = self.sim.step(self._inject(plan, t) if with_input else None)
            np.add(counts, spikes, out=counts)  # in-place: avoids a closure rebinding
            if not learn:
                return spikes
            due = self.reward.pop(self.sim.t_step - 1)
            if due or self.mod.current:
                self.mod.deliver(due)
            if credit_mode == "end_of_trial":
                # trace building only: the value that scales it has not arrived yet
                self.rule.accumulate(
                    PlasticityContext(
                        t_step=self.sim.t_step, trial=episode, spikes=spikes,
                        spikes_bool=np.asarray(spikes, dtype=bool),
                        modulator=0.0, dt_ms=p.dt_ms,
                    )
                )
            if credit_mode == "online":
                m = self.rule.apply(
                    PlasticityContext(
                        t_step=self.sim.t_step, trial=episode, spikes=spikes,
                        spikes_bool=np.asarray(spikes, dtype=bool),
                        modulator=self.mod.current, dt_ms=p.dt_ms,
                    )
                )
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        step_metrics[k] = step_metrics.get(k, 0.0) + v
            return spikes

        # --- phase 1: stimulus
        for t in range(plan.n_steps):
            one_step(True, t)
        readout_counts = counts.copy()
        if learn and not hasattr(self, "_silent_warned"):
            n_in_readout = int((readout_counts[self.decoder.output_neurons] > 0).sum())
            if n_in_readout == 0:
                log.warning(
                    "readout population received NO spikes during the %d-step stimulus window "
                    "(population activity: %s spikes over %d neurons). The task is currently "
                    "undriveable: increase train.steps_per_trial / lif gains, or use a selection "
                    "mode whose subgraph actually connects inputs to outputs.",
                    plan.n_steps, f"{int(counts.sum()):,}", self.conn.n_neurons,
                )
                self._silent_warned = True

        # --- phase 2: decision on the input-phase activity only
        decision: Decision = self.decoder.decode(readout_counts)
        pred = decision.label
        correct = self.env.is_correct(pred, trial.target)

        reward_value = 0.0
        if learn:
            reward_value = self._deliver_outcome(correct, episode)
            # --- phase 3: reinforcement window (no stimulus): delayed credit lands here
            for _ in range(self._post_steps()):
                one_step(False, -1)
            if credit_mode == "end_of_trial":
                due = self.reward.pop(self.sim.t_step) or 0.0
                self.mod.deliver(due)
                m = self.rule.apply_at_trial_end(self.mod.current)
                step_metrics["edges_updated"] += int(m.get("edges_updated", 0))
                step_metrics["abs_dw_sum"] += float(m.get("abs_dw_sum", 0.0))
                step_metrics["credit_scale"] = float(m.get("credit_scale", 0.0))
            leftover = self.reward.end_trial(correct=correct)
            if leftover:
                # credit that would have arrived after the last post-window step
                self.mod.deliver(leftover)
                if credit_mode == "end_of_trial":
                    m = self.rule.apply_at_trial_end(self.mod.current)
                    step_metrics["edges_updated"] += int(m.get("edges_updated", 0))
                    step_metrics["abs_dw_sum"] += float(m.get("abs_dw_sum", 0.0))
            self.rule.end_trial()

        total_spikes = int(counts.sum())
        dur_s = (plan.n_steps + (self._post_steps() if learn else 0)) * p.dt_ms / 1000.0
        rec = TrialResult(
            episode=episode, prediction=pred, target=trial.target, correct=bool(correct),
            reward=float(reward_value), n_spikes=total_spikes,
            mean_rate_hz=total_spikes / max(n, 1) / max(dur_s, 1e-9),
            margin=float(decision.margin),
            n_active_output=int((readout_counts[self.decoder.output_neurons] > 0).sum()),
            edges_updated=int(step_metrics.get("edges_updated", 0)),
            abs_dw=float(step_metrics.get("abs_dw_sum", 0.0)),
            key=trial.key,
        )
        if episode >= 0:  # keep training activity separate from evaluation passes
            self.activity_records.append(
                {"episode": episode, "n_spikes": total_spikes, "active": int((readout_counts > 0).sum())}
            )
            if len(self.activity_records) > 400:
                del self.activity_records[: len(self.activity_records) - 400]
        return rec

    def _credit_mode(self) -> str:
        cm = (self.cfg.reward.credit_mode or "auto").lower()
        if cm == "auto":
            return "end_of_trial" if self.cfg.reward.delay_steps > 0 else "online"
        if cm not in {"online", "end_of_trial"}:
            raise ValueError(f"reward.credit_mode must be auto|online|end_of_trial, got {cm!r}")
        return cm

    def _post_steps(self) -> int:
        v = int(self.cfg.train.post_steps)
        if v < 0:
            return int(self.cfg.reward.delay_steps) + 6
        return v

    def _deliver_outcome(self, correct: bool, episode: int) -> float:
        ctrl = (self.cfg.train.control or "").strip().lower()
        if ctrl in {"sham_reward", "sham"}:
            sham = bool(self.rng.random() < 0.5)
            val = self.reward.value_for(sham)
            self.reward.deliver(val, step=self.sim.t_step, correct=correct)
            return float(val)
        val = self.reward.value_for(correct)
        self.reward.deliver(val, step=self.sim.t_step, correct=correct)
        return float(val)

    # ------------------------------------------------------------------- train
    def train(self, *, episodes: int | None = None, resume: bool = False) -> TrainOutcome:
        n = int(episodes if episodes is not None else self.cfg.train.episodes)
        start_ep = 0
        if resume:
            start_ep = int(self.metrics.series("episode")[-1]) + 1 if self.metrics.records else 0
        eval_before = self.evaluate(split="eval", tag="before") if self.cfg.train.eval_before_training else {"accuracy": None}

        acc = RunningStats()
        rw = RunningStats()
        iterator = self.env.epochs(n)
        t0 = time.perf_counter()
        stopped_early = None
        for i in range(n):
            trial = next(iterator)
            rec = self.run_trial(trial, learn=True, episode=start_ep + i)
            acc.update(1.0 if rec.correct else 0.0)
            rw.update(rec.reward)
            self._last_reward = rec.reward
            w = self.conn.weights
            if (start_ep + i) % 10 == 0 or i == n - 1:
                self.weight_history.append((start_ep + i, float(w.mean()), float(w.std())))
            self.metrics.log(
                episode=start_ep + i, correct=rec.correct, reward=rec.reward, pred=str(rec.prediction),
                target=str(rec.target), n_spikes=rec.n_spikes, rate=rec.mean_rate_hz, margin=rec.margin,
                edges_updated=rec.edges_updated, abs_dw=rec.abs_dw,
                weight_mean=float(w.mean()), weight_std=float(w.std()),
                modulator=self.mod.current, acc_window=acc.mean,
            )
            if (i + 1) % max(1, self.cfg.train.checkpoint_every) == 0:
                self._save(tag=f"step{start_ep + i + 1:07d}")
            if self.cfg.train.early_stop_accuracy and acc.n >= max(20, self.cfg.train.window) and acc.mean >= self.cfg.train.early_stop_accuracy:
                stopped_early = f"train window accuracy {acc.mean:.3f} >= {self.cfg.train.early_stop_accuracy} at episode {start_ep + i + 1}"
                log.info("early stop: %s", stopped_early)
                break
            if self.cfg.train.max_wallclock_minutes and (time.perf_counter() - t0) / 60 > self.cfg.train.max_wallclock_minutes:
                stopped_early = f"wallclock limit {self.cfg.train.max_wallclock_minutes} min reached"
                log.warning("stopping: %s", stopped_early)
                break
        self.metrics.flush()
        if self.cfg.train.save_checkpoints:
            self._save(tag="latest")
            outcome_cp = str(self.run_dir / "checkpoint" / "latest.npz")
        else:
            outcome_cp = None
        eval_after = self.evaluate(split="eval", tag="after")
        return self._outcome(eval_before, eval_after, acc, rw, stopped_early, checkpoint_path=outcome_cp)

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, *, split: str = "eval", tag: str = "eval", n: int | None = None) -> dict:
        items: Sequence[Trial] = self.env.eval_items() if split == "eval" else self.env.train_items()
        if n:
            items = list(items)[: int(n)]
        elif self.cfg.train.eval_trials:
            items = list(items)[: int(self.cfg.train.eval_trials)]
        n_correct = 0
        rates: list[float] = []
        preds: list[Any] = []
        for j, trial in enumerate(items):
            rec = self.run_trial(trial, learn=False, episode=-1)
            n_correct += int(rec.correct)
            rates.append(rec.mean_rate_hz)
            preds.append(rec.prediction)
        total = max(1, len(items))
        p_chance = chance_level(len(self.env.classes))
        acc = n_correct / total
        return {
            "tag": tag,
            "n_trials": len(items),
            "n_correct": n_correct,
            "accuracy": round(acc, 4),
            "chance": round(p_chance, 4),
            "p_value_vs_chance": round(binom_p_above_chance(n_correct, total, p_chance), 6),
            "mean_rate_hz": round(float(np.mean(rates)), 4) if rates else 0.0,
            "readout_silent_trials": int(sum(1 for a in self.activity_records[-len(items):] if a.get("n_spikes", 0) == 0)),
            "prediction_histogram": {str(k): preds.count(k) for k in sorted({str(p) for p in preds}, key=lambda s: (s == "None", s))},
            "n_none_predictions": sum(1 for p in preds if p is None),
        }

    # -------------------------------------------------------------- checkpoint
    def _save(self, *, tag: str = "latest") -> str:
        rng_state = np.random.RandomState(abs(self.cfg.train.seed) % (2**31))
        path = save_checkpoint(
            self.run_dir,
            conn=self.conn,
            rule_state=self.rule.state_dict(),
            reward_state=self.reward.state_dict(),
            modulator_state=self.mod.state_dict(),
            encoder_state=self.encoder.state_dict(),
            decoder_state=self.decoder.state_dict(),
            episode=len(self.metrics.records),
            seed=self.cfg.train.seed,
            config_hash=self.cfg.config_hash,
            rng_state=rng_state,
            extra={"control": getattr(self, "control_note", "none"), "weight_history": self.weight_history[-50:]},
            tag=tag,
        )
        from .checkpoint import prune

        pruned = prune(self.run_dir, self.cfg.train.keep_last)
        if pruned:
            log.debug("pruned old checkpoints: %s", pruned)
        return str(path)

    def resume(self, source: str | Path) -> dict:
        """Restore weights/rule state from a checkpoint directory or tag."""
        src = Path(source)
        run_dir = src if src.is_dir() else self.run_dir
        tag = "latest" if src.is_dir() else src.stem
        cp = load_checkpoint(run_dir, tag=tag)
        if cp.config_hash and cp.config_hash != self.cfg.config_hash:
            log.warning(
                "checkpoint config hash %s != current %s: resuming anyway, but the run is "
                "no longer comparable to a from-scratch run", cp.config_hash, self.cfg.config_hash
            )
        rule_state = restore_connectome(cp, self.conn)
        self.sim.set_weights(self.conn.copy_weights())
        self.rule.load_state(rule_state)
        meta = cp.metadata
        self.reward.load_state(meta.get("reward", {}))
        self.mod.load_state(meta.get("modulator", {}))
        log.info("resumed from %s at episode %s", cp.path, cp.episode)
        return {"episode": cp.episode, "path": str(cp.path)}

    # ----------------------------------------------------------------- results
    def _outcome(self, eval_before: dict, eval_after: dict, acc: RunningStats, rw: RunningStats, stopped: str | None, *, checkpoint_path: str | None = None) -> TrainOutcome:
        accs = self.metrics.series("correct")
        rws = self.metrics.series("reward")
        w = self.conn.weights
        dw_total = float(np.abs(w - self.w0).sum())
        diag = self.rule.diagnostics()
        p_chance = chance_level(len(self.env.classes))
        improved = (
            (eval_after["accuracy"] - eval_before["accuracy"])
            if eval_before.get("accuracy") is not None and eval_after.get("accuracy") is not None
            else None
        )
        # The untrained network must not already be able to do the task. Without this
        # guard a run with a favourable (uncounterbalanced) readout mapping -- measured
        # at 0.875 held-out accuracy *before* training, reaching 1.000 after -- satisfies
        # "improvement above chance" while demonstrating nothing but a ceiling effect.
        pre_acc = eval_before.get("accuracy")
        baseline_at_chance = pre_acc is None or pre_acc <= p_chance + 0.05
        criterion = {
            "min_gain_vs_chance": 0.05,
            "min_gain_vs_baseline": 0.05,
            "p_threshold": 0.05,
            "chance": p_chance,
            "improvement": None if improved is None else round(float(improved), 4),
            "p_value_after": eval_after.get("p_value_vs_chance"),
            "total_abs_dw": diag["total_abs_dw"],
            "baseline_at_chance": bool(baseline_at_chance),
        }
        learning = bool(
            improved is not None
            and improved > 0.05
            and eval_after["accuracy"] > p_chance + 0.05
            and eval_after["p_value_vs_chance"] < 0.05
            and diag["total_abs_dw"] > 0
            and baseline_at_chance
        )
        if learning:
            claim = (
                "measurable improvement in held-out accuracy accompanied by non-zero synaptic "
                "weight change, in this configuration only. Not evidence about biological learning, "
                "and not a Luau capability."
            )
        elif not baseline_at_chance and pre_acc is not None:
            claim = (
                f"NOT counted as learning: the untrained network already scores {pre_acc:.3f} on the "
                f"held-out split (chance {p_chance:.3f}), so any gain is a ceiling effect of this "
                "readout mapping. Enable graph.counterbalance_init to adopt the mapping the "
                "untrained network is worst at, then re-measure."
            )
        else:
            claim = (
                "NO learning detected in this configuration (criterion: +5% held-out accuracy "
                "above chance and above pre-training, p<0.05, non-zero |dW|, and a baseline at "
                "chance). Reported as a null result rather than tuned until it passes."
            )
        acts = self.activity_records or [{}]
        outcome = TrainOutcome(
            run_dir=self.run_dir,
            config={**self.cfg.to_dict(), "scale_label": self.population.scale_label, "control": getattr(self, "control_note", "none")},
            n_neurons=self.conn.n_neurons,
            n_edges=self.conn.n_edges,
            n_episodes=len(self.metrics.records),
            train_accuracy_window=None if acc.n == 0 else round(acc.mean, 4),
            eval_before=eval_before,
            eval_after=eval_after,
            delta_eval_accuracy=None if improved is None else round(improved, 4),
            reward_curve={"raw_last_200": [float(x) for x in rws[-200:]], "summary": summarize_curve(rws, window=self.cfg.train.window)},
            accuracy_curve={"raw_last_200": [float(x) for x in accs[-200:]], "summary": summarize_curve(accs, window=self.cfg.train.window)},
            weight_stats={**diag, "total_abs_dw_vs_init": dw_total, "mean_abs_dw_vs_init": dw_total / max(w.size, 1),
                          "weight_history": self.weight_history[-200:]},
            plasticity_diagnostics=diag,
            activity={
                "spikes_per_trial_mean": float(np.mean([a.get("n_spikes", 0) for a in acts])),
                "spikes_per_trial_max": float(np.max([a.get("n_spikes", 0) for a in acts])),
                "active_fraction_mean": float(np.mean([a.get("active", 0) / max(self.conn.n_neurons, 1) for a in acts])),
            },
            timings={"train_s": round(time.perf_counter() - self.started, 2)},
            controls={"note": getattr(self, "control_note", "none"), "stopped_early": stopped,
                      "leakage_audit": self.env.overlap_check()},
            learning_detected=learning,
            learning_criterion=criterion,
            claim_bounds=claim,
            checkpoint_path=checkpoint_path,
        )
        self.metrics.close()
        out = {k: v for k, v in outcome.__dict__.items() if k != "config"}
        out["run_dir"] = str(self.run_dir)
        write_json(self.run_dir / "results.json", out)
        (self.run_dir / "SUMMARY.txt").write_text(outcome.summary_text() + "\n", encoding="utf-8")
        return outcome
