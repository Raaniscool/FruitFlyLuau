"""Synaptic plasticity rules operating directly on the sparse weight array.

Why this is "learning" and not a lookup table
---------------------------------------------
The only state a rule writes is ``Connectome.matrix.data`` --- the synaptic weight
between two real FAFB neurons (or two real nodes of the sample graph). There is no
question/answer store anywhere in this module.

Implementation
--------------
Once per run we snapshot the CSR row/col index arrays, so a step is pure NumPy over
edges with no Python per-edge loop:

    trace_pre, trace_post  <- decay + spike injection (per neuron)
    elig[e] += A+ * spike[post_e] * trace_pre[pre_e] - A- * spike[pre_e] * trace_post[post_e]
    elig     <- elig * exp(-dt / tau_trace)
    w[e]     <- clamp(w[e] + eta * modulator * elig[e])

The clamp respects each edge's initial sign when ``polarity_lock`` is on (a
Dale-like engineering constraint, not a biological claim), so inhibitory edges
cannot silently become excitatory and vice versa.

Eligibility traces are what let a *delayed* reward still credit synapses that
were co-active seconds earlier -- the mechanism the project exists to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import PlasticityConfig
from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class PlasticityContext:
    """What a rule is allowed to see at one simulation step."""

    t_step: int = 0
    trial: int = 0
    spikes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    spikes_bool: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    modulator: float = 0.0  # dopamine-like signal (scalar) or per-neuron array
    dt_ms: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapseState:
    """Edge bookkeeping aligned with the CSR data array."""

    n_edges: int
    edge_pre: np.ndarray  # int32 source per edge
    edge_post: np.ndarray  # int32 target per edge
    plastic_mask: np.ndarray  # bool: which edges may change
    eligibility: np.ndarray  # float32 (n_edges,)
    updates: np.ndarray  # int64 count of actual changes per edge
    w_lo: np.ndarray  # float64 per-edge lower clamp
    w_hi: np.ndarray  # float64 per-edge upper clamp
    row_target: np.ndarray | None = None  # initial |w| row sums for normalisation
    col_target: np.ndarray | None = None  # initial |w| column sums for normalisation
    weight_history: list = field(default_factory=list)

    @staticmethod
    def from_connectome(
        conn: Connectome, cfg: PlasticityConfig, *, seed: int = 0
    ) -> "SynapseState":
        m = conn.matrix
        n = int(m.nnz)
        if n == 0:
            raise ValueError("connectome has no edges to plasticise")
        edge_pre = np.repeat(np.arange(m.shape[0], dtype=np.int32), np.diff(m.indptr))
        edge_post = np.asarray(m.indices, dtype=np.int32)
        mask = np.ones(n, dtype=bool)
        frac = float(cfg.plastic_fraction)
        if frac < 1.0:
            rng = np.random.default_rng(seed)
            keep = max(1, int(round(n * frac)))
            rng.shuffle(mask)
            mask[keep:] = False
            log.info(
                "plasticity restricted to %s of %s edges (%.0f%%) for performance",
                f"{keep:,}", f"{n:,}", 100 * frac,
            )
        w0 = np.asarray(m.data, dtype=np.float64)
        w_max = float(cfg.w_max)
        if cfg.w_min is not None:
            lo_floor, hi_ceil = float(cfg.w_min), w_max
        else:  # auto: symmetric magnitude bound; all-positive graphs stay positive
            lo_floor = -w_max if (w0 < 0).any() else 0.0
            hi_ceil = w_max
        if cfg.polarity_lock:
            # each edge may only move within the sign it started with
            lo = np.where(w0 < 0, lo_floor, 0.0).astype(np.float64)
            hi = np.where(w0 > 0, hi_ceil, 0.0).astype(np.float64)
            zero = w0 == 0
            if zero.any():
                lo[zero] = min(lo_floor, 0.0)
                hi[zero] = hi_ceil
        else:
            lo = np.full(n, min(lo_floor, 0.0), dtype=np.float64)
            hi = np.full(n, hi_ceil, dtype=np.float64)
        lo = np.minimum(lo, w0)
        hi = np.maximum(hi, w0)
        row_target = np.bincount(edge_pre, weights=np.abs(w0), minlength=m.shape[0]).astype(np.float64)
        col_target = np.bincount(edge_post, weights=np.abs(w0), minlength=m.shape[1]).astype(np.float64)
        return SynapseState(
            n_edges=n, edge_pre=edge_pre, edge_post=edge_post, plastic_mask=mask,
            eligibility=np.zeros(n, dtype=np.float32), updates=np.zeros(n, dtype=np.int64),
            w_lo=lo, w_hi=hi, row_target=row_target, col_target=col_target,
        )


class PlasticityRule:
    """Interface every learning rule implements; override :meth:`apply`."""

    name: str = "base"
    #: does the rule consume the dopamine-like modulatory signal?
    uses_modulator: bool = False

    def __init__(self, cfg: PlasticityConfig, conn: Connectome) -> None:
        self.cfg = cfg
        self.conn = conn
        self.state = SynapseState.from_connectome(conn, cfg, seed=cfg.seed)
        self.trace_pre = np.zeros(conn.n_neurons, dtype=np.float32)
        self.trace_post = np.zeros(conn.n_neurons, dtype=np.float32)
        self.n_steps = 0
        self.total_abs_dw = 0.0

    # --------------------------------------------------------------- lifecycle
    def begin_trial(self) -> None:
        """Reset eligibility between trials (the credit window is per-trial)."""
        self.state.eligibility[:] = 0.0

    def end_trial(self) -> dict:
        norm = (self.cfg.normalize or "none").lower()
        if norm in {"rowsum", "both"}:
            self._normalise_rows()
        if norm in {"colsum", "both"}:
            self._normalise_columns()
        out: dict[str, float] = {}
        if self.cfg.normalize == "max":
            hi = float(np.abs(self.conn.weights).max()) if self.conn.n_edges else 1.0
            if hi > self.cfg.w_max > 0:
                self.conn.weights *= self.cfg.w_max / hi
                out["row_normalised_to"] = self.cfg.w_max
        out["weight_mean"] = float((float(self.conn.weights.mean()) if self.conn.weights.size else 0.0))
        return out

    # ------------------------------------------------------------------ update
    def _decay_traces(self, spikes: np.ndarray, dt_ms: float) -> None:
        c = self.cfg
        self.trace_pre *= float(np.exp(-dt_ms / max(c.tau_plus_ms, 1e-6)))
        self.trace_post *= float(np.exp(-dt_ms / max(c.tau_minus_ms, 1e-6)))
        hit = spikes.astype(bool)
        if hit.any():
            self.trace_pre[hit] += 1.0
            self.trace_post[hit] += 1.0

    def _eligibility_step(self, ctx: PlasticityContext) -> None:
        s = self.state
        c = self.cfg
        if not s.plastic_mask.any():
            return
        sp = ctx.spikes_bool
        d_plus = sp[s.edge_post] * self.trace_pre[s.edge_pre]
        d_minus = sp[s.edge_pre] * self.trace_post[s.edge_post]
        delta = c.a_plus * d_plus - c.a_minus * d_minus
        sel = s.plastic_mask
        s.eligibility[sel] += delta[sel]
        s.eligibility *= float(np.exp(-ctx.dt_ms / max(c.tau_trace_ms, 1e-6)))

    def _write_weights(self, increments: np.ndarray) -> tuple[int, float]:
        """Apply ``increments`` to plastic edges, respecting per-edge clamps."""
        s = self.state
        w = self.conn.weights
        sel = s.plastic_mask
        old = w[sel]
        new = np.clip(old + increments, s.w_lo[sel], s.w_hi[sel])
        changed = new != old
        w[sel] = new
        s.updates[sel] += changed.astype(np.int64)
        abs_dw = float(np.abs(new - old).sum())
        self.total_abs_dw += abs_dw
        return int(changed.sum()), abs_dw

    def apply(self, ctx: PlasticityContext) -> dict:
        """One simulation step. Returns metrics for this step."""
        self.n_steps += 1
        if ctx.spikes_bool is None or ctx.spikes_bool.size != ctx.spikes.size:
            ctx.spikes_bool = ctx.spikes.astype(bool)
        self._decay_traces(ctx.spikes, ctx.dt_ms)
        self._eligibility_step(ctx)
        s = self.state
        if not s.plastic_mask.any():
            return {"edges_updated": 0, "abs_dw_sum": 0.0}
        mod = ctx.modulator
        elig = self._elig_for_write()
        if isinstance(mod, np.ndarray) and mod.ndim:
            inc = self.cfg.eta * elig * np.asarray(mod, dtype=np.float64)[s.edge_post[s.plastic_mask]]
        else:
            inc = self.cfg.eta * float(mod) * elig
        n_changed, abs_dw = self._write_weights(inc)
        if self.n_steps % 100 == 0:
            w = self.conn.weights
            s.weight_history.append((self.n_steps, float(w.mean()), float(w.std())))
        return {"edges_updated": n_changed, "abs_dw_sum": abs_dw}

    def accumulate(self, ctx: PlasticityContext) -> dict:
        """Build eligibility traces for this step *without* writing weights.

        Used by the ``end_of_trial`` credit mode: the co-activity that earns credit
        happens while the animal is still responding, and the (possibly delayed)
        reinforcement signal only arrives afterwards. The trace must therefore be
        filled in during the trial and cashed in once, at :meth:`apply_at_trial_end`.
        Calling :meth:`apply` here instead would be wrong: it writes with the
        *current* modulator value, which is zero for most of the trial.
        """
        self.n_steps += 1
        if ctx.spikes_bool is None or ctx.spikes_bool.size != ctx.spikes.size:
            ctx.spikes_bool = ctx.spikes.astype(bool)
        self._decay_traces(ctx.spikes, ctx.dt_ms)
        self._eligibility_step(ctx)
        e = self.state.eligibility
        return {
            "edges_updated": 0,
            "abs_dw_sum": 0.0,
            "elig_max_abs": (float(np.abs(e).max()) if e.size else 0.0),
        }

    def _elig_for_write(self) -> np.ndarray:
        """Eligibility to be applied now, optionally mean-centred over plastic edges."""
        s = self.state
        e = s.eligibility[s.plastic_mask]
        if getattr(self.cfg, "center_eligibility", False) and e.size > 1:
            e = e - float(e.mean())
        return e

    def _normalise_columns(self) -> None:
        """Scale each post-synaptic neuron's incoming edges back to its initial total.

        This is what makes the reinforcement signal competitive instead of merely
        excitatory/depressory for everything.
        """
        s, m = self.state, self.conn.matrix
        w = self.conn.weights
        cols = s.edge_post
        cur = np.bincount(cols, weights=np.abs(w), minlength=m.shape[1])
        target = s.col_target if s.col_target is not None else cur
        scale = np.ones(m.shape[1], dtype=np.float64)
        ok = cur > 1e-9
        scale[ok] = np.minimum(target[ok] / cur[ok], 10.0)
        w *= scale[cols]
        np.clip(w, s.w_lo, s.w_hi, out=w)

    def apply_at_trial_end(self, modulator_value: float) -> dict:
        """One-shot credit assignment over the eligibility accumulated in a trial.

        This is the ``end_of_trial`` credit path: the trial's co-activity has left a
        trace per edge, and the (possibly delayed) reinforcement signal now scales it.
        """
        s = self.state
        if not s.plastic_mask.any():
            return {"edges_updated": 0, "abs_dw_sum": 0.0, "credit_scale": float(modulator_value)}
        inc = self.cfg.eta * float(modulator_value) * self._elig_for_write()
        n_changed, abs_dw = self._write_weights(inc)
        return {"edges_updated": n_changed, "abs_dw_sum": abs_dw, "credit_scale": float(modulator_value)}

    # ------------------------------------------------------------------ utils
    def _normalise_rows(self) -> None:
        """Scale each presynaptic row back to its initial total |w| (homeostasis)."""
        s, m = self.state, self.conn.matrix
        w = self.conn.weights
        rows = s.edge_pre
        cur = np.bincount(rows, weights=np.abs(w), minlength=m.shape[0])
        target = s.row_target if s.row_target is not None else cur
        scale = np.ones(m.shape[0], dtype=np.float64)
        ok = cur > 1e-9
        scale[ok] = np.minimum(target[ok] / cur[ok], 10.0)
        w *= scale[rows]
        np.clip(w, s.w_lo, s.w_hi, out=w)

    def diagnostics(self) -> dict:
        w = self.conn.weights
        s = self.state
        e = s.eligibility
        return {
            "rule": self.name,
            "n_steps": self.n_steps,
            "n_edges": int(w.size),
            "plastic_edges": int(s.plastic_mask.sum()),
            "weight_mean": float((float(w.mean()) if w.size else 0.0)),
            "weight_std": float((float(w.std()) if w.size else 0.0)),
            "weight_min": float((w.min() if w.size else 0.0)),
            "weight_max": float((w.max() if w.size else 0.0)),
            "total_abs_dw": float(self.total_abs_dw),
            "n_edges_changed_at_least_once": int((s.updates > 0).sum()),
            "elig_mean_abs": float(np.abs(e).mean()) if e.size else 0.0,
            "elig_max_abs": (float(np.abs(e).max()) if e.size else 0.0),
        }

    def state_dict(self) -> dict:
        s = self.state
        return {
            "rule": self.name,
            "n_steps": self.n_steps,
            "total_abs_dw": self.total_abs_dw,
            "trace_pre": self.trace_pre.astype(np.float32),
            "trace_post": self.trace_post.astype(np.float32),
            "eligibility": s.eligibility.astype(np.float32),
            "updates": s.updates.astype(np.int64),
        }

    def load_state(self, d: dict) -> None:
        self.n_steps = int(d.get("n_steps", 0))
        self.total_abs_dw = float(d.get("total_abs_dw", 0.0))
        s = self.state
        for key, target in (("trace_pre", "trace_pre"), ("trace_post", "trace_post")):
            if key in d:
                arr = np.asarray(d[key], dtype=np.float32)
                if arr.size == getattr(self, target).size:
                    setattr(self, target, arr)
        if "eligibility" in d:
            arr = np.asarray(d["eligibility"], dtype=np.float32)
            if arr.size == s.eligibility.size:
                s.eligibility = arr
        if "updates" in d:
            arr = np.asarray(d["updates"], dtype=np.int64)
            if arr.size == s.updates.size:
                s.updates = arr


class STDP(PlasticityRule):
    """Pure spike-timing plasticity, unmodulated.

    This is the control that keeps us honest: if an unmodulated rule solves the
    task, then reinforcement was not what solved it, and we report exactly that.
    """

    name = "stdp"
    uses_modulator = False

    def apply(self, ctx: PlasticityContext) -> dict:
        ctx.modulator = 1.0
        return super().apply(ctx)


class RewardModulatedSTDP(PlasticityRule):
    """STDP eligibility x dopamine-like modulatory signal (the primary experiment)."""

    name = "reward_stdp"
    uses_modulator = True


class HebbianTrace(PlasticityRule):
    """Co-activity potentiation with a leak; ignores precise spike timing."""

    name = "hebbian"
    uses_modulator = True

    def _update_trace(self, ctx: PlasticityContext) -> None:
        s = self.state
        sp = ctx.spikes.astype(np.float32)
        co = sp[s.edge_pre] * sp[s.edge_post]
        sel = s.plastic_mask
        s.eligibility[sel] = np.clip(s.eligibility[sel] * 0.9 + co[sel], 0.0, 5.0)

    def accumulate(self, ctx: PlasticityContext) -> dict:
        self.n_steps += 1
        if not self.state.plastic_mask.any():
            return {"edges_updated": 0, "abs_dw_sum": 0.0}
        self._update_trace(ctx)
        e = self.state.eligibility
        return {"edges_updated": 0, "abs_dw_sum": 0.0,
                "elig_max_abs": (float(np.abs(e).max()) if e.size else 0.0)}

    def apply(self, ctx: PlasticityContext) -> dict:
        self.n_steps += 1
        s = self.state
        if not s.plastic_mask.any():
            return {"edges_updated": 0, "abs_dw_sum": 0.0}
        self._update_trace(ctx)
        inc = self.cfg.eta * float(ctx.modulator) * s.eligibility[s.plastic_mask]
        n_changed, abs_dw = self._write_weights(inc)
        return {"edges_updated": n_changed, "abs_dw_sum": abs_dw}


class FrozenWeights(PlasticityRule):
    """No learning at all: proves the task is not solvable by the initial state alone."""

    name = "none"

    def apply(self, ctx: PlasticityContext) -> dict:
        return {"edges_updated": 0, "abs_dw_sum": 0.0}


_RULES: dict[str, type[PlasticityRule]] = {cls.name: cls for cls in (STDP, RewardModulatedSTDP, HebbianTrace, FrozenWeights)}
_RULES["rstdp"] = RewardModulatedSTDP
_RULES["frozen"] = FrozenWeights
_RULES["no_plasticity"] = FrozenWeights


def register_rule(cls: type[PlasticityRule]) -> type[PlasticityRule]:
    """Register a new rule (used by later phases without editing this file)."""
    if not issubclass(cls, PlasticityRule):
        raise TypeError("rule must subclass PlasticityRule")
    if cls.name in _RULES and _RULES[cls.name] is not cls:
        log.warning("overriding plasticity rule %r", cls.name)
    _RULES[cls.name] = cls
    return cls


def available_rules() -> list[str]:
    return sorted(set(_RULES))


def build_rule(cfg: PlasticityConfig, conn: Connectome) -> PlasticityRule:
    key = (cfg.rule or "none").strip().lower()
    if key not in _RULES:
        raise ValueError(f"unknown plasticity rule {cfg.rule!r}; available: {available_rules()}")
    cls = _RULES[key]
    rule = cls(cfg, conn)
    log.info(
        "plasticity=%s eta=%g modulated=%s over %s/%s edges",
        cls.name, cfg.eta, cls.uses_modulator, f"{rule.state.plastic_mask.sum():,}", f"{conn.n_edges:,}",
    )
    return rule
