"""Vectorised leaky integrate-and-fire network over the sparse connectome.

Model (per dt step, explicit Euler), documented so nobody mistakes it for biology:

    g      <- g * exp(-dt/tau_syn) + (Wᵀ @ s_prev)              synaptic drive
    I_syn  <- gain * g * (1 | E_rev - v)                        current or conductance
    v      <- v + (dt/tau) * ( -leak*(v - v_rest) + I_syn + I_ext + noise )
    spike  <- (v > v_thresh + adaptation) & (refractory == 0)
    v[spike] = v_reset ;  refractory = ceil(refractory_ms/dt) ;  adaptation += a_step

W is the CSR data array of the :class:`~fruitfly.graph.connectome.Connectome`,
rows = presynaptic, cols = postsynaptic, so ``W.T @ spikes`` is the input current
each neuron receives from neurons that spiked in the previous step (that one-step
lag is our axonal-conduction delay). Weight signs encode the E/I convention
configured at build time.

Everything is NumPy/SciPy array arithmetic: no per-neuron Python objects, so the
cost per step is one sparse mat-vec plus a handful of O(n) vector ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ..config import LIFConfig
from ..graph.connectome import Connectome
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class Diagnostics:
    """Per-run activity statistics, computed from real spikes (never assumed)."""

    n_steps: int = 0
    n_spikes: int = 0
    mean_rate_hz: float = 0.0
    fraction_neurons_ever_active: float = 0.0
    max_single_neuron_rate_hz: float = 0.0
    cv_isi: float = 0.0
    sync_index: float = 0.0  # population-wide synchrony: var(pop_rate)/mean(pop_rate)
    notes: list[str] = field(default_factory=list)


class NetworkSimulator:
    """One connectome + one LIF parameter set + integrator state."""

    def __init__(
        self,
        connectome: Connectome,
        params: LIFConfig | None = None,
        *,
        seed: int = 0,
        weight_source: np.ndarray | None = None,
    ) -> None:
        self.conn = connectome
        self.p = params or LIFConfig()
        self.n = connectome.n_neurons
        self.rng = np.random.default_rng(seed)
        # Live view onto the CSR arrays. A.T is an O(1) *view* sharing the same
        # data buffer, so plasticity writing into W.data is immediately visible to
        # the propagation mat-vec with no per-step rebuild (verified in
        # tests/test_signal_propagation.py::test_plasticity_is_visible_to_propagation).
        self._W = connectome.matrix
        self._WT = self._W.T
        self._validate()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        p = self.p
        if p.dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        if p.tau_ms <= 0:
            raise ValueError("tau_ms must be > 0")
        if p.v_thresh <= p.v_reset:
            raise ValueError("v_thresh must be above v_reset or the neuron cannot recover")
        if p.dt_ms > p.tau_ms:
            log.warning(
                "dt_ms=%.3f > tau_ms=%.3f: Euler integration of the leak is inaccurate; reduce dt",
                p.dt_ms, p.tau_ms,
            )
        if self._W.shape[0] != self.n or self._W.shape[1] != self.n:
            raise ValueError("connectome matrix shape does not match population size")

    def reset(self, *, v: float | np.ndarray | None = None) -> None:
        p = self.p
        self.v = np.full(self.n, p.v_rest if v is None else v, dtype=np.float64)
        self.g = np.zeros(self.n, dtype=np.float64)  # signed synaptic drive state
        self.adapt = np.zeros(self.n, dtype=np.float64)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.inhibition = 0.0  # pooled APL-like inhibitory state (scalar)
        self._spikes = np.zeros(self.n, dtype=np.float32)
        self.spike_count = np.zeros(self.n, dtype=np.int64)
        self.last_spike_step = np.full(self.n, -1, dtype=np.int64)
        self.isi_sq_sum = 0.0
        self.isi_sum = 0.0
        self.isi_count = 0
        self.t_step = 0
        self._sync_num = 0.0
        self._sync_den = 0.0

    # ------------------------------------------------------------------ step
    def step(self, i_ext: np.ndarray | float | None = None) -> np.ndarray:
        """Advance one ``dt``. ``i_ext`` is injected drive per neuron (mV per tau-unit)."""
        p = self.p
        s_prev = self._spikes

        # 1) synaptic drive: decay, then add input from neurons that spiked last step
        self.g *= np.exp(-p.dt_ms / max(p.tau_syn_ms, 1e-6))
        if s_prev.any():
            self.g += np.asarray(self._WT.dot(s_prev)).ravel()  # W.T @ spikes_prev

        # 2) synaptic current
        if p.reversal_mode == "conductance":
            e_rev = np.where(self.g > 0, p.e_rev_exc, p.e_rev_inh)
            i_syn = p.synaptic_gain * self.g * (e_rev - self.v)
        else:
            i_syn = p.synaptic_gain * self.g

        # 3) external input + noise
        i_ext_arr = 0.0 if i_ext is None else i_ext
        if p.noise:
            i_ext_arr = i_ext_arr + self.rng.normal(0.0, p.noise, size=self.n)
        if p.background_current:
            i_ext_arr = i_ext_arr + p.background_current

        # 4) leak + integrate
        # APL-like pooled inhibition: one scalar driven by how much of the population
        # just fired, subtracted from every neuron. Applied before integration so it can
        # actually prevent the next volley rather than react to it.
        if p.global_inhibition > 0.0:
            decay_inh = np.exp(-p.dt_ms / max(p.inhibition_tau_ms, 1e-6))
            self.inhibition = self.inhibition * decay_inh + float(s_prev.mean())
            i_syn = i_syn - p.global_inhibition * self.inhibition

        d = (p.dt_ms / p.tau_ms) * (-p.leak * (self.v - p.v_rest) + i_syn + i_ext_arr)
        self.v += d
        if p.clip_v:
            np.clip(self.v, p.v_rest - p.clip_v, p.v_rest + p.clip_v, out=self.v)

        # 5) refractory countdown
        active = self.refractory <= 0
        thresh = p.v_thresh + self.adapt

        # 6) spike + reset
        spiked = active & (self.v > thresh)
        if spiked.any():
            self.v[spiked] = p.v_reset
            self.refractory[spiked] = max(1, int(round(p.refractory_ms / p.dt_ms)))
            if p.adaptation:
                self.adapt[spiked] += p.adaptation
            idx = np.flatnonzero(spiked)
            self.spike_count[idx] += 1
            gaps = self.t_step - self.last_spike_step[idx]
            first = self.last_spike_step[idx] < 0
            gaps = gaps[~first]
            if gaps.size:
                self.isi_sum += float(gaps.sum()) * p.dt_ms
                self.isi_sq_sum += float((gaps.astype(np.float64) ** 2).sum()) * p.dt_ms ** 2
                self.isi_count += int(gaps.size)
            self.last_spike_step[idx] = self.t_step
        else:
            idx = np.empty(0, dtype=np.int64)

        if p.adaptation:
            self.adapt *= np.exp(-p.dt_ms / 100.0)

        self.refractory[self.refractory > 0] -= 1
        self.refractory[~(self.refractory > 0)] = 0
        self._spikes = spiked.astype(np.float32)
        self.t_step += 1

        pop_rate = float(spiked.sum())
        self._sync_num += pop_rate * pop_rate
        self._sync_den += pop_rate
        return spiked

    # ------------------------------------------------------------------- run
    def run(
        self,
        steps: int,
        *,
        current_schedule: np.ndarray | None = None,
        collect: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Run ``steps`` steps.

        Parameters
        ----------
        current_schedule:
            Optional ``(steps, n)`` external input array (float32). Kept as a
            dense array because encoders produce exactly that shape for small
            populations; for full-brain runs pass a sparse ``(steps, idx, val)``
            triple via :meth:`run_sparse` instead.
        collect:
            Whether to keep the boolean spike matrix (memory: steps*n bytes).

        Returns
        -------
        (spike_counts_per_neuron, spike_matrix_or_None, per_step_population_rate)
        """
        counts = np.zeros(self.n, dtype=np.int64)
        rates = np.zeros(steps, dtype=np.float64)
        spikes = np.zeros((steps, self.n), dtype=bool) if collect else None
        for t in range(steps):
            i = None if current_schedule is None else current_schedule[t]
            s = self.step(i)
            counts += s
            rates[t] = float(s.sum())
            if spikes is not None:
                spikes[t] = s
        return counts, spikes, rates

    def run_sparse(
        self,
        steps: int,
        inject_per_step: list[tuple[np.ndarray, np.ndarray]] | None = None,
        *,
        collect: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Memory-friendly run for large populations.

        ``inject_per_step[t] = (neuron_indices, values)``; only those neurons get
        external current, so no dense ``(steps, n)`` array is ever allocated.
        """
        counts = np.zeros(self.n, dtype=np.int64)
        rates = np.zeros(steps, dtype=np.float64)
        spikes = np.zeros((steps, self.n), dtype=bool) if collect else None
        for t in range(steps):
            i_ext = None
            if inject_per_step is not None and t < len(inject_per_step):
                idx, val = inject_per_step[t]
                if idx.size:
                    i_ext = np.zeros(self.n, dtype=np.float64)
                    np.add.at(i_ext, idx, val)
            s = self.step(i_ext)
            counts += s
            rates[t] = float(s.sum())
            if spikes is not None:
                spikes[t] = s
        return counts, spikes, rates

    # --------------------------------------------------------------- reports
    def diagnostics(self, steps: int) -> Diagnostics:
        p = self.p
        total_sec = steps * p.dt_ms / 1000.0
        n_spikes = int(self.spike_count.sum())
        active = int((self.spike_count > 0).sum())
        rate = n_spikes / max(self.n, 1) / total_sec if total_sec else 0.0
        cv = 0.0
        if self.isi_count > 1:
            mean_isi = self.isi_sum / self.isi_count
            second = self.isi_sq_sum / self.isi_count
            var = max(second - mean_isi**2, 0.0)
            cv = float(np.sqrt(var) / mean_isi) if mean_isi > 0 else 0.0
        sync = 0.0
        if self._sync_den > 0:
            mean_pop = self._sync_den / steps
            mean_sq = self._sync_num / steps
            sync = float((mean_sq - mean_pop**2) / mean_pop) if mean_pop > 0 else 0.0
        return Diagnostics(
            n_steps=steps, n_spikes=n_spikes, mean_rate_hz=rate,
            fraction_neurons_ever_active=active / max(self.n, 1),
            max_single_neuron_rate_hz=(float(self.spike_count.max()) if self.spike_count.size else 0) / total_sec if total_sec else 0.0,
            cv_isi=cv, sync_index=sync,
        )

    # ------------------------------------------------------------ live weights
    @property
    def weights(self) -> np.ndarray:
        """Alias to the CSR data array: the physical state plasticity modifies."""
        return self._W.data

    def set_weights(self, data: np.ndarray) -> None:
        if data.shape != self._W.data.shape:
            raise ValueError("weight array shape mismatch")
        self._W.data[:] = data
        # _WT shares the same buffer as _W, so no rebuild is needed.
