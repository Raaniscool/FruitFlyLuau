"""Reservoir characterisation: what can a FROZEN connectome compute?

Why this module exists
----------------------
Every learning experiment in this project so far has returned a null: the
plasticity rules change weights by up to six orders of magnitude and held-out
accuracy never leaves chance (see EXPERIMENTS.md). That result has two possible
causes and the experiments could not tell them apart:

1. the learning rule is wrong, or
2. the network's *state* never contains the information a readout would need,
   in which case no learning rule could succeed.

This module tests (2) directly, and it cannot be blocked by (1), because
**nothing here is trained except a closed-form linear readout**. The connectome
is frozen. That is the standard reservoir-computing / liquid-state-machine
methodology (Jaeger 2001; Maass, Natschlaeger & Markram 2002; Legenstein &
Maass 2007), applied here to a biologically derived graph rather than a random
one.

What it measures
----------------
``memory_capacity``
    Jaeger's MC. Drive the network with an i.i.d. random scalar signal, then ask
    a ridge readout to reconstruct the input delayed by k steps. MC_k is the
    coefficient of determination on **held-out** timesteps; MC is the sum over
    k. Units: "effective past timesteps recoverable". MC is bounded above by the
    number of readout units.

``separation``
    Legenstein & Maass's separation property: mean state distance between
    responses to *different* input streams, divided by mean state distance
    between responses to the *same* stream under noise. > 1 means distinct
    inputs leave distinguishable traces; ~1 means the network smears everything
    together and no decoder can help.

``kernel_rank`` / ``generalisation_rank``
    Effective dimensionality of the state space for distinct inputs, and for
    noisy versions of one input. A useful reservoir has high kernel rank and low
    generalisation rank; ``kernel_rank - generalisation_rank`` is the usual
    single-number summary.

Honest limits
-------------
- These are properties of *this model* (LIF dynamics, our E/I sign convention,
  our weight scaling) driven by *this* wiring. They are not measurements of the
  fly. A connectome is a wiring map; everything dynamic here is our invention.
- A good score does not mean the network can learn a task. It means the
  information is present for a decoder to use, which is a necessary — not
  sufficient — condition.
- Every number is computed on held-out timesteps and reported with the sample
  size used, so it can be compared against the shuffled-wiring control.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace

import numpy as np
from scipy import sparse

from ..neuro.network import NetworkSimulator
from ..utils import get_logger

log = get_logger(__name__)

__all__ = [
    "ReservoirStates",
    "calibrate_drive",
    "MemoryCapacity",
    "Separation",
    "ReservoirReport",
    "collect_states",
    "ridge_readout",
    "memory_capacity",
    "separation",
    "rank_measures",
    "branching_ratio",
    "shuffle_connectome",
    "characterise",
]


# --------------------------------------------------------------------- states
@dataclass
class ReservoirStates:
    """Filtered spike traces over time. ``X[t, i]`` is unit i's trace at step t."""

    X: np.ndarray
    u: np.ndarray  # the driving signal, aligned with X
    washout: int
    input_neurons: np.ndarray
    readout_neurons: np.ndarray
    fraction_active: float
    mean_rate_hz: float
    #: spikes per timestep, used for the branching ratio. Defaulted so a caller can
    #: construct states by hand (the metric tests do) without inventing spike trains.
    pop_spikes: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_steps(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.X.shape[1])


def collect_states(
    sim: NetworkSimulator,
    signal: np.ndarray,
    *,
    input_neurons: np.ndarray,
    readout_neurons: np.ndarray | None = None,
    amplitude: float = 22.0,
    bias: float = 0.0,
    trace_tau_steps: float = 6.0,
    washout: int = 50,
) -> ReservoirStates:
    """Drive the frozen network with ``signal`` and return its filtered state.

    ``bias`` is a constant background current delivered to every neuron. It is a
    modelling choice, not biology: without it a sparse subgraph sits below
    threshold and produces no spikes at all, and every measurement below would
    describe a silent network. Use :func:`calibrate_drive` to pick it by
    measurement rather than by guess.

    The readout state is an exponentially filtered spike train, which is the
    standard "liquid state": instantaneous spikes are too sparse for a linear
    decoder to work with at these rates, and the filter is a property of the
    *decoder*, not of the network. No weight in the connectome changes here.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n_steps = signal.size
    idx_in = np.asarray(input_neurons, dtype=np.int64).ravel()
    idx_out = (np.arange(sim.n, dtype=np.int64) if readout_neurons is None
               else np.asarray(readout_neurons, dtype=np.int64).ravel())
    if idx_in.size == 0:
        raise ValueError("no input neurons: the reservoir would never be driven")
    if washout >= n_steps:
        raise ValueError(f"washout {washout} >= signal length {n_steps}")

    decay = float(np.exp(-1.0 / max(trace_tau_steps, 1e-6)))
    trace = np.zeros(sim.n, dtype=np.float64)
    X = np.empty((n_steps, idx_out.size), dtype=np.float64)
    pop = np.zeros(n_steps, dtype=np.float64)
    i_ext = np.zeros(sim.n, dtype=np.float64)
    ever = np.zeros(sim.n, dtype=bool)
    n_spikes = 0

    for t in range(n_steps):
        i_ext[:] = bias
        i_ext[idx_in] = bias + amplitude * signal[t]
        s = sim.step(i_ext)
        trace *= decay
        trace += s
        X[t] = trace[idx_out]
        pop[t] = float(s.sum())
        ever |= s > 0
        n_spikes += int(s.sum())

    total_s = n_steps * sim.p.dt_ms / 1000.0
    states = ReservoirStates(
        X=X[washout:],
        u=signal[washout:],
        pop_spikes=pop[washout:],
        washout=washout,
        input_neurons=idx_in,
        readout_neurons=idx_out,
        fraction_active=float(ever.mean()),
        mean_rate_hz=float(n_spikes / max(sim.n, 1) / total_s) if total_s else 0.0,
    )
    log.info("reservoir states: %d steps x %d units, %.1f%% of neurons ever spiked, %.2f Hz",
             states.n_steps, states.n_units, 100 * states.fraction_active, states.mean_rate_hz)
    return states


def calibrate_drive(
    sim_factory,
    *,
    input_neurons: np.ndarray,
    amplitude: float = 22.0,
    candidates: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0),
    gain_candidates: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01),
    inhibition_candidates: tuple[float, ...] = (0.0, 20.0, 60.0, 120.0),
    target_rate_hz: tuple[float, float] = (5.0, 80.0),
    steps: int = 200,
    seed: int = 0,
) -> tuple[float, dict]:
    """Pick the background current -- and, if needed, the synaptic gain -- by measuring.

    Three knobs, because one was not enough and two still were not. Measured on the real mushroom-body subgraph:
    the LIF parameters were calibrated on a sparse synthetic graph with mean degree 2.6,
    and the real circuit has mean degree 20.8. At that connectivity the network
    self-ignites from its own recurrence -- **every** bias candidate including zero gave
    380-480 Hz, roughly 100x a plausible rate, and the network is saturated rather than
    computing. Sweeping the bias alone could not fix that, so the gain is swept too.

    The third knob is ``global_inhibition``, an APL-like pooled inhibitory feedback.
    Measured on a graph matched to the real subgraph (all-excitatory, mean degree 20.8):
    with no inhibition the network only ever sits at 474 Hz / 100% active or 9 Hz / 7%
    active -- there is no middle. At inhibition 60 it reaches 82 Hz with a branching
    ratio of 0.98, i.e. an actually near-critical regime. The knob exists because the
    subgraph is missing the fly's own inhibition, not to make results look better.

    Among settings whose firing rate is acceptable, the one with branching ratio closest
    to 1 is chosen: rate alone cannot tell "18 driven cells and dead recurrence" from
    "a network propagating activity".

    Returns ``(bias, trace)``. ``trace["gain_scale"]`` multiplies ``synaptic_gain``,
    ``trace["global_inhibition"]`` sets the pooled inhibition; ``trace["saturated"]`` is True if nothing worked, and
    in that case every downstream number describes a saturated network and must not be
    read as a property of the wiring.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=steps)
    lo, hi = target_rate_hz
    tried: list[dict] = []

    def probe(bias: float, gain_scale: float, inhibition: float) -> float:
        sim = sim_factory()
        sim.p = replace(sim.p,
                        synaptic_gain=sim.p.synaptic_gain * gain_scale,
                        global_inhibition=inhibition)
        st = collect_states(sim, u, input_neurons=input_neurons,
                            amplitude=amplitude, bias=bias, washout=min(40, steps // 4))
        sigma, _ = branching_ratio(st.pop_spikes)
        tried.append({"bias": float(bias), "gain_scale": float(gain_scale),
                      "global_inhibition": float(inhibition),
                      "mean_rate_hz": round(st.mean_rate_hz, 3),
                      "fraction_active": round(st.fraction_active, 4),
                      "branching_ratio": None if np.isnan(sigma) else round(sigma, 3)})
        return st.mean_rate_hz

    in_range: list[dict] = []
    for inhibition in inhibition_candidates:
        for gain_scale in gain_candidates:
            for b in candidates:
                rate = probe(b, gain_scale, inhibition)
                if lo <= rate <= hi:
                    in_range.append(tried[-1])
        # deliberately NOT breaking here: the first inhibition level that merely lands in
        # the rate window is usually a tiny-gain setting where the recurrence is dead.
        # Sweep every level and choose globally.
    if in_range:
            # among settings with an acceptable firing rate, prefer the one closest to
            # criticality: rate alone cannot distinguish "18 driven cells and no
            # recurrence" from "a network actually propagating activity"
        best = min(in_range, key=lambda r: abs((r.get("branching_ratio") or 0.0) - 1.0))
        return float(best["bias"]), {
            "chosen_bias": float(best["bias"]),
            "gain_scale": float(best["gain_scale"]),
            "global_inhibition": float(best.get("global_inhibition", 0.0)),
            "branching_ratio": best.get("branching_ratio"),
            "target_rate_hz": [lo, hi], "tried": tried,
            "saturated": False,
            "note": "" if abs((best.get("branching_ratio") or 0.0) - 1.0) <= 0.15 else
                    (f"best available branching ratio is {best.get('branching_ratio')}, "
                     "not near 1: the network is in range by firing rate but is not "
                     "propagating activity well"),
        }
        # if even zero drive is above the window at this gain, more bias cannot help;
        # drop the gain and try again
    mid = 0.5 * (lo + hi)
    best = min(tried, key=lambda r: abs(r["mean_rate_hz"] - mid))
    note = (f"NO (bias, gain) combination produced a mean rate in [{lo}, {hi}] Hz. Closest was "
            f"{best['mean_rate_hz']} Hz at bias {best['bias']}, gain x{best['gain_scale']}. "
            "The network is outside its usable dynamic range and the measurements below do "
            "NOT describe the wiring.")
    return float(best["bias"]), {"chosen_bias": float(best["bias"]),
                                 "gain_scale": float(best["gain_scale"]),
                                 "global_inhibition": float(best.get("global_inhibition", 0.0)),
                                 "target_rate_hz": [lo, hi], "tried": tried,
                                 "saturated": True, "note": note}


def branching_ratio(pop_spikes: np.ndarray) -> tuple[float, str]:
    """Estimate sigma = E[A(t+1) | A(t)] / A(t): how many spikes one spike begets.

    The single most useful summary of where a recurrent network sits.
      sigma < 1  subcritical -- activity dies out; the network forgets immediately
      sigma ~ 1  critical    -- the regime with the longest memory and richest dynamics
      sigma > 1  supercritical -- activity amplifies into saturation (our 433 Hz run)

    Estimated by least-squares regression of A(t+1) on A(t) through the origin, over
    steps where A(t) > 0. Returns (sigma, verdict). This is a coarse estimator -- it
    assumes a linear relationship and ignores the external drive, which inflates it --
    so it is used for steering the calibration, not reported as a physics result.
    """
    a = np.asarray(pop_spikes, dtype=np.float64)
    if a.size < 3:
        return float("nan"), "too few steps"
    x, y = a[:-1], a[1:]
    mask = x > 0
    if mask.sum() < 3:
        return 0.0, "subcritical (the network is essentially silent)"
    sigma = float((x[mask] @ y[mask]) / (x[mask] @ x[mask]))
    if sigma < 0.85:
        verdict = "subcritical -- activity dies out, little memory"
    elif sigma > 1.15:
        verdict = "supercritical -- activity amplifies toward saturation"
    else:
        verdict = "near-critical -- the regime where a reservoir works"
    return sigma, verdict


# -------------------------------------------------------------------- readout
def ridge_readout(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                  *, alpha: float = 1e-6) -> np.ndarray:
    """Closed-form ridge regression. Returns predictions for ``X_test``.

    Ridge, not gradient descent: the readout must be a deterministic, seedless
    function of the states so that any effect we see belongs to the reservoir.
    A bias column is added so the decoder can centre its output.
    """
    A = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    B = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    n_feat = A.shape[1]
    # solve (AᵀA + alpha*I) w = Aᵀy  -- scaled so alpha means the same thing at
    # any state magnitude
    G = A.T @ A
    scale = float(np.trace(G) / max(n_feat, 1)) or 1.0
    w = np.linalg.solve(G + alpha * scale * np.eye(n_feat), A.T @ y_train)
    return B @ w


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, clipped to [0, 1].

    Clipped because MC_k is conventionally a squared correlation in [0, 1]; a
    readout that is worse than predicting the mean carries no memory, it does
    not carry negative memory.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


# ------------------------------------------------------------ memory capacity
@dataclass
class MemoryCapacity:
    total: float
    per_delay: list[float]
    max_delay: int
    n_train: int
    n_test: int
    half_life_delay: int | None
    note: str = ""

    def describe(self) -> str:
        hl = "never reaches 0.5" if self.half_life_delay is None else f"MC_k drops below 0.5 at k={self.half_life_delay}"
        return (f"memory capacity {self.total:.2f} effective steps over delays 1..{self.max_delay} "
                f"({hl}); held-out n={self.n_test}")


def memory_capacity(states: ReservoirStates, *, max_delay: int = 20,
                    train_fraction: float = 0.7, alpha: float = 1e-6) -> MemoryCapacity:
    """Jaeger memory capacity on held-out timesteps.

    For each delay k, a ridge readout is fitted on the first ``train_fraction``
    of the run to reconstruct ``u[t-k]`` from the state at t, and scored on the
    remaining timesteps. The split is contiguous, not random: shuffling
    timesteps would leak information across the temporal correlations we are
    trying to measure.
    """
    X, u = states.X, states.u
    n = X.shape[0]
    if n < 20:
        return MemoryCapacity(0.0, [], max_delay, 0, 0, None, "too few timesteps to estimate")
    per: list[float] = []
    n_train = n_test = 0
    for k in range(1, max_delay + 1):
        if k >= n - 10:
            per.append(0.0)
            continue
        Xk, yk = X[k:], u[:-k]
        cut = int(len(Xk) * train_fraction)
        if cut < 5 or len(Xk) - cut < 5:
            per.append(0.0)
            continue
        pred = ridge_readout(Xk[:cut], yk[:cut], Xk[cut:], alpha=alpha)
        per.append(_r2(yk[cut:], pred))
        n_train, n_test = cut, len(Xk) - cut
    half = next((i + 1 for i, v in enumerate(per) if v < 0.5), None)
    return MemoryCapacity(total=float(np.sum(per)), per_delay=[float(v) for v in per],
                          max_delay=max_delay, n_train=n_train, n_test=n_test,
                          half_life_delay=half)


# ------------------------------------------------------------------ separation
@dataclass
class Separation:
    ratio: float
    between_input_distance: float
    within_input_distance: float
    n_pairs: int
    note: str = ""

    def describe(self) -> str:
        if self.ratio > 100:
            # a ratio in the thousands is not 100x better than a ratio of 10. It means
            # nearby inputs diverge exponentially -- chaos, not computation. Such a
            # network separates everything, including two copies of the same input plus
            # noise, which is why generalisation rank collapses to kernel rank.
            verdict = ("IMPLAUSIBLY HIGH -- this is chaotic amplification, not useful "
                       "separation; check that generalisation rank has not collapsed")
        elif self.ratio > 1.5:
            verdict = "distinct inputs are distinguishable in state space"
        else:
            verdict = "inputs are NOT well separated: a decoder has little to work with"
        return f"separation ratio {self.ratio:.2f} ({verdict}); n_pairs={self.n_pairs}"


def separation(
    sim_factory,
    *,
    input_neurons: np.ndarray,
    n_pairs: int = 4,
    steps: int = 200,
    washout: int = 50,
    noise_scale: float = 0.05,
    amplitude: float = 22.0,
    bias: float = 0.0,
    seed: int = 0,
) -> Separation:
    """Legenstein & Maass separation property.

    ``sim_factory()`` must return a *fresh* simulator with identical weights, so
    that each input stream starts from the same initial state. Distances are
    Euclidean between final filtered states, averaged over pairs.
    """
    rng = np.random.default_rng(seed)
    between: list[float] = []
    within: list[float] = []
    for _ in range(n_pairs):
        a = rng.uniform(0.0, 1.0, size=steps)
        b = rng.uniform(0.0, 1.0, size=steps)
        a_noisy = a + rng.normal(0.0, noise_scale, size=steps)
        sa = collect_states(sim_factory(), a, input_neurons=input_neurons,
                            amplitude=amplitude, bias=bias, washout=washout)
        sb = collect_states(sim_factory(), b, input_neurons=input_neurons,
                            amplitude=amplitude, bias=bias, washout=washout)
        sn = collect_states(sim_factory(), a_noisy, input_neurons=input_neurons,
                            amplitude=amplitude, bias=bias, washout=washout)
        between.append(float(np.linalg.norm(sa.X[-1] - sb.X[-1])))
        within.append(float(np.linalg.norm(sa.X[-1] - sn.X[-1])))
    b_mean = float(np.mean(between)) if between else 0.0
    w_mean = float(np.mean(within)) if within else 0.0
    if w_mean <= 1e-12:
        note = ("within-input distance is ~0: either the network is silent, or it is "
                "perfectly deterministic and noise-insensitive. Check fraction_active before "
                "reading the ratio.")
        ratio = float("inf") if b_mean > 1e-12 else 0.0
        return Separation(ratio, b_mean, w_mean, len(between), note)
    return Separation(b_mean / w_mean, b_mean, w_mean, len(between))


# ----------------------------------------------------------------- rank / dim
def rank_measures(
    sim_factory,
    *,
    input_neurons: np.ndarray,
    n_streams: int = 8,
    steps: int = 150,
    washout: int = 40,
    noise_scale: float = 0.02,
    amplitude: float = 22.0,
    bias: float = 0.0,
    seed: int = 0,
    tol: float | None = None,
) -> dict:
    """Kernel rank (distinct inputs) and generalisation rank (noisy repeats)."""
    rng = np.random.default_rng(seed)
    distinct, noisy = [], []
    base = rng.uniform(0.0, 1.0, size=steps)
    for _ in range(n_streams):
        u = rng.uniform(0.0, 1.0, size=steps)
        distinct.append(collect_states(sim_factory(), u, input_neurons=input_neurons,
                                       amplitude=amplitude, bias=bias, washout=washout).X[-1])
        v = base + rng.normal(0.0, noise_scale, size=steps)
        noisy.append(collect_states(sim_factory(), v, input_neurons=input_neurons,
                                    amplitude=amplitude, bias=bias, washout=washout).X[-1])
    K = np.vstack(distinct)
    G = np.vstack(noisy)
    kr = int(np.linalg.matrix_rank(K, tol=tol))
    gr = int(np.linalg.matrix_rank(G, tol=tol))
    return {
        "kernel_rank": kr,
        "generalisation_rank": gr,
        "separation_rank": kr - gr,
        "n_streams": n_streams,
        "max_possible_rank": int(min(K.shape)),
        "note": ("kernel_rank is capped by n_streams; raise n_streams if it saturates"
                 if kr >= n_streams else ""),
    }


# ------------------------------------------------------------------- controls
def shuffle_connectome(conn, *, seed: int = 0):
    """Degree-preserving target shuffle: same out-degree and weight multiset.

    This is the control every number in :func:`characterise` should be read
    against. If the real wiring scores no better than its shuffle, the result is
    a property of the degree distribution, not of the connectome.
    """
    import copy

    rng = np.random.default_rng(seed)
    out = copy.deepcopy(conn)
    m: sparse.csr_matrix = out.matrix
    n = m.shape[1]
    indptr = m.indptr
    new_idx = np.asarray(m.indices).copy()
    for i in range(m.shape[0]):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        if hi - lo > 1:
            new_idx[lo:hi] = rng.choice(n, size=hi - lo, replace=False)
    m.indices[:] = new_idx
    m.sort_indices()
    out.provenance = dict(getattr(out, "provenance", {}) or {})
    out.provenance["control"] = f"degree-preserving target shuffle (seed={seed})"
    return out


# ------------------------------------------------------------------- top level
@dataclass
class ReservoirReport:
    n_neurons: int
    n_edges: int
    n_input: int
    n_readout: int
    mean_rate_hz: float
    fraction_active: float
    memory: MemoryCapacity
    sep: Separation
    ranks: dict
    sigma: float = float("nan")
    sigma_verdict: str = ""
    params: dict = field(default_factory=dict)
    control: str = "none (real wiring)"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["memory"] = asdict(self.memory)
        d["sep"] = asdict(self.sep)
        return d

    def describe(self) -> str:
        lines = [
            f"reservoir: {self.n_neurons:,} neurons / {self.n_edges:,} edges  [{self.control}]",
            f"  activity        : {self.mean_rate_hz:.2f} Hz mean, "
            f"{100 * self.fraction_active:.1f}% of neurons ever spiked",
            f"  branching ratio : {self.sigma:.3f} ({self.sigma_verdict})",
            f"  {self.memory.describe()}",
            f"  {self.sep.describe()}",
            f"  kernel rank {self.ranks['kernel_rank']} - generalisation rank "
            f"{self.ranks['generalisation_rank']} = {self.ranks['separation_rank']} "
            f"(max possible {self.ranks['max_possible_rank']})",
        ]
        cal = (self.params or {}).get("drive_calibration") or {}
        if cal.get("saturated"):
            lines.append("  *** SATURATED: " + str(cal.get("note", "")).strip())
            lines.append("  *** The memory/separation/rank numbers above are properties of a "
                         "saturated network, NOT of the connectome. Do not compare arms.")
        if self.mean_rate_hz > 150:
            lines.append(f"  WARNING: {self.mean_rate_hz:.0f} Hz mean rate is far outside any "
                         "plausible range; treat every number above as uninterpretable.")
        if self.fraction_active < 0.01:
            lines.append("  WARNING: the network is essentially silent. Every number above "
                         "describes a silent network, not the wiring. Raise the drive amplitude.")
        return "\n".join(lines)


def characterise(
    conn,
    params=None,
    *,
    input_neurons: np.ndarray,
    readout_neurons: np.ndarray | None = None,
    steps: int = 600,
    washout: int = 100,
    max_delay: int = 20,
    amplitude: float = 22.0,
    bias: float | None = None,
    seed: int = 0,
    n_sep_pairs: int = 3,
    n_rank_streams: int = 8,
    control: str = "none",
) -> ReservoirReport:
    """Full frozen-connectome characterisation. No weight is ever modified."""
    if control in {"shuffled", "shuffled_wiring", "shuffle"}:
        conn = shuffle_connectome(conn, seed=seed)
        control_note = conn.provenance.get("control", "shuffled")
    elif control in {"", "none"}:
        control_note = "none (real wiring)"
    else:
        raise ValueError(f"unknown control {control!r} (use 'none' or 'shuffled')")

    weights_before = np.asarray(conn.matrix.data).copy()

    gain_scale = 1.0
    inhibition: float | None = None

    def factory() -> NetworkSimulator:
        sim = NetworkSimulator(conn, params, seed=seed)
        if gain_scale != 1.0 or inhibition is not None:
            sim.p = replace(
                sim.p,
                synaptic_gain=sim.p.synaptic_gain * gain_scale,
                global_inhibition=sim.p.global_inhibition if inhibition is None else inhibition,
            )
        return sim

    if bias is None:
        bias, cal = calibrate_drive(factory, input_neurons=input_neurons,
                                    amplitude=amplitude, seed=seed)
        gain_scale = float(cal.get("gain_scale", 1.0))
        inhibition = float(cal.get("global_inhibition", 0.0))
    else:
        cal = {"bias": bias, "note": "bias supplied by caller, not calibrated"}

    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=steps)
    st = collect_states(factory(), u, input_neurons=input_neurons,
                        readout_neurons=readout_neurons, amplitude=amplitude,
                        bias=bias, washout=washout)
    mc = memory_capacity(st, max_delay=max_delay)
    sigma, sigma_verdict = branching_ratio(st.pop_spikes)
    sep = separation(factory, input_neurons=input_neurons, n_pairs=n_sep_pairs,
                     steps=max(120, steps // 4), washout=min(washout, 40),
                     amplitude=amplitude, bias=bias, seed=seed)
    ranks = rank_measures(factory, input_neurons=input_neurons, n_streams=n_rank_streams,
                          steps=max(120, steps // 4), washout=min(washout, 40),
                          amplitude=amplitude, bias=bias, seed=seed)

    # the whole premise is that the connectome is frozen; assert it rather than trust it
    if not np.array_equal(weights_before, np.asarray(conn.matrix.data)):
        raise AssertionError("connectome weights changed during characterisation -- "
                             "this measurement is only meaningful on a frozen reservoir")

    return ReservoirReport(
        n_neurons=int(conn.n_neurons), n_edges=int(conn.n_edges),
        n_input=int(np.size(input_neurons)), n_readout=st.n_units,
        mean_rate_hz=st.mean_rate_hz, fraction_active=st.fraction_active,
        memory=mc, sep=sep, ranks=ranks, control=control_note,
        sigma=sigma, sigma_verdict=sigma_verdict,
        params={"steps": steps, "washout": washout, "max_delay": max_delay,
                "amplitude": amplitude, "bias": bias, "gain_scale": gain_scale,
                "global_inhibition": inhibition,
                "drive_calibration": cal, "seed": seed,
                "n_sep_pairs": n_sep_pairs, "n_rank_streams": n_rank_streams},
    )
