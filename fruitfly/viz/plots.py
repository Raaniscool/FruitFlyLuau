"""Matplotlib figures: activity, connectivity sample, reward, learning, weights.

All plotting is optional (``pip install 'fruitflyluau[viz]'``) and defensive: a
missing backend degrades to a warning instead of killing a training run, since
the numbers -- not the pictures -- are the product here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("matplotlib unavailable (%s); skipping figures", exc)
        return None


def plot_activity(rates: np.ndarray, counts: np.ndarray, *, out: str = "activity.png", dpi: int = 110) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(np.asarray(rates), lw=1.0)
    ax[0].set_title("population spikes per step")
    ax[0].set_xlabel("step")
    ax[1].hist(np.asarray(counts), bins=30, color="steelblue")
    ax[1].set_title("spikes per neuron")
    ax[1].set_xlabel("count")
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_raster(spikes: np.ndarray, *, out: str = "raster.png", max_neurons: int = 40, dpi: int = 110) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    s = np.asarray(spikes)[: max(1, max_neurons), :]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.eventplot([np.flatnonzero(s[i]) for i in range(s.shape[0])], lineoffsets=np.arange(s.shape[0]), linelengths=0.8)
    ax.set_xlabel("time step")
    ax.set_ylabel("neuron (first %d)" % s.shape[0])
    ax.set_title("spike raster (sampled)")
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_connectivity_sample(conn, *, out: str = "connectivity.png", max_nodes: int = 60, max_edges: int = 4000, dpi: int = 110) -> Path | None:
    """Sampled layout of the graph actually being simulated (never the whole fly)."""
    plt = _plt()
    if plt is None:
        return None
    n = min(max_nodes, conn.n_neurons)
    idx = np.linspace(0, conn.n_neurons - 1, n).round().astype(int)
    sub = conn.matrix[idx][:, idx]
    pos = np.column_stack([np.cos(2 * np.pi * np.arange(n) / n), np.sin(2 * np.pi * np.arange(n) / n)])
    in_deg, out_deg = conn.degrees()
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    rows, cols = sub.nonzero()
    w = np.abs(sub.data)
    if rows.size > max_edges:
        pick = np.argsort(-w)[:max_edges]
        rows, cols, w = rows[pick], cols[pick], w[pick]
    for r, c, ww in zip(rows[:max_edges], cols[:max_edges], w[:max_edges]):
        ax.plot([pos[r, 0], pos[c, 0]], [pos[r, 1], pos[c, 1]], lw=0.35, color="0.6", alpha=0.5, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=12 + 900 * (in_deg[idx] + out_deg[idx]) / max(1.0, (in_deg + out_deg).max()),
               c="tab:blue", zorder=2)
    ax.set_title(f"sampled subgraph: {n} of {conn.n_neurons:,} neurons, {sub.nnz:,} edges shown")
    ax.set_axis_off()
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def _legend(ax) -> None:
    """Only call legend() when something is actually labelled (matplotlib warns otherwise)."""
    labelled = [t for t in ax.get_texts() if t.get_text()] if False else []
    artists = list(ax.get_lines()) + list(ax.patches) + list(ax.collections)
    if any(getattr(a, "get_label", lambda: "")() not in ("", None) and not str(
            getattr(a, "get_label", lambda: "")()).startswith("_") for a in artists):
        ax.legend(fontsize=7)
    del labelled


def plot_learning(runner, *, out_dir: str = "figures", dpi: int = 110, spatial: bool = True) -> list[Path]:
    """Reward, accuracy, weight statistics and (optionally) spatial activity."""
    plt = _plt()
    if plt is None:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = runner.metrics

    def series(key: str) -> np.ndarray:
        """A logged column, or an empty array. Never assumes record counts match.

        ``metrics.jsonl`` holds both training rows and evaluation rows, so a given
        key can legitimately be shorter than ``len(records)`` -- plotting must use
        each series' own length rather than a shared episode axis.
        """
        try:
            y = np.asarray(m.series(key), dtype=np.float64)
        except Exception:  # pragma: no cover - defensive
            return np.zeros(0)
        return y[np.isfinite(y)] if y.dtype.kind == "f" else y

    window = 25

    def curve(ax, key, *, color: str, title: str, ylabel: str = "", ylim=None,
              reference: float | None = None, ref_label: str = "", extra=None):
        y = series(key)
        ax.set_title(title if y.size else f"{title} (no data logged)")
        ax.set_xlabel("episode")
        if ylabel:
            ax.set_ylabel(ylabel)
        if y.size == 0:
            return
        x = np.arange(y.size)
        ax.plot(x, y, lw=0.6, color="0.8")
        if y.size > window:
            sm = np.convolve(y, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(sm.size) + window - 1, sm, lw=1.6, color=color,
                    label=f"{window}-episode mean")
        if reference is not None:
            ax.axhline(reference, color="r", ls="--", lw=1.0, label=ref_label)
        if ylim:
            ax.set_ylim(*ylim)
        if extra:
            extra(ax, x, y)
        _legend(ax)

    fig, ax = plt.subplots(2, 3, figsize=(14, 6.4))
    from ..experiment.metrics import chance_level

    curve(ax[0, 0], "reward", color="tab:orange", title="reward per episode", reference=0.0)
    curve(ax[0, 1], "correct", color="tab:green", title="accuracy per episode", ylim=(-0.05, 1.05),
          reference=chance_level(len(runner.env.classes)), ref_label="chance")
    curve(ax[0, 2], "rate", color="tab:blue", title="population firing rate (Hz)")
    curve(ax[1, 0], "weight_mean", color="tab:purple", title="mean |w| over training",
          extra=lambda a_, x_, y_: a_.plot(x_, series("weight_std"), lw=1.0, color="tab:red", label="std"))
    curve(ax[1, 1], "edges_updated", color="tab:brown", title="edges changed per episode",
          extra=lambda a_, x_, y_: a_.plot(x_, series("abs_dw"), lw=0.8, color="k", label="|dW| sum"))

    w = np.asarray(runner.conn.weights, dtype=np.float64)
    hist, bin_edges = np.histogram(w, bins=40)
    ax[1, 2].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", color="steelblue", label="now")
    w0 = np.asarray(getattr(runner, "w0", w), dtype=np.float64)
    if w0.size == w.size and w.size:
        h0, _ = np.histogram(w0, bins=bin_edges)
        ax[1, 2].step(bin_edges[:-1], h0, where="post", color="0.5", lw=1.0, label="initial")
    ax[1, 2].set_title("weight distribution (now vs init)")
    ax[1, 2].set_xlabel("|w|")
    ax[1, 2].legend(fontsize=7)
    fig.tight_layout()
    paths = [out / "learning.png"]
    fig.savefig(paths[0], dpi=dpi)
    plt.close(fig)

    if spatial and runner.conn.coords is not None:
        coords = np.asarray(runner.conn.coords, dtype=np.float64)
        if np.isfinite(coords).any():
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            ok = np.isfinite(coords).all(axis=1)
            counts = np.zeros(runner.conn.n_neurons)
            sc = np.asarray(runner.sim.spike_count, dtype=float)
            sc = sc / max(sc.max(), 1.0)
            im = ax.scatter(coords[ok, 0], coords[ok, 1], c=sc[ok], cmap="viridis", s=8)
            fig.colorbar(im, ax=ax, label="normalised spikes")
            ax.set_title("spatial activity (marked-neuron coordinates, if loaded)")
            p = out / "spatial_activity.png"
            fig.tight_layout()
            fig.savefig(p, dpi=dpi)
            plt.close(fig)
            paths.append(p)

    paths.append(plot_connectivity_sample(runner.conn, out=str(out / "connectivity.png"), dpi=dpi) or out / "connectivity.png")
    log.info("figures written to %s", out)
    return [p for p in paths if p]
