"""Neuron population selection: which slice of the connectome gets simulated.

Modes are pluggable (see :data:`REGISTRY`) so later phases can ask things like
"only cholinergic visual projection neurons" without touching the simulator.
All selection works on *integer codes* into the sorted id array, so nothing here
scales with the number of Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import sparse

from ..utils import get_logger

log = get_logger(__name__)

Selector = Callable[["SelectionContext"], np.ndarray]
_REGISTRY: dict[str, Selector] = {}


def register(name: str) -> Callable[[Selector], Selector]:
    def deco(fn: Selector) -> Selector:
        _REGISTRY[name] = fn
        return fn

    return deco


@dataclass
class SelectionContext:
    """Everything a selector may look at."""

    ids: np.ndarray  # sorted int64 neuron ids of the loaded graph
    n: int
    adjacency: sparse.csr_matrix | None = None  # over ``ids`` (full loaded graph)
    metadata: dict[str, np.ndarray] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    kwargs: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    def meta(self, key: str) -> np.ndarray | None:
        return self.metadata.get(key)

    def match(self, key: str, values: str | Sequence[str] | None, *, regex: str | None = None) -> np.ndarray:
        """Boolean mask over neurons where metadata ``key`` equals one of ``values`` (or matches regex)."""
        col = self.meta(key)
        mask = np.zeros(self.n, dtype=bool)
        if col is None:
            log.warning("selection needs metadata %r, which is not available; selecting nothing", key)
            return mask
        arr = np.asarray([("" if v is None else str(v)) for v in col], dtype="<U64")
        if regex is not None:
            import re

            pat = re.compile(regex, re.IGNORECASE)
            return np.array([bool(pat.search(v)) for v in arr], dtype=bool)
        if isinstance(values, str):
            values = [values]
        wanted = {str(v).strip().lower() for v in (values or [])}
        return np.array([v.strip().lower() in wanted for v in arr], dtype=bool)

    def by_weight(self, count: int, *, degree: str = "total") -> np.ndarray:
        """Indices of the ``count`` highest-degree neurons (a cheap 'hub' prior)."""
        if self.adjacency is None:
            return self.rng.permutation(min(count, self.n))[:count]
        out_deg = np.diff(self.adjacency.indptr)
        in_deg = np.asarray((self.adjacency != 0).sum(axis=0)).ravel()
        score = {"out": out_deg, "in": in_deg}.get(degree, out_deg + in_deg)
        count = int(min(max(count, 1), self.n))
        if count >= self.n:
            return np.arange(self.n, dtype=np.int64)
        part = np.argpartition(-score, count)[:count]
        return np.sort(part)


def get(name: str) -> Selector:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown selection {name!r}; available: {sorted(_REGISTRY)}") from None


def list_selectors() -> list[str]:
    return sorted(_REGISTRY)


# ------------------------------------------------------------------- selectors
@register("random")
def _random(ctx: SelectionContext) -> np.ndarray:
    k = int(ctx.kwargs.get("n", 100))
    return ctx.rng.choice(ctx.n, size=min(k, ctx.n), replace=False)


@register("hubs")
def _hubs(ctx: SelectionContext) -> np.ndarray:
    k = int(ctx.kwargs.get("n", 100))
    return ctx.by_weight(k, degree=ctx.kwargs.get("degree", "total"))


@register("manual")
def _manual(ctx: SelectionContext) -> np.ndarray:
    """Explicit FlyWire root ids, e.g. ``ids=[720575940603231916, ...]``."""
    raw = ctx.kwargs.get("ids") or ctx.kwargs.get("root_ids")
    if not raw:
        raise ValueError("selection 'manual' needs kwargs.ids = [root_id, ...]")
    want = np.unique(np.asarray(list(raw), dtype=np.int64))
    pos = np.searchsorted(ctx.ids, want)
    inb = (pos < ctx.n) & (ctx.ids[np.clip(pos, 0, max(ctx.n - 1, 0))] == want)
    missing = want[~inb]
    if missing.size:
        log.warning("%d requested neuron id(s) are not in the loaded graph: %s", missing.size, missing[:5])
    return pos[inb].astype(np.int64)


@register("neuron_type")
def _neuron_type(ctx: SelectionContext) -> np.ndarray:
    """Filter by a categorical annotation column, e.g. ``field="cell_types.primary_type"``."""
    field_key = ctx.kwargs.get("field", "cell_types.primary_type")
    values = ctx.kwargs.get("values")
    regex = ctx.kwargs.get("regex")
    if values is None and regex is None:
        raise ValueError("selection 'neuron_type' needs kwargs.values or kwargs.regex")
    mask = ctx.match(field_key, values, regex=regex)
    k = int(ctx.kwargs.get("n", 0))
    idx = np.flatnonzero(mask)
    if k and idx.size > k:
        idx = ctx.rng.choice(idx, size=k, replace=False)
    if idx.size == 0:
        log.warning("neuron_type selection matched 0 neurons on %r", field_key)
    return idx


@register("neurotransmitter")
def _nt(ctx: SelectionContext) -> np.ndarray:
    """Filter by predicted transmitter: ``values="gaba"`` or e.g. ``["glut", "cholin"]``."""
    values = ctx.kwargs.get("values", "gaba")
    mask = ctx.match("nt_predictions.nt_type", values)
    idx = np.flatnonzero(mask)
    k = int(ctx.kwargs.get("n", 0))
    if k and idx.size > k:
        idx = ctx.rng.choice(idx, size=k, replace=False)
    return idx


@register("mushroom_body")
def _mushroom_body(ctx: SelectionContext) -> np.ndarray:
    """The associative-learning circuit, selected by `classification.class`.

    Measured in the real v783 download (see fafb_schema_report.md section 12): the
    `class` column contains Kenyon_Cell, MBON, MBIN and DAN as literal values, so this
    circuit can be selected with the annotations the portal already ships -- no extra
    file, no manual id list.

    Why this circuit: it is the one the fly genuinely uses for associative learning,
    which makes it the honest first target for a learning experiment. Selecting it does
    NOT mean the model learns the way the fly does -- the dynamics and the plasticity
    rule remain our inventions.

    kwargs: ``classes`` (default the four above), ``n`` (cap), ``sample`` --
    "connected" (default; greedy densest connected subgraph within the circuit) or
    "random" (an explicit baseline that is mostly isolated cells).
    """
    classes = ctx.kwargs.get("classes") or ["Kenyon_Cell", "MBON", "MBIN", "DAN"]
    mask = ctx.match("classification.class", classes)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        log.warning(
            "mushroom_body selection matched 0 neurons. It needs classification.csv.gz; "
            "without it the class column is unavailable and nothing can be selected by cell class."
        )
        return idx
    total = idx.size
    k = int(ctx.kwargs.get("n", 0))
    if not k or total <= k:
        log.info("mushroom_body: %d neurons selected (no cap applied)", total)
        return np.sort(idx)

    how = str(ctx.kwargs.get("sample", "connected")).lower()
    if how == "random":
        # kept as an explicit, honest baseline: a random k of the circuit is mostly
        # disconnected, because the MB is sparse and k is small relative to it.
        picked = ctx.rng.choice(idx, size=k, replace=False)
        log.warning(
            "mushroom_body: random sample of %d/%d neurons -- expect most of them to be "
            "isolated. Use sample='connected' for a graph that can carry activity.", k, total,
        )
        return np.sort(picked)

    picked = _densest_connected_within(ctx, idx, k)
    log.info("mushroom_body: %d/%d neurons, grown as a connected subgraph within the circuit",
             picked.size, total)
    return picked


def _densest_connected_within(ctx: SelectionContext, members: np.ndarray, k: int) -> np.ndarray:
    """Grow a connected set of <=k nodes using ONLY edges internal to ``members``.

    Why this is needed, measured: the mushroom body has roughly 5k neurons spread over
    Kenyon cells, MBONs, DANs and MBINs. A uniform random 100 of them shares almost no
    edges -- the first real run produced 36 edges and 62 isolated neurons out of 100,
    which cannot carry activity no matter what the dynamics are.

    Strategy: restrict the adjacency to ``members``, start from the highest internal-degree
    node, and repeatedly add the candidate with the most edges to the set already chosen
    (a greedy densest-subgraph expansion). Ties are broken by the context rng, so the
    result is seeded and reproducible.
    """
    if ctx.adjacency is None:
        log.warning("no adjacency available; falling back to a random sample of the circuit")
        return np.sort(ctx.rng.choice(members, size=k, replace=False))

    sub = ctx.adjacency[members][:, members].tocsr()
    sym = (sub + sub.T).tocsr()          # connectivity, direction-agnostic for growth
    deg = np.asarray((sym != 0).sum(axis=1)).ravel()
    if deg.max(initial=0) == 0:
        log.warning("the selected circuit has no internal edges at all; returning a random sample")
        return np.sort(ctx.rng.choice(members, size=k, replace=False))

    start = int(np.argmax(deg))
    chosen = [start]
    in_set = np.zeros(members.size, dtype=bool)
    in_set[start] = True
    # score[i] = number of edges from candidate i into the chosen set
    score = np.asarray(sym[:, start].todense()).ravel().astype(np.int64)
    score[start] = -1

    while len(chosen) < k:
        best = int(np.argmax(score))
        if score[best] <= 0:
            # the connected component is exhausted; restart from the best unused node
            remaining = np.flatnonzero(~in_set)
            if remaining.size == 0:
                break
            best = int(remaining[np.argmax(deg[remaining])])
            if deg[best] == 0:
                break
        in_set[best] = True
        chosen.append(best)
        score += np.asarray(sym[:, best].todense()).ravel().astype(np.int64)
        score[in_set] = -1

    return np.sort(members[np.asarray(chosen, dtype=np.int64)])


@register("visual")
def _visual(ctx: SelectionContext) -> np.ndarray:
    mask = ctx.match("visual_types.type", ctx.kwargs.get("values"), regex=ctx.kwargs.get("regex"))
    if not mask.any():  # fall back to any visual annotation at all
        col = ctx.meta("visual_types.type")
        if col is not None:
            mask = np.asarray([v not in (None, "", "nan") for v in col], dtype=bool)
    idx = np.flatnonzero(mask)
    k = int(ctx.kwargs.get("n", 0))
    if k and idx.size > k:
        idx = ctx.rng.choice(idx, size=k, replace=False)
    return idx


def _neighbour_operator(adj, *, reverse: bool):
    """CSR view whose *rows* are the neighbours we want to expand.

    ``adj`` is CSR with rows = presynaptic, so rows are "outputs". For upstream
    traversal we need the transpose as a real CSR matrix (one conversion, cached
    by the caller via :func:`upstream_ancestors`), because ``adj[rows]`` on a CSC
    matrix would silently slice *rows* again and walk the wrong direction.
    """
    from scipy import sparse as _sp

    return (_sp.csr_matrix(adj.T) if reverse else adj)


def _bfs_layers(adj, seeds: np.ndarray, depth: int, *, reverse: bool = False) -> np.ndarray:
    """Neurons reachable from ``seeds`` within ``depth`` hops (forward or upstream)."""
    op = _neighbour_operator(adj, reverse=reverse)
    seen = np.zeros(adj.shape[0], dtype=bool)
    seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
    seen[seeds] = True
    frontier = seeds
    for _ in range(max(0, int(depth))):
        if frontier.size == 0:
            break
        nxt = np.unique(op[frontier].indices)
        nxt = nxt[~seen[nxt]]
        if nxt.size == 0:
            break
        seen[nxt] = True
        frontier = nxt
    return np.flatnonzero(seen)


def _seeds(ctx: SelectionContext) -> np.ndarray:
    raw = ctx.kwargs.get("seeds") or ctx.kwargs.get("ids")
    if raw:
        want = np.unique(np.asarray(list(raw), dtype=np.int64))
        pos = np.searchsorted(ctx.ids, want)
        inb = (pos < ctx.n) & (ctx.ids[np.clip(pos, 0, max(ctx.n - 1, 0))] == want)
        return pos[inb].astype(np.int64)
    k = int(ctx.kwargs.get("n_seeds", 1))
    return ctx.rng.choice(ctx.n, size=min(k, ctx.n), replace=False)


def _trophic_selector(forward: bool) -> Selector:
    def fn(ctx: SelectionContext) -> np.ndarray:
        if ctx.adjacency is None:
            raise ValueError(f"selection {'downstream' if forward else 'upstream'}_of needs an adjacency matrix")
        seeds = _seeds(ctx)
        depth = int(ctx.kwargs.get("depth", 2))
        layers = _bfs_layers(ctx.adjacency, seeds, depth, reverse=not forward)
        if ctx.kwargs.get("exclude_seeds", False):
            layers = np.setdiff1d(layers, np.atleast_1d(seeds))
        out = layers
        cap = int(ctx.kwargs.get("n", 0))
        if cap and out.size > cap:
            out = ctx.rng.choice(out, size=cap, replace=False)
        return np.sort(out)

    return fn


_REGISTRY["downstream_of"] = _trophic_selector(True)
_REGISTRY["upstream_of"] = _trophic_selector(False)


@register("connected_subgraph")
def _connected_subgraph(ctx: SelectionContext) -> np.ndarray:
    """BFS-expanding subgraph so the population is not a bag of isolated cells.

    Seed choice biases towards well-connected neurons (degree prior) because a
    purely random seed in this graph frequently lands in a tiny component.
    """
    k = int(ctx.kwargs.get("n", 100))
    if ctx.adjacency is None or ctx.adjacency.nnz == 0:
        return ctx.rng.choice(ctx.n, size=min(k, ctx.n), replace=False)
    seeds = _seeds(ctx) if ctx.kwargs.get("seeds") else ctx.by_weight(int(ctx.kwargs.get("n_seeds", 12)), degree="out")
    undirected = (ctx.adjacency + ctx.adjacency.T).tocsr()
    grown = _bfs_layers(undirected, np.atleast_1d(seeds), int(ctx.kwargs.get("depth", 6)))
    if grown.size >= k:
        return np.sort(ctx.rng.choice(grown, size=k, replace=False))
    # pad with high-degree neurons to reach the requested size
    extra = np.setdiff1d(ctx.by_weight(k * 3, degree="total"), grown)
    need = k - grown.size
    if extra.size:
        grown = np.concatenate([grown, ctx.rng.choice(extra, size=min(need, extra.size), replace=False)])
    if grown.size < k:  # last resort: random fill so population size always equals request
        rest = np.setdiff1d(np.arange(ctx.n), grown)
        if rest.size:
            grown = np.concatenate([grown, ctx.rng.choice(rest, size=min(k - grown.size, rest.size), replace=False)])
    return np.sort(grown.astype(np.int64))


@register("all")
def _all(ctx: SelectionContext) -> np.ndarray:
    return np.arange(ctx.n, dtype=np.int64)


def select_population(
    mode: str,
    *,
    ids: np.ndarray,
    adjacency: sparse.csr_matrix | None,
    metadata: dict[str, np.ndarray] | None,
    rng: np.random.Generator,
    kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Dispatch to a registered selector, always returning a sorted index array."""
    kwargs = dict(kwargs or {})
    if mode in {"", "none"}:
        return np.arange(ids.size, dtype=np.int64)
    ctx = SelectionContext(
        ids=ids, n=int(ids.size), adjacency=adjacency, metadata=metadata or {}, rng=rng, kwargs=kwargs
    )
    idx = get(mode)(ctx)
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    if idx.size == 0:
        raise ValueError(
            f"selection mode {mode!r} produced an empty population. Check that the "
            f"annotation asset is present (metadata keys available: "
            f"{sorted((metadata or {}).keys())[:8]}...)"
        )
    return idx


# ------------------------------------------------------- input/output assignment
def upstream_ancestors(conn, targets: np.ndarray, *, depth: int = 3) -> np.ndarray:
    """Neurons with a directed path of length <= ``depth`` INTO ``targets``.

    Traverses the transpose, i.e. walks backwards along edges. The targets
    themselves are excluded so a caller can never pick a readout cell as an
    input cell by accident.
    """
    targets = np.atleast_1d(np.asarray(targets, dtype=np.int64))
    seen = _bfs_layers(conn.matrix, targets, depth, reverse=True)
    keep = np.ones(seen.size, dtype=bool)
    mask = np.isin(seen, targets)
    return seen[~mask]


def input_output_sets(
    conn,
    *,
    n_inputs: int,
    n_outputs: int,
    rng,
    depth: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Choose a readout population and a *reachable* input drive population.

    Rationale (measured with ``scripts/calibrate_dynamics.py``, not assumed): a
    uniformly random input set frequently has no directed path to the high
    in-degree "sink" neurons within the trial window, which leaves the readout
    blind and makes the task unlearnable no matter what the learning rule is.
    So we pick outputs as strong sinks, then draw inputs from their true
    upstream ancestors (preferring well-connected ancestors), with an explicit
    audit dict describing what was actually possible.
    """
    n = conn.n_neurons
    in_deg, out_deg = conn.degrees()
    n_out = int(max(2, min(n_outputs, max(2, n // 4))))
    n_in = int(max(2, min(n_inputs, max(2, n // 4))))

    sink_order = np.argsort(-(in_deg.astype(np.float64) - 0.25 * out_deg), kind="stable")
    outputs = np.sort(sink_order[:n_out]).astype(np.int64)

    ancestors = upstream_ancestors(conn, outputs, depth=depth)
    audit: dict[str, object] = {
        "n_outputs": int(outputs.size),
        "n_ancestors_within_depth": int(ancestors.size),
        "depth": int(depth),
        "readout_reachable_from_inputs": bool(ancestors.size > 0),
    }
    if ancestors.size:
        score = out_deg[ancestors].astype(np.float64) + 0.5 * in_deg[ancestors]
        ranked = ancestors[np.argsort(-score, kind="stable")]
        pool = ranked[: max(n_in, min(4 * n_in, ranked.size))]
        chosen = rng.choice(pool, size=min(n_in, pool.size), replace=False) if pool.size > n_in else pool
        inputs = np.sort(chosen).astype(np.int64)
        audit["input_source"] = "upstream_ancestors"
    else:
        inputs = np.sort(np.argsort(-out_deg, kind="stable")[:n_in]).astype(np.int64)
        audit["input_source"] = "out_degree_fallback"
        log.warning(
            "no neuron reaches the readout population within depth=%d; falling back to "
            "high-out-degree inputs, which may drive it only weakly", depth,
        )
    if inputs.size < n_in:  # pad deterministically, never with readout cells
        spare = np.setdiff1d(np.argsort(-out_deg, kind="stable"), np.concatenate([inputs, outputs]))
        inputs = np.sort(np.concatenate([inputs, spare[: n_in - inputs.size]])).astype(np.int64)
    audit["n_inputs"] = int(inputs.size)
    audit["input_output_overlap"] = int(np.intersect1d(inputs, outputs).size)
    return inputs, outputs, audit
