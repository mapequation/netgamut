"""flowgen -- generative benchmark networks from the modular flow of a blueprint.

The non-redundant map equation turns a network + (hierarchical) partition into a
proper joint distribution over node pairs (the *flow model* ``P``).  Running that
model backwards samples networks that reproduce the blueprint's modular flow.  Two
samplers are provided, both **parameter-free** (no temperature):

``sample_multinomial`` (canonical)
    Draw ``E`` links i.i.d. from the symmetrized joint ``P``.  Faithful to the model
    as an ensemble, but at realistic sparsity it produces multi-links and
    disconnected modules -- the direct, leak-free analogue of similarity-softmax
    samplers.

``sample_stub_matching`` (microcanonical, recommended for benchmarks)
    Match stubs from the flow statistics: degrees follow visit rates, each module's
    internal/boundary split follows its observed mixing ``mu = q/p`` (exactly the LFR
    mixing parameter), boundary stubs route up the module tree by hierarchical exit
    rates, and stubs pair by a sequential hub-first pairer subject to the *non-redundant
    constraint* (never pair back into the branch just exited).  Preserves the full degree sequence,
    yields connected simple graphs, handles hierarchy, and reduces to LFR for the
    special case power-law degrees + power-law community sizes + flat ``mu``.

``sample_benchmark`` is the convenience entry point: give it a blueprint (and
optionally a partition; otherwise Infomap supplies one) and it returns a sample.
``mixing_scale`` rescales every module's mixing to sweep detectability from a single
blueprint.

Directed blueprints get ``sample_stub_matching_directed`` (enter != exit rates, both
degree sequences preserved, hierarchy routed).  The maths is documented in the project
notebooks "Non-redundant map equation - expanded form" and "Generative map equation -
properties and regularization".

What the microcanonical sampler fixes, and how (the specification):

* **degrees** -- ``k_u = apportion(2E, p_u)``: the flow-apportioned degree sequence
  (equal to the plain degrees of an unweighted blueprint at the same ``E``);
* **per-module mixing** -- each leaf module's internal/boundary stub totals from
  ``mu_A = q_A / p_A``; the split *between its nodes* follows each node's **observed**
  internal share in the blueprint (``_observed_shares``), capped at ``k_u``.  A
  degree-proportional split hands hubs more internal stubs than their module has other
  nodes (anybeat: internal demand 3309 for a module of 2827 nodes) and was the source of
  the residual "saturation" deficits; measured 2026-09-05 (output/sampler_study/joint);
* **hierarchy** -- boundary stubs stop at the LCA depth read off the blueprint
  (``routing="model"``, the flow model's own escalation probability), see
  :meth:`ModuleTree.escalation_probs`;
* **pairing** -- stubs pair by a *sequential hub-first* pairer (``_pair_stubs``,
  ``_pair_bipartite``): the entity with most remaining stubs draws its partners without
  replacement, proportional to remaining stubs (Bayati, Kim & Saberi 2010; Efraimidis &
  Spirakis 2006 keys), with a Havel--Hakimi take-all fallback and a bounded edge-split
  repair.  It places every pool's maximum (checked against an exact max-flow b-matching)
  where random pairing plus repair sweeps stranded 3-12% of the links and, worse, leaked
  unpaired internal stubs into the boundary pools (football: 387 within-module edges for
  a blueprint with 429).  Order bias against an MCMC-uniformised reference: 0.1 sd.
  Cost ``O((S + n) log n)`` per pool of ``n`` entities and ``S`` stubs (Fenwick trees over
  remaining stubs and over tags, a lazy max-heap for the hub order): one root pool of
  ``3 x 10^5`` entities pairs in about a second where the ``O(n)``-per-hub scan it replaced
  ran for hours.  Verified equal in distribution to that scan on six blueprints x 8 seeds
  (2026-09-18): deficit, per-node degrees and mixing identical, average clustering and
  recovery AMI within 2.4 standard errors (largest: football ``C`` -0.004 +- 0.002).

Together these give, on every unweighted blueprint measured (15 networks, 3 seeds): link
deficit exactly 0, per-node degrees exact, mixing error <= 0.004, no top-up needed.  Weighted
blueprints are sampled in the flow-degree frame (see ``generate.weighted_benchmark``).

``raise_clustering`` (opt-in via ``sample_benchmark(match_clustering=True)``) then steers
each node's local clustering coefficient to the blueprint's with degree-, mixing-,
hierarchy-, block- and connectivity-preserving swaps (numba-compiled when available).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import networkx as nx
import numpy as np

try:
    import infomap
except ImportError:  # infomap is only needed for the Infomap helpers
    infomap = None

try:  # numba accelerates the pairing kernels and raise_clustering; the pure-Python path is the same code
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# --------------------------------------------------------------------------- #
# Flow statistics and the module tree
# --------------------------------------------------------------------------- #
def node_flows(G):
    """Stationary visit rates of an undirected random walk: p_i = s_i / sum_j s_j."""
    s = dict(G.degree(weight="weight"))
    total = sum(s.values())
    return {u: su / total for u, su in s.items()}


def stationary_flows(G, teleport=0.15):
    """Stationary visit rates of a directed random walk with teleportation.

    Infomap's default directed flow model: a PageRank-style walk that teleports with
    probability *teleport* (0.15) and, from dangling nodes, always.  Returns the
    normalised visit rates ``p_i``; for an undirected graph this is not used --
    :func:`node_flows` gives the exact stationary distribution in closed form.
    """
    if G.number_of_nodes() == 0:
        return {}
    p = nx.pagerank(G, alpha=1.0 - teleport, weight="weight")
    total = sum(p.values()) or 1.0
    return {u: v / total for u, v in p.items()}


def as_paths(partition):
    """Normalize a partition to tuple paths (two-level ids become 1-tuples)."""
    return {u: (m if isinstance(m, tuple) else (m,)) for u, m in partition.items()}


def common_prefix_len(a, b):
    L = 0
    for x, y in zip(a, b):
        if x != y:
            break
        L += 1
    return L


class ModuleTree:
    """Module tree with flow statistics from a network + (hierarchical) partition.

    One tree serves both flow directions.  For an **undirected** blueprint the enter and
    exit rates of every module coincide, and the module statistics are the usual
    map-equation ones: ``p[u] = s_u / 2W`` and ``q[nu]`` the boundary flow of module
    ``nu``.  For a **directed** blueprint they differ, and the statistics are read off the
    arc weights crossing each tree node's boundary::

        p_out[u] = s_u^out / W        exit[nu]  = weight of arcs leaving subtree nu
        p_in[u]  = s_u^in  / W        enter[nu] = weight of arcs entering subtree nu

    so ``mu_out(nu) = exit[nu] / flow_out[nu]`` and ``mu_in(nu) = enter[nu] /
    flow_in[nu]`` generalise :func:`directed_mixing_stats` from leaf modules to every
    level of the tree.  Directed statistics are deliberately *count*-based rather than
    stationary-walk based: the microcanonical sampler constrains degree sequences and
    per-module mixing, both count quantities, so no teleportation parameter enters.

    Attributes::

        p, paths                   node -> visit rate (undirected: s_u/2W; directed: mean of
                                   p_out and p_in), node -> module path tuple
        flow, enter, exit          module path -> p_nu, q_nu^enter, q_nu^exit
        q                          alias of ``exit`` (undirected: enter == exit)
        flow_out, flow_in          module path -> out-/in-endpoint flow (directed)
        children, nodes_in, leaves tree structure
    """

    def __init__(self, G, partition, directed=None, flow_stats="stationary", p=None):
        paths = as_paths(partition)
        assert set(paths) == set(G.nodes()), "partition must cover all nodes"
        if directed is None:
            directed = G.is_directed()
        self.directed = bool(directed)
        if flow_stats not in ("stationary", "counts"):
            raise ValueError('flow_stats must be "stationary" or "counts"')
        self.flow_stats = flow_stats
        children, nodes_in = defaultdict(set), defaultdict(list)
        for u, path in paths.items():
            for t in range(1, len(path) + 1):
                children[path[:t - 1]].add(path[:t])
            nodes_in[path].append(u)
        self.paths = paths
        self.children = {m: sorted(ch) for m, ch in children.items()}
        self.nodes_in = {m: sorted(ns) for m, ns in nodes_in.items()}
        self.leaves = sorted(nodes_in)
        if self.directed:
            self._init_directed(G, p=p)
        else:
            self._init_undirected(G)

    # -- undirected: p_i = s_i / 2W, enter == exit == q ---------------------- #
    def _init_undirected(self, G):
        paths = self.paths
        p = node_flows(G)
        flow, q = defaultdict(float), defaultdict(float)
        for u, path in paths.items():
            for t in range(1, len(path) + 1):
                flow[path[:t]] += p[u]
        total_strength = sum(dict(G.degree(weight="weight")).values())  # = 2W
        stop = defaultdict(lambda: defaultdict(float))
        for u, v, w in G.edges(data="weight", default=1.0):
            if u == v:
                continue
            f = w / total_strength  # flow along the edge, per direction
            pu, pv = paths[u], paths[v]
            L = common_prefix_len(pu, pv)
            for t in range(L + 1, len(pu) + 1):
                q[pu[:t]] += f
            for t in range(L + 1, len(pv) + 1):
                q[pv[:t]] += f
            if L < len(pu):
                stop[pu][L] += f
            if L < len(pv):
                stop[pv][L] += f
        self.p = p
        self.flow = dict(flow)
        self.q = {m: q[m] for m in flow}
        self.enter = self.exit = self.q
        self.stop_out = {A: dict(h) for A, h in stop.items()}
        self.stop_in = self.stop_out
        self.flow_out = self.flow_in = self.flow
        self.p_out = self.p_in = p

    # -- directed: one visit rate per node, enter != exit per module --------- #
    def _init_directed(self, G, p=None):
        """Directed statistics in one of two frames (see the class docstring).

        ``flow_stats="stationary"`` (the flow model): a *single* visit rate per node --
        at stationarity a node's in-flow equals its out-flow, so only modules carry an
        enter/exit asymmetry, exactly as the manuscript states.  Link flow is
        ``p_u w_uv / s_u^out`` and module enter/exit rates are that flow summed across
        each subtree boundary.  This is the frame in which ``P`` is proper.

        ``flow_stats="counts"`` (the microcanonical sampler): out- and in-endpoint
        counts and boundary counts read straight off the arc weights.  The sampler
        constrains both degree sequences and both mixings, all count quantities, so no
        teleportation parameter enters -- but the resulting rates are not a stationary
        distribution and must not be fed to :func:`flow_matrix`.
        """
        paths = self.paths
        nodes = list(G.nodes())
        W = float(sum(w for _, _, w in G.edges(data="weight", default=1.0))) or 1.0
        s_out = {u: G.out_degree(u, weight="weight") for u in nodes}

        if self.flow_stats == "counts":
            p_out = {u: s_out[u] / W for u in nodes}
            p_in = {u: G.in_degree(u, weight="weight") / W for u in nodes}
            link_flow = lambda u, v, w: w / W                      # noqa: E731
        else:
            if p is None:
                p = stationary_flows(G)
            p_out = p_in = dict(p)
            link_flow = lambda u, v, w: (p[u] * w / s_out[u]) if s_out[u] > 0 else 0.0  # noqa: E731

        flow_out, flow_in = defaultdict(float), defaultdict(float)
        enter, exit_ = defaultdict(float), defaultdict(float)
        for u, path in paths.items():
            for t in range(1, len(path) + 1):
                flow_out[path[:t]] += p_out[u]
                flow_in[path[:t]] += p_in[u]
        stop_out = defaultdict(lambda: defaultdict(float))
        stop_in = defaultdict(lambda: defaultdict(float))
        for u, v, w in G.edges(data="weight", default=1.0):
            if u == v:
                continue
            f = link_flow(u, v, w)
            pu, pv = paths[u], paths[v]
            L = common_prefix_len(pu, pv)
            for t in range(L + 1, len(pu) + 1):
                exit_[pu[:t]] += f          # the arc leaves every ancestor of u below the LCA
            for t in range(L + 1, len(pv) + 1):
                enter[pv[:t]] += f          # and enters every ancestor of v below the LCA
            if L < len(pu):
                stop_out[pu][L] += f        # ... and re-enters the tree at depth L
            if L < len(pv):
                stop_in[pv][L] += f
        self.p_out, self.p_in = p_out, p_in
        self.p = {u: 0.5 * (p_out[u] + p_in[u]) for u in nodes}
        self.flow_out = dict(flow_out)
        self.flow_in = {m: flow_in[m] for m in flow_out}
        self.flow = ({m: 0.5 * (flow_out[m] + flow_in[m]) for m in flow_out}
                     if self.flow_stats == "counts" else dict(flow_out))
        self.enter = {m: enter[m] for m in flow_out}
        self.exit = {m: exit_[m] for m in flow_out}
        self.q = self.exit
        self.stop_out = {A: dict(h) for A, h in stop_out.items()}
        self.stop_in = {A: dict(h) for A, h in stop_in.items()}

    # -- mixing ------------------------------------------------------------- #
    def mu_out(self, nu):
        """Out-mixing of module *nu*: share of its out-endpoints that leave it."""
        return self.exit[nu] / self.flow_out[nu] if self.flow_out[nu] > 0 else 0.0

    def mu_in(self, nu):
        """In-mixing of module *nu*: share of its in-endpoints that come from outside."""
        return self.enter[nu] / self.flow_in[nu] if self.flow_in[nu] > 0 else 0.0

    # -- leave-one-out normalisers ------------------------------------------ #
    def Z(self, nu, excluded_child, non_redundant=True):
        """Normalization at internal node nu for a walker that exited excluded_child.

        ``Q^enter(nu) - q^enter[excluded_child]``: the enter rates of nu's children plus
        nu's own exit-to-parent rate, with the branch just left removed.
        """
        z = sum(self.enter[ch] for ch in self.children[nu]
                if (ch != excluded_child or not non_redundant))
        if nu != ():
            z += self.exit[nu]
        return z

    def Z_in(self, nu, excluded_child, non_redundant=True):
        """Time-reversed :meth:`Z`: normalizer for an in-stub entering excluded_child."""
        z = sum(self.exit[ch] for ch in self.children[nu]
                if (ch != excluded_child or not non_redundant))
        if nu != ():
            z += self.enter[nu]
        return z

    # -- hierarchical boundary routing -------------------------------------- #
    def escalation_probs(self, A, kind="out", routing="model"):
        """Stop-level distribution for a boundary stub of leaf module *A*.

        Returns ``[(nu, tag, prob), ...]``: the walker that left ``A`` re-enters the tree
        at internal node ``nu`` having exited branch ``tag``, so it pairs inside pool
        ``nu`` and must not pair back into ``tag`` (the non-redundant constraint).
        Deepest level first; the probabilities sum to one.

        ``kind="out"`` routes out-stubs by exit rates, ``kind="in"`` in-stubs by enter
        rates; the two coincide when undirected.  All three rules are identical for a
        two-level partition, where every boundary stub stops at the root, so only a
        hierarchical blueprint can separate them.

        ``"model"`` (default) -- the flow model's own escalation probability
        ``q^exit[nu] / Z(nu, child)``, with the leave-one-out normaliser of
        :func:`cross_R`, so sampler and joint ``P`` describe the same walk, and the only
        rule that needs no blueprint at sampling time.

        ``"observed"`` -- the blueprint's own distribution.  A boundary link re-enters the
        tree at exactly the depth of its endpoints' lowest common ancestor, so this
        histogram is measured rather than modelled, per leaf module.

        ``"boundary"`` -- escalate out of ``nu`` with probability
        ``q[nu] / sum_children q[ch]``, the share of the children's boundary flow that
        also leaves ``nu``.  A pooled rate rather than a per-module one.

        Re-measured 2026-09-18 on the pool's nine hierarchical blueprints, 85 samples per
        rule, with the current pairer and split (the earlier comparison, which made
        ``"observed"`` the default, ran with the random pairer whose within-module failures
        escalated internal stubs into exactly these pools).  Routing through the tree is
        what matters and flattening the partition to its leaves is decisively wrong: total
        variation to the blueprint's distribution of boundary links over levels 0.397
        flattened against 0.019-0.036 for the three rules, and level-1 AMI 0.00 against
        0.79-0.90 on the power grid.  Among the three rules no criterion dominates:
        ``"observed"`` leads on that total variation (0.019 against 0.021 and 0.036) and on
        the pooled gap to the self-recovery ceiling (0.345 against 0.359 and 0.400),
        ``"model"`` loses the planted structure in the fewest samples (4/85 against 8/85 and
        9/85) and has the smallest depth error (0.151 against 0.158 and 0.166), and
        ``"boundary"`` sits between them throughout.  The default follows the robustness
        gate declared before the deepest blueprints were run, which selects ``"model"``.
        """
        depth = len(A)
        if routing == "observed":
            hist = (self.stop_out if kind == "out" else self.stop_in).get(A, {})
            total = sum(hist.values())
            if total > 0:
                return [(A[:t], A[:t + 1], hist.get(t, 0.0) / total)
                        for t in range(depth - 1, -1, -1)]
            routing = "boundary"        # no boundary links to learn from
        if routing not in ("boundary", "model"):
            raise ValueError(f"unknown routing {routing!r}")
        own = self.exit if kind == "out" else self.enter
        out, carry = [], 1.0
        for t in range(depth - 1, -1, -1):
            nu, tag = A[:t], A[:t + 1]
            if t == 0:
                c = 0.0                                    # nothing above the root
            elif routing == "boundary":
                denom = sum(own[ch] for ch in self.children[nu])
                c = own[nu] / denom if denom > 0 else 0.0
            else:
                z = (self.Z(nu, tag) if kind == "out" else self.Z_in(nu, tag))
                c = own[nu] / z if z > 0 else 0.0
            c = min(max(c, 0.0), 1.0)
            out.append((nu, tag, carry * (1.0 - c)))
            carry *= c
            if carry <= 0.0:
                break
        if out and carry > 0.0:                            # numerical remainder to the root
            nu, tag, pr = out[-1]
            out[-1] = (nu, tag, pr + carry)
        return out


def cross_R(tree, A, B, non_redundant=True):
    """R-value (= p_ij / (p_i^out p_j^in)) for i in leaf module A, j in leaf module B != A.

    Exits are priced by exit rates, entries by enter rates; the two coincide for an
    undirected blueprint, so this is the same expression the undirected model always used.
    """
    L = common_prefix_len(A, B)
    val = tree.exit[A] / (tree.flow[A] + tree.exit[A])       # exit leaf A
    for t in range(len(A) - 1, L, -1):            # keep exiting ancestors below the LCA
        nu, c = A[:t], A[:t + 1]
        val *= tree.exit[nu] / tree.Z(nu, c, non_redundant)
    nu, c = A[:L], A[:L + 1]                       # at the LCA: enter the branch holding B
    val *= tree.enter[B[:L + 1]] / tree.Z(nu, c, non_redundant)
    for t in range(L + 1, len(B)):                # descend to leaf B by enter rates
        nu = B[:t]
        val *= tree.enter[B[:t + 1]] / sum(tree.enter[ch] for ch in tree.children[nu])
    return val / tree.flow[B]                      # choose node j within B: p_j / p_B


def flow_matrix(G, partition, non_redundant=True, node_order=None, directed=None):
    """Stochastic flow matrix P (joint distribution over ordered node pairs).

    Row sums equal the nodes' out-flow exactly (properness, P1 of the manuscript);
    column sums equal the in-flow only for symmetric modules.  Directed blueprints are
    priced with separate enter and exit rates; ``directed`` defaults to
    ``G.is_directed()``.

    O(n^2) dense -- practical only for small/medium blueprints.
    """
    tree = ModuleTree(G, partition, directed=directed)
    nodes = list(node_order) if node_order is not None else sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    p = np.array([tree.p[u] for u in nodes])
    R = np.zeros((len(nodes), len(nodes)))
    members = {A: [idx[u] for u in tree.nodes_in[A]] for A in tree.leaves}
    for A in tree.leaves:
        ia = members[A]
        R[np.ix_(ia, ia)] = 1.0 / (tree.flow[A] + tree.exit[A])
        for B in tree.leaves:
            if B != A and tree.exit[A] > 0:
                R[np.ix_(ia, members[B])] = cross_R(tree, A, B, non_redundant)
    return R * np.outer(p, p), nodes, tree


def mixing_stats(G, partition):
    """Observed per-module mixing mu_A = q_A / p_A of a network given a partition."""
    tree = ModuleTree(G, partition, directed=False)
    return {A: (tree.q[A] / tree.flow[A] if tree.flow[A] > 0 else float("nan"))
            for A in tree.leaves}


# --------------------------------------------------------------------------- #
# Flow-model distances (for D ~ 0 blueprint/sample fidelity checks)
# --------------------------------------------------------------------------- #
def jsd(a, b):
    """Jensen-Shannon divergence, base 2 (in [0, 1])."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = 0.5 * (a + b)
    ma, mb = a > 0, b > 0
    return max(0.0, 0.5 * float(np.sum(a[ma] * np.log2(a[ma] / m[ma])))
                  + 0.5 * float(np.sum(b[mb] * np.log2(b[mb] / m[mb]))))


def element_distances(P1, P2):
    """d_i = JS distance between row conditionals q_i = P_i / p_i."""
    p1, p2 = P1.sum(1), P2.sum(1)
    return np.array([math.sqrt(jsd(P1[i] / p1[i], P2[i] / p2[i]))
                     for i in range(P1.shape[0])])


def S_dist(P1, P2):
    """Flow-weighted arithmetic mean of element-wise JS distances (the manuscript's S)."""
    p1, p2 = P1.sum(1), P2.sum(1)
    return float(np.sum(0.5 * (p1 + p2) * element_distances(P1, P2)))


def D_dist(P1, P2):
    """Joint JS distance between flow models (Proposition 6: an unconditional metric)."""
    return math.sqrt(jsd(P1.ravel(), P2.ravel()))


# --------------------------------------------------------------------------- #
# Canonical sampler (draw links from the joint P)
# --------------------------------------------------------------------------- #
def sample_multinomial(P, E, rng=None, no_self_loops=True):
    """Draw E undirected edges i.i.d. from symmetrized P; weights = multiplicities."""
    if rng is None:
        rng = np.random.default_rng()
    Ps = 0.5 * (P + P.T)
    if no_self_loops:
        np.fill_diagonal(Ps, 0.0)
    counts = rng.multinomial(E, Ps.ravel() / Ps.sum()).reshape(Ps.shape)
    counts = counts + counts.T
    G = nx.Graph()
    G.add_nodes_from(range(P.shape[0]))
    for i, j in zip(*np.nonzero(np.triu(counts, k=1))):
        G.add_edge(int(i), int(j), weight=int(counts[i, j]))
    return G


# --------------------------------------------------------------------------- #
# Microcanonical stub-matching sampler (recommended benchmark generator)
# --------------------------------------------------------------------------- #
def apportion(total, weights, rng=None):
    """Allocate `total` integer items proportionally to weights (largest remainder)."""
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=int)
    quota = total * w / w.sum()
    base = np.floor(quota).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        frac = quota - base
        if rng is not None:
            frac = frac + rng.random(len(w)) * 1e-9
        base[np.argsort(-frac)[:rem]] += 1
    return base


# --------------------------------------------------------------------------- #
# Sequential hub-first pairing in O((E + n) log n) per pool
# --------------------------------------------------------------------------- #
# Weighted sampling without replacement over the *eligible* entities of a pool is done on a
# Fenwick tree over remaining stubs, with entities laid out in tag-sorted order and a second
# Fenwick tree over the per-tag totals.  Excluding the hub's own tag (the non-redundant rule)
# is then one temporary update on the tag tree, and each partner draw is two O(log n)
# descents -- pick a tag with probability proportional to its remaining eligible stubs, then
# an entity inside that tag -- so no rejection loop and no O(n_pool) scan.  Successive draws
# proportional to remaining weight give exactly the distribution of the Efraimidis--Spirakis
# top-r keys the previous mask implementation used.
@_njit(cache=True)
def _fw_build(values):
    """Fenwick tree (1-indexed array of length ``n + 1``) over ``values`` (int64)."""
    n = values.shape[0]
    tree = np.zeros(n + 1, np.int64)
    for i in range(n):
        tree[i + 1] = values[i]
    for i in range(1, n + 1):
        j = i + (i & (-i))
        if j <= n:
            tree[j] += tree[i]
    return tree


@_njit(cache=True)
def _fw_add(tree, i, delta):
    """Add ``delta`` at 0-based position ``i``."""
    n = tree.shape[0] - 1
    i += 1
    while i <= n:
        tree[i] += delta
        i += i & (-i)


@_njit(cache=True)
def _fw_prefix(tree, i):
    """Sum of the first ``i`` entries (0-based positions ``0 .. i-1``)."""
    s = 0
    while i > 0:
        s += tree[i]
        i -= i & (-i)
    return s


@_njit(cache=True)
def _fw_find(tree, target):
    """Smallest 0-based position ``j`` with ``prefix(j + 1) > target``, for
    ``0 <= target < total``."""
    n = tree.shape[0] - 1
    pos = 0
    step = 1
    while step * 2 <= n:
        step *= 2
    while step > 0:
        nxt = pos + step
        if nxt <= n and tree[nxt] <= target:
            pos = nxt
            target -= tree[nxt]
        step //= 2
    return pos


@_njit(cache=True)
def _zero_entity(tree, ttree, rem, tag, i):
    """Remove entity ``i`` from both trees (its stubs are placed or stranded)."""
    w = rem[i]
    if w > 0:
        _fw_add(tree, i, -w)
        _fw_add(ttree, tag[i], -w)
        rem[i] = 0


@_njit(cache=True)
def _draw_partners(tree, ttree, rem, tag, tstart, excl, excl_tag, r, urand, picks):
    """Draw up to ``r`` distinct partner entities without replacement, each with probability
    proportional to its remaining stubs, among the entities with stubs left whose tag is not
    ``excl_tag`` (``-1``: no tag excluded) and which are not listed in ``excl`` (distinct
    indices: the hub's own node and the nodes already adjacent to it).

    ``tree`` is the Fenwick tree over ``rem`` in tag-sorted entity order, ``ttree`` the tree
    over the tag totals, ``tstart`` the tag offsets, ``urand`` ``r`` uniforms.  Picked entities
    lose one stub in ``rem`` and both trees.  Returns the number picked; fewer than ``r`` only
    when the eligible weight ran out (the Havel--Hakimi take-all case).
    """
    for k in range(excl.shape[0]):                    # exclusions: temporarily zero
        j = excl[k]
        w = rem[j]
        if w > 0:
            _fw_add(tree, j, -w)
            _fw_add(ttree, tag[j], -w)
    wt = 0
    if excl_tag >= 0:
        wt = _fw_prefix(ttree, excl_tag + 1) - _fw_prefix(ttree, excl_tag)
        if wt > 0:
            _fw_add(ttree, excl_tag, -wt)
    T = ttree.shape[0] - 1
    W = _fw_prefix(ttree, T)
    cnt = 0
    for d in range(r):
        if W <= 0:
            break
        target = np.int64(urand[d] * W)
        if target >= W:
            target = W - 1
        g = _fw_find(ttree, target)                     # tag, proportional to eligible stubs
        resid = target - _fw_prefix(ttree, g)
        j = _fw_find(tree, _fw_prefix(tree, tstart[g]) + resid)   # entity inside the tag
        picks[cnt] = j
        cnt += 1
        w = rem[j]
        _fw_add(tree, j, -w)                            # not drawn twice for this hub
        _fw_add(ttree, g, -w)
        W -= w
    for k in range(cnt):                                # picks return with one stub less
        j = picks[k]
        rem[j] -= 1
        if rem[j] > 0:
            _fw_add(tree, j, rem[j])
            _fw_add(ttree, tag[j], rem[j])
    if wt > 0:
        _fw_add(ttree, excl_tag, wt)
    for k in range(excl.shape[0]):                      # exclusions return in full
        j = excl[k]
        w = rem[j]
        if w > 0:
            _fw_add(tree, j, w)
            _fw_add(ttree, tag[j], w)
    return cnt


def _heap_top(heap, rem):
    """Entity with the most remaining stubs (ties broken by the random key stored in the heap),
    or ``-1`` when none has stubs left.  Entries are ``(-stubs, key, index)``; stale entries
    (stubs decreased since the push) are refreshed lazily, so the heap costs ``O(log n)`` per
    decrement instead of an ``O(n)`` argmax per hub."""
    while heap:
        negr, key, i = heap[0]
        r = int(rem[i])
        if r == 0:
            heapq.heappop(heap)
        elif -negr != r:
            heapq.heapreplace(heap, (-r, key, i))
        else:
            return i
    return -1


def _pool_layout(stubs, rng, tagid):
    """Aggregate stubs ``[(node, tag), ...]`` into entities with a remaining count, shuffled
    (random tie-breaking) and then stably sorted by tag id so every tag is a contiguous block.
    ``tagid`` is extended in place (shared across the two sides of a bipartite pool).

    Returns ``(ents, tag, rem, pos_of)``."""
    cnt = defaultdict(int)
    for st in stubs:
        cnt[st] += 1
    ents = list(cnt)
    rng.shuffle(ents)
    for e in ents:
        tagid.setdefault(e[1], len(tagid))
    ents.sort(key=lambda e: tagid[e[1]])
    n = len(ents)
    tag = np.fromiter((tagid[e[1]] for e in ents), np.int64, n)
    rem = np.fromiter((cnt[e] for e in ents), np.int64, n)
    pos_of = defaultdict(list)
    for j, e in enumerate(ents):
        pos_of[e[0]].append(j)
    return ents, tag, rem, pos_of


def _pool_trees(tag, rem, rng, n_tags):
    """Fenwick trees over the entities and the tags, the tag offsets, and the hub heap."""
    tstart = np.searchsorted(tag, np.arange(n_tags + 1)).astype(np.int64)
    tree = _fw_build(rem)
    ttree = _fw_build(np.bincount(tag, weights=rem, minlength=n_tags).astype(np.int64))
    heap = list(zip((-rem).tolist(), rng.permutation(len(rem)).tolist(), range(len(rem))))
    heapq.heapify(heap)
    return tree, ttree, tstart, heap


_NO_EXCL = np.empty(0, np.int64)


def _excl_array(indices):
    return np.fromiter(indices, np.int64, len(indices)) if indices else _NO_EXCL


def _pair_stubs(stubs, rng, repair_tries=200):
    """Pair stubs ``[(node, tag), ...]`` into links: tags must differ within a pair (a tag
    is the node itself for internal stubs, the branch just exited for boundary stubs -- the
    non-redundant constraint), node pairs must not repeat, no self-pairs.

    Sequential hub-first pairing: aggregate stubs into entities ``(node, tag)`` with a
    remaining count; repeatedly take the entity with the most remaining stubs and draw its
    partners without replacement from the eligible entities (other node, other tag, not yet
    adjacent, stubs left) with probability proportional to their remaining stubs.  When
    fewer entities are eligible than stubs, take them all (Havel--Hakimi) and strand the
    rest; a bounded repair then places stranded pairs by splitting an existing pair
    ``(x, y) -> (u, x), (u', y)``, which keeps every degree.

    Cost ``O((S + n) log n)`` for a pool of ``n`` entities and ``S`` stubs (Fenwick trees and
    a lazy heap, see :func:`_draw_partners`), so one giant root pool -- the parametric mode
    plants a two-level partition and routes every boundary stub to the root -- costs seconds
    where the ``O(n)``-per-hub mask scan took hours (web graph, ``n = 3 x 10^5``: killed
    after 5 h 20 min on 2026-09-06).  Same distribution as that scan (successive weighted
    draws without replacement are the Efraimidis--Spirakis top-``r`` keys); the change of
    random stream moves individual samples, not the ensemble.

    Returns ``(pairs, leftover)``.  Each unordered node pair arises in exactly one pool, so
    per-pool dedup is global dedup.  Verified pool-optimal against an exact max-flow
    b-matching on the pools where any stub strands (2026-09-05).
    """
    stubs = list(stubs)
    if not stubs:
        return [], []
    tagid = {}
    ents, tag, rem, pos_of = _pool_layout(stubs, rng, tagid)
    tree, ttree, tstart, heap = _pool_trees(tag, rem, rng, len(tagid))
    adj = defaultdict(set)
    pairs, leftover = [], []
    while True:
        i = _heap_top(heap, rem)
        if i < 0:
            break
        u = ents[i][0]
        r = int(rem[i])
        ex = {j for v in adj[u] for j in pos_of[v]}
        ex.update(j for j in pos_of[u] if j != i)
        picks = np.empty(r, np.int64)
        got = _draw_partners(tree, ttree, rem, tag, tstart, _excl_array(ex), int(tag[i]),
                             r, rng.random(r), picks)
        for j in picks[:got].tolist():
            v = ents[j][0]
            pairs.append((ents[i], ents[j]))
            adj[u].add(v)
            adj[v].add(u)
        if got < r:                                    # Havel-Hakimi: eligible weight ran out
            leftover.extend([ents[i]] * (r - got))
        _zero_entity(tree, ttree, rem, tag, i)
    if len(leftover) >= 2:
        leftover = _repair_undirected(pairs, leftover, adj, rng, repair_tries)
    return pairs, leftover


def _repair_undirected(pairs, leftover, adj, rng, tries):
    """Place leftover stubs two at a time, directly or by splitting an existing pair of this
    pool ``(x, y) -> (a, x), (b, y)``: degrees of ``x``, ``y`` unchanged, tags stay
    consistent because only pairs of the same pool are split.  Bounded tries."""
    def ok(a, b):
        return a[0] != b[0] and a[1] != b[1] and b[0] not in adj[a[0]]
    rest = []
    rng.shuffle(leftover)
    while len(leftover) >= 2 and pairs:
        a, b = leftover.pop(), leftover.pop()
        if ok(a, b):
            pairs.append((a, b))
            adj[a[0]].add(b[0])
            adj[b[0]].add(a[0])
            continue
        done = False
        for _ in range(tries):
            k = int(rng.integers(len(pairs)))
            x, y = pairs[k]
            for (p, q) in ((x, y), (y, x)):
                if ok(a, p) and ok(b, q) and not (a[0] == b[0] and p[0] == q[0]):
                    adj[x[0]].discard(y[0])
                    adj[y[0]].discard(x[0])
                    pairs[k] = (a, p)
                    pairs.append((b, q))
                    adj[a[0]].add(p[0]); adj[p[0]].add(a[0])
                    adj[b[0]].add(q[0]); adj[q[0]].add(b[0])
                    done = True
                    break
            if done:
                break
        if not done:
            rest.append(a)
            leftover.insert(0, b)                      # retry b with another partner
            if len(rest) > 4 * tries:                  # everything is stuck
                break
    return rest + leftover


class _ArcSet:
    """The arcs realized so far, with successor/predecessor maps so eligibility checks in
    :func:`_pair_bipartite` are ``O(deg)``.  Shared across the pools of one directed sample
    so no directed edge is realized twice."""

    __slots__ = ("succ", "pred", "n")

    def __init__(self):
        self.succ, self.pred, self.n = defaultdict(set), defaultdict(set), 0

    def add(self, o, i):
        self.succ[o].add(i)
        self.pred[i].add(o)
        self.n += 1

    def discard(self, o, i):
        self.succ[o].discard(i)
        self.pred[i].discard(o)
        self.n -= 1

    def __contains__(self, arc):
        return arc[1] in self.succ.get(arc[0], ())

    def __len__(self):
        return self.n


def _pair_bipartite(out_stubs, in_stubs, rng, forbid_same_group=False, used=None, repair_tries=200):
    """Pair out-stubs ``[(node, group), ...]`` to in-stubs into arcs, forbidding self-loops,
    duplicate arcs (via the shared :class:`_ArcSet` ``used``) and, with ``forbid_same_group``,
    same-group pairs (the non-redundant boundary rule).

    Sequential hub-first over both sides: the entity with the most remaining stubs, out or
    in, draws its partners from the other side without replacement, proportional to their
    remaining stubs; Havel--Hakimi take-all fallback; then a bounded repair places leftover
    ``(u, g), (v, h)`` directly or by splitting an arc of this pool ``(x, y) -> (u, y), (x, v)``
    (``x`` keeps its out-degree, ``y`` its in-degree).  Each side has its own Fenwick trees
    and heap, so the cost is ``O((S + n) log n)`` like :func:`_pair_stubs`.

    Returns ``(edges, leftover_out, leftover_in, used)``.
    """
    used = _ArcSet() if used is None else used
    out_stubs, in_stubs = list(out_stubs), list(in_stubs)
    if not out_stubs or not in_stubs:
        return [], out_stubs, in_stubs, used
    gid = {}
    oe, ogrp, orem, opos = _pool_layout(out_stubs, rng, gid)
    ie, igrp, irem, ipos = _pool_layout(in_stubs, rng, gid)
    otree, ottree, ostart, oheap = _pool_trees(ogrp, orem, rng, len(gid))
    itree, ittree, istart, iheap = _pool_trees(igrp, irem, rng, len(gid))

    edges, lo, li = [], [], []
    while True:
        a, b = _heap_top(oheap, orem), _heap_top(iheap, irem)
        if a < 0 or b < 0:
            break
        if orem[a] >= irem[b]:                         # out-hub draws in-partners
            u, r = oe[a][0], int(orem[a])
            ex = set(ipos.get(u, ()))
            for v in used.succ.get(u, ()):
                ex.update(ipos.get(v, ()))
            picks = np.empty(r, np.int64)
            got = _draw_partners(itree, ittree, irem, igrp, istart, _excl_array(ex),
                                 int(ogrp[a]) if forbid_same_group else -1, r, rng.random(r), picks)
            for j in picks[:got].tolist():
                v = ie[j][0]
                edges.append((u, v))
                used.add(u, v)
            if got < r:
                lo.extend([oe[a]] * (r - got))
            _zero_entity(otree, ottree, orem, ogrp, a)
        else:                                          # in-hub draws out-partners
            v, r = ie[b][0], int(irem[b])
            ex = set(opos.get(v, ()))
            for o in used.pred.get(v, ()):
                ex.update(opos.get(o, ()))
            picks = np.empty(r, np.int64)
            got = _draw_partners(otree, ottree, orem, ogrp, ostart, _excl_array(ex),
                                 int(igrp[b]) if forbid_same_group else -1, r, rng.random(r), picks)
            for j in picks[:got].tolist():
                o = oe[j][0]
                edges.append((o, v))
                used.add(o, v)
            if got < r:
                li.extend([ie[b]] * (r - got))
            _zero_entity(itree, ittree, irem, igrp, b)
    for j in np.flatnonzero(orem).tolist():            # the other side ran out first
        lo.extend([oe[j]] * int(orem[j]))
    for j in np.flatnonzero(irem).tolist():
        li.extend([ie[j]] * int(irem[j]))
    if lo and li and edges:
        og = {e[0]: e[1] for e in oe}
        ig = {e[0]: e[1] for e in ie}
        lo, li = _repair_directed(edges, lo, li, used, forbid_same_group, og, ig, rng, repair_tries)
    return edges, lo, li, used


def _repair_directed(edges, lo, li, used, forbid, og, ig, rng, tries):
    """Place a leftover out-stub ``(u, g)`` with a leftover in-stub ``(v, h)`` directly, or by
    splitting an arc of this pool ``(x, y) -> (u, y), (x, v)``.  ``og``/``ig``: node -> group
    on the out/in side of this pool.  Bounded tries."""
    def ok(u, g, v, h):
        return u != v and (not forbid or g != h) and v not in used.succ[u]
    rng.shuffle(lo)
    rng.shuffle(li)
    rest_o = []
    while lo and li and edges:
        (u, g), (v, h) = lo.pop(), li.pop()
        if ok(u, g, v, h):
            edges.append((u, v))
            used.add(u, v)
            continue
        done = False
        for _ in range(tries):
            k = int(rng.integers(len(edges)))
            x, y = edges[k]
            if ok(u, g, y, ig[y]) and ok(x, og[x], v, h):
                used.discard(x, y)
                edges[k] = (u, y)
                edges.append((x, v))
                used.add(u, y)
                used.add(x, v)
                done = True
                break
        if not done:
            rest_o.append((u, g))
            li.insert(0, (v, h))
            if len(rest_o) > 4 * tries:
                break
    return rest_o + lo, li


def _capped_apportion(total, weights, caps, rng):
    """Largest-remainder apportion of ``total`` by ``weights`` that never exceeds ``caps``;
    what a capped node cannot take is redistributed among nodes with spare capacity."""
    w = np.asarray(weights, float).copy()
    caps = np.asarray(caps, int)
    alloc = np.zeros(len(w), int)
    remaining = int(total)
    for _ in range(50):
        if remaining <= 0:
            break
        active = (alloc < caps) & (w > 0)
        if not active.any():
            break
        a = np.minimum(apportion(remaining, w[active], rng), caps[active] - alloc[active])
        alloc[active] += a
        remaining -= int(a.sum())
    if remaining > 0:                                  # weights exhausted: fill where room is left
        active = np.flatnonzero(alloc < caps)
        if len(active):
            a = np.minimum(apportion(remaining, np.ones(len(active)), rng), caps[active] - alloc[active])
            alloc[active] += a
    return alloc


def _observed_shares(G, tree):
    """Per node, the blueprint's strength inside its own leaf module and outside it (out- and
    in-side for directed blueprints; identical for undirected).  These shares decide how a
    module's internal/boundary stub totals are split between its nodes, so a hub keeps its
    observed boundary share instead of a degree-proportional one."""
    paths, directed = tree.paths, G.is_directed()
    out_int, out_bnd, in_int, in_bnd = (defaultdict(float) for _ in range(4))
    for u, v, w in G.edges(data="weight", default=1.0):
        if u == v:
            continue
        if paths[u] == paths[v]:
            out_int[u] += w
            in_int[v] += w
            if not directed:
                out_int[v] += w
        else:
            out_bnd[u] += w
            in_bnd[v] += w
            if not directed:
                out_bnd[v] += w
    if not directed:
        in_int, in_bnd = out_int, out_bnd
    return dict(out_int=out_int, out_bnd=out_bnd, in_int=in_int, in_bnd=in_bnd)


def _split_by_shares(us, ks, total, shares, rng):
    """Distribute ``total`` stubs of one kind over nodes ``us`` with sampled degrees ``ks``,
    proportional to ``k_u * share_u`` and capped at ``k_u``."""
    ks = np.asarray(ks, int)
    w = np.array([k * shares[u] for u, k in zip(us, ks)], float)
    if w.sum() <= 0:
        w = ks.astype(float)
    return _capped_apportion(total, w, ks, rng)


def _route_stubs(tree, per_node, kind, rng, routing="model"):
    """Route each leaf module's boundary stubs to their stop level in the module tree.

    ``per_node`` maps node -> number of boundary stubs of that node.  Returns
    ``{internal node nu: [(node, tag), ...]}``: a stub in pool ``nu`` exited branch
    ``tag`` (a child of ``nu``) and so must not pair back into it.  Two-level partitions
    collapse to the single root pool, reproducing the flat behaviour exactly.
    """
    pools = defaultdict(list)
    for A in tree.leaves:
        B_A = sum(per_node[u] for u in tree.nodes_in[A])
        if B_A == 0:
            continue
        levels = tree.escalation_probs(A, kind=kind, routing=routing)
        counts = apportion(B_A, [pr for _, _, pr in levels], rng)
        stub_nodes = [u for u in tree.nodes_in[A] for _ in range(per_node[u])]
        rng.shuffle(stub_nodes)
        pos = 0
        for (nu, tag, _pr), c_t in zip(levels, counts):
            for u in stub_nodes[pos:pos + c_t]:
                pools[nu].append((u, tag))
            pos += c_t
    return pools


def sample_stub_matching(G, partition, E=None, rng=None, degree_noise=False,
                         boundary_scale=1.0, min_degree=1, routing="model"):
    """Microcanonical (LFR-like) benchmark sampler from a blueprint's flow statistics.

    ``routing`` selects the hierarchical escalation rule shared with the directed
    sampler -- ``"observed"`` (default), ``"boundary"`` or ``"model"``, see
    :meth:`ModuleTree.escalation_probs`.  They differ only for hierarchical partitions.

    Returns (simple undirected nx.Graph, info dict).
    """
    if rng is None:
        rng = np.random.default_rng()
    tree = ModuleTree(G, partition)
    nodes = sorted(G.nodes())
    p = np.array([tree.p[u] for u in nodes])
    if E is None:
        E = G.number_of_edges()

    # 1. degrees ~ visit rates
    k = rng.multinomial(2 * E, p) if degree_noise else apportion(2 * E, p, rng)
    k = np.maximum(k, min_degree)
    k_of = {u: int(k[i]) for i, u in enumerate(nodes)}

    # 2. internal/boundary split: module totals from the mixing mu_A = q_A / p_A, distributed
    #    over the module's nodes by each node's observed internal share (capped at k_u)
    obs = _observed_shares(G, tree)
    internal, boundary = {}, {}
    for A in tree.leaves:
        us = tree.nodes_in[A]
        mu = tree.q[A] / tree.flow[A] if tree.flow[A] > 0 else 0.0
        mu = min(mu * boundary_scale, 1.0)
        ks = np.array([k_of[u] for u in us])
        I_A = min(int(round(ks.sum() * (1.0 - mu) / 2.0)) * 2, int(ks.sum()))
        share = {}
        for u in us:
            tot = obs["out_int"][u] + obs["out_bnd"][u]
            share[u] = obs["out_int"][u] / tot if tot > 0 else 1.0 - mu
        ints = _split_by_shares(us, ks, I_A, share, rng)
        for u, ki, ii in zip(us, ks, ints):
            internal[u], boundary[u] = int(ii), int(ki - ii)

    # 3. route boundary stubs to stop levels (hierarchical exit rates), apportioned
    pools = _route_stubs(tree, boundary, "out", rng, routing)

    edges = []
    info = {"dropped_stubs": 0, "realized_E": 0, "target_E": E, "stub_sum": int(k.sum())}

    # 4a. internal stubs: pair within each leaf module (tag = node -> no self-loops)
    for A in tree.leaves:
        stubs = [(u, u) for u in tree.nodes_in[A] for _ in range(internal[u])]
        pairs, leftover = _pair_stubs(stubs, rng)
        for (u, _tag) in leftover:  # escalate to the parent-level pool
            pools[A[:-1]].append((u, A))
        edges.extend((a[0], b[0]) for a, b in pairs)

    # 4b. boundary pools, deepest first so leftovers escalate upward (dropped at root)
    max_depth = max((len(nu) for nu in pools), default=0)
    for depth in range(max_depth, -1, -1):
        for nu in sorted(kk for kk in pools if len(kk) == depth):
            pairs, leftover = _pair_stubs(pools[nu], rng)
            edges.extend((a[0], b[0]) for a, b in pairs)
            if nu == ():
                info["dropped_stubs"] += len(leftover)
            else:
                for (u, _tag) in leftover:
                    pools[nu[:-1]].append((u, nu[:len(nu[:-1]) + 1]))

    info["realized_E"] = len(edges)
    info["routing"] = routing
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    assert H.number_of_edges() == len(edges), "unexpected cross-pool duplicate"
    return H, info


# --------------------------------------------------------------------------- #
# Directed microcanonical sampler (preserves in- AND out-degree)
# --------------------------------------------------------------------------- #
def directed_mixing_stats(G, partition, hierarchical=False):
    """Per-module directed mixing: mu_out = cross-out / total-out, likewise mu_in.

    Leaf modules by default; ``hierarchical=True`` reports every internal node of the
    module tree as well, which is what the hierarchical boundary routing of
    :func:`sample_stub_matching_directed` needs.
    """
    tree = ModuleTree(G, partition, directed=True, flow_stats="counts")
    keys = sorted(tree.flow) if hierarchical else tree.leaves
    return {A: dict(mu_out=tree.mu_out(A), mu_in=tree.mu_in(A),
                    within=tree.flow_out[A] - tree.exit[A],
                    cout=tree.exit[A], cin=tree.enter[A])
            for A in keys}


def sample_stub_matching_directed(G, partition, E=None, rng=None, boundary_scale=1.0,
                                  min_degree=1, routing="model"):
    """Directed microcanonical benchmark sampler from a blueprint's directed modular flow.

    Preserves the in- and out-degree sequences exactly (every unweighted blueprint measured;
    weighted blueprints are sampled in the flow-degree frame), reproduces each module's
    directed mixing ``mu^exit`` and ``mu^enter`` separately, and enforces the non-redundant
    boundary rule (a module's exit stubs never re-enter that module).  The split of a module's
    boundary stubs between its nodes follows each node's observed boundary share
    (:func:`_observed_shares`); pairing is the sequential pairer of :func:`_pair_bipartite`.

    **Hierarchical**: boundary stubs route up the module tree by the hierarchical exit
    rates (out-stubs) and enter rates (in-stubs), pair inside the pool of the internal
    node they stopped at, and escalate to the parent pool when they cannot be placed --
    the directed counterpart of :func:`sample_stub_matching`.  A two-level partition
    reduces to a single root pool, i.e. to the flat sampler.

    ``routing`` selects the hierarchical escalation rule -- ``"observed"`` (default),
    ``"boundary"`` or ``"model"``, see :meth:`ModuleTree.escalation_probs`.

    Returns (simple nx.DiGraph, info dict).
    """
    if rng is None:
        rng = np.random.default_rng()
    tree = ModuleTree(G, partition, directed=True, flow_stats="counts")
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    Wout = np.array([G.out_degree(u, weight="weight") for u in nodes], float)
    Win = np.array([G.in_degree(u, weight="weight") for u in nodes], float)
    E = G.number_of_edges() if E is None else E

    kout = apportion(E, Wout, rng)
    kin = apportion(E, Win, rng)
    for i in range(len(nodes)):                       # no isolated nodes
        if kout[i] + kin[i] < min_degree:
            if Wout[i] >= Win[i]:
                kout[i] += min_degree
            else:
                kin[i] += min_degree
    kout_of = {u: int(kout[idx[u]]) for u in nodes}
    kin_of = {u: int(kin[idx[u]]) for u in nodes}

    # 1. internal/boundary split per leaf module: totals from its directed mixing, distributed
    #    over its nodes by each node's observed boundary share (out and in separately)
    obs = _observed_shares(G, tree)
    iout, bout, iin, bin_ = {}, {}, {}, {}
    for A in tree.leaves:
        us = tree.nodes_in[A]
        mo = min(tree.mu_out(A) * boundary_scale, 1.0)
        mi = min(tree.mu_in(A) * boundary_scale, 1.0)
        ko = np.array([kout_of[u] for u in us])
        ki = np.array([kin_of[u] for u in us])
        so, si = {}, {}
        for u in us:
            t = obs["out_int"][u] + obs["out_bnd"][u]
            so[u] = obs["out_bnd"][u] / t if t > 0 else mo
            t = obs["in_int"][u] + obs["in_bnd"][u]
            si[u] = obs["in_bnd"][u] / t if t > 0 else mi
        bo = _split_by_shares(us, ko, min(int(round(ko.sum() * mo)), int(ko.sum())), so, rng)
        bi = _split_by_shares(us, ki, min(int(round(ki.sum() * mi)), int(ki.sum())), si, rng)
        for u, k, b in zip(us, ko, bo):
            bout[u], iout[u] = int(b), int(k - b)
        for u, k, b in zip(us, ki, bi):
            bin_[u], iin[u] = int(b), int(k - b)

    # 2. route boundary stubs up the module tree
    pools_out = _route_stubs(tree, bout, "out", rng, routing)
    pools_in = _route_stubs(tree, bin_, "in", rng, routing)

    # 3a. internal arcs: pair within each leaf module; leftovers escalate to its parent
    used, edges = _ArcSet(), []
    for A in tree.leaves:
        us = tree.nodes_in[A]
        o = [(u, A) for u in us for _ in range(iout[u])]
        i = [(u, A) for u in us for _ in range(iin[u])]
        e, lo, li, used = _pair_bipartite(o, i, rng, forbid_same_group=False, used=used)
        edges.extend(e)
        pools_out[A[:-1]].extend((u, A) for u, _ in lo)
        pools_in[A[:-1]].extend((u, A) for u, _ in li)

    # 3b. boundary pools, deepest first, so leftovers escalate upward (dropped at the root)
    depth = max([len(nu) for nu in pools_out] + [len(nu) for nu in pools_in], default=0)
    root_lo, root_li = [], []
    for d in range(depth, -1, -1):
        for nu in sorted(set(k for k in pools_out if len(k) == d)
                         | set(k for k in pools_in if len(k) == d)):
            e, lo, li, used = _pair_bipartite(pools_out.get(nu, []), pools_in.get(nu, []),
                                              rng, forbid_same_group=True, used=used)
            edges.extend(e)
            if nu == ():
                root_lo, root_li = lo, li
            else:
                pools_out[nu[:-1]].extend((u, nu) for u, _ in lo)
                pools_in[nu[:-1]].extend((u, nu) for u, _ in li)

    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    return H, {"target_E": E, "realized_E": H.number_of_edges(),
               "dropped_out": len(root_lo), "dropped_in": len(root_li),
               "levels": max(len(A) for A in tree.leaves), "routing": routing}


def connect_components(H, rng, partition=None, max_passes=12, tries=200):
    """Merge components with degree-preserving double-edge swaps (directed or undirected).

    Each non-giant component is joined to the giant by one swap ``(a->b),(c->d) -> (a->d),(c->b)``.
    With ``partition`` given (as the samplers do), the swap is chosen **module-aware**: ``c`` is
    drawn from the giant's edges with ``comm[c] == comm[a]`` and ``comm[d] == comm[b]``, so the two
    new links have exactly the module pairs of the two removed ones and every module's internal and
    boundary link counts -- hence every ``mu_A`` and every module-pair count -- are unchanged.  Only
    when no such giant edge exists after ``tries`` draws does it fall back to an unconstrained swap.
    Isolated nodes are attached by splitting a giant edge, preferably one inside their own module
    (a small, controlled degree perturbation).  Passes repeat until one component remains or no
    progress is made.  Modifies ``H`` in place.

    Why module-aware: on sparse blueprints the repair is not rare -- the 209-link interactome sample
    needs ~20 merges under a two-level partition -- and module-blind swaps turned internal links
    into boundary links, moving the mean per-module mixing from 0.18 to 0.28 (2026-09-05).

    A swap joins the two components only if at least one of the removed edges is not a bridge:
    removing ``(a, b)`` splits a tree-like small component into an ``a``-side and a ``b``-side,
    and if ``(c, d)`` is a bridge of the giant as well, ``a``-side + ``d``-side becomes a *new*
    component.  So when ``(a, b)`` is a bridge of its component, the giant edge is drawn from the
    non-bridge edges (bridges found once per pass, Tarjan); attaching an isolated node never
    splits anything.  Without this rule each pass manufactured fresh two-node fragments on sparse
    samples and the loop never converged (60k-node test graph: 4 components left after 40 passes,
    1 with the rule; 2026-09-18).

    Cost ``O(passes * (n + E))``: the giant's edge list, its per-module-pair index and its bridge
    set are built once per pass and the lists grown incrementally as components join (a
    swapped-away edge stays listed and is skipped when drawn, so draws are still uniform over the
    live edges).  Rebuilding the lists after every merge made the repair quadratic in the number
    of components -- 37 s for a 20k-node graph with 1.3k components and, on the 1.1M-node road
    network, hours per sample (2026-09-18).
    """
    directed = H.is_directed()
    comp_fn = nx.weakly_connected_components if directed else nx.connected_components
    comm = as_paths(partition) if partition is not None else None
    info = {"merges": 0, "merges_module_aware": 0, "singletons_attached": 0}

    def ok_swap(a, b, c, d):
        return (a != d and c != b and H.has_edge(c, d)                 # (c, d) may have been swapped away
                and not H.has_edge(a, d) and not H.has_edge(c, b))

    def index(edges):
        pairs = defaultdict(list)
        if comm is not None:
            for c, d in edges:
                pairs[(comm[c], comm[d])].append((c, d))
                if not directed:
                    pairs[(comm[d], comm[c])].append((d, c))
        return pairs

    def absorb(new_edges):                                               # the giant grew: extend, never rebuild
        giant_edges.extend(new_edges)
        if comm is not None:
            for c, d in new_edges:
                by_pair[(comm[c], comm[d])].append((c, d))
                if not directed:
                    by_pair[(comm[d], comm[c])].append((d, c))

    for _ in range(max_passes):
        comps = sorted(comp_fn(H), key=len, reverse=True)
        if len(comps) <= 1:
            break
        giant = set(comps[0])
        giant_edges = [(u, v) for u, v in H.edges() if u in giant]
        by_pair = index(giant_edges)
        Hu = H.to_undirected(as_view=True) if directed else H
        bridges = set()
        for u, v in nx.bridges(Hu.subgraph(giant)):
            bridges.add((u, v)); bridges.add((v, u))
        progressed = False
        for small in comps[1:]:
            small = list(small)
            small_edges = [(u, v) for u, v in H.edges(small) if u in small] if directed else \
                          [(u, v) for u, v in H.edges(small)]
            small_bridges = set()
            if len(small) > 1:
                for u, v in nx.bridges(Hu.subgraph(small)):
                    small_bridges.add((u, v)); small_bridges.add((v, u))
            done = False
            if small_edges and giant_edges:
                # module-aware attempts first (``tries`` draws), then unconstrained ones
                for attempt in range(2 * tries):
                    a, b = small_edges[int(rng.integers(len(small_edges)))]
                    if not directed and rng.random() < 0.5:
                        a, b = b, a
                    ab_bridge = (a, b) in small_bridges
                    cands = by_pair.get((comm[a], comm[b])) if comm is not None else None
                    if cands and attempt < tries:
                        k = int(rng.integers(len(cands)))
                        c, d = cands[k]
                        if not H.has_edge(c, d):                # swapped away earlier: purge, redraw
                            cands[k] = cands[-1]
                            cands.pop()
                            continue
                        aware = True
                    else:
                        c, d = giant_edges[int(rng.integers(len(giant_edges)))]
                        if not directed and rng.random() < 0.5:
                            c, d = d, c
                        aware = False
                    if ab_bridge and (c, d) in bridges:       # would split off a new component
                        continue
                    if ok_swap(a, b, c, d):
                        H.remove_edge(a, b); H.remove_edge(c, d)
                        H.add_edge(a, d); H.add_edge(c, b)
                        info["merges"] += 1
                        info["merges_module_aware"] += aware
                        joined = [(a, d), (c, b)] + [e for e in small_edges if e != (a, b) and e != (b, a)]
                        if ab_bridge:                              # the two new links are bridges too
                            bridges.update({(a, d), (d, a), (c, b), (b, c)})
                        bridges.update(small_bridges)
                        done = True
                        break
            if not done and len(small) == 1 and giant_edges:      # isolated node: split a giant edge
                iso = small[0]
                cands = by_pair.get((comm[iso], comm[iso])) if comm is not None else None
                for attempt in range(2 * tries):
                    if cands and attempt < tries:               # inside its own module first
                        k = int(rng.integers(len(cands)))
                        c, d = cands[k]
                        if not H.has_edge(c, d):
                            cands[k] = cands[-1]
                            cands.pop()
                            continue
                    else:
                        c, d = giant_edges[int(rng.integers(len(giant_edges)))]
                    if H.has_edge(c, d) and iso not in (c, d) and not H.has_edge(c, iso) and not H.has_edge(iso, d):
                        H.remove_edge(c, d); H.add_edge(c, iso); H.add_edge(iso, d)
                        info["singletons_attached"] += 1
                        joined = [(c, iso), (iso, d)]
                        done = True
                        break
            if done:
                giant.update(small)
                absorb(joined)
                progressed = True
        if not progressed:
            break
    info["components"] = (nx.number_weakly_connected_components(H) if directed
                          else nx.number_connected_components(H))
    return H, info


@_njit(cache=True)
def _cl_has(indptr, nbrs, u, v):
    lo, hi = indptr[u], indptr[u + 1]
    while lo < hi:
        mid = (lo + hi) >> 1
        if nbrs[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo < indptr[u + 1] and nbrs[lo] == v


@_njit(cache=True)
def _cl_replace(indptr, nbrs, u, old, new):
    """In u's sorted neighbour row replace ``old`` by ``new`` and keep the row sorted."""
    st, en = indptr[u], indptr[u + 1]
    lo, hi = st, en
    while lo < hi:
        mid = (lo + hi) >> 1
        if nbrs[mid] < old:
            lo = mid + 1
        else:
            hi = mid
    j = lo
    if new > old:
        while j + 1 < en and nbrs[j + 1] < new:
            nbrs[j] = nbrs[j + 1]
            j += 1
    else:
        while j - 1 >= st and nbrs[j - 1] > new:
            nbrs[j] = nbrs[j - 1]
            j -= 1
    nbrs[j] = new


@_njit(cache=True)
def _cl_common(indptr, nbrs, u, v, out):
    """Common neighbours of u and v (sorted-row merge) into ``out``; returns their number."""
    i, ie = indptr[u], indptr[u + 1]
    j, je = indptr[v], indptr[v + 1]
    k = 0
    while i < ie and j < je:
        a, b = nbrs[i], nbrs[j]
        if a == b:
            out[k] = a
            k += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return k


@_njit(cache=True)
def _cl_triangles(indptr, nbrs, t, buf):
    n = len(t)
    for u in range(n):
        c = 0
        for i in range(indptr[u], indptr[u + 1]):
            c += _cl_common(indptr, nbrs, u, nbrs[i], buf)
        t[u] = c // 2


@_njit(cache=True)
def _cl_reach(indptr, nbrs, src, dst, hop_limit, stamp, stampval, queue):
    """Breadth-first search from ``src``; True if ``dst`` is within ``hop_limit`` hops."""
    if src == dst:
        return True
    head, tail = 0, 0
    queue[tail] = src
    tail += 1
    stamp[src] = stampval
    level_end, hops = tail, 0
    while head < tail and hops < hop_limit:
        u = queue[head]
        head += 1
        for i in range(indptr[u], indptr[u + 1]):
            v = nbrs[i]
            if v == dst:
                return True
            if stamp[v] != stampval:
                stamp[v] = stampval
                queue[tail] = v
                tail += 1
        if head == level_end:
            hops += 1
            level_end = tail
    return False


@_njit(cache=True)
def _cl_refresh(indptr, t, tgt, cum):
    """Apex-sampling weights: a node's triangle deficit plus a floor, zero below degree 2."""
    n = len(t)
    base, cnt = 0.0, 0
    for u in range(n):
        if indptr[u + 1] - indptr[u] >= 2:
            d = tgt[u] - t[u]
            if d > 0:
                base += d
                cnt += 1
    base = 0.1 * base / cnt if cnt > 0 else 1.0
    total = 0.0
    for u in range(n):
        if indptr[u + 1] - indptr[u] >= 2:
            d = tgt[u] - t[u]
            total += (d if d > 0 else 0.0) + base
        cum[u] = total
    return total


@_njit(cache=True)
def _cl_pick(indptr, nbrs, u, want_comm, comm, excl1, excl2, k):
    """The k-th neighbour of u (0-based) whose module is ``want_comm`` and that is not excl1/excl2;
    -1 when there is none.  With k < 0 returns the count of such neighbours."""
    c = 0
    for i in range(indptr[u], indptr[u + 1]):
        v = nbrs[i]
        if comm[v] == want_comm and v != excl1 and v != excl2:
            if c == k:
                return v
            c += 1
    return -1 if k >= 0 else c


@_njit(cache=True)
def _cl_kernel(indptr, nbrs, comm, t, tgt, w, cum, dt, mark, touched, buf, stamp, queue,
               istate, fstate, chunk, max_attempts, stall_window, hop_limit, refresh_every, seed):
    """Run up to ``chunk`` swap attempts.  istate = [swaps, attempts, last_accept, stop, stampval,
    mark_epoch, seeded]; fstate = [D, C, total_weight].  stop: 0 running, 1 budget, 2 stall."""
    n = len(t)
    if istate[6] == 0:
        np.random.seed(seed)
        istate[6] = 1
        fstate[2] = _cl_refresh(indptr, t, tgt, cum)
    swaps, attempts, last_accept = istate[0], istate[1], istate[2]
    done = 0
    while done < chunk:
        if attempts >= max_attempts:
            istate[3] = 1
            break
        if attempts - last_accept > stall_window:
            istate[3] = 2
            break
        attempts += 1
        done += 1
        if attempts % refresh_every == 0:
            fstate[2] = _cl_refresh(indptr, t, tgt, cum)
        if fstate[2] <= 0.0:
            istate[3] = 2
            break
        a = np.searchsorted(cum, np.random.random() * fstate[2])
        if a >= n:
            a = n - 1
        sa, ea = indptr[a], indptr[a + 1]
        if ea - sa < 2:
            continue
        i = sa + np.random.randint(0, ea - sa)
        j = sa + np.random.randint(0, ea - sa)
        if i == j:
            continue
        b, c = nbrs[i], nbrs[j]
        if _cl_has(indptr, nbrs, b, c):
            continue                                    # wedge already closed
        A, B = comm[b], comm[c]
        if A == B:                                      # types 1/2: rewire two internal edges of A
            nx_ = _cl_pick(indptr, nbrs, b, A, comm, a, c, -1)
            ny_ = _cl_pick(indptr, nbrs, c, A, comm, a, b, -1)
        else:                                           # type 3: x in B (c's module), y in A (b's)
            nx_ = _cl_pick(indptr, nbrs, b, B, comm, a, c, -1)
            ny_ = _cl_pick(indptr, nbrs, c, A, comm, a, b, -1)
        if nx_ <= 0 or ny_ <= 0:
            continue
        if A == B:
            x = _cl_pick(indptr, nbrs, b, A, comm, a, c, np.random.randint(0, nx_))
            y = _cl_pick(indptr, nbrs, c, A, comm, a, b, np.random.randint(0, ny_))
        else:
            x = _cl_pick(indptr, nbrs, b, B, comm, a, c, np.random.randint(0, nx_))
            y = _cl_pick(indptr, nbrs, c, A, comm, a, b, np.random.randint(0, ny_))
        if x == y or _cl_has(indptr, nbrs, x, y):
            continue
        # triangle deltas: removals on the current graph, then rewire, then additions
        istate[5] += 1
        epoch = istate[5]
        nt = 0
        for (p, q, sgn) in ((b, x, -1), (c, y, -1)):
            k = _cl_common(indptr, nbrs, p, q, buf)
            for z in (p, q):
                if mark[z] != epoch:
                    mark[z] = epoch
                    dt[z] = 0
                    touched[nt] = z
                    nt += 1
                dt[z] += sgn * k
            for m in range(k):
                z = buf[m]
                if mark[z] != epoch:
                    mark[z] = epoch
                    dt[z] = 0
                    touched[nt] = z
                    nt += 1
                dt[z] += sgn
        _cl_replace(indptr, nbrs, b, x, c)
        _cl_replace(indptr, nbrs, x, b, y)
        _cl_replace(indptr, nbrs, c, y, b)
        _cl_replace(indptr, nbrs, y, c, x)
        for (p, q) in ((b, c), (x, y)):
            k = _cl_common(indptr, nbrs, p, q, buf)
            for z in (p, q):
                if mark[z] != epoch:
                    mark[z] = epoch
                    dt[z] = 0
                    touched[nt] = z
                    nt += 1
                dt[z] += k
            for m in range(k):
                z = buf[m]
                if mark[z] != epoch:
                    mark[z] = epoch
                    dt[z] = 0
                    touched[nt] = z
                    nt += 1
                dt[z] += 1
        dD = 0.0
        dC = 0.0
        for m in range(nt):
            z = touched[m]
            dD += w[z] * (abs(t[z] + dt[z] - tgt[z]) - abs(t[z] - tgt[z]))
            dC += w[z] * dt[z]
        ok = dD < -1e-12
        if ok:                                          # connectivity: both removed edges bypassed
            istate[4] += 1
            ok = _cl_reach(indptr, nbrs, b, x, hop_limit, stamp, istate[4], queue)
            if ok:
                istate[4] += 1
                ok = _cl_reach(indptr, nbrs, c, y, hop_limit, stamp, istate[4], queue)
        if not ok:
            _cl_replace(indptr, nbrs, b, c, x)
            _cl_replace(indptr, nbrs, x, y, b)
            _cl_replace(indptr, nbrs, c, b, y)
            _cl_replace(indptr, nbrs, y, x, c)
            continue
        for m in range(nt):
            z = touched[m]
            t[z] += dt[z]
        fstate[0] += dD
        fstate[1] += dC / n
        swaps += 1
        last_accept = attempts
    istate[0], istate[1], istate[2] = swaps, attempts, last_accept
    return done


def raise_clustering(H, partition, rng, blueprint=None, targets=None, swap_factor=50,
                     stall_window=None, time_cap=None, hop_limit=4, chunk=20000):
    """Steer each node's local clustering coefficient toward the blueprint's, preserving
    everything the sampler fixed.

    Objective: ``D = sum_u w_u |t_u - t_u^*|`` with ``t_u`` the node's triangle count,
    ``t_u^*`` its target and ``w_u = 1 / C(k_u, 2)``, so ``D`` is the L1 distance between local
    clustering coefficients and the average clustering ``C`` is matched node by node.  A swap
    is accepted iff it strictly lowers ``D``.  Targets come from the ``blueprint`` (its own
    triangle counts; nodes are aligned because the sampler keeps node identities and degrees)
    or, for parametric samples with no blueprint, from ``targets``: a ``{node: triangles}``
    dict or a callable ``c(k)`` giving the local clustering of a degree-``k`` node, e.g. the
    fitted profile ``c(k) = c_0 k^{-alpha}``; ``t_u^* = c(k_u) C(k_u, 2)``.  LFR is the case
    ``c = 0`` (no pass).  Newman-style edge-disjoint triangle stubs were measured as an
    alternative and rejected: they can plant at most ``k/2`` triangles per node, which covers
    11-12% of the deficit on email and HepPh (ideas.tex, 2026-09-05).

    Moves are double-edge swaps ``remove (b,x),(c,y) -> add (b,c),(x,y)`` closing an open
    wedge ``b--a--c`` (apex ``a`` drawn with probability proportional to its triangle deficit):
    when ``b, c`` share leaf module ``A`` the two removed edges are internal edges of ``A``
    (the apex may sit in another module, which creates the two-boundary-edge triangles a
    within-module pass cannot); when ``b`` in ``B`` and ``c`` in ``C`` differ, ``x`` is a
    neighbour of ``b`` in ``C`` and ``y`` a neighbour of ``c`` in ``B``, so two ``B--C``
    boundary edges become two ``B--C`` boundary edges.  Invariants, all exact: every degree,
    every module's internal and boundary edge counts (hence every ``mu_A``), every LCA-depth
    histogram (hence the hierarchical routing), every realized module-pair link count, simple
    graph, and connectivity -- a swap is kept only if both removed edges are bypassed within
    ``hop_limit`` hops in the new graph, so no bridge is ever removed and no repair pass is
    needed afterwards.

    Stops at ``swap_factor * E`` attempts, after ``stall_window`` attempts without an accepted
    swap (default ``max(2000, E)``), or at ``time_cap`` seconds.  Compiled with numba when
    available (same code otherwise).  Undirected only.  Returns ``(nx.Graph, info)``.

    Measured 2026-09-05 on samples of the sequential sampler (output/sampler_study/
    clustering_v2): mean ``|c_u - c_u^bp|`` polbooks 0.180 -> 0.026, email 0.149 -> 0.017,
    polblogs 0.147 -> 0.007; recovery AMI unchanged or higher and the number of modules
    Infomap finds stays at the planted count, where the earlier within-module wedge pass
    (transitivity target, no per-node profile, no block constraint) inflated it (email 31.5
    modules for 12 planted).  A global-``C`` target with greedy triangle creation and the
    unconstrained type-3 move were both rejected on those numbers (ideas.tex, 2026-09-05).
    """
    import time
    if H.is_directed():
        raise ValueError("raise_clustering is undirected only")
    t0 = time.time()
    nodes = sorted(H.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    deg = np.array([H.degree(u) for u in nodes], dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg)
    nbrs = np.empty(int(indptr[-1]), dtype=np.int64)
    for u in nodes:
        i = idx[u]
        row = np.fromiter((idx[v] for v in H.neighbors(u)), dtype=np.int64, count=deg[i])
        row.sort()
        nbrs[indptr[i]:indptr[i + 1]] = row
    paths = as_paths(partition)
    mid = {}
    comm = np.array([mid.setdefault(paths[u], len(mid)) for u in nodes], dtype=np.int64)
    if targets is None:
        if blueprint is None:
            raise ValueError("raise_clustering needs the blueprint the sample was drawn from, or "
                             "per-node targets (a {node: triangles} dict or a callable c(k))")
        U = nx.Graph(blueprint) if blueprint.is_directed() else blueprint
        if not all(u in U for u in nodes):
            raise ValueError("blueprint and sample must share their node set (per-node targets)")
        tb = nx.triangles(U)
        tgt = np.array([float(tb[u]) for u in nodes])
    elif callable(targets):                              # parametric: local clustering c(k)
        tgt = np.array([float(targets(int(k))) * k * (k - 1) / 2.0 for k in deg])
    else:
        tgt = np.array([float(targets[u]) for u in nodes])
    w = np.where(deg >= 2, 2.0 / np.maximum(deg * (deg - 1), 1), 0.0)
    t = np.zeros(n, dtype=np.int64)
    kmax = int(deg.max()) if n else 0
    buf = np.empty(max(kmax, 1), dtype=np.int64)
    _cl_triangles(indptr, nbrs, t, buf)
    m = int(indptr[-1]) // 2
    if m == 0 or kmax < 2:
        return H.copy(), {"swaps": 0, "attempts": 0, "reason": "nothing to do"}
    D0 = float(np.sum(w * np.abs(t - tgt)))
    C0 = float(np.sum(w * t)) / n
    cum = np.zeros(n)
    dt = np.zeros(n, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    touched = np.empty(4 * kmax + 4, dtype=np.int64)
    stamp = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    istate = np.zeros(7, dtype=np.int64)
    fstate = np.array([D0, C0, 0.0])
    max_attempts = int(swap_factor * m)
    if stall_window is None:
        stall_window = max(2000, m)
    seed = int(rng.integers(0, 2**31 - 1))
    reason = "budget"
    while True:
        _cl_kernel(indptr, nbrs, comm, t, tgt, w, cum, dt, mark, touched, buf, stamp, queue,
                   istate, fstate, int(chunk), max_attempts, int(stall_window), int(hop_limit),
                   5000, seed)
        if istate[3] == 1:
            reason = "budget"
            break
        if istate[3] == 2:
            reason = "stall"
            break
        if time_cap is not None and time.time() - t0 > time_cap:
            reason = "time_cap"
            break
    Hc = nx.Graph()
    Hc.add_nodes_from(nodes)
    Hc.add_edges_from((nodes[u], nodes[nbrs[i]]) for u in range(n)
                      for i in range(indptr[u], indptr[u + 1]) if u < nbrs[i])
    return Hc, {"swaps": int(istate[0]), "attempts": int(istate[1]), "reason": reason,
                "D_before": D0, "D_after": float(fstate[0]), "C_before": C0,
                "C_after": float(fstate[1]), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# Infomap helpers
# --------------------------------------------------------------------------- #
def run_infomap(G, two_level=False, num_trials=20, seed=1, directed=None, extra=None,
                stats=False, **options):
    """Run Infomap on a networkx graph; returns ``({node: path tuple}, codelength)``.

    With ``stats=True`` the second element is a dict of Infomap's own codelength
    statistics instead -- ``codelength``, ``one_level_codelength``, ``savings`` (the
    relative codelength savings), ``num_levels`` and ``num_leaf_modules`` -- which is
    what a compressibility fingerprint needs.

    Uses Infomap's keyword API, so engine settings are keyword arguments
    (``markov_time=1.5``, ``flow_model="rawdir"``, ...) rather than a CLI string, and
    node labels of any type survive the round trip -- integer labels come back
    unchanged, other labels are mapped back through ``Result.names``.

    ``directed`` defaults to ``G.is_directed()``; when true Infomap runs with directed
    flow, so a module's enter and exit rates differ.  ``extra`` still accepts a raw
    argument string for CLI parity when an option has no keyword.

    Paths are the solution's own, so they are *ragged*: a leaf sitting directly under a
    top-level module has a shorter path than one nested deeper.
    """
    if infomap is None:
        raise ImportError("infomap package not available")
    if directed is None:
        directed = G.is_directed()
    result = infomap.run(G, num_trials=num_trials, seed=seed, two_level=two_level,
                         directed=directed, silent=True,
                         **({"args": extra} if extra else {}), **options)
    # Leaf paths from the tree itself, dropping each node's index within its module.
    # NOT Result.multilevel_modules(), which pads every leaf to uniform depth: that
    # inserts singleton levels into a ragged hierarchy, inflating depth statistics and
    # giving ModuleTree tree levels the solution does not have.
    names = result.names
    partition = {(names[node.node_id] if names else node.node_id): tuple(node.path[:-1])
                 for node in result.nodes()}
    if not stats:
        return partition, result.codelength
    return partition, {
        "codelength": float(result.codelength),
        "one_level_codelength": float(result.one_level_codelength),
        "savings": float(result.relative_codelength_savings),
        "num_levels": int(result.num_levels),
        "num_leaf_modules": int(result.num_leaf_modules),
    }


def infomap_partition(G, two_level=True, num_trials=20, seed=1, directed=None, extra=None,
                      **options):
    """Convenience: just the ``{node: path tuple}`` partition from Infomap."""
    return run_infomap(G, two_level=two_level, num_trials=num_trials, seed=seed,
                       directed=directed, extra=extra, **options)[0]


def read_pajek(path):
    """Minimal Pajek reader (integer node ids, undirected, weights aggregated)."""
    G, mode = nx.Graph(), None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("*vertices"):
                mode = "v"
                continue
            if low.startswith(("*edges", "*arcs", "*links")):
                mode = "e"
                continue
            parts = line.split()
            if mode == "v":
                G.add_node(int(parts[0]))
            elif mode == "e":
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2]) if len(parts) > 2 else 1.0
                G.add_edge(u, v, weight=G[u][v]["weight"] + w if G.has_edge(u, v) else w)
    return G


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
def sample_benchmark(G, partition=None, target_edges=None, mixing_scale=1.0,
                     method="stub", directed=None, connect=True, min_degree=1,
                     match_clustering=False, rng=None, degree_noise=False, num_trials=20,
                     seed=1, routing="model"):
    """Generate one benchmark network from a blueprint's modular flow.

    Parameters
    ----------
    G : networkx.Graph or DiGraph
        Blueprint network (weights honored).
    partition : dict or None
        {node: module-id or path-tuple}.  Hierarchical paths welcome, for directed and
        undirected blueprints alike.  If None, Infomap supplies a hierarchical one.
    target_edges : int or None
        Number of edges to realize (default: same as the blueprint).
    mixing_scale : float
        Multiply every module's mixing mu -> min(mixing_scale * mu, 1).
        1.0 is faithful; > 1 blurs communities (detectability sweeps).
    method : {"stub", "multinomial"}
        "stub" = microcanonical stub matching (degree-preserving, connected).  For a directed
        blueprint this preserves in- AND out-degree.  "multinomial" = canonical draw from the dense
        joint P (undirected, small graphs only).
    routing : {"observed", "boundary", "model"}
        Hierarchical boundary-stub escalation rule; see
        :meth:`ModuleTree.escalation_probs`.  Only matters for hierarchical partitions.
    directed : bool or None
        Force directed/undirected; defaults to G.is_directed().
    connect : bool
        Run a degree-preserving connectivity-repair pass so the sample is a single component.
    min_degree : int
        Floor on sampled node degree (avoids isolated nodes).
    match_clustering : bool
        Opt-in (default False, undirected only): run :func:`raise_clustering`, which steers every
        node's local clustering coefficient to the blueprint's with swaps that keep degrees,
        mixing, hierarchy, block structure and connectivity.  Off by default because it costs
        more than the sampling itself and because clustering is a blueprint statistic the pass
        matches, not a property of the flow model.

    Returns
    -------
    (H, info) : the sampled graph and a dict of diagnostics (incl. connectivity repair).

    .. warning::
       Called without ``partition`` this plants with the RECURSIVE search
       (``infomap_partition`` passes no ``extra``).  The generative-map-equation project
       runs the columnar engine (``-C``) on both sides of every plant/re-detect comparison,
       so a partition-less call there would silently make its recovery numbers measure
       cross-engine agreement instead.  Every call site in that project passes a partition
       today (checked 2026-09-19), and the default is left alone because this package is
       shared with other projects that may rely on it; pass ``partition=`` explicitly, or
       ``infomap_partition(..., extra="-C")``, when the engine matters.
    """
    if rng is None:
        rng = np.random.default_rng()
    if directed is None:
        directed = G.is_directed()
    if partition is None:
        partition = infomap_partition(G, two_level=False, num_trials=num_trials,
                                      seed=seed, directed=directed)
    if method == "multinomial":
        if directed:
            raise ValueError("multinomial sampler is undirected only; use method='stub'")
        P, _, _ = flow_matrix(G, partition, non_redundant=True)
        E = G.number_of_edges() if target_edges is None else target_edges
        H = sample_multinomial(P, E, rng)
        info = {"realized_E": H.number_of_edges(), "target_E": E, "method": "multinomial"}
    elif method == "stub":
        if directed:
            H, info = sample_stub_matching_directed(G, partition, E=target_edges, rng=rng,
                                                    boundary_scale=mixing_scale,
                                                    min_degree=min_degree, routing=routing)
        else:
            H, info = sample_stub_matching(G, partition, E=target_edges, rng=rng,
                                           degree_noise=degree_noise, boundary_scale=mixing_scale,
                                           min_degree=min_degree, routing=routing)
    else:
        raise ValueError(f"unknown method {method!r}")

    if connect:
        H, rep = connect_components(H, rng, partition=partition)   # module-aware: keeps every mu_A
        info["connect"] = rep

    if match_clustering:
        if directed:
            raise ValueError("match_clustering is undirected only")
        H, cinfo = raise_clustering(H, partition, rng, blueprint=G)   # connectivity-preserving
        info["clustering"] = cinfo
    return H, info
