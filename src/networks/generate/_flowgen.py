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
    rates, and stubs pair uniformly subject to the *non-redundant constraint* (never
    pair back into the branch just exited).  Preserves the full degree sequence,
    yields connected simple graphs, handles hierarchy, and reduces to LFR for the
    special case power-law degrees + power-law community sizes + flat ``mu``.

``sample_benchmark`` is the convenience entry point: give it a blueprint (and
optionally a partition; otherwise Infomap supplies one) and it returns a sample.
``mixing_scale`` rescales every module's mixing to sweep detectability from a single
blueprint.

This module treats networks as **undirected**; directed flow (enter != exit rates)
is a planned extension.  The maths is documented in the project notebooks
"Non-redundant map equation - expanded form" and
"Generative map equation - properties and regularization".
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np

try:
    import infomap
except ImportError:  # infomap is only needed for the Infomap helpers
    infomap = None


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

    Attributes
    ----------
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
    def escalation_probs(self, A, kind="out", routing="boundary"):
        """Stop-level distribution for a boundary stub of leaf module *A*.

        Returns ``[(nu, tag, prob), ...]``: the walker that left ``A`` re-enters the tree
        at internal node ``nu`` having exited branch ``tag``, so it pairs inside pool
        ``nu`` and must not pair back into ``tag`` (the non-redundant constraint).
        Deepest level first; the probabilities sum to one.

        ``kind="out"`` routes out-stubs by exit rates, ``kind="in"`` in-stubs by enter
        rates; the two coincide when undirected.  All three rules are identical for a
        two-level partition, where every boundary stub stops at the root:

        ``"boundary"`` (default) -- escalate with ``q[nu] / sum_children q[ch]``, the
        share of the children's boundary flow that also leaves ``nu``.  Measured best
        overall: it reproduces the blueprint's per-level boundary counts on undirected
        blueprints exactly and its cross-branch counts on directed ones to 2%.

        ``"model"`` -- the flow model's own escalation probability
        ``q^exit[nu] / Z(nu, child)``, with the leave-one-out normaliser of
        :func:`cross_R`, so sampler and joint ``P`` describe the same walk.  Ties
        ``"boundary"`` on recovery but places 14% too many links across the top-level
        split; prefer it when generating from model parameters rather than a blueprint.

        ``"observed"`` -- the blueprint's own histogram.  A boundary link re-enters the
        tree at exactly the depth of its endpoints' lowest common ancestor, so this
        distribution is observed rather than modelled.  It preserves per-level counts
        but its out- and in-histograms need not balance at a pool, and the leftovers
        that escalate cost recovery (directed level-1 AMI 0.917 against 0.988).
        """
        depth = len(A)
        if routing == "observed":
            hist = (self.stop_out if kind == "out" else self.stop_in).get(A, {})
            total = sum(hist.values())
            if total > 0:
                return [(A[:t], A[:t + 1], hist.get(t, 0.0) / total)
                        for t in range(depth - 1, -1, -1)]
            routing = "model"           # no boundary links to learn from
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


def _pair_stubs(stubs, rng, max_sweeps=200):
    """Pair stubs [(node, tag), ...]: tags must differ within a pair, node pairs must not repeat.

    Conflicts fixed by random swaps; unfixable pairs dissolve into leftovers.
    Each unordered node pair arises in exactly one pool, so per-pool dedup is global dedup.
    """
    stubs = list(stubs)
    rng.shuffle(stubs)
    leftover = []
    if len(stubs) % 2:
        leftover.append(stubs.pop())
    pairs = [(stubs[2 * k], stubs[2 * k + 1]) for k in range(len(stubs) // 2)]

    def find_bad(pairs):
        seen, bad = set(), []
        for k, (a, b) in enumerate(pairs):
            key = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
            if a[1] == b[1] or key in seen:
                bad.append(k)
            else:
                seen.add(key)
        return bad

    for _ in range(max_sweeps):
        bad = find_bad(pairs)
        if not bad or len(pairs) < 2:
            break
        for k in bad:
            j = int(rng.integers(len(pairs)))
            if j != k:
                (a, b), (c, d) = pairs[k], pairs[j]
                pairs[k], pairs[j] = (a, d), (c, b)
    bad = set(find_bad(pairs))
    good = [pr for k, pr in enumerate(pairs) if k not in bad]
    for k in bad:
        leftover.extend(pairs[k])
    return good, leftover


def _route_stubs(tree, per_node, kind, rng, routing="boundary"):
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
                         boundary_scale=1.0, min_degree=1, routing="boundary"):
    """Microcanonical (LFR-like) benchmark sampler from a blueprint's flow statistics.

    ``routing`` selects the hierarchical escalation rule shared with the directed
    sampler -- ``"boundary"``, ``"model"`` or ``"observed"``, see
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

    # 2. internal/boundary split per module (mixing mu_A = q_A / p_A), apportioned by degree
    internal, boundary = {}, {}
    for A in tree.leaves:
        mu = tree.q[A] / tree.flow[A] if tree.flow[A] > 0 else 0.0
        mu = min(mu * boundary_scale, 1.0)
        ks = np.array([k_of[u] for u in tree.nodes_in[A]])
        I_A = min(int(round(ks.sum() * (1.0 - mu) / 2.0)) * 2, int(ks.sum()))
        ints = np.minimum(apportion(I_A, ks, rng), ks)
        for u, ki, ii in zip(tree.nodes_in[A], ks, ints):
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


def _pair_bipartite(out_stubs, in_stubs, rng, forbid_same_group=False, used=None, max_sweeps=200):
    """Pair out-stubs [(node, group), ...] to in-stubs, forbidding self-loops, duplicate directed
    edges (via a shared `used` set) and optionally same-group pairs (non-redundant boundary rule).

    Returns (edges, leftover_out, leftover_in, used).
    """
    out_stubs, in_stubs = list(out_stubs), list(in_stubs)
    rng.shuffle(out_stubs)
    rng.shuffle(in_stubs)
    used = set() if used is None else used
    n = min(len(out_stubs), len(in_stubs))
    leftover_out, leftover_in = out_stubs[n:], in_stubs[n:]
    out_stubs, in_stubs = out_stubs[:n], in_stubs[:n]

    def bad_at(i):
        o, ii = out_stubs[i], in_stubs[i]
        return (o[0] == ii[0] or (forbid_same_group and o[1] == ii[1])
                or (o[0], ii[0]) in used)

    for _ in range(max_sweeps):
        bad = [i for i in range(n) if bad_at(i)]
        if not bad:
            break
        for i in bad:
            j = int(rng.integers(n))
            in_stubs[i], in_stubs[j] = in_stubs[j], in_stubs[i]

    edges = []
    for i in range(n):
        if bad_at(i):
            leftover_out.append(out_stubs[i])
            leftover_in.append(in_stubs[i])
        else:
            used.add((out_stubs[i][0], in_stubs[i][0]))
            edges.append((out_stubs[i][0], in_stubs[i][0]))
    return edges, leftover_out, leftover_in, used


def sample_stub_matching_directed(G, partition, E=None, rng=None, boundary_scale=1.0,
                                  min_degree=1, topup=True, routing="boundary"):
    """Directed microcanonical benchmark sampler from a blueprint's directed modular flow.

    Preserves the in- and out-degree sequences (up to a few dropped stubs), reproduces each
    module's directed mixing ``mu^exit`` and ``mu^enter`` separately, and enforces the
    non-redundant boundary rule (a module's exit stubs never re-enter that module).

    **Hierarchical**: boundary stubs route up the module tree by the hierarchical exit
    rates (out-stubs) and enter rates (in-stubs), pair inside the pool of the internal
    node they stopped at, and escalate to the parent pool when they cannot be placed --
    the directed counterpart of :func:`sample_stub_matching`.  A two-level partition
    reduces to a single root pool, i.e. to the flat sampler.

    With ``topup`` (default), leftover stubs that the non-redundant boundary pass could not
    place (mostly hub stubs blocked by the same-branch or duplicate-arc constraint) get one
    more pairing pass with the same-branch rule relaxed, recovering the link count and the
    hub degrees at the cost of a few extra within-module arcs.

    ``routing`` selects the hierarchical escalation rule -- ``"boundary"``, ``"model"``
    or ``"observed"``, see :meth:`ModuleTree.escalation_probs`.

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

    # 1. internal/boundary split per leaf module, from its directed mixing
    iout, bout, iin, bin_ = {}, {}, {}, {}
    for A in tree.leaves:
        us = tree.nodes_in[A]
        mo = min(tree.mu_out(A) * boundary_scale, 1.0)
        mi = min(tree.mu_in(A) * boundary_scale, 1.0)
        ko = np.array([kout_of[u] for u in us])
        ki = np.array([kin_of[u] for u in us])
        bo = np.minimum(apportion(min(int(round(ko.sum() * mo)), int(ko.sum())), ko, rng), ko)
        bi = np.minimum(apportion(min(int(round(ki.sum() * mi)), int(ki.sum())), ki, rng), ki)
        for u, k, b in zip(us, ko, bo):
            bout[u], iout[u] = int(b), int(k - b)
        for u, k, b in zip(us, ki, bi):
            bin_[u], iin[u] = int(b), int(k - b)

    # 2. route boundary stubs up the module tree
    pools_out = _route_stubs(tree, bout, "out", rng, routing)
    pools_in = _route_stubs(tree, bin_, "in", rng, routing)

    # 3a. internal arcs: pair within each leaf module; leftovers escalate to its parent
    used, edges = set(), []
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
    dropped_before, root_lo, root_li = 0, [], []
    for d in range(depth, -1, -1):
        for nu in sorted(set(k for k in pools_out if len(k) == d)
                         | set(k for k in pools_in if len(k) == d)):
            e, lo, li, used = _pair_bipartite(pools_out.get(nu, []), pools_in.get(nu, []),
                                              rng, forbid_same_group=True, used=used)
            edges.extend(e)
            if nu == ():
                dropped_before, root_lo, root_li = len(lo), lo, li
            else:
                pools_out[nu[:-1]].extend((u, nu) for u, _ in lo)
                pools_in[nu[:-1]].extend((u, nu) for u, _ in li)

    # 4. top-up: place what the non-redundant rule could not, same-branch allowed
    if topup and root_lo and root_li:
        e2, root_lo, root_li, used = _pair_bipartite(root_lo, root_li, rng,
                                                     forbid_same_group=False, used=used)
        edges.extend(e2)

    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)
    return H, {"target_E": E, "realized_E": H.number_of_edges(),
               "dropped_out": len(root_lo), "dropped_in": len(root_li),
               "dropped_before_topup": dropped_before,
               "levels": max(len(A) for A in tree.leaves), "routing": routing}


def connect_components(H, rng, max_passes=12):
    """Merge components with degree-preserving double-edge swaps (directed or undirected).

    Each non-giant component is joined to the giant by one swap
    (a->b),(c->d) -> (a->d),(c->b); isolated singletons are attached by splitting a giant edge
    (a small, controlled degree perturbation). Passes repeat until a single component remains or
    no further progress is made. Modifies H in place.
    """
    directed = H.is_directed()
    comp_fn = nx.weakly_connected_components if directed else nx.connected_components
    nbrs_fn = H.successors if directed else H.neighbors
    info = {"merges": 0, "singletons_attached": 0}

    def random_edge(pool):                             # a random edge with tail in `pool`
        for _ in range(200):
            c = pool[int(rng.integers(len(pool)))]
            nb = list(nbrs_fn(c))
            if nb:
                return c, nb[int(rng.integers(len(nb)))]
        return None

    for _ in range(max_passes):
        comps = sorted(comp_fn(H), key=len, reverse=True)
        if len(comps) <= 1:
            break
        giant = list(comps[0])
        progressed = False
        for small in comps[1:]:
            small = list(small)
            es, eg = random_edge(small), random_edge(giant)
            done = False
            if es and eg:
                for _ in range(20):
                    a, b = es
                    c, d = eg
                    if a != d and c != b and not H.has_edge(a, d) and not H.has_edge(c, b):
                        H.remove_edge(a, b); H.remove_edge(c, d)
                        H.add_edge(a, d); H.add_edge(c, b)
                        info["merges"] += 1
                        done = True
                        break
                    es, eg = random_edge(small), random_edge(giant)
            if not done and eg:                        # singleton: split a giant edge to attach it
                iso, (c, d) = small[0], eg
                if iso not in (c, d) and not H.has_edge(c, iso) and not H.has_edge(iso, d):
                    H.remove_edge(c, d); H.add_edge(c, iso); H.add_edge(iso, d)
                    info["singletons_attached"] += 1
                    done = True
            if done:
                giant.extend(small)
                progressed = True
        if not progressed:
            break
    info["components"] = (nx.number_weakly_connected_components(H) if directed
                          else nx.number_connected_components(H))
    return H, info


def raise_clustering(H, partition, rng, target=None, swap_factor=25):
    """Raise clustering by closing open wedges with within-module degree-preserving swaps.

    For an open wedge b--a--c (a adjacent to both, b not adjacent to c), pick edges (b,x) and
    (c,y) inside the same module and rewire remove (b,x),(c,y) -> add (b,c),(x,y): this closes the
    wedge into triangle a-b-c while preserving every node's degree. Keeping all endpoints inside
    one module preserves each module's boundary flow (mixing) and every cross-module edge, so only
    the within-module wiring changes. A swap is kept only if it does not lower the triangle count.

    ``target`` is a global transitivity ($3T/W$) to stop at; ``swap_factor`` bounds the attempts
    at ``swap_factor`` times the internal-edge count. Undirected only. Returns (nx.Graph, info).
    Partial by nature: it typically closes part of the gap to a real network's clustering, more on
    denser blueprints, and adds runtime---so it is opt-in (see ``sample_benchmark``).
    """
    comm = as_paths(partition)
    adj = {u: set(H.neighbors(u)) for u in H}
    mod_nodes = defaultdict(list)
    for u in H:
        mod_nodes[comm[u]].append(u)
    n_int = sum(1 for u, v in H.edges() if comm[u] == comm[v])
    mods = [m for m in mod_nodes if len(mod_nodes[m]) >= 3]
    if not mods or n_int == 0:
        return H, {"swaps": 0, "attempts": 0}
    sizes = np.array([len(mod_nodes[m]) for m in mods], float)
    cum = np.cumsum(sizes) / sizes.sum()

    W = sum(k * (k - 1) // 2 for k in (len(adj[u]) for u in adj))
    T = sum(len(adj[u] & adj[v]) for u, v in H.edges()) // 3
    T_target = None if target is None else target * W / 3.0

    def common(x, y):
        return len(adj[x] & adj[y])

    def same_mod_nbrs(u, A):
        return [w for w in adj[u] if comm[w] == A]

    swaps, attempts = 0, swap_factor * n_int
    for _ in range(attempts):
        A = mods[int(np.searchsorted(cum, rng.random()))]
        nodes = mod_nodes[A]
        a = nodes[int(rng.integers(len(nodes)))]
        na = same_mod_nbrs(a, A)
        if len(na) < 2:
            continue
        b = na[int(rng.integers(len(na)))]
        c = na[int(rng.integers(len(na)))]
        if b == c or c in adj[b]:                    # need an OPEN wedge b--a--c
            continue
        nb = [x for x in same_mod_nbrs(b, A) if x not in (a, c) and x not in adj[c]]
        nc = [y for y in same_mod_nbrs(c, A) if y not in (a, b)]
        if not nb or not nc:
            continue
        x = nb[int(rng.integers(len(nb)))]
        y = nc[int(rng.integers(len(nc)))]
        if x == y or y in adj[x] or x in adj[c] or y in adj[b]:
            continue
        # rewire remove (b,x),(c,y) -> add (b,c),(x,y); exact triangle delta while mutating
        dT = -common(b, x); adj[b].discard(x); adj[x].discard(b)
        dT -= common(c, y); adj[c].discard(y); adj[y].discard(c)
        dT += common(b, c); adj[b].add(c); adj[c].add(b)
        dT += common(x, y); adj[x].add(y); adj[y].add(x)
        if dT <= 0:                                  # revert non-improving swaps
            adj[b].discard(c); adj[c].discard(b)
            adj[x].discard(y); adj[y].discard(x)
            adj[b].add(x); adj[x].add(b)
            adj[c].add(y); adj[y].add(c)
            continue
        T += dT
        swaps += 1
        if T_target is not None and swaps % max(n_int // 4, 1) == 0 and T >= T_target:
            break

    Hc = nx.Graph()
    Hc.add_nodes_from(H.nodes())
    for u in adj:
        for v in adj[u]:
            if u < v:
                Hc.add_edge(u, v)
    return Hc, {"swaps": swaps, "attempts": attempts}


# --------------------------------------------------------------------------- #
# Infomap helpers
# --------------------------------------------------------------------------- #
def run_infomap(G, two_level=False, num_trials=20, seed=1, directed=None, extra=None,
                **options):
    """Run Infomap on a networkx graph; returns ``({node: path tuple}, codelength)``.

    Uses Infomap's keyword API, so engine settings are keyword arguments
    (``markov_time=1.5``, ``flow_model="rawdir"``, ...) rather than a CLI string, and
    node labels of any type survive the round trip -- integer labels come back
    unchanged, other labels are mapped back through ``Result.names``.

    ``directed`` defaults to ``G.is_directed()``; when true Infomap runs with directed
    flow, so a module's enter and exit rates differ.  ``extra`` still accepts a raw
    argument string for CLI parity when an option has no keyword.
    """
    if infomap is None:
        raise ImportError("infomap package not available")
    if directed is None:
        directed = G.is_directed()
    result = infomap.run(G, num_trials=num_trials, seed=seed, two_level=two_level,
                         directed=directed, silent=True,
                         **({"args": extra} if extra else {}), **options)
    names = result.names
    paths = result.multilevel_modules()
    if names:
        return {names[i]: path for i, path in paths.items()}, result.codelength
    return dict(paths), result.codelength


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
                     match_clustering=False, clustering_swap_factor=25,
                     rng=None, degree_noise=False, num_trials=20, seed=1,
                     routing="boundary", topup=True):
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
    routing : {"boundary", "model", "observed"}
        Hierarchical boundary-stub escalation rule; see
        :meth:`ModuleTree.escalation_probs`.  Only matters for hierarchical partitions.
    topup : bool
        Directed only: place stubs the non-redundant rule could not, recovering the link
        count and hub degrees at the cost of a few extra within-module arcs.
    directed : bool or None
        Force directed/undirected; defaults to G.is_directed().
    connect : bool
        Run a degree-preserving connectivity-repair pass so the sample is a single component.
    min_degree : int
        Floor on sampled node degree (avoids isolated nodes).
    match_clustering : bool
        Opt-in (default False, undirected only): run a within-module, degree- and mixing-preserving
        wedge-closing pass (``raise_clustering``) toward the blueprint's transitivity. Adds realism
        (real networks are more clustered than a configuration-model benchmark) at extra runtime; it
        only partially closes the gap on sparse blueprints. Off by default because of that cost and
        because it perturbs recovery slightly.
    clustering_swap_factor : int
        Attempt budget for the clustering pass (attempts = factor * internal-edge count).

    Returns
    -------
    (H, info) : the sampled graph and a dict of diagnostics (incl. connectivity repair).
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
                                                    min_degree=min_degree, topup=topup,
                                                    routing=routing)
        else:
            H, info = sample_stub_matching(G, partition, E=target_edges, rng=rng,
                                           degree_noise=degree_noise, boundary_scale=mixing_scale,
                                           min_degree=min_degree, routing=routing)
    else:
        raise ValueError(f"unknown method {method!r}")

    if connect:
        H, rep = connect_components(H, rng)
        info["connect"] = rep

    if match_clustering:
        if directed:
            raise ValueError("match_clustering is undirected only")
        H, cinfo = raise_clustering(H, partition, rng, target=nx.transitivity(G),
                                    swap_factor=clustering_swap_factor)
        if connect:
            H, _ = connect_components(H, rng)   # wedge swaps can rarely fragment
        info["clustering"] = cinfo
    return H, info
