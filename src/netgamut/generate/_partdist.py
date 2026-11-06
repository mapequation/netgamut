"""Flow-model partition distances: the generative map distance and its adjusted form.

The generative map equation turns a partition ``M`` of a network into a proper joint
distribution ``P^M`` over node pairs (:func:`flow_matrix`), so two partitions of the same
network compare as distributions.  The Jensen--Shannon distance between them is the
**generative map distance**

    GMD(A, B) = D(A, B) = d_JS(P^A, P^B) in [0, 1],

a metric with a Hilbertian (exactly Euclidean-embeddable) geometry and an exact
element-centric decomposition ``D^2 = sum_i p_i d_i^2``.  :func:`gms` reports the
similarity ``1 - D``; :func:`agmd` and :func:`agms` report the *adjusted* pair, a
Steinhaus transform against the one-module null partition that restores the
zero-at-trivial convention label-based measures have (see :func:`D_adjusted`).

Evaluation is blockwise and never materializes the ``n x n`` matrices.  Each row of
``P`` is piecewise proportional to the visit-rate vector ``p``,

    P[i, j] = C[a(i), b(j)] * p_i * p_j,

with ``a(i)`` the leaf module of ``i`` and ``C`` an ``m x m`` constant table built from
the module-tree path products (diagonal ``1 / (p_A + q_A)``, off-diagonal
:func:`cross_R`).  So the row conditional ``q_i = P_i / p_i`` depends on ``i`` only
through its leaf module, every node in a meet cell ``(leaf_A(i), leaf_B(i))`` has the
same element distance ``d_i``, and the total cost is
``O(cells^2 + n + m_A^2 + m_B^2)`` instead of ``O(n^2)``.

Both flow models share the exact factor ``p_i`` per row, so the joint JSD decomposes
exactly and is accumulated per meet cell.

Verified against the dense reference ``D_dist(flow_matrix(G, A), flow_matrix(G, B))``
and per-node against :func:`element_distances` to 1e-12, on six structures -- a toy
bridge, planted blocks, structure-ignoring random partitions, karate, nine triangles in
a ring, and a two-level directed hierarchy -- each in an undirected *and* a directed
form (``tests/test_generate.py``, which is where this is checked: the measure lives here,
so its checks do too).  Both forms are needed, because undirected ``ModuleTree`` sets
``enter is exit`` -- literally one dict -- so a path priced entirely by exit rates agrees
with the model by construction and the descending half goes untested.  The per-node
check compares the divergences ``d_i^2``, not ``d_i``: where ``d_i`` is a true zero this
path clamps to exactly 0 and the dense path returns the sqrt of its ~1e-17 rounding, a
~4e-9 difference carrying no information about either.  Nodes carrying no flow (``p_i = 0``,
e.g. isolated nodes) have zero weight in the aggregate: two partitions differing only in
where they put zero-flow nodes are flow-equivalent, and ``D`` is exactly ``0``.
"""

from __future__ import annotations

import math

import numpy as np

from . import _flowgen


# --------------------------------------------------------------------------- #
# Leaf-module constant tables from the module tree
# --------------------------------------------------------------------------- #
def two_level_constants(p_mod, exits, enters=None, non_redundant=True):
    """Leaf constants of a flat (two-level) flow model, as three arrays.

    ``C[a, b] = up[a] * down[b]`` off the diagonal and ``C[a, a] = diag[a]`` on it, so
    that ``P[i, j] = C[a(i), a(j)] p_i p_j``.  This is the two-level case of
    :func:`_leaf_constants`, taking module aggregates directly instead of a
    ``ModuleTree``: the shape a hot loop wants, where a search over candidate
    partitions already holds the aggregates and never builds a tree.
    :func:`_leaf_constants` calls it for ``maxd == 1``, so there is one derivation of
    the flat constants rather than one per caller.

    ``p_mod`` is the module flow, ``exits`` and ``enters`` its exit and enter rates;
    ``enters`` defaults to ``exits``, which is the undirected case -- there the two
    rates coincide, and a caller that has only ``q`` is saying its blueprint is
    undirected.  ``non_redundant`` excludes the module just left from the root
    normalizer.  A module with no flow, no exit or an empty normalizer scores 0
    rather than dividing by zero.
    """
    p_mod = np.asarray(p_mod, dtype=float)
    exits = np.asarray(exits, dtype=float)
    enters = exits if enters is None else np.asarray(enters, dtype=float)
    tot = p_mod + exits
    # Summed sequentially, in the order given.  ``ModuleTree`` sums a node's children
    # the same way, so a caller handing over the leaves in tree order gets the tree's
    # own normalizer bit for bit -- which is what keeps D between a partition and that
    # same partition wrapped in one extra level at exactly 0, where ``np.sum``'s
    # pairwise order differs by an ulp and the square root turns it into ~5e-9.
    Z = float(sum(enters.tolist())) - (enters if non_redundant else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        diag = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1.0), 0.0)
        up = np.where(tot > 0, exits / np.where(tot > 0, tot, 1.0), 0.0)
        up = np.where(Z > 0, up / np.where(Z > 0, Z, 1.0), 0.0)
        down = np.where(p_mod > 0, enters / np.where(p_mod > 0, p_mod, 1.0), 0.0)
    return diag, up, down


def two_level_table(p_mod, exits, enters=None, non_redundant=True):
    """The dense ``K x K`` table of :func:`two_level_constants`, diagonal filled in."""
    diag, up, down = two_level_constants(p_mod, exits, enters, non_redundant)
    C = np.outer(up, down)
    np.fill_diagonal(C, diag)
    return C


def _leaf_constants(tree, non_redundant=True):
    """C (m x m) with C[a, b] = P[i, j] / (p_i p_j) for i in leaf a, j in leaf b.

    Matches flow_matrix exactly, directed included: diagonal 1/(p_A + q_A^exit),
    off-diagonal cross_R via tree path products -- exit rates going up to the least
    common ancestor, enter rates coming back down -- and rows of leaves with
    q_A^exit = 0 stay 0 off-diagonal.  A flat partition is handed to
    :func:`two_level_table`, so the flat constants have one derivation however they
    are reached.
    Returns (C, leaves list, {leaf path: index}).
    """
    leaves = tree.leaves
    m = len(leaves)
    depths = np.array([len(A) for A in leaves], dtype=np.int64)
    maxd = int(depths.max())

    # no leaf path may be a proper prefix of another (flowgen assumes this too)
    leafset = set(leaves)
    for A in leaves:
        for t in range(1, len(A)):
            assert A[:t] not in leafset, f"leaf {A[:t]} is a prefix of leaf {A}"

    index = {A: a for a, A in enumerate(leaves)}
    if maxd == 1:  # flat two-level partition: the LCA is always the root
        return (
            two_level_table(
                np.array([tree.flow[A] for A in leaves]),
                np.array([tree.exit[A] for A in leaves]),
                np.array([tree.enter[A] for A in leaves]),
                non_redundant,
            ),
            leaves,
            index,
        )

    flow, children = tree.flow, tree.children
    # Exits are priced by exit rates, entries by enter rates, exactly as cross_R does.
    # For an undirected blueprint ModuleTree sets enter is exit is q (the same dict), so
    # this is bit-identical to pricing everything with q; they differ only when directed.
    enter, exit_ = tree.enter, tree.exit
    S = {nu: sum(enter[ch] for ch in ch_list) for nu, ch_list in children.items()}

    # UZ[a, L]  = U(A, L) / Z(A[:L], A[:L+1])   (exit leaf A up to LCA depth L,
    #             including the LCA normalization; cross_R's ascending part -- exit rates)
    # DEN[b, L] = enter[B[:L+1]] * prod_t enter[B[:t+1]]/S[B[:t]] / flow[B]
    #             (enter the depth-L+1 branch, descend to leaf B, pick node j -- enter rates)
    UZ = np.zeros((m, maxd))
    DEN = np.zeros((m, maxd))
    for a, A in enumerate(leaves):
        dA = len(A)
        denom = flow[A] + exit_[A]
        U = exit_[A] / denom if denom > 0 else 0.0  # exit leaf A
        for L in range(dA - 1, -1, -1):
            nu, c = A[:L], A[: L + 1]
            Z = (
                S.get(nu, 0.0)
                - (enter[c] if non_redundant else 0.0)
                + (exit_[nu] if nu != () else 0.0)
            )
            UZ[a, L] = U / Z if Z > 0 else 0.0
            if L > 0:  # keep exiting: factor exit[nu] / Z(nu, c)
                U = U * exit_[nu] / Z if Z > 0 else 0.0
        # descending part
        den = enter[A] / flow[A] if flow[A] > 0 else 0.0
        DEN[a, dA - 1] = den
        for L in range(dA - 2, -1, -1):
            Ssub = S.get(A[: L + 1], 0.0)
            den = den * enter[A[: L + 1]] / Ssub if Ssub > 0 else 0.0
            DEN[a, L] = den

    # integer-encode paths, padded with -1; LCA depth = first differing position
    codes = {}
    P = np.full((m, maxd), -1, dtype=np.int64)
    for a, A in enumerate(leaves):
        for t, x in enumerate(A):
            P[a, t] = codes.setdefault((t, x), len(codes))
    C = np.empty((m, m))
    cols = np.arange(m)
    chunk = max(1, min(m, 2048))
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        neq = P[lo:hi, None, :] != P[None, :, :]
        L = neq.argmax(axis=2)  # distinct leaves always differ somewhere
        C[lo:hi] = UZ[cols[lo:hi, None], L] * DEN[cols[None, :], L]

    within = np.array(
        [1.0 / (flow[A] + exit_[A]) if (flow[A] + exit_[A]) > 0 else 0.0 for A in leaves]
    )
    np.fill_diagonal(C, within)
    return C, leaves, index


# --------------------------------------------------------------------------- #
# Blockwise D
# --------------------------------------------------------------------------- #
def D_blockwise(G, partA, partB, non_redundant=True, return_info=False):
    """Flow-based partition distance D = d_JS(P^A, P^B) plus per-node distances.

    Exact (to floating point) equivalent of
    ``D_dist(flow_matrix(G, partA), flow_matrix(G, partB))`` and of
    ``element_distances`` per node, at O(cells^2 + n) cost.

    Returns (D, {node: d_i}); with ``return_info`` also a dict with m_A, m_B, cells.
    """
    treeA = _flowgen.ModuleTree(G, partA)
    treeB = _flowgen.ModuleTree(G, partB)
    CA, _, idxA = _leaf_constants(treeA, non_redundant)
    CB, _, idxB = _leaf_constants(treeB, non_redundant)
    mB = len(idxB)

    nodes = sorted(G.nodes())
    p = np.array([treeA.p[u] for u in nodes])
    ua = np.array([idxA[treeA.paths[u]] for u in nodes], dtype=np.int64)
    vb = np.array([idxB[treeB.paths[u]] for u in nodes], dtype=np.int64)

    # meet cells: unique (leaf_A, leaf_B) pairs, with flow weight W = sum of p
    cell_ids, node_cell = np.unique(ua * mB + vb, return_inverse=True)
    K = len(cell_ids)
    cu = cell_ids // mB  # A-module index of each cell
    cv = cell_ids % mB  # B-module index of each cell
    w = np.bincount(node_cell, weights=p, minlength=K)  # cell flow

    # per-cell JSD between row conditionals: every node j in cell (a, b) contributes
    # p_j * f(C_A[u, a], C_B[v, b]); p_j cancels inside the logs.
    jsd_cell = np.zeros(K)
    chunk = max(1, min(K, 512))
    for lo in range(0, K, chunk):
        hi = min(lo + chunk, K)
        X = CA[cu[lo:hi, None], cu[None, :]]
        Y = CB[cv[lo:hi, None], cv[None, :]]
        M = X + Y  # = 2 * mixture / (that factor of 2 folded into the logs below)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(
                X > 0, X * np.log2(np.divide(2.0 * X, M, out=np.ones_like(X), where=M > 0)), 0.0
            )
            ty = np.where(
                Y > 0, Y * np.log2(np.divide(2.0 * Y, M, out=np.ones_like(Y), where=M > 0)), 0.0
            )
        jsd_cell[lo:hi] = 0.5 * ((tx + ty) @ w)
    jsd_cell = np.maximum(jsd_cell, 0.0)

    d_cell = np.sqrt(jsd_cell)
    D = math.sqrt(max(0.0, float(w @ jsd_cell)))  # D^2 = sum_i p_i d_i^2
    d_node = {u: float(d_cell[node_cell[i]]) for i, u in enumerate(nodes)}
    if return_info:
        return D, d_node, {"n": len(nodes), "m_A": len(idxA), "m_B": mB, "cells": K}
    return D, d_node


# --------------------------------------------------------------------------- #
# Adjusted distance: Steinhaus transform with the one-module null partition
# --------------------------------------------------------------------------- #
def null_partition(G):
    """The one-module (null) partition N of G: every node in a single module."""
    return {u: 1 for u in G.nodes()}


def D_null(G, part, non_redundant=True):
    """D(part, N) where N is the one-module null partition of G."""
    D, _ = D_blockwise(G, part, null_partition(G), non_redundant=non_redundant)
    return D


def D_adjusted(G, partA, partB, non_redundant=True):
    """Adjusted generative map distance (Steinhaus transform of D, base point N).

        D_hat(A, B) = 2 D(A, B) / (D(A, N) + D(B, N) + D(A, B)),

    with N the one-module null partition of G and the convention 0/0 -> 0
    (both partitions flow-equivalent to N, hence to each other).  Because the
    Steinhaus transform preserves metricity, D_hat is a (pseudo)metric on the
    partitions of a FIXED network; the base point N differs across networks,
    so D_hat values are not comparable between networks.  Properties:
    D_hat in [0, 1], D_hat(A, N) = 1 whenever D(A, N) > 0, and
    D_hat = 0 iff D = 0 (same kernel: flow-equivalent partitions).

    Returns (D_hat, D, D_A_null, D_B_null).
    """
    D, _ = D_blockwise(G, partA, partB, non_redundant=non_redundant)
    D_A_null = D_null(G, partA, non_redundant=non_redundant)
    D_B_null = D_null(G, partB, non_redundant=non_redundant)
    denom = D_A_null + D_B_null + D
    D_hat = 2.0 * D / denom if denom > 0.0 else 0.0
    return D_hat, D, D_A_null, D_B_null


# --------------------------------------------------------------------------- #
# Named public API: the generative map distance and similarity, raw and adjusted
# --------------------------------------------------------------------------- #
def gmd(G, partA, partB, non_redundant=True, element=False):
    """Generative map distance GMD = D(A, B) = d_JS(P^A, P^B), in [0, 1].

    A metric on the flow models of partitions of ``G``: zero exactly on
    flow-equivalent partitions (relabelings, and hierarchy levels that route no
    flow), one when the two flow models share no support.  ``G`` may be directed,
    weighted, and either partition may be hierarchical (``{node: path tuple}``).

    With ``element=True`` returns ``(D, {node: d_i})``, the element-centric
    decomposition satisfying ``D^2 = sum_i p_i d_i^2`` exactly.
    """
    D, d_node = D_blockwise(G, partA, partB, non_redundant=non_redundant)
    return (D, d_node) if element else D


def gms(G, partA, partB, non_redundant=True):
    """Generative map similarity GMS = 1 - GMD, in [0, 1] (1 = flow-equivalent).

    Similarities of random partitions do not approach 0: every partition's flow
    model shares the visit-rate skeleton ``p p^T``, so GMS reports how much flow
    two descriptions agree about, not a rescaled label overlap.  Use :func:`agms`
    for the zero-at-trivial convention.
    """
    return 1.0 - gmd(G, partA, partB, non_redundant=non_redundant)


def agmd(G, partA, partB, non_redundant=True):
    """Adjusted generative map distance AGMD = D_hat, in [0, 1] (see :func:`D_adjusted`).

    Steinhaus transform of GMD against the one-module null partition, so the
    trivial partition sits at distance 1 from every partition that carries flow
    structure.  Metric on the partitions of a FIXED network; the base point is
    network-specific, so values are not comparable across networks.
    """
    return D_adjusted(G, partA, partB, non_redundant=non_redundant)[0]


def agms(G, partA, partB, non_redundant=True):
    """Adjusted generative map similarity AGMS = 1 - AGMD, in [0, 1].

    Reproduces the zero-at-trivial convention of chance-corrected label measures
    without a random model: AGMS = 0 whenever either partition is flow-equivalent
    to the one-module null, and 1 exactly on flow-equivalent partitions.
    """
    return 1.0 - agmd(G, partA, partB, non_redundant=non_redundant)
