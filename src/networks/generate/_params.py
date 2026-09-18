"""Parametric fits and flow-model edge weights from the domain benchmark.

Adapted from the paper's ``domainbench.py``: MLE family fitting with AIC model
selection (``fit_families``), the flow-weighted Beta mixing fit, the
generalized-LFR parametric generator (``sample_from_params``) and flow-model
edge weights for weighted benchmarks (``flow_edge_weights``).  Kept faithful
to the source -- these functions back the manuscript's published numbers.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from . import _flowgen as fg


def _und(G):
    """Undirected weighted view (symmetrized by summing both directions)."""
    if not G.is_directed():
        return G
    U = nx.Graph()
    U.add_nodes_from(G.nodes())
    for u, v, w in G.edges(data="weight", default=1.0):
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    return U


def infomap_full(G, two_level, trials, seed, directed):
    """Like _flowgen.run_infomap but also returns Infomap's native codelength
    statistics (respecting directedness), for the compressibility fingerprint."""
    args = f"--silent --num-trials {trials} --seed {seed}"
    if two_level:
        args += " --two-level"
    if directed:
        args += " --directed"
    import infomap as _infomap
    im = _infomap.Infomap(args)
    for u, v, w in G.edges(data="weight", default=1.0):
        im.add_link(int(u), int(v), float(w))
    im.run()
    part = {node.node_id: tuple(node.path[:-1]) for node in im.tree if node.is_leaf}
    stats = dict(codelength=float(im.codelength),
                 one_level_codelength=float(im.one_level_codelength),
                 savings=float(im.relative_codelength_savings))
    return part, stats


def fit_families(values, support_min=None):
    """MLE fits + AIC model selection over discrete families: Zipf power law,
    geometric, truncated Poisson, discretized log-normal.

    All families are fit on the SHIFTED variable k' = k - (support_min - 1)
    >= 1, where support_min defaults to min(values) -- otherwise a family
    anchored at 1 (Zipf) is structurally misfit to quantities that live on a
    shifted support, e.g. module sizes starting at 5 (round-2 review finding).
    Returns {family: {params, loglik, aic, aic_weight}, 'winner', 'support_min'}."""
    from scipy import optimize, special, stats

    raw = np.asarray([v for v in values if v >= 1], dtype=float)
    if support_min is None:
        support_min = int(raw.min()) if len(raw) else 1
    k = raw - (support_min - 1)
    k = k[k >= 1]
    n, m = len(k), k.mean() if len(k) else 0.0
    out = {}
    if n < 3:
        return {"winner": "insufficient", "n_values": n,
                "support_min": support_min}

    # Zipf: P(k) = k^-alpha / zeta(alpha)
    sum_log = np.log(k).sum()

    def nll(a):
        return n * np.log(special.zeta(a)) + a * sum_log

    res = optimize.minimize_scalar(nll, bounds=(1.05, 8.0), method="bounded")
    out["powerlaw"] = dict(params={"alpha": float(res.x)},
                           loglik=float(-res.fun), n_params=1)

    # Geometric on k>=1: P(k) = (1-p)^(k-1) p, MLE p = 1/mean
    p = 1.0 / m
    ll = float(((k - 1) * np.log1p(-p)).sum() + n * np.log(p)) if m > 1 else -1e18
    out["geometric"] = dict(params={"p": p}, loglik=ll, n_params=1)

    # Truncated Poisson (k>=1): solve lam / (1 - exp(-lam)) = mean
    if m > 1.0 + 1e-9:

        def f(lam):
            return lam / (1.0 - math.exp(-lam)) - m

        lam = optimize.brentq(f, 1e-9, m)
        ll = float((k * np.log(lam)).sum() - n * lam
                   - special.gammaln(k + 1).sum()
                   - n * np.log1p(-math.exp(-lam)))
        out["poisson"] = dict(params={"lam": lam}, loglik=ll, n_params=1)

    # Discretized log-normal (approximate MLE via moments of log k)
    logk = np.log(k)
    mu_l, s_l = float(logk.mean()), float(max(logk.std(), 1e-6))
    hi = stats.norm.cdf((np.log(k + 0.5) - mu_l) / s_l)
    lo = stats.norm.cdf((np.log(np.maximum(k - 0.5, 0.25)) - mu_l) / s_l)
    norm = 1.0 - stats.norm.cdf((math.log(0.5) - mu_l) / s_l)
    probs = np.maximum((hi - lo) / norm, 1e-300)
    out["lognormal"] = dict(params={"mu": mu_l, "sigma": s_l},
                            loglik=float(np.log(probs).sum()), n_params=2)

    aics = {f: 2 * d["n_params"] - 2 * d["loglik"] for f, d in out.items()}
    amin = min(aics.values())
    weights = {f: math.exp(-(a - amin) / 2) for f, a in aics.items()}
    wsum = sum(weights.values())
    for f in out:
        out[f]["aic"] = aics[f]
        out[f]["aic_weight"] = weights[f] / wsum
    out["winner"] = min(aics, key=aics.get)
    out["n_values"] = n
    out["support_min"] = support_min
    return out


def fit_mixing_beta(mixing, flows):
    """Flow-weighted Beta fit (method of moments) to per-module mixing values."""
    mu = np.clip(np.asarray(mixing, dtype=float), 0.0, 1.0)
    w = np.asarray(flows, dtype=float)
    w = w / w.sum()
    mean = float((w * mu).sum())
    var = float((w * (mu - mean) ** 2).sum())
    if var < 1e-6 or mean <= 0 or mean >= 1:
        return dict(family="delta", mu=mean, var=var)
    conc = mean * (1 - mean) / var - 1.0
    if conc <= 0:
        return dict(family="delta", mu=mean, var=var)
    return dict(family="beta", alpha=mean * conc, beta=(1 - mean) * conc,
                mu=mean, var=var)


def _draw_from_family(family, params, size, rng, support_min=1):
    """Draw integers from a fitted family, shifted back to its support."""
    if family == "powerlaw":
        out = rng.zipf(params["alpha"], size=size)
    elif family == "geometric":
        out = rng.geometric(params["p"], size=size)
    elif family == "poisson":
        out = np.zeros(size, dtype=int)
        todo = np.arange(size)
        while len(todo):  # rejection: truncate at k >= 1
            draw = rng.poisson(params["lam"], size=len(todo))
            ok = draw >= 1
            out[todo[ok]] = draw[ok]
            todo = todo[~ok]
    elif family == "lognormal":
        out = np.maximum(np.rint(rng.lognormal(
            params["mu"], params["sigma"], size=size)).astype(int), 1)
    else:
        raise ValueError(f"unknown family {family!r}")
    return out + (support_min - 1)


def fit_clustering_profile(G, min_degree=2):
    """Fit the local-clustering profile ``c(k) = c0 * k^-alpha`` by least squares on
    ``log c`` over the nodes of degree ``>= min_degree`` with ``c_u > 0``.

    The fingerprint's clustering component, in the same spirit as the degree and
    module-size families: a blueprint reduces to two numbers instead of its own triangle
    counts, so the parametric generator can be handed a clustering target without a
    blueprint at sampling time.  Returns ``{"c0", "alpha", "n_fit", "c_mean"}``, and
    ``c(k)`` as the callable :func:`clustering_profile` builds from it.

    ``c0`` is the fitted clustering of a degree-1 node, which the power law extrapolates;
    the returned callable clips ``c(k)`` to ``[0, 1]`` so a steep fit cannot ask for more
    triangles than a node's degree allows.
    """
    import networkx as nx

    U = nx.Graph(G) if G.is_directed() else G
    cl = nx.clustering(U)
    ks, cs = [], []
    for u, c in cl.items():
        k = U.degree(u)
        if k >= min_degree and c > 0:
            ks.append(float(k))
            cs.append(float(c))
    c_mean = float(np.mean(list(cl.values()))) if cl else 0.0
    if len(ks) < 3:                       # nothing to fit: a flat profile at the mean
        return dict(c0=c_mean, alpha=0.0, n_fit=len(ks), c_mean=c_mean)
    x = np.log(np.asarray(ks))
    y = np.log(np.asarray(cs))
    A = np.vstack([np.ones_like(x), x]).T
    (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(c0=float(np.exp(b0)), alpha=float(-b1), n_fit=len(ks), c_mean=c_mean)


def clustering_profile(fit):
    """The callable ``c(k)`` of a :func:`fit_clustering_profile` result, clipped to
    ``[0, 1]``; ``raise_clustering`` turns it into per-node triangle targets."""
    c0, alpha = float(fit["c0"]), float(fit["alpha"])

    def c(k):
        k = max(int(k), 1)
        return float(min(max(c0 * k ** (-alpha), 0.0), 1.0))

    return c


def sample_from_params(n, deg_fit, size_fit, mixing_fit, rng,
                       min_size=2, connect=True, k_mean=None, k_max=None,
                       size_max=None, clustering_fit=None, clustering_kwargs=None):
    """Generate a two-level benchmark purely from fitted parameters -- the
    generalized-LFR generator.  Draw module sizes and degrees from their
    families, per-module mixing from the Beta (or delta), then stub-match
    exactly like the microcanonical sampler.

    Mirroring LFR's (tau, <k>, k_max, max_community) parameterization,
    drawn degrees are rescaled to `k_mean` and capped at `k_max`, and sizes
    capped at `size_max` -- otherwise an unbounded heavy-tailed draw changes
    the density or collapses the partition into one giant module, and the
    comparison between family variants is not density-matched.
    Returns (H, partition, info)."""
    smin_s = size_fit.get("support_min", 1)
    sizes = []
    while sum(sizes) < n:
        s = int(_draw_from_family(size_fit["family"], size_fit["params"],
                                  1, rng, support_min=smin_s)[0])
        s = max(s, min_size)
        if size_max is not None:
            s = min(s, int(size_max))
        sizes.append(s)
    sizes[-1] = max(n - sum(sizes[:-1]), min_size)  # trim to n (approximately)
    partition, nodes_in = {}, {}
    node = 0
    for mod, s in enumerate(sizes):
        members = list(range(node, node + s))
        for u in members:
            partition[u] = (mod,)
        nodes_in[(mod,)] = members
        node += s
    n_real = node

    # degrees: draw, rescale to the target mean, cap
    k = _draw_from_family(deg_fit["family"], deg_fit["params"], n_real, rng,
                          support_min=deg_fit.get("support_min", 1)).astype(float)
    if k_mean is not None and k.mean() > 0:
        k *= k_mean / k.mean()
    k = np.maximum(np.rint(k).astype(int), 1)
    cap = n_real - 1 if k_max is None else min(int(k_max), n_real - 1)
    k = np.minimum(k, cap)

    # per-module mixing
    if mixing_fit["family"] == "delta":
        mus = np.full(len(sizes), mixing_fit["mu"])
    else:
        mus = rng.beta(mixing_fit["alpha"], mixing_fit["beta"], size=len(sizes))

    # stub matching: internal within module, boundary in one global pool
    pool = []
    edges = []
    for mod, s in enumerate(sizes):
        members = nodes_in[(mod,)]
        ks = k[members]
        I_A = min(int(round(ks.sum() * (1.0 - mus[mod]) / 2.0)) * 2, int(ks.sum()))
        ints = np.minimum(fg.apportion(I_A, ks, rng), ks)
        stubs = [(u, u) for u, ki in zip(members, ints) for _ in range(int(ki))]
        pairs, leftover = fg._pair_stubs(stubs, rng)
        edges.extend((a[0], b[0]) for a, b in pairs)
        pool.extend((u, (mod,)) for (u, _t) in leftover)
        pool.extend((u, (mod,)) for u, ki, ii in zip(members, ks, ints)
                    for _ in range(int(ki - ii)))
    pairs, leftover = fg._pair_stubs(pool, rng)
    edges.extend((a[0], b[0]) for a, b in pairs)

    H = nx.Graph()
    H.add_nodes_from(range(n_real))
    H.add_edges_from(edges)
    info = dict(n=n_real, realized_E=H.number_of_edges(),
                dropped_stubs=len(leftover), n_modules=len(sizes))
    if connect:
        H, rep = fg.connect_components(H, rng, partition=partition)   # module-aware repair
        info["connect"] = rep
    if clustering_fit is not None:
        # The same pass the blueprint sampler runs, driven by the fitted profile c(k)
        # instead of the blueprint's own triangle counts: LFR's absence of clustering is
        # the case c == 0.  It preserves every degree, every module's internal and
        # boundary link counts and connectivity, so the components repaired above stay
        # repaired.
        H, cinfo = fg.raise_clustering(H, partition, rng,
                                       targets=clustering_profile(clustering_fit),
                                       **(clustering_kwargs or {}))
        info["clustering"] = cinfo
    return H, partition, info


def flow_edge_weights(U, partition, H):
    """Flow-model weights for sample H's edges: w_uv proportional to
    p_u p_v R(A_u, A_v) -- the entries of the joint flow matrix P restricted to
    realized edges (O(E) with per-module-pair memoization) -- rescaled so the
    total weight matches the blueprint's."""
    tree = fg.ModuleTree(U, partition)
    paths = tree.paths
    memo = {}

    def R(A, B):
        if A == B:
            return 1.0 / (tree.flow[A] + tree.q[A])
        if (A, B) not in memo:
            memo[(A, B)] = fg.cross_R(tree, A, B) if tree.q[A] > 0 else 0.0
        return memo[(A, B)]

    W = sum(w for _, _, w in U.edges(data="weight", default=1.0))
    raw = {(u, v): tree.p[u] * tree.p[v] * R(paths[u], paths[v])
           for u, v in H.edges()}
    positive = [w for w in raw.values() if w > 0]
    floor = min(positive) / 10.0 if positive else 1.0
    raw = {e: max(w, floor) for e, w in raw.items()}
    total = sum(raw.values())
    return {e: w * W / total for e, w in raw.items()}
