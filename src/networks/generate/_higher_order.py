"""Generative benchmarks for higher-order networks: memory and multilayer.

The flow model needs no change for higher-order flows.  Multilayer and memory networks
*are* state networks -- one physical node represented by several state nodes -- and the
map equation already operates on their modular flows, so ``L*`` and the non-redundant
construction apply as they stand: they touch only the module-level enter and exit
codebooks.  What changes is the **sampler**, because a state link is not valid between
arbitrary states.

Both cases are the same constraint, *typed stub matching*: every state carries an
out-type and an in-type, and a link may only join an out-stub to an in-stub of the same
type.  Pairing therefore happens inside each type class -- internally per module first,
then across modules under the non-redundant rule, then a relaxed top-up.

    second-order memory   out-type = the state's current node, in-type = its remembered
                          predecessor, so a link (i,j) -> (j,k) matches at the physical
                          junction j
    multiplex             out-type = in-type = the state's layer, so intra-layer links
                          pair within their layer

Because every out-link of a state passes through its own type class, link counts are
apportioned class by class, which balances each class's stub supply against its demand
exactly instead of leaving the balance to global rounding.  And because multi-edges
aggregate to link weights -- weights and multiplicity are the same thing to the map
equation -- hub stubs never strand on a simple-graph constraint: every stub is realized.

Verified against the two pilots behind the manuscript's higher-order section: the
Wikispeedia second-order memory network (23,872 states over 1,000 articles) and the
London transport multiplex (3 layers, 399 state nodes).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import _flowgen as fg


def _pair_multi(out_stubs, in_stubs, rng, forbid_same_group=False, max_sweeps=200):
    """Pair out- to in-stubs allowing DUPLICATE links, which aggregate to weights.

    Multi-edges and weights are the same thing to Infomap, so hub stubs never strand on
    a distinct-pair constraint.  Self-loops and (optionally) same-module pairs -- the
    non-redundant boundary rule -- are still repaired by swap sweeps.

    Returns ``(pairs, leftover_out, leftover_in)``.
    """
    out_stubs, in_stubs = list(out_stubs), list(in_stubs)
    rng.shuffle(out_stubs)
    rng.shuffle(in_stubs)
    n = min(len(out_stubs), len(in_stubs))
    leftover_out, leftover_in = out_stubs[n:], in_stubs[n:]
    out_stubs, in_stubs = out_stubs[:n], in_stubs[:n]

    def bad_at(i):
        o, ii = out_stubs[i], in_stubs[i]
        return o[0] == ii[0] or (forbid_same_group and o[1] == ii[1])

    for _ in range(max_sweeps):
        bad = [i for i in range(n) if bad_at(i)]
        if not bad:
            break
        for i in bad:
            j = int(rng.integers(n))
            in_stubs[i], in_stubs[j] = in_stubs[j], in_stubs[i]

    pairs = []
    for i in range(n):
        if bad_at(i):
            leftover_out.append(out_stubs[i])
            leftover_in.append(in_stubs[i])
        else:
            pairs.append((out_stubs[i][0], in_stubs[i][0]))
    return pairs, leftover_out, leftover_in


def _capped_apportion(total, weights, cap, rng, passes=6):
    """Apportion *total* by *weights* with a per-item cap, redistributing the excess.

    A state cannot link to more distinct partners than its type class offers, so the
    uncapped apportionment would demand more links through heavy classes than they have
    distinct state pairs.
    """
    k = fg.apportion(total, weights, rng).astype(int)
    for _ in range(passes):
        excess = k - cap
        over = int(excess[excess > 0].sum())
        if over == 0:
            break
        k = np.minimum(k, cap)
        headroom = cap - k
        if headroom.sum() == 0:
            break
        add = fg.apportion(min(over, int(headroom.sum())), headroom.astype(float), rng)
        k = np.minimum(k + add, cap)
    return np.minimum(k, cap)


def sample_state_links(links, partition, out_type, in_type, rng=None, seed=None,
                       topup=True):
    """Typed microcanonical sampler for a state network.

    Parameters
    ----------
    links : dict
        ``{(state, state): weight}`` -- the blueprint's state-level links.
    partition : dict
        ``{state: module id or path tuple}``; the planted ground truth.
    out_type, in_type : dict or callable
        The type of a state's out-stubs and in-stubs.  A link may only join an out-stub
        to an in-stub of the same type (the junction for memory, the layer for a
        multiplex).
    rng, seed : numpy Generator or int
        ``rng`` takes precedence; ``seed`` builds one.
    topup : bool
        After the non-redundant boundary pass, place what it could not with the
        same-module rule relaxed.  The type constraint is hard; the module rule is not.

    Returns
    -------
    (sampled, info) : ``{(state, state): weight}`` and a dict of diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    ot = out_type if callable(out_type) else out_type.__getitem__
    it = in_type if callable(in_type) else in_type.__getitem__

    states = sorted({s for e in links for s in e} | set(partition))
    idx = {s: i for i, s in enumerate(states)}
    wout, win = np.zeros(len(states)), np.zeros(len(states))
    for (a, b), w in links.items():
        wout[idx[a]] += w
        win[idx[b]] += w

    # 1. link counts per type class -- the blueprint's own, feasible by construction
    cls_count = defaultdict(int)
    for a, _b in links:
        cls_count[ot(a)] += 1
    out_states, in_states = defaultdict(list), defaultdict(list)
    for s in states:
        out_states[ot(s)].append(s)
        in_states[it(s)].append(s)

    kout = np.zeros(len(states), dtype=int)
    kin = np.zeros(len(states), dtype=int)
    for cls in sorted(cls_count, key=repr):
        e_c = cls_count[cls]
        so, si = out_states.get(cls, []), in_states.get(cls, [])
        if not so or not si or e_c == 0:
            continue
        # distinct-partner capacity, minus the state itself when it sits on both sides
        cap_o = np.array([len(si) - int(it(s) == cls) for s in so])
        cap_i = np.array([len(so) - int(ot(s) == cls) for s in si])
        for s, k in zip(so, _capped_apportion(
                e_c, np.array([wout[idx[s]] for s in so]), cap_o, rng), strict=True):
            kout[idx[s]] = k
        for s, k in zip(si, _capped_apportion(
                e_c, np.array([win[idx[s]] for s in si]), cap_i, rng), strict=True):
            kin[idx[s]] = k

    # 2. per-module directed mixing on the weighted state graph
    comm = fg.as_paths(partition)
    within, cout, cin = defaultdict(float), defaultdict(float), defaultdict(float)
    for (a, b), w in links.items():
        if comm[a] == comm[b]:
            within[comm[a]] += w
        else:
            cout[comm[a]] += w
            cin[comm[b]] += w

    # 3. split each state's stubs into internal and boundary by its module's mixing
    o_int, i_int = defaultdict(list), defaultdict(list)   # keyed (type, module)
    o_bnd, i_bnd = defaultdict(list), defaultdict(list)   # keyed type
    for s in states:
        A = comm[s]
        to, ti = within[A] + cout[A], within[A] + cin[A]
        mo = cout[A] / to if to > 0 else 0.0
        mi = cin[A] / ti if ti > 0 else 0.0
        ko, ki = int(kout[idx[s]]), int(kin[idx[s]])
        bo, bi = min(int(round(ko * mo)), ko), min(int(round(ki * mi)), ki)
        o_int[(ot(s), A)].extend([(s, A)] * (ko - bo))
        i_int[(it(s), A)].extend([(s, A)] * (ki - bi))
        o_bnd[ot(s)].extend([(s, A)] * bo)
        i_bnd[it(s)].extend([(s, A)] * bi)

    # 4. internal pass per (type, module); leftovers escalate to the type's boundary pool
    pairs, dropped = [], 0
    for key in sorted(set(o_int) | set(i_int), key=repr):
        e, lo, li = _pair_multi(o_int.get(key, []), i_int.get(key, []), rng)
        pairs.extend(e)
        o_bnd[key[0]].extend(lo)
        i_bnd[key[0]].extend(li)

    # 5. boundary pass across modules within each type class, then a relaxed top-up
    for cls in sorted(set(o_bnd) | set(i_bnd), key=repr):
        o, i_ = o_bnd.get(cls, []), i_bnd.get(cls, [])
        if not o or not i_:
            dropped += len(o) + len(i_)
            continue
        e, lo, li = _pair_multi(o, i_, rng, forbid_same_group=True)
        pairs.extend(e)
        if topup and lo and li:
            e2, lo, li = _pair_multi(lo, li, rng)
            pairs.extend(e2)
        dropped += len(lo) + len(li)

    sampled = defaultdict(float)
    for e in pairs:
        sampled[e] += 1.0
    return dict(sampled), {"target_E": len(links), "realized_E": len(sampled),
                           "realized_stubs": len(pairs), "dropped_stubs": dropped,
                           "n_states": len(states), "n_type_classes": len(cls_count)}


def memory_benchmark(links, windows, partition, rng=None, seed=None, topup=True):
    """Sample a memory-network benchmark from a state-level blueprint.

    A second-order state ``s`` stands for a path window ``(i, j)``: its out-links leave
    the current node ``windows[s][-1]`` and its in-links arrive at the remembered
    predecessor ``windows[s][0]``, so a link ``(i,j) -> (j,k)`` matches at the physical
    junction ``j``.  Higher orders work unchanged -- only the two ends of the window are
    used.

    Args:
        links: ``{(state, state): weight}`` blueprint state links.
        windows: ``{state: tuple of physical nodes}``, oldest first.
        partition: ``{state: module}`` planted ground truth.
        rng, seed: ``numpy`` Generator, or a seed for a fresh one.
        topup: Place stubs the non-redundant rule could not; see
            :func:`sample_state_links`.

    Returns:
        ``(sampled, info)`` -- ``{(state, state): weight}`` and diagnostics.
    """
    return sample_state_links(
        links, partition,
        out_type={s: w[-1] for s, w in windows.items()},
        in_type={s: w[0] for s, w in windows.items()},
        rng=rng, seed=seed, topup=topup)


def multilayer_benchmark(layers, partition, rng=None, seed=None, **kwargs):
    """Sample a multiplex benchmark layer by layer from a state-level blueprint.

    Intra-layer links pair within their layer, so each layer is resampled on its own
    with the state partition restricted to it -- typed stub matching with the layer as
    the type.  Inter-layer coupling is not sampled: in a multiplex it is supplied by the
    relax rate at analysis time, not by links in the data.

    Args:
        layers: ``{layer: networkx graph}`` or ``{layer: {(u, v): weight}}``.
        partition: ``{(node, layer): module}`` planted ground truth, as Infomap reports
            a multilayer partition.
        rng, seed: ``numpy`` Generator, or a seed for a fresh one.
        **kwargs: Passed to :func:`~networks.generate._flowgen.sample_benchmark`.

    Returns:
        ``(sampled, info)`` -- ``{layer: {(u, v): weight}}`` and diagnostics.
    """
    import networkx as nx

    if rng is None:
        rng = np.random.default_rng(seed)
    sampled, info = {}, {"layers": {}}
    for layer in sorted(layers, key=repr):
        spec = layers[layer]
        if isinstance(spec, dict):
            G = nx.Graph()
            for (u, v), w in spec.items():
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
        else:
            G = spec
        part_l = {u: partition.get((u, layer), (0,)) for u in G.nodes()}
        H, li = fg.sample_benchmark(G, partition=part_l, rng=rng, **kwargs)
        sampled[layer] = {(u, v): float(d.get("weight", 1.0))
                          for u, v, d in H.edges(data=True)}
        info["layers"][layer] = li
    info["realized_E"] = sum(len(e) for e in sampled.values())
    info["target_E"] = sum(
        (len(s) if isinstance(s, dict) else s.number_of_edges()) for s in layers.values())
    return sampled, info
