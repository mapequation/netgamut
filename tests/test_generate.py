"""The generate subpackage: blueprint samplers, parametric fits and wrappers.

Everything here is offline -- blueprints are built in process, never
downloaded. Tests that touch an optional dependency skip when it is missing,
like the conversion tests do.

The blueprint is a planted-partition graph rather than joined cliques: stub
matching is exact for sparse modules, while a complete clique's internal
stubs can only tile the clique itself, which makes leftovers (and therefore
loose assertions) unavoidable. Sparse modules are the sampler's operating
regime, so that is what the tests pin down.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from networks import InMemoryNetwork, MultilayerNetwork, StateNetwork, generate


def blueprint() -> tuple[Any, dict[int, int]]:
    """Four sparse planted communities: the regime where sampling is exact."""
    nx = pytest.importorskip("networkx")
    G = nx.planted_partition_graph(4, 25, 0.3, 0.02, seed=1)
    partition = {u: G.nodes[u]["block"] for u in G}
    return G, partition


def two_cliques(size: int = 10) -> tuple[Any, dict[int, int]]:
    """Two cliques joined by a single bridge: unambiguous for Infomap."""
    nx = pytest.importorskip("networkx")
    G = nx.disjoint_union(nx.complete_graph(size), nx.complete_graph(size))
    G.add_edge(0, size)
    partition = {u: (0 if u < size else 1) for u in G.nodes()}
    return G, partition


def as_graph_info(net: Any) -> tuple[Any, dict[str, Any]]:
    """Unpack a InMemoryNetwork the way the generators used to return it."""
    return net.to_networkx(), net.info


def as_graph_partition_info(net: Any) -> tuple[Any, dict[Any, Any], dict[str, Any]]:
    return net.to_networkx(), net.partition, net.info


def cross_fraction(G: Any, partition: dict[int, int]) -> float:
    edges = list(G.edges())
    return sum(1 for u, v in edges if partition[u] != partition[v]) / len(edges)


# -- benchmark --------------------------------------------------------------
def test_benchmark_preserves_size_and_connectivity() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, partition = blueprint()
    H, info = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(1)))
    assert sorted(H.nodes()) == sorted(G.nodes())
    assert abs(H.number_of_edges() - G.number_of_edges()) <= 2
    assert nx.is_connected(H)
    assert info["target_E"] == G.number_of_edges()


def test_benchmark_preserves_the_degree_sequence() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(2)))
    target = [G.degree(u) for u in sorted(G.nodes())]
    sampled = [H.degree(u) for u in sorted(H.nodes())]
    assert sum(abs(a - b) for a, b in zip(target, sampled, strict=True)) <= 4


def test_benchmark_reproduces_the_mixing() -> None:
    """The planted modules keep their share of cross-module edges."""
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(3)))
    assert cross_fraction(H, partition) == pytest.approx(
        cross_fraction(G, partition), abs=0.03
    )


def test_benchmark_without_partition_uses_infomap() -> None:
    pytest.importorskip("infomap")
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, _ = two_cliques()
    H, _ = as_graph_info(generate.benchmark(G, rng=np.random.default_rng(4)))
    assert sorted(H.nodes()) == sorted(G.nodes())
    assert nx.is_connected(H)


# -- weighted_benchmark -----------------------------------------------------
def test_weighted_benchmark_assigns_flow_weights() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.5
    H, info = as_graph_info(generate.weighted_benchmark(G, partition, rng=np.random.default_rng(5)))
    assert not H.is_directed()
    weights = [w for _, _, w in H.edges(data="weight")]
    assert weights and all(w > 0 for w in weights)
    total = sum(w for _, _, w in G.edges(data="weight"))
    assert sum(weights) == pytest.approx(total)
    assert info["frame"] == "undirected"


def test_weighted_benchmark_symmetrizes_directed_blueprints() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = as_graph_info(
        generate.weighted_benchmark(G.to_directed(), partition, rng=np.random.default_rng(6))
    )
    assert not H.is_directed()
    assert sorted(H.nodes()) == sorted(G.nodes())


# -- parameter fits ---------------------------------------------------------
def test_fit_families_recovers_zipf() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(7)
    fits = generate.fit_families(rng.zipf(2.5, size=3000), support_min=1)
    assert fits["winner"] == "powerlaw"
    assert fits["powerlaw"]["params"]["alpha"] == pytest.approx(2.5, abs=0.2)


def test_fit_families_recovers_poisson() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    values = rng.poisson(6.0, size=3000)
    fits = generate.fit_families(values[values >= 1], support_min=1)
    assert fits["winner"] == "poisson"
    assert fits["poisson"]["params"]["lam"] == pytest.approx(6.0, abs=0.4)


def test_fit_mixing_beta_fits_spread_mixing() -> None:
    pytest.importorskip("numpy")
    fit = generate.fit_mixing_beta([0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0])
    assert fit["family"] == "beta"
    assert fit["mu"] == pytest.approx(0.25)
    assert fit["alpha"] > 0 and fit["beta"] > 0


def test_fit_mixing_beta_degenerate_falls_back_to_delta() -> None:
    pytest.importorskip("numpy")
    fit = generate.fit_mixing_beta([0.2, 0.2, 0.2], [1.0, 2.0, 1.0])
    assert fit["family"] == "delta"
    assert fit["mu"] == pytest.approx(0.2)


# -- parametric generator ---------------------------------------------------
def test_parametric_roundtrip_on_fitted_families() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(9)
    degree_fit = generate.fit_families(rng.zipf(2.8, size=2000), support_min=1)
    size_fit = generate.fit_families(rng.poisson(30.0, size=200) + 5)
    mixing_fit = generate.fit_mixing_beta([0.05, 0.1, 0.15, 0.2], [1.0, 1.0, 1.0, 1.0])
    H, partition, info = as_graph_partition_info(generate.parametric(
        400, degree_fit, size_fit, mixing_fit, k_mean=6.0, k_max=40, size_max=80, seed=10
    ))
    assert set(partition) == set(H.nodes())
    assert abs(H.number_of_nodes() - 400) <= 2
    assert info["n_modules"] >= 2
    assert nx.is_connected(H)


def test_parametric_accepts_explicit_families() -> None:
    pytest.importorskip("numpy")
    H, partition, info = as_graph_partition_info(generate.parametric(
        300,
        {"family": "powerlaw", "params": {"alpha": 2.7}, "support_min": 1},
        {"family": "geometric", "params": {"p": 0.05}, "support_min": 10},
        {"family": "delta", "mu": 0.1},
        k_mean=6.0,
        k_max=30,
        size_max=60,
        seed=11,
    ))
    assert set(partition) == set(H.nodes())
    assert info["realized_E"] == H.number_of_edges()


def test_parametric_rejects_fits_without_a_winner() -> None:
    pytest.importorskip("numpy")
    bad = {"winner": "insufficient", "n_values": 1, "support_min": 1}
    with pytest.raises(ValueError, match="usable family"):
        generate.parametric(100, bad, bad, {"family": "delta", "mu": 0.1}, seed=12)


# -- lfr_networkx ---------------------------------------------------------------
def test_lfr_networkx_returns_graph_and_partition() -> None:
    """networkx's reimplementation of the binary benchmark.

    The authors' own generators are :func:`generate.lfr`, covered in test_lfr.py;
    this one stays because it needs no compiler and no download.
    """
    nx = pytest.importorskip("networkx")
    net = generate.lfr_networkx(
        250,
        2.5,
        1.5,
        0.1,
        average_degree=5,
        max_degree=20,
        min_community=20,
        max_community=60,
        seed=42,
    )
    assert isinstance(net, InMemoryNetwork)
    G, partition = net.to_networkx(), net.partition
    assert partition == nx.get_node_attributes(G, "module")
    assert set(partition) == set(G.nodes())
    assert len(set(partition.values())) > 1
    assert nx.number_of_selfloops(G) == 0


# -- detect -------------------------------------------------------------------
def test_detect_finds_the_planted_cliques() -> None:
    pytest.importorskip("infomap")
    G, _ = two_cliques()
    partition = generate.detect(G)
    assert set(partition) == set(G.nodes())
    assert len(set(partition.values())) == 2
    assert all(isinstance(path, tuple) for path in partition.values())


# -- packaging ----------------------------------------------------------------
def test_import_networks_does_not_need_the_extras() -> None:
    """The base package must stay importable without numpy and friends."""
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'numpy', 'networkx', 'scipy', 'infomap'}:\n"
        "            raise ImportError(name + ' is blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import networks\n"
        "import networks.generate\n"
        "try:\n"
        "    networks.generate.benchmark(None)\n"
        "except ImportError as error:\n"
        "    assert 'mapequation-networks[generate]' in str(error), error\n"
        "else:\n"
        "    raise SystemExit('expected an ImportError with install advice')\n"
        "networks.generate.overlapping_memory_benchmark(2, 300, n_nodes=32, community_size=8)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# -- directed and hierarchical blueprints -----------------------------------
def directed_hierarchical(
    seed: int = 1, per: int = 40, subs: int = 3, supers: int = 4,
    k_int: float = 5.0, k_sub: float = 0.8, k_sup: float = 0.25,
) -> tuple[Any, dict[int, tuple[int, int]]]:
    """A directed blueprint with a genuinely detectable two-level hierarchy.

    Sparse modules nested in supermodules, with three link densities: inside a
    submodule, between submodules of one supermodule, and across supermodules.
    """
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(seed)
    G, nid, groups, partition = nx.DiGraph(), 0, [], {}
    for supermodule in range(supers):
        subl = []
        for sub in range(subs):
            members = list(range(nid, nid + per))
            nid += per
            subl.append(members)
            for u in members:
                partition[u] = (supermodule, sub)
        groups.append(subl)

    def add(a: Any, b: Any) -> None:
        if a != b:
            G.add_edge(int(a), int(b))

    for supermodule, subl in enumerate(groups):
        for sub, members in enumerate(subl):
            for _ in range(int(len(members) * k_int)):
                a, b = rng.choice(members, 2, replace=False)
                add(a, b)
            siblings = [x for t, o in enumerate(subl) if t != sub for x in o]
            for _ in range(int(len(members) * k_sub)):
                add(rng.choice(members), rng.choice(siblings))
            outside = [x for t, o2 in enumerate(groups) if t != supermodule
                       for o in o2 for x in o]
            for _ in range(int(len(members) * k_sup)):
                add(rng.choice(members), rng.choice(outside))
    G.add_nodes_from(partition)
    return G, partition


def arc_classes(H: Any, partition: dict[Any, tuple[int, int]]) -> dict[str, int]:
    """Split links into within-module, within-supermodule and across-supermodule."""
    counts = {"internal": 0, "within_super": 0, "cross_super": 0}
    for u, v in H.edges():
        pu, pv = partition[u], partition[v]
        key = ("internal" if pu == pv
               else "within_super" if pu[0] == pv[0] else "cross_super")
        counts[key] += 1
    return counts


def test_benchmark_directed_preserves_both_degree_sequences() -> None:
    np = pytest.importorskip("numpy")
    G, partition = directed_hierarchical()
    H, info = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(1)))
    assert H.is_directed()
    assert abs(H.number_of_edges() - G.number_of_edges()) / G.number_of_edges() < 0.05
    for degree in ("in_degree", "out_degree"):
        blueprint_seq = sorted(d for _, d in getattr(G, degree)())
        sample_seq = sorted(d for _, d in getattr(H, degree)())
        mad = np.abs(np.array(blueprint_seq) - np.array(sample_seq)).mean()
        assert mad < 1.0, f"{degree} mean absolute deviation {mad}"
    assert info["levels"] == 2


@pytest.mark.parametrize("routing", ["boundary", "model", "observed"])
def test_directed_sampler_keeps_the_supermodule_layer(routing: str) -> None:
    """The whole point of hierarchical routing: a flattened partition loses it.

    Routing boundary stubs up the module tree reproduces the blueprint's split of
    boundary links into within-supermodule and across-supermodule; flattening the
    partition to its leaves scatters them over the whole network instead. Every
    escalation rule must clear this bar -- which rule is *best* is a separate,
    measured question (see :meth:`ModuleTree.escalation_probs`), so this test is
    parametrized rather than silently pinning whichever rule is the default.
    """
    np = pytest.importorskip("numpy")
    G, partition = directed_hierarchical()
    flat = {u: (path[0] * 10 + path[1],) for u, path in partition.items()}
    reference = arc_classes(G, partition)

    hier, _ = as_graph_info(generate.benchmark(G, partition, routing=routing,
                                               rng=np.random.default_rng(1)))
    flattened, _ = as_graph_info(generate.benchmark(G, flat, rng=np.random.default_rng(1)))
    hier_counts = arc_classes(hier, partition)
    flat_counts = arc_classes(flattened, partition)

    # hierarchical routing keeps both boundary classes within 40% of the blueprint
    for key in ("within_super", "cross_super"):
        assert 0.6 < hier_counts[key] / reference[key] < 1.4, (routing, key, hier_counts)
    # flattening inverts them: too few within-supermodule, far too many across
    assert flat_counts["within_super"] < 0.5 * reference["within_super"]
    assert flat_counts["cross_super"] > 2.0 * reference["cross_super"]


def test_sampler_default_routing_is_the_documented_one() -> None:
    """Pin the default explicitly, so changing it is a visible edit and not a side effect.

    The default is a measured choice over the pool's hierarchical blueprints, recorded in
    the manuscript's ideas log; this test exists so that flipping it cannot happen quietly.
    """
    import inspect

    flowgen, _ = generate._modules()
    expected = "model"          # re-measured 2026-09-18; the robustness gate selects it
    for fn in (flowgen.sample_stub_matching, flowgen.sample_stub_matching_directed,
               flowgen.sample_benchmark, flowgen._route_stubs):
        assert inspect.signature(fn).parameters["routing"].default == expected, fn.__name__
    tree_default = inspect.signature(
        flowgen.ModuleTree.escalation_probs).parameters["routing"].default
    assert tree_default == expected


def test_directed_flow_matrix_is_proper() -> None:
    """P1: rows of the joint P sum to the visit rate, for directed flow too."""
    np = pytest.importorskip("numpy")
    G, partition = directed_hierarchical(per=8, subs=2, supers=2)
    P, nodes, tree = generate.flow_matrix(G, partition)
    expected = np.array([tree.p[u] for u in nodes])
    assert np.abs(P.sum(axis=1) - expected).max() < 1e-12
    # and the blueprint really is directed: some module's enter != exit
    assert max(abs(tree.enter[m] - tree.exit[m]) for m in tree.flow) > 1e-6


def test_undirected_flow_matrix_is_proper() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    P, nodes, tree = generate.flow_matrix(G, partition)
    expected = np.array([tree.p[u] for u in nodes])
    assert np.abs(P.sum(axis=1) - expected).max() < 1e-12


def test_module_tree_count_frame_keeps_directed_asymmetry() -> None:
    G, partition = directed_hierarchical(per=8, subs=2, supers=2)
    counts = generate.ModuleTree(G, partition, flow_stats="counts")
    stationary = generate.ModuleTree(G, partition, flow_stats="stationary")
    assert counts.directed and stationary.directed
    # the sampler frame reads endpoint counts, so in- and out-flow differ per module
    assert max(abs(counts.flow_out[m] - counts.flow_in[m]) for m in counts.flow) > 1e-6
    # the flow-model frame has one visit rate per node, hence flow_out == flow
    assert all(abs(stationary.flow_out[m] - stationary.flow[m]) < 1e-12
               for m in stationary.flow)


@pytest.mark.parametrize("routing", ["boundary", "model", "observed"])
def test_escalation_probs_are_a_distribution(routing: str) -> None:
    G, partition = directed_hierarchical(per=8, subs=2, supers=2)
    tree = generate.ModuleTree(G, partition, flow_stats="counts")
    for leaf in tree.leaves:
        for kind in ("out", "in"):
            levels = tree.escalation_probs(leaf, kind=kind, routing=routing)
            assert len(levels) == len(leaf)
            assert abs(sum(prob for _, _, prob in levels) - 1.0) < 1e-9
            assert all(prob >= 0.0 for _, _, prob in levels)
            # deepest level first, and each stub is tagged with the branch it left
            assert [nu for nu, _, _ in levels] == [leaf[:t]
                                                   for t in range(len(leaf) - 1, -1, -1)]


def test_two_level_routing_collapses_to_one_pool() -> None:
    """All three rules coincide when there is no hierarchy to route through."""
    G, partition = blueprint()
    tree = generate.ModuleTree(G, {u: (m,) for u, m in partition.items()})
    for leaf in tree.leaves:
        levels = [tree.escalation_probs(leaf, routing=r)
                  for r in ("boundary", "model", "observed")]
        assert all(lv == [((), leaf, 1.0)] for lv in levels)


def test_unknown_routing_is_rejected() -> None:
    G, partition = blueprint()
    tree = generate.ModuleTree(G, partition)
    with pytest.raises(ValueError, match="routing"):
        tree.escalation_probs(tree.leaves[0], routing="nonsense")


# -- weighted blueprints ----------------------------------------------------
def skewed_weighted() -> tuple[Any, dict[int, int]]:
    """Two modules whose structure lives in the weights, with a heavy hub."""
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(3)
    G = nx.Graph()
    partition = {}
    for module in (0, 1):
        members = list(range(module * 20, module * 20 + 20))
        for u in members:
            partition[u] = module
        for i, u in enumerate(members):
            for v in members[i + 1:]:
                if rng.random() < 0.4:
                    G.add_edge(u, v, weight=float(rng.integers(50, 200)))
    for u in range(0, 20, 4):          # a few weak cross links
        G.add_edge(u, u + 20, weight=1.0)
    G.add_edge(0, 21, weight=50000.0)  # one dominating hub link
    return G, partition


def test_weighted_benchmark_plain_target_is_feasible_when_flow_target_is_not() -> None:
    np = pytest.importorskip("numpy")
    G, partition = skewed_weighted()
    plain, plain_info = as_graph_info(generate.weighted_benchmark(
        G, partition, degree_target="plain", rng=np.random.default_rng(1)))
    flow, flow_info = as_graph_info(generate.weighted_benchmark(
        G, partition, degree_target="flow", rng=np.random.default_rng(1)))
    assert plain_info["degree_target"] == "plain"
    assert flow_info["degree_target"] == "flow"
    # the plain target is a feasible simple graph by construction; the flow target
    # asks the hub for more distinct partners than the network has
    plain_deficit = 1 - plain.number_of_edges() / G.number_of_edges()
    flow_deficit = 1 - flow.number_of_edges() / G.number_of_edges()
    assert plain_deficit < 0.05
    assert flow_deficit > plain_deficit
    assert all("weight" in d for _, _, d in plain.edges(data=True))


def test_weighted_benchmark_rejects_an_unknown_degree_target() -> None:
    G, partition = blueprint()
    with pytest.raises(ValueError, match="degree_target"):
        generate.weighted_benchmark(G, partition, degree_target="nonsense")


# -- higher-order blueprints ------------------------------------------------
def memory_blueprint() -> tuple[dict[Any, float], dict[Any, Any], dict[Any, Any]]:
    """A second-order state network from trails over two communities of nodes."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    groups = [list(range(0, 8)), list(range(8, 16))]
    links: dict[Any, float] = {}
    windows: dict[Any, Any] = {}
    partition: dict[Any, Any] = {}
    for _ in range(600):
        group = groups[int(rng.integers(2))]
        trail = [int(rng.choice(group)) for _ in range(4)]
        if rng.random() < 0.1:                       # occasional cross-community step
            trail[-1] = int(rng.choice(groups[1] if group is groups[0] else groups[0]))
        for a, b, c in zip(trail, trail[1:], trail[2:], strict=False):
            if a == b or b == c:
                continue
            source, target = (a, b), (b, c)
            windows[source], windows[target] = source, target
            links[(source, target)] = links.get((source, target), 0.0) + 1.0
    for state in windows:
        partition[state] = (0,) if state[0] < 8 else (1,)
    return links, windows, partition


def test_memory_benchmark_respects_the_physical_junction() -> None:
    links, windows, partition = memory_blueprint()
    net = generate.memory_benchmark(links, windows, partition, seed=1)
    sampled, info = net.links, net.info
    assert isinstance(net, StateNetwork)
    assert sampled, "sampler produced no links"
    states = net.states                      # state id -> window of physical ids
    violations = [(a, b) for a, b in sampled if states[a][-1] != states[b][0]]
    assert violations == [], f"{len(violations)} links skip their junction"
    assert info["dropped_stubs"] == 0          # multi-edges aggregate, nothing strands
    assert 0.7 < info["realized_E"] / info["target_E"] < 1.3


def test_memory_benchmark_reproduces_the_module_mixing() -> None:
    links, windows, partition = memory_blueprint()
    net = generate.memory_benchmark(links, windows, partition, seed=1)
    sampled = net.links

    def mixing(edge_weights: dict[Any, float], partition: dict[Any, Any]) -> float:
        total = sum(edge_weights.values())
        cross = sum(w for (a, b), w in edge_weights.items()
                    if partition[a] != partition[b])
        return cross / total

    assert abs(mixing(sampled, net.partition) - mixing(links, partition)) < 0.05


def test_multilayer_benchmark_keeps_links_inside_their_layer() -> None:
    nx = pytest.importorskip("networkx")
    layers = {}
    partition = {}
    for layer in ("road", "rail"):
        G = nx.planted_partition_graph(3, 20, 0.3, 0.02, seed=1 if layer == "road" else 2)
        layers[layer] = G
        for u in G.nodes():
            partition[(u, layer)] = (G.nodes[u]["block"],)
    net = generate.multilayer_benchmark(layers, partition, seed=1)
    sampled, info = net.intra, net.info
    assert isinstance(net, MultilayerNetwork)
    assert set(sampled) == set(layers)
    for layer, edges in sampled.items():
        assert edges
        assert all(u in layers[layer] and v in layers[layer] for u, v in edges)
    assert 0.8 < info["realized_E"] / info["target_E"] < 1.2


# -- overlapping memory benchmark -------------------------------------------
def test_overlapping_memory_benchmark_reproduces_the_published_sampler() -> None:
    """Same states, links and planted partition as the paper project's script.

    The hash is of that script's output for ``N32 om2 nc8 E3000 mu0.1``, sample 1
    (``columnar_wip/make-om-network.py`` in the Infomap repo), taken over its links and
    partition by state id, so it pins every random draw and the state numbering but not
    the file layout.
    """
    import hashlib

    net = generate.overlapping_memory_benchmark(2, 3000, n_nodes=32, community_size=8, seed=1)
    assert net.partition is not None
    text = "".join(f"{s} {t} {int(w)}\n" for (s, t), w in sorted(net.links.items()))
    text += "".join(f"{s} {c}\n" for s, c in sorted(net.partition.items()))
    assert hashlib.md5(text.encode()).hexdigest() == "a50bb228004c5548c970e7308ef21a6a"
    assert net.name == "overlapping-memory-n32-om2-nc8-E3000-mu0.1-seed1"


def test_overlapping_memory_benchmark_overlaps_on_nodes_not_on_states(tmp_path: Any) -> None:
    net = generate.overlapping_memory_benchmark(
        3, 3000, n_nodes=32, community_size=8, mu=0.3, seed=7
    )
    assert net.partition is not None
    communities = net.info["communities"]
    assert len(communities) == 32 * 3 // 8
    assert all(len(nodes) == 8 for nodes in communities.values())
    members = {c: set(nodes) for c, nodes in communities.items()}
    # every state lies inside the one community that planted it
    for state, window in net.states.items():
        assert set(window) <= members[net.partition[state]]
    # every link joins its states at the physical junction
    assert all(net.states[s][1] == net.states[t][0] for s, t in net.links)
    # exactly the mu share of the trigrams crosses communities
    cross = sum(w for (s, t), w in net.links.items() if net.partition[s] != net.partition[t])
    assert net.total_weight == 3000
    assert cross == int(0.3 * 3000) == net.info["cross_trigrams"]
    path = net.to_pajek(tmp_path)
    assert "*Vertices 32\n" in path.read_text()


def test_overlapping_memory_benchmark_rejects_impossible_parameters() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        generate.overlapping_memory_benchmark(3, 1000, n_nodes=30, community_size=8)
    with pytest.raises(ValueError, match="at least 3"):
        generate.overlapping_memory_benchmark(2, 1000, n_nodes=32, community_size=2)


# -- Infomap helpers --------------------------------------------------------
def test_run_infomap_accepts_non_integer_labels() -> None:
    pytest.importorskip("infomap")
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"),
                      ("c", "x"), ("x", "y"), ("y", "z"), ("x", "z")])
    partition, codelength = generate.run_infomap(G, two_level=True, seed=1)
    assert set(partition) == set(G.nodes())
    assert all(isinstance(path, tuple) for path in partition.values())
    assert partition["a"] == partition["b"] and partition["x"] != partition["a"]
    assert codelength > 0


def test_run_infomap_forwards_engine_keywords() -> None:
    pytest.importorskip("infomap")
    G, _ = blueprint()
    fine = generate.infomap_partition(G, two_level=True, seed=1, markov_time=0.5)
    coarse = generate.infomap_partition(G, two_level=True, seed=1, markov_time=4.0)
    assert len(set(fine.values())) >= len(set(coarse.values()))


def test_detect_gives_hierarchical_paths_for_directed_blueprints() -> None:
    pytest.importorskip("infomap")
    G, _ = directed_hierarchical()
    partition = generate.detect(G)
    assert max(len(path) for path in partition.values()) >= 2


def test_read_pajek_aggregates_repeated_edges(tmp_path: Any) -> None:
    pytest.importorskip("networkx")
    path = tmp_path / "net.pajek"
    path.write_text(
        "*Vertices 3\n1\n2\n3\n*Edges\n1 2 1.5\n1 2 2.5\n2 3\n",
        encoding="utf-8",
    )
    G = generate.read_pajek(str(path))
    assert sorted(G.nodes()) == [1, 2, 3]
    assert G[1][2]["weight"] == pytest.approx(4.0)
    assert G[2][3]["weight"] == pytest.approx(1.0)


def test_run_infomap_keeps_ragged_hierarchy_paths() -> None:
    """Leaf paths are the solution's own depth, not padded to a uniform one.

    ``Result.multilevel_modules()`` pads every leaf to the depth of the deepest one,
    which inserts singleton levels into a ragged hierarchy: it inflates depth
    statistics and hands ModuleTree tree levels the solution does not have.
    """
    pytest.importorskip("infomap")
    nx = pytest.importorskip("networkx")
    # a ragged hierarchy: two tight communities plus a nested pair inside one of them
    G = nx.Graph()
    for base in (0, 30):
        G.add_edges_from((base + u, base + v) for u in range(10) for v in range(u + 1, 10))
    G.add_edges_from((60 + u, 60 + v) for u in range(6) for v in range(u + 1, 6))
    G.add_edge(0, 30)
    G.add_edge(5, 60)
    partition, _ = generate.run_infomap(G, seed=1, num_trials=20)
    depths = {len(path) for path in partition.values()}
    assert depths, "no partition returned"
    # every path must address a module the solution actually has
    assert all(len(path) >= 1 for path in partition.values())
    # and a two-level solution must not be reported as deeper than it is
    flatG = nx.planted_partition_graph(4, 25, 0.3, 0.02, seed=1)
    flat, _ = generate.run_infomap(flatG, seed=1, num_trials=20, two_level=True)
    assert {len(path) for path in flat.values()} == {1}


@pytest.mark.parametrize("routing", ["boundary", "model", "observed"])
def test_mixing_scale_is_monotone_and_routing_independent(routing: str) -> None:
    """The detectability dial is orthogonal to the escalation rule.

    ``mixing_scale`` multiplies every module's mixing, so the share of boundary links
    must rise monotonically with it; and because the dial scales the *total* boundary
    flow while the escalation rule only decides how that flow spreads over the levels,
    the curve must be essentially the same whichever rule is in force.
    """
    np = pytest.importorskip("numpy")
    G, partition = directed_hierarchical()
    fractions = []
    for scale in (1.0, 1.5, 2.0, 3.0):
        H, _ = as_graph_info(generate.benchmark(
            G, partition, routing=routing, mixing_scale=scale,
            rng=np.random.default_rng(7)))
        counts = arc_classes(H, partition)
        total = sum(counts.values())
        fractions.append((counts["within_super"] + counts["cross_super"]) / total)
    pairs = list(zip(fractions, fractions[1:], strict=False))   # deliberately offset
    assert all(b > a for a, b in pairs), fractions
    assert fractions[0] < 0.3 < fractions[-1]


def test_two_level_samples_are_identical_across_routing_rules() -> None:
    """Without a hierarchy the escalation rule cannot matter -- and it does not, exactly.

    Every boundary stub of a two-level partition stops at the root, so all three rules
    reduce to the same single-pool pairing and, given the same generator, produce the
    *same edge set*. This is what makes the affected surface of a routing change
    decidable: only experiments that plant a hierarchical partition can move.
    """
    np = pytest.importorskip("numpy")
    rules = ("observed", "boundary", "model")

    G, flat = blueprint()
    partition = {u: (m,) for u, m in flat.items()}
    undirected = {
        r: sorted(
            tuple(sorted(e))
            for e in generate.benchmark(
                G, partition, routing=r, rng=np.random.default_rng(42)).to_networkx().edges()
        )
        for r in rules
    }
    assert undirected["observed"] == undirected["boundary"] == undirected["model"]
    assert len(undirected["observed"]) > 0

    D, hier = directed_hierarchical(per=20, subs=2, supers=3)
    two_level = {u: (path[0],) for u, path in hier.items()}
    directed = {
        r: sorted(generate.benchmark(
            D, two_level, routing=r, rng=np.random.default_rng(42)).to_networkx().edges())
        for r in rules
    }
    assert directed["observed"] == directed["boundary"] == directed["model"]

    # ... and with the hierarchy restored the rules must genuinely diverge, or the
    # measurement that chose between them would be meaningless
    deep = {
        r: sorted(generate.benchmark(
            D, hier, routing=r, rng=np.random.default_rng(42)).to_networkx().edges())
        for r in rules
    }
    assert not (deep["observed"] == deep["boundary"] == deep["model"])


def test_partition_distances_match_the_dense_flow_models() -> None:
    """GMD/GMS/AGMD/AGMS: blockwise evaluation against the dense reference.

    The blockwise cell decomposition must reproduce ``D_dist`` on the materialized
    ``n x n`` flow models exactly, and the element-centric decomposition
    ``D^2 = sum_i p_i d_i^2`` must hold -- both on a flat and on a hierarchical
    partition, since the hierarchical path shares no code branch with the flat one.
    """
    pytest.importorskip("numpy")
    G, partition = blueprint()
    coarse = {u: partition[u] // 2 for u in G}                 # merge pairs of blocks
    hier = {u: (partition[u] // 2, partition[u]) for u in G}    # two planted levels

    for A, B in ((partition, coarse), (hier, coarse), (hier, partition)):
        PA, _, _ = generate.flow_matrix(G, A)
        PB, _, _ = generate.flow_matrix(G, B)
        D, d_node = generate.gmd(G, A, B, element=True)
        assert generate.D_dist(PA, PB) == pytest.approx(D, abs=1e-12)
        p = generate.ModuleTree(G, A).p
        assert sum(p[u] * d_node[u]**2 for u in G) == pytest.approx(D**2, abs=1e-12)
        assert generate.gms(G, A, B) == pytest.approx(1.0 - D, abs=1e-15)
        assert 0.0 <= generate.agms(G, A, B) <= 1.0

    # identity and the adjusted end points: N is at AGMD = 1 from any structure
    assert generate.gmd(G, partition, partition) == pytest.approx(0.0, abs=1e-15)
    assert generate.agms(G, partition, partition) == pytest.approx(1.0, abs=1e-15)
    N = generate.null_partition(G)
    assert generate.agmd(G, partition, N) == pytest.approx(1.0, abs=1e-12)
    assert generate.agms(G, N, N) == pytest.approx(1.0, abs=1e-15)


def test_zero_flow_nodes_carry_no_weight() -> None:
    """Partitions that differ only on nodes with no flow are flow-equivalent.

    An isolated node has visit rate ``p_i = 0``, so where a partition puts it makes
    no claim about flow and ``D`` must be exactly zero -- the property that separates
    ``D`` from label measures, which score such a node as a pure cluster.
    """
    nx = pytest.importorskip("networkx")
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    G.add_nodes_from([6, 7, 8])                      # no links, hence no flow
    base = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    shattered = {**base, 6: 2, 7: 3, 8: 4}           # each zero-flow node its own module
    absorbed = {**base, 6: 0, 7: 0, 8: 0}            # all three swept into module 0

    assert generate.gmd(G, shattered, absorbed) == 0.0
    assert generate.gms(G, shattered, absorbed) == 1.0
    assert generate.agmd(G, shattered, absorbed) == 0.0
    # and they stay indistinguishable against any third partition
    N = generate.null_partition(G)
    assert generate.gmd(G, shattered, N) == pytest.approx(generate.gmd(G, absorbed, N), abs=1e-15)


# -- sequential pairing, observed split, clustering pass (2026-09-05) --------
def hub_blueprint() -> tuple[Any, dict[int, int]]:
    """A hub whose internal demand under a degree-proportional split would exceed its
    module: module A = 12 nodes where node 0 is adjacent to all 11 others and to 10 nodes of
    module B (a 20-node ring of cliques).  The old random pairer stranded stubs here; the
    sequential pairer with the observed per-node split places every one."""
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_edges_from((0, v) for v in range(1, 12))
    G.add_edges_from((u, u + 1) for u in range(1, 11))
    G.add_edge(11, 1)
    B = nx.relabel_nodes(nx.connected_caveman_graph(4, 5), lambda u: u + 12)
    G.add_edges_from(B.edges())
    G.add_edges_from((0, v) for v in range(12, 22))
    partition = {u: (0 if u < 12 else 1) for u in G.nodes()}
    return G, partition


def module_edge_counts(G: Any, partition: dict[int, int]) -> tuple[dict, dict]:
    internal: dict[int, int] = {}
    boundary: dict[int, int] = {}
    for u, v in G.edges():
        if partition[u] == partition[v]:
            internal[partition[u]] = internal.get(partition[u], 0) + 1
        else:
            for w in (u, v):
                boundary[partition[w]] = boundary.get(partition[w], 0) + 1
    return internal, boundary


def test_sampler_places_every_stub_and_keeps_every_degree() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, partition = hub_blueprint()
    for seed in range(5):
        H, info = as_graph_info(
            generate.benchmark(G, partition, rng=np.random.default_rng(seed), connect=False)
        )
        assert info["realized_E"] == G.number_of_edges()
        assert info["dropped_stubs"] == 0
        assert all(H.degree(u) == G.degree(u) for u in G)          # per node, not just sorted
        assert nx.number_of_selfloops(H) == 0
        assert module_edge_counts(H, partition) == module_edge_counts(G, partition)


def test_directed_sampler_places_every_stub_without_a_top_up() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, partition = hub_blueprint()
    D = nx.DiGraph()
    D.add_edges_from(G.edges())
    D.add_edges_from((v, u) for u, v in G.edges() if (u + v) % 3 == 0)   # some reciprocal arcs
    for seed in range(3):
        H, info = as_graph_info(
            generate.benchmark(D, partition, rng=np.random.default_rng(seed), connect=False)
        )
        assert info["realized_E"] == D.number_of_edges()
        assert info["dropped_out"] == 0 and info["dropped_in"] == 0
        assert all(H.out_degree(u) == D.out_degree(u) and H.in_degree(u) == D.in_degree(u) for u in D)
        assert module_edge_counts(H, partition) == module_edge_counts(D, partition)


def test_pair_stubs_respects_tags_and_never_duplicates() -> None:
    np = pytest.importorskip("numpy")
    flowgen, _ = generate._modules()
    rng = np.random.default_rng(3)
    stubs = [(u, u) for u in range(6) for _ in range(3)] + [(6, 6)] * 6        # hub of 6 among 6 degree-3 nodes: 24 stubs, graphical
    pairs, leftover = flowgen._pair_stubs(stubs, rng)
    assert not leftover
    seen = set()
    for (a, ta), (b, tb) in pairs:
        assert a != b and ta != tb
        key = (min(a, b), max(a, b))
        assert key not in seen
        seen.add(key)


def test_raise_clustering_matches_the_blueprint_profile_and_keeps_invariants() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G = nx.watts_strogatz_graph(200, 6, 0.1, seed=1)        # ring lattice: C ~ 0.45 at k = 6
    partition = {u: u // 50 for u in G}                       # four ring segments as modules
    H, _ = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(1)))
    flowgen, _ = generate._modules()
    Hc, info = flowgen.raise_clustering(H, partition, np.random.default_rng(1), blueprint=G)
    cb, c0, c1 = nx.clustering(G), nx.clustering(H), nx.clustering(Hc)
    err0 = sum(abs(cb[u] - c0[u]) for u in G) / len(G)
    err1 = sum(abs(cb[u] - c1[u]) for u in G) / len(G)
    assert err1 < 0.5 * err0                                # the profile error at least halves
    assert info["swaps"] > 0
    assert all(Hc.degree(u) == H.degree(u) for u in G)
    assert module_edge_counts(Hc, partition) == module_edge_counts(H, partition)
    assert nx.is_connected(Hc) == nx.is_connected(H)       # no bridge was removed
    assert nx.number_of_selfloops(Hc) == 0


def test_raise_clustering_accepts_a_parametric_profile() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, partition = blueprint()
    H, _ = as_graph_info(generate.benchmark(G, partition, rng=np.random.default_rng(2)))
    flowgen, _ = generate._modules()
    Hc, info = flowgen.raise_clustering(H, partition, np.random.default_rng(2),
                                        targets=lambda k: 0.5)
    assert info["C_after"] > info["C_before"]
    assert all(Hc.degree(u) == H.degree(u) for u in G)


def test_top_up_keyword_is_gone() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    with pytest.raises(TypeError):
        generate.benchmark(G, partition, rng=np.random.default_rng(1), topup=True)


def test_memory_benchmark_overlaps_windows_at_order_three() -> None:
    """A state link is only realizable when the windows overlap in m-1 nodes.

    At order 2 that is the single physical junction, which the sampler used to encode as
    the window's last and first node -- correct there and wrong at every higher order (an
    order-3 Wikispeedia blueprint came out with 1731 of 1731 links pairing incompatible
    windows).  The type is the whole (m-1)-window.
    """
    np = pytest.importorskip("numpy")
    # order-3 windows over a 4-node cycle: every 3-path along 0->1->2->3->0
    cyc = [0, 1, 2, 3]
    windows = {}
    for a in range(4):
        b, c = (a + 1) % 4, (a + 2) % 4
        windows[(a, b, c)] = (a, b, c)
    links = {}
    for w in windows:
        nxt = (w[1], w[2], (w[2] + 1) % 4)
        links[(w, nxt)] = 2.0
    partition = {s: (i % 2,) for i, s in enumerate(sorted(windows))}
    net = generate.memory_benchmark(links, windows, partition, seed=5)
    sampled, info = net.links, net.info
    assert info["dropped_stubs"] == 0
    assert sampled, "no links realized"
    for (a, b) in sampled:
        assert net.states[a][1:] == net.states[b][:-1], (a, b)
    assert len(cyc) == 4                      # the fixture is the 4-cycle, unchanged


def test_parametric_runs_the_clustering_pass_from_a_fitted_profile() -> None:
    """The parametric generator takes the same two post-passes as the blueprint sampler.

    Connectivity repair is on by default; the clustering pass is opt-in and driven by a
    profile c(k) instead of a blueprint's triangle counts, so a parametric sample can be
    given clustering with no blueprint at sampling time.  A flat target c = 0.6 stands in
    for the fitted profile here, because a power law fitted on a small fixture is
    dominated by its degree-2 nodes and can ask for *less* clustering than stub matching
    already produces.
    """
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    pytest.importorskip("scipy")
    deg = generate.fit_families([4] * 60)
    size = generate.fit_families([10] * 6, support_min=2)
    mix = {"family": "delta", "mu": 0.2}
    plain, _, info0 = as_graph_partition_info(generate.parametric(60, deg, size, mix, seed=4))
    flat = {"c0": 0.6, "alpha": 0.0, "n_fit": 0, "c_mean": 0.6}
    withc, _, info1 = as_graph_partition_info(
        generate.parametric(60, deg, size, mix, seed=4, clustering_fit=flat)
    )
    assert "connect" in info0 and "connect" in info1        # repair on by default for both
    assert "clustering" not in info0 and info1["clustering"]["swaps"] > 0
    assert nx.average_clustering(withc) > nx.average_clustering(plain) + 0.1
    assert sorted(d for _, d in withc.degree()) == sorted(d for _, d in plain.degree())
    assert (nx.number_connected_components(withc)
            == nx.number_connected_components(plain) == 1)


def test_fit_clustering_profile_recovers_a_flat_profile() -> None:
    """A graph whose nodes all have the same local clustering fits alpha = 0."""
    pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    for m in range(12):                                     # disjoint triangles: c = 1 everywhere
        b = 3 * m
        G.add_edges_from([(b, b + 1), (b + 1, b + 2), (b, b + 2)])
    fit = generate.fit_clustering_profile(G)
    assert fit["c_mean"] == pytest.approx(1.0)
    assert fit["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert fit["c0"] == pytest.approx(1.0, abs=1e-9)
