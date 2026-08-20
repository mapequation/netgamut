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

from networks import generate


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


def cross_fraction(G: Any, partition: dict[int, int]) -> float:
    edges = list(G.edges())
    return sum(1 for u, v in edges if partition[u] != partition[v]) / len(edges)


# -- benchmark --------------------------------------------------------------
def test_benchmark_preserves_size_and_connectivity() -> None:
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, partition = blueprint()
    H, info = generate.benchmark(G, partition, rng=np.random.default_rng(1))
    assert sorted(H.nodes()) == sorted(G.nodes())
    assert abs(H.number_of_edges() - G.number_of_edges()) <= 2
    assert nx.is_connected(H)
    assert info["target_E"] == G.number_of_edges()


def test_benchmark_preserves_the_degree_sequence() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = generate.benchmark(G, partition, rng=np.random.default_rng(2))
    target = [G.degree(u) for u in sorted(G.nodes())]
    sampled = [H.degree(u) for u in sorted(H.nodes())]
    assert sum(abs(a - b) for a, b in zip(target, sampled, strict=True)) <= 4


def test_benchmark_reproduces_the_mixing() -> None:
    """The planted modules keep their share of cross-module edges."""
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = generate.benchmark(G, partition, rng=np.random.default_rng(3))
    assert cross_fraction(H, partition) == pytest.approx(
        cross_fraction(G, partition), abs=0.03
    )


def test_benchmark_without_partition_uses_infomap() -> None:
    pytest.importorskip("infomap")
    np = pytest.importorskip("numpy")
    nx = pytest.importorskip("networkx")
    G, _ = two_cliques()
    H, _ = generate.benchmark(G, rng=np.random.default_rng(4))
    assert sorted(H.nodes()) == sorted(G.nodes())
    assert nx.is_connected(H)


# -- weighted_benchmark -----------------------------------------------------
def test_weighted_benchmark_assigns_flow_weights() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.5
    H, info = generate.weighted_benchmark(G, partition, rng=np.random.default_rng(5))
    assert not H.is_directed()
    weights = [w for _, _, w in H.edges(data="weight")]
    assert weights and all(w > 0 for w in weights)
    total = sum(w for _, _, w in G.edges(data="weight"))
    assert sum(weights) == pytest.approx(total)
    assert info["frame"] == "undirected"


def test_weighted_benchmark_symmetrizes_directed_blueprints() -> None:
    np = pytest.importorskip("numpy")
    G, partition = blueprint()
    H, _ = generate.weighted_benchmark(G.to_directed(), partition, rng=np.random.default_rng(6))
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
    H, partition, info = generate.parametric(
        400, degree_fit, size_fit, mixing_fit, k_mean=6.0, k_max=40, size_max=80, seed=10
    )
    assert set(partition) == set(H.nodes())
    assert abs(H.number_of_nodes() - 400) <= 2
    assert info["n_modules"] >= 2
    assert nx.is_connected(H)


def test_parametric_accepts_explicit_families() -> None:
    pytest.importorskip("numpy")
    H, partition, info = generate.parametric(
        300,
        {"family": "powerlaw", "params": {"alpha": 2.7}, "support_min": 1},
        {"family": "geometric", "params": {"p": 0.05}, "support_min": 10},
        {"family": "delta", "mu": 0.1},
        k_mean=6.0,
        k_max=30,
        size_max=60,
        seed=11,
    )
    assert set(partition) == set(H.nodes())
    assert info["realized_E"] == H.number_of_edges()


def test_parametric_rejects_fits_without_a_winner() -> None:
    pytest.importorskip("numpy")
    bad = {"winner": "insufficient", "n_values": 1, "support_min": 1}
    with pytest.raises(ValueError, match="usable family"):
        generate.parametric(100, bad, bad, {"family": "delta", "mu": 0.1}, seed=12)


# -- lfr ----------------------------------------------------------------------
def test_lfr_returns_graph_and_partition() -> None:
    nx = pytest.importorskip("networkx")
    G, partition = generate.lfr(
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
    H, info = generate.benchmark(G, partition, rng=np.random.default_rng(1))
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

    hier, _ = generate.benchmark(G, partition, routing=routing,
                                 rng=np.random.default_rng(1))
    flattened, _ = generate.benchmark(G, flat, rng=np.random.default_rng(1))
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
    expected = "boundary"
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
    plain, plain_info = generate.weighted_benchmark(
        G, partition, degree_target="plain", rng=np.random.default_rng(1))
    flow, flow_info = generate.weighted_benchmark(
        G, partition, degree_target="flow", rng=np.random.default_rng(1))
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
    sampled, info = generate.memory_benchmark(links, windows, partition, seed=1)
    assert sampled, "sampler produced no links"
    violations = [(a, b) for a, b in sampled if windows[a][-1] != windows[b][0]]
    assert violations == [], f"{len(violations)} links skip their junction"
    assert info["dropped_stubs"] == 0          # multi-edges aggregate, nothing strands
    assert 0.7 < info["realized_E"] / info["target_E"] < 1.3


def test_memory_benchmark_reproduces_the_module_mixing() -> None:
    links, windows, partition = memory_blueprint()
    sampled, _ = generate.memory_benchmark(links, windows, partition, seed=1)

    def mixing(edge_weights: dict[Any, float]) -> float:
        total = sum(edge_weights.values())
        cross = sum(w for (a, b), w in edge_weights.items()
                    if partition[a] != partition[b])
        return cross / total

    assert abs(mixing(sampled) - mixing(links)) < 0.05


def test_multilayer_benchmark_keeps_links_inside_their_layer() -> None:
    nx = pytest.importorskip("networkx")
    layers = {}
    partition = {}
    for layer in ("road", "rail"):
        G = nx.planted_partition_graph(3, 20, 0.3, 0.02, seed=1 if layer == "road" else 2)
        layers[layer] = G
        for u in G.nodes():
            partition[(u, layer)] = (G.nodes[u]["block"],)
    sampled, info = generate.multilayer_benchmark(layers, partition, seed=1)
    assert set(sampled) == set(layers)
    for layer, edges in sampled.items():
        assert edges
        assert all(u in layers[layer] and v in layers[layer] for u, v in edges)
    assert 0.8 < info["realized_E"] / info["target_E"] < 1.2


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
