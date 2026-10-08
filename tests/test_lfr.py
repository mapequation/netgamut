"""The LFR wrapper: argument resolution, output parsing, and the generators themselves.

Two layers, deliberately separated. The unit tests build no networks and start no
processes -- they pin the argument-to-variant mapping, the flag encoding and the
parsers against hand-written copies of what the programs write, so a broken parse is
diagnosable without a compiler. The integration tests run the real generators and are
marked ``network``: they download the sources once (``-m 'not network'`` skips them)
and need a C++ compiler.

The properties they pin are the ones a wrapper can plausibly get wrong: which program
runs, that the seed alone decides the network, that the mixing parameter survives the
round trip, that labels come back zero-based and cover the graph, and that the
hierarchical and multiplex shapes are the ones the rest of ``generate`` consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from networks import MultilayerNetwork, generate
from networks.generate import _lfr
from networks.generate._lfr import _io


# ---------------------------------------------------------------------------- #
# Which generator do the arguments select?
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mu": 0.2}, "undirected"),
        ({"mu": 0.2, "directed": True}, "directed"),
        ({"muw": 0.2}, "weighted"),
        ({"mut": 0.2, "muw": 0.2}, "weighted"),
        ({"beta": 1.5, "muw": 0.2}, "weighted"),
        ({"muw": 0.2, "directed": True}, "weighted_directed"),
        ({"mu1": 0.1, "mu2": 0.3}, "hierarchical"),
        ({"mu2": 0.3}, "hierarchical"),
        ({"mu": 0.2, "layers": 4}, "multiplex"),
    ],
)
def test_arguments_select_the_generator(kwargs: dict[str, Any], expected: str) -> None:
    """Every upstream program is reachable, and only through its own arguments."""
    variant = _lfr._resolve(
        directed=bool(kwargs.get("directed")),
        weighted=any(kwargs.get(name) is not None for name in ("muw", "mut", "beta")),
        hierarchical=any(kwargs.get(name) is not None for name in ("mu1", "mu2")),
        multiplex=kwargs.get("layers") is not None,
    )
    assert variant.key == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mu1": 0.1, "directed": True}, "hierarchical"),
        ({"mu1": 0.1, "muw": 0.2}, "hierarchical"),
        ({"layers": 3, "muw": 0.2}, "multiplex"),
        ({"mu": 0.2, "muw": 0.2}, "mut and muw"),
        ({}, "mu is required"),
        ({"muw": None, "mut": 0.2}, "muw is required"),
        ({"mu": 0.2, "bound": "upper"}, "bound must be"),
        ({"mu": 0.2, "directed": True, "clustering": 0.3}, "no -C flag"),
    ],
)
def test_impossible_combinations_are_refused(kwargs: dict[str, Any], message: str) -> None:
    """Upstream has no directed hierarchical generator and no -C for directed ones.

    Refusing in Python beats letting the program exit 255 on an unknown option, and
    beats silently dropping a flag the caller believed had an effect.
    """
    with pytest.raises(ValueError, match=message):
        generate.lfr(100, 10, 20, **kwargs)


def test_flags_are_encoded_the_way_the_programs_read_them() -> None:
    """``-flag value`` pairs, unset options omitted, floats at full precision.

    The programs parse their own command line and reject unknown options, so an
    accidental ``-minc None`` is an error rather than a default.
    """
    assert _lfr._flags(N=1000, k=15.0, maxk=50, minc=None, on=0) == [
        "-N", "1000", "-k", "15", "-maxk", "50", "-on", "0",
    ]
    assert _lfr._flags(mu=0.1 + 0.2) == ["-mu", "0.30000000000000004"]
    assert _lfr._flags(sup=True, inf=False) == ["-sup"]


# ---------------------------------------------------------------------------- #
# Parsing what the programs write
# ---------------------------------------------------------------------------- #
def test_undirected_edges_are_read_once_despite_being_written_twice(tmp_path: Path) -> None:
    """``network.dat`` lists every undirected edge in both directions.

    Reading it into a Graph collapses the pair; reading the directed variant's file,
    which lists each arc once, must not.
    """
    pytest.importorskip("networkx")
    path = tmp_path / "network.dat"
    path.write_text("1\t2\n2\t1\n1\t3\n3\t1\n2\t2\n")  # last line is a self-loop

    undirected = _io.read_network(path, directed=False, weighted=False)
    assert undirected.number_of_edges() == 2
    assert not undirected.has_edge(2, 2), "self-loops carry no flow and are dropped"

    directed = _io.read_network(path, directed=True, weighted=False)
    assert directed.number_of_edges() == 4


def test_weights_are_read_as_floats(tmp_path: Path) -> None:
    pytest.importorskip("networkx")
    path = tmp_path / "network.dat"
    path.write_text("1\t2\t1.38021\n2\t1\t1.38021\n")
    G = _io.read_network(path, directed=False, weighted=True)
    assert G[1][2]["weight"] == pytest.approx(1.38021)


def test_memberships_read_one_or_many_per_node(tmp_path: Path) -> None:
    """Overlapping nodes put several community ids on one line, with trailing space."""
    path = tmp_path / "community.dat"
    path.write_text("1\t1 \n2\t3 \n6\t1 4 \n\n")
    assert _io.read_memberships(path) == {1: (1,), 2: (3,), 6: (1, 4)}


def test_partition_shape_follows_the_call() -> None:
    """One shallow path is an int, one deep path a tuple, several a cover."""
    assert _io.as_partition({0: [(2,)], 1: [(0,)]}) == {0: 2, 1: 0}
    assert _io.as_partition({0: [(1, 5)]}) == {0: (1, 5)}
    assert _io.as_partition({0: [(1,), (4,)]}) == {0: (1, 4)}
    assert _io.as_partition({0: [(1, 5), (1, 7)]}) == {0: ((1, 5), (1, 7))}


def test_micro_communities_are_placed_by_their_unambiguous_nodes() -> None:
    """The hierarchical files never state the nesting, so it is read off the data.

    Nodes with a single membership at both levels fix which macro each micro sits in;
    an overlapping node is then resolved through that map rather than guessed at.
    """
    micro = {1: (1,), 2: (2,), 3: (1, 2), 4: (3,)}
    macro = {1: (1,), 2: (1,), 3: (1,), 4: (2,)}
    assert _io.macro_of_micro(micro, macro) == {1: 1, 2: 1, 3: 2}


def test_compact_renumbers_from_zero() -> None:
    assert _io.compact([3, 1, 1, 7]) == {1: 0, 3: 1, 7: 2}


def test_the_pinned_upstream_revision_is_recorded() -> None:
    """The pin is what makes a seed reproducible across machines, so it is asserted.

    Bumping it is a deliberate edit -- the generators are random, and a different
    revision can mean a different network for the same seed -- so it fails here first,
    where ``scripts/update_lfr_pin.py`` is the fix and the diff shows the change.
    """
    assert _lfr.COMMIT == "ec9a860282d8fc9d52e9e08fea9ce93ddeb473af"
    assert _lfr.URL.endswith(f"{_lfr.COMMIT}.tar.gz")
    assert set(_lfr.VARIANTS) == {
        "undirected", "weighted", "directed", "weighted_directed",
        "hierarchical", "multiplex",
    }


def test_the_pin_is_visible_without_fetching_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``lfr_source()`` is a status line: it must never download or compile.

    Pointed at an empty cache it reports "not fetched" rather than fetching, which is
    what makes it safe to call from a notebook header or a CI summary.
    """
    monkeypatch.setenv("NETWORKS_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NETWORKS_LFR_DIR", raising=False)
    from networks._core import cache

    monkeypatch.setattr(cache, "_override", None)

    source = generate.lfr_source()
    assert source.commit == _lfr.COMMIT
    assert source.sha256 in source.__str__()
    assert source.repository == "andrealancichinetti/LFRbenchmarks"
    assert source.fetched is False
    assert not any(tmp_path.iterdir()), "reporting the pin must not populate the cache"
    assert source.overridden is False
    assert f"{_lfr.COMMIT[:12]}" in str(source)


def test_an_override_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``NETWORKS_LFR_DIR`` wins over the pin, and the report says which is in force."""
    monkeypatch.setenv("NETWORKS_LFR_DIR", str(tmp_path))
    source = generate.lfr_source()
    assert source.overridden is True
    assert source.directory == tmp_path
    assert source.fetched is True  # the directory exists, whatever is in it
    assert "NETWORKS_LFR_DIR" in str(source)


# ---------------------------------------------------------------------------- #
# The generators themselves
# ---------------------------------------------------------------------------- #
def compiler_or_skip() -> None:
    from networks.generate._lfr._build import CompilerNotFound, compiler

    try:
        compiler()
    except CompilerNotFound as error:  # pragma: no cover - depends on the machine
        pytest.skip(str(error))


@pytest.mark.network
def test_the_binary_benchmark_plants_the_mixing_it_was_asked_for() -> None:
    """The headline property: the fraction of links leaving a community is mu.

    Checked at two values so a constant would not pass, and loosely, because LFR
    matches mu in expectation over a finite degree sequence, not exactly.
    """
    pytest.importorskip("networkx")
    compiler_or_skip()
    for mu in (0.1, 0.4):
        net = generate.lfr(500, 15, 50, mu=mu, minc=20, maxc=50, seed=3)
        G, partition = net.to_networkx(), net.partition
        cross = sum(1 for u, v in G.edges() if partition[u] != partition[v])
        assert cross / G.number_of_edges() == pytest.approx(mu, abs=0.02)


@pytest.mark.network
def test_the_seed_alone_decides_the_network() -> None:
    """The programs seed themselves from ``time_seed.dat``, which the wrapper writes.

    Left to itself the generator reads the clock, so this is the property that makes
    an LFR experiment repeatable -- and it has to hold across processes, not just
    within one.
    """
    pytest.importorskip("networkx")
    compiler_or_skip()
    first = generate.lfr(200, 10, 25, mu=0.2, minc=20, maxc=50, seed=7)
    again = generate.lfr(200, 10, 25, mu=0.2, minc=20, maxc=50, seed=7)
    other = generate.lfr(200, 10, 25, mu=0.2, minc=20, maxc=50, seed=8)

    assert sorted(first.to_networkx().edges()) == sorted(again.to_networkx().edges())
    assert first.partition == again.partition
    assert sorted(first.to_networkx().edges()) != sorted(other.to_networkx().edges())


@pytest.mark.network
def test_labels_are_zero_based_and_the_partition_covers_the_graph() -> None:
    """Upstream writes one-based ids; the rest of generate is zero-based."""
    pytest.importorskip("networkx")
    compiler_or_skip()
    nx = pytest.importorskip("networkx")
    net = generate.lfr(200, 10, 25, mu=0.2, minc=20, maxc=50, seed=1)
    G, partition = net.to_networkx(), net.partition
    assert min(G) == 0
    assert max(G) == G.number_of_nodes() - 1
    assert set(partition) == set(G)
    assert min(partition.values()) == 0
    # the network object carries the partition on the graph as the "module" attribute
    assert partition == nx.get_node_attributes(G, "module")


@pytest.mark.network
@pytest.mark.parametrize(
    ("kwargs", "directed", "weighted"),
    [
        ({"mu": 0.2}, False, False),
        ({"mut": 0.2, "muw": 0.2, "beta": 1.5}, False, True),
        ({"mu": 0.2, "directed": True}, True, False),
        ({"mut": 0.2, "muw": 0.2, "directed": True}, True, True),
    ],
)
def test_each_flat_variant_returns_the_graph_type_it_promises(
    kwargs: dict[str, Any], directed: bool, weighted: bool
) -> None:
    pytest.importorskip("networkx")
    compiler_or_skip()
    net = generate.lfr(200, 10, 25, minc=20, maxc=50, seed=1, **kwargs)
    G, partition = net.to_networkx(), net.partition
    assert G.is_directed() is directed
    assert G.number_of_nodes() == 200
    _, _, data = next(iter(G.edges(data=True)))
    assert ("weight" in data) is weighted
    assert all(isinstance(module, int) for module in partition.values())


@pytest.mark.network
def test_the_hierarchical_variant_returns_nested_paths() -> None:
    """``{node: (macro, micro)}`` is the path convention the flow model reads.

    Each micro community must sit inside exactly one macro community, or the paths
    would not form a tree and ModuleTree would reject them.
    """
    pytest.importorskip("networkx")
    numpy = pytest.importorskip("numpy")
    compiler_or_skip()
    net = generate.lfr(
        600, 20, 50, mu1=0.1, mu2=0.3, minc=10, maxc=50,
        min_macro=100, max_macro=300, seed=1,
    )
    G, partition = net.to_networkx(), net.partition
    assert {len(path) for path in partition.values()} == {2}
    inside: dict[int, int] = {}
    for macro, micro in partition.values():
        assert inside.setdefault(micro, macro) == macro

    tree = generate.ModuleTree(G, partition)
    assert len(tree.leaves) == len({p for p in partition.values()})
    P, nodes, tree = generate.flow_matrix(G, partition)
    p = numpy.array([tree.p[u] for u in nodes])
    assert numpy.allclose(P.sum(axis=1), p, atol=1e-14), "P must stay proper in its rows"

    # the two planted levels are two levels: mixing out of a micro community is mu1+mu2
    mixing = generate.mixing_stats(G, partition)
    assert float(numpy.mean(list(mixing.values()))) == pytest.approx(0.4, abs=0.05)


@pytest.mark.network
def test_the_multiplex_variant_returns_layers_and_a_state_partition() -> None:
    """``({layer: graph}, {(node, layer): module})`` -- what multilayer_benchmark takes.

    Layers cut from one generated graph share its communities; layers cut from
    different graphs must not, even though upstream numbers both from 1.
    """
    pytest.importorskip("networkx")
    compiler_or_skip()
    net = generate.lfr(
        150, 10, 25, mu=0.2, minc=20, maxc=40, layers=2, graphs=2, seed=1
    )
    assert isinstance(net, MultilayerNetwork)
    layers = {i: net.to_networkx(layer=i) for i in net.layers}
    partition = net.partition
    assert sorted(layers) == [0, 1, 2, 3]
    assert all(G.number_of_nodes() == 150 for G in layers.values())
    assert set(partition) == {(u, layer) for layer in layers for u in layers[layer]}

    first_graph = {module for (_, layer), module in partition.items() if layer < 2}
    second_graph = {module for (_, layer), module in partition.items() if layer >= 2}
    assert not (first_graph & second_graph), "layers of different graphs share no module"
    assert partition[(0, 0)] == partition[(0, 1)], "layers of one graph share communities"

    sampled = generate.multilayer_benchmark(layers, partition, seed=1).intra
    assert sorted(sampled) == sorted(layers)


@pytest.mark.network
def test_overlapping_nodes_come_back_as_a_cover() -> None:
    """``on``/``om`` make a cover, not a partition, and the shape says so."""
    pytest.importorskip("networkx")
    compiler_or_skip()
    net = generate.lfr(200, 10, 25, mu=0.2, minc=20, maxc=50, on=20, om=2, seed=1)
    cover = net.partition
    overlapping = [value for value in cover.values() if isinstance(value, tuple)]
    assert len(overlapping) == 20
    assert {len(memberships) for memberships in overlapping} == {2}


@pytest.mark.network
def test_infeasible_parameters_raise_with_the_generator_s_own_words() -> None:
    """The programs print the problem and still exit 0, writing nothing.

    Trusting the exit code would hand back an empty graph, so the wrapper checks for
    the output files and quotes what the program said.
    """
    compiler_or_skip()
    with pytest.raises(RuntimeError, match="average degree"):
        generate.lfr(50, 100, 5, mu=0.2, seed=1)
