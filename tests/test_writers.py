"""Writing every kind of network to file, and the objects the generators return."""

from __future__ import annotations

from pathlib import Path

import pytest

from networks import InMemoryNetwork, MultilayerNetwork, StateNetwork
from networks._core import formats
from networks._core.columns import Columns
from networks._core.models import DataFile
from networks._core.network import Network
from networks.snap._models import Dataset


def make_network(
    path: Path,
    *,
    type_: str | None = "Undirected",
    name: str = "toy",
    columns: Columns | None = None,
) -> Network:
    dataset = Dataset(
        name=name,
        categories=("socnets",),
        category_names=("Social networks",),
        type=type_,
        columns={},
    )
    return Network(
        dataset=dataset,
        file=DataFile(path.name, "u"),
        archive_path=path,
        columns=columns or Columns(),
    )


def lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if not line.startswith("#")]


def infomap_counts(path: Path) -> tuple[int, int]:
    infomap = pytest.importorskip("infomap")
    im = infomap.Infomap(silent=True, num_trials=1)
    im.read_file(str(path))
    im.run()
    return im.num_nodes, im.num_links


def api_counts(net: object) -> tuple[int, int]:
    pytest.importorskip("infomap")
    im = net.to_infomap(silent=True, num_trials=1)  # type: ignore[attr-defined]
    im.run()
    return im.num_nodes, im.num_links


# -- shared helpers -----------------------------------------------------------


def test_output_path_creates_the_directory_and_cleans_the_name(tmp_path: Path) -> None:
    target = formats.output_path(tmp_path / "a" / "b", "soc/Reddit hyper", ".net")
    assert target == tmp_path / "a" / "b" / "soc_Reddit_hyper.net"
    assert target.parent.is_dir()


def test_output_path_refuses_a_file_and_an_empty_name(tmp_path: Path) -> None:
    (tmp_path / "file.net").write_text("")
    with pytest.raises(NotADirectoryError):
        formats.output_path(tmp_path / "file.net", "x", ".net")
    with pytest.raises(ValueError, match="name="):
        formats.output_path(tmp_path, "", ".net")


def test_node_ids_keep_non_negative_integers_and_number_anything_else() -> None:
    assert formats.node_ids([3, 0, 7]) == {0: 0, 3: 3, 7: 7}
    assert formats.node_ids(["b", "a", "b"]) == {"b": 1, "a": 2}
    assert formats.node_ids([1, -1]) == {1: 1, -1: 2}
    assert formats.node_ids([True, 2]) == {True: 1, 2: 2}


def test_ordered_sorts_integer_layers_numerically() -> None:
    assert formats.ordered([10, 2, 1]) == [1, 2, 10]


# -- downloaded networks --------------------------------------------------------


def test_network_to_pajek_keeps_integer_ids(gzipped_edges: Path, tmp_path: Path) -> None:
    net = make_network(gzipped_edges)
    path = net.to_pajek(tmp_path / "out")
    assert path == tmp_path / "out" / "toy.net"
    body = lines(path)
    assert body[0] == "*Vertices 4"
    assert body[1:5] == ['1 "1"', '2 "2"', '3 "3"', '4 "4"']
    assert body[5] == "*Edges"
    assert body[6:] == ["1 2", "2 3", "3 4", "4 1"]


def test_network_to_pajek_follows_direction_and_name_override(
    gzipped_edges: Path, tmp_path: Path
) -> None:
    path = make_network(gzipped_edges, type_="Directed").to_pajek(tmp_path, name="custom")
    assert path.name == "custom.net"
    assert "*Arcs" in lines(path)


def test_network_to_pajek_names_string_labels(tmp_path: Path) -> None:
    edges = tmp_path / "words.txt"
    edges.write_text("alpha beta\nbeta gamma\n")
    body = lines(make_network(edges).to_pajek(tmp_path / "out"))
    assert body[:4] == ["*Vertices 3", '1 "alpha"', '2 "beta"', '3 "gamma"']
    assert body[5:] == ["1 2", "2 3"]


def test_network_writes_weights_exactly_as_to_infomap_reads_them(tmp_path: Path) -> None:
    edges = tmp_path / "weighted.txt"
    edges.write_text("1 2 0.5\n2 3 2\n3 1 1\n")
    net = make_network(edges, columns=Columns(weight=2))
    body = lines(net.to_pajek(tmp_path / "out"))
    assert body[-3:] == ["1 2 0.5", "2 3 2.0", "3 1 1.0"]
    assert lines(net.to_link_list(tmp_path / "out"))[-3:] == ["1 2 0.5", "2 3 2.0", "3 1 1.0"]


def test_network_csv_columns_are_not_weights(csv_edges: Path, tmp_path: Path) -> None:
    # to_infomap reads only the first two columns of a CSV file (ratings, times).
    body = lines(make_network(csv_edges).to_link_list(tmp_path))
    assert body == ["6 2", "6 5"]


def test_network_files_load_the_same_network_as_to_infomap(
    gzipped_edges: Path, tmp_path: Path
) -> None:
    net = make_network(gzipped_edges)
    assert infomap_counts(net.to_pajek(tmp_path)) == api_counts(net)
    assert infomap_counts(net.to_link_list(tmp_path)) == api_counts(net)


def test_link_list_refuses_names_and_leaves_no_partial_file(tmp_path: Path) -> None:
    edges = tmp_path / "words.txt"
    edges.write_text("1 2\nalpha beta\n")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="to_pajek"):
        make_network(edges).to_link_list(out)
    assert list(out.iterdir()) == []


def test_weights_are_read_as_infomap_reads_them(tmp_path: Path) -> None:
    # Infomap's parser takes the numeric prefix of a field; Python's float()
    # would read "1_000" as 1000.
    edges = tmp_path / "odd.txt"
    edges.write_text("0 1 1_000\n1 2 2.5x\n2 0 abc\n")
    net = make_network(edges, columns=Columns(weight=2))
    assert lines(net.to_link_list(tmp_path / "out")) == ["0 1 1.0", "1 2 2.5", "2 0 1.0"]


def test_downloaded_networks_have_no_planted_partition(gzipped_edges: Path) -> None:
    assert make_network(gzipped_edges).partition is None


# -- synthetic first-order networks ----------------------------------------------


@pytest.fixture
def synthetic() -> InMemoryNetwork:
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_nodes_from(range(5))  # node 4 stays isolated
    G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 1.0), (2, 0, 1.0), (3, 2, 0.5)])
    return InMemoryNetwork.from_networkx(
        G,
        name="toy-benchmark",
        partition={0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
        info={"generator": "test", "parameters": {"n": 5, "seed": 1}},
    )


def test_synthetic_to_networkx_carries_the_planted_partition(synthetic: InMemoryNetwork) -> None:
    nx = pytest.importorskip("networkx")
    G = synthetic.to_networkx()
    assert nx.get_node_attributes(G, "module") == synthetic.partition
    assert G[0][1]["weight"] == 2.0
    assert G.name == "toy-benchmark"
    assert "module" not in synthetic.to_networkx(partition_attribute=None).nodes[0]


def test_synthetic_to_pajek_records_how_it_was_made(
    synthetic: InMemoryNetwork, tmp_path: Path
) -> None:
    path = synthetic.to_pajek(tmp_path)
    assert path.name == "toy-benchmark.net"
    text = path.read_text().splitlines()
    assert text[:2] == ["# toy-benchmark", "# generated by test: n=5 seed=1"]
    body = lines(path)
    assert body[0] == "*Vertices 5" and '4 "4"' in body  # the isolated node is kept
    assert "0 1 2.0" in body


def test_synthetic_files_match_to_infomap(synthetic: InMemoryNetwork, tmp_path: Path) -> None:
    assert infomap_counts(synthetic.to_pajek(tmp_path))[1] == api_counts(synthetic)[1]
    assert infomap_counts(synthetic.to_link_list(tmp_path))[1] == api_counts(synthetic)[1]


def test_numpy_integer_labels_are_kept_as_ids(tmp_path: Path) -> None:
    nx = pytest.importorskip("networkx")
    np = pytest.importorskip("numpy")
    G = nx.Graph()
    G.add_edges_from(np.array([[0, 1], [1, 2]], dtype=np.int64))
    net = InMemoryNetwork.from_networkx(G, name="np")
    assert lines(net.to_link_list(tmp_path)) == ["0 1", "1 2"]
    assert lines(net.to_pajek(tmp_path))[:2] == ["*Vertices 3", '0 "0"']


def test_unweighted_synthetic_writes_no_weight_column(tmp_path: Path) -> None:
    nx = pytest.importorskip("networkx")
    net = InMemoryNetwork.from_networkx(nx.path_graph(3), name="path")
    assert not net.weighted
    assert lines(net.to_link_list(tmp_path)) == ["0 1", "1 2"]


# -- multilayer networks -------------------------------------------------------------


@pytest.fixture
def multilayer() -> MultilayerNetwork:
    return MultilayerNetwork(
        name="toy-multiplex",
        intra={0: {(0, 1): 1.0, (1, 2): 1.0}, 1: {(0, 2): 1.0}},
        partition={(0, 0): 0, (1, 0): 0, (2, 0): 1, (0, 1): 0, (2, 1): 0, (3, 1): 1},
    )


def test_multilayer_to_pajek_writes_intra_only(
    multilayer: MultilayerNetwork, tmp_path: Path
) -> None:
    body = lines(multilayer.to_pajek(tmp_path))
    assert body[0] == "*Vertices 4"  # node 3 is a partitioned isolate in layer 1
    intra = body.index("*Intra")
    assert body[intra + 1 :] == ["0 0 1 1.0", "0 1 2 1.0", "1 0 2 1.0"]
    assert "*Inter" not in body


def test_multilayer_inter_links_on_one_node_use_the_inter_section(tmp_path: Path) -> None:
    net = MultilayerNetwork(name="m", intra={0: {(0, 1): 1.0}}, inter={((0, 0), (0, 1)): 0.5})
    body = lines(net.to_pajek(tmp_path))
    assert body[body.index("*Inter") + 1 :] == ["0 0 1 0.5"]


def test_multilayer_inter_links_must_stay_on_one_node() -> None:
    with pytest.raises(ValueError, match="one physical node"):
        MultilayerNetwork(name="m", intra={0: {(0, 1): 1.0}}, inter={((0, 0), (1, 1)): 0.5})


def test_multilayer_inter_links_load_the_same_from_file_and_api(tmp_path: Path) -> None:
    net = MultilayerNetwork(
        name="m",
        intra={0: {(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}, 1: {(0, 1): 1.0, (1, 2): 2.0}},
        inter={((0, 0), (0, 1)): 0.5, ((2, 1), (2, 0)): 0.3},
    )
    assert infomap_counts(net.to_pajek(tmp_path)) == api_counts(net)


def test_multilayer_file_matches_to_infomap(multilayer: MultilayerNetwork, tmp_path: Path) -> None:
    assert infomap_counts(multilayer.to_pajek(tmp_path)) == api_counts(multilayer)


def test_multilayer_has_no_link_list(multilayer: MultilayerNetwork, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="to_pajek"):
        multilayer.to_link_list(tmp_path)


def test_multilayer_to_networkx_supra_graph_and_single_layer(
    multilayer: MultilayerNetwork,
) -> None:
    pytest.importorskip("networkx")
    supra = multilayer.to_networkx()
    assert supra.number_of_nodes() == 6 and supra.number_of_edges() == 3
    assert supra.nodes[(2, 1)] == {"node": 2, "layer": 1, "module": 0}
    layer = multilayer.to_networkx(layer=1)
    assert sorted(layer.nodes()) == [0, 2, 3]
    assert list(layer.edges()) == [(0, 2)]
    assert layer.nodes[3]["module"] == 1
    assert multilayer.layers == [0, 1]


# -- state networks --------------------------------------------------------------------


def test_state_network_to_pajek_writes_the_states_format(tmp_path: Path) -> None:
    net = StateNetwork.from_paths([["a", "b", "c"], ["a", "b", "d"]], name="trails")
    path = net.to_pajek(tmp_path)
    assert path.name == "trails.net"
    assert path.read_text() == net.write(tmp_path / "direct.net").read_text()
    assert "*States" in lines(path)
    with pytest.raises(TypeError, match="to_pajek"):
        net.to_link_list(tmp_path)


def test_state_network_write_is_atomic(tmp_path: Path) -> None:
    (tmp_path / "s.net").write_text("kept\n")
    broken = StateNetwork(
        order=2,
        physical_nodes={1: "a", 2: "b"},
        states={1: (1, 2), 2: (2, 3)},  # physical node 3 is missing
        links={(1, 2): 1.0},
        name="s",
    )
    with pytest.raises(KeyError):
        broken.to_pajek(tmp_path)
    assert (tmp_path / "s.net").read_text() == "kept\n"
    assert [path.name for path in tmp_path.iterdir()] == ["s.net"]


def test_first_order_state_network_writes_plain_arcs(tmp_path: Path) -> None:
    net = StateNetwork.from_paths([(["a", "b", "c"], 2.0)], order=1, name="first")
    body = lines(net.to_pajek(tmp_path))
    assert body == ["*Vertices 3", '1 "a"', '2 "b"', '3 "c"', "*Arcs", "1 2 2.0", "2 3 2.0"]
    assert lines(net.to_link_list(tmp_path)) == ["1 2 2.0", "2 3 2.0"]


def test_state_network_to_networkx_names_states() -> None:
    pytest.importorskip("networkx")
    G = StateNetwork.from_paths([["a", "b", "c"]]).to_networkx()
    assert G.nodes[1] == {"physical": 2, "name": "a -> b"}
    assert G[1][2]["weight"] == 1.0


# -- generators ------------------------------------------------------------------------


def test_multilayer_gm_is_reproducible_and_named() -> None:
    pytest.importorskip("numpy")
    from networks import generate

    a = generate.multilayer_gm(60, 3, p=0.8, mu=0.2, k_min=3, k_max=20, seed=4)
    b = generate.multilayer_gm(60, 3, p=0.8, mu=0.2, k_min=3, k_max=20, seed=4)
    c = generate.multilayer_gm(60, 3, p=0.8, mu=0.2, k_min=3, k_max=20, seed=5)
    assert isinstance(a, MultilayerNetwork)
    assert a.intra == b.intra and a.partition == b.partition
    assert a.intra != c.intra
    assert a.name == "mlgm-multiplex-n60-L3-p0.8-mu0.2-seed4"
    assert len(a.partition or {}) == 60 * 3
    assert not a.inter
    assert all(u != v for links in a.intra.values() for u, v in links)


def test_multilayer_gm_reuses_a_partition_and_checks_its_shape() -> None:
    pytest.importorskip("numpy")
    from networks import generate

    net = generate.multilayer_gm(40, 2, p=0.5, seed=1)
    again = generate.multilayer_gm(40, 2, p=0.5, mu=0.5, partition=net.partition, seed=2)
    assert again.partition == net.partition
    assert again.info["parameters"]["partition"] == "given"
    with pytest.raises(ValueError, match="state nodes"):
        generate.multilayer_gm(41, 2, p=0.5, partition=net.partition)
    with pytest.raises(ValueError, match="coupling"):
        generate.multilayer_gm(40, 2, p=0.5, coupling="ring")


def test_multilayer_gm_temporal_takes_per_layer_probabilities() -> None:
    pytest.importorskip("numpy")
    from networks import generate

    net = generate.multilayer_gm(30, 4, p=[0.9, 0.5, 0.1], coupling="temporal", seed=1)
    assert net.name == "mlgm-temporal-n30-L4-mu0.1-seed1"
    assert net.info["parameters"]["p"] == [0.9, 0.5, 0.1]


def test_memory_benchmark_returns_a_state_network_with_its_partition() -> None:
    pytest.importorskip("numpy")
    from networks import generate

    paths = [["a", "b", "c", "a", "b", "d"], ["c", "b", "d", "b", "c"]] * 5
    blueprint = StateNetwork.from_paths(paths)
    partition = {sid: (0,) if sid % 2 else (1,) for sid in blueprint.states}
    net = generate.memory_benchmark(blueprint.links, blueprint.states, partition, seed=1)
    assert isinstance(net, StateNetwork)
    assert net.order == 2
    assert net.states == blueprint.states  # positive integer ids are kept
    assert net.partition == partition
    assert set(net.links) <= {
        (a, b) for a in net.states for b in net.states if net.states[a][1:] == net.states[b][:-1]
    }


def test_first_order_memory_benchmark_keys_its_partition_like_its_files() -> None:
    pytest.importorskip("numpy")
    from networks import generate

    net = generate.memory_benchmark(
        {(10, 11): 1.0, (11, 12): 1.0, (12, 10): 1.0, (11, 10): 1.0},
        {10: ("a",), 11: ("b",), 12: ("c",)},
        {10: 0, 11: 0, 12: 1},
        seed=1,
    )
    assert net.order == 1
    assert set(net.partition or {}) == set(net.physical_nodes) == set(net.states)
    assert all(net.states[sid] == (sid,) for sid in net.states)


def test_multinomial_benchmark_keeps_the_blueprint_labels() -> None:
    nx = pytest.importorskip("networkx")
    np = pytest.importorskip("numpy")
    from networks import generate

    G = nx.relabel_nodes(nx.karate_club_graph(), {i: f"n{i}" for i in range(34)})
    partition = {u: (0,) if G.nodes[u]["club"] == "Mr. Hi" else (1,) for u in G}
    net = generate.benchmark(
        G, partition, method="multinomial", connect=False, rng=np.random.default_rng(1)
    )
    assert set(net.nodes) <= set(G)
    assert net.partition == {u: partition[u] for u in net.nodes}
    assert len(net.partition) == 34


def test_multilayer_benchmark_reports_the_partition_it_planted() -> None:
    nx = pytest.importorskip("networkx")
    pytest.importorskip("numpy")
    from networks import generate

    K = nx.karate_club_graph()
    layers = {0: K, 1: K.copy()}
    partition = {(u, layer): (0,) for u in K if u != 5 for layer in layers}
    net = generate.multilayer_benchmark(layers, partition, seed=1)
    assert net.partition is not None
    assert net.partition[(5, 0)] == (0,)  # the sampler's default for unlisted nodes
    assert len(net.partition) == 2 * 34
