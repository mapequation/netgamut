"""The Network handle: reading, streaming and converting."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from networks.snap._models import DataFile, Dataset
from networks.snap._network import Network


def make_network(path: Path, *, type_: str | None = "Undirected", name: str = "toy") -> Network:
    dataset = Dataset(
        name=name,
        categories=("socnets",),
        category_names=("Social networks",),
        type=type_,
        columns={},
    )
    return Network(dataset=dataset, file=DataFile(path.name, "u"), archive_path=path)


def test_path_decompresses_a_gzipped_file(gzipped_edges: Path) -> None:
    network = make_network(gzipped_edges)
    plain = network.path
    assert plain.suffix == ".txt"
    assert plain.is_file()
    assert "1\t2" in plain.read_text()


def test_decompression_happens_once(gzipped_edges: Path) -> None:
    network = make_network(gzipped_edges)
    first = network.path
    stamp = first.stat().st_mtime_ns
    assert network.path.stat().st_mtime_ns == stamp


def test_edges_skip_comments_and_stream(gzipped_edges: Path) -> None:
    network = make_network(gzipped_edges)
    assert list(network.edges()) == [("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")]


def test_edges_can_cast_node_ids(gzipped_edges: Path) -> None:
    network = make_network(gzipped_edges)
    assert list(network.edges(cast=int)) == [(1, 2), (2, 3), (3, 4), (4, 1)]


def test_header_exposes_snap_comment_lines(gzipped_edges: Path) -> None:
    network = make_network(gzipped_edges)
    assert network.header[0] == "Undirected graph: toy.txt"
    assert "Nodes: 4 Edges: 4" in network.header


def test_rows_expose_every_column(gzipped_communities: Path) -> None:
    network = make_network(gzipped_communities)
    assert list(network.rows()) == [("1", "2", "3"), ("3", "4")]


def test_communities_parse_variable_length_lines(gzipped_communities: Path) -> None:
    network = make_network(gzipped_communities)
    assert list(network.communities(cast=int)) == [(1, 2, 3), (3, 4)]


def test_plain_uncompressed_files_work(tmp_path: Path) -> None:
    path = tmp_path / "edges.txt"
    path.write_text("# c\n1 2\n2 3\n")
    network = make_network(path)
    assert network.path == path
    assert list(network.edges()) == [("1", "2"), ("2", "3")]


def test_archive_is_detected_and_extracted(tmp_path: Path) -> None:
    member = tmp_path / "inner.txt"
    member.write_text("1\t2\n")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(member, arcname="inner.txt")

    network = make_network(archive)
    assert network.is_archive
    directory = network.extract()
    assert (directory / "inner.txt").read_text() == "1\t2\n"


def test_opening_an_archive_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "b.zip"
    archive.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError, match="archive"):
        make_network(archive).open()


def test_extract_refused_on_a_plain_file(gzipped_edges: Path) -> None:
    with pytest.raises(ValueError, match="single file"):
        make_network(gzipped_edges).extract()


# -- conversions ----------------------------------------------------------
def test_to_networkx_honours_the_declared_direction(gzipped_edges: Path) -> None:
    networkx = pytest.importorskip("networkx")

    undirected = make_network(gzipped_edges, type_="Undirected").to_networkx()
    assert isinstance(undirected, networkx.Graph)
    assert not undirected.is_directed()
    assert undirected.number_of_nodes() == 4
    assert undirected.number_of_edges() == 4

    directed = make_network(gzipped_edges, type_="Directed").to_networkx()
    assert directed.is_directed()


def test_to_networkx_can_be_overridden(gzipped_edges: Path) -> None:
    pytest.importorskip("networkx")
    graph = make_network(gzipped_edges, type_="Undirected").to_networkx(directed=True)
    assert graph.is_directed()


def test_to_networkx_falls_back_to_string_ids(tmp_path: Path) -> None:
    pytest.importorskip("networkx")
    path = tmp_path / "named.txt"
    path.write_text("alice\tbob\nbob\tcarol\n")
    graph = make_network(path).to_networkx()
    assert set(graph.nodes) == {"alice", "bob", "carol"}


def test_to_infomap_loads_the_network(gzipped_edges: Path) -> None:
    pytest.importorskip("infomap")
    instance = make_network(gzipped_edges, type_="Undirected").to_infomap(silent=True)
    result = instance.run()
    assert instance.num_nodes == 4
    assert result.num_top_modules >= 1
