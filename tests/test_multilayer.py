"""Multilayer networks stacked from separate networks, and Netzschleuder's layers."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.parse import unquote

import pytest

from netgamut import MultilayerNetwork
from netgamut import netzschleuder as nz
from netgamut._core.models import DataFile
from netgamut._core.network import Network
from netgamut.snap._models import Dataset

#: Two subgraphs that number their nodes on their own: alice is node 0 in the
#: first and node 1 in the second, so only the ``id`` column matches them.
BUNDLES = {
    "calls": (
        "# source, target, timestamp\n0,1,10\n1,2,20\n0,1,30\n",
        "# index, id, name\n0,7,alice\n1,8,bob\n2,9,carol\n",
    ),
    "sms": (
        "# source, target\n0,1\n",
        "# index, id, name\n0,8,bob\n1,7,alice\n",
    ),
    "repeated": (
        "# source, target\n0,1\n",
        "# index, id\n0,7\n1,7\n",
    ),
}


@pytest.fixture
def bundles(monkeypatch):
    """Serve each subgraph's CSV bundle from :data:`BUNDLES`, counting downloads."""
    fetched: list[str] = []

    def download(url, target, **kwargs):
        stem = unquote(url.rsplit("/", 1)[1]).removesuffix(".csv.zip")
        fetched.append(stem)
        edges, nodes = BUNDLES[stem]
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as bundle:
            bundle.writestr("edges.csv", edges)
            bundle.writestr("nodes.csv", nodes)
        return target

    monkeypatch.setattr(nz, "_download_file", download)
    return fetched


def plain(path: Path, text: str, *, type_: str = "Undirected") -> Network:
    path.write_text(text)
    dataset = Dataset(name=path.stem, categories=(), category_names=(), type=type_, columns={})
    return Network(dataset=dataset, file=DataFile(path.name, "u"), archive_path=path)


def test_networks_sharing_ids_stack_as_layers(tmp_path: Path) -> None:
    """SNAP's higgs-twitter files share one id space, so no key is needed."""
    retweet = plain(tmp_path / "retweet.txt", "1 2\n2 3\n2 1\n")
    reply = plain(tmp_path / "reply.txt", "3 4\n")
    ml = MultilayerNetwork.from_networks({"retweet": retweet, "reply": reply})
    # Undirected, so 2 1 is the link 1 2 again.
    assert ml.intra == {"retweet": {(1, 2): 2.0, (2, 3): 1.0}, "reply": {(3, 4): 1.0}}
    assert ml.physical_nodes == [1, 2, 3, 4]
    assert (ml.name, ml.directed, ml.weighted) == ("retweet", False, True)


def test_any_directed_network_makes_the_stack_directed(tmp_path: Path) -> None:
    follows = plain(tmp_path / "follows.txt", "1 2\n2 1\n", type_="Directed")
    friends = plain(tmp_path / "friends.txt", "1 2\n")
    ml = MultilayerNetwork.from_networks({"f": follows, "g": friends}, name="both")
    assert ml.directed
    assert ml.intra["f"] == {(1, 2): 1.0, (2, 1): 1.0}
    assert ml.name == "both"


def test_stacking_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no networks"):
        MultilayerNetwork.from_networks({})


def test_subgraphs_match_their_nodes_by_a_column(bundles, tmp_path: Path) -> None:
    ml = nz.load_multilayer("celegansneural", ["calls", "sms"])
    # Matched by id, not by row: sms's 0 1 is bob to alice, calls's 0 1 alice to bob.
    assert ml.intra["calls"] == {(7, 8): 2.0, (8, 9): 1.0}
    assert ml.intra["sms"] == {(8, 7): 1.0}
    assert ml.physical_nodes == [7, 8, 9]
    assert ml.names == {7: "alice", 8: "bob", 9: "carol"}
    assert ml.info == {"source": "celegansneural", "layers": "subgraph", "key": "id"}
    text = ml.to_pajek(tmp_path).read_text()
    assert '7 "alice"' in text
    assert "*Intra" in text


def test_the_key_can_be_named_or_be_the_row_number(bundles) -> None:
    by_name = nz.load_multilayer("celegansneural", ["calls", "sms"], key="name")
    assert by_name.physical_nodes == ["alice", "bob", "carol"]
    # Row numbers: sms's node 0 is calls's node 0, whoever they are.
    by_row = nz.load_multilayer("celegansneural", ["calls", "sms"], key="index")
    assert by_row.intra["sms"] == {(0, 1): 1.0}
    assert by_row.info["key"] == "index"


def test_a_key_that_repeats_inside_a_layer_is_refused(bundles) -> None:
    with pytest.raises(ValueError, match="such as '7', belong to more than one node"):
        nz.load_multilayer("celegansneural", ["calls", "repeated"])


def test_a_known_entry_stacks_its_own_subgraphs(bundles, monkeypatch) -> None:
    monkeypatch.setitem(nz._SUBGRAPH_LAYERS, "celegansneural", (("calls", "sms"), "name"))
    ml = nz.load_multilayer("celegansneural")
    assert bundles == ["calls", "sms"]
    assert ml.layers == ["calls", "sms"]
    assert ml.info["key"] == "name"


def test_a_layer_column_is_used_where_there_is_one(monkeypatch) -> None:
    def download(url, target, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as bundle:
            bundle.writestr("edges.csv", "# source, target, layer\n0,1,1\n1,2,2\n")
            bundle.writestr("nodes.csv", "# index, nodeLabel\n0,a\n1,b\n2,c\n")
        return target

    monkeypatch.setattr(nz, "_download_file", download)
    ml = nz.load_multilayer("london_transport")
    assert ml.intra == {1: {(0, 1): 1.0}, 2: {(1, 2): 1.0}}
    # Node names come along, as they do for to_pajek.
    assert ml.names == {0: "a", 1: "b", 2: "c"}


def test_an_entry_without_layers_says_how_to_name_them(bundles, monkeypatch) -> None:
    monkeypatch.setitem(BUNDLES, "celegansneural", ("# source, target\n0,1\n", "# index\n0\n1\n"))
    with pytest.raises(LookupError, match="subgraphs="):
        nz.load_multilayer("celegansneural")


def test_subgraphs_without_a_shared_key_ask_for_one(bundles, monkeypatch) -> None:
    monkeypatch.setitem(BUNDLES, "bare", ("# source, target\n0,1\n", "# index\n0\n1\n"))
    with pytest.raises(LookupError, match="key='index'"):
        nz.load_multilayer("celegansneural", ["calls", "bare"])


def test_every_known_layout_names_real_subgraphs_and_columns() -> None:
    for name, (subgraphs, _) in nz._SUBGRAPH_LAYERS.items():
        assert set(subgraphs) <= set(nz.info(name).subgraphs), name
    for name, column in nz._LAYER_COLUMNS.items():
        properties = [entry[0] for entry in nz.info(name).analyses["edge_properties"]]
        assert column in properties, name
