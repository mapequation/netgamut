"""The PyG source: what gets catalogued, and the materialise-once contract.

The parser tests run against hand-written docstrings rather than live PyG source, so
they pin the shapes that matter -- collection vs single graph, named vs unnamed class
-- without depending on PyG's current text. The load tests need ``torch_geometric``
and skip without it; the reload test is the important one, since the whole design
rests on the second call not needing torch.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from networks import pyg
from networks.pyg import _scrape

SINGLE = '''
class Planetoid(InMemoryDataset):
    r"""Citation networks.

    Args:
        root (str): Root directory.
        name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
            :obj:`"PubMed"`).

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 2,708
          - 10,556
          - 1,433
          - 7
        * - CiteSeer
          - 3,327
          - 9,104
          - 3,703
          - 6
    """
'''

COLLECTION = '''
class TUDataset(InMemoryDataset):
    r"""Benchmark collections.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - MUTAG
          - 188
          - ~17.9
          - ~39.6
          - 7
          - 2
    """
'''

UNNAMED = '''
class KarateClub(InMemoryDataset):
    r"""Zachary's karate club.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 34
          - 156
          - 34
          - 4
    """
'''


def test_a_class_with_a_name_column_yields_one_dataset_per_row() -> None:
    entries, collection = _scrape.parse_stats("Planetoid", SINGLE)
    assert collection is False
    assert [e.name for e in entries] == ["Cora", "CiteSeer"]
    assert (entries[0].nodes, entries[0].edges, entries[0].classes) == (2708, 10556, 7)


def test_a_class_without_a_name_column_is_itself_one_dataset() -> None:
    entries, collection = _scrape.parse_stats("KarateClub", UNNAMED)
    assert collection is False
    assert [e.name for e in entries] == ["KarateClub"]
    assert entries[0].nodes == 34


def test_a_graphs_column_above_one_marks_a_collection() -> None:
    """The rule that keeps 151 TUDataset collections out of a one-graph catalog.

    Their node and edge figures are ``~`` averages over thousands of graphs, which is
    the other tell -- and why those cells must not be read as counts.
    """
    entries, collection = _scrape.parse_stats("TUDataset", COLLECTION)
    assert collection is True
    assert entries == []


def test_dataset_names_survive_rst_markup() -> None:
    """PyG writes names as :obj:`"Cora"`; a naive strip leaves ``obj:`Cora``."""
    assert _scrape.parse_names("Planetoid", SINGLE) == ["Cora", "CiteSeer", "PubMed"]


UNTYPED_ARG = '''
class Airports(InMemoryDataset):
    r"""Airport networks.

    Args:
        root: Root directory.
        name: The name of the dataset (:obj:`"USA"`, :obj:`"Brazil"`,
            :obj:`"Europe"`).
        transform: A transform.
    """
'''


def test_names_parse_whether_or_not_the_arg_is_type_annotated() -> None:
    """Newer PyG docstrings write ``name:`` where older ones write ``name (str):``.

    Requiring the annotation silently reduced Airports to one nameless entry that
    could not load at all, since the class requires a name.
    """
    assert _scrape.parse_names("Airports", UNTYPED_ARG) == ["USA", "Brazil", "Europe"]


def test_classes_needing_a_name_are_catalogued_with_their_children() -> None:
    """A nameless entry for a class that requires ``name=`` is an unloadable row."""
    airports = {d.name for d in pyg.datasets() if d.cls == "Airports"}
    assert airports == {"Airports/USA", "Airports/Brazil", "Airports/Europe"}


def test_multi_class_import_lines_are_all_resolved() -> None:
    """``from .citation_full import CitationFull, CoraFull`` defines two classes."""
    mapping = _scrape.module_of(
        "from .citation_full import CitationFull, CoraFull\nfrom .planetoid import Planetoid\n"
    )
    assert mapping == {
        "CitationFull": "citation_full",
        "CoraFull": "citation_full",
        "Planetoid": "planetoid",
    }


def test_the_catalog_holds_single_graph_datasets_with_their_reasons_for_omission() -> None:
    assert len(pyg.datasets()) >= 60
    omitted = pyg.omitted()
    assert "TUDataset" in omitted
    assert "collection" in omitted["TUDataset"]
    assert "SNAPDataset" in omitted
    assert not any(d.cls == "TUDataset" for d in pyg.datasets())


def test_lookup_accepts_the_full_name_a_bare_name_and_the_pyg_prefix() -> None:
    assert pyg.info("Planetoid/Cora").dataset == "Cora"
    assert pyg.info("pyg/Planetoid/Cora").dataset == "Cora"
    assert pyg.info("KarateClub").cls == "KarateClub"


def test_an_ambiguous_bare_name_is_an_error_not_a_guess() -> None:
    """Cora is offered by Planetoid, CitationFull and AttributedGraphDataset."""
    with pytest.raises(KeyError, match="several PyG classes"):
        pyg.info("Cora")


def test_unsized_datasets_are_excluded_from_size_filters_not_assumed_small() -> None:
    """PyG's cheatsheet is a work in progress; a missing size must not read as small.

    Named by property rather than by dataset, since a ``--with-direction`` pass fills
    sizes in and any specific example goes stale.
    """
    unsized = [d for d in pyg.datasets() if d.nodes is None]
    assert unsized, "expected some datasets with no published or measured size"
    matched = {d.name for d in pyg.search(max_nodes=10**9)}
    assert not (matched & {d.name for d in unsized})
    assert "Planetoid/Cora" in matched


def test_every_catalogued_dataset_names_a_class() -> None:
    for dataset in pyg.datasets():
        assert dataset.cls, f"{dataset.name} names no class"


def test_pyg_is_not_an_undirected_only_collection() -> None:
    """The assumption its docs invite, and it is wrong.

    PyG never states direction, so it is easy to conclude there is none. WebKB, Actor,
    the Airports networks and the JODIE interaction graphs are all directed.
    """
    directed = {d.name for d in pyg.search(directed=True)}
    undirected = {d.name for d in pyg.search(directed=False)}
    assert "WebKB/Cornell" in directed
    assert "Actor" in directed
    assert "Planetoid/Cora" in undirected
    assert len(directed) > 15 and len(undirected) > 15


def test_the_same_network_can_differ_by_the_class_that_serves_it() -> None:
    """Cora is undirected through Planetoid and directed through AttributedGraph.

    Which is the argument for measuring direction rather than attaching it to a name:
    the preprocessing, not the network, decides.
    """
    assert pyg.info("Planetoid/Cora").directed is False
    assert pyg.info("AttributedGraphDataset/Cora").directed is True


def test_unmeasured_direction_is_excluded_not_assumed() -> None:
    """A dataset nobody has loaded has no direction, and must not default to either."""
    unmeasured = [d for d in pyg.datasets() if d.directed is None]
    assert unmeasured, "expected some unmeasured datasets in the vendored catalog"
    names = {d.name for d in pyg.search(directed=True)} | {
        d.name for d in pyg.search(directed=False)
    }
    assert not (names & {d.name for d in unmeasured})


def test_structurally_unloadable_datasets_are_marked_and_refused() -> None:
    """Some PyG classes need arguments the loader does not supply.

    Recording that is the difference between a catalog row that explains itself and
    one that fails with a TypeError from deep inside PyG.
    """
    broken = pyg.info("GraphLandDataset/city-roads-M")
    assert broken.supported is False
    assert "split" in broken.load_error
    with pytest.raises(LookupError, match="cannot be built"):
        pyg.load("GraphLandDataset/city-roads-M")
    assert pyg.info("Planetoid/Cora").supported is True


def test_environmental_failures_are_not_baked_into_the_catalog() -> None:
    """An unreachable mirror or a missing rdflib says nothing about the dataset.

    Recording those would permanently mark datasets unloadable because one machine
    could not resolve a hostname on one afternoon.
    """
    for name in ("Twitch/DE", "LastFMAsia", "Entities/AIFB"):
        assert pyg.info(name).supported is True, f"{name} was wrongly marked unloadable"


# ---------------------------------------------------------------------------- #
# Materialising, which is the only part that needs torch
# ---------------------------------------------------------------------------- #
def pyg_or_skip() -> None:
    pytest.importorskip("torch_geometric")


@pytest.mark.network
def test_karate_club_materialises_to_plain_files() -> None:
    """KarateClub needs no download at all -- PyG builds it in memory.

    It also takes neither ``root`` nor ``name``, which is why the loader filters
    constructor arguments by signature instead of passing a fixed set.
    """
    pyg_or_skip()
    pytest.importorskip("networkx")
    net = pyg.load("KarateClub")
    graph = net.to_networkx()
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (34, 78)
    assert not graph.is_directed()
    assert net.nodes().categories("label") == (0, 1, 2, 3)


@pytest.mark.network
def test_materialising_writes_only_what_a_network_needs(tmp_path: Path) -> None:
    """The dense feature matrix is a learning input, not metadata, and is not written.

    Cora's raw PyG directory is ~16 MB, almost all of it the 1,433-dimensional
    bag-of-words matrix; the materialised network is ~44 KB.
    """
    pyg_or_skip()
    net = pyg.load("KarateClub")
    base = net.archive_path.parent
    meta = json.loads((base / "meta.json").read_text())
    assert meta["columns"] == ["label"]
    assert meta["directed"] is False
    assert meta["nodes"] == 34

    with gzip.open(base / "nodes.csv.gz", "rt") as handle:
        header = next(csv.reader(handle))
    assert header == ["node", "label"], "no x, no train/val/test masks"


@pytest.mark.network
def test_reloading_a_materialised_dataset_never_imports_torch() -> None:
    """The property the whole design rests on: torch is a one-time cost.

    Run in a subprocess so the check is real -- torch may already be imported in this
    one by another test.
    """
    pyg_or_skip()
    import subprocess
    import sys

    pyg.load("KarateClub")  # make sure it is materialised
    code = (
        "import sys, networks;"
        "net = networks.load('KarateClub', source='pyg');"
        "assert net.nodes().partition('label');"
        "assert 'torch' not in sys.modules, 'torch was imported on reload';"
        "assert 'torch_geometric' not in sys.modules;"
        "print('ok')"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout
