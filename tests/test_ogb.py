"""The OGB source: catalog, loadability rules, and the metadata that makes it worth it.

Catalog tests are offline against the vendored snapshot. The two tests that download
are marked ``network`` -- ``ogbl-ddi`` at 46 MB is the smallest OGB ships.
"""

from __future__ import annotations

import pytest

from networks import ogb
from networks.ogb._catalog import _parse_master

MASTER = """,ogbn-demo,ogbn-hetero
num classes,7,3
task type,multiclass classification,multiclass classification
download_name,demo,hetero
version,1,1
url,http://example.invalid/demo.zip,http://example.invalid/hetero.zip
add_inverse_edge,True,False
additional node files,node_year,None
is hetero,False,True
binary,False,False
"""


def test_master_csv_is_transposed_and_parsed_per_column() -> None:
    """OGB writes one *column* per dataset, not one row -- the usual way to misread it."""
    demo, hetero = _parse_master("nodeproppred", MASTER)
    assert demo.name == "ogbn-demo"
    assert demo.num_classes == 7
    assert demo.download_name == "demo"
    assert demo.node_files == ("node_year",)
    assert hetero.name == "ogbn-hetero"


def test_add_inverse_edge_is_inverted_into_directedness() -> None:
    """OGB adds the reverse of every stored edge exactly when the network is undirected.

    Reading the flag straight through would label every undirected OGB network as
    directed, and silently double its edge count on conversion.
    """
    demo, hetero = _parse_master("nodeproppred", MASTER)
    assert demo.directed is False  # add_inverse_edge=True  -> undirected
    assert hetero.directed is True  # add_inverse_edge=False -> directed


def test_none_is_not_a_node_file() -> None:
    """``additional node files`` is the literal string "None" when there are none."""
    _, hetero = _parse_master("nodeproppred", MASTER)
    assert hetero.node_files == ()


def test_the_catalog_ships_the_single_graph_families_only() -> None:
    """graphproppred is 15 collections of small molecules; one Network cannot hold one."""
    families = {dataset.family for dataset in ogb.datasets()}
    assert families == {"nodeproppred", "linkproppred"}
    assert len(ogb.datasets()) >= 12
    assert not any(name.startswith("ogbg-") for name in (d.name for d in ogb.datasets()))


def test_unloadable_datasets_are_catalogued_with_a_reason() -> None:
    """Better a listed dataset that explains itself than a silent omission."""
    assert ogb.info("ogbn-mag").supported is False
    assert "heterogeneous" in ogb.info("ogbn-mag").unsupported_reason
    assert ogb.info("ogbn-papers100M").supported is False
    assert "npz" in ogb.info("ogbn-papers100M").unsupported_reason
    assert ogb.info("ogbn-arxiv").supported is True
    assert ogb.info("ogbn-arxiv").unsupported_reason == ""


def test_load_refuses_the_unsupported_ones_before_downloading() -> None:
    """ogbn-papers100M is 57 GB; the refusal must come before the transfer."""
    with pytest.raises(LookupError, match="npz"):
        ogb.load("ogbn-papers100M")
    with pytest.raises(LookupError, match="heterogeneous"):
        ogb.load("ogbn-mag")


def test_search_filters_on_loadability_and_domain() -> None:
    assert all(d.supported for d in ogb.search(supported=True))
    assert {d.name for d in ogb.search(supported=False)} == {
        "ogbn-mag",
        "ogbn-papers100M",
        "ogbl-biokg",
        "ogbl-vessel",
    }
    assert all("Biological" in d.domains for d in ogb.search(domain="Biological"))


def test_every_dataset_has_a_domain_and_an_archive() -> None:
    """A dataset with no domain drops out of every domain query, silently."""
    for dataset in ogb.datasets():
        assert dataset.domains, f"{dataset.name} has no domain"
        assert dataset.primary_file is not None, f"{dataset.name} has no archive"


def test_sizes_are_recorded_only_where_they_were_measured() -> None:
    """Counts come from the archives' own num-node-list, never from a web page.

    The big ones stay None rather than being guessed, which is why size filters
    exclude them instead of matching.
    """
    arxiv = ogb.info("ogbn-arxiv")
    assert (arxiv.nodes, arxiv.edges) == (169343, 1166243)
    assert ogb.info("ogbl-ddi").nodes == 4267
    assert ogb.info("ogbn-papers100M").nodes is None


def test_unknown_dataset_suggests_near_misses() -> None:
    with pytest.raises(KeyError, match="Did you mean"):
        ogb.info("ogbn-arxi")


@pytest.mark.network
def test_ddi_loads_and_carries_its_entity_mapping() -> None:
    """The smallest OGB dataset, 46 MB. No labels, but a drug id per node."""
    pytest.importorskip("networkx")
    net = ogb.load("ogbl-ddi", progress=False)
    graph = net.to_networkx()
    assert graph.number_of_nodes() == 4267
    assert not graph.is_directed()

    nodes = net.nodes()
    assert nodes.columns == ("drug",)
    assert nodes["drug"][0] == "DB00001"


@pytest.mark.network
def test_arxiv_labels_decode_to_their_names() -> None:
    """The reason OGB is worth reading: the codes have meanings, and they ship.

    83 MB. ``mapping/labelidx2arxivcategeory.csv.gz`` is what turns module 3 into
    "arxiv cs cy", which is the difference between a describable covariate and an
    anonymous integer.
    """
    net = ogb.load("ogbn-arxiv", progress=False)
    nodes = net.nodes()
    assert set(nodes.columns) == {"arxiv_category", "year", "paper"}

    categories = nodes.categories("arxiv_category")
    assert len(categories) == 40
    assert all(name.startswith("arxiv cs ") for name in categories)

    partition = nodes.partition("arxiv_category")
    assert len(partition) == 169343
    assert categories[partition[0]] == nodes["arxiv_category"][0]
    assert nodes["year"][0] == 2013
