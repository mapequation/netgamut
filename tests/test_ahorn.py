"""The AHORN source: the exchange format, the catalog, and the projections.

The format fixtures are hand-written rather than copies of real files, and they
pin the quirks the real ones actually have: the shipped v0.2 spelling of the
reserved keys, which differs from the published v0.3 specification; interactions
of any size, with and without attributes; explicit node declarations; and the
bare JSON markers that separate the constituent networks of a multi-network file.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from networks import ahorn
from networks._core.domains import domains_for_ahorn
from networks._core.hypergraph import Hypergraph
from networks._core.models import DataFile
from networks.ahorn._catalog import Catalog, load_catalog
from networks.ahorn._models import Dataset, Revision

# The reserved keys are spelled the way every published file spells them:
# underscore-prefixed, from format version 0.2. The specification documents 0.3
# without the underscores, and both must read.
SAMPLE = """{"name": "toy", "_format-version": "0.2", "_revision": 2}
1 {"party": "Democrat"}
2 {"party": "Republican"}
1,2,3 {"category": "algebra", "weight": 2.5}
2,3 {"category": "geometry"}
1,3,4

3,4 {"category": "algebra"}
"""

MULTI = """{"name": "toy-multi", "_num_networks": 2}
{"id": "net-a", "label": "first"}
1,2,3 {"manifold_id": "net-a"}
1,2,4 {"manifold_id": "net-a"}
{"id": "net-b", "label": "second"}
5,6 {"manifold_id": "net-b"}
"""


def record(name: str = "toy", **overrides) -> Dataset:
    fields: dict = {
        "name": name,
        "nodes": 4,
        "interactions": 4,
        "revisions": (
            Revision(number=1, url="https://example.invalid/toy.txt", size=10),
            Revision(
                number=2,
                url="https://example.invalid/toy.txt.gz",
                hif_url="https://example.invalid/toy.hif.json",
                size=5,
                changelog=("Renamed an attribute.",),
            ),
        ),
    }
    fields.update(overrides)
    return Dataset(**fields)


def handle(tmp_path: Path, body: str = SAMPLE, *, gz: bool = False) -> Hypergraph:
    path = tmp_path / ("toy.txt.gz" if gz else "toy.txt")
    path.write_bytes(gzip.compress(body.encode()) if gz else body.encode())
    return Hypergraph(
        dataset=record(),
        file=DataFile(name=path.name, url="https://example.invalid/toy.txt"),
        archive_path=path,
        source="ahorn",
        revision=2,
    )


# -- the exchange format --------------------------------------------------
def test_dataset_metadata_is_the_first_line_verbatim(tmp_path: Path) -> None:
    metadata = handle(tmp_path).metadata
    assert metadata == {"name": "toy", "_format-version": "0.2", "_revision": 2}


@pytest.mark.parametrize(
    "line",
    [
        '{"name": "toy", "_format-version": "0.2", "_revision": 2, "_num_networks": 3}',
        '{"name": "toy", "format-version": "0.2", "revision": 2, "num-networks": 3}',
        '{"name": "toy", "format_version": "0.2", "revision": 2, "num_networks": 3}',
    ],
)
def test_reserved_keys_read_in_every_spelling(tmp_path: Path, line: str) -> None:
    """v0.2 files underscore-prefix these; the v0.3 specification does not."""
    subject = handle(tmp_path, f"{line}\n1,2\n")
    assert subject.format_version == "0.2"
    assert subject.file_revision == 2
    assert subject.num_networks == 3


def test_interactions_carry_their_attributes(tmp_path: Path) -> None:
    found = list(handle(tmp_path).interactions(cast=int))
    assert found == [
        ((1, 2, 3), {"category": "algebra", "weight": 2.5}),
        ((2, 3), {"category": "geometry"}),
        ((1, 3, 4), {}),
        ((3, 4), {"category": "algebra"}),
    ]


def test_a_lone_identifier_declares_a_node_rather_than_an_interaction(tmp_path: Path) -> None:
    """The format cannot distinguish the two, and reads a single member as a node."""
    subject = handle(tmp_path)
    assert [node for node, _ in subject.node_records(cast=int)] == [1, 2]
    assert all(len(members) > 1 for members, _ in subject.interactions())


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    assert len(list(handle(tmp_path).interactions())) == 4


def test_tab_separated_attributes_parse(tmp_path: Path) -> None:
    subject = handle(tmp_path, '{"name": "toy"}\n1,2\t{"weight": 3}\n')
    assert list(subject.interactions(cast=int)) == [((1, 2), {"weight": 3})]


def test_interaction_labels_stay_aligned_with_the_interactions(tmp_path: Path) -> None:
    """An interaction missing the attribute contributes None, not a gap."""
    assert handle(tmp_path).interaction_labels("category") == [
        "algebra",
        "geometry",
        None,
        "algebra",
    ]


# -- multi-network files --------------------------------------------------
def test_markers_are_read_as_constituent_networks(tmp_path: Path) -> None:
    subject = handle(tmp_path, MULTI)
    assert subject.num_networks == 2
    assert [marker["label"] for marker in subject.networks()] == ["first", "second"]


def test_interactions_can_be_taken_from_one_constituent_network(tmp_path: Path) -> None:
    subject = handle(tmp_path, MULTI)
    assert len(list(subject.interactions())) == 3
    assert list(subject.interactions(network="net-b", cast=int)) == [
        ((5, 6), {"manifold_id": "net-b"})
    ]


# -- reading a gzipped file -----------------------------------------------
def test_a_gzipped_dataset_reads_the_same(tmp_path: Path) -> None:
    """Half the published revisions are gzipped and half are not."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert list(handle(tmp_path, gz=True).interactions(cast=int)) == list(
        handle(plain).interactions(cast=int)
    )


# -- projections ----------------------------------------------------------
def test_clique_projection_weights_an_edge_by_shared_interactions(tmp_path: Path) -> None:
    graph = handle(tmp_path).to_networkx(projection="clique")
    # 3 and 4 share {1,3,4} and {3,4}; 1 and 2 share only {1,2,3}.
    assert graph[3][4]["weight"] == 2
    assert graph[1][2]["weight"] == 1


def test_star_projection_keeps_the_interactions_as_nodes(tmp_path: Path) -> None:
    graph = handle(tmp_path).to_networkx(projection="star")
    features = [node for node, data in graph.nodes(data=True) if data.get("bipartite") == 1]
    assert len(features) == 4
    # e0 is {1,2,3}: three members, so three edges.
    assert graph.degree("e0") == 3


def test_an_unknown_projection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown projection"):
        handle(tmp_path).to_networkx(projection="hyper")


def test_to_network_writes_an_edge_list_and_reuses_it(tmp_path: Path) -> None:
    subject = handle(tmp_path)
    network = subject.to_network(projection="clique")
    assert network.path.is_file()
    first = network.path.read_text()

    network.path.write_text("1 2 1\n")  # a rewrite would clobber this
    assert subject.to_network(projection="clique").path.read_text() == "1 2 1\n"
    assert subject.to_network(projection="clique", force=True).path.read_text() == first


def test_the_projected_network_describes_the_projection_not_the_dataset(
    tmp_path: Path,
) -> None:
    """Reporting the interaction count as an edge count would be plainly wrong."""
    subject = handle(tmp_path)
    assert subject.dataset.interactions == 4

    # {1,2,3}, {2,3}, {1,3,4}, {3,4} give 5 distinct pairs and 10 memberships,
    # so neither projection has 4 of anything.
    clique = subject.to_network(projection="clique")
    star = subject.to_network(projection="star")
    assert sum(1 for _ in clique.edges()) == 5
    assert sum(1 for _ in star.edges()) == 10

    assert clique.dataset.edges is None and clique.dataset.nodes is None
    assert clique.dataset.domains == subject.dataset.domains
    assert "clique projection" in str(clique.dataset)


def test_the_projected_network_carries_no_source_tag(tmp_path: Path) -> None:
    """A projection is not what AHORN published, so node metadata stays behind."""
    network = handle(tmp_path).to_network()
    assert network.source == ""
    with pytest.raises(LookupError, match="without a source tag"):
        network.nodes()


def test_to_infomap_builds_the_bipartite_representation(tmp_path: Path) -> None:
    instance = handle(tmp_path).to_infomap(silent=True, seed=1)
    # Four distinct members across the interactions, then one node per interaction.
    assert instance.bipartite_start_id == 4
    assert instance.run().num_top_modules >= 1


# -- node metadata --------------------------------------------------------
def test_node_attributes_become_a_node_table(tmp_path: Path) -> None:
    table = handle(tmp_path).nodes()
    assert table.columns == ("party",)
    assert table.categories("party") == ("Democrat", "Republican")
    assert table.partition("party") == {"1": 0, "2": 1}


def test_a_dataset_without_node_lines_says_where_to_look_instead(tmp_path: Path) -> None:
    subject = handle(tmp_path, '{"name": "toy"}\n1,2,3\n')
    with pytest.raises(LookupError, match="implied by the interactions"):
        subject.nodes()


# -- records and revisions ------------------------------------------------
def test_the_latest_revision_is_the_default() -> None:
    assert record().revision().number == 2
    assert record().revision(1).number == 1


def test_an_unknown_revision_lists_the_ones_that_exist() -> None:
    with pytest.raises(LookupError, match="Available: 1, 2"):
        record().revision(7)


def test_a_dataset_with_no_revision_says_so() -> None:
    with pytest.raises(LookupError, match="no downloadable revision"):
        record(revisions=()).revision()


def test_edges_reports_the_interaction_count_for_cross_source_search() -> None:
    """An interaction is not an edge; the alias exists so max_edges= keeps working."""
    assert record(interactions=17).edges == 17


def test_records_round_trip_through_json() -> None:
    original = record(tags=("domain: reviews",), citations=("@article{a,\n}",))
    assert Dataset.from_dict(json.loads(json.dumps(original.to_dict()))) == original


# -- domains --------------------------------------------------------------
def test_domain_tags_translate_to_the_shared_vocabulary() -> None:
    assert domains_for_ahorn("email-enron", ("domain: communication",)) == ("Social",)
    assert domains_for_ahorn("amazon-reviews", ("domain: reviews",)) == ("Economic",)


def test_similarity_is_a_derivation_not_a_domain() -> None:
    assert domains_for_ahorn("x", ("domain: similarity",)) == ()


def test_a_per_dataset_override_beats_the_tags() -> None:
    """Co-citation is tagged collaboration, but its nodes are papers."""
    assert domains_for_ahorn("cora-cocitation", ("domain: collaboration",)) == ("Informational",)


# -- the catalog ----------------------------------------------------------
def test_lookup_is_case_insensitive_and_suggests_near_misses() -> None:
    catalog = Catalog(datasets=(record("karate-club"),))
    assert catalog.get("KARATE-CLUB").name == "karate-club"
    with pytest.raises(KeyError, match="Did you mean: karate-club"):
        catalog.get("karate-clob")


def test_catalog_round_trips_through_json() -> None:
    catalog = Catalog(datasets=(record("a"), record("b")), retrieved="2026-08-28T00:00:00Z")
    assert Catalog.from_dict(json.loads(json.dumps(catalog.to_dict()))) == catalog


# -- the vendored snapshot ------------------------------------------------
def test_the_bundled_catalog_covers_the_published_collection() -> None:
    catalog = load_catalog()
    # AHORN held 83 datasets when its paper was published and is expected to grow.
    assert len(catalog.datasets) >= 80
    assert all(dataset.revisions for dataset in catalog.datasets)
    assert all(dataset.description for dataset in catalog.datasets)


def test_every_dataset_is_mapped_to_the_shared_vocabulary_or_deliberately_not() -> None:
    """Only the triangulated-manifold collections have no empirical domain."""
    unmapped = {dataset.name for dataset in ahorn.datasets() if not dataset.domains}
    assert unmapped == {"MANTRA-2-manifolds", "MANTRA-3-manifolds"}


def test_the_collection_is_higher_order_throughout() -> None:
    types = {kind for dataset in ahorn.datasets() for kind in dataset.network_types}
    assert "hypergraph" in types and "simplicial-complex" in types
    assert len(ahorn.datasets("simplicial-complex")) >= 40


def test_search_filters_on_formalism_size_and_tag() -> None:
    assert all(d.temporal for d in ahorn.search(temporal=True))
    assert all(d.nodes <= 500 for d in ahorn.search(max_nodes=500))
    assert all("benson" in " ".join(d.tags) for d in ahorn.search(tag="benson"))


def test_parents_only_drops_the_derived_children() -> None:
    everything = ahorn.search()
    parents = ahorn.search(parents_only=True)
    assert len(parents) < len(everything)
    assert not any(dataset.parent for dataset in parents)


def test_describe_reports_the_revisions_and_the_upstream_citation() -> None:
    text = ahorn.describe("algebra-questions")
    assert "Revisions" in text and "(* fetched by default)" in text
    assert "Cite the source" in text and "@article" in text


def test_is_cached_is_false_for_an_unknown_dataset() -> None:
    assert ahorn.is_cached("no-such-dataset") is False


# -- registration ---------------------------------------------------------
def test_ahorn_is_a_source_of_the_top_level_package() -> None:
    import networks

    assert "ahorn" in networks.sources()
    assert networks.search(source="ahorn", max_nodes=100)


# -- live ----------------------------------------------------------------
@pytest.mark.network
def test_building_the_catalog_matches_the_bundled_one() -> None:
    pytest.importorskip("yaml")
    from networks.ahorn._catalog import build_catalog

    fresh = build_catalog(sizes=False)
    bundled = load_catalog()
    assert {d.name for d in fresh.datasets} >= {d.name for d in bundled.datasets}


@pytest.mark.network
def test_loading_a_small_dataset_end_to_end() -> None:
    subject = ahorn.load("karate-club")
    members = [nodes for nodes, _ in subject.interactions(cast=int)]
    assert members and all(len(group) >= 2 for group in members)
    assert ahorn.is_cached("karate-club")


@pytest.mark.network
def test_a_pinned_revision_returns_the_bytes_that_revision_published() -> None:
    """Revision 2 renamed the attribute revision 1 called ``genre``."""
    first = ahorn.load("algebra-questions", revision=1)
    latest = ahorn.load("algebra-questions")
    assert first.archive_path != latest.archive_path
    assert first.interaction_labels("genre")[0] == "lie-algebras"
    assert latest.interaction_labels("category")[0] == "lie-algebras"


@pytest.mark.network
def test_files_predating_the_revision_key_report_none_rather_than_guessing() -> None:
    """Revision 1 of this dataset is format 0.1, which recorded no revision."""
    first = ahorn.load("algebra-questions", revision=1)
    assert first.format_version == "0.1"
    assert first.file_revision is None
    assert first.revision == 1  # what was asked for, which is always known
