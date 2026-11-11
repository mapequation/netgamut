"""The normalised domain vocabulary and the cross-source API."""

from __future__ import annotations

import pytest

import netgamut
from netgamut._core.domains import (
    DOMAINS,
    domains_for_netzschleuder_tags,
    domains_for_snap,
)


# -- the mapping ----------------------------------------------------------
def test_snap_categories_map_to_domains() -> None:
    assert domains_for_snap("roadNet-PA", ("road",)) == ("Transportation",)
    assert domains_for_snap("web-Google", ("web",)) == ("Informational",)
    assert domains_for_snap("as-Skitter", ("as",)) == ("Technological",)
    assert domains_for_snap("amazon0302", ("amazon",)) == ("Economic",)


def test_cross_listed_snap_dataset_spans_domains() -> None:
    assert domains_for_snap("wiki-Vote", ("socnets", "wikipedia")) == (
        "Social",
        "Informational",
    )


def test_per_dataset_override_beats_the_category() -> None:
    """SNAP files com-Amazon under ground-truth communities, which says nothing."""
    assert domains_for_snap("com-Amazon", ("communities",)) == ("Economic",)
    assert domains_for_snap("com-DBLP", ("communities",)) == ("Social",)


def test_property_only_categories_contribute_nothing_alone() -> None:
    """ "Temporal networks" describes the edges, not the system."""
    assert domains_for_snap("unknown-thing", ("temporal",)) == ()


def test_netzschleuder_tags_are_filtered_not_translated() -> None:
    assert domains_for_netzschleuder_tags(("Biological", "Food web", "Weighted")) == ("Biological",)
    assert domains_for_netzschleuder_tags(("Weighted",)) == ()


def test_domains_are_always_in_canonical_order() -> None:
    assert domains_for_netzschleuder_tags(("Economic", "Social")) == ("Social", "Economic")
    assert domains_for_snap("wiki-talk-temporal", ()) == ("Social", "Informational")


# -- cross-source API -----------------------------------------------------
def test_sources_are_registered() -> None:
    assert netgamut.sources() == (
        "snap",
        "netzschleuder",
        "konect",
        "ogb",
        "pyg",
        "paths",
        "ahorn",
    )


def test_domain_counts_sum_across_sources() -> None:
    counts = netgamut.domains()
    assert set(counts) == set(DOMAINS)
    # SNAP contributes no biological networks at all, so this comes entirely
    # from Netzschleuder -- the reason the second source exists.
    assert counts["Biological"] >= 30
    assert counts["Social"] > counts["Transportation"]


def test_snap_has_no_biological_networks() -> None:
    """Documents the gap that motivated adding a second source."""
    from netgamut import snap

    assert not [item for item in snap.datasets() if "Biological" in item.domains]


def test_cross_source_search_reaches_every_source() -> None:
    """One query, one vocabulary, whichever catalog the network came from.

    Transportation is the domain every source happens to cover, so it is the one that
    proves the vocabulary is genuinely shared rather than per-source.
    """
    sources = {match.source for match in netgamut.search(domain="Transportation")}
    assert {"snap", "netzschleuder", "paths", "pyg"} <= sources

    # and each source can still be reached on its own terms
    for source in netgamut.sources():
        assert netgamut.search(source=source), f"{source} returned nothing"


def test_matches_are_uniquely_keyed() -> None:
    keys = [match.key for match in netgamut.search()]
    assert len(keys) == len(set(keys))
    assert all("/" in key for key in keys)


def test_search_can_be_restricted_to_one_source() -> None:
    found = netgamut.search(domain="Social", source="snap")
    assert found
    assert {match.source for match in found} == {"snap"}


def test_search_filters_by_size_across_sources() -> None:
    found = netgamut.search(max_nodes=500)
    assert found
    assert all(match.nodes is not None and match.nodes <= 500 for match in found)


def test_unknown_domain_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown domain"):
        netgamut.search(domain="Chemical")


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown source"):
        netgamut.search(source="nope")


def test_load_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="No dataset named"):
        netgamut.load("definitely-not-a-network")


def test_load_refuses_to_guess_between_sources(monkeypatch) -> None:
    """An ambiguous name must be an error, not an arbitrary pick."""
    from netgamut import netzschleuder, snap

    monkeypatch.setattr(snap, "catalog", lambda **_: {"shared"})
    monkeypatch.setattr(netzschleuder, "catalog", lambda **_: {"shared"})
    with pytest.raises(KeyError, match="several sources"):
        netgamut.load("shared")


# -- node metadata across sources -----------------------------------------------
def test_node_metadata_filter_leaves_out_the_unknown() -> None:
    with_metadata = netgamut.search(node_metadata=True)
    without = netgamut.search(node_metadata=False)
    assert all(match.node_metadata is True for match in with_metadata)
    assert all(match.node_metadata is False for match in without)
    # AHORN keeps node lines inside the dataset file, so it answers neither way.
    assert not any(match.source == "ahorn" for match in with_metadata + without)
    assert {match.source for match in with_metadata} >= {"snap", "netzschleuder", "konect"}


def test_trait_filters_leave_out_the_unknown() -> None:
    for name in netgamut._TRAITS:
        yes = netgamut.search(**{name: True})
        no = netgamut.search(**{name: False})
        assert yes and no, name
        assert all(getattr(match, name) is True for match in yes), name
        assert all(getattr(match, name) is False for match in no), name
    # Only Netzschleuder's catalog says whether a network is multilayer.
    assert {match.source for match in netgamut.search(multilayer=False)} == {"netzschleuder"}
    # SNAP's type lists a graph's properties, and says nothing for its other collections.
    signed = {match.key for match in netgamut.search(signed=True, source="snap")}
    assert "snap/soc-sign-bitcoin-alpha" in signed
    assert netgamut.search("amazon-meta", source="snap")[0].weighted is None
    # Netzschleuder tags a network Weighted or Unweighted, and a few neither.
    assert netgamut.search("chess", source="netzschleuder")[0].weighted is None


def test_unknown_keeps_what_a_filter_cannot_answer() -> None:
    unweighted = {match.key for match in netgamut.search(weighted=False)}
    either = netgamut.search(weighted=False, unknown=True)
    assert all(match.weighted is not True for match in either)
    assert unweighted < {match.key for match in either}
    chess = netgamut.search("chess", source="netzschleuder")[0]
    assert chess.key in {match.key for match in either}
    # A size bound keeps the datasets whose source publishes no size.
    small = netgamut.search(max_nodes=1_000, unknown=True)
    assert any(match.dataset.nodes is None for match in small)
    assert all(match.dataset.nodes is None or match.dataset.nodes <= 1_000 for match in small)
    # And a column filter those whose catalog cannot name their columns.
    gender = netgamut.search(node_attribute="gender", unknown=True)
    assert any(match.node_attributes is None for match in gender)
    assert any(match.node_attributes for match in gender)


def test_node_attribute_finds_a_named_column_in_any_source() -> None:
    keys = {match.key for match in netgamut.search(node_attribute="Department")}
    assert "snap/email-Eu-core" in keys
    species = {match.source for match in netgamut.search(node_attribute="species")}
    assert {"konect", "ogb"} <= species
    assert all(
        "gender" in (match.node_attributes or ())
        for match in netgamut.search(node_attribute="gender")
    )


def test_node_attributes_leave_out_bookkeeping_columns() -> None:
    """Netzschleuder's index and _pos layout are not attributes."""
    karate = netgamut.netzschleuder.info("karate")
    assert karate.node_attributes == ("name", "groups")
    assert netgamut.snap.info("email-Eu-core").node_attributes == ("department",)
    assert netgamut.ogb.info("ogbn-arxiv").node_attributes == ("year",)


def test_sources_that_cannot_name_columns_still_answer_whether() -> None:
    labelled = next(item for item in netgamut.pyg.datasets() if item.num_classes)
    assert labelled.node_metadata is True and labelled.node_attributes == ()
    assert all(item.node_metadata is False for item in netgamut.paths.datasets())
