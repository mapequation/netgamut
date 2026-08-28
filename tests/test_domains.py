"""The normalised domain vocabulary and the cross-source API."""

from __future__ import annotations

import pytest

import networks
from networks._core.domains import (
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
    assert networks.sources() == ("snap", "netzschleuder", "ogb", "pyg", "paths")


def test_domain_counts_sum_across_sources() -> None:
    counts = networks.domains()
    assert set(counts) == set(DOMAINS)
    # SNAP contributes no biological networks at all, so this comes entirely
    # from Netzschleuder -- the reason the second source exists.
    assert counts["Biological"] >= 30
    assert counts["Social"] > counts["Transportation"]


def test_snap_has_no_biological_networks() -> None:
    """Documents the gap that motivated adding a second source."""
    from networks import snap

    assert not [item for item in snap.datasets() if "Biological" in item.domains]


def test_cross_source_search_reaches_every_source() -> None:
    """One query, one vocabulary, whichever catalog the network came from.

    Transportation is the domain every source happens to cover, so it is the one that
    proves the vocabulary is genuinely shared rather than per-source.
    """
    sources = {match.source for match in networks.search(domain="Transportation")}
    assert {"snap", "netzschleuder", "paths", "pyg"} <= sources

    # and each source can still be reached on its own terms
    for source in networks.sources():
        assert networks.search(source=source), f"{source} returned nothing"


def test_matches_are_uniquely_keyed() -> None:
    keys = [match.key for match in networks.search()]
    assert len(keys) == len(set(keys))
    assert all("/" in key for key in keys)


def test_search_can_be_restricted_to_one_source() -> None:
    found = networks.search(domain="Social", source="snap")
    assert found
    assert {match.source for match in found} == {"snap"}


def test_search_filters_by_size_across_sources() -> None:
    found = networks.search(max_nodes=500)
    assert found
    assert all(match.nodes is not None and match.nodes <= 500 for match in found)


def test_unknown_domain_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown domain"):
        networks.search(domain="Chemical")


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown source"):
        networks.search(source="konect")


def test_load_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="No dataset named"):
        networks.load("definitely-not-a-network")


def test_load_refuses_to_guess_between_sources(monkeypatch) -> None:
    """An ambiguous name must be an error, not an arbitrary pick."""
    from networks import netzschleuder, snap

    monkeypatch.setattr(snap, "catalog", lambda **_: {"shared"})
    monkeypatch.setattr(netzschleuder, "catalog", lambda **_: {"shared"})
    with pytest.raises(KeyError, match="several sources"):
        networks.load("shared")
