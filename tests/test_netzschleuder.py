"""The Netzschleuder subpackage and its vendored catalog."""

from __future__ import annotations

import pytest

from netgamut import netzschleuder as nz
from netgamut._core.domains import DOMAINS
from netgamut.netzschleuder._catalog import Catalog, load_catalog
from netgamut.netzschleuder._models import Dataset


def make(name: str = "toy", **overrides) -> Dataset:
    fields = {"name": name}
    fields.update(overrides)
    return Dataset(**fields)


# -- metadata records -----------------------------------------------------
def test_domain_tags_are_separated_from_properties_and_subtypes() -> None:
    dataset = make(tags=("Biological", "Food web", "Weighted"))
    assert dataset.domains == ("Biological",)
    assert dataset.subtypes == ("Food web",)
    assert dataset.weighted is True
    assert dataset.multilayer is False


def test_online_offline_are_not_domains() -> None:
    """They qualify social networks rather than naming a domain."""
    dataset = make(tags=("Social", "Online"))
    assert dataset.domains == ("Social",)


def test_domains_are_reported_in_canonical_order() -> None:
    dataset = make(tags=("Economic", "Social"))
    assert dataset.domains == ("Social", "Economic")


def test_sizes_come_from_the_analyses_block() -> None:
    dataset = make(analyses={"num_vertices": 297, "num_edges": 2359, "is_directed": True})
    assert (dataset.nodes, dataset.edges, dataset.directed) == (297, 2359, True)


def test_missing_analyses_yield_none() -> None:
    dataset = make()
    assert dataset.nodes is None
    assert dataset.edges is None
    assert dataset.directed is None


def test_csv_bundle_is_the_default_download() -> None:
    """It needs nothing beyond the standard library, unlike .gt.zst."""
    dataset = make("celegansneural")
    assert dataset.primary_file is not None
    assert dataset.primary_file.name == "celegansneural.csv.zip"
    assert dataset.primary_file.url.endswith("/celegansneural.csv.zip")


def test_page_url_includes_the_net_path() -> None:
    # https://networks.skewed.de/celegansneural is a 404.
    assert make("celegansneural").url == "https://networks.skewed.de/net/celegansneural"


def test_other_formats_are_reachable_by_suffix() -> None:
    dataset = make("power")
    assert dataset.file("gml.zst").name == "power.gml.zst"
    with pytest.raises(KeyError):
        dataset.file("nope")


def test_round_trips_through_json() -> None:
    dataset = make(
        "power",
        tags=("Technological", "Power grid"),
        analyses={"num_vertices": 4941},
        subgraph_count=3,
        subgraphs=("a", "b"),
    )
    assert Dataset.from_dict(dataset.to_dict()) == dataset


def test_subgraph_preview_is_not_the_full_list() -> None:
    """openstreetmap has 90k+ subgraphs; only a preview is vendored."""
    dataset = make(subgraph_count=91040, subgraphs=("a", "b"))
    assert len(dataset.subgraphs) < dataset.subgraph_count


# -- catalog --------------------------------------------------------------
def test_vendored_catalog_covers_the_collection() -> None:
    catalog = load_catalog(reload=True)
    assert catalog.origin == "package"
    assert len(catalog.datasets) >= 250


def test_every_domain_is_represented() -> None:
    """The point of this source: SNAP cannot fill Biological or Transportation."""
    counts = nz.domains()
    for domain in DOMAINS:
        assert counts[domain] >= 5, f"{domain} has only {counts[domain]}"


def test_search_filters_by_domain() -> None:
    biological = nz.search(domain="Biological")
    assert len(biological) >= 20
    assert all("Biological" in item.domains for item in biological)


def test_search_filters_by_structural_property() -> None:
    multilayer = nz.search(multilayer=True)
    assert len(multilayer) >= 10
    assert all(item.multilayer for item in multilayer)


def test_search_filters_by_subtype() -> None:
    """Subtype matching is case-insensitive because the site is inconsistent.

    Most food webs are tagged "Food web" but messal_shale uses "Food Web".
    """
    found = nz.search(subtype="Food web")
    assert len(found) >= 5
    assert all(any(tag.casefold() == "food web" for tag in item.subtypes) for item in found)
    assert nz.search(subtype="FOOD WEB") == found


def test_search_combines_domain_and_size() -> None:
    small = nz.search(domain="Biological", max_nodes=1000)
    assert small
    assert all(item.nodes is not None and item.nodes <= 1000 for item in small)


def test_known_networks_are_present_with_expected_domains() -> None:
    assert "Biological" in nz.info("celegansneural").domains
    assert "Transportation" in nz.info("london_transport").domains
    assert "Technological" in nz.info("power").domains
    assert "Economic" in nz.info("fao_trade").domains


def test_unknown_network_suggests_a_near_match() -> None:
    with pytest.raises(KeyError, match="Did you mean"):
        nz.info("celegansneual")


def test_describe_reports_without_downloading() -> None:
    text = nz.describe("celegansneural")
    assert "Biological" in text
    assert "Reported measures" in text
    assert not nz.is_cached("celegansneural")


def test_tags_are_counted() -> None:
    tags = nz.tags()
    assert tags["Social"] > tags["Transportation"]


def test_future_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported catalog schema"):
        Catalog.from_dict({"schema": 999, "datasets": []})


def test_build_reads_single_and_multiple_upstream_urls(monkeypatch) -> None:
    """The API gives ``url`` as a string, or as a list for multi-publisher entries."""
    import json

    from netgamut.netzschleuder import _catalog

    pages = {
        "nets": ["one", "many"],
        "net/one": {"url": "https://a.example/one", "nets": ["one"]},
        "net/many": {"url": ["https://b.example/x", "https://c.example/y"], "nets": ["x", "y"]},
    }
    monkeypatch.setattr(
        _catalog,
        "fetch_text",
        lambda url: json.dumps(pages[url.removeprefix(f"{_catalog.API_URL}/")]),
    )
    catalog = _catalog.build_catalog(workers=1)
    assert catalog["one"].upstream_url == "https://a.example/one"
    assert catalog["many"].upstream_url == "https://b.example/x"


def test_cached_catalog_overrides_the_vendored_one(isolated_cache) -> None:
    Catalog(datasets=(make("only-me"),)).write(isolated_cache / "netzschleuder" / "catalog.json")
    catalog = load_catalog(reload=True)
    assert catalog.origin == "cache"
    assert [item.name for item in catalog.datasets] == ["only-me"]


# -- opt-in network tests -------------------------------------------------
@pytest.mark.network
def test_really_loads_a_csv_bundle() -> None:
    """The CSV path must work end to end without graph-tool or zstd."""
    network = nz.load("celegansneural", progress=False)
    assert network.delimiter == ","
    edges = list(network.edges(cast=int))
    assert len(edges) == 2359
    # The bundle's other members stay reachable despite edges.csv being "the"
    # network.
    assert network.is_bundle and not network.is_archive
    members = {item.name for item in network.extract().iterdir()}
    assert {"edges.csv", "nodes.csv"} <= members


@pytest.mark.network
def test_live_api_still_matches_the_vendored_shape() -> None:
    """Fails loudly if Netzschleuder changes its API."""
    import json

    from netgamut._core.download import fetch_text
    from netgamut.netzschleuder._catalog import API_URL

    names = json.loads(fetch_text(f"{API_URL}/nets"))
    assert len(names) >= 250
    payload = json.loads(fetch_text(f"{API_URL}/net/celegansneural"))
    assert "Biological" in payload["tags"]
    assert payload["analyses"]["num_vertices"] == 297
