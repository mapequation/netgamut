"""Catalog lookup, filtering, and the vendored snapshot itself."""

from __future__ import annotations

import json

import pytest

from networks import snap
from networks._core.models import DataFile
from networks.snap._catalog import Catalog, load_catalog
from networks.snap._models import Category, Dataset


def build(*datasets: Dataset, categories: tuple[Category, ...] = ()) -> Catalog:
    return Catalog(categories=categories, datasets=datasets)


def dataset(name: str, **overrides) -> Dataset:
    fields = {
        "name": name,
        "categories": ("socnets",),
        "category_names": ("Social networks",),
        "columns": {},
    }
    fields.update(overrides)
    return Dataset(**fields)


# -- lookup ---------------------------------------------------------------
def test_lookup_is_case_insensitive() -> None:
    catalog = build(dataset("ca-GrQc"))
    assert catalog.dataset("CA-GRQC").name == "ca-GrQc"
    assert "ca-grqc" in catalog


def test_unknown_dataset_suggests_a_near_match() -> None:
    catalog = build(dataset("ca-GrQc"))
    with pytest.raises(KeyError, match="Did you mean: ca-GrQc"):
        catalog.dataset("ca-GrQx")


def test_category_reachable_by_id_heading_and_toc_label() -> None:
    category = Category(
        id="amazon",
        name="Product co-purchasing networks",
        toc_name="Amazon networks",
        dataset_names=("amazon0302",),
    )
    catalog = build(dataset("amazon0302", categories=("amazon",)), categories=(category,))
    for key in ("amazon", "Product co-purchasing networks", "Amazon networks"):
        assert catalog.category(key).id == "amazon"


def test_cross_listed_dataset_appears_in_each_category() -> None:
    categories = (
        Category(id="socnets", name="Social networks"),
        Category(id="temporal", name="Temporal networks"),
    )
    catalog = build(
        dataset("soc-Reddit", categories=("socnets", "temporal")), categories=categories
    )
    assert catalog.datasets_in("socnets")[0].name == "soc-Reddit"
    assert catalog.datasets_in("temporal")[0].name == "soc-Reddit"


# -- search ---------------------------------------------------------------
def test_search_matches_name_and_description() -> None:
    catalog = build(
        dataset("ca-GrQc", description="Arxiv General Relativity"),
        dataset("web-Google", description="Web graph from Google"),
    )
    assert [item.name for item in catalog.search("relativity")] == ["ca-GrQc"]
    assert [item.name for item in catalog.search("google")] == ["web-Google"]


def test_search_filters_by_size() -> None:
    catalog = build(
        dataset("small", columns={"Edges": "100"}),
        dataset("large", columns={"Edges": "1,000,000"}),
    )
    assert [item.name for item in catalog.search(max_edges=1000)] == ["small"]
    assert [item.name for item in catalog.search(min_edges=1000)] == ["large"]


def test_size_filters_exclude_datasets_with_no_reported_size() -> None:
    """A dataset SNAP gives no edge count for cannot be claimed to match."""
    catalog = build(dataset("unknown", columns={"Number of graphs": "6"}))
    assert catalog.search(max_edges=10**12) == ()
    assert catalog.search() != ()


def test_search_filters_by_ground_truth() -> None:
    catalog = build(
        dataset("com-DBLP", files=(DataFile("com-dblp.top5000.cmty.txt.gz", "u"),)),
        dataset("ca-GrQc", files=(DataFile("ca-GrQc.txt.gz", "u"),)),
    )
    assert [item.name for item in catalog.search(ground_truth=True)] == ["com-DBLP"]
    assert [item.name for item in catalog.search(ground_truth=False)] == ["ca-GrQc"]


def test_search_filters_by_direction() -> None:
    catalog = build(dataset("a", type="Undirected"), dataset("b", type="Directed"))
    assert [item.name for item in catalog.search(directed=False)] == ["a"]


# -- persistence ----------------------------------------------------------
def test_catalog_round_trips_through_json(tmp_path) -> None:
    catalog = build(
        dataset("ca-GrQc"), categories=(Category(id="socnets", name="Social networks"),)
    )
    path = catalog.write(tmp_path / "catalog.json")
    restored = Catalog.from_dict(json.loads(path.read_text()))
    assert restored.datasets == catalog.datasets
    assert restored.categories == catalog.categories


def test_future_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported catalog schema"):
        Catalog.from_dict({"schema": 999, "categories": [], "datasets": []})


def test_cached_catalog_overrides_the_vendored_one(isolated_cache) -> None:
    replacement = build(dataset("only-me"))
    replacement.write(isolated_cache / "snap" / "catalog.json")
    catalog = load_catalog(reload=True)
    assert catalog.origin == "cache"
    assert [item.name for item in catalog.datasets] == ["only-me"]


def test_corrupt_cached_catalog_falls_back_to_the_package(isolated_cache) -> None:
    """A bad cache file must never make the package unusable."""
    target = isolated_cache / "snap" / "catalog.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{ not json")
    catalog = load_catalog(reload=True)
    assert catalog.origin == "package"
    assert len(catalog.datasets) > 100


# -- the shipped snapshot -------------------------------------------------
def test_vendored_catalog_covers_the_collection() -> None:
    catalog = load_catalog(reload=True)
    assert catalog.origin == "package"
    assert len(catalog.categories) >= 20
    assert len(catalog.datasets) >= 100


def test_vendored_catalog_has_no_duplicate_names() -> None:
    names = [item.name for item in load_catalog(reload=True).datasets]
    assert len(names) == len(set(names))


def test_every_vendored_dataset_is_well_formed() -> None:
    for item in load_catalog(reload=True).datasets:
        assert item.name
        assert item.categories, f"{item.name} has no category"
        assert len(item.categories) == len(item.category_names)
        assert item.url.startswith("https://snap.stanford.edu/data/")


def test_every_category_resolves_and_is_non_empty() -> None:
    catalog = load_catalog(reload=True)
    for category in catalog.categories:
        assert catalog.datasets_in(category.id), f"{category.name} is empty"


def test_expected_categories_are_present() -> None:
    names = {category.name for category in load_catalog(reload=True).categories}
    assert {
        "Social networks",
        "Networks with ground-truth communities",
        "Collaboration networks",
        "Temporal networks",
        "Web graphs",
    } <= names


def test_known_dataset_matches_snap() -> None:
    grqc = snap.info("ca-GrQc")
    assert grqc.nodes == 5242
    assert grqc.edges == 14496
    assert grqc.directed is False
    assert grqc.category_name == "Collaboration networks"
    assert grqc.primary_file is not None
    assert grqc.primary_file.name == "ca-GrQc.txt.gz"


def test_ground_truth_datasets_are_discoverable() -> None:
    names = {item.name for item in snap.search(ground_truth=True)}
    assert {"com-DBLP", "com-Amazon", "com-Youtube"} <= names


def test_describe_reports_metadata_without_downloading() -> None:
    text = snap.describe("ca-GrQc")
    assert "ca-GrQc" in text
    assert "Collaboration networks" in text
    assert "Dataset statistics" in text
    assert not snap.is_cached("ca-GrQc")
