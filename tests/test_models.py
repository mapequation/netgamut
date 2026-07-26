"""Behaviour of the metadata records themselves."""

from __future__ import annotations

import pytest

from networks.snap._models import Category, DataFile, Dataset
from networks.snap._scrape import parse_dataset_page, parse_index

BASE = "https://snap.stanford.edu/data/"


def make(**overrides) -> Dataset:
    fields = {
        "name": "toy",
        "categories": ("socnets",),
        "category_names": ("Social networks",),
        "columns": {},
    }
    fields.update(overrides)
    return Dataset(**fields)


def test_counts_strip_thousands_separators() -> None:
    dataset = make(columns={"Nodes": "1,134,890", "Edges": "2,987,624"})
    assert dataset.nodes == 1_134_890
    assert dataset.edges == 2_987_624


def test_missing_counts_are_none() -> None:
    assert make(columns={"Number of graphs": "6"}).nodes is None
    assert make(columns={"Nodes": "-"}).nodes is None


def test_static_edges_preferred_for_temporal_tables() -> None:
    dataset = make(columns={"Temporal Edges": "858,490", "Static Edges": "571,927"})
    assert dataset.edges == 571_927


def test_statistics_are_a_fallback_for_counts() -> None:
    dataset = make(columns={}, statistics={"Nodes": "4039", "Edges": "88234"})
    assert (dataset.nodes, dataset.edges) == (4039, 88234)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("Undirected", False),
        ("Directed", True),
        ("Weighted, Directed, Temporal", True),
        ("Bipartite, Undirected", False),
        (None, None),
        ("Reviews", None),
    ],
)
def test_direction_is_read_from_the_type_column(declared, expected) -> None:
    assert make(type=declared).directed is expected


def test_primary_file_prefers_the_named_edge_list() -> None:
    dataset = make(
        name="ca-GrQc",
        files=(
            DataFile("readme.txt", "u/readme.txt"),
            DataFile("other.txt.gz", "u/other.txt.gz"),
            DataFile("ca-GrQc.txt.gz", "u/ca-GrQc.txt.gz"),
        ),
    )
    assert dataset.primary_file is not None
    assert dataset.primary_file.name == "ca-GrQc.txt.gz"


def test_primary_file_never_picks_a_readme() -> None:
    dataset = make(files=(DataFile("readme-Ego.txt", "u"), DataFile("data.tar.gz", "u")))
    assert dataset.primary_file is not None
    assert dataset.primary_file.name == "data.tar.gz"


def test_primary_file_is_none_when_withdrawn() -> None:
    assert make(files=()).primary_file is None


def test_category_properties_expose_the_first_listing() -> None:
    dataset = make(
        categories=("socnets", "temporal"),
        category_names=("Social networks", "Temporal networks"),
    )
    assert dataset.category == "socnets"
    assert dataset.category_name == "Social networks"


def test_ground_truth_detection(ground_truth_html: str) -> None:
    parsed = parse_dataset_page(ground_truth_html, BASE + "com-DBLP.html")
    dataset = make(name="com-DBLP", files=parsed.files)
    assert dataset.has_ground_truth
    assert dataset.community_file("top5000").name == "com-dblp.top5000.cmty.txt.gz"
    assert dataset.community_file("all").name == "com-dblp.all.cmty.txt.gz"
    # The network itself must not be mistaken for a community file.
    assert dataset.primary_file is not None
    assert dataset.primary_file.name == "com-dblp.ungraph.txt.gz"


def test_label_style_ground_truth_is_recognised() -> None:
    """email-Eu-core ships a node-label partition, not community lists."""
    dataset = make(
        name="email-Eu-core",
        files=(
            DataFile("email-Eu-core.txt.gz", "u"),
            DataFile("email-Eu-core-department-labels.txt.gz", "u"),
        ),
    )
    assert dataset.has_ground_truth
    assert dataset.ground_truth_kind == "labels"
    assert dataset.community_files == ()
    assert dataset.community_file().name == "email-Eu-core-department-labels.txt.gz"
    # The network itself must still be the default download.
    assert dataset.primary_file is not None
    assert dataset.primary_file.name == "email-Eu-core.txt.gz"


def test_ground_truth_kind_distinguishes_the_two_shapes() -> None:
    assert make(files=(DataFile("x.top5000.cmty.txt.gz", "u"),)).ground_truth_kind == (
        "communities"
    )
    assert make(files=(DataFile("x-labels.txt.gz", "u"),)).ground_truth_kind == "labels"
    assert make(files=(DataFile("x.txt.gz", "u"),)).ground_truth_kind is None


def test_ground_truth_absent_raises_lookup_error() -> None:
    with pytest.raises(LookupError):
        make().community_file()


def test_unknown_file_lists_the_alternatives() -> None:
    dataset = make(files=(DataFile("a.txt.gz", "u"),))
    with pytest.raises(KeyError, match="a.txt.gz"):
        dataset.file("nope.txt.gz")


def test_dataset_round_trips_through_json(index_html: str) -> None:
    _, datasets = parse_index(index_html, BASE)
    original = next(item for item in datasets if item.name == "soc-Reddit")
    assert Dataset.from_dict(original.to_dict()) == original


def test_category_round_trips_through_json(index_html: str) -> None:
    categories, _ = parse_index(index_html, BASE)
    assert Category.from_dict(categories[0].to_dict()) == categories[0]
