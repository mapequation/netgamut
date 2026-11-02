"""Parser tests, pinned to the quirks SNAP's real markup has."""

from __future__ import annotations

from netgamut.snap._scrape import parse_dataset_page, parse_index

BASE = "https://snap.stanford.edu/data/"


def test_categories_come_from_section_headings(index_html: str) -> None:
    categories, _ = parse_index(index_html, BASE)
    names = [category.name for category in categories]
    assert names == [
        "Social networks",
        "Temporal networks",
        "Product co-purchasing networks",
        "Online reviews",
    ]


def test_category_keeps_anchor_and_toc_gloss(index_html: str) -> None:
    categories, _ = parse_index(index_html, BASE)
    amazon = next(item for item in categories if item.id == "amazon")
    # SNAP's contents label and its section heading genuinely differ here.
    assert amazon.name == "Product co-purchasing networks"
    assert amazon.toc_name == "Amazon networks"
    assert amazon.description == "co-purchasing products"


def test_commented_out_sections_are_skipped(index_html: str) -> None:
    categories, datasets = parse_index(index_html, BASE)
    assert all(item.id != "hidden" for item in categories)
    assert all(item.name != "gone" for item in datasets)


def test_non_dataset_tables_are_ignored(index_html: str) -> None:
    """The trailing "Network types" glossary table is not a category."""
    categories, _ = parse_index(index_html, BASE)
    assert all(item.name != "Network types" for item in categories)


def test_cross_listed_dataset_becomes_one_record(index_html: str) -> None:
    _, datasets = parse_index(index_html, BASE)
    reddit = [item for item in datasets if item.name == "soc-Reddit"]
    assert len(reddit) == 1
    assert reddit[0].categories == ("socnets", "temporal")
    assert reddit[0].category_names == ("Social networks", "Temporal networks")


def test_cross_listing_merges_column_schemas(index_html: str) -> None:
    """The social table reports Edges; the temporal one adds the split counts."""
    _, datasets = parse_index(index_html, BASE)
    reddit = next(item for item in datasets if item.name == "soc-Reddit")
    assert reddit.columns["Edges"] == "858,490"
    assert reddit.columns["Static Edges"] == "571,927"
    assert reddit.columns["Temporal Edges"] == "858,490"


def test_category_listing_keeps_every_row(index_html: str) -> None:
    """A cross-listed dataset still appears in each section that lists it."""
    categories, _ = parse_index(index_html, BASE)
    by_id = {item.id: item for item in categories}
    assert "soc-Reddit" in by_id["socnets"].dataset_names
    assert "soc-Reddit" in by_id["temporal"].dataset_names


def test_item_oriented_category_has_no_edges(index_html: str) -> None:
    _, datasets = parse_index(index_html, BASE)
    reviews = next(item for item in datasets if item.name == "web-Reviews")
    assert reviews.columns["Number of items"] == "1,586,614"
    assert reviews.edges is None
    assert reviews.nodes is None


def test_detail_urls_are_absolute(index_html: str) -> None:
    _, datasets = parse_index(index_html, BASE)
    assert all(item.url.startswith("https://snap.stanford.edu/data/") for item in datasets)


def test_statistics_are_parsed(detail_html: str) -> None:
    parsed = parse_dataset_page(detail_html, BASE + "ego-Facebook.html")
    statistics = parsed.statistics
    assert statistics["Nodes"] == "4039"
    assert statistics["Average clustering coefficient"] == "0.6055"
    # The table's own caption row must not become a statistic.
    assert "Dataset statistics" not in statistics


def test_files_carry_descriptions(detail_html: str) -> None:
    parsed = parse_dataset_page(detail_html, BASE + "ego-Facebook.html")
    files = {item.name: item for item in parsed.files}
    assert files["facebook.tar.gz"].description == "Facebook data (10 networks)"
    assert files["facebook.tar.gz"].url == BASE + "facebook.tar.gz"


def test_files_outside_tables_are_found(detail_html: str) -> None:
    """Several SNAP pages list downloads in a paragraph rather than a table."""
    parsed = parse_dataset_page(detail_html, BASE + "ego-Facebook.html")
    files = {item.name for item in parsed.files}
    assert "extra-2009.txt.gz" in files


def test_withdrawn_files_are_not_offered(detail_html: str) -> None:
    """SNAP disables downloads by commenting out the link, keeping the row."""
    parsed = parse_dataset_page(detail_html, BASE + "ego-Facebook.html")
    assert all(item.name != "withdrawn.txt.gz" for item in parsed.files)


def test_citations_are_parsed(detail_html: str) -> None:
    parsed = parse_dataset_page(detail_html, BASE + "ego-Facebook.html")
    assert parsed.citations == (
        "J. McAuley and J. Leskovec. Learning to Discover Social Circles. NIPS, 2012.",
    )
