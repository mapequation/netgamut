"""NodeTable: the one shape every source's node metadata arrives in.

These are pure unit tests over hand-built tables -- no downloads, no sources -- so the
contract is pinned independently of who produces it. What matters is that
``partition`` and ``categories`` agree, that decoding is applied consistently, and
that ordering is natural rather than lexicographic, since the category order is what
numbers the modules.
"""

from __future__ import annotations

import pytest

from netgamut._core.nodes import NodeTable


def table() -> NodeTable:
    return NodeTable(
        values={"dept": {0: 2, 1: 10, 2: 2}, "year": {0: 2013, 1: 2015, 2: 2013}},
        names={"dept": {2: "physics", 10: "chemistry"}},
        origin="test",
    )


def test_values_are_decoded_where_the_source_names_the_codes() -> None:
    """A named column reads out as names; an unnamed one keeps its raw values."""
    nodes = table()
    assert nodes["dept"] == {0: "physics", 1: "chemistry", 2: "physics"}
    assert nodes["year"] == {0: 2013, 1: 2015, 2: 2013}


def test_partition_and_categories_are_two_halves_of_one_answer() -> None:
    """``categories(c)[partition(c)[node]]`` must be that node's value, always.

    This is the property that lets a caller score a partition and then say which
    category each module lines up with. If the two used different orders the labels
    on a figure would be silently wrong.
    """
    nodes = table()
    for column in nodes.columns:
        categories, partition = nodes.categories(column), nodes.partition(column)
        assert set(partition.values()) == set(range(len(categories)))
        for node, value in nodes[column].items():
            assert categories[partition[node]] == value


def test_categories_order_numerically_not_lexicographically() -> None:
    """Sources write codes as bare text, and plain sorted() puts 10 before 2.

    The order numbers the modules, so a lexicographic sort would make ``partition``
    disagree with anyone reading the source file.
    """
    nodes = NodeTable(values={"c": {i: text for i, text in enumerate(["10", "2", "1"])}})
    assert nodes.categories("c") == ("1", "2", "10")
    assert nodes.partition("c") == {0: 2, 1: 1, 2: 0}


def test_mixed_value_types_order_numerically_and_do_not_raise() -> None:
    """One column may mix ints and strings; plain sorted() would raise TypeError.

    Numbers and numeric text interleave on value -- 1, 3, "10" -- and non-numeric
    text sorts after them, so a column that is numeric apart from a stray "unknown"
    still orders sensibly.
    """
    nodes = NodeTable(values={"c": {0: 3, 1: "apple", 2: "10", 3: 1}})
    assert nodes.categories("c") == (1, 3, "10", "apple")


def test_a_node_missing_a_value_is_absent_rather_than_pooled() -> None:
    """Silence is not a category: an unlabelled node gets no module, not a shared one.

    Pooling them would invent a community the source never claimed, and it would show
    up as a real module in any comparison.
    """
    nodes = NodeTable(values={"c": {0: "a", 2: "b"}})
    assert nodes.partition("c") == {0: 0, 2: 1}
    assert 1 not in nodes.partition("c")


def test_unknown_column_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="Available: dept, year"):
        table()["missing"]


def test_columns_nodes_and_membership() -> None:
    nodes = table()
    assert nodes.columns == ("dept", "year")
    assert nodes.nodes == (0, 1, 2)
    assert "dept" in nodes
    assert len(nodes) == 2
    assert list(nodes) == ["dept", "year"]


def test_select_keeps_names_and_origin() -> None:
    narrowed = table().select(["dept"])
    assert narrowed.columns == ("dept",)
    assert narrowed["dept"][0] == "physics"
    assert narrowed.origin == "test"


def test_rows_are_decoded_and_one_per_node() -> None:
    rows = table().rows()
    assert rows[0] == {"node": 0, "dept": "physics", "year": 2013}
    assert len(rows) == 3


def test_repr_shows_the_categorical_widths() -> None:
    """The repr is the discovery path: it has to say what a table carries."""
    assert repr(table()) == "NodeTable(3 nodes from test: dept[2], year)"
