"""The paths -> state network conversion in netgamut._core.state_network."""

from __future__ import annotations

from pathlib import Path

import pytest

from netgamut import StateNetwork


def test_trigrams_become_state_links() -> None:
    net = StateNetwork.from_paths([["a", "b", "c", "d"]])
    assert net.order == 2
    assert net.num_physical_nodes == 4
    # Windows (a,b), (b,c), (c,d) in first-seen order.
    assert net.num_state_nodes == 3
    assert net.links == {(1, 2): 1.0, (2, 3): 1.0}
    assert net.state_name(1) == "a -> b"
    # State (a,b) lives at physical node b.
    assert net.states[1][-1] == 2


def test_weights_accumulate_across_paths() -> None:
    net = StateNetwork.from_paths(
        [
            ["a", "b", "c"],
            ["a", "b", "c"],
            (["a", "b", "c"], 3.0),
        ]
    )
    assert net.num_links == 1
    assert net.total_weight == 5.0


def test_short_paths_are_counted_not_silently_dropped() -> None:
    net = StateNetwork.from_paths([["a", "b"], ["a", "b", "c"]])
    assert net.num_paths == 2
    assert net.num_short_paths == 1
    # The short path contributes no states at all, not even a dangling one.
    assert net.num_state_nodes == 2


def test_first_order_recovers_the_plain_directed_network() -> None:
    net = StateNetwork.from_paths([["a", "b", "c"]], order=1)
    assert net.num_state_nodes == 3
    assert net.links == {(1, 2): 1.0, (2, 3): 1.0}
    assert net.state_name(1) == "a"


def test_third_order_uses_four_grams() -> None:
    net = StateNetwork.from_paths([["a", "b", "c", "d"]], order=3)
    assert net.num_state_nodes == 2
    assert net.num_links == 1
    assert net.state_name(1) == "a -> b -> c"


def test_order_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="order"):
        StateNetwork.from_paths([], order=0)


def test_integer_labels_are_accepted() -> None:
    net = StateNetwork.from_paths([[1, 2, 3]])
    assert net.physical_nodes == {1: "1", 2: "2", 3: "3"}


def test_numeric_weight_types_beyond_float_are_weights() -> None:
    # pandas group-bys hand back numpy scalars, which are numbers.Real but
    # not int/float subclasses; anything Real must count as a weight.
    from fractions import Fraction

    net = StateNetwork.from_paths([(["a", "b", "c"], Fraction(3, 2))])
    assert net.num_short_paths == 0
    assert net.total_weight == 1.5


def test_any_sequence_can_carry_a_weight() -> None:
    net = StateNetwork.from_paths([(range(3), 2.0)])
    assert net.total_weight == 2.0
    assert net.physical_nodes == {1: "0", 2: "1", 3: "2"}


def test_frozen_networks_are_hashable() -> None:
    trips = [["a", "b", "c"]]
    first, second = StateNetwork.from_paths(trips), StateNetwork.from_paths(trips)
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_whitespace_in_labels_cannot_break_the_file(tmp_path: Path) -> None:
    net = StateNetwork.from_paths([["a\nnewline", "b\ttab", "c"]])
    text = net.write(tmp_path / "states.net").read_text(encoding="utf-8")
    assert '"a newline"' in text
    assert '"b tab"' in text
    # Every name stays on its own line: sections + one line per record.
    assert len(text.splitlines()) == 1 + 4 + 4 + 2


def test_written_file_is_the_infomap_states_format(tmp_path: Path) -> None:
    net = StateNetwork.from_paths([["a", "b", "c"]])
    target = net.write(tmp_path / "states.net")
    assert target.read_text(encoding="utf-8") == (
        "# State network with Markov order 2 built from 1 paths\n"
        "*Vertices 3\n"
        '1 "a"\n'
        '2 "b"\n'
        '3 "c"\n'
        "*States\n"
        "# state_id physical_id name\n"
        '1 2 "a -> b"\n'
        '2 3 "b -> c"\n'
        "*Links\n"
        "1 2 1.0\n"
    )


def test_written_file_is_deterministic(tmp_path: Path) -> None:
    trips = [["x", "y", "z"], (["y", "z", "x"], 2.0), ["x", "y", "z", "x"]]
    first = StateNetwork.from_paths(trips).write(tmp_path / "a.net").read_text()
    second = StateNetwork.from_paths(trips).write(tmp_path / "b.net").read_text()
    assert first == second


def test_quotes_in_labels_cannot_break_the_file(tmp_path: Path) -> None:
    net = StateNetwork.from_paths([['say_"hi"', "b", "c"]])
    text = net.write(tmp_path / "states.net").read_text(encoding="utf-8")
    assert "\"say_'hi'\"" in text


def test_to_infomap_runs_the_memory_model() -> None:
    pytest.importorskip("infomap")
    # Two return flows through the hub j that keep their identity: with
    # memory they form two modules that overlap on j; a first-order network
    # of the same trips cannot separate them.
    trips = [["a", "j", "b", "j", "a", "j", "b"]] * 5 + [["c", "j", "d", "j", "c", "j", "d"]] * 5
    net = StateNetwork.from_paths(trips)
    assert net.num_state_nodes == 8

    result = net.to_infomap(silent=True, num_trials=5, seed=42).run()
    assert result.have_memory
    assert result.num_top_modules == 2
