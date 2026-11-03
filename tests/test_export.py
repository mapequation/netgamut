"""The web mirror: gzipped Pajek files and the manifest the explorer and Navigator read."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import netgamut
from netgamut import __main__ as cli
from netgamut import _export
from netgamut._core import formats


@dataclass
class FakeDataset:
    name: str
    nodes: int | None
    edges: int | None
    directed: bool = False
    duplicate_of: str | None = None
    type: str = "Undirected"
    fails: bool = False


@dataclass
class FakeNetwork:
    dataset: FakeDataset

    def to_pajek(self, directory: str | Path, *, name: str) -> Path:
        if self.dataset.fails:
            raise OSError("upstream is down")
        count = self.dataset.edges or 0
        return formats.write_pajek(
            formats.output_path(directory, name, ".net"),
            nodes=range(1, count + 2),
            links=lambda: ((i, i + 1, None) for i in range(1, count + 1)),
            directed=self.dataset.directed,
            comments=[name],
        )


@dataclass
class FakeMatch:
    source: str
    dataset: FakeDataset
    loads: list[dict[str, Any]] = field(default_factory=list)

    name = property(lambda self: self.dataset.name)
    nodes = property(lambda self: self.dataset.nodes)
    edges = property(lambda self: self.dataset.edges)
    key = property(lambda self: f"{self.source}/{self.name}")

    def load(self, **kwargs: Any) -> FakeNetwork:
        self.loads.append(kwargs)
        return FakeNetwork(self.dataset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "title": "",
            "url": f"https://example.org/{self.name}",
            "citations": [{"text": "A. Author,\nA network (2020).", "url": "https://doi.org/x"}],
        }


@pytest.fixture
def catalog(monkeypatch):
    sources = {
        "konect": [
            FakeMatch("konect", FakeDataset("large", 50, 40)),
            FakeMatch("konect", FakeDataset("copy", 3, 2, duplicate_of="snap/tiny")),
            FakeMatch("konect", FakeDataset("unsized", None, None)),
        ],
        "snap": [
            FakeMatch("snap", FakeDataset("tiny", 3, 2, directed=True, type="Directed")),
            FakeMatch("snap", FakeDataset("middle", 11, 10)),
            FakeMatch("snap", FakeDataset("reviews", 5, 4, type="Beer reviews")),
        ],
    }
    monkeypatch.setattr(netgamut, "search", lambda source: sources.get(source, []))
    return sources


def _keys(document: dict[str, Any]) -> list[str]:
    return [entry["key"] for entry in document["datasets"]]


def test_writes_gzipped_pajek_with_attribution(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path)

    assert _keys(document) == ["konect/large", "snap/middle", "snap/tiny"]
    tiny = document["datasets"][2]
    # Counts and direction come from the written file, not the catalog.
    assert (tiny["nodes"], tiny["edges"], tiny["directed"]) == (3, 2, True)
    assert tiny["file"] == "snap/tiny.net.gz"
    assert tiny["bytes"] == (tmp_path / tiny["file"]).stat().st_size
    assert document["bytes"] == sum(entry["bytes"] for entry in document["datasets"])
    text = gzip.decompress((tmp_path / tiny["file"]).read_bytes()).decode()
    assert text.startswith("# snap/tiny: tiny\n# From https://example.org/tiny\n")
    # A newline inside a citation must not end the comment line.
    assert "# Cite: A. Author, A network (2020). https://doi.org/x\n" in text
    assert "*Arcs\n1 2\n2 3\n" in text
    assert json.loads((tmp_path / "index.json").read_text()) == document


def test_keeps_the_source_ids_and_says_where_they_run(tmp_path) -> None:
    # Ambassador's 1985-1989 snapshot: 11 of its 16 actors have links, and they
    # keep the ids they have in every other snapshot.
    pajek = tmp_path / "ambassador.net"
    pajek.write_text('# ambassador\n*Vertices 3\n0 "0"\n4 "4"\n14 "14"\n*Edges\n0 4\n4 14\n')
    assert _export._id_note(pajek) == [
        "Node ids are the source's own, so they can start at 0 and skip numbers: "
        "3 nodes, ids 0 to 14."
    ]
    packed = tmp_path / "ambassador.net.gz"
    _export._compress(pajek, packed, _export._id_note(pajek))
    assert "\n0 4\n4 14\n" in gzip.decompress(packed.read_bytes()).decode()


def test_leaves_out_copies_non_graphs_and_unknown_sizes(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path)
    assert {entry["key"] for entry in document["skipped"]} == {"konect/unsized"}
    assert catalog["konect"][1].loads == []
    assert catalog["snap"][2].loads == []


def test_passes_the_download_limit(tmp_path, catalog) -> None:
    _export.export_networks(tmp_path, sources=("snap",), max_download=123)
    assert catalog["snap"][0].loads == [{"progress": False, "max_bytes": 123}]


def test_stops_at_the_budget_and_deletes_what_no_longer_fits(tmp_path, catalog) -> None:
    whole = _export.export_networks(tmp_path, sources=("snap",))
    tiny = whole["datasets"][1]
    document = _export.export_networks(tmp_path, sources=("snap",), budget=tiny["bytes"])
    assert _keys(document) == ["snap/tiny"]
    assert document["skipped"] == [{"key": "snap/middle", "reason": "over budget"}]
    assert not (tmp_path / "snap" / "middle.net.gz").exists()


def test_rejects_a_file_over_the_limit_and_remembers_it(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path, sources=("snap",), max_file=150)
    middle = next(entry for entry in document["skipped"] if entry["key"] == "snap/middle")
    assert middle["reason"].startswith("too large: ")
    catalog["snap"][1].loads.clear()

    again = _export.export_networks(tmp_path, sources=("snap",), max_file=150)
    middle = next(entry for entry in again["skipped"] if entry["key"] == "snap/middle")
    assert middle["reason"] == "too large on an earlier build"
    assert catalog["snap"][1].loads == []


def test_an_unchanged_network_is_not_downloaded_again(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    tiny = catalog["snap"][0]
    tiny.loads.clear()

    again = _export.export_networks(tmp_path, sources=("snap",))
    assert again["datasets"] == first["datasets"]
    assert tiny.loads == []

    _export.export_networks(tmp_path, sources=("snap",), force=True)
    assert len(tiny.loads) == 1


def test_a_changed_network_is_rebuilt(tmp_path, catalog) -> None:
    _export.export_networks(tmp_path, sources=("snap",))
    catalog["snap"][1].dataset.edges = 12
    document = _export.export_networks(tmp_path, sources=("snap",))
    middle = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert middle["edges"] == 12


def test_a_failed_rebuild_keeps_the_previous_file(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    catalog["snap"][1].dataset.fails = True  # also changes its fingerprint
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert document["datasets"] == first["datasets"]
    assert (tmp_path / "snap" / "middle.net.gz").is_file()


def test_a_failed_network_without_a_file_is_skipped(tmp_path, catalog) -> None:
    catalog["snap"][1].dataset.fails = True
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["snap/tiny"]
    (middle,) = document["skipped"]
    assert (middle["key"], middle["reason"]) == ("snap/middle", "OSError: upstream is down")


def test_one_source_at_a_time_keeps_the_others(tmp_path, catalog) -> None:
    _export.export_networks(tmp_path)
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["konect/large", "snap/middle", "snap/tiny"]
    assert {entry["key"] for entry in document["skipped"]} == {"konect/unsized"}


def test_cli_export_networks(tmp_path, catalog, capsys) -> None:
    target = tmp_path / "mirror"
    assert cli.main(["export-networks", "--output", str(target), "--source", "snap"]) == 0
    assert "2 networks" in capsys.readouterr().err
    assert len(json.loads((target / "index.json").read_text())["datasets"]) == 2
