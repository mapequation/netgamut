"""The web mirror: gzipped Pajek files and the manifest the explorer and Navigator read."""

from __future__ import annotations

import gzip
import json
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import netgamut
from netgamut import __main__ as cli
from netgamut import _export
from netgamut._core import formats
from netgamut._core.download import DownloadError


@dataclass
class FakeDataset:
    name: str
    nodes: int | None
    edges: int | None
    directed: bool = False
    duplicate_of: str | None = None
    type: str = "Undirected"
    fails: bool = False
    #: Reads as a network with no nodes, as product_space once did.
    empty: bool = False
    #: Raised by the download, as the real loaders raise DownloadError.
    error: Exception | None = None


def download_error(cause: Exception) -> DownloadError:
    """A DownloadError chained to *cause*, as ``_core.download._open`` raises it."""
    try:
        raise DownloadError(f"Could not fetch http://example.org: {cause}") from cause
    except DownloadError as error:
        return error


@dataclass
class FakeNetwork:
    dataset: FakeDataset

    def to_pajek(self, directory: str | Path, *, name: str) -> Path:
        if self.dataset.fails:
            raise OSError("upstream is down")
        count = 0 if self.dataset.empty else self.dataset.edges or 0
        return formats.write_pajek(
            formats.output_path(directory, name, ".net"),
            nodes=range(1, count + 2) if count else (),
            links=lambda: ((i, i + 1, None) for i in range(1, count + 1)),
            directed=self.dataset.directed,
            comments=[name],
        )


@dataclass
class FakeMatch:
    source: str
    dataset: FakeDataset
    loads: list[dict[str, Any]] = field(default_factory=list)
    #: Match.weighted and the rest; derived, so not part of the fingerprint.
    kinds: dict[str, bool] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in netgamut._KINDS:
            return self.kinds.get(name)
        raise AttributeError(name)

    name = property(lambda self: self.dataset.name)
    nodes = property(lambda self: self.dataset.nodes)
    edges = property(lambda self: self.dataset.edges)
    key = property(lambda self: f"{self.source}/{self.name}")

    def load(self, **kwargs: Any) -> FakeNetwork:
        self.loads.append(kwargs)
        if self.dataset.error is not None:
            raise self.dataset.error
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


def test_a_new_file_format_rebuilds_that_source_alone(tmp_path, catalog, monkeypatch) -> None:
    _export.export_networks(tmp_path, sources=("konect", "snap"))
    large, tiny = catalog["konect"][0], catalog["snap"][0]
    large.loads.clear()
    tiny.loads.clear()
    monkeypatch.setitem(_export._FORMATS, "konect", _export._FORMATS["konect"] + 1)
    _export.export_networks(tmp_path, sources=("konect", "snap"))
    assert len(large.loads) == 1
    assert tiny.loads == []


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


def test_a_network_that_reads_as_empty_is_skipped(tmp_path, catalog) -> None:
    catalog["snap"][1].dataset.empty = True
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["snap/tiny"]
    (middle,) = document["skipped"]
    assert middle["key"] == "snap/middle"
    assert middle["reason"] == "empty: 0 nodes and 0 links read"
    assert not (tmp_path / "snap" / "middle.net.gz").exists()


def test_an_empty_file_from_an_earlier_build_is_built_again(tmp_path, catalog) -> None:
    """Unchanged in the catalog, but its file had nothing in it: not worth keeping."""
    _export.export_networks(tmp_path, sources=("snap",))
    manifest = tmp_path / "index.json"
    earlier = json.loads(manifest.read_text())
    for entry in earlier["datasets"]:
        if entry["key"] == "snap/middle":
            entry["nodes"] = entry["edges"] = 0
    manifest.write_text(json.dumps(earlier))
    middle = catalog["snap"][1]
    middle.loads.clear()
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert len(middle.loads) == 1
    rebuilt = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert (rebuilt["nodes"], rebuilt["edges"]) == (11, 10)


def test_one_source_at_a_time_keeps_the_others(tmp_path, catalog) -> None:
    _export.export_networks(tmp_path)
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["konect/large", "snap/middle", "snap/tiny"]
    assert {entry["key"] for entry in document["skipped"]} == {"konect/unsized"}


def test_starts_no_download_after_the_time_limit(tmp_path, catalog) -> None:
    stopped = _export.export_networks(tmp_path, sources=("snap",), stop_after=0)
    assert stopped["complete"] is False
    assert stopped["datasets"] == []
    assert stopped["skipped"] == [
        {"key": "snap/middle", "reason": "not built yet"},
        {"key": "snap/tiny", "reason": "not built yet"},
    ]
    assert catalog["snap"][0].loads == []

    finished = _export.export_networks(tmp_path, sources=("snap",))
    assert finished["complete"] is True
    assert _keys(finished) == ["snap/middle", "snap/tiny"]


def test_a_stopped_run_keeps_what_earlier_runs_built(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    catalog["snap"][1].dataset.edges = 12  # middle changed upstream
    stopped = _export.export_networks(tmp_path, sources=("snap",), stop_after=0)
    # Out of time: the stale file stays listed rather than vanishing.
    assert stopped["complete"] is False
    assert stopped["datasets"] == first["datasets"]
    assert (tmp_path / "snap" / "middle.net.gz").is_file()

    resumed = _export.export_networks(tmp_path, sources=("snap",))
    assert resumed["complete"] is True
    middle = next(entry for entry in resumed["datasets"] if entry["key"] == "snap/middle")
    assert middle["edges"] == 12


def test_an_idle_run_leaves_the_manifest_untouched(tmp_path, catalog) -> None:
    _export.export_networks(tmp_path, sources=("snap",))
    manifest = tmp_path / "index.json"
    earlier = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**earlier, "built": "2026-01-01T00:00:00+00:00"}))
    before = manifest.read_bytes()

    again = _export.export_networks(tmp_path, sources=("snap",))
    assert again["built"] == "2026-01-01T00:00:00+00:00"
    assert json.loads(manifest.read_bytes()) == json.loads(before)


def test_a_source_that_stops_answering_waits_for_a_later_run(tmp_path, catalog) -> None:
    tiny, middle = catalog["snap"][0], catalog["snap"][1]
    tiny.dataset.error = download_error(urllib.error.URLError("timed out"))
    document = _export.export_networks(tmp_path)

    # One host that does not answer is enough: middle is not tried at all.
    assert middle.loads == []
    assert _keys(document) == ["konect/large"]
    skipped = {entry["key"]: entry for entry in document["skipped"]}
    assert skipped["snap/tiny"]["reason"].startswith("DownloadError: Could not fetch")
    assert skipped["snap/middle"]["reason"] == "snap unreachable in this run"
    # Their estimated space is held, so building them later evicts nothing.
    assert document["reserved"] == (2 + 10) * 4
    assert document["bytes"] == document["datasets"][0]["bytes"]
    assert document["complete"] is True

    tiny.dataset.error = None
    again = _export.export_networks(tmp_path)
    assert _keys(again) == ["konect/large", "snap/middle", "snap/tiny"]
    assert again["reserved"] == 0


def test_a_missing_file_does_not_stop_its_source(tmp_path, catalog) -> None:
    gone = urllib.error.HTTPError("http://example.org", 404, "Not Found", None, None)  # type: ignore[arg-type]
    catalog["snap"][0].dataset.error = download_error(gone)
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["snap/middle"]
    assert document["reserved"] == 0


def test_an_unreachable_source_keeps_its_earlier_files(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    # A changed catalog entry asks for a rebuild, but the host does not answer.
    catalog["snap"][0].dataset.error = download_error(TimeoutError("timed out"))
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert document["datasets"] == first["datasets"]
    assert document["reserved"] == 0


def test_lists_the_kinds_of_network_from_the_catalog(tmp_path, catalog) -> None:
    """The Navigator filters on them; a file kept from an earlier run gains them too."""
    tiny = catalog["snap"][0]
    first = _export.export_networks(tmp_path)
    assert all(entry["temporal"] is None for entry in first["datasets"])

    tiny.kinds = {"temporal": True, "weighted": False}
    document = _export.export_networks(tmp_path)

    assert len(tiny.loads) == 1
    entry = next(entry for entry in document["datasets"] if entry["key"] == "snap/tiny")
    assert {name: entry[name] for name in netgamut._KINDS} == {
        "weighted": False,
        "bipartite": None,
        "temporal": True,
        "multilayer": None,
        "signed": None,
        "multigraph": None,
    }


def test_cli_export_networks(tmp_path, catalog, capsys) -> None:
    target = tmp_path / "mirror"
    assert cli.main(["export-networks", "--output", str(target), "--source", "snap"]) == 0
    assert "2 networks" in capsys.readouterr().err
    assert len(json.loads((target / "index.json").read_text())["datasets"]) == 2


def test_cli_says_how_many_wait_for_the_next_run(tmp_path, catalog, capsys) -> None:
    target = tmp_path / "mirror"
    argv = ["export-networks", "--output", str(target), "--source", "snap", "--stop-after", "0"]
    assert cli.main(argv) == 0
    assert "0 networks (0 MB) written" in (err := capsys.readouterr().err)
    assert "2 not built yet" in err
