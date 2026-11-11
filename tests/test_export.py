"""The web mirror: Parquet files and the manifest the explorer and Navigator read."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import netgamut
from netgamut import __main__ as cli
from netgamut import _export
from netgamut._core.columns import Columns
from netgamut._core.download import DownloadError
from netgamut._core.network import ArchiveError
from netgamut._core.nodes import NodeTable


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
    #: The edge list's rows; by default a path, ``1 2``, ``2 3``, ...
    rows: list[tuple[str, ...]] | None = None
    columns: Columns = Columns()
    #: What ``nodes()`` returns; ``None`` raises LookupError, as a source without one does.
    table: NodeTable | None = None


def download_error(cause: Exception) -> DownloadError:
    """A DownloadError chained to *cause*, as ``_core.download._open`` raises it."""
    try:
        raise DownloadError(f"Could not fetch http://example.org: {cause}") from cause
    except DownloadError as error:
        return error


@dataclass
class FakeNetwork:
    dataset: FakeDataset

    @property
    def columns(self) -> Columns:
        return self.dataset.columns

    def rows(self) -> Iterator[tuple[str, ...]]:
        if self.dataset.fails:
            raise RuntimeError("the reader broke")
        if self.dataset.rows is not None:
            yield from self.dataset.rows
            return
        count = 0 if self.dataset.empty else self.dataset.edges or 0
        for i in range(1, count + 1):
            yield str(i), str(i + 1)

    def nodes(self) -> NodeTable:
        if self.dataset.table is None:
            raise LookupError("no node table")
        return self.dataset.table


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


def _read(folder: Path, entry: dict[str, Any], role: str = "links") -> Any:
    return pq.read_table(folder / entry["files"][role])


def _network(tmp_path: Path, monkeypatch, dataset: FakeDataset, **kinds: bool) -> dict[str, Any]:
    """Export a catalog of one network and return its manifest entry."""
    match = FakeMatch("snap", dataset, kinds=dict(kinds))
    monkeypatch.setattr(netgamut, "search", lambda source: [match] if source == "snap" else [])
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert document["failed"] == []
    (entry,) = document["datasets"]
    return entry


def test_writes_parquet_with_attribution(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path)

    assert document["format"] == 2
    assert _keys(document) == ["konect/large", "snap/middle", "snap/tiny"]
    tiny = document["datasets"][2]
    # Counts and direction come from the written file, not the catalog.
    assert (tiny["nodes"], tiny["links"], tiny["directed"]) == (3, 2, True)
    assert tiny["files"] == {"links": "snap/tiny.links.parquet"}
    assert tiny["bytes"] == (tmp_path / tiny["files"]["links"]).stat().st_size
    assert document["bytes"] == sum(entry["bytes"] for entry in document["datasets"])
    links = _read(tmp_path, tiny)
    assert links.column_names == ["source", "target"]
    assert links.to_pydict() == {"source": [1, 2], "target": [2, 3]}
    metadata = pq.read_metadata(tmp_path / tiny["files"]["links"])
    assert metadata.row_group(0).column(0).compression == "ZSTD"
    found = {key.decode(): value.decode() for key, value in metadata.metadata.items()}
    assert {key: value for key, value in found.items() if key.startswith("netgamut:")} == {
        "netgamut:key": "snap/tiny",
        "netgamut:title": "tiny",
        "netgamut:url": "https://example.org/tiny",
        # A newline inside a citation must not split it in two.
        "netgamut:citation": "A. Author, A network (2020). https://doi.org/x",
        "netgamut:directed": "true",
        "netgamut:version": netgamut.__version__,
    }
    assert "dropped" not in tiny and "note" not in tiny
    assert json.loads((tmp_path / "index.json").read_text()) == document


@pytest.mark.parametrize(
    ("ids", "kind"),
    [
        (["0", "4", "14"], "int32"),
        (["0", "4", str(2**40)], "int64"),
        (["0", "4", "007"], "string"),
        (["0", "4", str(2**64)], "string"),
        (["a", "b", "c"], "string"),
    ],
)
def test_ids_are_the_sources_own_in_the_smallest_type(tmp_path, monkeypatch, ids, kind) -> None:
    rows = [(ids[0], ids[1]), (ids[1], ids[2])]
    entry = _network(tmp_path, monkeypatch, FakeDataset("ids", 3, 2, rows=rows))
    links = _read(tmp_path, entry)
    assert str(links.schema.field("source").type) == kind
    assert str(links.schema.field("target").type) == kind
    assert [str(node) for node in links.column("source").to_pylist()] == ids[:2]
    assert entry["nodes"] == 3


@pytest.mark.parametrize(
    ("times", "kind", "values"),
    [
        (["5", "", "7"], "int64", [5, None, 7]),
        (["5", "1.5", "7"], "double", [5.0, 1.5, 7.0]),
        (
            ["2020-01-01", "", "2020-01-02 12:00:00.250"],
            "timestamp[ms, tz=UTC]",
            [datetime(2020, 1, 1, tzinfo=UTC), None, datetime(2020, 1, 2, 12, 0, 0, 250_000, UTC)],
        ),
        (["early", "late", "2020-01-01"], "string", ["early", "late", "2020-01-01"]),
        # numpy reads these words as dates, but they are none.
        (["now", "today", "2020-01-01"], "string", ["now", "today", "2020-01-01"]),
        (["5", "2020-01-01", "7"], "string", ["5", "2020-01-01", "7"]),
    ],
)
def test_writes_weight_and_time_as_the_source_declares(
    tmp_path, monkeypatch, times, kind, values
) -> None:
    rows = [("1", "2", "0.5", times[0]), ("2", "3", "x", times[1]), ("3", "1", "2", times[2])]
    columns = Columns(weight=2, time=3, names=("source", "target", "weight", "when"))
    entry = _network(tmp_path, monkeypatch, FakeDataset("timed", 3, 3, rows=rows, columns=columns))
    links = _read(tmp_path, entry)
    assert links.column_names == ["source", "target", "weight", "time"]
    # Read as Infomap reads a weight: one that is not a number is 1.
    assert links.column("weight").to_pylist() == [0.5, 1.0, 2.0]
    assert str(links.schema.field("time").type) == kind
    column = links.column("time")
    if kind.startswith("timestamp"):
        # The stored milliseconds since 1970 UTC. Turning them into datetimes
        # needs a time zone database, which Windows has only with tzdata.
        expected = [None if value is None else round(value.timestamp() * 1000) for value in values]
        assert column.cast(pa.int64()).to_pylist() == expected
    else:
        assert column.to_pylist() == values


def test_a_free_text_link_column_is_left_out(tmp_path, monkeypatch) -> None:
    rows = [("1", "2", "a\nb"), ("2", "3", "c")]
    columns = Columns(time=2, names=("source", "target", "remark"))
    entry = _network(tmp_path, monkeypatch, FakeDataset("free", 3, 2, rows=rows, columns=columns))
    assert _read(tmp_path, entry).column_names == ["source", "target"]
    assert entry["dropped"] == {"nodes": [], "links": ["remark"]}


@pytest.mark.parametrize(
    ("labels", "kind", "count"),
    [(["1", "2", "1"], "int32", 2), (["bus", "", "tram"], "string", 2)],
)
def test_a_layer_column_is_written_as_layer(tmp_path, monkeypatch, labels, kind, count) -> None:
    rows = [("1", "2", labels[0]), ("2", "3", labels[1]), ("3", "1", labels[2])]
    columns = Columns(layer=2)
    dataset = FakeDataset("layered", 3, 3, rows=rows, columns=columns)
    entry = _network(tmp_path, monkeypatch, dataset, multilayer=True)
    links = _read(tmp_path, entry)
    assert str(links.schema.field("layer").type) == kind
    assert links.column("layer").to_pylist() == [
        (int(label) if kind == "int32" else label) or None for label in labels
    ]
    assert entry["layers"] == count
    assert "note" not in entry


def test_nodes_file_holds_names_and_metadata(tmp_path, monkeypatch) -> None:
    long = "x" * 150
    table = NodeTable(
        values={
            "id": {0: "a", 1: "b", 2: "c", 3: "d"},
            "Name": {0: "Ann", 1: long, 2: "", 3: "Dee"},
            "age": {0: "31", 1: "", 2: "40", 3: "7"},
            "score": {0: "1", 1: "2.5", 2: "", 3: "3"},
            "group": {0: "x", 1: "y", 2: "x"},
            "bio": {0: "short", 1: "y" * 101, 2: "", 3: ""},
            "notes": {0: "one\ntwo", 1: "", 2: "", 3: ""},
        }
    )
    rows = [("0", "1"), ("1", "2"), ("2", "4")]  # 3 is an isolate, 4 has no row
    entry = _network(tmp_path, monkeypatch, FakeDataset("named", 4, 3, rows=rows, table=table))
    assert entry["files"] == {
        "links": "snap/named.links.parquet",
        "nodes": "snap/named.nodes.parquet",
    }
    assert entry["bytes"] == sum(
        (tmp_path / path).stat().st_size for path in entry["files"].values()
    )
    assert entry["nodes"] == 5
    assert entry["dropped"] == {"nodes": ["bio", "notes"], "links": []}
    nodes = _read(tmp_path, entry, "nodes")
    assert [(field.name, str(field.type)) for field in nodes.schema] == [
        ("id", "int32"),
        ("name", "string"),
        ("id_", "string"),
        ("age", "int64"),
        ("score", "double"),
        ("group", "string"),
    ]
    assert nodes.to_pydict() == {
        "id": [0, 1, 2, 3],
        "name": ["Ann", "x" * 99 + "…", None, "Dee"],
        "id_": ["a", "b", "c", "d"],
        "age": [31, None, 40, 7],
        "score": [1.0, 2.5, None, 3.0],
        "group": ["x", "y", "x", None],
    }
    found = pq.read_metadata(tmp_path / entry["files"]["nodes"]).metadata
    assert found[b"netgamut:key"] == b"snap/named"


def test_columns_that_repeat_the_ids_or_names_are_not_written(tmp_path, monkeypatch) -> None:
    # Netzschleuder's index is the id; london_transport's nodeLabel is its name.
    table = NodeTable(
        values={
            "index": {0: "0", 1: "1", 2: "2"},
            "name": {0: "abbeyroad", 1: "westham", 2: ""},
            "nodeLabel": {0: "abbeyroad", 1: "westham", 2: ""},
            "alias": {0: "abbeyroad", 1: "westham", 2: "bank"},
            "_pos": {0: "array([1.67, 6.85])", 1: "array([1.69, 6.86])", 2: ""},
        }
    )
    rows = [("0", "1", "a"), ("1", "2", "b")]
    columns = Columns(names=("source", "target", "_pos"))
    dataset = FakeDataset("london", 3, 2, rows=rows, columns=columns, table=table)
    entry = _network(tmp_path, monkeypatch, dataset)
    assert _read(tmp_path, entry, "nodes").column_names == ["id", "name", "alias"]
    # Graph-tool's internal properties are left out of both files, and listed.
    assert entry["dropped"] == {"nodes": ["_pos"], "links": ["_pos"]}


def test_a_renamed_id_column_clears_every_other_name(tmp_path, monkeypatch) -> None:
    table = NodeTable(
        values={
            "label": {0: "x", 1: "y"},
            "id": {0: "p", 1: "q"},
            "id_": {0: "r", 1: "s"},
            "id__": {0: "t", 1: "u"},
        }
    )
    dataset = FakeDataset("clash", 2, 1, rows=[("0", "1")], table=table)
    entry = _network(tmp_path, monkeypatch, dataset)
    nodes = _read(tmp_path, entry, "nodes").to_pydict()
    assert list(nodes) == ["id", "name", "id___", "id_", "id__"]
    assert (nodes["id___"], nodes["id_"], nodes["id__"]) == (["p", "q"], ["r", "s"], ["t", "u"])


def test_no_nodes_file_without_names_or_metadata(tmp_path, monkeypatch) -> None:
    # A column with no value is no metadata, nor a name column without a name.
    table = NodeTable(values={"name": {0: "", 1: ""}, "group": {0: " ", 1: ""}})
    rows = [("0", "1")]
    entry = _network(tmp_path, monkeypatch, FakeDataset("bare", 2, 1, rows=rows, table=table))
    assert entry["files"] == {"links": "snap/bare.links.parquet"}
    assert not (tmp_path / "snap" / "bare.nodes.parquet").exists()


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
    assert not (tmp_path / "snap" / "middle.links.parquet").exists()


def test_rejects_a_file_over_the_limit_and_remembers_it(tmp_path, catalog) -> None:
    sizes = {
        entry["key"]: entry["bytes"]
        for entry in _export.export_networks(tmp_path / "sized", sources=("snap",))["datasets"]
    }
    limit = sizes["snap/middle"] - 1
    assert sizes["snap/tiny"] <= limit
    document = _export.export_networks(tmp_path, sources=("snap",), max_file=limit)
    assert _keys(document) == ["snap/tiny"]
    middle = next(entry for entry in document["skipped"] if entry["key"] == "snap/middle")
    assert middle["reason"].startswith("too large: ")
    catalog["snap"][1].loads.clear()

    again = _export.export_networks(tmp_path, sources=("snap",), max_file=limit)
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
    assert middle["links"] == 12


def test_a_new_file_format_rebuilds_that_source_alone(tmp_path, catalog, monkeypatch) -> None:
    _export.export_networks(tmp_path, sources=("konect", "snap"))
    large, tiny = catalog["konect"][0], catalog["snap"][0]
    large.loads.clear()
    tiny.loads.clear()
    monkeypatch.setitem(_export._FORMATS, "konect", _export._FORMATS["konect"] + 1)
    _export.export_networks(tmp_path, sources=("konect", "snap"))
    assert len(large.loads) == 1
    assert tiny.loads == []


BROKE = [{"key": "snap/middle", "reason": "RuntimeError: the reader broke"}]


def test_a_failed_rebuild_keeps_the_previous_file_and_is_listed(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    assert first["failed"] == []
    catalog["snap"][1].dataset.fails = True  # also changes its fingerprint
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert document["datasets"] == first["datasets"]
    assert (tmp_path / "snap" / "middle.links.parquet").is_file()
    assert document["failed"] == BROKE


def test_a_failed_network_without_a_file_is_listed_as_failed(tmp_path, catalog) -> None:
    catalog["snap"][1].dataset.fails = True
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["snap/tiny"]
    assert document["skipped"] == []
    assert document["failed"] == BROKE


def test_a_failure_stands_until_the_network_builds(tmp_path, catalog) -> None:
    catalog["snap"][1].dataset.fails = True
    _export.export_networks(tmp_path, sources=("snap",))
    # A run that stops before reaching it still lists it.
    stopped = _export.export_networks(tmp_path, sources=("snap",), stop_after=0)
    assert stopped["failed"] == BROKE
    # So does a run of another source.
    assert _export.export_networks(tmp_path, sources=("konect",))["failed"] == BROKE
    catalog["snap"][1].dataset.fails = False
    assert _export.export_networks(tmp_path, sources=("snap",))["failed"] == []


@pytest.mark.parametrize(
    ("error", "failed"),
    [
        (ArchiveError("bundle.tar.gz is an archive."), False),
        (LookupError("'x' has no downloadable files."), False),
        (KeyError("source"), True),
        (ModuleNotFoundError("No module named 'numpy'"), True),
    ],
)
def test_only_netgamuts_own_errors_count_as_failed(tmp_path, catalog, error, failed) -> None:
    catalog["snap"][0].dataset.error = error
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert bool(document["failed"]) == failed
    assert bool(document["skipped"]) != failed


def test_a_network_that_reads_as_empty_is_skipped(tmp_path, catalog) -> None:
    catalog["snap"][1].dataset.empty = True
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert _keys(document) == ["snap/tiny"]
    (middle,) = document["skipped"]
    assert middle["key"] == "snap/middle"
    assert middle["reason"] == "empty: 0 nodes and 0 links read"
    assert not (tmp_path / "snap" / "middle.links.parquet").exists()


def test_an_empty_file_from_an_earlier_build_is_built_again(tmp_path, catalog) -> None:
    """Unchanged in the catalog, but its file had nothing in it: not worth keeping."""
    _export.export_networks(tmp_path, sources=("snap",))
    manifest = tmp_path / "index.json"
    earlier = json.loads(manifest.read_text())
    for entry in earlier["datasets"]:
        if entry["key"] == "snap/middle":
            entry["nodes"] = entry["links"] = 0
    manifest.write_text(json.dumps(earlier))
    middle = catalog["snap"][1]
    middle.loads.clear()
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert len(middle.loads) == 1
    rebuilt = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert (rebuilt["nodes"], rebuilt["links"]) == (11, 10)


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
    assert (tmp_path / "snap" / "middle.links.parquet").is_file()

    resumed = _export.export_networks(tmp_path, sources=("snap",))
    assert resumed["complete"] is True
    middle = next(entry for entry in resumed["datasets"] if entry["key"] == "snap/middle")
    assert middle["links"] == 12


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
    assert document["reserved"] == (2 + 10) * 3
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
    assert document["failed"] == []


def test_an_unreachable_source_keeps_its_earlier_files(tmp_path, catalog) -> None:
    first = _export.export_networks(tmp_path, sources=("snap",))
    # A changed catalog entry asks for a rebuild, but the host does not answer.
    catalog["snap"][0].dataset.error = download_error(TimeoutError("timed out"))
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert document["datasets"] == first["datasets"]
    assert document["reserved"] == 0


def test_a_subgraph_stack_is_written_in_its_layers(tmp_path, catalog, monkeypatch) -> None:
    middle = catalog["snap"][1]
    middle.kinds["multilayer"] = True
    asked: list[tuple[str, dict[str, Any]]] = []
    # Each subgraph numbers its nodes on its own; "id" matches them.
    sms = FakeDataset(
        "sms",
        2,
        2,
        directed=True,
        rows=[("0", "1", "5"), ("1", "0", "6")],
        columns=Columns(time=2),
        table=NodeTable(values={"id": {0: "10", 1: "11"}, "name": {0: "Ann", 1: ""}}),
    )
    calls = FakeDataset(
        "calls",
        3,
        1,
        rows=[("0", "2")],
        table=NodeTable(values={"id": {0: "11", 1: "12", 2: "13"}, "name": {0: "", 1: "", 2: ""}}),
    )

    def subgraph_layers(name: str, **kwargs: Any) -> tuple[dict[str, Any], str]:
        asked.append((name, kwargs))
        return {"sms": FakeNetwork(sms), "calls": FakeNetwork(calls)}, "id"

    monkeypatch.setattr(netgamut.snap, "_subgraph_layers", subgraph_layers, raising=False)
    document = _export.export_networks(tmp_path, sources=("snap",), max_download=123)
    entry = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert asked == [("middle", {"progress": False, "max_bytes": 123})]
    assert middle.loads == []  # the flat network is not downloaded besides
    assert (entry["nodes"], entry["links"], entry["layers"], entry["directed"]) == (4, 3, 2, True)
    assert entry["multilayer"] is True
    links = _read(tmp_path, entry)
    assert links.to_pydict() == {
        "source": [10, 11, 11],
        "target": [11, 10, 13],
        "time": [5, 6, None],
        "layer": ["sms", "sms", "calls"],
    }
    nodes = _read(tmp_path, entry, "nodes")
    assert nodes.to_pydict() == {"id": [10, 11, 12, 13], "name": ["Ann", None, None, None]}
    # A network without layers has no layers field.
    tiny = next(entry for entry in document["datasets"] if entry["key"] == "snap/tiny")
    assert "layers" not in tiny


def _stack(monkeypatch, catalog, *layers: FakeDataset) -> None:
    """Make snap/middle a stack of *layers*, matched by ``id``."""
    catalog["snap"][1].kinds["multilayer"] = True

    def subgraph_layers(name: str, **kwargs: Any) -> tuple[dict[str, Any], str]:
        return {layer.name: FakeNetwork(layer) for layer in layers}, "id"

    monkeypatch.setattr(netgamut.snap, "_subgraph_layers", subgraph_layers, raising=False)


def test_a_stack_node_without_a_key_has_no_row(tmp_path, catalog, monkeypatch) -> None:
    # Node 2 of a has no id at all, node 0 of b an empty one: neither keeps its
    # own id, which could be another node's key (b's node 1 is "2").
    a = FakeDataset(
        "a",
        3,
        1,
        rows=[("0", "1")],
        table=NodeTable(values={"id": {0: "5", 1: "6"}, "name": {0: "A", 1: "B", 2: "Orphan"}}),
    )
    b = FakeDataset(
        "b",
        3,
        1,
        rows=[("1", "2")],
        table=NodeTable(values={"id": {0: "", 1: "2", 2: "6"}, "name": {0: "Zero", 1: "Two"}}),
    )
    _stack(monkeypatch, catalog, a, b)
    document = _export.export_networks(tmp_path, sources=("snap",))
    entry = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    nodes = _read(tmp_path, entry, "nodes").to_pydict()
    assert nodes == {"id": [5, 6, 2], "name": ["A", "B", "Two"]}
    assert entry["nodes"] == 3


def test_a_stack_link_to_a_node_without_a_key_fails(tmp_path, catalog, monkeypatch) -> None:
    a = FakeDataset("a", 2, 1, rows=[("0", "1")], table=NodeTable(values={"id": {0: "5", 1: ""}}))
    _stack(monkeypatch, catalog, a)
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert "snap/middle" not in _keys(document)
    (failed,) = document["failed"]
    assert failed["key"] == "snap/middle"
    assert "node 1 of layer 'a' has no key" in failed["reason"]


def test_a_stack_with_a_subgraph_without_nodes_is_written_as_one(
    tmp_path, catalog, monkeypatch
) -> None:
    a = FakeDataset("a", 2, 1, rows=[("0", "1")], table=NodeTable(values={"id": {0: "5"}}))
    _stack(monkeypatch, catalog, a, FakeDataset("b", 2, 1, rows=[("0", "1")]))
    document = _export.export_networks(tmp_path, sources=("snap",))
    entry = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert "layers" not in entry
    assert len(catalog["snap"][1].loads) == 1
    assert entry["note"] == "snap tags this network multilayer, but its data has no layers"


def test_a_bug_in_a_stack_is_not_taken_for_a_missing_table(tmp_path, catalog, monkeypatch) -> None:
    catalog["snap"][1].kinds["multilayer"] = True

    def subgraph_layers(name: str, **kwargs: Any) -> None:
        raise KeyError("a bug")

    monkeypatch.setattr(netgamut.snap, "_subgraph_layers", subgraph_layers, raising=False)
    document = _export.export_networks(tmp_path, sources=("snap",))
    assert [entry["key"] for entry in document["failed"]] == ["snap/middle"]
    assert catalog["snap"][1].loads == []


def test_a_multilayer_network_without_layers_is_written_as_one(
    tmp_path, catalog, monkeypatch
) -> None:
    middle = catalog["snap"][1]
    middle.kinds["multilayer"] = True

    def subgraph_layers(name: str, **kwargs: Any) -> None:
        raise LookupError("no subgraphs")

    monkeypatch.setattr(netgamut.snap, "_subgraph_layers", subgraph_layers, raising=False)
    document = _export.export_networks(tmp_path, sources=("snap",))
    entry = next(entry for entry in document["datasets"] if entry["key"] == "snap/middle")
    assert "layers" not in entry
    assert len(middle.loads) == 1
    assert entry["note"] == "snap tags this network multilayer, but its data has no layers"
    assert _read(tmp_path, entry).column_names == ["source", "target"]


def test_a_format_1_mirror_is_built_again_and_its_files_deleted(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path, sources=("snap",))
    old = {"format": None, **document}
    del old["format"]
    for entry in old["datasets"]:
        old_file = entry["files"]["links"].replace(".links.parquet", ".net.gz")
        (tmp_path / old_file).write_bytes(b"")
        (tmp_path / entry.pop("files")["links"]).unlink()
        entry["file"] = old_file
        entry["edges"] = entry.pop("links")
    (tmp_path / "index.json").write_text(json.dumps(old))
    for match in catalog["snap"]:
        match.loads.clear()

    rebuilt = _export.export_networks(tmp_path, sources=("snap",))
    assert rebuilt["format"] == 2
    assert [len(match.loads) for match in catalog["snap"][:2]] == [1, 1]
    assert all("file" not in entry for entry in rebuilt["datasets"])
    assert sorted(path.name for path in (tmp_path / "snap").iterdir()) == [
        "middle.links.parquet",
        "tiny.links.parquet",
    ]


def test_a_one_source_run_over_a_format_1_mirror_lists_format_2_alone(tmp_path, catalog) -> None:
    document = _export.export_networks(tmp_path)
    old = {key: value for key, value in document.items() if key != "format"}
    for entry in old["datasets"]:
        if entry["key"].startswith("konect/"):
            old_file = entry["files"]["links"].replace(".links.parquet", ".net.gz")
            (tmp_path / old_file).write_bytes(b"x" * 1000)
            (tmp_path / entry.pop("files")["links"]).unlink()
            entry["file"] = old_file
            entry["edges"] = entry.pop("links")
    (tmp_path / "index.json").write_text(json.dumps(old))

    again = _export.export_networks(tmp_path, sources=("snap",))
    assert all("files" in entry for entry in again["datasets"])
    assert _keys(again) == ["snap/middle", "snap/tiny"]
    assert {"key": "konect/large", "reason": "not built yet"} in again["skipped"]
    assert list((tmp_path / "konect").iterdir()) == []
    assert again["complete"] is False
    # It stays incomplete until konect's own run builds it.
    assert _export.export_networks(tmp_path, sources=("snap",))["complete"] is False
    built = _export.export_networks(tmp_path, sources=("konect",))
    assert built["complete"] is True
    assert "konect/large" in _keys(built)


def test_a_new_mirror_format_rebuilds_every_network(tmp_path, catalog, monkeypatch) -> None:
    _export.export_networks(tmp_path, sources=("snap",))
    _export.export_networks(tmp_path, sources=("snap",))
    built = catalog["snap"][:2]  # reviews is not a graph
    assert [len(match.loads) for match in built] == [1, 1]
    monkeypatch.setattr(_export, "FORMAT", _export.FORMAT + 1)
    rebuilt = _export.export_networks(tmp_path, sources=("snap",))
    assert [len(match.loads) for match in built] == [2, 2]
    assert rebuilt["format"] == _export.FORMAT
    assert _keys(rebuilt) == ["snap/middle", "snap/tiny"]


def test_writing_layers_rebuilds_the_multilayer_networks_alone(tmp_path, catalog) -> None:
    large, tiny = catalog["konect"][0], catalog["snap"][0]
    before = _export._fingerprint(large), _export._fingerprint(tiny)
    large.kinds["multilayer"] = True
    assert _export._fingerprint(large) != before[0]
    assert _export._fingerprint(tiny) == before[1]


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


def test_cli_says_how_many_failed(tmp_path, catalog, capsys) -> None:
    catalog["snap"][1].dataset.fails = True
    target = tmp_path / "mirror"
    assert cli.main(["export-networks", "--output", str(target), "--source", "snap"]) == 0
    assert "0 skipped, 1 failed" in capsys.readouterr().err


def test_cli_says_how_many_wait_for_the_next_run(tmp_path, catalog, capsys) -> None:
    target = tmp_path / "mirror"
    argv = ["export-networks", "--output", str(target), "--source", "snap", "--stop-after", "0"]
    assert cli.main(argv) == 0
    assert "0 networks (0 MB) written" in (err := capsys.readouterr().err)
    assert "2 not built yet" in err
