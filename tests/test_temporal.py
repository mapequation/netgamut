"""Column roles, link times, time snapshots and multilayer networks from downloads."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from networks import InMemoryNetwork, MultilayerNetwork
from networks._core.columns import Columns
from networks._core.models import DataFile
from networks._core.network import Network
from networks.snap._models import Dataset

np = pytest.importorskip("numpy")


def make_network(path: Path, columns: Columns, *, type_: str = "Undirected") -> Network:
    dataset = Dataset(name="toy", categories=(), category_names=(), type=type_, columns={})
    return Network(
        dataset=dataset, file=DataFile(path.name, "u"), archive_path=path, columns=columns
    )


@pytest.fixture
def contacts(tmp_path: Path) -> Network:
    """SNAP's temporal layout, ``SRC DST UNIXTS``, deliberately out of time order."""
    path = tmp_path / "contacts.txt"
    path.write_text(
        "# FromNodeId ToNodeId Timestamp\n"
        "1 2 100\n"
        "2 3 350\n"
        "2 1 110\n"  # the same undirected link as the first row
        "3 4 900\n"
        "1 3 1000\n"
    )
    return make_network(path, Columns(time=2, names=("source", "target", "time")))


def test_a_timestamp_is_not_a_weight(contacts: Network, tmp_path: Path) -> None:
    assert not contacts.weighted_links
    assert [
        line
        for line in contacts.to_link_list(tmp_path).read_text().splitlines()
        if not line.startswith("#")
    ] == ["1 2", "2 3", "2 1", "3 4", "1 3"]
    pytest.importorskip("infomap")
    im = contacts.to_infomap(silent=True, num_trials=1)
    im.run()
    assert im.num_links == 5 and im.num_nodes == 4


def test_times_are_an_array_in_file_order(contacts: Network) -> None:
    assert contacts.times.dtype == np.int64
    assert contacts.times.tolist() == [100, 350, 110, 900, 1000]
    assert contacts.times is contacts.times  # read once


def test_time_bins_follow_numpy(contacts: Network) -> None:
    assert (
        contacts.time_bins(3).tolist()
        == np.histogram_bin_edges([100, 350, 110, 900, 1000], bins=3).tolist()
    )
    assert contacts.time_bins([0, 500, 1000]).tolist() == [0, 500, 1000]


def test_snapshots_are_lazy_and_count_events(contacts: Network) -> None:
    snapshots = contacts.snapshots(bins=[100, 400, 700, 1000])
    first = next(snapshots)
    assert isinstance(first, InMemoryNetwork)
    assert first.name == "toy-t0"
    assert first.links == {(1, 2): 2.0, (2, 3): 1.0}  # 1->2 and 2->1 are one link
    assert first.info["events"] == 3 and first.info["start"] == 100
    rest = list(snapshots)
    assert [s.info["bin"] for s in rest] == [1, 2]
    assert rest[0].links == {}  # an empty bin still yields
    assert rest[1].links == {(3, 4): 1.0, (1, 3): 1.0}  # the last bin is closed


def test_range_leaves_out_times_outside_it(contacts: Network) -> None:
    (only,) = contacts.snapshots(bins=1, range=(0, 200))
    assert only.links == {(1, 2): 2.0}


def test_directed_snapshots_keep_both_directions(contacts: Network) -> None:
    (only,) = contacts.snapshots(bins=1, range=(0, 200), directed=True)
    assert only.links == {(1, 2): 1.0, (2, 1): 1.0}


def test_time_layers_make_a_multilayer_network(contacts: Network, tmp_path: Path) -> None:
    ml = contacts.to_multilayer(bins=[100, 400, 700, 1000])
    assert isinstance(ml, MultilayerNetwork)
    assert ml.intra == {
        0: {(1, 2): 2.0, (2, 3): 1.0},
        1: {},
        2: {(3, 4): 1.0, (1, 3): 1.0},
    }
    assert ml.info["bin_edges"].tolist() == [100, 400, 700, 1000]
    assert ml.weighted
    assert "*Intra" in ml.to_pajek(tmp_path).read_text()


def test_weights_add_up_inside_a_bin(tmp_path: Path) -> None:
    path = tmp_path / "konect.tsv"
    path.write_text("1 2 3 10\n1 2 4 20\n2 3 1 30\n")
    net = make_network(path, Columns(weight=2, time=3))
    (only,) = net.snapshots(bins=1)
    assert only.links == {(1, 2): 7.0, (2, 3): 1.0}


def test_iso_dates_become_datetimes(tmp_path: Path) -> None:
    path = tmp_path / "dated.csv"
    path.write_text("1,2,2010-11-08 18:45:11\n2,3,2010-11-09 18:45:11\n3,1,2010-11-10 18:45:11\n")
    net = make_network(path, Columns(time=2))
    assert net.times.dtype == np.dtype("datetime64[s]")
    edges = net.time_bins(2)
    assert edges.dtype == np.dtype("datetime64[us]")
    assert [s.info["events"] for s in net.snapshots(bins=2)] == [1, 2]


def test_layer_columns_make_a_multilayer_network(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    path.write_text("# source, target, weight, layer\n0,1,2,1\n1,2,1,1\n0,2,5,2\n")
    names = ("source", "target", "weight", "layer")
    net = make_network(path, Columns(weight=2, layer=3, names=names))
    ml = net.to_multilayer()
    assert ml.intra == {1: {(0, 1): 2.0, (1, 2): 1.0}, 2: {(0, 2): 5.0}}
    assert ml.info["layers"] == "layer"


def test_any_categorical_column_can_be_the_layer(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    path.write_text("0,1,100,RT\n1,2,200,MT\n0,2,300,RT\n")
    names = ("source", "target", "timestamp", "interaction")
    net = make_network(path, Columns(time=2, names=names))
    ml = net.to_multilayer(layer="interaction")
    assert ml.intra == {"MT": {(1, 2): 1.0}, "RT": {(0, 1): 1.0, (0, 2): 1.0}}
    with pytest.raises(LookupError, match="no column named"):
        net.to_multilayer(layer="kind")
    with pytest.raises(ValueError, match="not both"):
        net.to_multilayer(bins=2, layer="interaction")


def test_missing_roles_say_how_to_declare_them(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("1 2 100\n")
    net = make_network(path, Columns())
    with pytest.raises(LookupError, match="no time column"):
        net.times  # noqa: B018
    with pytest.raises(LookupError, match="no layer column"):
        net.to_multilayer()
    declared = dataclasses.replace(net, columns=Columns(time=2))
    assert declared.times.tolist() == [100]


def test_columns_describe_themselves() -> None:
    assert str(Columns(time=2, names=("source", "target", "time"))) == ("source target time (time)")
    assert str(Columns(weight=2, time=3)) == "source target ? (weight) ? (time)"


def test_every_snap_layout_names_a_real_file() -> None:
    from networks import snap
    from networks.snap import _COLUMNS

    files = {f.name for dataset in snap.datasets() for f in dataset.files}
    assert set(_COLUMNS) <= files


def _nz_network(tmp_path: Path, header: str, properties: list[list[str]]) -> Network:
    from networks import netzschleuder
    from networks.netzschleuder import _columns as nz_columns

    path = tmp_path / "edges.csv"
    path.write_text(f"# {header}\n0,1,1,5\n")
    dataset = dataclasses.replace(
        netzschleuder.info("london_transport"), analyses={"edge_properties": properties}
    )
    net = Network(dataset=dataset, file=DataFile("e", "u"), archive_path=path)
    return dataclasses.replace(net, columns=nz_columns(net, dataset))


def test_netzschleuder_roles_come_from_the_file_header(tmp_path: Path) -> None:
    net = _nz_network(
        tmp_path, "source, target, weight, layer", [["weight", "double"], ["layer", "int16_t"]]
    )
    assert (net.columns.weight, net.columns.time, net.columns.layer) == (2, None, 3)
    # A boolean weight only marks presence; the header, not the catalog, orders columns.
    net = _nz_network(
        tmp_path,
        "source, target, weight, timestamp",
        [["timestamp", "int32_t"], ["weight", "bool"]],
    )
    assert (net.columns.weight, net.columns.time) == (None, 3)
    assert net.columns.names == ("source", "target", "weight", "timestamp")


def test_signed_netzschleuder_weights_are_not_weights(tmp_path: Path) -> None:
    from networks import netzschleuder
    from networks.netzschleuder import _columns as nz_columns

    path = tmp_path / "edges.csv"
    path.write_text("# source, target, weight\n0,1,-1\n1,2,1\n")
    dataset = dataclasses.replace(
        netzschleuder.info("new_guinea_tribes"),
        analyses={"edge_properties": [["weight", "int16_t"]]},
    )
    net = Network(dataset=dataset, file=DataFile("e", "u"), archive_path=path)
    assert nz_columns(net, dataset).weight is None


def _timed(tmp_path: Path, times: list[str], *, layers: list[str] | None = None) -> Network:
    path = tmp_path / "timed.csv"
    rows = [f"{i},{i + 1},{t}" for i, t in enumerate(times)]
    if layers is not None:
        rows = [f"{row},{layer}" for row, layer in zip(rows, layers, strict=True)]
    path.write_text("\n".join(rows) + "\n")
    return make_network(path, Columns(time=2, layer=3 if layers is not None else None))


def test_missing_times_are_left_out_of_every_bin(tmp_path: Path) -> None:
    dated = _timed(tmp_path, ["2020-01-01", "", "2020-01-05", "2020-01-09"])
    assert np.isnat(dated.times[1])
    snapshots = list(dated.snapshots(bins=2))
    assert [s.info["events"] for s in snapshots] == [1, 2]
    assert str(snapshots[0].info["start"]).startswith("2020-01-01")
    numbers = _timed(tmp_path, ["100", "", "200", "300"])
    assert [s.info["events"] for s in numbers.snapshots(bins=2)] == [1, 2]


def test_a_missing_first_time_does_not_hide_dates(tmp_path: Path) -> None:
    path = tmp_path / "late.txt"
    path.write_text("1 2\n2 3 2020-01-01\n3 4 2020-01-03\n")
    net = make_network(path, Columns(time=2))
    assert net.times.dtype == np.dtype("datetime64[s]")
    assert [s.info["events"] for s in net.snapshots(bins=1)] == [2]


def test_datetime_bins_can_be_dates_or_the_edges_time_bins_gave(tmp_path: Path) -> None:
    dated = _timed(tmp_path, ["2020-01-01", "2020-01-03", "2020-01-05"])
    again = list(dated.snapshots(bins=dated.time_bins(2)))
    assert [s.info["events"] for s in again] == [1, 2]
    by_date = list(dated.snapshots(bins=["2020-01-01", "2020-01-02", "2020-01-06"]))
    assert [s.info["events"] for s in by_date] == [1, 2]


def test_reported_datetime_edges_are_the_edges_used(tmp_path: Path) -> None:
    # Edges at 3.33 s and 6.67 s: whole-second edges would claim bin 0 ends at
    # 00:00:03 while holding the 00:00:03 event.
    times = ["00:00:00", "00:00:03", "00:00:04", "00:00:10"]
    net = _timed(tmp_path, [f"1970-01-01T{t}" for t in times])
    snapshots = list(net.snapshots(bins=3))
    assert [s.info["events"] for s in snapshots] == [2, 1, 1]
    assert snapshots[0].info["end"] == np.datetime64("1970-01-01T00:00:03.333333")
    assert snapshots[0].info["end"] > net.times[1]


def test_rows_without_a_layer_belong_to_no_layer(tmp_path: Path) -> None:
    net = _timed(tmp_path, ["1", "2", "3"], layers=["1", "", "2"])
    assert set(net.to_multilayer().intra) == {1, 2}


def test_leading_zero_labels_stay_strings_everywhere(tmp_path: Path) -> None:
    path = tmp_path / "zeros.txt"
    path.write_text("007 1 5\n1 2 6\n")
    net = make_network(path, Columns(time=2))
    (only,) = net.snapshots(bins=1)
    assert set(only.nodes) == {"007", "1", "2"}
    body = net.to_pajek(tmp_path / "out").read_text()
    assert '"007"' in body


def test_higgs_interactions_are_its_layers() -> None:
    from networks.snap import _COLUMNS

    assert _COLUMNS["higgs-activity_time.txt.gz"].layer == 3
