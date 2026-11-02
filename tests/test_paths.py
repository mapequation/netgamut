"""The curated paths source: catalog, parsers, handle and registration."""

from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

import netgamut
from netgamut import paths
from netgamut.__main__ import main
from netgamut.paths import _parsers
from netgamut.paths._models import PathData

# -- fixtures shaped like the real downloads --------------------------------

WIKISPEEDIA_TSV = (
    "# Comment line, as the real file has.\n"
    "\n"
    "hash1\t1297740409\t166\t14th_century;Time;Science\tNULL\n"
    "hash2\t1297740410\t88\tA;B;C;<;D\t3\n"
    "hash3\t1297740411\t31\t%C3%89douard_Manet;Earth%27s_atmosphere\tNULL\n"
    "hash4\t1297740412\t42\tX;Y;<;<;Z\t2\n"
)


def _wikispeedia_bundle(directory: Path) -> Path:
    """A tar.gz shaped like wikispeedia_paths-and-graph.tar.gz."""
    archive = directory / "wikispeedia_paths-and-graph.tar.gz"
    payload = WIKISPEEDIA_TSV.encode()
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("wikispeedia_paths-and-graph/paths_finished.tsv")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    return archive


DB1B_HEADER = "ItinID,MktID,SeqNum,Coupons,Year,Origin,Dest,Passengers,"
# The real file is globally sorted by SeqNum descending, so one itinerary's
# rows are scattered; the fixture reproduces that.
DB1B_ROWS = (
    "20191,m,3,3,2019,LAX,ATL,1.00,",
    "20192,m,2,2,2019,SLC,OAK,2.00,",
    "20190,m,2,2,2019,CCC,DDD,1.00,",
    "20191,m,2,3,2019,SLC,LAX,1.00,",
    "20192,m,1,2,2019,OAK,SLC,2.00,",
    "20190,m,1,2,2019,AAA,BBB,1.00,",
    "20191,m,1,3,2019,OAK,SLC,1.00,",
)


def _db1b_archive(directory: Path) -> Path:
    archive = directory / "Origin_and_Destination_Survey_DB1BCoupon_2019_1.zip"
    body = DB1B_HEADER + "\n" + "\n".join(DB1B_ROWS) + "\n"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Origin_and_Destination_Survey_DB1BCoupon_2019_1.csv", body)
    return archive


# Airport and city-market columns side by side, as the real header has them.
# JFK/LGA share city market 31703, ORD/MDW 30977, OAK/SFO 32457.
DB1B_CITY_HEADER = (
    "ItinID,MktID,SeqNum,Coupons,Year,OriginCityMarketID,Origin,DestCityMarketID,Dest,Passengers,"
)
DB1B_CITY_ROWS = (
    "1,m,2,2,2019,30977,MDW,31703,LGA,4.00,",  # airport change in Chicago on the way back
    "3,m,1,1,2019,32457,OAK,32457,SFO,1.00,",  # one coupon inside a single city market
    "1,m,1,2,2019,31703,JFK,30977,ORD,4.00,",
    "2,m,2,2,2019,32457,SFO,32101,LAX,2.00,",
    "2,m,1,2,2019,32457,OAK,32457,SFO,2.00,",  # a same-city hop before flying on
)


def _db1b_city_archive(directory: Path) -> Path:
    archive = directory / "Origin_and_Destination_Survey_DB1BCoupon_2019_1.zip"
    body = DB1B_CITY_HEADER + "\n" + "\n".join(DB1B_CITY_ROWS) + "\n"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Origin_and_Destination_Survey_DB1BCoupon_2019_1.csv", body)
    return archive


# OpenFlights airports.dat: header-less, IATA in the fifth field, \N when absent.
AIRPORTS_DAT = (
    '3453,"Metropolitan Oakland International Airport","Oakland","United States","OAK",'
    '"KOAK",37.721298,-122.221001,9,-8,"A","America/Los_Angeles","airport","OurAirports"\n'
    '3484,"Los Angeles International Airport","Los Angeles","United States","LAX","KLAX",'
    '33.94250107,-118.4079971,125,-8,"A","America/Los_Angeles","airport","OurAirports"\n'
    '9999,"Some Heliport","Nowhere","United States",\\N,"KXXX",1.0,2.0,0,-8,"A","UTC",'
    '"airport","OurAirports"\n'
    '9998,"Duplicate Code","Elsewhere","United States","LAX","XXXX",0.0,0.0,0,0,"A","UTC",'
    '"airport","OurAirports"\n'
    '9997,"Two Letter","Odd","United States","XY","XXXY",0.0,0.0,0,0,"A","UTC",'
    '"airport","OurAirports"\n'
)

CITY_MARKET_LOOKUP = (
    "Code,Description\n"
    '"30977","Chicago, IL"\n'
    '"31703","New York City, NY (Metropolitan Area)"\n'
    '"32457","San Francisco, CA (Metropolitan Area)"\n'
)


def _handle(name: str, archive: Path) -> PathData:
    dataset = paths.info(name)
    return PathData(
        dataset=dataset, file=dataset.files[0], archive_path=archive, parser=dataset.parser
    )


# -- catalog and registration ------------------------------------------------


def test_source_is_registered_everywhere() -> None:
    assert "paths" in netgamut.sources()
    assert netgamut.domains()  # must not KeyError on the new source's domains
    found = netgamut.search(domain="Transportation", source="paths")
    assert [match.name for match in found] == ["db1b-coupon", "db1b-city"]
    assert found[0].key == "paths/db1b-coupon"


def test_catalog_entries_are_complete() -> None:
    for dataset in paths.datasets():
        assert dataset.parser in _parsers.PARSERS
        assert dataset.files, dataset.name
        assert dataset.domains, dataset.name
        assert dataset.citations, dataset.name
        for role, item in dataset.metadata:
            assert role and item.name and item.url.startswith("https://"), dataset.name
        if dataset.shares_download_with:
            shared = paths.info(dataset.shares_download_with)
            assert shared.files == dataset.files and shared.url_template == dataset.url_template


def test_wikispeedia_is_ambiguous_across_sources() -> None:
    # SNAP lists wikispeedia too; the top level must refuse to guess.
    with pytest.raises(KeyError, match="several sources"):
        netgamut.load("wikispeedia")


def test_unknown_name_suggests_alternatives() -> None:
    with pytest.raises(KeyError, match="db1b-coupon"):
        paths.info("db1b_coupon")


def test_describe_mentions_the_default_quarter() -> None:
    text = paths.describe("db1b-coupon")
    assert "2019 Q1" in text
    assert "public domain" in text


def test_file_for_validates_year_and_quarter() -> None:
    quarterly = paths.info("db1b-coupon")
    assert "2018_3" in quarterly.file_for(year=2018, quarter=3).url
    # Passing only one of the pair fills the other from the defaults.
    assert "2021_1" in quarterly.file_for(year=2021).url
    with pytest.raises(ValueError, match="quarter"):
        quarterly.file_for(year=2019, quarter=5)
    with pytest.raises(ValueError, match="year"):
        quarterly.file_for(year=1980, quarter=1)
    single = paths.info("wikispeedia")
    with pytest.raises(ValueError, match="single collection"):
        single.file_for(year=2019, quarter=1)


# -- wikispeedia parser -------------------------------------------------------


def test_wikispeedia_paths_resolve_back_clicks(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    trails = list(handle.paths())
    assert trails == [
        (("14th_century", "Time", "Science"), 1.0),
        (("A", "B", "C", "B", "D"), 1.0),
        (("Édouard_Manet", "Earth's_atmosphere"), 1.0),
        (("X", "Y", "X", "Z"), 1.0),
    ]


def test_wikispeedia_state_network(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    net = handle.to_state_network(order=2)
    assert net.num_paths == 4
    # The two-article trail has no trigram.
    assert net.num_short_paths == 1
    assert net.state_name(1) == "14th_century -> Time"


def test_edges_are_the_first_order_projection(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    pairs = list(handle.edges())
    assert ("14th_century", "Time") in pairs
    assert ("C", "B") in pairs  # the resolved back click is a real step
    assert len(pairs) == 2 + 4 + 1 + 3


def test_to_networkx_works_on_path_data(tmp_path: Path) -> None:
    networkx = pytest.importorskip("networkx")
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    graph = handle.to_networkx(cast=None)
    assert isinstance(graph, networkx.DiGraph)
    assert graph.has_edge("Time", "Science")


def test_to_infomap_defaults_to_first_order(tmp_path: Path) -> None:
    pytest.importorskip("infomap")
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    first_order = handle.to_infomap(silent=True).run()
    assert not first_order.have_memory  # order=1 is a plain directed network
    second_order = handle.to_infomap(order=2, silent=True).run()
    assert second_order.have_memory


# -- subsetting ---------------------------------------------------------------


def test_max_paths_keeps_the_first_paths(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    trails = list(handle.paths(max_paths=2))
    assert trails == [
        (("14th_century", "Time", "Science"), 1.0),
        (("A", "B", "C", "B", "D"), 1.0),
    ]


def test_max_nodes_keeps_the_most_visited_nodes(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    # Weighted visits: B and X are visited twice, everything else once; ties
    # among the once-visited nodes break by first appearance, so the top 5
    # are B, X, then the first trail's three articles -- which is the only
    # trail that survives intact.
    trails = list(handle.paths(max_nodes=5))
    assert trails == [(("14th_century", "Time", "Science"), 1.0)]


def test_max_nodes_cuts_paths_instead_of_bridging_them(tmp_path: Path) -> None:
    archive = tmp_path / "split.zip"
    rows = "\n".join(
        f"1,m,{seq},5,2019,{origin},{dest},3.00,"
        for seq, (origin, dest) in enumerate(
            [("OAK", "SLC"), ("SLC", "LAX"), ("LAX", "OAK"), ("OAK", "SLC"), ("SLC", "OAK")],
            start=1,
        )
    )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("split.csv", DB1B_HEADER + "\n" + rows + "\n")
    handle = _handle("db1b-coupon", archive)
    # The full itinerary is OAK SLC LAX OAK SLC OAK; dropping LAX must yield
    # two separate segments, never a fabricated SLC -> OAK shortcut over LAX.
    segments = list(handle.paths(max_nodes=2))
    assert segments == [
        (("OAK", "SLC"), 3.0),
        (("OAK", "SLC", "OAK"), 3.0),
    ]


def test_subset_limits_compose_and_feed_the_state_network(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    # max_paths first, then the node ranking on that subset only.
    trails = list(handle.paths(max_paths=1, max_nodes=2))
    assert trails == [(("14th_century", "Time"), 1.0)]

    net = handle.to_state_network(order=1, max_paths=1, max_nodes=2)
    assert net.num_physical_nodes == 2
    assert net.num_links == 1


def test_subset_limits_are_validated(tmp_path: Path) -> None:
    handle = _handle("wikispeedia", _wikispeedia_bundle(tmp_path))
    with pytest.raises(ValueError, match="max_paths"):
        handle.paths(max_paths=0)
    with pytest.raises(ValueError, match="max_nodes"):
        handle.paths(max_nodes=0)


# -- db1b parser --------------------------------------------------------------


def test_db1b_itineraries_are_grouped_and_ordered(tmp_path: Path) -> None:
    handle = _handle("db1b-coupon", _db1b_archive(tmp_path))
    trips = list(handle.paths())
    assert trips == [
        (("AAA", "BBB", "CCC", "DDD"), 1.0),  # a gap keeps both airports
        (("OAK", "SLC", "LAX", "ATL"), 1.0),
        (("OAK", "SLC", "OAK"), 2.0),  # weight = sampled passengers
    ]


def test_db1b_derivation_happens_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _db1b_archive(tmp_path)
    handle = _handle("db1b-coupon", archive)
    first = list(handle.paths())
    derived = archive.with_name(archive.stem + ".paths.tsv")
    assert derived.is_file()

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("derivation ran twice")

    monkeypatch.setattr(_parsers, "_derive_db1b", explode)
    assert list(handle.paths()) == first


def test_db1b_empty_csv_is_not_cached_as_truth(tmp_path: Path) -> None:
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("empty.csv", DB1B_HEADER + "\n")
    handle = _handle("db1b-coupon", archive)
    with pytest.raises(LookupError, match="no itineraries"):
        list(handle.paths())
    assert not archive.with_name(archive.stem + ".paths.tsv").exists()


def test_db1b_zero_passengers_never_yield_zero_weights(tmp_path: Path) -> None:
    archive = tmp_path / "zeropax.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "zeropax.csv",
            DB1B_HEADER + "\n1,m,1,2,2019,AAA,BBB,0.00,\n1,m,2,2,2019,BBB,CCC,0.00,\n",
        )
    handle = _handle("db1b-coupon", archive)
    assert list(handle.paths()) == [(("AAA", "BBB", "CCC"), 1.0)]


def test_db1b_missing_column_fails_loudly(tmp_path: Path) -> None:
    archive = tmp_path / "renamed.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("renamed.csv", "ItinID,SeqNum,Origin,Arrival\n1,1,AAA,BBB\n")
    handle = _handle("db1b-coupon", archive)
    with pytest.raises(LookupError, match="Dest"):
        list(handle.paths())


def test_db1b_memory_network(tmp_path: Path) -> None:
    handle = _handle("db1b-coupon", _db1b_archive(tmp_path))
    net = handle.to_state_network(order=2)
    # OAK -> SLC -> OAK carries 2 passengers into state link (OAK,SLC) -> (SLC,OAK).
    source = next(sid for sid in net.states if net.state_name(sid) == "OAK -> SLC")
    target = next(sid for sid in net.states if net.state_name(sid) == "SLC -> OAK")
    assert net.links[(source, target)] == 2.0


# -- db1b city view -------------------------------------------------------------


def test_db1b_city_itineraries_collapse_repeated_markets(tmp_path: Path) -> None:
    archive = _db1b_city_archive(tmp_path)
    cities = list(_handle("db1b-city", archive).paths())
    assert cities == [
        (("31703", "30977", "31703"), 4.0),  # MDW after ORD is still Chicago
        (("32457", "32101"), 2.0),  # OAK -> SFO -> LAX starts with one Bay Area node
        # itinerary 3 never leaves its city market, so it is no path at all
    ]
    # The airport view of the same archive keeps every airport, as before.
    airports = list(_handle("db1b-coupon", archive).paths())
    assert airports == [
        (("JFK", "ORD", "MDW", "LGA"), 4.0),
        (("OAK", "SFO", "LAX"), 2.0),
        (("OAK", "SFO"), 1.0),
    ]
    assert archive.with_name(archive.stem + ".paths.tsv").is_file()
    assert archive.with_name(archive.stem + ".city-paths.tsv").is_file()


def test_db1b_city_missing_columns_fail_loudly(tmp_path: Path) -> None:
    # The airport-only fixture header has no city market columns.
    handle = _handle("db1b-city", _db1b_archive(tmp_path))
    with pytest.raises(LookupError, match="OriginCityMarketID, DestCityMarketID"):
        list(handle.paths())
    assert not handle.archive_path.with_name(handle.archive_path.stem + ".city-paths.tsv").exists()


def test_db1b_city_names_from_the_bts_lookup(tmp_path: Path) -> None:
    archive = _db1b_city_archive(tmp_path)
    (tmp_path / "L_CITY_MARKET_ID.csv").write_text(CITY_MARKET_LOOKUP, encoding="utf-8")
    handle = _handle("db1b-city", archive)
    names = handle.names()
    assert names["31703"] == "New York City, NY (Metropolitan Area)"
    assert set(names) == {"30977", "31703", "32457"}
    with pytest.raises(LookupError, match="no 'names' table"):
        _handle("db1b-coupon", archive).names()


def test_bts_error_page_is_not_read_as_an_empty_lookup(tmp_path: Path) -> None:
    (tmp_path / "L_CITY_MARKET_ID.csv").write_text("<!DOCTYPE html><html>...", encoding="utf-8")
    handle = _handle("db1b-city", _db1b_city_archive(tmp_path))
    with pytest.raises(LookupError, match="Code,Description"):
        handle.names()


# -- airport coordinates --------------------------------------------------------


def test_coordinates_come_from_openflights(tmp_path: Path) -> None:
    archive = _db1b_archive(tmp_path)
    (tmp_path / "airports.dat").write_text(AIRPORTS_DAT, encoding="utf-8")
    where = _handle("db1b-coupon", archive).coordinates()
    assert where == {
        "OAK": (37.721298, -122.221001),
        "LAX": (33.94250107, -118.4079971),  # first row wins over the duplicate code
    }
    # Only db1b-coupon's nodes are IATA codes; the other entries ship no table.
    with pytest.raises(LookupError, match="no 'coordinates' table"):
        _handle("db1b-city", archive).coordinates()
    with pytest.raises(LookupError, match="no 'coordinates' table"):
        _handle("wikispeedia", _wikispeedia_bundle(tmp_path)).coordinates()


def test_side_tables_are_fetched_once_next_to_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from netgamut.paths import _models

    archive = _db1b_archive(tmp_path)
    fetched: list[str] = []
    real_download = _models._download_file

    def fake_download(url: str, target: Path, **kwargs: Any) -> Path:
        fetched.append(url)
        target.write_text(AIRPORTS_DAT, encoding="utf-8")
        return target

    monkeypatch.setattr(_models, "_download_file", fake_download)
    handle = _handle("db1b-coupon", archive)
    assert "OAK" in handle.coordinates()
    assert fetched == [paths.info("db1b-coupon").metadata_file("coordinates").url]
    assert (archive.parent / "airports.dat").is_file()

    # The real downloader is a no-op on a cached file, so this stays offline.
    monkeypatch.setattr(_models, "_download_file", real_download)
    assert "LAX" in handle.coordinates()


def test_describe_lists_side_tables(capsys: pytest.CaptureFixture[str]) -> None:
    text = paths.describe("db1b-coupon")
    assert "+ airports.dat  [coordinates]" in text
    text = paths.describe("db1b-city")
    assert "+ L_CITY_MARKET_ID.csv  [names]" in text
    assert "shared with db1b-coupon" in text


# -- load(), cache and CLI ----------------------------------------------------


@pytest.fixture
def offline_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Redirect the source's downloader at local fixture archives."""
    bundles = {
        "wikispeedia": _wikispeedia_bundle(tmp_path),
        "db1b-coupon": _db1b_archive(tmp_path),
    }
    seen: dict[str, Any] = {"urls": []}

    def fake_download(url: str, target: Path, **kwargs: Any) -> Path:
        seen["urls"].append(url)
        source = bundles["wikispeedia"] if "wikispeedia" in url else bundles["db1b-coupon"]
        shutil.copy(source, target)
        return Path(target)

    monkeypatch.setattr(paths, "_download_file", fake_download)
    return seen


def test_load_returns_a_path_data_handle(offline_paths: dict[str, Any]) -> None:
    assert not paths.is_cached("wikispeedia")
    data = paths.load("wikispeedia")
    assert isinstance(data, PathData)
    assert data.parser == "wikispeedia"
    assert paths.is_cached("wikispeedia")
    assert len(list(data.paths())) == 4


def test_load_builds_quarterly_urls(offline_paths: dict[str, Any]) -> None:
    paths.load("db1b-coupon", year=2023, quarter=3)
    assert offline_paths["urls"][-1].endswith("DB1BCoupon_2023_3.zip")
    assert paths.is_cached("db1b-coupon", year=2023, quarter=3)
    assert not paths.is_cached("db1b-coupon", year=2024, quarter=4)


def test_db1b_city_shares_the_coupon_download(offline_paths: dict[str, Any]) -> None:
    assert paths.info("db1b-city").cache_name == "db1b-coupon"
    assert not paths.is_cached("db1b-city")
    airports = paths.load("db1b-coupon")
    assert paths.is_cached("db1b-city")  # the same archive serves both views

    cities = paths.load("db1b-city")
    # Same URL resolved to the same cached file, so the real downloader
    # (a no-op on an existing target) never fetches the archive twice.
    assert len(set(offline_paths["urls"])) == 1
    assert cities.archive_path == airports.archive_path
    assert cities.parser == "db1b-city"
    assert paths.is_cached("db1b-city", year=2019, quarter=1)
    assert not paths.is_cached("db1b-city", year=2020, quarter=1)


def test_cli_lists_and_describes_the_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--source", "paths"]) == 0
    output = capsys.readouterr().out
    assert "db1b-coupon" in output
    assert "wikispeedia" in output

    assert main(["info", "db1b-coupon"]) == 0
    assert "DB1B" in capsys.readouterr().out

    # Ambiguous without a source, and the error must say so.
    assert main(["info", "wikispeedia"]) == 1
    assert "snap" in capsys.readouterr().err

    assert main(["info", "wikispeedia", "--source", "paths"]) == 0
    assert "51,318" in capsys.readouterr().out


# -- the real thing -----------------------------------------------------------


@pytest.mark.network
def test_wikispeedia_downloads_and_converts() -> None:
    data = paths.load("wikispeedia", progress=False)
    trails = list(data.paths())
    assert len(trails) == 51318
    net = data.to_state_network(order=2)
    assert net.num_physical_nodes > 4000
    assert net.num_state_nodes > net.num_physical_nodes


@pytest.mark.network
def test_db1b_url_is_alive_without_downloading() -> None:
    from netgamut._core.download import head

    dataset = paths.info("db1b-coupon")
    size = head(dataset.files[0].url)["bytes"]
    assert isinstance(size, int) and size > 100_000_000
