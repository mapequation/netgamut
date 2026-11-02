"""Public API surface and the command line, plus the opt-in network tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from netgamut import __main__ as cli
from netgamut import snap
from netgamut._core import download as _download


@pytest.fixture
def offline_snap(monkeypatch):
    """Serve a toy edge list instead of snap.stanford.edu."""
    payload = gzip.compress(b"# toy\n1\t2\n2\t3\n")
    calls: list[str] = []

    def fake_download(url, target, **kwargs):
        # Mirror the real download()'s contract: a cached file is a no-op, so
        # `calls` records genuine transfers only.
        if target.exists() and not kwargs.get("force"):
            return target
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(snap, "_download_file", fake_download)
    return calls


# -- API ------------------------------------------------------------------
def test_categories_match_the_snap_page() -> None:
    names = [category.name for category in snap.categories()]
    assert names[0] == "Social networks"
    assert "Networks with ground-truth communities" in names
    assert len(names) == len(set(names))


def test_datasets_can_be_filtered_by_category() -> None:
    collaboration = snap.datasets("Collaboration networks")
    assert {item.name for item in collaboration} >= {"ca-GrQc", "ca-AstroPh"}


def test_info_never_downloads(offline_snap) -> None:
    snap.info("ca-GrQc")
    assert offline_snap == []


def test_load_downloads_then_caches(offline_snap) -> None:
    first = snap.load("ca-GrQc", progress=False)
    assert first.dataset.name == "ca-GrQc"
    assert list(first.edges(cast=int)) == [(1, 2), (2, 3)]
    assert snap.is_cached("ca-GrQc")

    snap.load("ca-GrQc", progress=False)
    assert len(offline_snap) == 1


def test_load_accepts_an_explicit_file(offline_snap) -> None:
    network = snap.load("ego-Facebook", file="facebook.tar.gz", progress=False)
    assert network.file.name == "facebook.tar.gz"


def test_load_rejects_an_unknown_file(offline_snap) -> None:
    with pytest.raises(KeyError):
        snap.load("ca-GrQc", file="nope.txt.gz", progress=False)


def test_load_reports_withdrawn_datasets_clearly(offline_snap) -> None:
    withdrawn = [item for item in snap.datasets() if not item.files]
    if not withdrawn:
        pytest.skip("SNAP currently publishes files for every dataset")
    with pytest.raises(LookupError, match="withdrawn"):
        snap.load(withdrawn[0].name)


def test_load_ground_truth_picks_the_variant(offline_snap) -> None:
    network = snap.load_ground_truth("com-DBLP", variant="top5000", progress=False)
    assert "top5000" in network.file.name
    assert network.dataset.has_ground_truth


def test_load_ground_truth_refuses_datasets_without_one(offline_snap) -> None:
    with pytest.raises(LookupError, match="no ground-truth"):
        snap.load_ground_truth("ca-GrQc")


def test_unknown_dataset_is_not_reported_as_cached() -> None:
    assert not snap.is_cached("does-not-exist")


def test_benchmark_selection_is_deduplicated_and_bounded() -> None:
    """The shape an Infomap benchmark suite is built from."""
    suite = snap.search(ground_truth=True, max_edges=5_000_000)
    names = [item.name for item in suite]
    assert names, "expected at least one small ground-truth dataset"
    assert len(names) == len(set(names))
    assert all(item.edges is not None and item.edges <= 5_000_000 for item in suite)
    assert all(item.has_ground_truth for item in suite)


# -- CLI ------------------------------------------------------------------
def test_cli_categories(capsys) -> None:
    assert cli.main(["categories"]) == 0
    assert "Social networks" in capsys.readouterr().out


def test_cli_list_filters(capsys) -> None:
    assert cli.main(["list", "--category", "Collaboration networks"]) == 0
    out = capsys.readouterr().out
    assert "ca-GrQc" in out
    assert "dataset(s)" in out


def test_cli_info(capsys) -> None:
    assert cli.main(["info", "ca-GrQc"]) == 0
    assert "Dataset statistics" in capsys.readouterr().out


def test_cli_reports_unknown_datasets(capsys) -> None:
    assert cli.main(["info", "not-a-dataset"]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_get_prints_the_cached_path(offline_snap, capsys) -> None:
    assert cli.main(["get", "ca-GrQc"]) == 0
    printed = capsys.readouterr().out.strip()
    assert Path(printed).is_file()


def test_cli_cache_reports_state(offline_snap, capsys) -> None:
    cli.main(["get", "ca-GrQc"])
    capsys.readouterr()
    assert cli.main(["cache"]) == 0
    out = capsys.readouterr().out
    assert "ca-GrQc" in out
    assert "Location" in out


# -- opt-in network tests -------------------------------------------------
@pytest.mark.network
def test_really_downloads_a_small_dataset() -> None:
    network = snap.load("ca-GrQc", progress=False)
    assert network.path.is_file()
    assert sum(1 for _ in network.edges()) == 28980  # both directions listed


@pytest.mark.network
def test_live_index_still_parses() -> None:
    """Fails loudly if SNAP redesigns its markup."""
    from netgamut.snap._scrape import INDEX_URL, SNAP_BASE_URL, parse_index

    categories, datasets = parse_index(_download.fetch_text(INDEX_URL), SNAP_BASE_URL)
    assert len(categories) >= 20
    assert len(datasets) >= 100
