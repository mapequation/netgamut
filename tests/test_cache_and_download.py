"""Cache location, atomicity, locking, and the download guard rails."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import pytest

from netgamut._core import cache as _cache
from netgamut._core import download as _download


# -- location -------------------------------------------------------------
def test_environment_variable_selects_the_cache(isolated_cache: Path) -> None:
    assert _cache.cache_dir() == isolated_cache
    assert isolated_cache.is_dir()


def test_set_cache_dir_overrides_the_environment(tmp_path: Path) -> None:
    elsewhere = tmp_path / "shared"
    try:
        assert _cache.set_cache_dir(elsewhere) == elsewhere
        assert _cache.cache_dir() == elsewhere
    finally:
        _cache.set_cache_dir(None)


@pytest.fixture
def default_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve the default (no environment, no override) cache inside tmp_path."""
    monkeypatch.delenv(_cache.CACHE_ENV_VAR, raising=False)
    monkeypatch.setattr(_cache, "_override", None)
    monkeypatch.setattr(_cache, "user_cache_dir", lambda app, author: str(tmp_path / app))
    return tmp_path


def test_the_cache_under_the_old_name_is_moved(default_cache: Path) -> None:
    legacy = default_cache / "networks" / "snap" / "files" / "ca-GrQc"
    legacy.mkdir(parents=True)
    (legacy / "ca-GrQc.txt").write_text("1 2\n")

    root = _cache.cache_dir()

    assert root == default_cache / "netgamut"
    assert (root / "snap" / "files" / "ca-GrQc" / "ca-GrQc.txt").read_text() == "1 2\n"
    assert not (default_cache / "networks").exists()


def test_a_foreign_networks_directory_is_left_alone(default_cache: Path) -> None:
    (default_cache / "networks" / "unrelated").mkdir(parents=True)

    root = _cache.cache_dir()

    assert root == default_cache / "netgamut"
    assert (default_cache / "networks" / "unrelated").is_dir()
    assert not any(root.iterdir())


def test_an_existing_new_cache_is_not_replaced(default_cache: Path) -> None:
    (default_cache / "networks" / "snap").mkdir(parents=True)
    (default_cache / "netgamut" / "konect").mkdir(parents=True)

    root = _cache.cache_dir()

    assert (root / "konect").is_dir()
    assert not (root / "snap").exists()
    assert (default_cache / "networks" / "snap").is_dir()


def test_cache_is_shared_not_per_project(isolated_cache: Path) -> None:
    """Two independent resolutions must land in the same directory.

    This is the property that makes a dataset download once per machine
    instead of once per project.
    """
    first = _cache.dataset_dir("snap", "ca-GrQc")
    second = _cache.dataset_dir("snap", "ca-GrQc")
    assert first == second == isolated_cache / "snap" / "files" / "ca-GrQc"


def test_dataset_names_are_sanitised_into_one_path_component() -> None:
    directory = _cache.dataset_dir("snap", "../../etc/passwd")
    assert directory.parent.name == "files"
    assert "/" not in directory.name
    assert ".." not in directory.name


# -- atomicity and locking -----------------------------------------------
def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    _cache.write_atomic(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [target]


def test_lock_is_released_even_when_the_body_raises(tmp_path: Path) -> None:
    target = tmp_path / "thing"
    with pytest.raises(RuntimeError), _cache.file_lock(target):
        raise RuntimeError("boom")
    assert not list(tmp_path.glob(".*.lock"))


def test_a_stale_lock_is_broken(tmp_path: Path, monkeypatch) -> None:
    """A process killed mid-download must not block the cache forever."""
    target = tmp_path / "thing"
    lock = target.with_name(f".{target.name}.lock")
    lock.write_text("999999 0")
    monkeypatch.setattr(_cache, "_LOCK_STALE_SECONDS", -1)
    with _cache.file_lock(target, timeout=1):
        pass
    assert not lock.exists()


def test_a_fresh_lock_with_a_dead_owner_is_stale(tmp_path: Path) -> None:
    """A hard-killed process must not block its retry until the age cutoff."""
    import subprocess
    import sys

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    lock = tmp_path / ".thing.lock"

    lock.write_text(f"{gone.pid} 0")
    assert _cache._lock_is_stale(lock)

    lock.write_text(f"{os.getpid()} 0")  # a live owner is never stale
    assert not _cache._lock_is_stale(lock)

    lock.write_text("")  # just created, not yet written: leave it alone
    assert not _cache._lock_is_stale(lock)


def test_lock_times_out_rather_than_hanging(tmp_path: Path) -> None:
    """Correctness rests on the atomic rename, so a busy lock is not fatal."""
    target = tmp_path / "thing"
    lock = target.with_name(f".{target.name}.lock")
    lock.write_text("1 0")
    try:
        with _cache.file_lock(target, timeout=0.05):
            pass  # proceeds unlocked instead of deadlocking
    finally:
        lock.unlink(missing_ok=True)


# -- reporting and clearing ----------------------------------------------
def test_cache_info_reports_sizes(isolated_cache: Path) -> None:
    directory = _cache.dataset_dir("snap", "ca-GrQc")
    (directory / "ca-GrQc.txt.gz").write_bytes(b"x" * 100)
    _cache.write_sidecar(directory / "ca-GrQc.txt.gz", {"bytes": 100})

    info = _cache.cache_info()
    assert info.bytes == 100
    # The .meta.json sidecar is bookkeeping, not payload.
    assert info.datasets["snap/ca-GrQc"].files == ("ca-GrQc.txt.gz",)


def test_clear_cache_targets_one_dataset(isolated_cache: Path) -> None:
    for name in ("keep", "drop"):
        (_cache.dataset_dir("snap", name) / "f.txt").write_bytes(b"y" * 10)
    assert _cache.clear_cache("drop") == 10
    assert _cache.dataset_dir("snap", "keep").exists()
    assert not (isolated_cache / "snap" / "files" / "drop").exists()


def test_clear_cache_removes_everything(isolated_cache: Path) -> None:
    (_cache.dataset_dir("snap", "a") / "f.txt").write_bytes(b"z" * 5)
    assert _cache.clear_cache() == 5
    assert _cache.cache_info().datasets == {}


# -- downloading ----------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]):
        self._stream = iter([payload[i : i + 8] for i in range(0, len(payload), 8)])
        self.headers = _FakeHeaders(headers)

    def read(self, _size: int = -1) -> bytes:
        return next(self._stream, b"")

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None


class _FakeHeaders(dict):
    def get_content_charset(self):
        return "utf-8"


@pytest.fixture
def served(monkeypatch):
    """Serve a fixed payload in place of snap.stanford.edu."""
    calls: list[str] = []

    def serve(payload: bytes, **headers: str):
        def fake_open(url, *, method="GET", timeout=None):
            calls.append(url)
            merged = {"Content-Length": str(len(payload)), **headers}
            return _FakeResponse(b"" if method == "HEAD" else payload, merged)

        monkeypatch.setattr(_download, "_open", fake_open)
        return calls

    return serve


def test_download_writes_the_payload_and_a_sidecar(tmp_path: Path, served) -> None:
    served(b"hello world")
    target = tmp_path / "f.txt"
    _download.download("https://x/f.txt", target, progress=False)

    assert target.read_bytes() == b"hello world"
    sidecar = _cache.read_sidecar(target)
    assert sidecar["bytes"] == 11
    assert sidecar["sha256"]
    assert _download.verify(target)


def test_second_download_reuses_the_cache(tmp_path: Path, served) -> None:
    """The whole point: importing from another project must not refetch."""
    calls = served(b"payload")
    target = tmp_path / "f.txt"
    _download.download("https://x/f.txt", target, progress=False)
    _download.download("https://x/f.txt", target, progress=False)
    assert len(calls) == 1


def test_force_refetches(tmp_path: Path, served) -> None:
    calls = served(b"payload")
    target = tmp_path / "f.txt"
    _download.download("https://x/f.txt", target, progress=False)
    _download.download("https://x/f.txt", target, progress=False, force=True)
    assert len(calls) == 2


def test_truncated_transfer_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A short read must not be published as a complete file."""

    def fake_open(url, *, method="GET", timeout=None):
        return _FakeResponse(b"partial", {"Content-Length": "999"})

    monkeypatch.setattr(_download, "_open", fake_open)
    target = tmp_path / "f.txt"
    with pytest.raises(_download.DownloadError, match="declared"):
        _download.download("https://x/f.txt", target, progress=False)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.part"))


def test_max_bytes_refuses_before_transferring(tmp_path: Path, served) -> None:
    served(b"x" * 5000)
    with pytest.raises(_download.DownloadError, match="exceeds"):
        _download.download("https://x/big.gz", tmp_path / "big.gz", progress=False, max_bytes=100)
    assert not (tmp_path / "big.gz").exists()


def test_progress_callback_receives_totals(tmp_path: Path, served) -> None:
    served(b"x" * 32)
    seen: list[tuple[int, int | None]] = []
    _download.download(
        "https://x/f", tmp_path / "f", progress=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (32, 32)


def test_client_errors_do_not_retry(tmp_path: Path, monkeypatch) -> None:
    import urllib.error

    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(_download.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(_download.DownloadError, match="404"):
        _download.fetch_text("https://x/missing.html")
    assert len(attempts) == 1


def test_gzip_extraction_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "d.txt.gz"
    source.write_bytes(gzip.compress(b"1\t2\n"))
    target = tmp_path / "d.txt"
    assert _download.extract_gzip(source, target).read_bytes() == b"1\t2\n"
    before = os.stat(target).st_mtime_ns
    assert _download.extract_gzip(source, target).read_bytes() == b"1\t2\n"
    assert os.stat(target).st_mtime_ns == before
