"""Where downloaded data lives, and how concurrent readers stay out of each
other's way.

The cache is deliberately *user-level*, not project-level: importing
``netgamut`` from ten different virtualenvs, notebooks or checkouts resolves to
one directory, so any given dataset is fetched from Stanford exactly once per
machine.

Layout::

    <root>/snap/catalog.json          refreshed catalog, overrides the vendored one
    <root>/snap/files/<name>/<file>   downloaded artifacts, plus .meta.json sidecars

Override the root with the ``NETGAMUT_CACHE_DIR`` environment variable, or at
runtime with :func:`set_cache_dir`.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

__all__ = [
    "CACHE_ENV_VAR",
    "CacheInfo",
    "CachedDataset",
    "cache_dir",
    "cache_info",
    "clear_cache",
    "set_cache_dir",
]

CACHE_ENV_VAR = "NETGAMUT_CACHE_DIR"

_APP_NAME = "netgamut"
_APP_AUTHOR = "mapequation"

# The default cache was named after the package's old import name. It is moved
# to the new name on first use rather than abandoned, so nothing re-downloads.
# "networks" is generic enough to belong to another tool, so it is only taken
# when it holds one of our own source directories.
_LEGACY_APP_NAME = "networks"
_LEGACY_MARKERS = ("snap", "netzschleuder", "konect", "ogb", "pyg", "paths", "ahorn", "lfr")

# Set by set_cache_dir(); takes precedence over the environment.
_override: Path | None = None

# A lock older than this is assumed to belong to a process that died mid
# download. Generous, because some SNAP archives are tens of gigabytes.
_LOCK_STALE_SECONDS = 6 * 60 * 60
_LOCK_POLL_SECONDS = 0.25


def cache_dir() -> Path:
    """Return the root cache directory, creating it if necessary."""
    if _override is not None:
        root = _override
    else:
        env_value = os.environ.get(CACHE_ENV_VAR)
        root = Path(env_value).expanduser() if env_value else _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_cache_dir() -> Path:
    root = Path(user_cache_dir(_APP_NAME, _APP_AUTHOR))
    if not root.exists():
        _adopt_legacy_cache(root)
    return root


def _adopt_legacy_cache(root: Path) -> None:
    """Move the cache kept under the old package name to *root*, if there is one."""
    legacy = Path(user_cache_dir(_LEGACY_APP_NAME, _APP_AUTHOR))
    if not any((legacy / marker).is_dir() for marker in _LEGACY_MARKERS):
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    # A concurrent process may have moved it first; then there is nothing left
    # to move and the new directory already holds the data.
    with contextlib.suppress(OSError):
        legacy.rename(root)


def set_cache_dir(path: str | os.PathLike[str] | None) -> Path:
    """Point the cache at *path* for the rest of this process.

    Passing ``None`` restores the default lookup. Returns the directory now
    in effect.
    """
    global _override
    _override = Path(path).expanduser() if path is not None else None
    return cache_dir()


def source_dir(source: str) -> Path:
    """Directory for one data source, e.g. ``"snap"``."""
    path = cache_dir() / _safe_name(source)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(source: str, name: str) -> Path:
    """Directory holding the downloaded files for one dataset.

    Namespaced by source, so two sources publishing a network under the same
    name cannot collide.
    """
    path = source_dir(source) / "files" / _safe_name(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(name: str) -> str:
    """Make a dataset or file name safe to use as a single path component."""
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in name)
    return cleaned.strip("._") or "unnamed"


def sidecar_path(target: Path) -> Path:
    return target.with_name(target.name + ".meta.json")


def read_sidecar(target: Path) -> dict[str, object]:
    path = sidecar_path(target)
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_sidecar(target: Path, data: dict[str, object]) -> None:
    write_atomic(sidecar_path(target), json.dumps(data, indent=2).encode("utf-8"))


def write_atomic(target: Path, payload: bytes) -> None:
    """Write *payload* to *target* so readers never observe a partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(target: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Serialise work on *target* across processes.

    Correctness does not depend on this lock -- every writer publishes its
    result with an atomic rename regardless. The lock exists so that two
    processes asked for the same dataset at the same time do not both spend
    twenty minutes downloading it.
    """
    lock_path = target.with_name(f".{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    handle: int | None = None

    while True:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if _lock_is_stale(lock_path):
                lock_path.unlink(missing_ok=True)
                continue
            if deadline is not None and time.monotonic() > deadline:
                # Proceed unlocked rather than fail: the atomic rename keeps
                # the result correct even if we duplicate the work.
                yield
                return
            time.sleep(_LOCK_POLL_SECONDS)
        except OSError as error:  # pragma: no cover - read-only cache dirs
            if error.errno in (errno.EACCES, errno.EROFS):
                yield
                return
            raise

    try:
        os.write(handle, f"{os.getpid()} {time.time()}".encode())
        os.close(handle)
        handle = None
        yield
    finally:
        if handle is not None:  # pragma: no cover - defensive
            os.close(handle)
        lock_path.unlink(missing_ok=True)


def _lock_is_stale(lock_path: Path) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    if age > _LOCK_STALE_SECONDS:
        return True
    # A lock whose owner has died is stale regardless of age; without this, a
    # hard-killed download or derivation blocks its retry for hours. The pid
    # is only meaningful on the machine that wrote it: on a shared cache a
    # foreign pid may coincide with a live local process, in which case this
    # falls back to the age check -- no worse than before, and the atomic
    # rename keeps results correct even if a live lock is ever stolen.
    try:
        pid = int(lock_path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False  # just created and not yet written, or unreadable
    if pid <= 0:
        return False
    return not _process_alive(pid)


def _process_alive(pid: int) -> bool:
    """Whether a process with this pid runs on this machine; True when unsure."""
    if sys.platform == "win32":
        # os.kill(pid, 0) is not a probe on Windows: signal 0 is CTRL_C_EVENT,
        # which would interrupt a live owner and fails for a dead one with an
        # error that does not say "no such process".
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        process_query_limited_information, still_active, error_invalid_parameter = 0x1000, 259, 87
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # No such pid; any other failure (access denied) means it exists.
            return ctypes.get_last_error() != error_invalid_parameter
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            # An exited process stays openable while anyone holds a handle to it.
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM: alive but owned by someone else
    return True


@dataclass(frozen=True, slots=True)
class CachedDataset:
    """What is on disk for one dataset."""

    source: str
    name: str
    files: tuple[str, ...]
    bytes: int

    @property
    def key(self) -> str:
        return f"{self.source}/{self.name}"


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """A summary of the local cache."""

    path: Path
    bytes: int
    #: Keyed ``"<source>/<name>"``, so the two sources stay distinguishable.
    datasets: dict[str, CachedDataset]
    #: Sources with a refreshed catalog in the cache shadowing the bundled one.
    refreshed_catalogs: tuple[str, ...] = ()

    @property
    def catalog_refreshed(self) -> bool:
        return bool(self.refreshed_catalogs)

    def __str__(self) -> str:
        return f"{len(self.datasets)} dataset(s), {self.bytes / 1e6:.1f} MB in {self.path}"


def cache_info() -> CacheInfo:
    """Summarise what is currently cached on this machine, across all sources."""
    root = cache_dir()
    datasets: dict[str, CachedDataset] = {}
    refreshed: list[str] = []
    total = 0

    for source_path in sorted(root.iterdir()) if root.is_dir() else []:
        if not source_path.is_dir():
            continue
        source = source_path.name
        if (source_path / "catalog.json").is_file():
            refreshed.append(source)

        files_root = source_path / "files"
        if not files_root.is_dir():
            continue
        for entry in sorted(files_root.iterdir()):
            if not entry.is_dir():
                continue
            names: list[str] = []
            size = 0
            for item in sorted(entry.iterdir()):
                # Sidecars, locks and in-flight parts are bookkeeping, not data.
                if not item.is_file() or item.name.endswith(".meta.json"):
                    continue
                if item.name.startswith(".") or item.name.endswith(".lock"):
                    continue
                names.append(item.name)
                size += item.stat().st_size
            if names:
                record = CachedDataset(source, entry.name, tuple(names), size)
                datasets[record.key] = record
                total += size

    return CacheInfo(
        path=root,
        bytes=total,
        datasets=datasets,
        refreshed_catalogs=tuple(refreshed),
    )


def clear_cache(name: str | None = None, *, source: str | None = None) -> int:
    """Delete cached downloads and return the number of bytes reclaimed.

    With *name*, clears just that dataset; with *source*, everything from that
    source; with neither, the whole cache including any refreshed catalogs. The
    catalogs shipped inside the package are never touched.
    """
    import shutil

    root = cache_dir()
    sources = [source] if source else [p.name for p in root.iterdir() if p.is_dir()]
    targets: list[Path] = []

    for current in sources:
        base = root / _safe_name(current)
        if name is None:
            targets.append(base / "files")
            refreshed = base / "catalog.json"
            if refreshed.is_file():
                targets.append(refreshed)
        else:
            targets.append(base / "files" / _safe_name(name))

    reclaimed = 0
    for target in targets:
        if not target.exists():
            continue
        if target.is_file():
            reclaimed += target.stat().st_size
            target.unlink()
            continue
        for item in target.rglob("*"):
            if item.is_file():
                reclaimed += item.stat().st_size
        shutil.rmtree(target, ignore_errors=True)

    return reclaimed
