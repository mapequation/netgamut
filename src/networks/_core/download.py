"""HTTP access to snap.stanford.edu.

Uses only :mod:`urllib` so the installed package pulls in no HTTP dependency.
Downloads stream to a temporary file and are published with an atomic rename,
so a cancelled or crashed run can never leave a truncated file that a later run
would mistake for a complete one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cache import file_lock, read_sidecar, write_sidecar

__all__ = ["DownloadError", "download", "fetch_bytes", "fetch_text", "head"]

USER_AGENT = "mapequation-networks (+https://github.com/mapequation/networks)"

_CHUNK_SIZE = 1 << 20  # 1 MiB
_DEFAULT_TIMEOUT = 60.0
_RETRIES = 3
_BACKOFF_SECONDS = 1.5

#: Called with ``(downloaded_bytes, total_bytes_or_None)``.
ProgressCallback = Callable[[int, int | None], None]


class DownloadError(RuntimeError):
    """Raised when a file could not be retrieved from SNAP."""


def _request(url: str, *, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def _open(url: str, *, method: str = "GET", timeout: float = _DEFAULT_TIMEOUT) -> Any:
    """Open *url*, retrying transient failures.

    Returns the raw urllib response. Typed as ``Any`` because urllib may hand
    back an ``HTTPResponse`` or an ``addinfourl``, and the tests substitute a
    stand-in; all we rely on is ``.read()``, ``.headers`` and the context
    manager protocol.
    """
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            return urllib.request.urlopen(_request(url, method=method), timeout=timeout)
        except urllib.error.HTTPError as error:
            # Client errors will not fix themselves; fail immediately.
            if 400 <= error.code < 500:
                raise DownloadError(f"{error.code} {error.reason} for {url}") from error
            last = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise DownloadError(f"Could not fetch {url}: {last}") from last


def fetch_text(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Retrieve *url* and decode it as text."""
    with _open(url, timeout=timeout) as response:
        raw: bytes = response.read()
        charset: str = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_bytes(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> bytes:
    """Retrieve *url* into memory, undecoded.

    For build-time archives that are read once and never cached -- AHORN's
    catalog is a repository tarball. Anything a user asks for goes through
    :func:`download` instead, which streams to disk and caches.
    """
    with _open(url, timeout=timeout) as response:
        payload: bytes = response.read()
    return payload


def head(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> dict[str, str | int | None]:
    """Return size and validators for *url* without downloading its body."""
    try:
        with _open(url, method="HEAD", timeout=timeout) as response:
            headers = response.headers
            length = headers.get("Content-Length")
            return {
                "bytes": int(length) if length and length.isdigit() else None,
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
            }
    except DownloadError:
        # Not every server answers HEAD; callers treat this as "size unknown".
        return {"bytes": None, "etag": None, "last_modified": None}


def _terminal_progress(label: str) -> ProgressCallback:
    """A single-line progress indicator, used when stderr is a terminal."""
    state = {"last": 0.0}

    def report(done: int, total: int | None) -> None:
        now = time.monotonic()
        finished = total is not None and done >= total
        if not finished and now - state["last"] < 0.1:
            return
        state["last"] = now
        if total:
            fraction = done / total
            filled = int(fraction * 24)
            bar = "#" * filled + "-" * (24 - filled)
            text = f"\r{label} [{bar}] {fraction:6.1%} of {_human_bytes(total)}"
        else:
            text = f"\r{label} {_human_bytes(done)}"
        sys.stderr.write(text.ljust(72))
        if finished:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return report


def _human_bytes(count: int | None) -> str:
    if count is None:
        return "unknown size"
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover


def download(
    url: str,
    target: Path,
    *,
    force: bool = False,
    progress: ProgressCallback | bool | None = None,
    max_bytes: int | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download *url* to *target*, reusing an existing complete copy.

    Returns *target*. When the file is already cached this is a no-op, which is
    what makes repeated use across projects cheap.

    Set *force* to re-download, *max_bytes* to refuse anything larger than a
    given size, and *progress* to ``True`` for a terminal bar, ``False`` to
    silence it, or a callable to receive ``(downloaded, total)``.
    """
    target = Path(target)

    if target.exists() and not force:
        return target

    if progress is None:
        progress = sys.stderr.isatty()
    if progress is True:
        report: ProgressCallback | None = _terminal_progress(target.name)
    elif progress is False or progress is None:
        report = None
    else:
        report = progress

    with file_lock(target, timeout=_DEFAULT_TIMEOUT * 10):
        # Another process may have finished while we waited for the lock.
        if target.exists() and not force:
            return target

        if max_bytes is not None:
            size = head(url, timeout=timeout)["bytes"]
            if isinstance(size, int) and size > max_bytes:
                raise DownloadError(
                    f"{url} is {_human_bytes(size)}, which exceeds the "
                    f"{_human_bytes(max_bytes)} limit. Raise max_bytes to download it."
                )

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        digest = hashlib.sha256()
        downloaded = 0

        try:
            with _open(url, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
                if max_bytes is not None and total is not None and total > max_bytes:
                    raise DownloadError(
                        f"{url} is {_human_bytes(total)}, which exceeds the "
                        f"{_human_bytes(max_bytes)} limit. Raise max_bytes to download it."
                    )

                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if max_bytes is not None and downloaded > max_bytes:
                            raise DownloadError(
                                f"{url} exceeded the {_human_bytes(max_bytes)} limit "
                                f"while downloading."
                            )
                        if report is not None:
                            report(downloaded, total)

                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")

            if total is not None and downloaded != total:
                raise DownloadError(f"{url} returned {downloaded} bytes but declared {total}.")

            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)

        write_sidecar(
            target,
            {
                "url": url,
                "bytes": downloaded,
                "sha256": digest.hexdigest(),
                "etag": etag,
                "last_modified": last_modified,
                "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    return target


def extract_gzip(source: Path, target: Path) -> Path:
    """Decompress a ``.gz`` file, publishing the result atomically."""
    import gzip

    if target.exists():
        return target

    with file_lock(target, timeout=_DEFAULT_TIMEOUT * 10):
        if target.exists():
            return target
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        try:
            with gzip.open(source, "rb") as reader, partial.open("wb") as writer:
                shutil.copyfileobj(reader, writer, _CHUNK_SIZE)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)

    return target


def verify(target: Path) -> bool:
    """Re-hash a cached file and compare it with its recorded checksum."""
    recorded = read_sidecar(target).get("sha256")
    if not isinstance(recorded, str) or not target.is_file():
        return False
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest() == recorded
