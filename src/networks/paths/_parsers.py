"""Recipes that turn each cached download into a stream of paths.

Every parser takes the :class:`~networks.paths._models.PathData` handle and
yields ``(nodes, weight)`` pairs. Anything expensive -- extracting a bundle,
grouping a scattered CSV -- happens once and lands next to the download in
the cache, so a second run streams a derived file instead of repeating the
work.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING
from urllib.parse import unquote

from .._core.cache import file_lock

if TYPE_CHECKING:
    from ._models import PathData

__all__ = ["PARSERS"]

PathStream = Iterator[tuple[tuple[str, ...], float]]

_FINISHED_PATHS = "paths_finished.tsv"

_DB1B_COLUMNS = ("ItinID", "SeqNum", "Origin", "Dest", "Passengers")
_DB1B_BATCH = 50_000
_DERIVE_LOCK_TIMEOUT = 3600.0


def wikispeedia(data: PathData) -> PathStream:
    """Finished navigation trails from the Wikispeedia game.

    Each line of ``paths_finished.tsv`` carries one trail as semicolon-joined,
    URL-encoded article names, where ``<`` is a back click. Back clicks are
    resolved against the browsing history, so the yielded trail is the
    sequence of articles actually visited: ``A;B;C;<;D`` becomes
    ``A, B, C, B, D``. Unfinished trails are not included.
    """
    directory = data.extract()
    source = next(directory.rglob(_FINISHED_PATHS), None)
    if source is None:
        raise LookupError(f"{data.name!r} bundle contains no {_FINISHED_PATHS}.")
    with source.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) < 4:
                continue
            trail: list[str] = []
            history: list[str] = []
            for token in fields[3].split(";"):
                if token == "<":
                    if history:
                        history.pop()
                    if history:
                        trail.append(history[-1])
                else:
                    article = unquote(token)
                    history.append(article)
                    trail.append(article)
            if trail:
                yield tuple(trail), 1.0


def db1b_coupon(data: PathData) -> PathStream:
    """Airline itineraries from a BTS DB1B coupon archive.

    The raw CSV is globally sorted by coupon sequence number, so one
    itinerary's rows are scattered across ~2 GB. The first call groups them
    through a temporary SQLite table into a derived ``.paths.tsv`` next to
    the download; later calls stream that file. Weights are the itinerary's
    sampled passengers (DB1B is a 10% ticket sample).
    """
    derived = data.archive_path.with_name(data.archive_path.stem + ".paths.tsv")
    if not derived.is_file():
        _derive_db1b(data.archive_path, derived)
    with derived.open(encoding="utf-8") as stream:
        for line in stream:
            airports, _, weight = line.rstrip("\n").partition("\t")
            if airports and weight:
                yield tuple(airports.split()), float(weight)


def _derive_db1b(archive: Path, derived: Path) -> None:
    """Group the coupon CSV by itinerary, once, publishing atomically."""
    with file_lock(derived, timeout=_DERIVE_LOCK_TIMEOUT):
        if derived.is_file():
            return
        database = derived.with_name(f".{derived.name}.{os.getpid()}.sqlite")
        partial = derived.with_name(f".{derived.name}.{os.getpid()}.part")
        try:
            # A crashed earlier run with a since-reused pid may have left
            # these behind; a stale table would derail CREATE TABLE.
            database.unlink(missing_ok=True)
            connection = sqlite3.connect(database)
            try:
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("PRAGMA synchronous = OFF")
                connection.execute(
                    "CREATE TABLE coupons "
                    "(itin TEXT, seq INTEGER, origin TEXT, dest TEXT, pax REAL)"
                )
                _load_coupons(archive, connection)
                with partial.open("w", encoding="utf-8") as out:
                    written = _write_itineraries(connection, out)
            finally:
                connection.close()
            if written == 0:
                # Publishing an empty file would cache "no paths" as truth.
                raise LookupError(
                    f"{archive.name} yielded no itineraries. The download may be "
                    "malformed; retry with load(..., force=True)."
                )
            os.replace(partial, derived)
        finally:
            partial.unlink(missing_ok=True)
            database.unlink(missing_ok=True)


def _load_coupons(archive: Path, connection: sqlite3.Connection) -> None:
    """Stream the zipped CSV into the coupons table without extracting it."""
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise LookupError(f"{archive.name} contains no CSV member.")
        with bundle.open(members[0]) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None:
                raise LookupError(f"{archive.name} has an empty CSV member.")
            # Columns are located by name: BTS occasionally reshuffles them.
            positions = {column.strip().casefold(): index for index, column in enumerate(header)}
            missing = [column for column in _DB1B_COLUMNS if column.casefold() not in positions]
            if missing:
                raise LookupError(
                    f"{archive.name} is missing the expected column(s) {', '.join(missing)}. "
                    f"BTS may have renamed its fields; found: {', '.join(header)}"
                )
            indices = [positions[column.casefold()] for column in _DB1B_COLUMNS]
            itin_at, seq_at, origin_at, dest_at, pax_at = indices
            widest = max(indices)

            batch: list[tuple[str, int, str, str, float]] = []
            insert = "INSERT INTO coupons VALUES (?, ?, ?, ?, ?)"
            for record in reader:
                if len(record) <= widest:
                    continue
                try:
                    seq = int(float(record[seq_at]))
                except ValueError:
                    continue
                try:
                    pax = float(record[pax_at])
                except ValueError:
                    pax = 1.0
                if not pax > 0:  # also catches NaN
                    pax = 1.0
                batch.append(
                    (
                        record[itin_at].strip(),
                        seq,
                        record[origin_at].strip(),
                        record[dest_at].strip(),
                        pax,
                    )
                )
                if len(batch) >= _DB1B_BATCH:
                    connection.executemany(insert, batch)
                    batch.clear()
            if batch:
                connection.executemany(insert, batch)
    connection.commit()


def _write_itineraries(connection: sqlite3.Connection, out: IO[str]) -> int:
    """Walk the coupons in itinerary order and write one path per line.

    Consecutive coupons chain when each destination is the next origin, which
    holds for well-formed itineraries; a gap keeps both airports so nothing is
    silently dropped. Returns the number of itineraries written, so the caller
    can tell an empty result from a successful one.
    """
    cursor = connection.execute(
        "SELECT itin, seq, origin, dest, pax FROM coupons ORDER BY itin, seq"
    )
    current: str | None = None
    airports: list[str] = []
    weight = 1.0
    written = 0

    def flush() -> None:
        nonlocal written
        if len(airports) >= 2:
            out.write(f"{' '.join(airports)}\t{weight:.10g}\n")
            written += 1

    for itinerary, _seq, origin, dest, pax in cursor:
        if itinerary != current:
            flush()
            current = itinerary
            airports = []
            weight = pax
        if origin and (not airports or origin != airports[-1]):
            airports.append(origin)
        if dest:
            airports.append(dest)
    flush()
    return written


#: Dataset parser name -> recipe. The name is stored in the catalog.
PARSERS: dict[str, Callable[[PathData], PathStream]] = {
    "wikispeedia": wikispeedia,
    "db1b-coupon": db1b_coupon,
}
