"""What the fields of an edge-list row hold, and reading them as arrays.

Beyond source and target, a downloaded edge list may carry a weight, a
timestamp, a layer, a sign, a rating -- in an order that differs by source and
by dataset. A third column is a weight in one file and a Unix timestamp in the
next, so guessing is wrong: each source declares a :class:`Columns` record for
what it loads, from the metadata it publishes, and everything that reads the
file -- :meth:`Network.to_infomap`, the writers, :attr:`Network.times` --
follows it.

Times are binned the way :func:`numpy.histogram_bin_edges` bins: ``bins`` is a
count, a sequence of edges or an estimator name, and ``range`` clips. Bin ``i``
is the half-open ``[edges[i], edges[i + 1])``, except the last, which is closed,
and times outside every bin are left out.
"""

from __future__ import annotations

import csv
import math
import re
from array import array
from collections.abc import Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Columns", "Table", "allow_long_fields", "assign_bins", "read_table"]

# The numeric prefix a C++ stream reads as a double -- what Infomap's own parser
# makes of a weight field -- so "1_000" is 1 here too, not Python's 1000.
LEADING_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# csv stops at fields over 128 KiB, and some Netzschleuder fields are longer:
# littlesis's. 2**31 - 1 is the most the limit takes on every platform.
_FIELD_LIMIT = 2**31 - 1


def allow_long_fields() -> None:
    """Lift csv's field size limit, which a few CSV files exceed.

    The limit is process-wide, so this only ever raises it.
    """
    if csv.field_size_limit() < _FIELD_LIMIT:
        csv.field_size_limit(_FIELD_LIMIT)


@dataclass(frozen=True)
class Columns:
    """Which field of each row holds what. Field 0 is the source, 1 the target.

    ``None`` means the file has no such field. The default -- no weight, no
    time, no layer -- reads every file as an unweighted edge list, which is
    the safe reading of a column nobody has vouched for.
    """

    #: Link weight, read as a number.
    weight: int | None = None
    #: Timestamp of the link, for :attr:`Network.times` and time snapshots.
    time: int | None = None
    #: Layer of the link, for :meth:`Network.to_multilayer`.
    layer: int | None = None
    #: Every field's name, where the source publishes them.
    names: tuple[str, ...] = ()
    #: The field separator, where the source says what it is: ``","`` for a CSV
    #: whose quoted fields hold commas or line breaks, ``"\t"`` for a TSV whose
    #: fields hold spaces. ``None`` detects it from the first lines.
    delimiter: str | None = None
    #: The character that escapes a quote inside a quoted field, where the
    #: source escapes quotes rather than doubling them: Netzschleuder writes
    #: ``\"``. Only a declared *delimiter* reads it.
    escapechar: str | None = None
    #: Whether the first row that is not a comment names the fields rather than
    #: holding a link, as in SNAP's Reddit hyperlink TSVs.
    header_row: bool = False

    def find(self, name: str) -> int:
        """The index of the field called *name*.

        Raises:
            LookupError: No field has that name.
        """
        if name in self.names:
            return self.names.index(name)
        known = ", ".join(self.names) if self.names else "none published"
        raise LookupError(f"no column named {name!r}; columns: {known}.")

    def __str__(self) -> str:
        roles = {self.weight: "weight", self.time: "time", self.layer: "layer"}
        if not self.names:
            width = max((i for i in roles if i is not None), default=1) + 1
            names: Sequence[str] = ["source", "target", *(["?"] * (width - 2))]
        else:
            names = self.names
        return " ".join(
            f"{name} ({roles[i]})" if i in roles and i is not None else name
            for i, name in enumerate(names)
        )


def is_int_label(text: str) -> bool:
    """Whether *text* reads as an integer id and back unchanged: ``"7"``, not ``"007"``.

    The one rule for when labels from a file become integers -- here, in
    :meth:`Network.to_pajek` and in :meth:`Network.to_link_list` -- so a label
    with a leading zero stays the string it is everywhere.
    """
    return text.isascii() and text.isdigit() and (text == "0" or text[0] != "0")


class _Labels:
    """Node labels: compact integers while every label is an integer, strings after."""

    def __init__(self) -> None:
        self.ints: array[int] | None = array("q")
        self.strings: list[str] = []

    def add(self, text: str) -> None:
        if self.ints is not None:
            if is_int_label(text):
                try:
                    self.ints.append(int(text))
                    return
                except OverflowError:
                    pass
            self.strings = [str(value) for value in self.ints]
            self.ints = None
        self.strings.append(text)

    def values(self) -> Sequence[Hashable]:
        return self.ints if self.ints is not None else self.strings


class _Values:
    """A numeric column: integers, then floats on the first non-integer, then text.

    A missing or empty field is NaN, so the column turns float; text that is
    not a number keeps the column as strings for the caller to interpret, with
    missing fields as empty strings, which numpy reads as NaT.
    """

    def __init__(self) -> None:
        self.kind = "int"
        self.data: Any = array("q")

    def _widen(self, kind: str) -> None:
        if kind == "float":
            self.data = array("d", (float(value) for value in self.data))
        else:
            self.data = [
                "" if isinstance(value, float) and math.isnan(value) else str(value)
                for value in self.data
            ]
        self.kind = kind

    def add(self, text: str | None) -> None:
        if text is not None and not text.strip():
            text = None
        if text is None:
            if self.kind == "int":
                self._widen("float")
            self.data.append(math.nan if self.kind == "float" else "")
            return
        if self.kind == "int":
            try:
                self.data.append(int(text))
                return
            except (ValueError, OverflowError):
                self._widen("float")
        if self.kind == "float":
            try:
                self.data.append(float(text))
                return
            except ValueError:
                self._widen("str")
        self.data.append(text)

    def array(self) -> Any:
        import numpy as np

        if self.kind == "int":
            return np.frombuffer(self.data, dtype=np.int64).copy()
        if self.kind == "float":
            return np.frombuffer(self.data, dtype=np.float64).copy()
        try:
            # ISO 8601 dates and times, the usual text form of a timestamp.
            return np.array(self.data, dtype="datetime64[s]")
        except ValueError:
            return np.array(self.data, dtype=object)


@dataclass
class Table:
    """The fields of every edge row, column by column, in file order."""

    sources: Sequence[Hashable]
    targets: Sequence[Hashable]
    #: Floats, or ``None`` when the file has no weight column.
    weights: Any
    #: The time column as an array, or ``None`` when not asked for.
    times: Any
    #: Layer labels, ``None`` for a row without one; ``None`` when not asked for.
    layers: Sequence[Hashable | None] | None


def read_table(
    rows: Iterable[tuple[str, ...]],
    columns: Columns,
    *,
    time: int | None = None,
    layer: int | None = None,
    nodes: bool = True,
) -> Table:
    """Read *rows* once into columns: labels, weights, and the asked-for extras.

    Labels are integers when every label is digits, as :meth:`Network.to_pajek`
    decides, and strings otherwise. Rows with fewer than two fields are
    skipped, as everywhere else in the package.
    """
    import numpy as np

    # One label column for both ends, so sources and targets are always the
    # same kind: integers only if every label on either side is digits.
    ends = _Labels()
    weights = array("d") if columns.weight is not None else None
    times = _Values() if time is not None else None
    layers: list[str | None] | None = [] if layer is not None else None
    for fields in rows:
        if len(fields) < 2:
            continue
        if nodes:
            ends.add(fields[0])
            ends.add(fields[1])
        if weights is not None:
            weights.append(weight_of(fields, columns.weight))
        if times is not None and time is not None:
            times.add(fields[time] if len(fields) > time else None)
        if layers is not None and layer is not None:
            text = fields[layer].strip() if len(fields) > layer else ""
            layers.append(text or None)
    labels = ends.values()
    return Table(
        sources=labels[0::2],
        targets=labels[1::2],
        weights=None if weights is None else np.frombuffer(weights, dtype=np.float64).copy(),
        times=None if times is None else times.array(),
        layers=None if layers is None else _layer_labels(layers),
    )


def _layer_labels(texts: list[str | None]) -> list[Hashable | None]:
    """Integers when every present label is one; ``None`` marks a missing layer."""
    if all(text is None or is_int_label(text) for text in texts):
        return [None if text is None else int(text) for text in texts]
    return list(texts)


def weight_of(fields: Sequence[str], column: int | None) -> float:
    """The weight in *fields*, read as Infomap's parser reads it; 1.0 if absent."""
    if column is None or len(fields) <= column:
        return 1.0
    number = LEADING_NUMBER.match(fields[column])
    return float(number.group()) if number else 1.0


def numeric(times: Any) -> tuple[Any, Any]:
    """``(values, missing)``: times as numbers for binning, and which are absent.

    Datetimes become seconds since the epoch; NaT and NaN are missing.
    """
    import numpy as np

    if np.issubdtype(times.dtype, np.datetime64):
        missing = np.isnat(times)
        return times.astype("datetime64[s]").astype(np.int64), missing
    if times.dtype == object:
        present = [value for value in times if value]
        raise ValueError(
            "the time column holds text that is neither a number nor an ISO date, "
            f"e.g. {present[0] if present else times[0]!r}; it cannot be binned."
        )
    if times.dtype.kind == "f":
        return times, np.isnan(times)
    return times, np.zeros(len(times), dtype=bool)


def _seconds(value: Any) -> Any:
    """A datetime-like value (or a sequence of them) as seconds since the epoch."""
    import numpy as np

    as_datetime = np.asarray(value, dtype="datetime64[us]")
    return as_datetime.astype(np.int64) / 1e6


def assign_bins(times: Any, bins: Any, range: tuple[Any, Any] | None) -> tuple[Any, Any]:
    """``(edges, index)``: :func:`numpy.histogram_bin_edges` and each time's bin.

    *index* is ``-1`` for a time outside every bin, or missing. For datetime
    times, *bins* edges and *range* may be datetimes or ISO strings, and the
    edges come back as microsecond datetimes -- the same edges the times were
    binned with, so a bin never holds a time its reported span excludes.
    """
    import numpy as np

    values, missing = numeric(times)
    dated = np.issubdtype(times.dtype, np.datetime64)
    if dated:
        if range is not None:
            range = (float(_seconds(range[0])), float(_seconds(range[1])))
        if not isinstance(bins, (int, np.integer, str)):
            bins = _seconds(bins)
    edges = np.histogram_bin_edges(values[~missing], bins=bins, range=range)
    if dated:
        edges = np.round(edges * 1e6).astype(np.int64).astype("datetime64[us]")
        values = values.astype(np.float64)
        bounds = edges.astype(np.int64) / 1e6
    else:
        bounds = edges
    index = np.searchsorted(bounds, values, side="right") - 1
    last = len(bounds) - 2
    index[values == bounds[-1]] = last  # the last bin is closed, as in numpy.histogram
    index[(index < 0) | (index > last) | missing] = -1
    return edges, index


def groups(index: Any, count: int) -> Iterator[tuple[int, Any]]:
    """Yield ``(bin, row indices)`` for every bin, empty ones included, in order."""
    import numpy as np

    order = np.argsort(index, kind="stable")
    ordered = index[order]
    bounds = np.searchsorted(ordered, np.arange(-1, count + 1))
    for b in np.arange(count):
        yield int(b), order[bounds[b + 1] : bounds[b + 2]]
