"""The web mirror's files: a network as Parquet, links and nodes apart.

Format 2 of the mirror, which :func:`netgamut._export.export_networks` builds.
Each network is an ``<name>.links.parquet`` file -- one row per row of the
source's link list, with ``source``, ``target`` and, where the network has
them, ``weight``, ``time`` and ``layer`` -- and, where the source names its
nodes or has node metadata, an ``<name>.nodes.parquet`` file.

The link list is read twice and never held: the first pass decides each
column's type, the second writes it in row groups. Columns are typed the way
:mod:`netgamut._core.columns` reads them -- integers, then floats, then dates,
then text -- except that a missing value stays missing rather than turning a
column of integers into floats.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._core import columns as _columns
from ._core.multilayer import node_identity
from ._core.nodes import NodeTable

#: Rows per row group: few enough that a reader holds one at a time.
ROW_GROUP = 1_000_000
#: The longest node name kept whole, and the longest value of a column that is
#: not free text.
TEXT_LIMIT = 100
#: What str.splitlines() breaks a line at: besides "\n" and "\r", Unicode's line
#: and paragraph separators, which littlesis's biographies hold.
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")

_INT32 = 2**31 - 1
_INT64 = 2**63 - 1


def require() -> Any:
    """The pyarrow module, which only writing the mirror needs."""
    try:
        import pyarrow
        import pyarrow.parquet  # noqa: F401
    except ImportError as error:
        raise ImportError(
            'Writing the web mirror needs pyarrow: pip install "netgamut[export]"'
        ) from error
    return pyarrow


@dataclass(frozen=True)
class Part:
    """Rows of the links file: a network's link list, or one subgraph of a stack."""

    #: Anything with ``rows()`` and ``columns``, as :class:`~netgamut.Network` has.
    network: Any
    #: The layer every row is in, for a subgraph; ``None`` reads the network's
    #: layer column, if it has one.
    layer: str | None = None
    #: ``{source id: written id}``, for subgraphs that number their nodes on
    #: their own; ``None`` writes the source's ids.
    ids: Mapping[str, str] | None = None

    def rows(self) -> Iterator[tuple[str, str, tuple[str, ...]]]:
        """``(source, target, fields)`` for every row with two fields."""
        for fields in self.network.rows():
            if len(fields) < 2:
                continue
            u, v = fields[0], fields[1]
            if self.ids is not None:
                try:
                    u, v = self.ids[u], self.ids[v]
                except KeyError as missing:
                    raise ValueError(
                        f"node {missing.args[0]} of layer {self.layer!r} has no key."
                    ) from None
            yield u, v, fields


@dataclass
class Nodes:
    """The nodes file's rows: ids, names, and the other node columns, in order."""

    ids: list[str]
    names: dict[str, str] = field(default_factory=dict)
    #: ``{source column: values}``, aligned with :attr:`ids`; ``None`` is missing.
    columns: dict[str, list[str | None]] = field(default_factory=dict)
    #: Source columns left out before the values are typed: internal properties.
    dropped: list[str] = field(default_factory=list)


@dataclass
class Written:
    """What :func:`write` wrote, for the manifest entry."""

    files: dict[str, Path]
    nodes: int
    links: int
    layers: int | None
    dropped: dict[str, list[str]]


def node_rows(table: NodeTable) -> Nodes:
    """A source's node table, decoded, its name column apart and capped.

    A column that only repeats the ids (Netzschleuder's ``index``) or the names
    (london_transport's ``nodeLabel`` beside its ``name``) is not written again.
    An internal property, one whose name starts with ``_`` such as graph-tool's
    ``_pos`` layout, is left out and listed in :attr:`Nodes.dropped`.
    """
    names = {str(node): label for node, label in table.labels().items()}
    from_names = set(table.name_columns())
    nodes = table.nodes
    ids = [str(node) for node in nodes]
    named = [names.get(node) for node in ids]
    columns = {}
    dropped = []
    for column in table.columns:
        if column in from_names:
            continue
        if column.startswith("_"):
            dropped.append(column)
            continue
        decoded = table[column]
        values = [_text(decoded.get(node)) for node in nodes]
        if _repeats(values, ids) or (names and _repeats(values, named)):
            continue
        columns[column] = values
    return Nodes(ids=ids, names=names, columns=columns, dropped=dropped)


def _repeats(values: Sequence[str | None], other: Sequence[str | None]) -> bool:
    """Whether *values* has a value and says nothing *other* does not, node by node."""
    stripped = [None if value is None else value.strip() for value in values]
    return any(stripped) and stripped == list(other)


def stack(layers: Mapping[str, Any], key: str) -> tuple[list[Part], Nodes | None]:
    """The parts and nodes of subgraphs that are layers, matched by node *key*.

    Ids are mapped through *key* as :meth:`MultilayerNetwork.from_networks`
    maps them; ``"index"`` keeps them, since the subgraphs number the same
    nodes alike. The nodes file has ids and names only, and none at all when
    no subgraph names its nodes. A node without a *key* value has no row, as
    it has no name in :meth:`MultilayerNetwork.from_networks`; a link to one
    raises :class:`ValueError` when the parts are read.

    Raises:
        LookupError: A subgraph has no node table to take *key* from.
    """
    parts = []
    ids: dict[str, None] = {}
    names: dict[str, str] = {}
    for layer, network in layers.items():
        identity: dict[str, str] | None = None
        try:
            table = network.nodes()
        except (KeyError, IndexError):
            raise
        except LookupError:
            if key != "index":
                raise
            table = None
        if table is not None:
            if key != "index":
                identity = _identity(table, key, layer)
            for node in table.nodes:
                mapped = _mapped(identity, node)
                if mapped is not None:
                    ids.setdefault(mapped, None)
            for node, label in table.labels().items():
                mapped = _mapped(identity, node)
                if mapped is not None:
                    names.setdefault(mapped, label)
        parts.append(Part(network, layer=str(layer), ids=identity))
    return parts, Nodes(ids=list(ids), names=names) if names else None


def _identity(table: NodeTable, key: str, layer: Any) -> dict[str, str]:
    """:func:`node_identity`, over the nodes with a *key* value: an empty one is missing."""
    present = {node: text for node, value in table[key].items() if (text := _text(value))}
    return node_identity(NodeTable(values={key: present}), key, layer)


def _mapped(identity: Mapping[str, str] | None, node: Any) -> str | None:
    """*node*'s written id; ``None`` for one the identity has no key for."""
    return str(node) if identity is None else identity.get(str(node))


def write(
    parts: Sequence[Part],
    nodes: Nodes | None,
    directory: Path,
    name: str,
    metadata: Mapping[str, str],
) -> Written:
    """Write ``<name>.links.parquet``, and ``<name>.nodes.parquet`` when *nodes* has any."""
    pa = require()
    import pyarrow.parquet as pq

    survey = _Survey(parts)
    node_ids = nodes.ids if nodes is not None else []
    for node in node_ids:
        survey.ids.add(node)
    survey.seen.update(node_ids)
    id_type = survey.ids.arrow(pa)
    dropped: dict[str, list[str]] = {"nodes": [], "links": []}

    # Internal properties, such as graph-tool's ``_pos``, are never written.
    for part in parts:
        for name_ in part.network.columns.names[2:]:
            if str(name_).startswith("_") and name_ not in dropped["links"]:
                dropped["links"].append(str(name_))
    link_columns: dict[str, Any] = {"source": id_type, "target": id_type}
    if survey.weighted:
        link_columns["weight"] = pa.float64()
    time_kind = survey.time.final() if survey.time is not None else None
    if time_kind == "str" and survey.time is not None and survey.time.free:
        dropped["links"].append(survey.time_name)
        time_kind = None
    if time_kind is not None:
        link_columns["time"] = _arrow_type(pa, time_kind)
    layer_kind = survey.layer.final() if survey.layer is not None else None
    if layer_kind == "str" and survey.layer is not None and survey.layer.free:
        dropped["links"].append(survey.layer_name)
        layer_kind = None
    if layer_kind is not None:
        assert survey.layer is not None
        layer_kind = "int32" if layer_kind == "int" and survey.layer.small else "str"
        link_columns["layer"] = _arrow_type(pa, layer_kind)

    files = {"links": directory / f"{name}.links.parquet"}
    schema = pa.schema(list(link_columns.items()), metadata=dict(metadata))
    cast = int if survey.ids.kind != "str" else str
    with pq.ParquetWriter(files["links"], schema, compression="zstd") as writer:
        batch: dict[str, list[Any]] = {column: [] for column in link_columns}
        for part in parts:
            columns = part.network.columns
            for u, v, fields in part.rows():
                batch["source"].append(cast(u))
                batch["target"].append(cast(v))
                if "weight" in batch:
                    batch["weight"].append(
                        None
                        if columns.weight is None
                        else _columns.weight_of(fields, columns.weight)
                    )
                if "time" in batch:
                    batch["time"].append(_value(_field(fields, columns.time), time_kind))
                if "layer" in batch:
                    label = part.layer or _field(fields, columns.layer)
                    batch["layer"].append(_value(label, layer_kind))
                if len(batch["source"]) >= ROW_GROUP:
                    _flush(pa, writer, schema, batch)
        _flush(pa, writer, schema, batch)

    if nodes is not None:
        table, node_dropped = _node_table(pa, nodes, id_type, cast, metadata)
        dropped["nodes"] = node_dropped
        if table is not None:
            files["nodes"] = directory / f"{name}.nodes.parquet"
            pq.write_table(table, files["nodes"], compression="zstd", row_group_size=ROW_GROUP)

    return Written(
        files=files,
        nodes=len(survey.seen),
        links=survey.links,
        layers=len(survey.layers) if "layer" in link_columns else None,
        dropped=dropped if any(dropped.values()) else {},
    )


def _flush(pa: Any, writer: Any, schema: Any, batch: dict[str, list[Any]]) -> None:
    if batch["source"]:
        writer.write_table(pa.Table.from_pydict(batch, schema=schema), row_group_size=ROW_GROUP)
    for values in batch.values():
        values.clear()


def _node_table(
    pa: Any, nodes: Nodes, id_type: Any, cast: Any, metadata: Mapping[str, str]
) -> tuple[Any, list[str]]:
    """The nodes file as a table, or ``None`` with nothing past the ids; and what was dropped."""
    data: dict[str, Any] = {"id": pa.array([cast(node) for node in nodes.ids], type=id_type)}
    if nodes.names:
        data["name"] = pa.array([_cap(nodes.names.get(node)) for node in nodes.ids], pa.string())
    dropped = list(nodes.dropped)
    taken = {"id", "name", *nodes.columns}
    for column, values in nodes.columns.items():
        kind = _Kind()
        for value in values:
            kind.add(value)
        if not kind.present:
            continue  # a column with no value says nothing
        final = "str" if kind.kind == "str" else kind.final()
        if final == "str" and kind.free:
            dropped.append(column)
            continue
        renamed = column
        if column in ("id", "name"):
            # Clear of every other column's name, ``id_`` included.
            renamed += "_"
            while renamed in taken:
                renamed += "_"
            taken.add(renamed)
        data[renamed] = pa.array(
            [_value(value, final) for value in values], type=_arrow_type(pa, final)
        )
    if len(data) == 1:
        return None, dropped
    table = pa.table(data).replace_schema_metadata(dict(metadata))
    return table, dropped


class _Survey:
    """The first pass over the link list: every column's type, and the counts."""

    def __init__(self, parts: Sequence[Part]) -> None:
        self.ids = _Ids()
        self.seen: set[str] = set()
        self.links = 0
        self.weighted = any(part.network.columns.weight is not None for part in parts)
        timed = [part.network.columns for part in parts if part.network.columns.time is not None]
        self.time = _Kind() if timed else None
        self.time_name = _role_name(timed, "time")
        layered = any(part.layer is not None for part in parts)
        by_column = [
            part.network.columns for part in parts if part.network.columns.layer is not None
        ]
        self.layer = _Kind() if layered or by_column else None
        self.layer_name = _role_name(by_column, "layer")
        self.layers: set[str] = set()
        for part in parts:
            columns = part.network.columns
            for u, v, fields in part.rows():
                self.links += 1
                for node in (u, v):
                    if node not in self.seen:
                        self.seen.add(node)
                        self.ids.add(node)
                if self.time is not None:
                    self.time.add(_field(fields, columns.time))
                if self.layer is not None:
                    label = part.layer or _field(fields, columns.layer)
                    self.layer.add(label)
                    if label is not None:
                        self.layers.add(label)


def _role_name(found: Iterable[Any], role: str) -> str:
    """The source's name for a role column, for ``dropped``; else the role."""
    for columns in found:
        index = getattr(columns, role)
        if index is not None and index < len(columns.names):
            return str(columns.names[index])
    return role


class _Ids:
    """Node ids: int32 while every one is an integer that fits, int64, then text."""

    def __init__(self) -> None:
        self.kind = "int32"

    def add(self, text: str) -> None:
        if self.kind == "str":
            return
        if not _columns.is_int_label(text):
            self.kind = "str"
            return
        number = int(text)
        if number > _INT64:
            self.kind = "str"
        elif number > _INT32:
            self.kind = "int64"

    def arrow(self, pa: Any) -> Any:
        return {"int32": pa.int32(), "int64": pa.int64(), "str": pa.string()}[self.kind]


class _Kind:
    """What a column's present values all are, widening as :class:`_columns._Values` does.

    Integers, then floats on the first value that is not one, then text.
    Text that is all dates is a timestamp column; a mix of numbers and dates
    stays text. Free text is a value over :data:`TEXT_LIMIT` characters, or
    one with a line break.
    """

    def __init__(self) -> None:
        self.kind = "int"
        self.numbers = False
        self.dates = True
        self.free = False
        self.present = False
        #: Every present value is an int32 integer, as a layer label must be.
        self.small = True

    def add(self, text: str | None) -> None:
        if text is None:
            return
        self.present = True
        if len(text) > TEXT_LIMIT or not _LINE_BREAKS.isdisjoint(text):
            self.free = True
        try:
            number = int(text)
            if abs(number) > _INT64:
                raise OverflowError
        except (ValueError, OverflowError):
            pass
        else:
            self.numbers = True
            self.small = self.small and _columns.is_int_label(text) and number <= _INT32
            return
        self.small = False
        try:
            float(text)
        except ValueError:
            self.kind = "str"
            self.dates = self.dates and _is_date(text)
        else:
            self.numbers = True
            if self.kind == "int":
                self.kind = "float"

    def final(self) -> str:
        """``int``, ``float``, ``date`` or ``str``."""
        if self.kind == "str" and self.dates and not self.numbers:
            return "date"
        return self.kind


def _is_date(text: str) -> bool:
    """Whether numpy reads *text* as a date; never its words ``now`` and ``today``."""
    import numpy as np

    if not any(character.isdigit() for character in text):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # numpy deprecates parsing time zones
            exact = np.datetime64(text, "ms")
            # A year past int64 milliseconds wraps round rather than failing.
            year = np.datetime64(text).astype("datetime64[Y]")
            return not np.isnat(exact) and exact.astype("datetime64[Y]") == year
    except (ValueError, OverflowError):
        return False


def _arrow_type(pa: Any, kind: str) -> Any:
    return {
        "int": pa.int64(),
        "int32": pa.int32(),
        "float": pa.float64(),
        "date": pa.timestamp("ms", tz="UTC"),
        "str": pa.string(),
    }[kind]


def _value(text: str | None, kind: str | None) -> Any:
    """*text* as a value of a column of *kind*; ``None`` stays missing."""
    if text is None:
        return None
    if kind in ("int", "int32"):
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "date":
        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return int(np.datetime64(text, "ms").astype(np.int64))
    return text


def _field(fields: tuple[str, ...], index: int | None) -> str | None:
    """Field *index* of a row; ``None`` when the row has none, or it is blank."""
    if index is None or len(fields) <= index:
        return None
    text = fields[index]
    return text if text.strip() else None


def _text(value: Any) -> str | None:
    """A decoded node value as the text it was read from; ``None`` when missing."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value)
    return text if text.strip() else None


def _cap(name: str | None) -> str | None:
    if name is None:
        return None
    if not _LINE_BREAKS.isdisjoint(name):  # a label is one line
        name = " ".join(name.splitlines())
    if len(name) <= TEXT_LIMIT:
        return name
    return name[: TEXT_LIMIT - 1] + "…"
