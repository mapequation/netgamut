"""KONECT's per-node attribute files, in the three shapes the collection uses.

* ``ent.<network>.<entity>.<attribute>`` -- one value per line, line *i* for
  node *i* of that entity. ``moreno_crime`` has ``person.name`` and
  ``person.sex``; in a bipartite network the entity says which side.
* ``ent.<network>`` opening with a ``% ent dat.<column> ...`` header -- a table
  of quoted, space-separated fields, the first of them the node id. The petster
  networks keep each pet's profile this way, with a row missing for a few nodes.
* ``ent.<network>`` with no such header -- one label per line (``pajek-erdos``
  lists each coauthor's name).

The catalog build sees only a file's name and leading ``%`` lines while it
streams the archive; :func:`netgamut.konect.node_table` reads the files whole.
Both name columns through :func:`columns`, so the catalog advertises exactly the
columns the table will hold.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["LABEL", "columns", "header_columns", "read"]

#: The column for a file of labels, which names no attribute of its own.
LABEL = "label"


def _parts(name: str) -> list[str]:
    """``['moreno_crime_crime', 'person', 'name']`` from the member name."""
    return name.split(".")[1:]


def header_columns(leading: Iterable[str]) -> list[str] | None:
    """The table columns a ``% ent dat.x dat.y`` header declares, else ``None``."""
    for line in leading:
        fields = line.lstrip("%").split()
        if fields and fields[0] == "ent":
            return [field.removeprefix("dat.") for field in fields[1:]]
    return None


def _attribute_column(parts: list[str], entities: set[str]) -> str:
    entity, attribute = (parts[-2], parts[-1]) if len(parts) >= 3 else ("", parts[-1])
    return f"{entity}.{attribute}" if len(entities) > 1 and entity else attribute


def columns(files: Sequence[tuple[str, Sequence[str]]]) -> tuple[str, ...]:
    """Column names for one network's ``ent.*`` files.

    *files* pairs each member name with its leading ``%`` lines. An attribute is
    named alone (``name``) unless the files cover two entities -- both sides of
    a bipartite network -- when it becomes ``entity.attribute``. Files are taken
    in name order, as :func:`read` takes them, whatever order the archive has.
    """
    files = sorted(files, key=lambda item: item[0])
    entities = {parts[-2] for name, _ in files if len(parts := _parts(name)) >= 3}
    found: list[str] = []
    for name, leading in files:
        parts = _parts(name)
        if len(parts) >= 2:
            found.append(_attribute_column(parts, entities))
        else:
            found.extend(header_columns(leading) or [LABEL])
    return tuple(found)


def _leading(lines: list[str]) -> list[str]:
    return [line for line in lines[:20] if line.startswith("%")]


def read(
    files: Sequence[Path], *, sides: Sequence[str], left_nodes: int | None, bipartite: bool
) -> dict[str, dict[int, str]]:
    """``{column: {node: value}}`` from the ``ent.*`` files of one network.

    Node ids are the edge list's: in a bipartite network the right side's are
    shifted past *left_nodes*. *sides* is the meta file's ``entity-names``, left
    first; without it, a file exactly as long as the left side belongs to it.
    """
    files = sorted(files, key=lambda path: path.name)
    entities = {parts[-2] for path in files if len(parts := _parts(path.name)) >= 3}
    left = left_nodes or 0
    values: dict[str, dict[int, str]] = {}
    for path in files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = [line for line in lines if not line.startswith("%")]
        parts = _parts(path.name)

        if len(parts) == 1 and (header := header_columns(_leading(lines))) is not None:
            for record in csv.reader(body, delimiter=" ", quotechar='"'):
                if not record or not record[0].isdigit():
                    continue
                node = int(record[0])
                for column, value in zip(header, record[1:], strict=False):
                    if value.strip():
                        values.setdefault(column, {})[node] = value.strip()
            continue

        offset = 0
        if bipartite and len(parts) >= 3:
            entity = parts[-2]
            if entity in sides:
                offset = left if list(sides).index(entity) == 1 else 0
            elif len(body) != left:
                offset = left
        column = _attribute_column(parts, entities) if len(parts) >= 2 else LABEL
        values[column] = {
            index + offset: value.strip()
            for index, value in enumerate(body, start=1)
            if value.strip()
        }
    return values
