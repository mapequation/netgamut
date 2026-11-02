"""Reading PyG's dataset registry out of its own source.

PyG has no machine-readable dataset index, but it does not need one: its documentation
cheatsheet is generated from statistics its dataset classes carry in their docstrings,
as reStructuredText ``list-table`` blocks under ``**STATS:**``. Parsing the source is
therefore reading PyG's own published numbers, not scraping a rendered page -- and it
needs neither ``torch`` nor ``torch_geometric`` installed, which keeps catalog
regeneration a light job.

Two shapes occur. A class covering several named datasets has a ``Name`` column, one
row per dataset (``Planetoid`` → Cora, CiteSeer, PubMed). A class that *is* one dataset
has no ``Name`` column and exactly one row (``KarateClub``).

A ``#graphs`` column above 1 marks a *collection* -- thousands of small molecule or
mesh graphs, with ``~`` averages in the node and edge columns. Those are dropped:
a :class:`~netgamut._core.network.Network` is one graph, so a collection would be an
unloadable row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["REPOSITORY", "Entry", "module_of", "parse_names", "parse_stats"]

REPOSITORY = "pyg-team/pytorch_geometric"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/master"
DATASETS_PATH = "torch_geometric/datasets"

# A module may export several classes on one line:
#   from .citation_full import CitationFull, CoraFull
_IMPORT = re.compile(r"^from \.(\S+) import (.+)$", re.M)
_ROW = re.compile(r"^\s*\* -", re.M)
# Name cells are often written as RST roles: :obj:`Cora`, ``Cora``.
_ROLE = re.compile(r":[a-z]+:`([^`]+)`|``([^`]+)``|`([^`]+)`")


@dataclass(frozen=True, slots=True)
class Entry:
    """One named dataset inside a PyG class."""

    cls: str
    name: str
    nodes: int | None = None
    edges: int | None = None
    features: int | None = None
    classes: int | None = None


def module_of(init_source: str) -> dict[str, str]:
    """``{class name: module stem}`` from ``datasets/__init__.py``'s imports."""
    mapping: dict[str, str] = {}
    for module, imported in _IMPORT.findall(init_source):
        for cls in imported.split("#")[0].split(","):
            name = cls.strip().strip("()")
            if name.isidentifier():
                mapping[name] = module
    return mapping


def homo_classes(init_source: str) -> list[str]:
    """The ``homo_datasets`` list, which is the set of homogeneous-graph classes."""
    start = init_source.index("homo_datasets = [")
    end = init_source.index("]", start)
    return re.findall(r"'([A-Za-z0-9_]+)'", init_source[start:end])


def _number(cell: str) -> int | None:
    """Parse one table cell. ``~17.9`` is a collection average, not a count."""
    text = cell.strip().replace(",", "")
    if not text or text.startswith("~"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _clean(cell: str) -> str:
    """Strip RST markup from a Name cell: ``:obj:`Cora``` and ``` ``Cora`` ``` -> Cora."""
    match = _ROLE.search(cell)
    if match:
        return next(group for group in match.groups() if group).strip()
    return cell.strip().strip("`*")


def _tables(docstring: str) -> list[list[list[str]]]:
    """Every ``list-table`` under ``**STATS:**``, as rows of cells."""
    if "**STATS:**" not in docstring:
        return []
    body = docstring.split("**STATS:**", 1)[1]
    tables = []
    for block in body.split(".. list-table::")[1:]:
        rows: list[list[str]] = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("* -"):
                rows.append([stripped[3:].strip()])
            elif stripped.startswith("- ") and rows:
                rows[-1].append(stripped[2:].strip())
            elif stripped.startswith(":") or not stripped:
                continue
            elif rows and not stripped.startswith(("*", "-")):
                break
        if len(rows) >= 2:
            tables.append(rows)
    return tables


def parse_stats(cls: str, source: str) -> tuple[list[Entry], bool]:
    """Read one dataset module. Returns ``(entries, is_collection)``.

    ``is_collection`` is true when the class holds many graphs per dataset, which is
    what disqualifies it: those cannot be one :class:`Network`.
    """
    match = re.search(rf"^class {cls}\b.*?:\n\s+r?\"\"\"(.*?)\"\"\"", source, re.S | re.M)
    if not match:
        return [], False
    tables = _tables(match.group(1))
    if not tables:
        return [], False

    entries: list[Entry] = []
    collection = False
    for rows in tables:
        header = [cell.strip().lower() for cell in rows[0]]
        if "#graphs" in header:
            position = header.index("#graphs")
            counts = [_number(row[position]) for row in rows[1:] if len(row) > position]
            if any(count is not None and count > 1 for count in counts):
                collection = True
                continue

        def column(name: str, row: list[str], header: list[str] = header) -> int | None:
            return (
                _number(row[header.index(name)])
                if name in header and len(row) > header.index(name)
                else None
            )

        named = header and header[0] == "name"
        for row in rows[1:]:
            if not row:
                continue
            entries.append(
                Entry(
                    cls=cls,
                    name=_clean(row[0]) if named else cls,
                    nodes=column("#nodes", row),
                    edges=column("#edges", row),
                    features=column("#features", row),
                    classes=column("#classes", row),
                )
            )
    return entries, collection


# Older docstrings write "name (str):", newer ones just "name:".
_NAME_ARG = re.compile(
    r"^\s*name(?: \([^)]*\))?:(.*?)(?=\n\s*\w+(?: \([^)]*\))?:|\n\s*\*\*|\Z)",
    re.S | re.M,
)
_QUOTED = re.compile(r":obj:`\"([^\"]+)\"`|:obj:`'([^']+)'`")


def parse_names(cls: str, source: str) -> list[str]:
    """The dataset names a class accepts, from its ``name (str):`` argument docs.

    The fallback for the 41 classes PyG documents without a statistics table. PyG
    describes the allowed values consistently as ``:obj:`"chameleon"``` literals in
    the Args section, so this recovers datasets the cheatsheet has not reached yet --
    at the price of no node or edge counts, which is why it is only used for classes
    the domain map vouches for.

    Returns ``[]`` for a class taking no ``name``, which means the class is itself one
    dataset.
    """
    match = re.search(rf"^class {cls}\b.*?:\n\s+r?\"\"\"(.*?)\"\"\"", source, re.S | re.M)
    if not match:
        return []
    arg = _NAME_ARG.search(match.group(1))
    if not arg:
        return []
    seen: list[str] = []
    for first, second in _QUOTED.findall(arg.group(1)):
        value = first or second
        if value and value not in seen:
            seen.append(value)
    return seen
