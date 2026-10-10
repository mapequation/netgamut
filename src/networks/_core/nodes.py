"""Per-node metadata, in one shape whatever the source published.

Every source ships node attributes differently -- SNAP as a separate two-column
artifact, Netzschleuder as ``nodes.csv`` inside a bundle, OGB as ``node-label.csv.gz``
plus a ``mapping/`` folder naming the codes, PyG as an anonymous integer tensor. What
they have in common is *named columns of per-node values, some of them categorical*,
so that is what :class:`NodeTable` is, and every source builds one:

    nodes = networks.load("ogbn-arxiv").nodes()
    nodes.columns                     # ('arxiv_category', 'year', 'paper')
    nodes["arxiv_category"]           # {0: 'arxiv cs na', ...}  decoded where named
    nodes.partition("arxiv_category") # {0: 3, ...}   contiguous 0..k-1, for AMI/D
    nodes.categories("arxiv_category")# ('arxiv cs ai', ...)  categories[i] names i

``partition`` and ``categories`` are two halves of one answer: the first is what you
score a community-detection run against, the second is what lets you say *which*
category a module lines up with. A source that does not name its codes -- PyG -- still
supports ``partition``; ``categories`` is then the sorted codes themselves, and the
docstring of the loader says where the real names live.

**Node metadata is not ground truth for community structure.** It is what someone
recorded about the nodes, and a partition that disagrees with it is not thereby wrong
(Peel, Larremore & Clauset, *Science Advances* 3, e1602548, 2017). Scoring a detected
partition against ``partition(col)`` measures agreement with an attribute, which is a
finding about the attribute as much as about the algorithm.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NodeTable"]


def _order(value: Any) -> tuple[int, float, str]:
    """Sort key placing numbers before text and ordering numbers numerically.

    Values arrive as whatever the source wrote -- ``"10"`` from a CSV, ``10`` from a
    tensor -- and a column may mix the two. This orders both the way a reader expects
    and never raises on an unorderable mix, which plain ``sorted`` would.
    """
    if isinstance(value, bool):
        return (1, 0.0, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    text = str(value).strip()
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text)


@dataclass(frozen=True)
class NodeTable:
    """Named per-node columns, with optional names for categorical codes.

    Built by the sources, not by hand. ``values`` maps a column to
    ``{node: value}``; ``names`` optionally maps a column to ``{code: name}`` for
    columns whose stored values are codes the source can name.
    """

    #: ``{column: {node: value}}``, in the order the source published them.
    values: Mapping[str, Mapping[Any, Any]]
    #: ``{column: {code: name}}`` for columns whose values are named codes.
    names: Mapping[str, Mapping[Any, str]] = field(default_factory=dict)
    #: Where the metadata came from, for error messages and provenance.
    origin: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        """The available columns, in the source's own order."""
        return tuple(self.values)

    @property
    def nodes(self) -> tuple[Any, ...]:
        """Every node mentioned by any column, sorted when sortable."""
        seen: set[Any] = set()
        for column in self.values.values():
            seen.update(column)
        try:
            return tuple(sorted(seen))
        except TypeError:  # mixed key types; keep insertion order instead
            return tuple(seen)

    def _column(self, column: str) -> Mapping[Any, Any]:
        try:
            return self.values[column]
        except KeyError:
            available = ", ".join(self.columns) or "none"
            where = f" in {self.origin}" if self.origin else ""
            raise KeyError(f"No node column {column!r}{where}. Available: {available}.") from None

    def __getitem__(self, column: str) -> dict[Any, Any]:
        """``{node: value}``, with codes decoded to names where the source has them.

        This is the reading view. Use :meth:`partition` when the values are going
        into a clustering comparison, which needs contiguous integers.
        """
        raw = self._column(column)
        naming = self.names.get(column)
        if not naming:
            return dict(raw)
        return {node: naming.get(value, value) for node, value in raw.items()}

    def __contains__(self, column: object) -> bool:
        return column in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def categories(self, column: str) -> tuple[Any, ...]:
        """The distinct values of *column*, ordered so index ``i`` names module ``i``.

        Pairs with :meth:`partition`: ``categories(c)[partition(c)[node]]`` is that
        node's value. Named codes come back as their names, unnamed ones as the codes.

        Ordering is natural rather than lexicographic, because most sources write
        their codes as bare text and this order is what numbers the modules -- plain
        ``sorted`` would place department 10 before department 2 and make
        ``partition`` disagree with anyone reading the file.
        """
        decoded = self.__getitem__(column)
        return tuple(sorted(set(decoded.values()), key=_order))

    def partition(self, column: str) -> dict[Any, int]:
        """``{node: module}`` with contiguous ``0..k-1`` codes, ready to be scored.

        The renumbering is by :meth:`categories` order, so the two line up. Nodes with
        no value for *column* are absent rather than pooled into a module of their
        own, which would be a claim the source never made.
        """
        decoded = self.__getitem__(column)
        index = {value: position for position, value in enumerate(self.categories(column))}
        return {node: index[value] for node, value in decoded.items()}

    def select(self, columns: Sequence[str]) -> NodeTable:
        """A table with only *columns*, keeping their names and origin."""
        for column in columns:
            self._column(column)
        return NodeTable(
            values={name: self.values[name] for name in columns},
            names={name: self.names[name] for name in columns if name in self.names},
            origin=self.origin,
        )

    def rows(self) -> list[dict[str, Any]]:
        """One dict per node, decoded -- the shape a dataframe constructor wants."""
        decoded = {column: self.__getitem__(column) for column in self.columns}
        return [
            {"node": node, **{column: decoded[column].get(node) for column in self.columns}}
            for node in self.nodes
        ]

    def __repr__(self) -> str:
        sizes = ", ".join(
            f"{column}[{len(self.categories(column))}]" if column in self.names else f"{column}"
            for column in self.columns
        )
        where = f" from {self.origin}" if self.origin else ""
        return f"NodeTable({len(self.nodes)} nodes{where}: {sizes})"
