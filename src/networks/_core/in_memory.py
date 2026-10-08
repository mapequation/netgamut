"""A first-order network held in memory.

:class:`InMemoryNetwork` is what every first-order generator in
:mod:`networks.generate` returns, and what :meth:`Network.snapshots` yields for
each time bin of a temporal network. It has the methods of a downloaded
:class:`~networks.Network` -- ``to_networkx`` / ``to_infomap`` / ``to_pajek`` /
``to_link_list`` -- plus a :attr:`partition`, the planted ground truth of a
generated network, and :attr:`info`, which says where the network came from.
The two are separate classes because a ``Network`` is a lazy handle on a cached
download -- streamed, never held in memory -- while these exist only in memory
until they are written.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import formats

__all__ = ["InMemoryNetwork"]


@dataclass(frozen=True)
class InMemoryNetwork:
    """A first-order network in memory: generated, or a time slice of a download.

    Instances are frozen; the containers are the generator's result and must
    not be mutated. Convert with :meth:`to_networkx` for anything that edits.
    """

    #: Used as the file name by :meth:`to_pajek` and :meth:`to_link_list`.
    name: str
    #: Every node, isolated ones included, in the generator's order.
    nodes: tuple[Hashable, ...] = field(hash=False)
    #: ``(source, target) -> weight``; weight 1.0 throughout when unweighted.
    links: dict[tuple[Hashable, Hashable], float] = field(hash=False)
    directed: bool = False
    #: Whether :attr:`links` carries weights worth writing.
    weighted: bool = False
    #: The planted ground truth of a generated network, ``{node: module}``; a
    #: module is an ``int``, a path tuple for a hierarchical partition, or a
    #: tuple of memberships for an overlapping cover. ``None`` for real data.
    partition: dict[Hashable, Any] | None = field(default=None, hash=False)
    #: For a generated network, ``"generator"`` and ``"parameters"`` and the
    #: generator's diagnostics; for a snapshot, its bin and time span.
    info: dict[str, Any] = field(default_factory=dict, hash=False)

    @classmethod
    def from_networkx(
        cls,
        graph: Any,
        *,
        name: str,
        partition: Mapping[Hashable, Any] | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> InMemoryNetwork:
        """Capture a networkx graph; ``"weight"`` is the only edge data kept."""
        weighted = any("weight" in data for _, _, data in graph.edges(data=True))
        links = {(u, v): float(data.get("weight", 1.0)) for u, v, data in graph.edges(data=True)}
        return cls(
            name=name,
            nodes=tuple(graph.nodes()),
            links=links,
            directed=graph.is_directed(),
            weighted=weighted,
            partition=None if partition is None else dict(partition),
            info=dict(info or {}),
        )

    # -- inspection --------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_links(self) -> int:
        return len(self.links)

    def edges(self) -> Iterator[tuple[Hashable, Hashable]]:
        """Yield ``(source, target)`` pairs, as :meth:`Network.edges` does."""
        yield from self.links

    def _links(self) -> Iterator[formats.Link]:
        for (u, v), w in self.links.items():
            yield u, v, (w if self.weighted else None)

    # -- conversions -------------------------------------------------------
    def to_networkx(self, *, partition_attribute: str | None = "module") -> Any:
        """Build a networkx graph, with the planted partition as a node attribute.

        Each node's module lands under *partition_attribute* (``"module"`` by
        default; ``None`` leaves it out), so
        ``networkx.get_node_attributes(G, "module")`` gives back
        :attr:`partition`. Weights are set only for a weighted network.
        """
        try:
            import networkx
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_networkx() needs networkx. Install it with "
                '`pip install "mapequation-networks[networkx]"`.'
            ) from error

        graph = networkx.DiGraph() if self.directed else networkx.Graph()
        graph.name = self.name
        graph.add_nodes_from(self.nodes)
        if self.weighted:
            graph.add_weighted_edges_from((u, v, w) for (u, v), w in self.links.items())
        else:
            graph.add_edges_from(self.links)
        if self.partition is not None and partition_attribute is not None:
            networkx.set_node_attributes(graph, self.partition, partition_attribute)
        return graph

    def to_infomap(self, **kwargs: Any) -> Any:
        """Build an :class:`infomap.Infomap` instance with this network loaded.

        Direction follows the network; extra keyword arguments go to the
        ``Infomap`` constructor. Node ids are the labels when those are
        non-negative integers, as every generator here produces; otherwise
        ``1..N`` with the labels as names.
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_infomap() needs infomap. Install it with "
                '`pip install "mapequation-networks[infomap]"`.'
            ) from error

        kwargs.setdefault("directed", self.directed)
        instance = Infomap(**kwargs)
        ids = formats.node_ids(self.nodes)
        for label, index in ids.items():
            # Labels kept as ids need no name; renumbered ones keep theirs.
            instance.add_node(index, None if label == index else str(label))
        for (u, v), w in self.links.items():
            instance.add_link(ids[u], ids[v], w if self.weighted else 1.0)
        return instance

    # -- output ------------------------------------------------------------
    def to_pajek(self, directory: str | Path, *, name: str | None = None) -> Path:
        """Write ``<directory>/<name>.net`` in Pajek format and return its path.

        *name* defaults to :attr:`name`. Generator names include the
        parameters that matter, but not all of them: pass *name* when writing a
        sweep that varies something else into one directory.
        """
        target = formats.output_path(directory, name or self.name, ".net")
        return formats.write_pajek(
            target,
            nodes=self.nodes,
            links=self._links,
            directed=self.directed,
            comments=formats.describe(self.name, self.info),
        )

    def to_link_list(self, directory: str | Path, *, name: str | None = None) -> Path:
        """Write ``<directory>/<name>.txt`` as a link list and return its path.

        A link list holds links only, so isolated nodes are not in the file;
        :meth:`to_pajek` keeps them.

        Raises:
            ValueError: A node label is not a non-negative integer; use
                :meth:`to_pajek`, which keeps labels as names.
        """
        target = formats.output_path(directory, name or self.name, ".txt")
        return formats.write_link_list(
            target,
            links=self._links(),
            directed=self.directed,
            comments=formats.describe(self.name, self.info),
        )

    def __str__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        weights = "weighted " if self.weighted else ""
        planted = ", planted partition" if self.partition is not None else ""
        return (
            f"InMemoryNetwork {self.name!r}: {self.num_nodes:,} nodes, "
            f"{self.num_links:,} {weights}{kind} links{planted}"
        )
