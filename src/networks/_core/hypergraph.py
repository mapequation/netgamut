"""Interactions between node *sets*, not node pairs.

An edge list says two things met. A hypergraph says *how many* met at once, and
that is not recoverable afterwards: three authors on one paper and three authors
on three pairwise papers project onto the same triangle. Sources that publish
higher-order data -- :mod:`networks.ahorn` -- hand back a :class:`Hypergraph`
rather than a :class:`~networks._core.network.Network` so the interaction stays
whole until you choose to flatten it.

The handle is the same bargain as ``Network``: a thin wrapper around a cached
file, reading nothing until asked, so an 8-million-interaction manifold
collection and a 34-node karate club use one API.

Two projections turn one into a graph, and they are not interchangeable:

* **clique** -- every pair inside an interaction becomes an edge. The
  conventional one-mode projection, and the lossy one: interaction sizes are
  gone and an interaction of *k* nodes contributes *k(k-1)/2* edges.
* **star** (bipartite) -- each interaction becomes a node, joined to its
  members. Lossless, and what :meth:`Hypergraph.to_infomap` uses, because
  Infomap reads bipartite networks natively and its modules then span both
  nodes and interactions.

The file format is AHORN's line-based exchange format. The first line is a JSON
object of dataset metadata; every later line is either a comma-separated list of
node identifiers followed by an optional JSON attribute object, a lone
identifier declaring a node, or -- in a multi-network file -- a bare JSON object
marking the start of the next constituent network.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .cache import write_atomic
from .models import DataFile, DatasetLike
from .network import Network
from .nodes import NodeTable

__all__ = ["Hypergraph", "Projection"]

#: The projections :meth:`Hypergraph.to_networkx` and :meth:`Hypergraph.to_network`
#: understand.
PROJECTIONS = ("clique", "star")


def _reserved(metadata: dict[str, Any], key: str) -> Any:
    """Read a reserved metadata key, however the writer spelled it.

    The published specification is version 0.3 and names these keys
    ``format-version`` and ``num-networks``. Every file actually shipped is
    version 0.2, which wrote ``_format-version`` and ``_num_networks`` --
    underscore-prefixed, and inconsistent about the separator. Rather than trust
    either spelling, try all four.
    """
    stem = key.replace("-", "_")
    for candidate in (key, stem, f"_{key}", f"_{stem}"):
        if candidate in metadata:
            return metadata[candidate]
    return None


def _json(text: str, *, number: int, where: str) -> Any:
    """Parse an attribute object, naming the line when it will not parse.

    Three versions of this format are in circulation and a fourth will follow, so
    a file this reader cannot understand should say where it gave up rather than
    surfacing a bare decoder error from somewhere in a stream.
    """
    try:
        return json.loads(text)
    except ValueError as error:
        raise ValueError(
            f"{where}: line {number} is not valid AHORN format -- expected an "
            f"optional JSON object after the node identifiers, got {text[:60]!r}."
        ) from error


def _split(
    line: str, *, number: int = 0, where: str = ""
) -> tuple[tuple[str, ...] | None, dict[str, Any]]:
    """Parse one data line into ``(identifiers, attributes)``.

    ``identifiers`` is ``None`` for a bare JSON object -- the dataset metadata
    line or a multi-network marker. Identifiers carry no whitespace or braces by
    specification, so the split is at the first whitespace run and everything
    after it is the attribute object.
    """
    stripped = line.strip()
    if stripped.startswith("{"):
        parsed = _json(stripped, number=number, where=where)
        return None, parsed if isinstance(parsed, dict) else {}

    parts = stripped.split(None, 1)
    identifiers = tuple(part.strip() for part in parts[0].split(",") if part.strip())
    attributes: dict[str, Any] = {}
    tail = parts[1].strip() if len(parts) > 1 else ""
    if tail:
        parsed = _json(tail, number=number, where=where)
        if isinstance(parsed, dict):
            attributes = parsed
    return identifiers, attributes


@dataclass(frozen=True, slots=True)
class Projection:
    """What :meth:`Hypergraph.to_network` hands its ``Network`` as a dataset.

    A projection is not what the source published, so it does not borrow the
    source's record: the interaction count is not an edge count, and reporting
    it as one would be wrong by a factor of the mean interaction size. Sizes are
    ``None`` here for the same reason every other source leaves an unpublished
    size ``None`` -- nobody measured it. The original record is one attribute
    away, on the hypergraph.
    """

    name: str
    #: Which projection produced it, for the repr.
    projection: str = ""
    #: Both projections are symmetric: sharing an interaction has no direction.
    directed: bool | None = False
    domains: tuple[str, ...] = ()
    nodes: int | None = None
    edges: int | None = None

    def __str__(self) -> str:
        return f"{self.name} ({self.projection} projection)"


@dataclass(frozen=True)
class Hypergraph:
    """A downloaded higher-order dataset on local disk."""

    #: Full metadata for the dataset this file belongs to.
    dataset: DatasetLike
    #: The specific artifact that was downloaded.
    file: DataFile
    #: The downloaded artifact exactly as the source serves it, gzip included.
    archive_path: Path
    #: Which ``networks.<source>`` produced this handle, for :meth:`nodes`.
    source: str = ""
    #: Which published revision this file is, when the source pins them.
    revision: int | None = None

    # -- paths and reading -------------------------------------------------
    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def path(self) -> Path:
        """The file on disk, still compressed if that is how it arrived."""
        return self.archive_path

    @property
    def size_bytes(self) -> int:
        return self.archive_path.stat().st_size if self.archive_path.is_file() else 0

    def open(self) -> IO[str]:
        """Open the dataset as a text stream, decompressing on the fly."""
        if self.archive_path.name.lower().endswith(".gz"):
            return io.TextIOWrapper(
                gzip.open(self.archive_path, "rb"), encoding="utf-8", errors="replace"
            )
        return self.archive_path.open(encoding="utf-8", errors="replace")

    @property
    def metadata(self) -> dict[str, Any]:
        """The dataset-level JSON object on the first line, exactly as written.

        Keys are the file's own -- ``_format-version`` on every currently
        published file. Use :attr:`format_version`, :attr:`file_revision` and
        :attr:`num_networks` to read the reserved ones without caring which
        spelling this file used.
        """
        with self.open() as stream:
            for number, line in enumerate(stream, start=1):
                if line.strip():
                    identifiers, attributes = _split(line, number=number, where=self.name)
                    return attributes if identifiers is None else {}
        return {}

    @property
    def format_version(self) -> str:
        """Which revision of the exchange format wrote this file."""
        value = _reserved(self.metadata, "format-version")
        return str(value) if value is not None else ""

    @property
    def file_revision(self) -> int | None:
        """The revision number the file itself records, or ``None``.

        Format 0.1 wrote no revision at all, so older published revisions have
        none to report. :attr:`revision` -- the one that was asked for -- is
        always known.
        """
        value = _reserved(self.metadata, "revision")
        return int(value) if isinstance(value, (int, str)) and str(value).isdigit() else None

    @property
    def num_networks(self) -> int:
        """How many constituent networks the file holds; 1 unless it says otherwise."""
        value = _reserved(self.metadata, "num-networks")
        if isinstance(value, int):
            return value
        return int(value) if isinstance(value, str) and value.isdigit() else 1

    # -- streaming ---------------------------------------------------------
    def _records(self) -> Iterator[tuple[str, tuple[str, ...], dict[str, Any]]]:
        """Yield ``(kind, identifiers, attributes)`` for every data line.

        *kind* is ``"marker"`` for a constituent network's header,
        ``"node"`` for a lone identifier and ``"interaction"`` otherwise. The
        dataset metadata line is consumed here rather than yielded.

        A single-member interaction is indistinguishable from a node
        declaration in this format, and is read as a node.
        """
        with self.open() as stream:
            first = True
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                identifiers, attributes = _split(line, number=number, where=self.name)
                if identifiers is None:
                    if first:
                        first = False  # the dataset metadata object
                        continue
                    yield "marker", (), attributes
                    continue
                first = False
                if not identifiers:
                    continue
                yield ("node" if len(identifiers) == 1 else "interaction"), identifiers, attributes

    def interactions(
        self, *, cast: Callable[[str], Any] | None = None, network: str | None = None
    ) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        """Stream ``(nodes, attributes)`` for every interaction.

        Members come back as strings unless you pass *cast*; ``cast=int`` is
        usual. Streaming means memory stays flat however large the file is::

            for members, attributes in hypergraph.interactions(cast=int):
                ...   # ((2, 12, 20), {'category': 'lie-algebras'})

        In a multi-network file, pass *network* to take only the constituent
        network with that ``id``; without it every interaction is streamed and
        the marker attributes are the way to tell them apart.
        """
        current: str | None = None
        for kind, identifiers, attributes in self._records():
            if kind == "marker":
                current = str(attributes.get("id", "")) or None
                continue
            if kind != "interaction":
                continue
            if network is not None and current != network:
                continue
            yield (
                identifiers if cast is None else tuple(cast(item) for item in identifiers),
                attributes,
            )

    def node_records(
        self, *, cast: Callable[[str], Any] | None = None
    ) -> Iterator[tuple[Any, dict[str, Any]]]:
        """Stream ``(node, attributes)`` for every explicitly declared node.

        Most datasets declare no nodes at all and let the interactions imply
        them, so an empty stream is normal rather than an error.
        """
        for kind, identifiers, attributes in self._records():
            if kind == "node":
                yield (identifiers[0] if cast is None else cast(identifiers[0])), attributes

    def networks(self) -> tuple[dict[str, Any], ...]:
        """The marker objects of a multi-network file, in file order.

        Each is that constituent network's own metadata -- ``MANTRA-2-manifolds``
        gives 43,214 of them, one per triangulated manifold, carrying Betti
        numbers, orientability and genus. Empty for an ordinary file.
        """
        return tuple(attributes for kind, _, attributes in self._records() if kind == "marker")

    def interaction_labels(self, attribute: str, *, network: str | None = None) -> list[Any]:
        """The value of *attribute* on each interaction, in file order.

        This is the reference labelling for a hyperedge-classification dataset:
        ``interaction_labels("category")`` on ``algebra-questions`` gives the
        question category each co-answering group belongs to. Interactions
        missing the attribute contribute ``None`` rather than being dropped, so
        the list stays aligned with :meth:`interactions`.
        """
        return [attributes.get(attribute) for _, attributes in self.interactions(network=network)]

    def nodes(self) -> NodeTable:
        """Read per-node metadata into the shape every source uses.

        Delegates to the source, as :meth:`networks._core.network.Network.nodes`
        does. Node metadata is not ground truth for community structure; see
        :mod:`networks._core.nodes`.

        Raises:
            LookupError: The source publishes no per-node metadata, or this
                dataset declares no nodes explicitly.
        """
        import importlib

        if not self.source:
            raise LookupError(
                f"{self.dataset.name!r} was built without a source tag, so its node "
                "metadata cannot be located."
            )
        module = importlib.import_module(f"networks.{self.source}")
        reader = getattr(module, "node_table", None)
        if reader is None:
            raise LookupError(f"networks.{self.source} publishes no per-node metadata.")
        table: NodeTable = reader(self)
        return table

    # -- projections -------------------------------------------------------
    @staticmethod
    def _check(projection: str) -> str:
        if projection not in PROJECTIONS:
            raise ValueError(
                f"Unknown projection {projection!r}. Available: {', '.join(PROJECTIONS)}."
            )
        return projection

    def _clique_pairs(self, *, cast: Callable[[str], Any] | None) -> dict[tuple[Any, Any], float]:
        """``{(u, v): multiplicity}`` over every pair sharing an interaction.

        Held in memory, unlike everything else here, because a pair recurs
        across interactions and the weight is the count. An interaction of *k*
        members contributes *k(k-1)/2* pairs, so check
        ``dataset.max_interaction_size`` before projecting a wide one.
        """
        pairs: dict[tuple[Any, Any], float] = {}
        for members, _ in self.interactions(cast=cast):
            unique = sorted(set(members), key=str)
            for index, left in enumerate(unique):
                for right in unique[index + 1 :]:
                    key = (left, right)
                    pairs[key] = pairs.get(key, 0.0) + 1.0
        return pairs

    def _star_pairs(self, *, cast: Callable[[str], Any] | None) -> Iterator[tuple[str, Any]]:
        """Stream ``(interaction label, member)`` for the bipartite projection."""
        for index, (members, _) in enumerate(self.interactions(cast=cast)):
            label = f"e{index}"
            for member in members:
                yield label, member

    def to_networkx(
        self,
        *,
        projection: str = "clique",
        cast: Callable[[str], Any] | None = int,
        weighted: bool = True,
    ) -> Any:
        """Build a networkx graph by projecting the interactions.

        ``projection="clique"`` connects every pair that shares an interaction,
        weighting each edge by how many interactions they share. ``"star"``
        keeps the interactions as nodes -- labelled ``e0``, ``e1``, ... and
        marked ``bipartite=1`` -- joined to their members, which loses nothing.

        Requires the ``networkx`` extra::

            pip install "mapequation-networks[networkx]"

        Raises:
            ValueError: Unknown *projection*.
        """
        self._check(projection)
        try:
            import networkx
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_networkx() needs networkx. Install it with "
                '`pip install "mapequation-networks[networkx]"`.'
            ) from error

        graph = networkx.Graph()
        graph.name = self.dataset.name

        def build(caster: Callable[[str], Any] | None) -> None:
            graph.clear()
            if projection == "clique":
                for (left, right), weight in self._clique_pairs(cast=caster).items():
                    graph.add_edge(left, right, **({"weight": weight} if weighted else {}))
                return
            for label, member in self._star_pairs(cast=caster):
                graph.add_node(label, bipartite=1)
                graph.add_node(member, bipartite=0)
                graph.add_edge(label, member)

        try:
            build(cast)
        except ValueError:
            # Node identifiers are not integers after all; keep them as strings.
            build(None)
        return graph

    def to_infomap(self, **kwargs: Any) -> Any:
        """Build an ``infomap.Infomap`` instance holding the bipartite projection.

        Infomap reads bipartite networks natively, so the interactions stay in
        the model rather than being flattened away: physical nodes take ids
        ``0..n-1``, each interaction takes an id at or above
        ``bipartite_start_id``, and the modules Infomap reports span both. That
        is the whole reason for keeping a hypergraph handle at all -- the clique
        projection would have thrown the interaction sizes away before Infomap
        ever saw them.

        The file is read twice: once to number the nodes in first-seen order,
        once to add the links. Extra keyword arguments go to the ``Infomap``
        constructor. Requires the ``infomap`` extra::

            pip install "mapequation-networks[infomap]"
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_infomap() needs infomap. Install it with "
                '`pip install "mapequation-networks[infomap]"`.'
            ) from error

        ids: dict[str, int] = {}
        for members, _ in self.interactions():
            for member in members:
                if member not in ids:
                    ids[member] = len(ids)

        instance = Infomap(**kwargs)
        instance.bipartite_start_id = len(ids)
        instance.set_names({identifier: label for label, identifier in ids.items()})

        for index, (members, _) in enumerate(self.interactions()):
            feature = len(ids) + index
            for member in members:
                instance.add_link(feature, ids[member])
        return instance

    def to_network(
        self,
        *,
        projection: str = "clique",
        cast: Callable[[str], Any] | None = None,
        force: bool = False,
    ) -> Network:
        """Flatten to a dyadic edge list and return a ``Network`` over it.

        The projected edge list is written next to the cached archive and reused
        on later calls, so this costs one pass the first time and nothing after.
        Use it to feed a hypergraph to anything that wants an ordinary graph --
        :mod:`networks.generate`, an external tool reading a two-column file.

        What comes back describes the projection, not the dataset: no source tag,
        so per-node metadata stays here rather than on the result, and no sizes,
        because an interaction count is not an edge count.

        The star projection labels its interaction nodes ``e0``, ``e1``, ... , so
        the file mixes those with the source's own identifiers and cannot be read
        with ``cast=int``. Prefer :meth:`to_infomap`, which builds the bipartite
        network directly and keeps the two kinds of node properly separated.

        Raises:
            ValueError: Unknown *projection*.
        """
        self._check(projection)
        stem = self.archive_path.name.split(".")[0]
        target = self.archive_path.parent / f"{stem}-{projection}.edges"

        if force or not target.is_file():
            lines: list[str] = []
            if projection == "clique":
                for (left, right), weight in self._clique_pairs(cast=cast).items():
                    lines.append(f"{left} {right} {weight:g}")
            else:
                lines.extend(f"{label} {member}" for label, member in self._star_pairs(cast=cast))
            payload = "".join(f"{line}\n" for line in lines)
            write_atomic(target, payload.encode())

        return Network(
            dataset=Projection(
                name=self.dataset.name,
                projection=projection,
                domains=self.dataset.domains,
            ),
            file=DataFile(
                name=target.name,
                url=self.file.url,
                description=f"{projection} projection of {self.file.name}",
            ),
            archive_path=target,
        )

    # -- display -----------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Hypergraph {self.dataset.name!r} file={self.file.name!r} at {self.archive_path}>"

    def __str__(self) -> str:
        shape = []
        nodes = getattr(self.dataset, "nodes", None)
        interactions = getattr(self.dataset, "interactions", None)
        if nodes is not None:
            shape.append(f"{nodes:,} nodes")
        if interactions is not None:
            shape.append(f"{interactions:,} interactions")
        revision = f" revision {self.revision}" if self.revision is not None else ""
        return f"{self.dataset.name}{revision}: {', '.join(shape) or 'unmeasured'}"
