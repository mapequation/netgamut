"""Metadata records and the download handle for the curated paths source.

A path dataset is not an edge list: the unit of observation is a whole
trajectory. The :class:`Dataset` here therefore carries a *parser* name --
the recipe that turns the cached download into a stream of paths -- and
:class:`PathData` is the :class:`~netgamut.Network` handle
extended with that stream and the conversion to a state network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from .._core.domains import domains_for_paths
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.network import Network
from .._core.state_network import StateNetwork

__all__ = ["Dataset", "PathData"]

_FIRST_DB1B_YEAR = 1993


@dataclass(frozen=True, slots=True)
class Dataset:
    """One curated path dataset."""

    name: str
    description: str = ""
    #: Which recipe turns the download into paths; a key in ``_parsers.PARSERS``.
    parser: str = ""
    files: tuple[DataFile, ...] = ()
    #: Small side tables fetched on demand rather than with the download, as
    #: ``(role, file)`` pairs. Roles are what the handle's accessors look up:
    #: ``"coordinates"`` for :meth:`PathData.coordinates`, ``"names"`` for
    #: :meth:`PathData.names`.
    metadata: tuple[tuple[str, DataFile], ...] = ()
    #: Another entry whose cache directory holds this one's downloads. Two views
    #: of one archive -- ``db1b-city`` reads the same coupon file as
    #: ``db1b-coupon`` -- then share a single copy of it.
    shares_download_with: str = ""
    #: The dataset's page, for humans.
    url: str = ""
    #: For quarterly collections: a format string with ``{year}`` and ``{quarter}``.
    url_template: str = ""
    default_year: int | None = None
    default_quarter: int | None = None
    #: Physical nodes, when the source states them.
    nodes: int | None = None
    #: Number of paths, when the source states them.
    num_paths: int | None = None
    citations: tuple[str, ...] = ()
    license: str = ""

    # -- shared interface -------------------------------------------------
    @property
    def domains(self) -> tuple[str, ...]:
        """Normalised domains, shared with every other source."""
        return domains_for_paths(self.name)

    @property
    def directed(self) -> bool | None:
        """Paths are traversed in order, so the derived networks are directed."""
        return True

    @property
    def edges(self) -> int | None:
        """Unknown up front: the link count depends on the chosen Markov order."""
        return None

    @property
    def node_attributes(self) -> tuple[str, ...]:
        """None: the paths source publishes no per-node metadata."""
        return ()

    @property
    def node_metadata(self) -> bool:
        return False

    @property
    def primary_file(self) -> DataFile | None:
        return self.files[0] if self.files else None

    @property
    def quarterly(self) -> bool:
        """Whether this dataset is one download per year and quarter."""
        return bool(self.url_template)

    @property
    def cache_name(self) -> str:
        """The dataset directory in the cache; another entry's when downloads are shared."""
        return self.shares_download_with or self.name

    def metadata_file(self, role: str) -> DataFile:
        """The side table filling *role*, e.g. ``"coordinates"``.

        Raises:
            LookupError: This dataset ships no such table.
        """
        for current, item in self.metadata:
            if current == role:
                return item
        available = ", ".join(current for current, _ in self.metadata) or "none"
        raise LookupError(f"{self.name!r} ships no {role!r} table. Available: {available}.")

    def file_for(self, *, year: int | None = None, quarter: int | None = None) -> DataFile:
        """The artifact to download, resolving *year*/*quarter* for quarterly data.

        Raises:
            ValueError: year/quarter passed for a single-collection dataset,
                or out of range for a quarterly one.
        """
        if year is None and quarter is None:
            if self.files:
                return self.files[0]
            raise LookupError(f"{self.name!r} lists no downloadable files.")
        if not self.quarterly:
            raise ValueError(
                f"{self.name!r} is a single collection; year and quarter do not apply."
            )
        year = year if year is not None else self.default_year
        quarter = quarter if quarter is not None else self.default_quarter
        if year is None or year < _FIRST_DB1B_YEAR:
            raise ValueError(f"year must be {_FIRST_DB1B_YEAR} or later, got {year!r}.")
        if quarter is None or not 1 <= quarter <= 4:
            raise ValueError(f"quarter must be between 1 and 4, got {quarter!r}.")
        url = self.url_template.format(year=year, quarter=quarter)
        return DataFile(url.rsplit("/", 1)[-1], url, f"{self.name}, {year} Q{quarter}")

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parser": self.parser,
            "files": [item.to_dict() for item in self.files],
            "metadata": {role: item.to_dict() for role, item in self.metadata},
            "shares_download_with": self.shares_download_with,
            "url": self.url,
            "url_template": self.url_template,
            "default_year": self.default_year,
            "default_quarter": self.default_quarter,
            "nodes": self.nodes,
            "num_paths": self.num_paths,
            "citations": list(self.citations),
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parser=data.get("parser", ""),
            files=tuple(DataFile.from_dict(item) for item in data.get("files", [])),
            metadata=tuple(
                (role, DataFile.from_dict(item)) for role, item in data.get("metadata", {}).items()
            ),
            shares_download_with=data.get("shares_download_with", ""),
            url=data.get("url", ""),
            url_template=data.get("url_template", ""),
            default_year=data.get("default_year"),
            default_quarter=data.get("default_quarter"),
            nodes=data.get("nodes"),
            num_paths=data.get("num_paths"),
            citations=tuple(data.get("citations", ())),
            license=data.get("license", ""),
        )

    def __str__(self) -> str:
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.num_paths is not None:
            shape.append(f"{self.num_paths:,} paths")
        domains = "/".join(self.domains) or "uncategorised"
        return f"{self.name} [{domains}]" + (f" ({', '.join(shape)})" if shape else "")


@dataclass(frozen=True)
class PathData(Network):
    """A downloaded path dataset on local disk.

    Everything a :class:`~netgamut.Network` offers, plus the
    stream of trajectories and the conversion into a memory network::

        data = paths.load("wikispeedia")
        for nodes, weight in data.paths():
            ...
        memory = data.to_state_network(order=2)
    """

    #: Which recipe in ``netgamut.paths._parsers`` reads this download.
    parser: str = ""

    def paths(
        self, *, max_paths: int | None = None, max_nodes: int | None = None
    ) -> Iterator[tuple[tuple[str, ...], float]]:
        """Stream every path as ``(nodes, weight)``.

        Weights count observations: passengers for flight itineraries, one
        per trail for navigation data. Streaming keeps memory flat; parsing
        details -- and any one-time derivation work -- live with the dataset's
        parser, see :mod:`netgamut.paths._parsers`.

        For demonstrations and small benchmarks the stream can be subset,
        deterministically:

        Args:
            max_paths: Keep only the first *max_paths* paths, in file order.
            max_nodes: Keep only the *max_nodes* most-visited physical nodes
                (visits weighted by path weight, ties broken by first
                appearance), applied after *max_paths*. Paths are *cut* at
                removed nodes rather than bridged across them -- a trajectory
                ``a -> x -> b`` with ``x`` removed yields no ``a -> b``
                transition -- and the surviving segments of at least two
                nodes are streamed as separate paths. Costs one extra pass
                over the (already limited) stream to rank the nodes.
        """
        if max_paths is not None and max_paths < 1:
            raise ValueError(f"max_paths must be at least 1, got {max_paths}.")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError(f"max_nodes must be at least 1, got {max_nodes}.")

        def limited() -> Iterator[tuple[tuple[str, ...], float]]:
            stream = self._parsed()
            if max_paths is None:
                return stream
            return islice(stream, max_paths)

        if max_nodes is None:
            return limited()

        visits: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        for nodes, weight in limited():
            for node in nodes:
                visits[node] = visits.get(node, 0.0) + weight
                first_seen.setdefault(node, len(first_seen))
        ranked = sorted(visits, key=lambda node: (-visits[node], first_seen[node]))
        keep = set(ranked[:max_nodes])

        def restricted() -> Iterator[tuple[tuple[str, ...], float]]:
            for nodes, weight in limited():
                segment: list[str] = []
                for node in nodes:
                    if node in keep:
                        segment.append(node)
                        continue
                    if len(segment) >= 2:
                        yield tuple(segment), weight
                    segment.clear()
                if len(segment) >= 2:
                    yield tuple(segment), weight

        return restricted()

    def _parsed(self) -> Iterator[tuple[tuple[str, ...], float]]:
        """The raw path stream from this dataset's parser."""
        from ._parsers import PARSERS

        try:
            parse = PARSERS[self.parser]
        except KeyError:
            known = ", ".join(sorted(PARSERS)) or "none"
            raise LookupError(f"No parser named {self.parser!r}. Available: {known}.") from None
        return parse(self)

    def to_state_network(
        self, *, order: int = 2, max_paths: int | None = None, max_nodes: int | None = None
    ) -> StateNetwork:
        """Build the memory network of the given Markov order from the paths.

        With the default order 2, every observed trigram ``i -> j -> k``
        contributes a link from state ``(i, j)`` to state ``(j, k)``; state
        ``(i, j)`` is physical node ``j`` remembering the arrival from ``i``.

        *max_paths* and *max_nodes* subset the trajectories first -- see
        :meth:`paths` -- which is the handle for demo-sized or benchmark-sized
        networks from a large dataset.
        """
        return StateNetwork.from_paths(
            self.paths(max_paths=max_paths, max_nodes=max_nodes), order=order, name=self.name
        )

    # -- side tables -------------------------------------------------------
    def coordinates(self) -> dict[str, tuple[float, float]]:
        """Node label -> ``(latitude, longitude)``, for plotting.

        Read from the dataset's ``"coordinates"`` table, fetched once into the
        same cache directory as the download. For the DB1B airport itineraries
        this is the OpenFlights airports table, keyed by three-letter IATA
        code, so ``coordinates().get(airport)`` positions a node -- a few small
        airfields in DB1B have no OpenFlights entry and come back ``None``.

        Raises:
            LookupError: This dataset ships no coordinates table.
        """
        from ._parsers import openflights_coordinates

        return openflights_coordinates(self._metadata_path("coordinates"))

    def names(self) -> dict[str, str]:
        """Node label -> human-readable name, from the dataset's ``"names"`` table.

        The DB1B city itineraries are labelled by BTS city-market id; this maps
        ``"31703"`` to ``"New York City, NY (Metropolitan Area)"`` via the BTS
        ``L_CITY_MARKET_ID`` lookup, fetched once next to the download.

        Raises:
            LookupError: This dataset ships no names table.
        """
        from ._parsers import code_descriptions

        return code_descriptions(self._metadata_path("names"))

    def _metadata_path(self, role: str) -> Path:
        """Fetch the side table with *role* on cache-once terms and return its path."""
        if not isinstance(self.dataset, Dataset):
            raise LookupError(f"{self.name!r} carries no paths metadata record.")
        artifact = self.dataset.metadata_file(role)
        target = self.archive_path.parent / Path(artifact.name).name
        _download_file(artifact.url, target, progress=False)
        return target

    def edges(self, *, cast: Callable[[str], Any] | None = None) -> Iterator[tuple[Any, Any]]:
        """Stream the first-order projection: consecutive pairs of every path.

        This is what makes :meth:`to_networkx` work unchanged on path data.
        Path weights are not repeated per pair; use :meth:`to_state_network`
        with ``order=1`` when weights matter.
        """
        for nodes, _weight in self.paths():
            for source, target in zip(nodes, nodes[1:], strict=False):
                if cast is None:
                    yield source, target
                else:
                    yield cast(source), cast(target)

    def to_pajek(
        self,
        directory: str | Path,
        *,
        name: str | None = None,
        directed: bool | None = None,
        order: int = 1,
    ) -> Path:
        """Write the network the paths make, weights kept, and return its path.

        Order 1 by default, matching :meth:`to_infomap`: a directed Pajek file
        of the first-order network. ``order=2`` and up writes the memory
        network in Infomap's states format; see
        :meth:`StateNetwork.to_pajek`. *directed* is accepted for signature
        parity with :meth:`Network.to_pajek` and must not be ``False``:
        transitions along a path are directed.
        """
        if directed is False:
            raise ValueError("path data is directed; transitions have a direction.")
        return self.to_state_network(order=order).to_pajek(directory, name=name)

    def to_link_list(
        self, directory: str | Path, *, name: str | None = None, directed: bool | None = None
    ) -> Path:
        """Write the first-order network of the paths as a link list.

        Node labels are numbered from 1 in first-seen order -- a link list
        cannot hold names -- so prefer :meth:`to_pajek` unless the labels are
        meaningless anyway.
        """
        if directed is False:
            raise ValueError("path data is directed; transitions have a direction.")
        return self.to_state_network(order=1).to_link_list(directory, name=name)

    def to_infomap(self, *, directed: bool | None = None, order: int = 1, **kwargs: Any) -> Any:
        """Build an ``infomap.Infomap`` instance from the paths, weights kept.

        Defaults to the first-order network, matching what ``to_infomap``
        means everywhere else in this package; pass ``order=2`` for the
        memory network, or call :meth:`to_state_network` first to keep the
        intermediate representation.
        """
        if directed is not None:
            kwargs["directed"] = directed
        return self.to_state_network(order=order).to_infomap(**kwargs)
