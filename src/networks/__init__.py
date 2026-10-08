"""Browse, download and cache network datasets.

Each repository lives in its own subpackage:

* :mod:`networks.snap` -- the Stanford Large Network Dataset Collection, 109
  datasets. Large, well known, and skewed to social, web and technological
  systems.
* :mod:`networks.netzschleuder` -- networks.skewed.de, 286 networks tagged by
  domain, including the biological, transport and multilayer networks SNAP has
  none of.
* :mod:`networks.konect` -- KONECT, the Koblenz Network Collection, 300
  networks described in one fixed vocabulary of format, edge type and loops.
  A network Netzschleuder or SNAP also holds is listed there but used from the
  other source here, so nothing is counted twice.
* :mod:`networks.ogb` -- Open Graph Benchmark, 12 datasets whose node
  categories are *named* rather than left as anonymous codes.
* :mod:`networks.pyg` -- the PyTorch Geometric corpus, 99 datasets: citation,
  web, social and heterophilous graphs.
* :mod:`networks.paths` -- curated path data (flight itineraries, navigation
  trails) plus the conversion into second-order state networks for Infomap's
  memory model, which no edge list can express.
* :mod:`networks.ahorn` -- the Aachen Higher-Order Repository of Networks, 83
  hypergraphs and simplicial complexes. Its unit is a *set* of nodes rather than
  a pair, so it returns a
  :class:`~networks._core.hypergraph.Hypergraph` rather than a ``Network``.

Metadata for every source ships inside the package, so browsing is offline and
instant; only ``load()`` touches the network, and it caches into a shared
user-level directory so a dataset is fetched once per machine.

Beyond the datasets, :mod:`networks.generate` samples synthetic benchmark
networks with planted communities from any blueprint in the pool (requires
the ``generate`` extra).

Every network -- downloaded or generated, first-order, state or multilayer --
converts and writes the same way: ``to_networkx()``, ``to_infomap()``, and
``to_pajek(directory)`` / ``to_link_list(directory)``, which return the path
of the file they wrote.

All sources are mapped onto one domain vocabulary, so a comparison can span
kinds of system rather than kinds of provenance::

    import networks

    networks.domains()
    # {'Social': 201, 'Informational': 68, 'Biological': 36, ...}

    for dataset in networks.search(domain="Biological", max_nodes=1000):
        print(dataset.source, dataset.name)

Please cite the datasets you use -- each source's records carry the reference
its site asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import ahorn, generate, konect, netzschleuder, ogb, paths, pyg, snap
from ._core.cache import (
    CACHE_ENV_VAR,
    CachedDataset,
    CacheInfo,
    cache_dir,
    cache_info,
    clear_cache,
    set_cache_dir,
)
from ._core.columns import Columns
from ._core.domains import DOMAINS
from ._core.hypergraph import Hypergraph
from ._core.in_memory import InMemoryNetwork
from ._core.multilayer import MultilayerNetwork
from ._core.network import Network
from ._core.state_network import StateNetwork

__all__ = [
    "CACHE_ENV_VAR",
    "DOMAINS",
    "CacheInfo",
    "CachedDataset",
    "Columns",
    "Hypergraph",
    "Match",
    "MultilayerNetwork",
    "Network",
    "StateNetwork",
    "InMemoryNetwork",
    "__version__",
    "ahorn",
    "cache_dir",
    "cache_info",
    "clear_cache",
    "domains",
    "generate",
    "konect",
    "load",
    "netzschleuder",
    "ogb",
    "paths",
    "pyg",
    "search",
    "set_cache_dir",
    "snap",
    "sources",
]

__version__ = "0.3.0"

#: Source name -> subpackage, in the order results are reported.
_SOURCES: dict[str, Any] = {
    "snap": snap,
    "netzschleuder": netzschleuder,
    "konect": konect,
    "ogb": ogb,
    "pyg": pyg,
    "paths": paths,
    "ahorn": ahorn,
}


def sources() -> tuple[str, ...]:
    """The available data sources."""
    return tuple(_SOURCES)


def _shadowed(source: str, dataset: Any) -> bool:
    """Whether cross-source queries skip *dataset* for another source's copy.

    Only KONECT defers: a third of it is Netzschleuder's or SNAP's data. See
    :func:`networks.konect.duplicates`.
    """
    return source == "konect" and dataset.duplicate_of is not None


def _owners(name: str) -> list[str]:
    """The sources holding *name*, leaving out a KONECT copy of another's."""
    found = [key for key, module in _SOURCES.items() if name in module.catalog()]
    if len(found) > 1 and "konect" in found and _shadowed("konect", konect.info(name)):
        found.remove("konect")
    return found


@dataclass(frozen=True, slots=True)
class Match:
    """One dataset found by a cross-source query.

    Keeps the source alongside the record, because the two catalogs use
    different names for the same idea and a few names appear in both.
    """

    source: str
    dataset: Any

    @property
    def name(self) -> str:
        name: str = self.dataset.name
        return name

    @property
    def domains(self) -> tuple[str, ...]:
        result: tuple[str, ...] = self.dataset.domains
        return result

    @property
    def nodes(self) -> int | None:
        return self.dataset.nodes  # type: ignore[no-any-return]

    @property
    def edges(self) -> int | None:
        return self.dataset.edges  # type: ignore[no-any-return]

    @property
    def directed(self) -> bool | None:
        return self.dataset.directed  # type: ignore[no-any-return]

    @property
    def node_metadata(self) -> bool | None:
        """Whether :meth:`Network.nodes` returns per-node attributes; ``None`` if unknown.

        Each source answers from its own catalog, before any download: KONECT and
        Netzschleuder from their file listings and analyses, SNAP from its label
        files, OGB and PyG from their class labels. AHORN keeps node lines inside
        the dataset file, so it cannot say.
        """
        value = getattr(self.dataset, "node_metadata", None)
        return None if value is None else bool(value)

    @property
    def node_attributes(self) -> tuple[str, ...] | None:
        """The per-node columns the source's catalog can name, ``None`` if unknown.

        Exact for Netzschleuder, KONECT and SNAP. OGB names only its extra files
        (``year``, ``species``) and PyG none, since their other column names come
        with the download; ``nodes()`` returns those too.
        """
        value = getattr(self.dataset, "node_attributes", None)
        return None if value is None else tuple(value)

    @property
    def loops(self) -> bool | None:
        """Whether any node links to itself, where some source has checked.

        KONECT checks every network; for another source's network, KONECT's
        check is used when it holds the same edge list. ``None`` means unknown.
        """
        value = getattr(self.dataset, "loops", None)
        if value is not None:
            return bool(value)
        return konect.loops_for(self.key)

    @property
    def key(self) -> str:
        """``"<source>/<name>"`` -- unique across sources."""
        return f"{self.source}/{self.name}"

    def load(self, **kwargs: Any) -> Network:
        """Download this dataset via its own source."""
        result: Network = _SOURCES[self.source].load(self.name, **kwargs)
        return result

    def __str__(self) -> str:
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.edges is not None:
            shape.append(f"{self.edges:,} edges")
        domains = "/".join(self.domains) or "uncategorised"
        tail = f" ({', '.join(shape)})" if shape else ""
        return f"{self.key} [{domains}]{tail}"


def domains() -> dict[str, int]:
    """Dataset counts per domain, summed over every source.

    A dataset listed under two domains counts once in each, and a network held
    by two sources counts once.
    """
    counts = dict.fromkeys(DOMAINS, 0)
    for source, module in _SOURCES.items():
        for dataset in module.datasets():
            if _shadowed(source, dataset):
                continue
            for domain in dataset.domains:
                counts[domain] += 1
    return counts


def search(
    query: str = "",
    *,
    domain: str | None = None,
    source: str | None = None,
    directed: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
    loops: bool | None = None,
    node_metadata: bool | None = None,
    node_attribute: str | None = None,
) -> tuple[Match, ...]:
    """Search every source at once, filtering on the shared domain vocabulary.

    Only the fields every source agrees on are accepted here. For anything
    source-specific -- SNAP's ground-truth partitions, Netzschleuder's multilayer
    flag, OGB's loadability, PyG's class -- use that source's own ``search()``.

    Size filters exclude datasets whose size the source does not publish rather than
    assuming anything: most PyG entries and the largest OGB ones have no counts.
    ``loops`` does the same with datasets no source has checked for self-links
    (see :attr:`Match.loops`), and ``node_metadata`` with datasets whose catalog
    cannot say (see :attr:`Match.node_metadata`). ``node_attribute`` finds
    datasets with a per-node column of that name, case-insensitively, among the
    names each catalog knows -- ``node_attribute="gender"`` for something to score
    a partition against.

    A network held by both KONECT and Netzschleuder or SNAP appears once, from
    the other source; ``source="konect"`` lists KONECT's copy too.

    Raises:
        KeyError: Unknown *domain* or *source*.
    """
    if domain is not None and domain not in DOMAINS:
        raise KeyError(f"Unknown domain {domain!r}. Available: {', '.join(DOMAINS)}")
    if source is not None and source not in _SOURCES:
        raise KeyError(f"Unknown source {source!r}. Available: {', '.join(_SOURCES)}")

    needle = query.casefold().strip()
    results: list[Match] = []

    def within(value: int | None, low: int | None, high: int | None) -> bool:
        if low is None and high is None:
            return True
        if value is None:
            return False
        return (low is None or value >= low) and (high is None or value <= high)

    for name, module in _SOURCES.items():
        if source is not None and name != source:
            continue
        for dataset in module.datasets():
            if source is None and _shadowed(name, dataset):
                continue
            if domain is not None and domain not in dataset.domains:
                continue
            text = f"{dataset.name} {getattr(dataset, 'description', '')}".casefold()
            if needle and needle not in text:
                continue
            if directed is not None and dataset.directed is not directed:
                continue
            if not within(dataset.nodes, min_nodes, max_nodes):
                continue
            if not within(dataset.edges, min_edges, max_edges):
                continue
            match = Match(source=name, dataset=dataset)
            if loops is not None and match.loops is not loops:
                continue
            if node_metadata is not None and match.node_metadata is not node_metadata:
                continue
            if node_attribute is not None and not any(
                column.casefold() == node_attribute.casefold()
                for column in match.node_attributes or ()
            ):
                continue
            results.append(match)

    return tuple(results)


def load(name: str, *, source: str | None = None, **kwargs: Any) -> Network:
    """Download a dataset by name, searching every source.

    Pass *source* to disambiguate. Without it, a name present in more than one
    source is an error rather than an arbitrary choice -- except that KONECT's
    copy of another source's network defers to that source.

    Raises:
        KeyError: No source has *name*, or several do and *source* was omitted.
    """
    if source is not None:
        if source not in _SOURCES:
            raise KeyError(f"Unknown source {source!r}. Available: {', '.join(_SOURCES)}")
        result: Network = _SOURCES[source].load(name, **kwargs)
        return result

    found = _owners(name)
    if not found:
        raise KeyError(f"No dataset named {name!r} in any source: {', '.join(_SOURCES)}.")
    if len(found) > 1:
        raise KeyError(
            f"{name!r} exists in several sources ({', '.join(found)}). Pass source= to choose."
        )
    loaded: Network = _SOURCES[found[0]].load(name, **kwargs)
    return loaded
