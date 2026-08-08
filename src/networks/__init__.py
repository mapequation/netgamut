"""Browse, download and cache network datasets.

Each repository lives in its own subpackage:

* :mod:`networks.snap` -- the Stanford Large Network Dataset Collection, 109
  datasets. Large, well known, and skewed to social, web and technological
  systems.
* :mod:`networks.netzschleuder` -- networks.skewed.de, 286 networks tagged by
  domain, including the biological, transport and multilayer networks SNAP has
  none of.
* :mod:`networks.paths` -- curated path data (flight itineraries, navigation
  trails) plus the conversion into second-order state networks for Infomap's
  memory model, which no edge list can express.

Metadata for every source ships inside the package, so browsing is offline and
instant; only ``load()`` touches the network, and it caches into a shared
user-level directory so a dataset is fetched once per machine.

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

from . import netzschleuder, paths, snap
from ._core.cache import (
    CACHE_ENV_VAR,
    CachedDataset,
    CacheInfo,
    cache_dir,
    cache_info,
    clear_cache,
    set_cache_dir,
)
from ._core.domains import DOMAINS
from ._core.network import Network
from ._core.state_network import StateNetwork

__all__ = [
    "CACHE_ENV_VAR",
    "DOMAINS",
    "CacheInfo",
    "CachedDataset",
    "Match",
    "Network",
    "StateNetwork",
    "__version__",
    "cache_dir",
    "cache_info",
    "clear_cache",
    "domains",
    "load",
    "netzschleuder",
    "paths",
    "search",
    "set_cache_dir",
    "snap",
    "sources",
]

__version__ = "0.2.0"

#: Source name -> subpackage, in the order results are reported.
_SOURCES: dict[str, Any] = {"snap": snap, "netzschleuder": netzschleuder, "paths": paths}


def sources() -> tuple[str, ...]:
    """The available data sources."""
    return tuple(_SOURCES)


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

    A dataset listed under two domains counts once in each.
    """
    counts = dict.fromkeys(DOMAINS, 0)
    for module in _SOURCES.values():
        for dataset in module.datasets():
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
) -> tuple[Match, ...]:
    """Search every source at once, filtering on the shared domain vocabulary.

    Only the fields both sources agree on are accepted here. For anything
    source-specific -- SNAP's ground-truth partitions, Netzschleuder's
    multilayer flag -- use that source's own ``search()``.

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
            results.append(Match(source=name, dataset=dataset))

    return tuple(results)


def load(name: str, *, source: str | None = None, **kwargs: Any) -> Network:
    """Download a dataset by name, searching every source.

    Pass *source* to disambiguate. Without it, a name present in more than one
    source is an error rather than an arbitrary choice.

    Raises:
        KeyError: No source has *name*, or several do and *source* was omitted.
    """
    if source is not None:
        if source not in _SOURCES:
            raise KeyError(f"Unknown source {source!r}. Available: {', '.join(_SOURCES)}")
        result: Network = _SOURCES[source].load(name, **kwargs)
        return result

    found = [key for key, module in _SOURCES.items() if name in module.catalog()]
    if not found:
        raise KeyError(f"No dataset named {name!r} in any source: {', '.join(_SOURCES)}.")
    if len(found) > 1:
        raise KeyError(
            f"{name!r} exists in several sources ({', '.join(found)}). Pass source= to choose."
        )
    loaded: Network = _SOURCES[found[0]].load(name, **kwargs)
    return loaded
