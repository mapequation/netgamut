"""Path datasets and the memory networks built from them.

Flows in real systems remember where they came from: a passenger's next
flight depends on the itinerary so far, a reader's next click on the trail
behind it. This source curates datasets whose unit of observation is a whole
*path*, and pairs them with the conversion into Infomap state networks::

    from networks import paths

    data = paths.load("wikispeedia")     # 9.5 MB, 51,318 navigation trails
    memory = data.to_state_network(order=2)
    memory.write("wikispeedia_states.net")   # *Vertices / *States / *Links
    im = memory.to_infomap(silent=True)
    im.run()

With order 2, every observed trigram ``i -> j -> k`` adds a link from state
``(i, j)`` to state ``(j, k)`` -- being at ``j``, remembering ``i``. Order 1
gives the plain directed network from the same paths, so first-order versus
second-order comparisons are a one-argument change.

The flight itineraries come in two views of one download -- ``db1b-coupon``
between airports, ``db1b-city`` between metropolitan areas -- and carry the
side tables a plot needs::

    flights = paths.load("db1b-coupon")
    where = flights.coordinates()        # {"LAX": (33.94, -118.41), ...}
    cities = paths.load("db1b-city")     # same archive, no second download
    cities.names()["31703"]              # 'New York City, NY (Metropolitan Area)'

The collection is curated here rather than scraped from a site, which is what
guarantees every entry is downloadable without registration and ships with a
parser that knows its quirks (Wikispeedia's back clicks, DB1B's scattered
coupon rows).

Please cite the datasets you use; see ``Dataset.citations``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .._core.cache import cache_dir, cache_info, clear_cache, set_cache_dir
from .._core.cache import dataset_dir as _dataset_dir
from .._core.domains import DOMAINS
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.state_network import StateNetwork
from ._catalog import Catalog, load_catalog, refresh
from ._models import Dataset, PathData

__all__ = [
    "DOMAINS",
    "Catalog",
    "DataFile",
    "Dataset",
    "DownloadError",
    "PathData",
    "StateNetwork",
    "cache_dir",
    "cache_info",
    "catalog",
    "clear_cache",
    "datasets",
    "describe",
    "domains",
    "info",
    "is_cached",
    "load",
    "refresh",
    "search",
    "set_cache_dir",
]

SOURCE = "paths"


def catalog(*, reload: bool = False) -> Catalog:
    """The curated catalog of path datasets, vendored inside the package."""
    return load_catalog(reload=reload)


def datasets(domain: str | None = None) -> tuple[Dataset, ...]:
    """Every path dataset, or just those in *domain*."""
    current = catalog()
    return current.search(domain=domain) if domain else current.datasets


def info(name: str) -> Dataset:
    """Metadata for one dataset. Never downloads anything."""
    return catalog().dataset(name)


def domains() -> dict[str, int]:
    """Dataset counts per normalised domain."""
    return catalog().domains()


def search(
    query: str = "",
    *,
    domain: str | None = None,
    directed: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
) -> tuple[Dataset, ...]:
    """Find path datasets by name, description or domain."""
    return catalog().search(
        query,
        domain=domain,
        directed=directed,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        max_edges=max_edges,
        min_edges=min_edges,
    )


def describe(name: str) -> str:
    """A readable metadata summary for one dataset, for printing."""
    dataset = info(name)
    lines = [dataset.name, "=" * len(dataset.name)]
    lines.append(f"Domains     {', '.join(dataset.domains) or '-'}")
    if dataset.nodes is not None:
        lines.append(f"Nodes       {dataset.nodes:,}")
    if dataset.num_paths is not None:
        lines.append(f"Paths       {dataset.num_paths:,}")
    if dataset.quarterly:
        lines.append(
            f"Quarterly   one download per quarter; default "
            f"{dataset.default_year} Q{dataset.default_quarter}"
        )
    if dataset.description:
        lines.append(f"Summary     {dataset.description[:400]}")
    lines.append(f"Page        {dataset.url}")

    if dataset.files or dataset.metadata:
        lines.append("")
        lines.append("Files")
        for index, item in enumerate(dataset.files):
            marker = "*" if index == 0 else " "
            lines.append(f"  {marker} {item.name}")
            if item.description:
                lines.append(f"    {item.description}")
        for role, item in dataset.metadata:
            lines.append(f"  + {item.name}  [{role}]")
            if item.description:
                lines.append(f"    {item.description}")
        legend = "* downloaded by default" + (", + fetched on demand" if dataset.metadata else "")
        lines.append(f"  ({legend})")
    if dataset.shares_download_with:
        lines.append(f"Download    shared with {dataset.shares_download_with}")

    if dataset.citations:
        lines.append("")
        lines.append("Source")
        for citation in dataset.citations:
            lines.append(f"  {citation[:200]}")
    if dataset.license:
        lines.append(f"License     {dataset.license}")

    return "\n".join(lines)


def load(
    name: str,
    *,
    year: int | None = None,
    quarter: int | None = None,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> PathData:
    """Download *name* if not already cached, and return a handle to it.

    The handle adds :meth:`~networks.paths.PathData.paths` and
    :meth:`~networks.paths.PathData.to_state_network` on top of the usual
    :class:`~networks.Network` interface, plus
    :meth:`~networks.paths.PathData.coordinates` and
    :meth:`~networks.paths.PathData.names` for the entries that ship those
    side tables. Entries that are views of one archive (``db1b-city`` reads
    ``db1b-coupon``'s file) share its cache directory, so the download
    happens once.

    Args:
        name: Entry name, case-insensitive.
        year: For quarterly collections (``db1b-coupon``, ``db1b-city``):
            which year to fetch. Defaults to the dataset's verified default
            quarter.
        quarter: 1-4, together with *year*.
    """
    dataset = info(name)
    artifact = dataset.file_for(year=year, quarter=quarter)
    target = _dataset_dir(SOURCE, dataset.cache_name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return PathData(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        parser=dataset.parser,
    )


def is_cached(name: str, *, year: int | None = None, quarter: int | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
        artifact = dataset.file_for(year=year, quarter=quarter)
    except (KeyError, LookupError, ValueError):
        return False
    return (_dataset_dir(SOURCE, dataset.cache_name) / Path(artifact.name).name).is_file()
