"""Access to the Netzschleuder network repository (networks.skewed.de).

Netzschleuder is curated by Tiago Peixoto and tags every network by domain,
which makes it the source to draw on when a comparison needs to span *kinds of
system* rather than kinds of provenance::

    from networks import netzschleuder as nz

    nz.domains()
    # {'Social': 148, 'Informational': 53, 'Biological': 36, ...}

    for dataset in nz.search(domain="Biological", max_nodes=1000):
        print(dataset)

It complements :mod:`networks.snap`, which has no biological, multilayer or
non-road transport networks at all.

Downloads use the site's CSV bundle, so nothing beyond the standard library is
required -- graph-tool is not needed despite it being the site's native format.

Please cite the networks you use; see ``Dataset.citations``.
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
from .._core.network import Network
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._models import SITE_URL, Dataset

__all__ = [
    "DOMAINS",
    "SITE_URL",
    "Catalog",
    "DataFile",
    "Dataset",
    "DownloadError",
    "Network",
    "build_catalog",
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
    "tags",
]

SOURCE = "netzschleuder"

#: The member of the CSV bundle holding the edge list.
_EDGES_MEMBER = "edges.csv"


def catalog(*, reload: bool = False) -> Catalog:
    """The Netzschleuder catalog, from the cache if refreshed, else the package."""
    return load_catalog(reload=reload)


def datasets(domain: str | None = None) -> tuple[Dataset, ...]:
    """Every network, or just those in *domain*."""
    current = catalog()
    return current.search(domain=domain) if domain else current.datasets


def info(name: str) -> Dataset:
    """Metadata for one network. Never downloads anything."""
    return catalog().dataset(name)


def domains() -> dict[str, int]:
    """Network counts per normalised domain."""
    return catalog().domains()


def tags() -> dict[str, int]:
    """Every tag the site uses, with counts, most common first."""
    return catalog().tags()


def search(
    query: str = "",
    *,
    domain: str | None = None,
    subtype: str | None = None,
    directed: bool | None = None,
    weighted: bool | None = None,
    multilayer: bool | None = None,
    bipartite: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
) -> tuple[Dataset, ...]:
    """Find networks by domain, subtype, size or structural property.

    For a domain-balanced comparison set::

        nz.search(domain="Biological", max_nodes=1000)

    For Infomap's multilayer model, which SNAP cannot serve at all::

        nz.search(multilayer=True)
    """
    return catalog().search(
        query,
        domain=domain,
        subtype=subtype,
        directed=directed,
        weighted=weighted,
        multilayer=multilayer,
        bipartite=bipartite,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        max_edges=max_edges,
        min_edges=min_edges,
    )


def describe(name: str) -> str:
    """A readable metadata summary for one network, for printing."""
    dataset = info(name)
    lines = [dataset.name, "=" * len(dataset.name)]
    lines.append(f"Domains     {', '.join(dataset.domains) or '-'}")
    if dataset.subtypes:
        lines.append(f"Subtypes    {', '.join(dataset.subtypes)}")
    properties = [
        label
        for label, present in (
            ("directed", bool(dataset.directed)),
            ("weighted", dataset.weighted),
            ("multilayer", dataset.multilayer),
            ("temporal", dataset.temporal),
            ("bipartite", bool(dataset.bipartite)),
        )
        if present
    ]
    lines.append(f"Properties  {', '.join(properties) or 'unweighted, undirected'}")
    if dataset.description:
        lines.append(f"Summary     {dataset.description[:400]}")
    lines.append(f"Page        {dataset.url}")

    if dataset.analyses:
        lines.append("")
        lines.append("Reported measures")
        for key in (
            "num_vertices",
            "num_edges",
            "average_degree",
            "global_clustering",
            "degree_assortativity",
            "diameter",
            "largest_component_fraction",
        ):
            if key in dataset.analyses:
                value = dataset.analyses[key]
                shown = f"{value:.4g}" if isinstance(value, float) else f"{value}"
                lines.append(f"  {key:<28} {shown}")

    if dataset.subgraphs and len(dataset.subgraphs) > 1:
        lines.append("")
        lines.append(f"Subgraphs   {len(dataset.subgraphs)}: {', '.join(dataset.subgraphs[:8])}")

    if dataset.citations:
        lines.append("")
        lines.append("Source")
        for citation in dataset.citations:
            lines.append(f"  {citation[:200]}")

    return "\n".join(lines)


def load(
    name: str,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download *name* if not already cached, and return a handle to it.

    The CSV bundle is fetched and its ``edges.csv`` used as the network, so
    ``Network.path``, ``.edges()`` and ``.to_infomap()`` work exactly as they do
    for SNAP data. The bundle's node and graph properties stay available
    alongside via :meth:`Network.extract`.
    """
    dataset = info(name)
    artifact = dataset.files[0]
    target = _dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        member=_EDGES_MEMBER,
    )


def is_cached(name: str) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.files[0]
    return (_dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name).is_file()
