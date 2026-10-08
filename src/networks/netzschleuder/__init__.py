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

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._core.cache import cache_dir, cache_info, clear_cache, set_cache_dir
from .._core.cache import dataset_dir as _dataset_dir
from .._core.columns import Columns
from .._core.domains import DOMAINS
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.network import Network
from .._core.nodes import NodeTable
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
    "node_table",
    "refresh",
    "search",
    "set_cache_dir",
    "tags",
]

SOURCE = "netzschleuder"

#: The member of the CSV bundle holding the edge list.
_EDGES_MEMBER = "edges.csv"

#: Edge-property names that carry each role, in order of preference. ``count``
#: is a multiplicity -- migrants, visits -- and so a weight; ``year`` is a time.
_WEIGHT_NAMES = ("weight", "count")
_TIME_NAMES = ("time", "timestamp", "year")
_LAYER_NAMES = ("layer",)
#: Property types that cannot be a link weight: a boolean ``weight`` only marks
#: presence, and text is not a number.
_NOT_NUMERIC = ("bool", "string")
_NODES_MEMBER = "nodes.csv"


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

    if dataset.subgraph_count > 1:
        lines.append("")
        lines.append(
            f"Subgraphs   {dataset.subgraph_count} "
            f"(default {dataset.default_subgraph}): {', '.join(dataset.subgraphs[:8])}"
        )

    if dataset.citations:
        lines.append("")
        lines.append("Source")
        for citation in dataset.citations:
            lines.append(f"  {citation[:200]}")

    return "\n".join(lines)


def load(
    name: str,
    *,
    subgraph: str | None = None,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download *name* if not already cached, and return a handle to it.

    The CSV bundle is fetched and its ``edges.csv`` used as the network, so
    ``Network.path``, ``.edges()`` and ``.to_infomap()`` work exactly as they do
    for SNAP data. The bundle's node and graph properties stay available
    alongside via :meth:`Network.extract` and :meth:`Network.nodes`.

    Args:
        name: Entry name, case-insensitive.
        subgraph: Which subgraph to fetch. Entries usually have one, named after
            the entry; some publish several -- ``karate`` holds ``77`` and
            ``78``. Defaults to ``Dataset.default_subgraph``. See
            ``info(name).subgraph_count``.
    """
    dataset = info(name)
    artifact = dataset.files_for(subgraph)[0]
    target = _dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    network = Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        member=_EDGES_MEMBER,
        source=SOURCE,
    )
    return dataclasses.replace(network, columns=_columns(network, dataset))


def _columns(network: Network, dataset: Dataset) -> Columns:
    """Column roles from the file's own ``# source, target, ...`` header.

    The catalog's edge properties are the union over a dataset's subgraphs, and
    one subgraph's file can hold fewer of them, in its own order -- copenhagen's
    ``sms`` has ``timestamp`` where the catalog lists ``timestamp, rssi`` -- so
    the header of the file actually loaded names the columns, and the catalog
    only says which of them are numeric.
    """
    header = network.header
    if not header:
        return Columns()
    names = tuple(field.strip() for field in header[0].split(","))
    types = {name: kind for name, kind in dataset.analyses.get("edge_properties") or ()}

    def first(candidates: tuple[str, ...], *, numeric: bool = False) -> int | None:
        for candidate in candidates:
            if candidate in names[2:] and not (numeric and types.get(candidate) in _NOT_NUMERIC):
                return names.index(candidate)
        return None

    return Columns(
        weight=first(_WEIGHT_NAMES, numeric=True),
        time=first(_TIME_NAMES),
        layer=first(_LAYER_NAMES),
        names=names,
    )


def node_table(network: Network) -> NodeTable:
    """Read a Netzschleuder network's per-node metadata into the shared shape.

    The CSV bundle already downloaded for the edge list carries ``nodes.csv`` beside
    it, so this costs an extraction and no network access. Every column the site
    published comes through under its own name: ``karate`` labels each member with
    their faction under ``groups``, ``football`` each team with its conference under
    ``value``. Netzschleuder has no separate ground-truth file, so these columns are
    what a community-detection run gets scored against -- with the usual caveat that
    an attribute is not ground truth for community structure.

    Node ids are the row order the bundle uses, which is what ``edges.csv`` refers to.

    Raises:
        LookupError: The bundle ships no ``nodes.csv``.
    """
    import csv

    directory = network.extract()
    path = directory / _NODES_MEMBER
    if not path.is_file():
        raise LookupError(f"{network.name!r} ships no {_NODES_MEMBER} in its bundle.")

    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        # The site prefixes its header row with "# ".
        columns = [cell.strip().lstrip("# ").strip() for cell in header]
        values: dict[str, dict[Any, Any]] = {column: {} for column in columns}
        for node, record in enumerate(reader):
            if not record:
                continue
            for column, cell in zip(columns, record, strict=False):
                values[column][node] = cell

    return NodeTable(values=values, origin=f"{SOURCE}/{network.name}/{_NODES_MEMBER}")


def is_cached(name: str, *, subgraph: str | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.files_for(subgraph)[0]
    return (_dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name).is_file()
