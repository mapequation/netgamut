"""Access to the Netzschleuder network repository (networks.skewed.de).

Netzschleuder is curated by Tiago Peixoto and tags every network by domain,
which makes it the source to draw on when a comparison needs to span *kinds of
system* rather than kinds of provenance::

    from netgamut import netzschleuder as nz

    nz.domains()
    # {'Social': 148, 'Informational': 53, 'Biological': 36, ...}

    for dataset in nz.search(domain="Biological", max_nodes=1000):
        print(dataset)

It complements :mod:`netgamut.snap`, which has no biological, multilayer or
non-road transport networks at all.

Downloads use the site's CSV bundle, so nothing beyond the standard library is
required -- graph-tool is not needed despite it being the site's native format.

Please cite the networks you use; see ``Dataset.citations``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .._core.cache import cache_dir, cache_info, clear_cache, set_cache_dir
from .._core.cache import dataset_dir as _dataset_dir
from .._core.columns import Columns
from .._core.domains import DOMAINS
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.multilayer import MultilayerNetwork
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
    "load_multilayer",
    "node_names",
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
#: Entries whose layer column has another name: a transmitter for each layer of
#: the C. elegans wireless connectome, a mode of transport for the others.
_LAYER_COLUMNS = {
    "celegans_extrasynaptic": "transmitter",
    "kujala_public_transport": "route_type",
    "national_public_transport": "route_type",
}
#: Entries that keep each layer in a subgraph of its own: the subgraphs that make
#: up the default multilayer network, and the ``nodes.csv`` column that names a
#: node in every one of them -- ``index``, the row number, where they all number
#: the same nodes alike. ugandan_village holds two layers for each of 17
#: villages, so its default is the first village.
_SUBGRAPH_LAYERS: dict[str, tuple[tuple[str, ...], str]] = {
    "copenhagen": (("sms", "fb_friends", "calls", "bt"), "id"),
    "malaria_genes": (tuple(f"HVR_{number}" for number in range(1, 10)), "name"),
    "twitter_higgs": (("social", "retweet", "reply", "mention"), "name"),
    "ugandan_village": (("friendship-1", "health-advice_1"), "name"),
    "youtube_interactions": (
        (
            "contacts",
            "shared_friends",
            "shared_subscriptions",
            "shared_subscribers",
            "shared_favorites",
        ),
        "index",
    ),
}
#: Columns that can name a node across subgraphs when an entry is not listed above.
_KEY_NAMES = ("id", "name")
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


def load_multilayer(
    name: str,
    subgraphs: Sequence[str] | None = None,
    *,
    key: str | None = None,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> MultilayerNetwork:
    """Download *name* and return it as a :class:`~netgamut.MultilayerNetwork`.

    Without *subgraphs*, the layers are where the entry keeps them. Most
    multilayer entries hold a layer column in their default subgraph:
    ``layer`` for london_transport, ``route_type`` for the public transport
    entries, ``transmitter`` for celegans_extrasynaptic. copenhagen,
    malaria_genes, twitter_higgs, ugandan_village and youtube_interactions
    keep one layer per subgraph, and those are stacked.

    With *subgraphs*, each one named is a layer, as
    :meth:`MultilayerNetwork.from_networks` stacks them. Every subgraph numbers
    its nodes ``0..n-1`` on its own, so a node is matched across them by the
    ``nodes.csv`` column *key*, such as copenhagen's ``id``. By default that is
    the entry's known key, else the first of ``id`` and ``name`` that the
    subgraphs have. Pass ``key="index"`` when they all number the same nodes
    alike.

    For a column-layered entry's other subgraphs, load one and lay it out:
    ``load("kujala_public_transport", subgraph="berlin").to_multilayer()``.
    The other arguments are :func:`load`'s, applied to each download.

    Raises:
        LookupError: The entry has no layer column, and no subgraphs were named
            that netgamut knows to be its layers; or no *key* fits.
    """
    dataset = info(name)

    def fetch(subgraph: str | None = None) -> Network:
        return load(
            dataset.name, subgraph=subgraph, force=force, progress=progress, max_bytes=max_bytes
        )

    known = _SUBGRAPH_LAYERS.get(dataset.name)
    if subgraphs is None:
        if known is None:
            network = fetch()
            if network.columns.layer is None:
                raise LookupError(
                    f"{dataset.name!r} has no layer column, and netgamut knows no subgraphs "
                    "that are its layers; name them with subgraphs=[...]."
                )
            return network.to_multilayer()
        subgraphs = known[0]
    layers = {subgraph: fetch(subgraph) for subgraph in subgraphs}
    if key is None:
        key = known[1] if known is not None else _node_key(layers)
    stacked = MultilayerNetwork.from_networks(
        layers, key=None if key == "index" else key, name=dataset.name
    )
    info_ = {**stacked.info, "layers": "subgraph", "key": key}
    return dataclasses.replace(stacked, info=info_)


def _node_key(layers: dict[str, Network]) -> str:
    """The first of :data:`_KEY_NAMES` that every subgraph's ``nodes.csv`` has."""
    shared: set[str] | None = None
    for network in layers.values():
        try:
            columns = set(network.nodes().columns)
        except LookupError:
            columns = set()
        shared = columns if shared is None else shared & columns
    for candidate in _KEY_NAMES:
        if candidate in (shared or ()):
            return candidate
    raise LookupError(
        f"the subgraphs share no {' or '.join(_KEY_NAMES)} column to match nodes by; "
        "pass key=, or key='index' if they all number the same nodes alike."
    )


def _columns(network: Network, dataset: Dataset) -> Columns:
    """Column roles from the file's own ``# source, target, ...`` header.

    The catalog's edge properties are the union over a dataset's subgraphs, and
    one subgraph's file can hold fewer of them, in its own order -- copenhagen's
    ``sms`` has ``timestamp`` where the catalog lists ``timestamp, rssi`` -- so
    the header of the file actually loaded names the columns, and the catalog
    only says which of them are numeric.

    The file is always CSV, so the delimiter is declared rather than detected:
    detection splits lines, and a quoted field can hold a comma (``wiki_rfa``'s
    dates) or a line break (``product_space``'s colours).
    """
    header = network.header
    if not header:
        return Columns(delimiter=",")
    names = tuple(field.strip() for field in header[0].split(","))
    types = {name: kind for name, kind in dataset.analyses.get("edge_properties") or ()}

    # Infomap takes no negative weight, so a signed weight is not one here.
    signed = "Signed" in dataset.tags

    def first(candidates: tuple[str, ...], *, numeric: bool = False) -> int | None:
        for candidate in candidates:
            if candidate in names[2:] and not (numeric and types.get(candidate) in _NOT_NUMERIC):
                return names.index(candidate)
        return None

    return Columns(
        weight=None if signed else first(_WEIGHT_NAMES, numeric=True),
        time=first(_TIME_NAMES),
        layer=first((*filter(None, [_LAYER_COLUMNS.get(dataset.name)]), *_LAYER_NAMES)),
        names=names,
        delimiter=",",
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


def node_names(network: Network) -> dict[Any, str]:
    """Each node's name from ``nodes.csv`` in the bundle :func:`load` fetched, or ``{}``.

    :meth:`Network.to_pajek` writes these as the ``*Vertices`` labels. The column is
    the one :meth:`NodeTable.labels` picks.
    """
    try:
        return node_table(network).labels()
    except LookupError:
        return {}


def is_cached(name: str, *, subgraph: str | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.files_for(subgraph)[0]
    return (_dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name).is_file()
