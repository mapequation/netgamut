"""AHORN: the higher-order datasets the other four sources cannot express.

The Aachen Higher-Order Repository of Networks curates datasets whose unit of
observation is a *set* of nodes rather than a pair: hypergraphs, simplicial
complexes, one combinatorial complex. Three authors on one paper is one
interaction here, not three edges, and no edge list can say which of the two it
was. That is the gap this source fills -- SNAP, Netzschleuder, OGB and PyG all
publish dyadic graphs, and AHORN's own survey files them under "higher-order
structure only implicit"::

    from networks import ahorn

    ahorn.search(network_type="simplicial-complex", max_nodes=1000)

    h = ahorn.load("algebra-questions")
    h.metadata                      # {'name': ..., '_format-version': '0.2'}
    for members, attributes in h.interactions(cast=int):
        ...                         # ((2, 12, 20), {'category': 'lie-algebras'})

    im = h.to_infomap()             # bipartite, so interaction sizes survive
    im.run()

:func:`load` returns a :class:`~networks._core.hypergraph.Hypergraph`, not a
``Network``: flattening is a choice you make with ``to_networkx(projection=...)``
or ``to_network(...)``, not something this does behind your back.

**Revisions are pinnable, and only here.** AHORN publishes each dataset as
versioned Zenodo records, so ``load(name, revision=1)`` keeps returning the same
bytes after a correction lands. Without it you get the latest, which is
convenient and not reproducible -- ``algebra-questions`` revision 2 renamed an
attribute its revision 1 called ``genre``.

Citation is deliberately two-part. Cite the original source publication -- the
BibTeX in ``Dataset.citations`` -- for the data, and the AHORN paper for the
repository; ``ahorn.CITATION`` carries the latter. Licences are the upstream
source's, recorded on the record and granting nothing beyond what that source
already allowed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._core.cache import dataset_dir as _dataset_dir
from .._core.download import download as _download_file
from .._core.hypergraph import Hypergraph
from .._core.nodes import NodeTable
from ._catalog import ARCHIVE_URL, INDEX_URL, Catalog, build_catalog, load_catalog, refresh
from ._models import CITATION, NETWORK_TYPES, REPO, SITE_URL, Dataset, Revision

__all__ = [
    "ARCHIVE_URL",
    "CITATION",
    "INDEX_URL",
    "NETWORK_TYPES",
    "REPO",
    "SITE_URL",
    "Catalog",
    "Dataset",
    "Hypergraph",
    "Revision",
    "build_catalog",
    "catalog",
    "datasets",
    "describe",
    "info",
    "is_cached",
    "load",
    "node_table",
    "refresh",
    "search",
]

SOURCE = "ahorn"


def catalog(*, reload: bool = False) -> Catalog:
    """The AHORN catalog, read from the vendored snapshot or a refreshed copy."""
    return load_catalog(reload=reload)


def datasets(network_type: str | None = None) -> tuple[Dataset, ...]:
    """Every catalogued dataset, optionally restricted to one formalism.

    The formalisms overlap: most hypergraphs also admit a simplicial reading, so
    a dataset can appear under both.
    """
    items = catalog().datasets
    if network_type is None:
        return items
    return tuple(item for item in items if network_type in item.network_types)


def info(name: str) -> Dataset:
    """Metadata for one dataset, without downloading anything."""
    return catalog().get(name)


def search(
    query: str | None = None,
    *,
    domain: str | None = None,
    network_type: str | None = None,
    tag: str | None = None,
    temporal: bool | None = None,
    parents_only: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_interactions: int | None = None,
    min_interactions: int | None = None,
    max_interaction_size: int | None = None,
) -> tuple[Dataset, ...]:
    """Filter the catalog. *query* matches the slug, title and description.

    *tag* matches AHORN's own namespaced tags as substrings, so ``tag="benson"``
    finds everything derived from the Benson collection and ``tag="node classes"``
    everything carrying a node labelling.

    *parents_only* drops the derived children -- the 20 cuisines split out of
    ``cooking``, the genre splits of the review datasets -- which otherwise
    dominate any unfiltered listing. *max_interaction_size* is the guard worth
    setting before a clique projection: one interaction of *k* members becomes
    *k(k-1)/2* edges.
    """
    results = []
    for dataset in catalog().datasets:
        if query:
            haystack = f"{dataset.name} {dataset.title} {dataset.description}".casefold()
            if query.casefold() not in haystack:
                continue
        if domain and domain not in dataset.domains:
            continue
        if network_type and network_type not in dataset.network_types:
            continue
        if tag and not any(tag.casefold() in item.casefold() for item in dataset.tags):
            continue
        if temporal is not None and dataset.temporal is not temporal:
            continue
        if parents_only and dataset.parent:
            continue
        if max_nodes is not None and (dataset.nodes is None or dataset.nodes > max_nodes):
            continue
        if min_nodes is not None and (dataset.nodes is None or dataset.nodes < min_nodes):
            continue
        if max_interactions is not None and (
            dataset.interactions is None or dataset.interactions > max_interactions
        ):
            continue
        if min_interactions is not None and (
            dataset.interactions is None or dataset.interactions < min_interactions
        ):
            continue
        if max_interaction_size is not None and (
            dataset.max_interaction_size is None
            or dataset.max_interaction_size > max_interaction_size
        ):
            continue
        results.append(dataset)
    return tuple(results)


def describe(name: str) -> str:
    """A readable metadata summary for one dataset, for printing."""
    dataset = info(name)
    lines = [f"{dataset.title or dataset.name}  [{', '.join(dataset.domains) or 'uncategorised'}]"]
    if dataset.description:
        lines.append(f"  {dataset.description}")
    lines.append(f"  Slug        {dataset.name}")
    lines.append(f"  Read as     {', '.join(dataset.network_types) or 'unspecified'}")
    lines.append(f"  Page        {dataset.url}")
    if dataset.nodes is not None:
        size = f"{dataset.nodes:,} nodes"
        if dataset.interactions is not None:
            size += f", {dataset.interactions:,} interactions"
        lines.append(f"  Size        {size}")
    if dataset.max_interaction_size is not None:
        mean = dataset.mean_interaction_size
        shape = f"max {dataset.max_interaction_size}"
        if mean is not None:
            shape = f"mean {mean:.1f}, " + shape
        lines.append(f"  Sizes       {shape} nodes per interaction")
    if dataset.parent:
        lines.append(f"  Derived from {dataset.parent}")
    if dataset.tags:
        lines.append(f"  Tags        {', '.join(dataset.tags)}")

    if dataset.revisions:
        lines.append("")
        lines.append("Revisions")
        for revision in dataset.revisions:
            marker = "*" if revision == dataset.latest else " "
            formats = "ahorn" + (", hif" if revision.hif_url else "")
            lines.append(f"  {marker} {revision.number}  ({formats})")
            lines.extend(f"      - {entry}" for entry in revision.changelog)
        lines.append("  (* fetched by default)")

    if dataset.license or dataset.source_url:
        lines.append("")
        if dataset.license:
            lines.append(f"Licence     {dataset.license}")
        if dataset.source_url:
            lines.append(f"Upstream    {dataset.source_url}")
    if dataset.citations:
        lines.append("")
        lines.append("Cite the source")
        # BibTeX verbatim, indented: this is meant to be copied, and AHORN's
        # point is that the dataset citation is the upstream paper's, not its own.
        for entry in dataset.citations:
            lines.extend(f"  {line}" for line in entry.strip().splitlines())
        lines.append("")
        lines.append("Cite the repository")
        lines.append(f"  {CITATION}")
    return "\n".join(lines)


def load(
    name: str,
    *,
    revision: int | None = None,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Hypergraph:
    """Download *name* if it is not already cached, and return a handle to it.

    Without *revision* you get the latest published one, which is what the
    website links. Pass a number to pin it: revisions are separate Zenodo
    records, so a pinned call keeps returning the same bytes and a benchmark
    stays reproducible across a correction.

    Revisions are cached side by side under the dataset's directory, so pinning
    one does not evict another.

    Raises:
        KeyError: No such dataset.
        LookupError: No such revision, or the dataset publishes none.
        DownloadError: The transfer failed or exceeded *max_bytes*.
    """
    dataset = info(name)
    chosen = dataset.revision(revision)
    artifact = chosen.file

    target = _dataset_dir(SOURCE, dataset.name) / f"r{chosen.number}-{artifact.name}"
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return Hypergraph(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        source=SOURCE,
        revision=chosen.number,
    )


def node_table(handle: Hypergraph) -> NodeTable:
    """Read an AHORN dataset's per-node metadata into the shared shape.

    Node attributes live on the explicit node lines of the dataset file, so this
    reads the file already downloaded and needs nothing else. Most datasets
    declare no nodes at all -- their nodes are implied by the interactions --
    and those raise rather than returning an empty table.

    Interaction attributes are not node attributes and are not here; stream them
    with ``Hypergraph.interactions()``, or take one column of them with
    ``Hypergraph.interaction_labels(...)``.

    Raises:
        LookupError: The dataset declares no nodes explicitly.
    """
    values: dict[str, dict[Any, Any]] = {}
    for node, attributes in handle.node_records():
        for column, value in attributes.items():
            values.setdefault(column, {})[node] = value

    if not values:
        raise LookupError(
            f"{handle.name!r} declares no nodes with attributes; its nodes are implied "
            "by the interactions. Interaction attributes are available through "
            ".interactions() and .interaction_labels()."
        )
    return NodeTable(values=values, origin=f"{SOURCE}/{handle.name}")


def is_cached(name: str, *, revision: int | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
        chosen = dataset.revision(revision)
    except LookupError:  # covers KeyError
        return False
    return (_dataset_dir(SOURCE, dataset.name) / f"r{chosen.number}-{chosen.file.name}").is_file()
