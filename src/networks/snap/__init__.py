"""Access to the Stanford Large Network Dataset Collection (SNAP).

Metadata is answered from a catalog bundled with the package, so you can browse
the whole collection -- categories, sizes, statistics, citations -- without
touching the network::

    from networks import snap

    for category in snap.categories():
        print(category)

    print(snap.describe("ca-GrQc"))

Only :func:`load` downloads anything, and it caches into a shared user-level
directory, so a dataset is fetched once per machine no matter how many
projects, notebooks or virtualenvs use it::

    net = snap.load("ca-GrQc")
    net.path                    # a plain text file on disk
    sum(1 for _ in net.edges()) # streamed, never fully in memory

SNAP is a third-party collection. Please cite the datasets you use; see
``Dataset.citations``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._core.cache import (
    CACHE_ENV_VAR,
    CachedDataset,
    CacheInfo,
    cache_dir,
    cache_info,
    clear_cache,
    set_cache_dir,
)
from .._core.cache import dataset_dir as _dataset_dir
from .._core.columns import Columns
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.network import Network
from .._core.nodes import NodeTable
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._models import Category, DataFile, Dataset, label_column
from ._scrape import INDEX_URL, SNAP_BASE_URL

__all__ = [
    "CACHE_ENV_VAR",
    "INDEX_URL",
    "SNAP_BASE_URL",
    "CacheInfo",
    "CachedDataset",
    "Catalog",
    "Category",
    "DataFile",
    "Dataset",
    "DownloadError",
    "Network",
    "build_catalog",
    "cache_dir",
    "cache_info",
    "catalog",
    "categories",
    "clear_cache",
    "datasets",
    "describe",
    "info",
    "is_cached",
    "load",
    "load_ground_truth",
    "node_table",
    "refresh",
    "search",
    "set_cache_dir",
]


def catalog(*, reload: bool = False) -> Catalog:
    """The SNAP catalog, read from the cache if refreshed, else from the package."""
    return load_catalog(reload=reload)


def categories() -> tuple[Category, ...]:
    """Every category, in the order SNAP lists them on its index page."""
    return catalog().categories


def datasets(category: str | None = None) -> tuple[Dataset, ...]:
    """Every dataset, or just those in *category*."""
    current = catalog()
    return current.datasets_in(category) if category else current.datasets


def info(name: str) -> Dataset:
    """Metadata for one dataset. Never downloads anything."""
    return catalog().dataset(name)


def search(
    query: str = "",
    *,
    category: str | None = None,
    directed: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
    ground_truth: bool | None = None,
) -> tuple[Dataset, ...]:
    """Find datasets by name, description, category, orientation or size.

    For example, every undirected network small enough to load comfortably::

        snap.search(directed=False, max_edges=1_000_000)

    Or a graded benchmark set with reference partitions to score against::

        snap.search(ground_truth=True, max_edges=10_000_000)
    """
    return catalog().search(
        query,
        category=category,
        directed=directed,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        max_edges=max_edges,
        min_edges=min_edges,
        ground_truth=ground_truth,
    )


def describe(name: str) -> str:
    """A readable metadata summary for one dataset, for printing."""
    dataset = info(name)
    label = "Categories" if len(dataset.category_names) > 1 else "Category  "
    lines = [
        dataset.name,
        "=" * len(dataset.name),
        f"{label}  {', '.join(dataset.category_names)}",
    ]
    if dataset.description:
        lines.append(f"Summary     {dataset.description}")
    if dataset.type:
        lines.append(f"Type        {dataset.type}")
    if dataset.url:
        lines.append(f"Page        {dataset.url}")

    reported = {key: value for key, value in dataset.columns.items() if key != "Description"}
    if reported:
        lines.append("")
        lines.append("Reported by SNAP")
        for key, value in reported.items():
            lines.append(f"  {key:<28} {value}")

    if dataset.statistics:
        lines.append("")
        lines.append("Dataset statistics")
        for key, value in dataset.statistics.items():
            lines.append(f"  {key:<28} {value}")

    if dataset.files:
        lines.append("")
        lines.append("Files")
        primary = dataset.primary_file
        for item in dataset.files:
            marker = "*" if item is primary else " "
            lines.append(f" {marker} {item.name:<28} {item.description}")
        lines.append("  (* downloaded by default)")
        if dataset.has_ground_truth:
            lines.append("  Ground-truth communities available: snap.load_ground_truth(...)")

    if dataset.citations:
        lines.append("")
        lines.append("Source")
        for citation in dataset.citations:
            lines.append(f"  {citation}")

    return "\n".join(lines)


#: Column roles for SNAP files with more than two fields, by file name. Anything
#: not listed is read as a plain edge list: SNAP publishes no column metadata,
#: and a guessed third column is as often a timestamp or a sign as a weight.
_TIME = Columns(time=2, names=("source", "target", "time"))
_SIGN = Columns(names=("source", "target", "sign"))
_COLUMNS: dict[str, Columns] = {
    **{
        f"sx-{site}{variant}.txt.gz": _TIME
        for site in ("stackoverflow", "mathoverflow", "superuser", "askubuntu")
        for variant in ("", "-a2q", "-c2q", "-c2a")
    },
    "wiki-talk-temporal.txt.gz": _TIME,
    "CollegeMsg.txt.gz": _TIME,
    **{
        f"email-Eu-core-temporal{part}.txt.gz": _TIME
        for part in ("", "-Dept1", "-Dept2", "-Dept3", "-Dept4")
    },
    # Retweet, mention or reply: the layer of a multiplex.
    "higgs-activity_time.txt.gz": Columns(
        time=2, layer=3, names=("source", "target", "time", "interaction")
    ),
    **{
        f"{kind}_network.edgelist.gz": Columns(weight=2, names=("source", "target", "weight"))
        for kind in ("retweet", "reply", "mention")
    },
    # Ratings run from -10 to 10, so they are not Infomap weights.
    "soc-sign-bitcoinotc.csv.gz": Columns(time=3, names=("source", "target", "rating", "time")),
    "soc-sign-bitcoinalpha.csv.gz": Columns(time=3, names=("source", "target", "rating", "time")),
    "soc-sign-epinions.txt.gz": _SIGN,
    "soc-sign-Slashdot081106.txt.gz": _SIGN,
    "soc-sign-Slashdot090216.txt.gz": _SIGN,
    "soc-sign-Slashdot090221.txt.gz": _SIGN,
    # Business relationship codes (-1 customer, 0 peer, 1 provider, 2 sibling).
    "as-caida20071105.txt.gz": Columns(names=("source", "target", "relationship")),
}


def load(
    name: str,
    *,
    file: str | None = None,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download *name* if it is not already cached, and return a handle to it.

    The download goes to a shared user-level cache (see :func:`cache_dir`), so
    the second call -- in this project or any other -- is free.

    Args:
        name: Dataset name, case-insensitive, e.g. ``"ca-GrQc"``.
        file: Which artifact to fetch. Defaults to the dataset's main edge
            list; see ``info(name).files`` for the alternatives.
        force: Re-download even if a cached copy exists.
        progress: ``True`` for a terminal progress bar, ``False`` to silence
            it, or a callable receiving ``(downloaded, total)``. Defaults to a
            bar when stderr is a terminal.
        max_bytes: Refuse to download anything larger. Useful as a guard, since
            some SNAP datasets run to tens of gigabytes.

    Raises:
        KeyError: No such dataset, or no such file within it.
        LookupError: The dataset has no downloadable file listed.
        DownloadError: The transfer failed or exceeded *max_bytes*.
    """
    dataset = info(name)
    artifact = dataset.file(file) if file else dataset.primary_file

    if artifact is None:
        raise LookupError(
            f"{dataset.name!r} has no downloadable files. SNAP has withdrawn a "
            f"handful of datasets while keeping them listed; see {dataset.url}. "
            f"If SNAP has restored it since this catalog was built, "
            f"snap.refresh() will pick it up."
        )

    target = _dataset_dir("snap", dataset.name) / Path(artifact.name).name
    _download_file(
        artifact.url,
        target,
        force=force,
        progress=progress,
        max_bytes=max_bytes,
    )
    return Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        source="snap",
        columns=_COLUMNS.get(artifact.name, Columns()),
    )


def load_ground_truth(
    name: str,
    *,
    variant: str = "top5000",
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download a dataset's ground-truth communities and return a handle.

    Only the *Networks with ground-truth communities* datasets have these.
    *variant* is ``"top5000"`` for the 5000 highest-quality communities, which
    is the usual benchmark reference, or ``"all"`` for the full set::

        truth = snap.load_ground_truth("com-DBLP")
        reference = [set(c) for c in truth.communities()]

    Raises:
        LookupError: The dataset ships no ground-truth partition. Use
            ``snap.search(ground_truth=True)`` to see the ones that do.
    """
    dataset = info(name)
    artifact = dataset.community_file(variant)
    target = _dataset_dir("snap", dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        source="snap",
        columns=_COLUMNS.get(artifact.name, Columns()),
    )


def node_table(network: Network) -> NodeTable:
    """Read a SNAP dataset's per-node metadata into the shared shape.

    SNAP keeps node attributes in a *separate* artifact rather than beside the edge
    list, so this downloads it on the same cache-once terms as :func:`load`. Only
    ``email-Eu-core`` publishes one today -- each person's department -- which is why
    the column is named after the file rather than from a schema SNAP does not have.

    Overlapping ground-truth communities are a different thing and stay on
    :func:`load_ground_truth` and :meth:`Network.communities`: a cover is not a column.

    Raises:
        LookupError: This dataset ships no node-label artifact.
    """
    dataset = network.dataset if isinstance(network.dataset, Dataset) else info(network.name)
    artifact = next(
        (item for item in dataset.files if "label" in Path(item.name).name.lower()), None
    )
    if artifact is None:
        extra = (
            " It does publish ground-truth communities: snap.load_ground_truth(...)."
            if dataset.has_ground_truth
            else ""
        )
        raise LookupError(f"{dataset.name!r} ships no node-label file.{extra}")

    target = _dataset_dir("snap", dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, progress=False)
    handle = Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        source="snap",
        columns=_COLUMNS.get(artifact.name, Columns()),
    )

    # Node ids are cast so they match the integers the edge list uses; the attribute
    # value stays exactly as SNAP wrote it, and NodeTable orders it naturally.
    values: dict[Any, Any] = {}
    for fields in handle.rows():
        if len(fields) >= 2:
            node = int(fields[0]) if fields[0].lstrip("-").isdigit() else fields[0]
            values[node] = fields[1]

    column = label_column(dataset.name, artifact.name)
    return NodeTable(values={column: values}, origin=f"snap/{Path(artifact.name).name}")


def is_cached(name: str, *, file: str | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.file(file) if file else dataset.primary_file
    if artifact is None:
        return False
    return (_dataset_dir("snap", dataset.name) / Path(artifact.name).name).is_file()
