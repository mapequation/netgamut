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

from ._cache import (
    CACHE_ENV_VAR,
    CachedDataset,
    CacheInfo,
    cache_dir,
    cache_info,
    clear_cache,
    set_cache_dir,
)
from ._cache import dataset_dir as _dataset_dir
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._download import DownloadError
from ._download import download as _download_file
from ._models import Category, DataFile, Dataset
from ._network import Network
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

    target = _dataset_dir(dataset.name) / Path(artifact.name).name
    _download_file(
        artifact.url,
        target,
        force=force,
        progress=progress,
        max_bytes=max_bytes,
    )
    return Network(dataset=dataset, file=artifact, archive_path=target)


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
    target = _dataset_dir(dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return Network(dataset=dataset, file=artifact, archive_path=target)


def is_cached(name: str, *, file: str | None = None) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.file(file) if file else dataset.primary_file
    if artifact is None:
        return False
    return (_dataset_dir(dataset.name) / Path(artifact.name).name).is_file()
