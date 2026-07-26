"""Loading, querying and refreshing the SNAP catalog.

Metadata is answered from a snapshot that ships inside the package, so listing
categories, comparing dataset sizes and reading statistics all work offline and
instantly -- no network call happens until you actually ask for data.

:func:`refresh` re-scrapes snap.stanford.edu and writes an updated catalog into
the user cache, which then takes precedence over the vendored copy.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from . import _cache
from ._download import fetch_text
from ._models import Category, Dataset
from ._scrape import INDEX_URL, SNAP_BASE_URL, parse_dataset_page, parse_index

__all__ = ["Catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"

# Detail pages are small; a modest pool keeps a full refresh to a few seconds
# without hammering Stanford's server.
_REFRESH_WORKERS = 8

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the SNAP collection."""

    categories: tuple[Category, ...]
    datasets: tuple[Dataset, ...]
    source: str = INDEX_URL
    retrieved: str = ""
    #: Where this catalog came from: ``"package"`` or ``"cache"``.
    origin: str = "package"

    # Lookup indexes, built once in __post_init__.
    _by_name: dict[str, Dataset] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    _by_category: dict[str, Category] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._by_name.update({item.name.casefold(): item for item in self.datasets})
        for item in self.categories:
            # A category is reachable by anchor id, section heading, or the
            # label SNAP uses for it in the table of contents.
            self._by_category[item.id.casefold()] = item
            self._by_category[item.name.casefold()] = item
            if item.toc_name:
                self._by_category.setdefault(item.toc_name.casefold(), item)

    # -- lookup ----------------------------------------------------------
    def dataset(self, name: str) -> Dataset:
        """Return the dataset called *name*, case-insensitively."""
        found = self._by_name.get(name.casefold())
        if found is not None:
            return found
        suggestions = difflib.get_close_matches(
            name, [item.name for item in self.datasets], n=3, cutoff=0.6
        )
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"No SNAP dataset named {name!r}.{hint}")

    def category(self, name: str) -> Category:
        """Return a category by its id, section heading, or contents label."""
        found = self._by_category.get(name.casefold())
        if found is not None:
            return found
        options = sorted({item.name for item in self.categories})
        suggestions = difflib.get_close_matches(name, options, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"No SNAP category named {name!r}.{hint}")

    def datasets_in(self, category: str) -> tuple[Dataset, ...]:
        """Every dataset filed under *category*.

        Cross-listed datasets appear under each of their categories.
        """
        target = self.category(category)
        return tuple(item for item in self.datasets if target.id in item.categories)

    def search(
        self,
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
        """Filter the catalog.

        *query* matches case-insensitively against the dataset name and
        description. The remaining arguments narrow by category, orientation
        and size. Datasets that do not report a size are excluded whenever a
        size bound is given. Set *ground_truth* to select only datasets that
        ship a reference partition, which is what you want when assembling a
        community-detection benchmark.
        """
        needle = query.casefold().strip()
        target = self.category(category).id if category else None
        results: list[Dataset] = []

        def within(value: int | None, low: int | None, high: int | None) -> bool:
            if low is None and high is None:
                return True
            # A bound was requested but SNAP reports no count for this dataset,
            # so we cannot claim it matches.
            if value is None:
                return False
            return (low is None or value >= low) and (high is None or value <= high)

        for item in self.datasets:
            if target is not None and target not in item.categories:
                continue
            if needle and needle not in f"{item.name} {item.description}".casefold():
                continue
            if directed is not None and item.directed is not directed:
                continue
            if ground_truth is not None and item.has_ground_truth is not ground_truth:
                continue
            if not within(item.nodes, min_nodes, max_nodes):
                continue
            if not within(item.edges, min_edges, max_edges):
                continue
            results.append(item)

        return tuple(results)

    # -- container protocol ---------------------------------------------
    def __iter__(self) -> Iterator[Dataset]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.casefold() in self._by_name

    def __getitem__(self, name: str) -> Dataset:
        return self.dataset(name)

    def __repr__(self) -> str:
        return (
            f"<Catalog {len(self.datasets)} datasets in {len(self.categories)} "
            f"categories from {self.origin}>"
        )

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "retrieved": self.retrieved,
            "categories": [item.to_dict() for item in self.categories],
            "datasets": [item.to_dict() for item in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, origin: str = "package") -> Catalog:
        schema = data.get("schema")
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported catalog schema {schema!r}; this version expects {SCHEMA_VERSION}."
            )
        return cls(
            categories=tuple(Category.from_dict(item) for item in data.get("categories", [])),
            datasets=tuple(Dataset.from_dict(item) for item in data.get("datasets", [])),
            source=data.get("source", INDEX_URL),
            retrieved=data.get("retrieved", ""),
            origin=origin,
        )

    def write(self, path: Path) -> Path:
        _cache.write_atomic(path, json.dumps(self.to_dict(), indent=1).encode("utf-8"))
        return path


def _refreshed_path() -> Path:
    return _cache.snap_dir() / _VENDORED_NAME


def _read_vendored() -> Catalog:
    source = resources.files(__package__).joinpath(_VENDORED_NAME)
    with resources.as_file(source) as path, path.open(encoding="utf-8") as handle:
        return Catalog.from_dict(json.load(handle), origin="package")


def load_catalog(*, reload: bool = False) -> Catalog:
    """Return the catalog, preferring a refreshed copy in the cache.

    The result is memoised for the life of the process. Pass ``reload=True``
    to re-read from disk.
    """
    global _cached
    if _cached is not None and not reload:
        return _cached

    refreshed = _refreshed_path()
    if refreshed.is_file():
        try:
            with refreshed.open(encoding="utf-8") as handle:
                _cached = Catalog.from_dict(json.load(handle), origin="cache")
                return _cached
        except (OSError, ValueError, json.JSONDecodeError):
            # A corrupt or outdated cached catalog must never break the
            # package; fall back to the copy we shipped.
            pass

    _cached = _read_vendored()
    return _cached


def build_catalog(
    *,
    details: bool = True,
    workers: int = _REFRESH_WORKERS,
    on_progress: Any = None,
    only: Sequence[str] | None = None,
) -> Catalog:
    """Scrape snap.stanford.edu and return a freshly built catalog.

    With *details* the per-dataset pages are fetched too, which is what
    supplies the statistics tables and download URLs. *only* restricts the
    detail pass to the named datasets, which keeps test runs cheap.
    """
    import time

    index_html = fetch_text(INDEX_URL)
    categories, datasets = parse_index(index_html, SNAP_BASE_URL)

    if details and datasets:
        wanted = {name.casefold() for name in only} if only else None
        targets = [item for item in datasets if wanted is None or item.name.casefold() in wanted]

        def enrich(item: Dataset) -> Dataset:
            try:
                extra = parse_dataset_page(fetch_text(item.url), item.url)
            except Exception:
                # One unreachable detail page must not sink the whole refresh;
                # the dataset simply keeps its index-table metadata.
                return item
            return Dataset(
                name=item.name,
                categories=item.categories,
                category_names=item.category_names,
                description=item.description,
                type=item.type,
                url=item.url,
                columns=item.columns,
                statistics=extra.statistics,
                files=extra.files,
                citations=extra.citations,
            )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            enriched = list(pool.map(enrich, targets))

        by_name = {item.name: item for item in enriched}
        datasets = [by_name.get(item.name, item) for item in datasets]

        if on_progress is not None:
            on_progress(len(enriched), len(targets))

    return Catalog(
        categories=tuple(categories),
        datasets=tuple(datasets),
        source=INDEX_URL,
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        origin="cache",
    )


def refresh(*, details: bool = True, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Re-scrape SNAP and store the result in the user cache.

    The refreshed catalog takes precedence over the one bundled with the
    package, so newly published SNAP datasets become available without waiting
    for a release. Requires network access.
    """
    global _cached
    catalog = build_catalog(details=details, workers=workers)
    catalog.write(_refreshed_path())
    _cached = catalog
    return catalog
