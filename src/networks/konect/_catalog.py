"""Loading, querying and refreshing the KONECT catalog.

Same contract as the other sources: a snapshot ships inside the package so
browsing is offline and instant, :func:`refresh` re-reads the live site into the
user cache, and a refreshed copy takes precedence.

KONECT has no API, so the build reads the network index and then each network's
page; see :mod:`networks.konect._scrape` for what is parsed and why.
"""

from __future__ import annotations

import difflib
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .._core import cache as _cache
from .._core.domains import DOMAINS
from .._core.download import fetch_text
from ._models import SITE_URL, Dataset
from ._scrape import is_series, parse_index, parse_page

__all__ = ["Catalog", "build_catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
INDEX_URL = f"{SITE_URL}/networks/"
# The site is one person's server; keep the build gentle.
_REFRESH_WORKERS = 4

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the KONECT collection."""

    datasets: tuple[Dataset, ...]
    source: str = INDEX_URL
    retrieved: str = ""
    origin: str = "package"

    _by_name: dict[str, Dataset] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._by_name.update({item.name.casefold(): item for item in self.datasets})

    # -- lookup ----------------------------------------------------------
    def dataset(self, name: str) -> Dataset:
        found = self._by_name.get(name.casefold())
        if found is not None:
            return found
        suggestions = difflib.get_close_matches(
            name, [item.name for item in self.datasets], n=3, cutoff=0.6
        )
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"No KONECT network named {name!r}.{hint}")

    def domains(self) -> dict[str, int]:
        """How many networks fall in each normalised domain."""
        counts = dict.fromkeys(DOMAINS, 0)
        for item in self.datasets:
            for domain in item.domains:
                counts[domain] += 1
        return counts

    def categories(self) -> dict[str, int]:
        """KONECT's own categories, with counts, most common first."""
        counts: dict[str, int] = {}
        for item in self.datasets:
            counts[item.category] = counts.get(item.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: -pair[1]))

    def search(
        self,
        query: str = "",
        *,
        domain: str | None = None,
        category: str | None = None,
        directed: bool | None = None,
        bipartite: bool | None = None,
        weighted: bool | None = None,
        signed: bool | None = None,
        temporal: bool | None = None,
        loops: bool | None = None,
        max_nodes: int | None = None,
        min_nodes: int | None = None,
        max_edges: int | None = None,
        min_edges: int | None = None,
    ) -> tuple[Dataset, ...]:
        """Filter the catalog by domain, category, size or structural property.

        A ``loops`` filter excludes networks whose loop status KONECT does not
        report rather than assuming either answer.
        """
        needle = query.casefold().strip()
        results: list[Dataset] = []

        def within(value: int | None, low: int | None, high: int | None) -> bool:
            if low is None and high is None:
                return True
            if value is None:
                return False
            return (low is None or value >= low) and (high is None or value <= high)

        for item in self.datasets:
            if domain is not None and domain not in item.domains:
                continue
            if category is not None and category.casefold() != item.category.casefold():
                continue
            text = f"{item.name} {item.title} {item.description}".casefold()
            if needle and needle not in text:
                continue
            if directed is not None and item.directed is not directed:
                continue
            if bipartite is not None and item.bipartite is not bipartite:
                continue
            if weighted is not None and item.weighted is not weighted:
                continue
            if signed is not None and item.signed is not signed:
                continue
            if temporal is not None and item.temporal is not temporal:
                continue
            if loops is not None and item.loops is not loops:
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
        return f"<Catalog {len(self.datasets)} KONECT networks from {self.origin}>"

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "retrieved": self.retrieved,
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
            datasets=tuple(Dataset.from_dict(item) for item in data.get("datasets", [])),
            source=data.get("source", INDEX_URL),
            retrieved=data.get("retrieved", ""),
            origin=origin,
        )

    def write(self, path: Path) -> Path:
        _cache.write_atomic(path, json.dumps(self.to_dict(), indent=1).encode("utf-8"))
        return path


def _refreshed_path() -> Path:
    return _cache.source_dir("konect") / _VENDORED_NAME


def load_catalog(*, reload: bool = False) -> Catalog:
    """Return the catalog, preferring a refreshed copy in the cache."""
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
            pass

    source = resources.files(__package__).joinpath(_VENDORED_NAME)
    with resources.as_file(source) as path, path.open(encoding="utf-8") as handle:
        _cached = Catalog.from_dict(json.load(handle), origin="package")
    return _cached


def build_catalog(*, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Read the downloadable, non-series part of KONECT from the live site."""
    rows = [
        row
        for row in parse_index(fetch_text(INDEX_URL))
        if row["available"] and not is_series(row["name"])
    ]

    def fetch(row: dict[str, Any]) -> Dataset | None:
        try:
            page = parse_page(fetch_text(f"{INDEX_URL}{row['name']}/"))
        except Exception:
            # One unreachable page must not sink the whole refresh.
            return None
        return Dataset(
            name=row["name"],
            title=row["title"],
            nodes=row["nodes"],
            edges=row["edges"],
            **{**page, "temporal": page["temporal"] or row["temporal"]},
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fetched = [item for item in pool.map(fetch, rows) if item is not None]

    return Catalog(
        datasets=tuple(fetched),
        source=INDEX_URL,
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        origin="cache",
    )


def refresh(*, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Re-read the live site and store the result in the user cache."""
    global _cached
    catalog = build_catalog(workers=workers)
    catalog.write(_refreshed_path())
    _cached = catalog
    return catalog
