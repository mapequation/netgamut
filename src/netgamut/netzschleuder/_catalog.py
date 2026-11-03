"""Loading, querying and refreshing the Netzschleuder catalog.

Same contract as the SNAP subpackage: a snapshot ships inside the package so
browsing is offline and instant, :func:`refresh` re-reads the live site into the
user cache, and a refreshed copy takes precedence.

The build step here talks to a JSON API rather than parsing HTML, so it is
markedly less fragile than the SNAP scraper.
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

__all__ = ["Catalog", "build_catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
API_URL = f"{SITE_URL}/api"
_REFRESH_WORKERS = 8
# Enough to list nearly every entry in full. The cap is for the few with
# thousands of subgraphs, such as openstreetmap's ~91,000.
_SUBGRAPH_PREVIEW = 100

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the Netzschleuder collection."""

    datasets: tuple[Dataset, ...]
    source: str = API_URL
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
        raise KeyError(f"No Netzschleuder network named {name!r}.{hint}")

    def domains(self) -> dict[str, int]:
        """How many networks fall in each normalised domain."""
        counts = {domain: 0 for domain in DOMAINS}
        for item in self.datasets:
            for domain in item.domains:
                counts[domain] += 1
        return counts

    def tags(self) -> dict[str, int]:
        """Every tag the site uses, with counts."""
        counts: dict[str, int] = {}
        for item in self.datasets:
            for tag in item.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda pair: -pair[1]))

    def search(
        self,
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
        """Filter the catalog, including by domain and by structural property."""
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
            if subtype is not None and not any(
                subtype.casefold() == tag.casefold() for tag in item.subtypes
            ):
                continue
            if needle and needle not in f"{item.name} {item.description}".casefold():
                continue
            if directed is not None and item.directed is not directed:
                continue
            if weighted is not None and item.weighted is not weighted:
                continue
            if multilayer is not None and item.multilayer is not multilayer:
                continue
            if bipartite is not None and item.bipartite is not bipartite:
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
        return f"<Catalog {len(self.datasets)} Netzschleuder networks from {self.origin}>"

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
            source=data.get("source", API_URL),
            retrieved=data.get("retrieved", ""),
            origin=origin,
        )

    def write(self, path: Path) -> Path:
        _cache.write_atomic(path, json.dumps(self.to_dict(), indent=1).encode("utf-8"))
        return path


def _refreshed_path() -> Path:
    return _cache.source_dir("netzschleuder") / _VENDORED_NAME


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


def _first_analysis(analyses: Any) -> dict[str, Any]:
    """Pick the statistics block to report.

    Most entries have one; those holding several subgraphs nest one block per
    subgraph, in which case the first stands in for the entry.
    """
    if not isinstance(analyses, dict):
        return {}
    if "num_vertices" in analyses:
        return analyses
    for value in analyses.values():
        if isinstance(value, dict) and "num_vertices" in value:
            return value
    return {}


def build_catalog(*, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Read the whole collection from the Netzschleuder JSON API."""
    names = json.loads(fetch_text(f"{API_URL}/nets"))

    def fetch(name: str) -> Dataset | None:
        try:
            payload = json.loads(fetch_text(f"{API_URL}/net/{name}"))
        except Exception:
            # One unreachable entry must not sink the whole refresh.
            return None
        analyses = payload.get("analyses") or {}
        nets = tuple(payload.get("nets") or ())
        # Entries drawn from several publishers list one URL each; keep the first.
        url = payload.get("url") or ""
        if isinstance(url, list):
            url = url[0] if url else ""
        return Dataset(
            name=name,
            tags=tuple(payload.get("tags") or ()),
            description=(payload.get("description") or "").strip(),
            analyses=_first_analysis(analyses),
            subgraph_count=len(nets),
            # A preview only; see Dataset.subgraphs for why.
            subgraphs=nets[:_SUBGRAPH_PREVIEW],
            citations=tuple(payload.get("citation") or ()),
            upstream_url=url,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fetched = [item for item in pool.map(fetch, names) if item is not None]

    return Catalog(
        datasets=tuple(fetched),
        source=API_URL,
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        origin="cache",
    )


def refresh(*, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Re-read the live API and store the result in the user cache."""
    global _cached
    catalog = build_catalog(workers=workers)
    catalog.write(_refreshed_path())
    _cached = catalog
    return catalog
