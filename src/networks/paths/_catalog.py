"""Loading and querying the curated paths catalog.

Unlike the SNAP and Netzschleuder subpackages there is no site behind this
one: the collection is curated in this repository, so the vendored snapshot
*is* the catalog and :func:`refresh` has nothing to fetch. The loading
contract is kept identical anyway, so the top-level API and CLI treat all
sources alike.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from .._core.domains import DOMAINS
from ._models import Dataset

__all__ = ["Catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the curated path datasets."""

    datasets: tuple[Dataset, ...]
    source: str = "curated"
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
        raise KeyError(f"No path dataset named {name!r}.{hint}")

    def domains(self) -> dict[str, int]:
        """How many datasets fall in each normalised domain."""
        counts = {domain: 0 for domain in DOMAINS}
        for item in self.datasets:
            for domain in item.domains:
                counts[domain] += 1
        return counts

    def search(
        self,
        query: str = "",
        *,
        domain: str | None = None,
        directed: bool | None = None,
        max_nodes: int | None = None,
        min_nodes: int | None = None,
        max_edges: int | None = None,
        min_edges: int | None = None,
    ) -> tuple[Dataset, ...]:
        """Filter the catalog with the same shape the other sources use."""
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
            if needle and needle not in f"{item.name} {item.description}".casefold():
                continue
            if directed is not None and item.directed is not directed:
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
        return f"<Catalog {len(self.datasets)} path datasets from {self.origin}>"

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
            source=data.get("source", "curated"),
            retrieved=data.get("retrieved", ""),
            origin=origin,
        )


def load_catalog(*, reload: bool = False) -> Catalog:
    """Return the vendored catalog. Curated, so there is no cached override."""
    global _cached
    if _cached is not None and not reload:
        return _cached

    source = resources.files(__package__).joinpath(_VENDORED_NAME)
    with resources.as_file(source) as path, path.open(encoding="utf-8") as handle:
        _cached = Catalog.from_dict(json.load(handle), origin="package")
    return _cached


def refresh() -> Catalog:
    """Re-read the vendored catalog.

    The collection is curated in the package itself, so unlike the other
    sources there is no live site to consult; new datasets arrive with new
    releases. This exists to honour the shared source contract.
    """
    return load_catalog(reload=True)
