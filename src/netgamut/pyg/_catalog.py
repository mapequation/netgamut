"""Loading and refreshing the PyTorch Geometric catalog.

Same contract as the other sources: a snapshot ships inside the package so browsing is
offline and instant, :func:`refresh` rebuilds it from PyG's source, and a refreshed
copy takes precedence.

The build reads ``torch_geometric/datasets/*.py`` straight from GitHub and parses the
statistics tables in the class docstrings (see :mod:`._scrape`). It needs neither
``torch`` nor ``torch_geometric`` installed -- browsing PyG's collection costs nothing
until you actually load something.
"""

from __future__ import annotations

import difflib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .._core import cache as _cache
from . import _scrape
from ._models import DOMAIN_MAP, SKIP, Dataset

__all__ = ["Catalog", "build_catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
SOURCE = "pyg"
_WORKERS = 8

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the PyG collection."""

    datasets: tuple[Dataset, ...]
    source: str = _scrape.REPOSITORY
    retrieved: str = ""
    #: Classes seen but not catalogued, and why -- collections, missing statistics,
    #: deliberate skips. Kept so the gap is visible rather than silent.
    omitted: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.omitted is None:
            object.__setattr__(self, "omitted", {})
        object.__setattr__(self, "_index", {d.name.lower(): d for d in self.datasets})

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._resolve(name) is not None

    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self) -> Any:
        return iter(self.datasets)

    def _resolve(self, name: str) -> Dataset | None:
        index: dict[str, Dataset] = self._index  # type: ignore[attr-defined]
        key = name.lower().removeprefix("pyg/")
        if key in index:
            return index[key]
        # A bare dataset name is enough when only one class offers it.
        matches = [d for d in self.datasets if d.dataset.lower() == key or d.cls.lower() == key]
        return matches[0] if len(matches) == 1 else None

    def get(self, name: str) -> Dataset:
        """Look up by ``Class/name``, or by a bare name when it is unambiguous."""
        found = self._resolve(name)
        if found is not None:
            return found
        index: dict[str, Dataset] = self._index  # type: ignore[attr-defined]
        key = name.lower().removeprefix("pyg/")
        ambiguous = [d.name for d in self.datasets if d.dataset.lower() == key]
        if ambiguous:
            raise KeyError(
                f"{name!r} is offered by several PyG classes ({', '.join(ambiguous)}). "
                f"Use the full Class/name."
            )
        close = difflib.get_close_matches(key, index, n=3)
        hint = f" Did you mean: {', '.join(index[c].name for c in close)}?" if close else ""
        raise KeyError(f"No PyG dataset named {name!r}.{hint}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "retrieved": self.retrieved,
            "omitted": self.omitted,
            "datasets": [d.to_dict() for d in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalog:
        return cls(
            datasets=tuple(Dataset.from_dict(item) for item in data.get("datasets", ())),
            source=data.get("source", _scrape.REPOSITORY),
            retrieved=data.get("retrieved", ""),
            omitted=data.get("omitted", {}),
        )


def _fetch(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "netgamut"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload: bytes = response.read()
    return payload.decode("utf-8", errors="replace")


def build_catalog(*, timeout: float = 60.0) -> Catalog:
    """Parse PyG's dataset classes into a catalog.

    Classes with a statistics table are catalogued from it. Classes without one are
    catalogued only when :data:`~netgamut.pyg._models.DOMAIN_MAP` vouches for them,
    using the dataset names in their ``name (str):`` argument documentation; their
    sizes stay unknown. Everything else is recorded in :attr:`Catalog.omitted`.
    """
    base = f"{_scrape.RAW}/{_scrape.DATASETS_PATH}"
    init = _fetch(f"{base}/__init__.py", timeout)
    classes = _scrape.homo_classes(init)
    modules = _scrape.module_of(init)

    def source_of(cls: str) -> tuple[str, str]:
        module = modules.get(cls)
        if not module:
            return cls, ""
        try:
            return cls, _fetch(f"{base}/{module}.py", timeout)
        except OSError:  # pragma: no cover - transient network failure
            return cls, ""

    with ThreadPoolExecutor(_WORKERS) as pool:
        sources = dict(pool.map(source_of, classes))

    datasets: list[Dataset] = []
    omitted: dict[str, str] = {}
    for cls in classes:
        if cls in SKIP:
            omitted[cls] = SKIP[cls]
            continue
        source = sources.get(cls, "")
        if not source:
            omitted[cls] = "source could not be read"
            continue

        entries, collection = _scrape.parse_stats(cls, source)
        if collection and not entries:
            omitted[cls] = "a collection of many graphs, which one Network cannot hold"
            continue

        if entries:
            for entry in entries:
                named = entry.name != cls
                datasets.append(
                    Dataset(
                        name=f"{cls}/{entry.name}" if named else cls,
                        cls=cls,
                        dataset=entry.name if named else "",
                        nodes=entry.nodes,
                        edges=entry.edges,
                        features=entry.features,
                        num_classes=entry.classes,
                        domains=DOMAIN_MAP.get(f"{cls}/{entry.name}", DOMAIN_MAP.get(cls, ())),
                    )
                )
            continue

        if cls not in DOMAIN_MAP:
            omitted[cls] = "no statistics table, and not on the allowlist"
            continue

        names = _scrape.parse_names(cls, source)
        for name in names or [""]:
            datasets.append(
                Dataset(
                    name=f"{cls}/{name}" if name else cls,
                    cls=cls,
                    dataset=name,
                    domains=DOMAIN_MAP.get(f"{cls}/{name}", DOMAIN_MAP.get(cls, ())),
                )
            )

    return Catalog(
        datasets=tuple(sorted(datasets, key=lambda d: d.name)),
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        omitted=dict(sorted(omitted.items())),
    )


def _cache_path() -> Path:
    return _cache.source_dir(SOURCE) / _VENDORED_NAME


def load_catalog(*, reload: bool = False) -> Catalog:
    """Return the catalog, preferring a refreshed copy in the cache."""
    global _cached
    if _cached is not None and not reload:
        return _cached

    refreshed = _cache_path()
    if refreshed.is_file():
        try:
            _cached = Catalog.from_dict(json.loads(refreshed.read_text()))
            return _cached
        except (OSError, ValueError, KeyError):  # pragma: no cover - corrupt cache
            pass

    vendored = resources.files(__package__).joinpath(_VENDORED_NAME).read_text()
    _cached = Catalog.from_dict(json.loads(vendored))
    return _cached


def refresh(*, timeout: float = 60.0) -> Catalog:
    """Rebuild from PyG's source into the user cache and return the new catalog."""
    global _cached
    catalog = build_catalog(timeout=timeout)
    target = _cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _cache.write_atomic(target, json.dumps(catalog.to_dict(), indent=2).encode())
    _cached = catalog
    return catalog
