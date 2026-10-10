"""Loading and refreshing the OGB catalog.

Same contract as the other sources: a snapshot ships inside the package so browsing is
offline and instant, :func:`refresh` re-reads OGB's ``master.csv`` files into the user
cache, and a refreshed copy takes precedence over the vendored one.

The build step reads CSV that OGB maintains as the authoritative dataset registry, so
it is the least fragile of the three refreshers -- no HTML, no API contract, just the
file the ``ogb`` package itself reads at import.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .._core import cache as _cache
from .._core.download import fetch_text
from ._models import CITATION, DESCRIPTIONS, DOMAIN_MAP, Dataset

__all__ = ["Catalog", "FAMILIES", "MASTER_URL", "build_catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
SOURCE = "ogb"

#: The task families holding single networks. ``graphproppred`` is deliberately absent:
#: its 15 entries are collections of thousands of small molecule graphs, which do not
#: fit a one-graph Network and would be unloadable rows in the catalog.
FAMILIES = ("nodeproppred", "linkproppred")

MASTER_URL = "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/{family}/master.csv"

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the OGB collection."""

    datasets: tuple[Dataset, ...]
    source: str = "snap-stanford/ogb"
    retrieved: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {d.name.lower(): d for d in self.datasets})

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._index  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self) -> Any:
        return iter(self.datasets)

    def get(self, name: str) -> Dataset:
        """Look up by name, case-insensitively, suggesting near misses."""
        index: dict[str, Dataset] = self._index  # type: ignore[attr-defined]
        try:
            return index[name.lower()]
        except KeyError:
            close = difflib.get_close_matches(name.lower(), index, n=3)
            hint = f" Did you mean: {', '.join(index[c].name for c in close)}?" if close else ""
            raise KeyError(f"No OGB dataset named {name!r}.{hint}") from None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "retrieved": self.retrieved,
            "datasets": [d.to_dict() for d in self.datasets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalog:
        return cls(
            datasets=tuple(Dataset.from_dict(item) for item in data.get("datasets", ())),
            source=data.get("source", "snap-stanford/ogb"),
            retrieved=data.get("retrieved", ""),
        )


def _parse_master(family: str, text: str) -> list[Dataset]:
    """Turn one ``master.csv`` into datasets. It is transposed: one column per set."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    names = [cell.strip() for cell in rows[0][1:]]
    fields = {row[0].strip(): row[1:] for row in rows[1:] if row}

    def value(key: str, column: int) -> str:
        cells = fields.get(key, ())
        return cells[column].strip() if column < len(cells) else ""

    datasets = []
    for column, name in enumerate(names):
        if not name:
            continue
        classes = value("num classes", column)
        extra = value("additional node files", column)
        url = value("url", column)
        datasets.append(
            Dataset(
                name=name,
                family=family,
                description=DESCRIPTIONS.get(name, ""),
                # OGB adds the reverse of every stored edge exactly when undirected.
                directed=value("add_inverse_edge", column).lower() != "true",
                num_classes=int(classes)
                if classes.lstrip("-").isdigit() and classes != "-1"
                else None,
                task_type=value("task type", column),
                download_name=value("download_name", column),
                version=int(value("version", column) or 1),
                split=value("split", column),
                node_files=tuple(
                    part.strip()
                    for part in extra.split(",")
                    if part.strip() and part.strip().lower() != "none"
                ),
                hetero=value("is hetero", column).lower() == "true",
                binary=value("binary", column).lower() == "true",
                domains=DOMAIN_MAP.get(name, ()),
                files=(_archive(name, url),) if url else (),
                citations=(CITATION,),
            )
        )
    return datasets


def _archive(name: str, url: str) -> Any:
    from .._core.models import DataFile

    return DataFile(name=Path(url).name, url=url, description=f"{name} archive")


def build_catalog(*, timeout: float = 60.0) -> Catalog:
    """Read OGB's ``master.csv`` files and build a fresh catalog."""
    datasets: list[Dataset] = []
    for family in FAMILIES:
        datasets.extend(
            _parse_master(family, fetch_text(MASTER_URL.format(family=family), timeout=timeout))
        )
    return Catalog(
        datasets=tuple(sorted(datasets, key=lambda d: d.name)),
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    """Re-read OGB's registry into the user cache and return the new catalog."""
    global _cached
    catalog = build_catalog(timeout=timeout)
    target = _cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _cache.write_atomic(target, json.dumps(catalog.to_dict(), indent=2).encode())
    _cached = catalog
    return catalog
