"""Loading, querying and refreshing the KONECT catalog.

Same contract as the other sources: a snapshot ships inside the package so
browsing is offline and instant, :func:`refresh` re-reads the live site into the
user cache, and a refreshed copy takes precedence.

KONECT has no API, so the build reads the network index and then each network's
page; see :mod:`networks.konect._scrape` for what is parsed and why. Whether a
network ships node attributes is on no page, so the build also lists each archive
up to 100 MB -- 268 of 300, 2.4 GB -- by streaming it through ``tarfile`` without
keeping it. The archives last changed in 2017-2018, so a refresh reuses every
listing whose archive size and ``Last-Modified`` are unchanged, and in practice
downloads nothing.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import tarfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .._core import cache as _cache
from .._core.domains import DOMAINS
from .._core.download import fetch_text, head, open_stream
from . import _ent
from ._models import SITE_URL, Dataset, is_edge_list_header
from ._scrape import is_series, parse_index, parse_page

__all__ = ["Catalog", "build_catalog", "load_catalog", "refresh"]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
INDEX_URL = f"{SITE_URL}/networks/"
# The site is one person's server; keep the build gentle.
_REFRESH_WORKERS = 4
#: Archives up to this size are listed by the build; larger ones stay unlisted.
LIST_CAP_BYTES = 100_000_000
# Listing streams the whole archive, so allow far longer than a page fetch.
_LIST_TIMEOUT = 600.0

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
        node_metadata: bool | None = None,
        max_nodes: int | None = None,
        min_nodes: int | None = None,
        max_edges: int | None = None,
        min_edges: int | None = None,
    ) -> tuple[Dataset, ...]:
        """Filter the catalog by domain, category, size or structural property.

        ``loops`` and ``node_metadata`` exclude networks whose answer is unknown
        -- no loop check reported, or an archive too large to have been listed --
        rather than assuming either answer.
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
            if node_metadata is not None and item.node_metadata is not node_metadata:
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


def _leading_comments(handle: Any, limit: int = 20) -> list[str]:
    """The ``%`` lines a member opens with."""
    lines: list[str] = []
    for raw in handle:
        line = raw.decode("utf-8", "replace")
        if not line.startswith("%") or len(lines) >= limit:
            break
        lines.append(line.rstrip("\n"))
    return lines


def list_archive(url: str) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    """Every file in a KONECT archive, its edge list, and its node columns.

    Streams the archive once, in memory, and keeps nothing. The edge list is the
    member whose first line is a ``% <format> <edge type>`` header; see
    :func:`~networks.konect._models.is_edge_list_header`. The node columns are
    named from each ``ent.*`` file's name and leading ``%`` lines, as
    :func:`networks.konect.node_table` will name them.
    """
    members: list[str] = []
    edge_member = ""
    entity_files: list[tuple[str, list[str]]] = []
    with (
        open_stream(url, timeout=_LIST_TIMEOUT) as response,
        tarfile.open(fileobj=response, mode="r|*") as archive,
    ):
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            members.append(name)
            if name.startswith("ent."):
                handle = archive.extractfile(member)
                entity_files.append((name, _leading_comments(handle) if handle else []))
            elif not edge_member and name.startswith(("out.", "rel.")):
                handle = archive.extractfile(member)
                if handle is not None and is_edge_list_header(handle.readline()):
                    edge_member = name
    return tuple(members), edge_member, _ent.columns(entity_files)


def _with_listing(dataset: Dataset, previous: Catalog | None, *, cap: int | None) -> Dataset:
    """*dataset* with its archive's size, date and listing filled in.

    A listing from *previous* is reused while the archive's size and date are
    unchanged; otherwise an archive up to *cap* bytes is listed afresh.
    """
    validators = head(dataset.primary_file.url)
    size = validators["bytes"] if isinstance(validators["bytes"], int) else None
    modified = str(validators["last_modified"] or "")
    stamped = dataclasses.replace(dataset, archive_bytes=size, archive_modified=modified)

    if previous is not None and dataset.name in previous:
        old = previous[dataset.name]
        if size is not None and (old.archive_bytes, old.archive_modified) == (size, modified):
            return dataclasses.replace(
                stamped,
                members=old.members,
                edge_member=old.edge_member,
                node_attributes=old.node_attributes,
            )

    if size is None or cap is None or size > cap:
        return stamped
    try:
        members, edge_member, node_attributes = list_archive(dataset.primary_file.url)
    except Exception:
        # Unlisted is an honest answer; one bad archive must not sink the build.
        return stamped
    return dataclasses.replace(
        stamped, members=members, edge_member=edge_member, node_attributes=node_attributes
    )


def build_catalog(
    *,
    workers: int = _REFRESH_WORKERS,
    previous: Catalog | None = None,
    list_cap: int | None = LIST_CAP_BYTES,
) -> Catalog:
    """Read the downloadable, non-series part of KONECT from the live site.

    *previous* supplies archive listings to reuse; with none, every archive up to
    *list_cap* bytes is downloaded once to list it. ``list_cap=None`` skips
    listing entirely.
    """
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
        dataset = Dataset(
            name=row["name"],
            title=row["title"],
            nodes=row["nodes"],
            edges=row["edges"],
            **{**page, "temporal": page["temporal"] or row["temporal"]},
        )
        return _with_listing(dataset, previous, cap=list_cap)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fetched = [item for item in pool.map(fetch, rows) if item is not None]

    return Catalog(
        datasets=tuple(fetched),
        source=INDEX_URL,
        retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        origin="cache",
    )


def refresh(*, workers: int = _REFRESH_WORKERS) -> Catalog:
    """Re-read the live site and store the result in the user cache.

    Archive listings are carried over from the current catalog while their
    archives are unchanged, so this downloads an archive only when KONECT has
    published a new or changed one.
    """
    global _cached
    catalog = build_catalog(workers=workers, previous=load_catalog())
    catalog.write(_refreshed_path())
    _cached = catalog
    return catalog
