"""Loading and refreshing the AHORN catalog.

Same contract as the other sources: a snapshot ships inside the package so
browsing is offline and instant, :func:`refresh` rebuilds it into the user cache,
and a refreshed copy takes precedence over the vendored one.

The build reads the repository behind the website rather than the website.
AHORN's Astro site keeps one ``src/datasets/<slug>.mdx`` per dataset -- YAML
frontmatter validated against a published Zod schema, then a prose body -- and
that frontmatter is the authoritative record. The alternative, the site's
``/api/datasets.json``, carries no description, no licence, no citation and two
statistics, which would make this the thinnest catalog of the five sources
rather than one of the richest.

The whole repository is fetched as a single tarball and read in memory: 85 raw
file requests would otherwise run into GitHub's unauthenticated rate limit.

Frontmatter is YAML, so building needs ``pyyaml``. Nothing else here does --
reading the vendored catalog and downloading datasets need no dependency at all,
which is why the parser is imported inside the function rather than at module
level.
"""

from __future__ import annotations

import difflib
import io
import json
import re
import tarfile
import time
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from .._core import cache as _cache
from .._core.domains import domains_for_ahorn
from .._core.download import DownloadError, fetch_bytes
from ._models import REPO, Dataset, Revision

__all__ = [
    "ARCHIVE_URL",
    "INDEX_URL",
    "Catalog",
    "build_catalog",
    "load_catalog",
    "refresh",
]

SCHEMA_VERSION = 1
_VENDORED_NAME = "catalog.json"
SOURCE = "ahorn"

ARCHIVE_URL = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"
#: The site's own machine-readable index. Not what the catalog is built from --
#: see the module docstring -- but the thing to check when a slug looks stale.
INDEX_URL = "https://ahorn.rwth-aachen.de/api/datasets.json"

_DATASET_MEMBER = re.compile(r"^[^/]+/src/datasets/(?P<slug>[^/]+)\.mdx$")
_FRONTMATTER = re.compile(r"\A---\r?\n(?P<body>.*?)\r?\n---\r?\n(?P<rest>.*)\Z", re.S)
_REVISION = re.compile(r"^revision-(\d+)$")
_MARKDOWN_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\([^)]*\)")

_cached: Catalog | None = None


@dataclass(frozen=True)
class Catalog:
    """An immutable snapshot of the AHORN collection."""

    datasets: tuple[Dataset, ...]
    source: str = REPO
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
        """Look up by slug, case-insensitively, suggesting near misses."""
        index: dict[str, Dataset] = self._index  # type: ignore[attr-defined]
        try:
            return index[name.lower()]
        except KeyError:
            close = difflib.get_close_matches(name.lower(), index, n=3)
            hint = f" Did you mean: {', '.join(index[c].name for c in close)}?" if close else ""
            raise KeyError(f"No AHORN dataset named {name!r}.{hint}") from None

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
            source=data.get("source", REPO),
            retrieved=data.get("retrieved", ""),
        )


def _overview(body: str) -> str:
    """The prose under ``## Overview``, flattened to one paragraph.

    The body is MDX: import statements, JSX components and markdown mixed
    together. Only the Overview section is prose meant to be read as a
    description, so that is all this takes; links collapse to their text and the
    section ends at the next heading.
    """
    lines = body.splitlines()
    collected: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if inside:
                break
            inside = stripped.lower().lstrip("# ").startswith("overview")
            continue
        if not inside or not stripped:
            continue
        if stripped.startswith(("import ", "<", "export ")):
            continue
        collected.append(_MARKDOWN_LINK.sub(lambda m: m.group("text"), stripped))
    return " ".join(collected).strip()


def _distribution(raw: Any) -> tuple[int | None, float | None]:
    """``(maximum, mean)`` from a ``{degree: count}`` histogram.

    AHORN publishes full node- and interaction-degree distributions. Keeping
    them whole would dominate the vendored catalog for no gain, so the two
    numbers worth filtering on are derived here and the rest dropped; the
    dataset page has the box plots.
    """
    if not isinstance(raw, dict) or not raw:
        return None, None
    pairs: list[tuple[int, int]] = []
    for key, count in raw.items():
        try:
            pairs.append((int(key), int(count)))
        except (TypeError, ValueError):  # pragma: no cover - schema says numbers
            continue
    if not pairs:
        return None, None
    total = sum(count for _, count in pairs)
    weighted = sum(degree * count for degree, count in pairs)
    return max(degree for degree, _ in pairs), (weighted / total if total else None)


def _revisions(raw: Any) -> tuple[Revision, ...]:
    """Turn the ``attachments`` mapping into ordered revisions.

    Each key is ``revision-<n>``; each value has an ``ahorn`` URL, sometimes an
    ``hif`` one, and a changelog. Anything not matching that shape is skipped
    rather than guessed at, so a new attachment format shows up as a missing
    revision instead of a wrong URL.
    """
    if not isinstance(raw, dict):
        return ()
    found: list[Revision] = []
    for key, value in raw.items():
        match = _REVISION.match(str(key))
        if match is None or not isinstance(value, dict):
            continue
        url = value.get("ahorn")
        if not isinstance(url, str) or not url:
            continue
        hif = value.get("hif")
        changelog = value.get("changelog") or ()
        found.append(
            Revision(
                number=int(match.group(1)),
                url=url,
                hif_url=hif if isinstance(hif, str) else "",
                changelog=tuple(str(item) for item in changelog),
            )
        )
    return tuple(sorted(found, key=lambda item: item.number))


def _licence(raw: Any) -> tuple[str, str]:
    """``(identifier, url)``. The schema allows an SPDX object or one literal."""
    if isinstance(raw, dict):
        return str(raw.get("spdx", "")), str(raw.get("link", ""))
    return (str(raw) if raw else ""), ""


def _dataset(slug: str, front: dict[str, Any], body: str) -> Dataset:
    tags = tuple(str(tag) for tag in front.get("tags", ()) or ())
    statistics = front.get("statistics") or {}
    if not isinstance(statistics, dict):
        statistics = {}
    max_edge, mean_edge = _distribution(statistics.get("edge-degrees"))
    max_node, mean_node = _distribution(statistics.get("node-degrees"))
    licence, licence_url = _licence(front.get("license"))
    parent = front.get("parent")

    def flag(name: str) -> bool:
        return f"network-type: {name}" in tags

    return Dataset(
        name=slug,
        title=str(front.get("title", "") or ""),
        description=_overview(body),
        network_types=tuple(str(item) for item in front.get("network-type", ()) or ()),
        tags=tags,
        domains=domains_for_ahorn(slug, tags),
        nodes=statistics.get("num-nodes"),
        interactions=statistics.get("num-interactions"),
        max_interaction_size=max_edge,
        mean_interaction_size=mean_edge,
        max_node_degree=max_node,
        mean_node_degree=mean_node,
        temporal=flag("temporal"),
        weighted=flag("weighted"),
        multi_network=flag("multi-network"),
        source_url=str(front.get("source", "") or ""),
        license=licence,
        license_url=licence_url,
        citations=tuple(str(item).strip() for item in front.get("citation", ()) or ()),
        parent=str(parent) if parent else "",
        related=tuple(str(item) for item in front.get("related", ()) or ()),
        revisions=_revisions(front.get("attachments")),
    )


def _sizes(*, timeout: float) -> dict[tuple[str, int], int]:
    """``{(slug, revision): bytes}`` from the site index.

    The frontmatter records revision URLs but not how big they are, and
    "how large is this before I fetch it" is the question this package exists to
    answer offline. The index is the only place AHORN publishes that, so it is
    read as a supplement. A failure here leaves sizes unknown rather than
    failing the build; the regeneration script reports the coverage.
    """
    try:
        index = json.loads(fetch_bytes(INDEX_URL, timeout=timeout).decode("utf-8"))
    except (DownloadError, ValueError):  # pragma: no cover - network shape
        return {}
    found: dict[tuple[str, int], int] = {}
    for slug, entry in (index.get("datasets") or {}).items():
        for key, attachment in (entry.get("attachments") or {}).items():
            match = _REVISION.match(str(key))
            payload = attachment.get("ahorn") if isinstance(attachment, dict) else None
            if match is None or not isinstance(payload, dict):
                continue
            size = payload.get("size")
            if isinstance(size, int):
                found[(slug, int(match.group(1)))] = size
    return found


def build_catalog(
    *, timeout: float = 120.0, archive: bytes | None = None, sizes: bool = True
) -> Catalog:
    """Read AHORN's dataset frontmatter and build a fresh catalog.

    Pass *archive* to parse a repository tarball you already have; otherwise it
    is fetched from GitHub. *sizes* additionally reads the site index for the
    per-revision file sizes the frontmatter omits.

    Raises:
        ImportError: ``pyyaml`` is not installed. Only building needs it; the
            vendored catalog reads without it.
        DownloadError: The repository archive could not be fetched.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on install
        raise ImportError(
            "Building the AHORN catalog needs pyyaml, because the repository keeps "
            "its dataset records as YAML frontmatter. Install it with "
            "`pip install pyyaml`. Reading the catalog that ships with this "
            "package needs nothing."
        ) from error

    payload = archive if archive is not None else fetch_bytes(ARCHIVE_URL, timeout=timeout)
    datasets: list[Dataset] = []

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            match = _DATASET_MEMBER.match(member.name)
            if match is None or not member.isfile():
                continue
            handle = bundle.extractfile(member)
            if handle is None:  # pragma: no cover - directories are filtered above
                continue
            text = handle.read().decode("utf-8", errors="replace")
            split = _FRONTMATTER.match(text)
            if split is None:  # pragma: no cover - the schema requires frontmatter
                continue
            front = yaml.safe_load(split.group("body")) or {}
            if not isinstance(front, dict) or front.get("disable"):
                continue
            datasets.append(_dataset(match.group("slug"), front, split.group("rest")))

    if sizes:
        known = _sizes(timeout=timeout)
        datasets = [
            replace(
                dataset,
                revisions=tuple(
                    replace(revision, size=known.get((dataset.name, revision.number)))
                    for revision in dataset.revisions
                ),
            )
            for dataset in datasets
        ]

    return Catalog(
        datasets=tuple(sorted(datasets, key=lambda d: d.name.lower())),
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


def refresh(*, timeout: float = 120.0) -> Catalog:
    """Rebuild the catalog into the user cache and return it.

    Needs ``pyyaml``; see :func:`build_catalog`.
    """
    global _cached
    catalog = build_catalog(timeout=timeout)
    target = _cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _cache.write_atomic(target, json.dumps(catalog.to_dict(), indent=2).encode())
    _cached = catalog
    return catalog
