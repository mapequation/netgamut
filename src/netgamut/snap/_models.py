"""Typed records describing the SNAP collection.

These objects are pure data: constructing them never touches the network or the
filesystem. They round-trip through :meth:`to_dict` / :meth:`from_dict` so the
whole catalog can be vendored as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._core.domains import domains_for_snap
from .._core.models import DataFile


def label_column(dataset: str, file_name: str) -> str:
    """The column a node-label file becomes, named after the file.

    ``email-Eu-core-department-labels.txt.gz`` gives ``department``: SNAP has no
    schema to name it from.
    """
    stem = file_name.rsplit("/", 1)[-1].split(".")[0].lower()
    return stem.removeprefix(dataset.lower()).strip("-").removesuffix("-labels") or "label"


__all__ = ["Category", "DataFile", "Dataset"]

# Fields the SNAP index uses for the node count, in the order we prefer them.
_NODE_COLUMNS = ("Nodes",)
# Categories differ in how they count edges; temporal tables have two columns,
# and the item-oriented tables have none at all.
_EDGE_COLUMNS = (
    "Edges",
    "Static Edges",
    "Temporal Edges",
    "Edges/Actions",
)


def _parse_count(raw: str | None) -> int | None:
    """Turn a SNAP table cell such as ``"1,768,149"`` into an ``int``.

    Returns ``None`` for cells that are blank, a dash, or otherwise not a plain
    number (several rows carry values like ``"5,000 graphs"``).
    """
    if not raw:
        return None
    cleaned = raw.replace(",", "").replace(" ", " ").strip()
    if not cleaned or cleaned in {"-", "--", "n/a", "N/A"}:
        return None
    # Keep only a leading integer, so "1,000 graphs" still yields 1000.
    digits = ""
    for char in cleaned:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


@dataclass(frozen=True, slots=True)
class Dataset:
    """Metadata for a single SNAP dataset.

    Everything here is available offline, so you can inspect a dataset's size
    and shape before deciding to download it.
    """

    name: str
    #: Every category SNAP files this dataset under, in page order. Most
    #: datasets have one, but the site genuinely cross-lists some: wiki-Vote
    #: sits in both *Social networks* and *Wikipedia networks*, and
    #: soc-RedditHyperlinks in three sections.
    categories: tuple[str, ...] = ()
    category_names: tuple[str, ...] = ()
    description: str = ""
    type: str | None = None
    url: str = ""
    #: The dataset's row in the SNAP index tables, keyed by those tables' own
    #: column headers. Categories use different columns, so this preserves what
    #: the page actually said rather than forcing a common schema. For a
    #: cross-listed dataset the columns from each section are merged, which is
    #: how a temporal network keeps both *Static Edges* and *Temporal Edges*.
    columns: dict[str, str] = field(default_factory=dict)
    #: The "Dataset statistics" table from the dataset's detail page.
    statistics: dict[str, str] = field(default_factory=dict)
    files: tuple[DataFile, ...] = ()
    #: Citation lines from the detail page's "Source (citation)" section.
    citations: tuple[str, ...] = ()

    @property
    def category(self) -> str:
        """The first category SNAP files this dataset under."""
        return self.categories[0] if self.categories else ""

    @property
    def category_name(self) -> str:
        """The heading of the first category SNAP files this dataset under."""
        return self.category_names[0] if self.category_names else ""

    @property
    def domains(self) -> tuple[str, ...]:
        """Normalised domains, shared with every other source.

        SNAP groups by provenance rather than by domain, so this is a mapping
        rather than a relabelling -- see :mod:`netgamut._core.domains`. A few
        SNAP sections mix domains and yield nothing here.
        """
        return domains_for_snap(self.name, self.categories)

    @property
    def nodes(self) -> int | None:
        """Node count from the index table, or ``None`` if not reported."""
        for column in _NODE_COLUMNS:
            value = _parse_count(self.columns.get(column))
            if value is not None:
                return value
        return _parse_count(self.statistics.get("Nodes"))

    @property
    def edges(self) -> int | None:
        """Edge count from the index table, or ``None`` if not reported.

        Temporal categories report both static and temporal edges; this returns
        the static count. Use :attr:`columns` for the full picture.
        """
        for column in _EDGE_COLUMNS:
            value = _parse_count(self.columns.get(column))
            if value is not None:
                return value
        return _parse_count(self.statistics.get("Edges"))

    @property
    def directed(self) -> bool | None:
        """``True`` for directed networks, ``False`` for undirected.

        ``None`` when SNAP does not state a direction, which happens for the
        item-oriented categories such as *Online reviews*.
        """
        if not self.type:
            return None
        lowered = self.type.lower()
        if "undirected" in lowered:
            return False
        if "directed" in lowered:
            return True
        return None

    def _type_says(self, word: str) -> bool | None:
        # A graph's type lists its properties ("Weighted, Signed, Directed,
        # Temporal") and names its direction or that it is bipartite; any other
        # type ("Tweets", "Movie reviews") says nothing about the rest.
        lowered = (self.type or "").lower()
        if "irected" not in lowered and "bipartite" not in lowered:
            return None
        return word in lowered

    @property
    def weighted(self) -> bool | None:
        """Whether SNAP's type for the graph says *Weighted*; ``None`` if it is no graph's."""
        return self._type_says("weighted")

    @property
    def signed(self) -> bool | None:
        """Whether SNAP's type for the graph says *Signed*; ``None`` if it is no graph's."""
        return self._type_says("signed")

    @property
    def temporal(self) -> bool | None:
        """Whether SNAP's type for the graph says *Temporal*; ``None`` if it is no graph's."""
        return self._type_says("temporal")

    @property
    def bipartite(self) -> bool | None:
        """Whether SNAP's type for the graph says *Bipartite*; ``None`` if it is no graph's."""
        return self._type_says("bipartite")

    @property
    def primary_file(self) -> DataFile | None:
        """The file :func:`netgamut.snap.load` downloads by default.

        Prefers a plain edge list named after the dataset, then any gzipped
        text file, then the first non-readme artifact.
        """
        if not self.files:
            return None

        def is_readme(item: DataFile) -> bool:
            return "readme" in item.name.lower()

        candidates = [item for item in self.files if not is_readme(item)] or list(self.files)

        stem = self.name.lower()
        for item in candidates:
            if item.name.lower().startswith(stem) and item.name.lower().endswith(".txt.gz"):
                return item
        for item in candidates:
            if item.name.lower().endswith(".txt.gz"):
                return item
        for item in candidates:
            if item.name.lower().endswith((".csv.gz", ".txt", ".csv")):
                return item
        return candidates[0]

    @property
    def community_files(self) -> tuple[DataFile, ...]:
        """Overlapping ground-truth community lists, if SNAP publishes any.

        One line per community, listing its node ids. The ``com-*`` datasets
        use this shape; communities overlap, so scoring needs an
        overlap-aware measure.
        """
        return tuple(item for item in self.files if "cmty" in item.name.lower())

    @property
    def label_files(self) -> tuple[DataFile, ...]:
        """Node-label ground truth, if SNAP publishes any.

        One ``node label`` pair per line -- ``email-Eu-core`` labels each
        person with their department. This shape is a hard partition, so it can
        be scored directly against a community-detection run with NMI or AMI.
        """
        return tuple(item for item in self.files if "label" in item.name.lower())

    @property
    def ground_truth_files(self) -> tuple[DataFile, ...]:
        """Every reference-partition file, in either shape."""
        return self.community_files + self.label_files

    @property
    def ground_truth_kind(self) -> str | None:
        """``"communities"``, ``"labels"``, or ``None``.

        The two shapes need different scoring, so check this before choosing a
        metric.
        """
        if self.community_files:
            return "communities"
        if self.label_files:
            return "labels"
        return None

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.ground_truth_files)

    def community_file(self, variant: str = "all") -> DataFile:
        """Pick a ground-truth file.

        For the ``com-*`` datasets SNAP ships two: ``all`` (every community)
        and ``top5000`` (the 5000 highest-quality ones, the usual choice for
        benchmarking). Datasets whose ground truth is a node-label partition
        have a single file, returned regardless of *variant*.
        """
        candidates = self.ground_truth_files
        if not candidates:
            raise LookupError(f"{self.name!r} has no ground-truth files.")
        if len(candidates) == 1:
            return candidates[0]
        for item in candidates:
            if variant.lower() in item.name.lower():
                return item
        available = ", ".join(item.name for item in candidates)
        raise KeyError(
            f"No {variant!r} ground-truth file for {self.name!r}. Available: {available}"
        )

    def file(self, name: str) -> DataFile:
        """Look up one of this dataset's files by name."""
        for item in self.files:
            if item.name == name:
                return item
        available = ", ".join(item.name for item in self.files) or "none"
        raise KeyError(f"{self.name!r} has no file {name!r}. Available: {available}")

    # -- node metadata, in the shared vocabulary ------------------------
    @property
    def node_attributes(self) -> tuple[str, ...]:
        """The column :func:`netgamut.snap.node_table` returns, named after the file."""
        return tuple(label_column(self.name, item.name) for item in self.label_files[:1])

    @property
    def node_metadata(self) -> bool:
        """Whether SNAP publishes a node-label file for this dataset."""
        return bool(self.label_files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "categories": list(self.categories),
            "category_names": list(self.category_names),
            "description": self.description,
            "type": self.type,
            "url": self.url,
            "columns": self.columns,
            "statistics": self.statistics,
            "files": [item.to_dict() for item in self.files],
            "citations": list(self.citations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            categories=tuple(data.get("categories", ())),
            category_names=tuple(data.get("category_names", ())),
            description=data.get("description", ""),
            type=data.get("type"),
            url=data.get("url", ""),
            columns=dict(data.get("columns", {})),
            statistics=dict(data.get("statistics", {})),
            files=tuple(DataFile.from_dict(item) for item in data.get("files", [])),
            citations=tuple(data.get("citations", ())),
        )

    def __str__(self) -> str:
        parts = [self.name]
        if self.type:
            parts.append(f"({self.type.lower()})")
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.edges is not None:
            shape.append(f"{self.edges:,} edges")
        if shape:
            parts.append(f"[{', '.join(shape)}]")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Category:
    """One of the section headings on the SNAP index page."""

    #: The page anchor, e.g. ``"socnets"``. Stable across SNAP edits.
    id: str
    #: The section heading as rendered, e.g. ``"Social networks"``.
    name: str
    #: The one-line gloss from the index page's table of contents.
    description: str = ""
    #: The label used in the table of contents, which occasionally differs from
    #: the section heading (``"Amazon networks"`` vs ``"Product co-purchasing
    #: networks"``).
    toc_name: str = ""
    dataset_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "toc_name": self.toc_name,
            "dataset_names": list(self.dataset_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Category:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            toc_name=data.get("toc_name", ""),
            dataset_names=tuple(data.get("dataset_names", ())),
        )

    def __str__(self) -> str:
        return f"{self.name} ({len(self.dataset_names)} datasets)"
