"""Metadata records for AHORN datasets.

AHORN publishes its catalog twice. The website serves a thin index at
``/api/datasets.json`` -- slugs, titles, tags, two counts -- and the repository
behind it keeps the real thing as YAML frontmatter in ``src/datasets/*.mdx``,
one file per dataset, validated against a Zod schema. Every field here comes
from the frontmatter, so a record carries what the site's dataset page shows:
the upstream source, the licence, the BibTeX AHORN asks you to cite, the
per-revision Zenodo URLs with their changelogs, and the full node- and
interaction-degree distributions.

Two things distinguish these records from the other sources'. First, the unit is
an *interaction* -- a set of nodes -- not an edge, so :attr:`Dataset.edges`
reports the interaction count and exists only so cross-source ``search()`` size
filters keep working; :attr:`Dataset.interactions` is the honest name. Second,
AHORN pins revisions on Zenodo, which no other source here does, so a dataset
holds several :class:`Revision` records and ``load(..., revision=n)`` is
reproducible in a way ``load(...)`` alone is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._core.models import DataFile

__all__ = ["CITATION", "REPO", "SITE_URL", "Dataset", "Revision"]

SITE_URL = "https://ahorn.rwth-aachen.de"
REPO = "netsci-rwth/ahorn"

CITATION = (
    "Florian Frantzen and Michael T. Schaub. A repository for discovery and reuse of "
    "higher-order network datasets. arXiv:2608.03491 (2026)."
)

#: The formalisms AHORN's schema allows a dataset to be read as. Multi-select:
#: most hypergraphs also admit a simplicial reading.
NETWORK_TYPES = (
    "graph",
    "simplicial-complex",
    "cell-complex",
    "combinatorial-complex",
    "hypergraph",
)


@dataclass(frozen=True, slots=True)
class Revision:
    """One published, citable version of a dataset.

    Revisions are archived as Zenodo records, so a pinned one keeps returning the
    same bytes after AHORN corrects or re-derives the dataset. :attr:`changelog`
    is what changed relative to the revision before it.
    """

    number: int
    #: The dataset in AHORN's own line-based exchange format.
    url: str
    #: The Hypergraph Interchange Format export, where the structure fits its
    #: schema. Empty for the roughly half of revisions that have none.
    hif_url: str = ""
    size: int | None = None
    changelog: tuple[str, ...] = ()

    @property
    def file(self) -> DataFile:
        name = self.url.split("?", 1)[0].rsplit("/", 1)[-1]
        return DataFile(name=name, url=self.url, description=f"revision {self.number}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "url": self.url,
            "hif_url": self.hif_url,
            "size": self.size,
            "changelog": list(self.changelog),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Revision:
        return cls(
            number=int(data["number"]),
            url=data["url"],
            hif_url=data.get("hif_url", ""),
            size=data.get("size"),
            changelog=tuple(data.get("changelog", ())),
        )


@dataclass(frozen=True, slots=True)
class Dataset:
    """One AHORN dataset."""

    #: AHORN's slug, which is what every function here takes.
    name: str
    title: str = ""
    description: str = ""
    #: The formalisms AHORN suggests reading it as -- ``hypergraph``,
    #: ``simplicial-complex``, ``combinatorial-complex``, ``graph``. Suggestions,
    #: in AHORN's words, not claims to completeness.
    network_types: tuple[str, ...] = ()
    #: AHORN's own namespaced tags, verbatim: ``domain: reviews``,
    #: ``source: benson``, ``network-type: temporal``, ``metadata: node classes``.
    tags: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    nodes: int | None = None
    #: Hyperedges, simplices, cells -- whatever this dataset's formalism calls
    #: them. The count AHORN reports as ``num-interactions``.
    interactions: int | None = None
    max_interaction_size: int | None = None
    mean_interaction_size: float | None = None
    max_node_degree: int | None = None
    mean_node_degree: float | None = None
    temporal: bool = False
    weighted: bool = False
    multi_network: bool = False
    #: Where AHORN got it, before conversion.
    source_url: str = ""
    #: An SPDX identifier, or ``unprovided-reuse-encouraged`` where the upstream
    #: source stated none. A dataset page documents the upstream conditions; it
    #: does not grant rights beyond them.
    license: str = ""
    license_url: str = ""
    #: BibTeX for the original source publication. Cite this, plus the Zenodo
    #: record when the exact version matters -- AHORN keeps repository citation
    #: and dataset citation deliberately separate.
    citations: tuple[str, ...] = ()
    #: The slug this was derived from, for the cuisine- and genre-specific
    #: children that share a converter with their parent.
    parent: str = ""
    related: tuple[str, ...] = ()
    revisions: tuple[Revision, ...] = ()

    #: Interactions are unordered sets, so there is nothing to be directed
    #: about. Present because every source's record must answer this.
    directed: bool | None = False

    @property
    def url(self) -> str:
        """The dataset's page on the AHORN site."""
        return f"{SITE_URL}/dataset/{self.name}"

    @property
    def edges(self) -> int | None:
        """The interaction count, under the name cross-source ``search()`` filters on.

        An interaction is not an edge. This exists so ``max_edges=`` keeps
        meaning "not bigger than this" across all sources; read
        :attr:`interactions` when you mean it.
        """
        return self.interactions

    @property
    def latest(self) -> Revision | None:
        """The highest-numbered revision, which is what ``load()`` fetches."""
        return max(self.revisions, key=lambda item: item.number) if self.revisions else None

    def revision(self, number: int | None = None) -> Revision:
        """One revision by number, or the latest when *number* is ``None``.

        Raises:
            LookupError: No such revision, or the dataset publishes none.
        """
        if number is None:
            latest = self.latest
            if latest is None:
                raise LookupError(f"{self.name!r} publishes no downloadable revision.")
            return latest
        for item in self.revisions:
            if item.number == number:
                return item
        available = ", ".join(str(item.number) for item in self.revisions) or "none"
        raise LookupError(f"{self.name!r} has no revision {number}. Available: {available}.")

    @property
    def files(self) -> tuple[DataFile, ...]:
        """The latest revision's artifact, in the shape the cache reports."""
        latest = self.latest
        return (latest.file,) if latest is not None else ()

    @property
    def hypergraph(self) -> bool:
        return "hypergraph" in self.network_types

    @property
    def simplicial(self) -> bool:
        return "simplicial-complex" in self.network_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "network_types": list(self.network_types),
            "tags": list(self.tags),
            "domains": list(self.domains),
            "nodes": self.nodes,
            "interactions": self.interactions,
            "max_interaction_size": self.max_interaction_size,
            "mean_interaction_size": self.mean_interaction_size,
            "max_node_degree": self.max_node_degree,
            "mean_node_degree": self.mean_node_degree,
            "temporal": self.temporal,
            "weighted": self.weighted,
            "multi_network": self.multi_network,
            "source_url": self.source_url,
            "license": self.license,
            "license_url": self.license_url,
            "citations": list(self.citations),
            "parent": self.parent,
            "related": list(self.related),
            "revisions": [item.to_dict() for item in self.revisions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            network_types=tuple(data.get("network_types", ())),
            tags=tuple(data.get("tags", ())),
            domains=tuple(data.get("domains", ())),
            nodes=data.get("nodes"),
            interactions=data.get("interactions"),
            max_interaction_size=data.get("max_interaction_size"),
            mean_interaction_size=data.get("mean_interaction_size"),
            max_node_degree=data.get("max_node_degree"),
            mean_node_degree=data.get("mean_node_degree"),
            temporal=bool(data.get("temporal", False)),
            weighted=bool(data.get("weighted", False)),
            multi_network=bool(data.get("multi_network", False)),
            source_url=data.get("source_url", ""),
            license=data.get("license", ""),
            license_url=data.get("license_url", ""),
            citations=tuple(data.get("citations", ())),
            parent=data.get("parent", ""),
            related=tuple(data.get("related", ())),
            revisions=tuple(Revision.from_dict(item) for item in data.get("revisions", ())),
        )

    def __str__(self) -> str:
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.interactions is not None:
            shape.append(f"{self.interactions:,} interactions")
        domains = "/".join(self.domains) or "uncategorised"
        tail = f" ({', '.join(shape)})" if shape else ""
        return f"{self.name} [{domains}]{tail}"
