"""Metadata records for Netzschleuder networks.

Netzschleuder publishes a JSON API, so unlike the SNAP subpackage there is no
HTML to parse and the numbers below are the site's own precomputed analyses
rather than figures scraped from a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._core.domains import domains_for_netzschleuder_tags
from .._core.models import DataFile

__all__ = ["SITE_URL", "Dataset"]

SITE_URL = "https://networks.skewed.de"

# Tags describing a property rather than a domain or a subtype.
_PROPERTY_TAGS = frozenset(
    {
        "Weighted",
        "Unweighted",
        "Multilayer",
        "Multigraph",
        "Temporal",
        "Timestamps",
        "Metadata",
        "Bipartite",
        "Projection",
        "Online",
        "Offline",
    }
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """One Netzschleuder network.

    A Netzschleuder entry may hold several *subgraphs* under one name (for
    example one per year). :attr:`subgraphs` lists them; the statistics here
    describe the default one.
    """

    name: str
    #: The site's tags, verbatim: domains, subtypes and properties mixed.
    tags: tuple[str, ...] = ()
    description: str = ""
    #: The site's own precomputed measures for this network.
    analyses: dict[str, Any] = field(default_factory=dict)
    #: How many subgraphs the entry holds.
    subgraph_count: int = 0
    #: The first few subgraph names, as a preview. The full list is not
    #: vendored: ``openstreetmap`` alone has over 90,000 subgraphs, which would
    #: dominate the shipped catalog. Use :meth:`subgraph_names` for all of them.
    subgraphs: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    upstream_url: str = ""

    # -- shared interface -------------------------------------------------
    @property
    def domains(self) -> tuple[str, ...]:
        """Normalised domains, shared with every other source."""
        return domains_for_netzschleuder_tags(self.tags)

    @property
    def nodes(self) -> int | None:
        value = self.analyses.get("num_vertices")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def edges(self) -> int | None:
        value = self.analyses.get("num_edges")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def directed(self) -> bool | None:
        value = self.analyses.get("is_directed")
        return bool(value) if value is not None else None

    @property
    def bipartite(self) -> bool | None:
        value = self.analyses.get("is_bipartite")
        return bool(value) if value is not None else None

    # -- netzschleuder specifics -----------------------------------------
    @property
    def subtypes(self) -> tuple[str, ...]:
        """Tags that name a kind of network, e.g. ``Food web``, ``Airport``."""
        from .._core.domains import DOMAINS

        return tuple(tag for tag in self.tags if tag not in _PROPERTY_TAGS and tag not in DOMAINS)

    @property
    def weighted(self) -> bool:
        return "Weighted" in self.tags

    @property
    def multilayer(self) -> bool:
        """Whether the site publishes this as a multilayer/multiplex network.

        These are the natural test cases for Infomap's multilayer model, and
        SNAP has none.
        """
        return "Multilayer" in self.tags

    @property
    def temporal(self) -> bool:
        return "Temporal" in self.tags or "Timestamps" in self.tags

    @property
    def url(self) -> str:
        return f"{SITE_URL}/{self.name}"

    def subgraph_names(self) -> tuple[str, ...]:
        """Every subgraph name, fetched from the API on demand.

        Only a preview is vendored, so this needs network access when the entry
        has more subgraphs than :attr:`subgraphs` holds.
        """
        if self.subgraph_count <= len(self.subgraphs):
            return self.subgraphs
        import json

        from .._core.download import fetch_text

        payload = json.loads(fetch_text(f"{SITE_URL}/api/net/{self.name}"))
        return tuple(payload.get("nets") or ())

    @property
    def default_subgraph(self) -> str:
        """The subgraph :func:`networks.netzschleuder.load` fetches by default.

        Downloads are named after the *subgraph*, not the entry. For most
        entries the two coincide, but some publish several: ``karate`` holds
        ``77`` and ``78``, the two versions of Zachary's data.
        """
        return self.subgraphs[0] if self.subgraphs else self.name

    def files_for(self, subgraph: str | None = None) -> tuple[DataFile, ...]:
        """The download formats the site offers for one subgraph.

        The CSV bundle is listed first because it needs nothing beyond the
        standard library; the others are here for callers that prefer them.
        """
        stem = subgraph or self.default_subgraph
        base = f"{SITE_URL}/net/{self.name}/files/{stem}"
        return (
            DataFile(f"{stem}.csv.zip", f"{base}.csv.zip", "Edges, nodes and graph properties"),
            DataFile(f"{stem}.gml.zst", f"{base}.gml.zst", "GML (needs zstd)"),
            DataFile(f"{stem}.xml.zst", f"{base}.xml.zst", "GraphML (needs zstd)"),
            DataFile(f"{stem}.gt.zst", f"{base}.gt.zst", "graph-tool native (needs graph-tool)"),
        )

    @property
    def files(self) -> tuple[DataFile, ...]:
        """Downloads for the default subgraph."""
        return self.files_for()

    @property
    def primary_file(self) -> DataFile | None:
        return self.files[0]

    def file(self, name: str, *, subgraph: str | None = None) -> DataFile:
        candidates = self.files_for(subgraph)
        for item in candidates:
            if item.name == name or item.name.endswith(name):
                return item
        available = ", ".join(item.name for item in candidates)
        raise KeyError(f"{self.name!r} has no file {name!r}. Available: {available}")

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tags": list(self.tags),
            "description": self.description,
            "analyses": self.analyses,
            "subgraph_count": self.subgraph_count,
            "subgraphs": list(self.subgraphs),
            "citations": list(self.citations),
            "upstream_url": self.upstream_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            tags=tuple(data.get("tags", ())),
            description=data.get("description", ""),
            analyses=dict(data.get("analyses", {})),
            subgraph_count=int(data.get("subgraph_count", 0)),
            subgraphs=tuple(data.get("subgraphs", ())),
            citations=tuple(data.get("citations", ())),
            upstream_url=data.get("upstream_url", ""),
        )

    def __str__(self) -> str:
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.edges is not None:
            shape.append(f"{self.edges:,} edges")
        domains = "/".join(self.domains) or "uncategorised"
        return f"{self.name} [{domains}]" + (f" ({', '.join(shape)})" if shape else "")
