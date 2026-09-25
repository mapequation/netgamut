"""Metadata records for KONECT networks.

KONECT describes every network in one fixed vocabulary -- a format (``sym``,
``asym``, ``bip``), an edge type (``unweighted``, ``positive``, ``signed``, ...)
and a handful of structural flags -- and the same two codes open every data file
as ``% <format> <edge type>``. That is what makes the collection machine-readable
where copies of it elsewhere are not, so the codes are kept verbatim here and
the convenience properties are derived from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .._core.domains import domains_for_konect
from .._core.models import DataFile

__all__ = ["EDGE_TYPES", "FORMATS", "SITE_URL", "Dataset", "attribute_name", "is_edge_list_header"]

SITE_URL = "http://konect.cc"

#: Network formats, as KONECT codes them.
FORMATS: dict[str, str] = {
    "sym": "Unipartite, undirected",
    "asym": "Unipartite, directed",
    "bip": "Bipartite, undirected",
}

#: Edge types, as KONECT codes them. ``positive`` means repeated unweighted
#: edges: a multigraph whose weight column is 1 on every line.
EDGE_TYPES: dict[str, str] = {
    "unweighted": "Unweighted, no multiple edges",
    "positive": "Unweighted, multiple edges",
    "posweighted": "Positive weights, no multiple edges",
    "signed": "Signed, possibly weighted, no multiple edges",
    "multisigned": "Signed, possibly weighted, multiple edges",
    "weighted": "Ratings, no multiple edges",
    "multiweighted": "Ratings, multiple edges",
}

# The first line of every KONECT edge list, whatever the file is called.
_EDGE_LIST_HEADER = re.compile(r"^%\s*(sym|asym|bip)\b")


def is_edge_list_header(line: str | bytes) -> bool:
    """Whether *line* opens a KONECT edge list: ``% <format> <edge type>``.

    This, not the file name, identifies the edge list. The current layout calls
    it ``out.<name>``, the older one ``rel.<relation>`` beside per-edge attribute
    files such as ``rel.<name>.person.role``, which have no such header.
    """
    text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    return bool(_EDGE_LIST_HEADER.match(text))


def attribute_name(member: str) -> str:
    """``person.name`` from ``ent.moreno_crime_crime.person.name``.

    KONECT names never contain a dot, so the last two components are always the
    entity and the attribute, whatever network token precedes them.
    """
    return ".".join(member.split(".")[1:][-2:])


# Edge types whose weight column Infomap can use as it stands.
_INFOMAP_WEIGHTS = frozenset({"unweighted", "positive", "posweighted"})
_UNWEIGHTED = frozenset({"unweighted", "positive"})
_MULTIGRAPH = frozenset({"positive", "multisigned", "multiweighted"})


@dataclass(frozen=True, slots=True)
class Dataset:
    """One KONECT network."""

    #: KONECT's internal name, which is also its URL slug and archive name.
    name: str
    #: The display name, e.g. ``Bison``.
    title: str = ""
    description: str = ""
    #: KONECT's category, without the trailing " network": ``Animal``, ``Rating``.
    category: str = ""
    #: ``sym``, ``asym`` or ``bip``; see :data:`FORMATS`.
    format: str = ""
    #: The edge type code; see :data:`EDGE_TYPES`.
    edge_type: str = ""
    #: Whether any node links to itself: from KONECT's own check, ``None`` if
    #: the page does not say. Bipartite networks cannot have loops.
    loops: bool | None = None
    #: Whether each edge carries a timestamp (a trailing column in the file).
    temporal: bool = False
    #: ``n``, ``m`` as KONECT reports them. ``m`` counts every line, so repeated
    #: edges in a multigraph count separately.
    nodes: int | None = None
    edges: int | None = None
    #: For a bipartite network, how many nodes are on the left. KONECT numbers
    #: both sides from 1; the loaded edge list shifts the right side past this.
    left_nodes: int | None = None
    node_meaning: str = ""
    edge_meaning: str = ""
    #: KONECT's statistics table, label -> value as shown.
    statistics: dict[str, str] = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    #: Where KONECT took the data from.
    upstream_url: str = ""
    #: The archive's size and ``Last-Modified`` header when it was listed. A
    #: refresh re-lists an archive only when either has changed.
    archive_bytes: int | None = None
    archive_modified: str = ""
    #: Every file in the archive, by base name. ``None`` when it was not listed:
    #: archives over 100 MB are left unread by the catalog build.
    members: tuple[str, ...] | None = None
    #: The member holding the edge list, when the archive was listed.
    edge_member: str = ""
    #: The columns :func:`networks.konect.node_table` returns (``name``, ``sex``),
    #: read from the archive's ``ent.*`` files when it was listed; ``()`` when it
    #: ships none, ``None`` when it was not listed.
    node_attributes: tuple[str, ...] | None = None

    # -- shared interface -------------------------------------------------
    @property
    def domains(self) -> tuple[str, ...]:
        """Normalised domains, shared with every other source."""
        return domains_for_konect(self.name, self.category)

    @property
    def directed(self) -> bool | None:
        return {"asym": True, "sym": False, "bip": False}.get(self.format)

    @property
    def bipartite(self) -> bool:
        return self.format == "bip"

    # -- konect specifics -------------------------------------------------
    @property
    def weighted(self) -> bool:
        """Whether the weight column carries information beyond 1."""
        return bool(self.edge_type) and self.edge_type not in _UNWEIGHTED

    @property
    def signed(self) -> bool:
        return self.edge_type in ("signed", "multisigned")

    @property
    def multigraph(self) -> bool:
        return self.edge_type in _MULTIGRAPH

    @property
    def infomap_weights(self) -> bool:
        """Whether Infomap can read the weight column as link weights."""
        return self.edge_type in _INFOMAP_WEIGHTS or not self.edge_type

    @property
    def edge_attributes(self) -> tuple[str, ...] | None:
        """Per-edge files (``person.role``), one line per edge; ``None`` if unlisted."""
        if self.members is None:
            return None
        return tuple(
            attribute_name(member)
            for member in self.members
            if member.startswith("rel.") and member != self.edge_member
        )

    @property
    def node_metadata(self) -> bool | None:
        """Whether the archive ships per-node attributes; ``None`` if unlisted."""
        attributes = self.node_attributes
        return None if attributes is None else bool(attributes)

    @property
    def duplicate_of(self) -> str | None:
        """``"<source>/<name>"`` of the same network elsewhere, if there is one.

        Cross-source queries use that copy instead of this one. See
        :func:`networks.konect.duplicates` for the rules.
        """
        from . import duplicates

        return duplicates().get(self.name)

    @property
    def url(self) -> str:
        return f"{SITE_URL}/networks/{self.name}/"

    @property
    def files(self) -> tuple[DataFile, ...]:
        return (
            DataFile(
                f"{self.name}.tar.bz2",
                f"{SITE_URL}/files/download.tsv.{self.name}.tar.bz2",
                "Edge list with KONECT's README and meta files",
            ),
        )

    @property
    def primary_file(self) -> DataFile:
        return self.files[0]

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "format": self.format,
            "edge_type": self.edge_type,
            "loops": self.loops,
            "temporal": self.temporal,
            "nodes": self.nodes,
            "edges": self.edges,
            "left_nodes": self.left_nodes,
            "node_meaning": self.node_meaning,
            "edge_meaning": self.edge_meaning,
            "statistics": self.statistics,
            "citations": list(self.citations),
            "upstream_url": self.upstream_url,
            "archive_bytes": self.archive_bytes,
            "archive_modified": self.archive_modified,
            "members": None if self.members is None else list(self.members),
            "edge_member": self.edge_member,
            "node_attributes": (
                None if self.node_attributes is None else list(self.node_attributes)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            format=data.get("format", ""),
            edge_type=data.get("edge_type", ""),
            loops=data.get("loops"),
            temporal=bool(data.get("temporal", False)),
            nodes=data.get("nodes"),
            edges=data.get("edges"),
            left_nodes=data.get("left_nodes"),
            node_meaning=data.get("node_meaning", ""),
            edge_meaning=data.get("edge_meaning", ""),
            statistics=dict(data.get("statistics", {})),
            citations=tuple(data.get("citations", ())),
            upstream_url=data.get("upstream_url", ""),
            archive_bytes=data.get("archive_bytes"),
            archive_modified=data.get("archive_modified", ""),
            members=None if data.get("members") is None else tuple(data["members"]),
            edge_member=data.get("edge_member", ""),
            node_attributes=(
                None if data.get("node_attributes") is None else tuple(data["node_attributes"])
            ),
        )

    def __str__(self) -> str:
        shape = []
        if self.nodes is not None:
            shape.append(f"{self.nodes:,} nodes")
        if self.edges is not None:
            shape.append(f"{self.edges:,} edges")
        domains = "/".join(self.domains) or "uncategorised"
        return f"{self.name} [{domains}]" + (f" ({', '.join(shape)})" if shape else "")
