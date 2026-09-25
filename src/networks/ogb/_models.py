"""Metadata records for Open Graph Benchmark datasets.

OGB publishes a machine-readable ``master.csv`` per task family, which is where every
field here comes from -- no scraping, no hand-transcription. What it does *not*
publish is node and edge counts; those are read out of each archive's
``raw/num-node-list.csv.gz`` when the catalog is regenerated with statistics, and stay
``None`` otherwise rather than being guessed at.

Two flags decide whether a dataset is loadable at all, and both come straight from
``master.csv``:

* ``hetero`` -- several node and edge *types* in one file. A :class:`Network` is one
  homogeneous graph, so these are catalogued and refused with an explanation.
* ``binary`` -- stored as ``data.npz`` instead of ``csv.gz``. Also catalogued and
  refused; ``ogbn-papers100M`` is 57 GB besides.

``add_inverse_edge`` is inverted into :attr:`directed`: when OGB says it adds the
reverse of every stored edge, the network is undirected and the file holds one
direction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._core.models import DataFile

__all__ = ["SITE_URL", "Dataset"]

SITE_URL = "https://ogb.stanford.edu"


@dataclass(frozen=True, slots=True)
class Dataset:
    """One OGB dataset."""

    name: str
    #: ``nodeproppred`` or ``linkproppred``. The ``graphproppred`` family is not
    #: catalogued: those are collections of thousands of small graphs, which the
    #: one-graph :class:`~networks._core.network.Network` model has nowhere to put.
    family: str = ""
    description: str = ""
    #: ``False`` exactly when OGB adds the reverse of every stored edge.
    directed: bool | None = None
    #: How many classes the node-label column has, when the family has labels.
    num_classes: int | None = None
    task_type: str = ""
    #: OGB's own name for the archive, e.g. ``arxiv`` for ``ogbn-arxiv``.
    download_name: str = ""
    version: int = 1
    #: The split OGB recommends. Recorded for provenance; splits are an ML concern
    #: and are not read.
    split: str = ""
    #: Extra per-node columns the archive carries, e.g. ``node_year``.
    node_files: tuple[str, ...] = ()
    #: Several node/edge types in one dataset.
    hetero: bool = False
    #: Stored as ``data.npz`` rather than ``csv.gz``.
    binary: bool = False
    nodes: int | None = None
    edges: int | None = None
    domains: tuple[str, ...] = ()
    files: tuple[DataFile, ...] = ()
    citations: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        """The dataset's page in OGB's documentation."""
        return f"{SITE_URL}/docs/{self.family[:4]}/"

    @property
    def supported(self) -> bool:
        """Whether :func:`~networks.ogb.load` can open this one."""
        return not (self.hetero or self.binary)

    @property
    def unsupported_reason(self) -> str:
        """Why :func:`~networks.ogb.load` refuses it, or ``""`` when it does not."""
        if self.hetero:
            return (
                "it is heterogeneous -- several node and edge types in one dataset, "
                "which a single homogeneous Network cannot represent"
            )
        if self.binary:
            return (
                "OGB stores it as data.npz rather than csv.gz, which this reader "
                "does not parse"
            )
        return ""

    @property
    def primary_file(self) -> DataFile | None:
        return self.files[0] if self.files else None

    # -- node metadata, in the shared vocabulary ------------------------
    @property
    def node_attributes(self) -> tuple[str, ...]:
        """The per-node columns the catalog can name: ``year``, ``species``.

        :meth:`Network.nodes` returns more -- the label column, named by a mapping
        file inside the archive (``arxiv_category``), and the entity mapping
        (``paper``) -- whose names only the download reveals.
        """
        return tuple(name.removeprefix("node_") for name in self.node_files)

    @property
    def node_metadata(self) -> bool | None:
        """``True`` with class labels or extra node files; otherwise unknown.

        A dataset with neither may still map its nodes to entities, which only
        the archive shows.
        """
        return True if self.num_classes or self.node_files else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "directed": self.directed,
            "num_classes": self.num_classes,
            "task_type": self.task_type,
            "download_name": self.download_name,
            "version": self.version,
            "split": self.split,
            "node_files": list(self.node_files),
            "hetero": self.hetero,
            "binary": self.binary,
            "nodes": self.nodes,
            "edges": self.edges,
            "domains": list(self.domains),
            "files": [item.to_dict() for item in self.files],
            "citations": list(self.citations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            family=data.get("family", ""),
            description=data.get("description", ""),
            directed=data.get("directed"),
            num_classes=data.get("num_classes"),
            task_type=data.get("task_type", ""),
            download_name=data.get("download_name", ""),
            version=int(data.get("version", 1)),
            split=data.get("split", ""),
            node_files=tuple(data.get("node_files", ())),
            hetero=bool(data.get("hetero", False)),
            binary=bool(data.get("binary", False)),
            nodes=data.get("nodes"),
            edges=data.get("edges"),
            domains=tuple(data.get("domains", ())),
            files=tuple(DataFile.from_dict(item) for item in data.get("files", ())),
            citations=tuple(data.get("citations", ())),
        )


#: Hand-assigned, because OGB publishes no domain field. One line per dataset so a
#: new one fails loudly in the regeneration script rather than defaulting to nothing.
DOMAIN_MAP: dict[str, tuple[str, ...]] = {
    "ogbn-arxiv": ("Informational",),
    "ogbn-products": ("Economic",),
    "ogbn-proteins": ("Biological",),
    "ogbn-mag": ("Informational",),
    "ogbn-papers100M": ("Informational",),
    "ogbl-ppa": ("Biological",),
    "ogbl-collab": ("Social",),
    "ogbl-citation2": ("Informational",),
    "ogbl-wikikg2": ("Informational",),
    "ogbl-ddi": ("Biological",),
    "ogbl-biokg": ("Biological",),
    "ogbl-vessel": ("Biological",),
}

#: What each network is, in one line. OGB's master.csv carries no description.
DESCRIPTIONS: dict[str, str] = {
    "ogbn-arxiv": "Citation network among arXiv CS papers; each node's label is its "
    "arXiv subject area.",
    "ogbn-products": "Amazon co-purchasing network; each node's label is its product "
    "category.",
    "ogbn-proteins": "Protein association network; labels are 112 binary presence "
    "indicators, and node_species is the categorical attribute.",
    "ogbn-mag": "Heterogeneous academic graph: papers, authors, institutions and "
    "fields of study.",
    "ogbn-papers100M": "Citation network of 111M papers, the largest OGB node "
    "dataset.",
    "ogbl-ppa": "Protein association network, for link prediction.",
    "ogbl-collab": "Author collaboration network, for link prediction.",
    "ogbl-citation2": "Paper citation network, for link prediction.",
    "ogbl-wikikg2": "Wikidata knowledge graph, for relation completion.",
    "ogbl-ddi": "Drug-drug interaction network, for link prediction.",
    "ogbl-biokg": "Heterogeneous biomedical knowledge graph.",
    "ogbl-vessel": "Mouse brain vasculature network.",
}

CITATION = (
    "Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, "
    "Michele Catasta and Jure Leskovec. Open Graph Benchmark: Datasets for Machine "
    "Learning on Graphs. NeurIPS 2020."
)
