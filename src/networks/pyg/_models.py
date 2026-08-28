"""Metadata records for PyTorch Geometric datasets.

PyG publishes no dataset index and no domain field, so two things here are
hand-maintained and everything else is read from its source:

:data:`DOMAIN_MAP` assigns this package's domain vocabulary, keyed by class or by
``Class/name`` where one class spans domains. It doubles as the allowlist for classes
PyG documents *without* a statistics table: those are catalogued only if they appear
here, because without stats there is no way to tell a single network from a collection
of ten thousand molecules, and guessing wrong would put unloadable rows in the catalog.

:data:`SKIP` names classes deliberately left out even though they would qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DOMAIN_MAP", "SKIP", "SITE_URL", "Dataset"]

SITE_URL = "https://pytorch-geometric.readthedocs.io"


@dataclass(frozen=True, slots=True)
class Dataset:
    """One named PyG dataset: a class plus, where the class spans several, a name."""

    #: ``Class/name``, or just ``Class`` when the class is one dataset.
    name: str
    #: The ``torch_geometric.datasets`` class to instantiate.
    cls: str = ""
    #: The ``name=`` argument, empty when the class takes none.
    dataset: str = ""
    description: str = ""
    #: PyG's own figures, from the class docstring. ``None`` where it publishes none:
    #: the cheatsheet is explicitly a work in progress.
    nodes: int | None = None
    edges: int | None = None
    features: int | None = None
    num_classes: int | None = None
    #: PyG stores undirected graphs as both directions; nothing in the docstrings says
    #: which are directed, so this stays unknown rather than being guessed.
    directed: bool | None = None
    domains: tuple[str, ...] = ()
    paper: str = ""
    #: Why this dataset cannot be constructed, when a measuring pass found out.
    #: Only *structural* failures are recorded -- a constructor argument we do not
    #: supply, or a name PyG refuses in the mode we use. Network errors and missing
    #: optional dependencies are environmental and deliberately not baked in, since
    #: they say nothing about the dataset.
    load_error: str = ""

    @property
    def supported(self) -> bool:
        """Whether :func:`~networks.pyg.load` is known to be able to build this."""
        return not self.load_error

    @property
    def url(self) -> str:
        return f"{SITE_URL}/en/latest/generated/torch_geometric.datasets.{self.cls}.html"

    @property
    def files(self) -> tuple[Any, ...]:
        """PyG downloads its own raw files; nothing is listed here to fetch directly."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cls": self.cls,
            "dataset": self.dataset,
            "description": self.description,
            "nodes": self.nodes,
            "edges": self.edges,
            "features": self.features,
            "num_classes": self.num_classes,
            "directed": self.directed,
            "domains": list(self.domains),
            "paper": self.paper,
            "load_error": self.load_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dataset:
        return cls(
            name=data["name"],
            cls=data.get("cls", ""),
            dataset=data.get("dataset", ""),
            description=data.get("description", ""),
            nodes=data.get("nodes"),
            edges=data.get("edges"),
            features=data.get("features"),
            num_classes=data.get("num_classes"),
            directed=data.get("directed"),
            domains=tuple(data.get("domains", ())),
            paper=data.get("paper", ""),
            load_error=data.get("load_error", ""),
        )


#: Domains for the classes worth cataloguing, and the allowlist for stats-less ones.
#: ``Class/name`` overrides ``Class`` where one class spans domains.
DOMAIN_MAP: dict[str, tuple[str, ...]] = {
    # -- documented with a statistics table ---------------------------------
    "Actor": ("Social",),
    "Amazon": ("Economic",),
    "AmazonProducts": ("Economic",),
    "AttributedGraphDataset": ("Informational",),
    "AttributedGraphDataset/BlogCatalog": ("Social",),
    "AttributedGraphDataset/Facebook": ("Social",),
    "AttributedGraphDataset/Flickr": ("Social",),
    "AttributedGraphDataset/TWeibo": ("Social",),
    "AttributedGraphDataset/PPI": ("Biological",),
    "CitationFull": ("Informational",),
    "CityNetwork": ("Transportation",),
    "Coauthor": ("Social",),
    "DGraphFin": ("Economic",),
    "EllipticBitcoinDataset": ("Economic",),
    "EllipticBitcoinTemporalDataset": ("Economic",),
    "Entities": ("Informational",),
    "Flickr": ("Social",),
    "GDELTLite": ("Informational",),
    "GitHub": ("Social",),
    # A mixed benchmark suite: its datasets span domains, so most are keyed per name.
    "GraphLandDataset": ("Social",),
    "GraphLandDataset/hm-categories": ("Economic",),
    "GraphLandDataset/hm-prices": ("Economic",),
    "GraphLandDataset/city-reviews": ("Economic",),
    "GraphLandDataset/avazu-ctr": ("Economic",),
    "GraphLandDataset/city-roads-M": ("Transportation",),
    "GraphLandDataset/city-roads-L": ("Transportation",),
    "GraphLandDataset/web-topics": ("Informational",),
    "GraphLandDataset/web-fraud": ("Informational",),
    "GraphLandDataset/web-traffic": ("Informational",),
    "HeterophilousGraphDataset": ("Social",),
    "HeterophilousGraphDataset/Roman-empire": ("Informational",),
    "HeterophilousGraphDataset/Amazon-ratings": ("Economic",),
    "JODIEDataset": ("Social",),
    "KarateClub": ("Social",),
    "MyketDataset": ("Economic",),
    "NELL": ("Informational",),
    "Planetoid": ("Informational",),
    "PolBlogs": ("Informational",),
    "Reddit": ("Social",),
    "Reddit2": ("Social",),
    "RelLinkPredDataset": ("Informational",),
    "Twitch": ("Social",),
    "WebKB": ("Informational",),
    "Yelp": ("Social",),
    # -- real networks PyG documents without a statistics table -------------
    # Listing them here is what lets the builder catalogue them: their dataset
    # names come from the class docstring's Args section instead, and their sizes
    # stay unknown until PyG fills its cheatsheet in.
    "Airports": ("Transportation",),
    "BitcoinOTC": ("Economic",),
    "CoraFull": ("Informational",),
    "DeezerEurope": ("Social",),
    "EmailEUCore": ("Social",),
    "FacebookPagePage": ("Social",),
    "GemsecDeezer": ("Social",),
    "LastFMAsia": ("Social",),
    "LINKXDataset": ("Social",),
    "WikiCS": ("Informational",),
    "WikipediaNetwork": ("Informational",),
}

#: Classes left out on purpose, with the reason, so the builder can report it rather
#: than silently omitting them.
SKIP: dict[str, str] = {
    "SNAPDataset": "duplicates networks.snap, which catalogs SNAP properly",
    "SuiteSparseMatrixCollection": "sparse matrices, not networks with metadata",
    "Wikidata5M": "a knowledge base dump; the loader needs its own entity handling",
}

CITATION = (
    "Matthias Fey and Jan Eric Lenssen. Fast Graph Representation Learning with "
    "PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and "
    "Manifolds, 2019."
)
