"""Open Graph Benchmark: the graph datasets the GNN literature is benchmarked on.

OGB is small next to the other sources -- a dozen single networks -- but the metadata
is the best of any of them. Where SNAP gives one labelled dataset and PyG gives
anonymous integers, OGB ships a ``mapping/`` folder that *names* every code:
``ogbn-arxiv`` node labels decode to ``cs.NA``, ``cs.LO`` and the rest, and
``nodeidx2paperid.csv.gz`` resolves each node to a real paper. That is what makes it
worth reading attribute-wise rather than just topologically::

    net   = netgamut.load("ogbn-arxiv")
    nodes = net.nodes()
    nodes.columns                       # ('arxiv_category', 'year', 'paper')
    nodes["arxiv_category"]             # {0: 'arxiv cs na', ...}
    nodes.partition("arxiv_category")   # {0: 3, ...}, for AMI/NMI/D

Columns are named the way the source names them -- ``arxiv_category`` comes from
``mapping/labelidx2arxivcategeory.csv.gz``, ``paper`` from ``nodeidx2paperid`` -- so a
figure legend can use them as they are. ``NodeTable``'s repr lists what a dataset
carries.

Nothing here needs ``torch`` or the ``ogb`` package: the archives are plain zips of
``csv.gz``, and this reads them directly. Only ``split/*.pt`` is torch's, and splits
are a machine-learning concern this package has no use for.

Not everything OGB publishes is loadable. The ``graphproppred`` family -- 15
collections of thousands of small molecule graphs -- is not catalogued at all, since a
:class:`~netgamut._core.network.Network` is one graph. Heterogeneous datasets
(``ogbn-mag``, ``ogbl-biokg``) and the two stored as ``data.npz``
(``ogbn-papers100M``, ``ogbl-vessel``) are catalogued but refused, each with its
reason; ``search(supported=True)`` filters them out.

**Node metadata is not ground truth for community structure.** An arXiv subject area
is what an author ticked, not a claim about flow; see :mod:`netgamut._core.nodes`.
"""

from __future__ import annotations

import csv
import gzip
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .._core.cache import dataset_dir as _dataset_dir
from .._core.download import download as _download_file
from .._core.network import Network
from .._core.nodes import NodeTable
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._models import SITE_URL, Dataset

__all__ = [
    "SITE_URL",
    "Catalog",
    "Dataset",
    "build_catalog",
    "catalog",
    "datasets",
    "describe",
    "info",
    "is_cached",
    "load",
    "node_table",
    "refresh",
    "search",
]

SOURCE = "ogb"

#: Files inside the archive, relative to its single top-level directory.
_EDGES = "raw/edge.csv.gz"
_NUM_NODES = "raw/num-node-list.csv.gz"
_NODE_LABEL = "raw/node-label.csv.gz"
_MAPPING = "mapping"


def catalog(*, reload: bool = False) -> Catalog:
    """The OGB catalog, read from the vendored snapshot or a refreshed copy."""
    return load_catalog(reload=reload)


def datasets(family: str | None = None) -> tuple[Dataset, ...]:
    """Every catalogued dataset, optionally restricted to one task family."""
    items = catalog().datasets
    if family is None:
        return items
    return tuple(item for item in items if item.family == family)


def info(name: str) -> Dataset:
    """Metadata for one dataset, without downloading anything."""
    return catalog().get(name)


def search(
    query: str | None = None,
    *,
    domain: str | None = None,
    directed: bool | None = None,
    supported: bool | None = None,
) -> tuple[Dataset, ...]:
    """Filter the catalog. *query* matches the name and description."""
    results = []
    for dataset in catalog().datasets:
        if query and query.lower() not in f"{dataset.name} {dataset.description}".lower():
            continue
        if domain and domain not in dataset.domains:
            continue
        if directed is not None and dataset.directed is not directed:
            continue
        if supported is not None and dataset.supported is not supported:
            continue
        results.append(dataset)
    return tuple(results)


def describe(name: str) -> str:
    """A readable metadata summary for one dataset, for printing."""
    dataset = info(name)
    lines = [
        f"{dataset.name}  [{', '.join(dataset.domains) or 'uncategorised'}]",
        f"  {dataset.description}" if dataset.description else "",
        f"  Family      {dataset.family}",
        f"  Direction   {'directed' if dataset.directed else 'undirected'}",
    ]
    if dataset.nodes is not None:
        lines.append(f"  Size        {dataset.nodes:,} nodes, {dataset.edges:,} edges")
    if dataset.num_classes is not None:
        lines.append(f"  Classes     {dataset.num_classes}")
    if dataset.node_files:
        lines.append(f"  Node files  {', '.join(dataset.node_files)}")
    if not dataset.supported:
        lines.append(f"  Unsupported {dataset.unsupported_reason}")
    return "\n".join(line for line in lines if line)


def _require_supported(dataset: Dataset) -> None:
    if not dataset.supported:
        raise LookupError(
            f"{dataset.name!r} cannot be loaded because {dataset.unsupported_reason}. "
            f"ogb.search(supported=True) lists the ones that can."
        )


def load(
    name: str,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download *name* if it is not already cached, and return a handle to it.

    The archive is fetched whole -- OGB ships one zip per dataset -- and the handle
    points at its ``raw/edge.csv.gz``, so ``.edges()``, ``.to_networkx()`` and
    ``.nodes()`` all work as they do for every other source.

    Sizes are real: ``ogbl-ddi`` is 46 MB, ``ogbn-arxiv`` 83 MB, ``ogbn-products``
    over 1 GB. Pass *max_bytes* as a guard.

    Raises:
        KeyError: No such dataset.
        LookupError: The dataset is catalogued but not loadable -- heterogeneous, or
            stored as ``data.npz``.
        DownloadError: The transfer failed or exceeded *max_bytes*.
    """
    dataset = info(name)
    _require_supported(dataset)
    artifact = dataset.primary_file
    if artifact is None:  # pragma: no cover - every catalogued entry has one
        raise LookupError(f"{dataset.name!r} lists no downloadable archive.")

    target = _dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name
    _download_file(artifact.url, target, force=force, progress=progress, max_bytes=max_bytes)
    return Network(
        dataset=dataset,
        file=artifact,
        archive_path=target,
        member=f"{dataset.download_name}/{_EDGES}",
        source=SOURCE,
    )


def _members(archive: Path) -> dict[str, str]:
    """``{path below the top-level directory: full member name}``."""
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
    return {name.split("/", 1)[1]: name for name in names if "/" in name}


def _read_csv(archive: Path, member: str) -> Iterable[list[str]]:
    with zipfile.ZipFile(archive) as bundle:
        payload = gzip.decompress(bundle.read(member)).decode("utf-8", errors="replace")
    return list(csv.reader(payload.splitlines()))


def _label_names(archive: Path, members: dict[str, str]) -> tuple[str, dict[int, str]]:
    """Find and read ``mapping/labelidx2<thing>.csv.gz``, OGB's names for the codes.

    Returns ``(column name, {code: name})``. The column is named after what the
    mapping calls the thing -- ``labelidx2arxivcategeory`` gives ``subject`` -- rather
    than a generic "label", because that is the whole point of OGB shipping it.
    """
    for relative, member in members.items():
        head, _, tail = relative.partition("/")
        if head != _MAPPING or not tail.startswith("labelidx2"):
            continue
        rows = list(_read_csv(archive, member))
        if len(rows) < 2:
            continue
        header = [cell.strip() for cell in rows[0]]
        column = header[1] if len(header) > 1 else "label"
        # "arxiv category" -> "subject"; otherwise use the mapping's own second column
        names = {int(row[0]): row[1].strip() for row in rows[1:] if len(row) >= 2 and row[0]}
        return column.replace(" ", "_"), names
    return "label", {}


def node_table(network: Network) -> NodeTable:
    """Read an OGB dataset's per-node metadata into the shared shape.

    Everything comes out of the archive already downloaded for the edge list: the
    label column from ``raw/node-label.csv.gz``, its names from
    ``mapping/labelidx2*.csv.gz`` where OGB provides them, any extra per-node columns
    the catalog lists (``node_year``, ``node_species``), and the entity mapping from
    ``mapping/nodeidx2*.csv.gz``.

    A multi-task dataset labels each node with a *vector* rather than a class --
    ``ogbn-proteins`` carries 112 binary indicators -- so those arrive as
    ``label_0 .. label_111``. Its meaningful categorical attribute is ``species``.

    Raises:
        LookupError: The archive carries no per-node files at all.
    """
    archive = network.archive_path
    members = _members(archive)
    values: dict[str, dict[Any, Any]] = {}
    names: dict[str, dict[Any, str]] = {}

    column, label_names = _label_names(archive, members)
    if _NODE_LABEL in members:
        rows = list(_read_csv(archive, members[_NODE_LABEL]))
        width = max((len(row) for row in rows), default=0)
        if width == 1:
            values[column] = {
                node: int(row[0]) for node, row in enumerate(rows) if row and row[0].strip()
            }
            if label_names:
                names[column] = label_names
        elif width > 1:
            for position in range(width):
                values[f"label_{position}"] = {
                    node: row[position]
                    for node, row in enumerate(rows)
                    if len(row) > position and row[position].strip()
                }

    # extra per-node columns the catalog lists: node_year, node_species, ...
    for extra in getattr(network.dataset, "node_files", ()):
        member = members.get(f"raw/{extra}.csv.gz")
        if member is None:
            continue
        rows = list(_read_csv(archive, member))
        label = extra.removeprefix("node_")
        values[label] = {
            node: (int(row[0]) if row[0].strip().lstrip("-").isdigit() else row[0].strip())
            for node, row in enumerate(rows)
            if row and row[0].strip()
        }

    # entity resolution: nodeidx2paperid, nodeidx2drugid, ...
    for relative, member in members.items():
        head, _, tail = relative.partition("/")
        if head != _MAPPING or not tail.startswith("nodeidx2"):
            continue
        rows = list(_read_csv(archive, member))
        if len(rows) < 2:
            continue
        label = tail.split(".")[0].removeprefix("nodeidx2").removesuffix("id") or "entity"
        values[label] = {
            int(row[0]): row[1].strip() for row in rows[1:] if len(row) >= 2 and row[0].strip()
        }

    if not values:
        raise LookupError(f"{network.name!r} ships no per-node files in its archive.")
    return NodeTable(values=values, names=names, origin=f"{SOURCE}/{network.name}")


def is_cached(name: str) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    artifact = dataset.primary_file
    if artifact is None:
        return False
    return (_dataset_dir(SOURCE, dataset.name) / Path(artifact.name).name).is_file()
