"""PyTorch Geometric datasets, materialised once into plain files.

PyG is where most of the graph-learning literature's networks live. This catalogues the
single-graph ones -- the collections of thousands of molecules or meshes are left out,
since a :class:`~networks._core.network.Network` is one graph -- and loads them through
the same call as every other source::

    net   = networks.load("Planetoid/Cora")
    nodes = net.nodes()
    nodes["label"]              # {0: 3, ...}
    nodes.partition("label")    # {0: 3, ...}, for AMI/NMI/D

**The torch dependency is a one-time cost.** The first load needs the ``[pyg]`` extra:
PyG downloads its raw files, and this converts the result to ``edges.csv.gz`` and
``nodes.csv.gz`` in the cache. Every load after that reads those two files and never
imports torch. The materialised copy is a plain, archivable network like any other.

What is deliberately not written: the dense node-feature matrix (Cora's 1,433
bag-of-words dimensions), which is a learning input rather than network metadata, and
the train/validation/test masks, which are a benchmarking protocol. Use
:func:`from_pyg` on a ``Data`` object when you want those.

**PyG labels are anonymous.** ``Data.y`` is an integer tensor and there is no
``class_names`` anywhere in its API, so ``nodes.categories("label")`` returns the codes
themselves. What they mean lives in each dataset's originating paper -- unlike
:mod:`networks.ogb`, which ships the names. And node metadata is not ground truth for
community structure either way; see :mod:`networks._core.nodes`.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .._core.cache import dataset_dir as _dataset_dir
from .._core.models import DataFile
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
    "from_pyg",
    "info",
    "is_cached",
    "load",
    "node_table",
    "omitted",
    "refresh",
    "search",
]

SOURCE = "pyg"
_EDGES = "edges.csv.gz"
_NODES = "nodes.csv.gz"
_META = "meta.json"
_INSTALL = 'pip install "mapequation-networks[pyg]"'


def catalog(*, reload: bool = False) -> Catalog:
    """The PyG catalog, read from the vendored snapshot or a refreshed copy."""
    return load_catalog(reload=reload)


def datasets(cls: str | None = None) -> tuple[Dataset, ...]:
    """Every catalogued dataset, optionally restricted to one PyG class."""
    items = catalog().datasets
    return items if cls is None else tuple(item for item in items if item.cls == cls)


def omitted() -> dict[str, str]:
    """Classes PyG offers that are not catalogued, and why.

    Worth reading before concluding a dataset is missing: most entries are collections
    of many graphs, and the rest are classes PyG has not written a statistics table
    for, which its cheatsheet describes as a work in progress.
    """
    return dict(catalog().omitted)


def info(name: str) -> Dataset:
    """Metadata for one dataset, without downloading or importing torch."""
    return catalog().get(name)


def search(
    query: str | None = None,
    *,
    domain: str | None = None,
    max_nodes: int | None = None,
    cls: str | None = None,
    directed: bool | None = None,
    supported: bool | None = None,
) -> tuple[Dataset, ...]:
    """Filter the catalog. *query* matches the name and description.

    ``max_nodes`` matches only datasets whose size PyG publishes, and ``directed``
    only those a measuring pass has loaded: PyG states neither, so an unmeasured
    dataset is excluded rather than assumed to be anything. ``supported=False`` lists
    the ones a pass found could not be constructed at all.
    """
    results = []
    for dataset in catalog().datasets:
        if query and query.lower() not in f"{dataset.name} {dataset.description}".lower():
            continue
        if domain and domain not in dataset.domains:
            continue
        if cls and dataset.cls != cls:
            continue
        if max_nodes is not None and (dataset.nodes is None or dataset.nodes > max_nodes):
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
        f"  Class       torch_geometric.datasets.{dataset.cls}"
        + (f'(name="{dataset.dataset}")' if dataset.dataset else "()"),
    ]
    if dataset.nodes is not None:
        lines.append(f"  Size        {dataset.nodes:,} nodes, {dataset.edges:,} edges")
    else:
        lines.append("  Size        not published by PyG, and not measured yet")
    if dataset.directed is not None:
        lines.append(f"  Direction   {'directed' if dataset.directed else 'undirected'} (measured)")
    else:
        lines.append("  Direction   unknown; PyG does not say and it has not been measured")
    if not dataset.supported:
        lines.append(f"  Unloadable  {dataset.load_error}")
    if dataset.num_classes is not None:
        lines.append(f"  Classes     {dataset.num_classes} (unnamed; see the source paper)")
    lines.append(f"  Docs        {dataset.url}")
    return "\n".join(lines)


def _paths(dataset: Dataset) -> tuple[Path, Path, Path, Path]:
    base = _dataset_dir(SOURCE, dataset.name.replace("/", "_"))
    return base, base / _EDGES, base / _NODES, base / _META


def _materialise(dataset: Dataset, base: Path) -> dict[str, Any]:
    """Run PyG once and write the network out as plain files.

    This is the only place torch is imported. Everything after it reads CSV.
    """
    try:
        import torch_geometric.datasets as pyg_datasets
        from torch_geometric.utils import is_undirected
    except ImportError as error:  # pragma: no cover - depends on install
        raise ImportError(
            f"Loading {dataset.name!r} for the first time needs PyTorch Geometric. "
            f"Install it with `{_INSTALL}`. Once materialised it is read from plain "
            f"files and torch is no longer needed."
        ) from error

    factory = getattr(pyg_datasets, dataset.cls, None)
    if factory is None:
        raise LookupError(
            f"torch_geometric.datasets has no {dataset.cls!r}. The catalog is from a "
            f"different PyG version; pyg.refresh() rebuilds it."
        )

    # PyG constructors are not uniform: most take (root, name), some take root only,
    # and the in-memory ones like KarateClub take neither. Pass what each accepts.
    import inspect

    root = base / "source"
    root.mkdir(parents=True, exist_ok=True)
    accepted = set(inspect.signature(factory.__init__).parameters)
    kwargs: dict[str, Any] = {}
    if "root" in accepted:
        kwargs["root"] = str(root)
    if dataset.dataset and "name" in accepted:
        kwargs["name"] = dataset.dataset
    elif dataset.dataset and "root" not in accepted:
        raise LookupError(
            f"torch_geometric.datasets.{dataset.cls} takes no name= argument, so "
            f"{dataset.name!r} cannot be selected. pyg.refresh() rebuilds the catalog "
            f"against the installed version."
        )
    collection = factory(**kwargs)
    if len(collection) != 1:
        raise LookupError(
            f"{dataset.name!r} holds {len(collection)} graphs, not one. It should not "
            f"have been catalogued; please report it."
        )
    data = collection[0]

    edge_index = data.edge_index
    directed = not bool(is_undirected(edge_index, num_nodes=data.num_nodes))
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()

    with gzip.open(base / _EDGES, "wt", newline="") as handle:
        writer = csv.writer(handle)
        for u, v in zip(sources, targets, strict=True):
            writer.writerow((u, v))

    columns, rows = _node_columns(data)
    with gzip.open(base / _NODES, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("node", *columns))
        for node in range(data.num_nodes):
            writer.writerow((node, *(row[node] for row in rows)))

    meta = {
        "directed": directed,
        "nodes": int(data.num_nodes),
        "edges": len(sources) if directed else len(sources) // 2,
        "columns": list(columns),
        "pyg_class": dataset.cls,
        "pyg_name": dataset.dataset,
    }
    (base / _META).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _node_columns(data: Any) -> tuple[list[str], list[list[Any]]]:
    """The per-node columns worth writing: labels, and any scalar node attribute.

    ``x`` is skipped -- a dense feature matrix is a learning input, not metadata --
    and so are the train/val/test masks, which encode a benchmarking protocol rather
    than anything about the nodes.
    """
    skip = {"x", "edge_index", "edge_attr", "edge_label", "edge_label_index", "pos"}
    columns: list[str] = []
    rows: list[list[Any]] = []
    for key, value in data.items():
        if key in skip or key.endswith("_mask") or not hasattr(value, "dim"):
            continue
        if value.dim() == 1 and value.size(0) == data.num_nodes:
            columns.append("label" if key == "y" else key)
            rows.append(value.tolist())
        elif value.dim() == 2 and value.size(0) == data.num_nodes and key == "y":
            for position in range(value.size(1)):
                columns.append(f"label_{position}")
                rows.append(value[:, position].tolist())
    return columns, rows


def load(
    name: str,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
) -> Network:
    """Materialise *name* if it is not already cached, and return a handle to it.

    The first call needs the ``[pyg]`` extra and runs PyG's own downloader; it writes
    ``edges.csv.gz`` and ``nodes.csv.gz`` into the cache. Later calls read those and
    import nothing heavy.

    Raises:
        KeyError: No such dataset in the catalog.
        LookupError: A measuring pass established this one cannot be constructed.
        ImportError: First load, and PyTorch Geometric is not installed.
        LookupError: The installed PyG has no such class, or it turned out to hold
            several graphs.
    """
    dataset = info(name)
    if not dataset.supported:
        raise LookupError(
            f"{dataset.name!r} cannot be built: {dataset.load_error} "
            f"pyg.search(supported=True) lists the ones that can."
        )
    base, edges, _, meta_path = _paths(dataset)
    if force or not edges.is_file():
        base.mkdir(parents=True, exist_ok=True)
        _materialise(dataset, base)

    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    resolved = replace(
        dataset,
        directed=meta.get("directed", dataset.directed),
        nodes=meta.get("nodes", dataset.nodes),
        edges=meta.get("edges", dataset.edges),
    )
    artifact = DataFile(name=_EDGES, url=dataset.url, description=f"{dataset.name} edge list")
    return Network(dataset=resolved, file=artifact, archive_path=edges, source=SOURCE)


def node_table(network: Network) -> NodeTable:
    """Read a materialised PyG dataset's per-node metadata into the shared shape.

    Reads the ``nodes.csv.gz`` written at materialisation, so it needs no torch. PyG
    names no categories, so ``categories(col)`` returns the integer codes; the meaning
    of a code lives in the dataset's originating paper.

    Raises:
        LookupError: The materialised copy carries no node columns -- some PyG
            datasets have no ``y`` at all.
    """
    path = network.archive_path.parent / _NODES
    if not path.is_file():
        raise LookupError(f"{network.name!r} has no materialised {_NODES}.")

    with gzip.open(path, "rt") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        columns = [cell.strip() for cell in header[1:]]
        values: dict[str, dict[Any, Any]] = {column: {} for column in columns}
        for record in reader:
            if not record:
                continue
            node = int(record[0])
            for column, cell in zip(columns, record[1:], strict=False):
                text = cell.strip()
                if not text:
                    continue
                values[column][node] = int(text) if text.lstrip("-").isdigit() else text

    if not any(values.values()):
        raise LookupError(f"{network.name!r} carries no per-node attributes.")
    return NodeTable(values=values, origin=f"{SOURCE}/{network.name}")


def from_pyg(data: Any, *, directed: bool | None = None) -> tuple[Any, NodeTable]:
    """Convert a PyG ``Data`` object you already have into a graph and a node table.

    The escape hatch from the catalog: anything PyG can produce -- a dataset this
    package does not list, one graph out of a collection, a transformed object -- can
    still enter the same API. Needs networkx, and torch only because *data* is already
    a torch object.

    Returns ``(graph, NodeTable)``.
    """
    try:
        import networkx
    except ImportError as error:  # pragma: no cover - depends on install
        raise ImportError(
            "from_pyg() needs networkx. Install it with "
            '`pip install "mapequation-networks[networkx]"`.'
        ) from error

    edge_index = data.edge_index
    if directed is None:
        from torch_geometric.utils import is_undirected

        directed = not bool(is_undirected(edge_index, num_nodes=data.num_nodes))

    graph = networkx.DiGraph() if directed else networkx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))

    columns, rows = _node_columns(data)
    values = {
        column: {node: row[node] for node in range(int(data.num_nodes))}
        for column, row in zip(columns, rows, strict=True)
    }
    return graph, NodeTable(values=values, origin="pyg/Data")


def is_cached(name: str) -> bool:
    """Whether *name* has been materialised already, without importing torch."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    return _paths(dataset)[1].is_file()
