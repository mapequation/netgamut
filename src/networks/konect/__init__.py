"""Access to KONECT, the Koblenz Network Collection (konect.cc).

KONECT, curated by Jérôme Kunegis, describes every network in one fixed
vocabulary: its format (undirected, directed, bipartite), its edge type
(unweighted, multigraph, positive or signed weights), timestamps, and whether it
contains loops. The loop check is what no other source here reports, and it is
what a benchmark of how community detection treats self-links needs::

    from networks import konect

    konect.search(loops=True, max_nodes=10_000)
    konect.search(signed=True)
    konect.search(bipartite=True, domain="Economic")

The catalog is KONECT's downloadable networks minus its per-language Wikimedia
series (about a thousand edit, hyperlink and talk networks, one per language
edition), of which only the English editions are kept.

Netzschleuder imported a third of KONECT, and some of it is SNAP's. Such a
network is still listed here, but the cross-source :func:`networks.search`,
:func:`networks.domains` and :func:`networks.load` use the other copy; see
:func:`duplicates`.

Please cite the networks you use; see ``Dataset.citations``.
"""

from __future__ import annotations

import os
import tarfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from .._core.cache import cache_dir, cache_info, clear_cache, file_lock, set_cache_dir
from .._core.cache import dataset_dir as _dataset_dir
from .._core.domains import DOMAINS
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.network import Network
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._models import EDGE_TYPES, FORMATS, SITE_URL, Dataset

__all__ = [
    "DOMAINS",
    "EDGE_TYPES",
    "FORMATS",
    "SITE_URL",
    "Catalog",
    "DataFile",
    "Dataset",
    "DownloadError",
    "Network",
    "build_catalog",
    "cache_dir",
    "cache_info",
    "catalog",
    "categories",
    "clear_cache",
    "datasets",
    "describe",
    "domains",
    "duplicates",
    "info",
    "is_cached",
    "load",
    "refresh",
    "search",
    "set_cache_dir",
]

SOURCE = "konect"


def catalog(*, reload: bool = False) -> Catalog:
    """The KONECT catalog, from the cache if refreshed, else the package."""
    return load_catalog(reload=reload)


def datasets(domain: str | None = None) -> tuple[Dataset, ...]:
    """Every network, or just those in *domain*."""
    current = catalog()
    return current.search(domain=domain) if domain else current.datasets


def info(name: str) -> Dataset:
    """Metadata for one network. Never downloads anything."""
    return catalog().dataset(name)


def domains() -> dict[str, int]:
    """Network counts per normalised domain, duplicates included."""
    return catalog().domains()


def categories() -> dict[str, int]:
    """KONECT's own categories, with counts, most common first."""
    return catalog().categories()


def search(
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
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
) -> tuple[Dataset, ...]:
    """Find networks by domain, KONECT category, size or structural property.

    ``loops=True`` finds networks with at least one self-link, by KONECT's own
    check; networks it has no answer for are excluded from either filter.
    """
    return catalog().search(
        query,
        domain=domain,
        category=category,
        directed=directed,
        bipartite=bipartite,
        weighted=weighted,
        signed=signed,
        temporal=temporal,
        loops=loops,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        max_edges=max_edges,
        min_edges=min_edges,
    )


# -- overlap with the other sources -----------------------------------------
_overlap: tuple[tuple[int, int, int], dict[str, str], dict[str, Dataset]] | None = None


def _compute_overlap() -> tuple[dict[str, str], dict[str, Dataset]]:
    global _overlap
    from .. import netzschleuder, snap

    nz, sn, kc = netzschleuder.catalog(), snap.catalog(), catalog()
    key = (id(nz), id(sn), id(kc))
    if _overlap is not None and _overlap[0] == key:
        return _overlap[1], _overlap[2]

    imported: dict[str, str] = {}
    for item in nz.datasets:
        # An entry of many subgraphs (one per language, say) describes its
        # default one, which need not be the KONECT network it links to.
        if item.subgraph_count > 1:
            continue
        host, _, path = item.upstream_url.partition("://")[2].partition("/")
        if "konect" in host and path.startswith("networks/"):
            imported[path.removeprefix("networks/").strip("/")] = f"netzschleuder/{item.name}"

    # Netzschleuder is filled last, so it wins a size tie with SNAP.
    by_size: dict[tuple[int, int], str] = {}
    for prefix, items in (("snap", sn.datasets), ("netzschleuder", nz.datasets)):
        for other in items:
            if other.nodes and other.edges:
                by_size[(other.nodes, other.edges)] = f"{prefix}/{other.name}"

    duplicate: dict[str, str] = {}
    twins: dict[str, Dataset] = {}
    for dataset in kc.datasets:
        size = (dataset.nodes or 0, dataset.edges or 0)
        twin = by_size.get(size) if all(size) else None
        if twin is not None:
            twins[twin] = dataset
        found = imported.get(dataset.name) or twin
        if found is not None:
            duplicate[dataset.name] = found

    _overlap = (key, duplicate, twins)
    return duplicate, twins


def duplicates() -> dict[str, str]:
    """KONECT name -> ``"<source>/<name>"`` of the copy cross-source queries use.

    A KONECT network is a duplicate on evidence, never on name alone -- KONECT's
    ``epinions`` and ``twitter`` are different networks from Netzschleuder's.
    The evidence is either of:

    * Netzschleuder's entry names it as its upstream and holds that one
      network, not a family of subgraphs: the data was imported.
    * A Netzschleuder or SNAP entry has exactly the same node and edge count.

    Near-copies that differ in count -- a different snapshot, the largest
    component only -- stay separate networks.
    """
    return dict(_compute_overlap()[0])


def loops_for(key: str) -> bool | None:
    """KONECT's loop check for another source's network, if KONECT holds it.

    *key* is ``"<source>/<name>"``. Only an exact-size twin counts, so the check
    describes the same edge list.
    """
    twin = _compute_overlap()[1].get(key)
    return twin.loops if twin is not None else None


# -- describe and load --------------------------------------------------------
def describe(name: str) -> str:
    """A readable metadata summary for one network, for printing."""
    dataset = info(name)
    lines = [f"{dataset.name} ({dataset.title})", "=" * len(dataset.name)]
    lines.append(f"Domains     {', '.join(dataset.domains) or '-'}")
    lines.append(f"Category    {dataset.category or '-'}")
    lines.append(f"Format      {FORMATS.get(dataset.format, dataset.format or '-')}")
    lines.append(f"Edge type   {EDGE_TYPES.get(dataset.edge_type, dataset.edge_type or '-')}")
    loops = {True: "contains loops", False: "no loops", None: "not reported"}[dataset.loops]
    lines.append(f"Loops       {loops}")
    if dataset.temporal:
        lines.append("Temporal    edges carry timestamps")
    if dataset.node_meaning or dataset.edge_meaning:
        lines.append(f"Meaning     nodes: {dataset.node_meaning}; edges: {dataset.edge_meaning}")
    if dataset.duplicate_of:
        lines.append(f"Also as     {dataset.duplicate_of} (used by networks.search/load)")
    if dataset.description:
        lines.append(f"Summary     {dataset.description[:400]}")
    lines.append(f"Page        {dataset.url}")

    if dataset.statistics:
        lines.append("")
        lines.append("Reported statistics")
        for label, value in list(dataset.statistics.items())[:12]:
            lines.append(f"  {label:<32} {value}")

    if dataset.citations:
        lines.append("")
        lines.append("Source")
        for citation in dataset.citations:
            lines.append(f"  {citation[:200]}")
    return "\n".join(lines)


def _normalise(
    lines: Iterable[str], *, bipartite: bool, left_nodes: int | None = None
) -> Iterator[str]:
    """KONECT's edge list, rewritten so every tool reads it the same way.

    ``%`` comments become ``#``, which Infomap's parser requires. In a bipartite
    network both sides are numbered from 1, so the right side is shifted past the
    left: node ids are then unique, and right-side ids start at
    ``left_nodes + 1``. The left size comes from the catalog, else from KONECT's
    optional ``% m n1 n2`` header line. Every column is kept -- weights, signs
    and timestamps stay available through ``Network.rows()``.
    """
    offset = left_nodes
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            body = stripped.lstrip("% ")
            fields = body.split()
            size_line = len(fields) == 3 and all(field.isdigit() for field in fields)
            if bipartite and offset is None and size_line:
                offset = int(fields[1])
            yield f"# {body}\n"
            continue
        fields = stripped.split()
        if bipartite:
            if offset is None:
                raise ValueError("bipartite network with no left-side size in catalog or header")
            fields[1] = str(int(fields[1]) + offset)
        yield "\t".join(fields) + "\n"


def _write_edge_file(dataset: Dataset, archive_path: Path, target: Path) -> None:
    with file_lock(target):
        if target.is_file():
            return
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        try:
            # Stream mode: one pass over the compressed archive, flat memory.
            with tarfile.open(archive_path, "r|*") as archive:
                for member in archive:
                    if not (member.isfile() and Path(member.name).name.startswith("out.")):
                        continue
                    handle = archive.extractfile(member)
                    assert handle is not None
                    lines = (raw.decode("utf-8", "replace") for raw in handle)
                    rewritten = _normalise(
                        lines, bipartite=dataset.bipartite, left_nodes=dataset.left_nodes
                    )
                    with partial.open("w", encoding="utf-8") as out:
                        out.writelines(rewritten)
                    break
                else:
                    raise LookupError(f"{archive_path.name} holds no out.* edge list.")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)


def load(
    name: str,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | bool | None = None,
    max_bytes: int | None = None,
) -> Network:
    """Download *name* if not already cached, and return a handle to it.

    The archive is kept as KONECT serves it (``Network.extract()`` reaches its
    README and meta files); the edge list is rewritten once beside it so that
    ``.path``, ``.edges()`` and ``.to_infomap()`` work as for every other source.
    Bipartite networks come back with unique ids: right-side nodes start at
    ``info(name).left_nodes + 1``. Signed and rating weights are kept in the file
    but not handed to Infomap, which accepts only non-negative weights.
    """
    dataset = info(name)
    artifact = dataset.primary_file
    directory = _dataset_dir(SOURCE, dataset.name)
    archive_path = directory / artifact.name
    edge_file = directory / f"{dataset.name}.tsv"
    _download_file(artifact.url, archive_path, force=force, progress=progress, max_bytes=max_bytes)
    if force:
        edge_file.unlink(missing_ok=True)
    if not edge_file.is_file():
        _write_edge_file(dataset, archive_path, edge_file)
    return Network(
        dataset=dataset,
        file=artifact,
        archive_path=archive_path,
        source=SOURCE,
        edge_file=edge_file,
        weighted_links=dataset.infomap_weights,
    )


def is_cached(name: str) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    return (_dataset_dir(SOURCE, dataset.name) / dataset.primary_file.name).is_file()
