"""Access to KONECT, the Koblenz Network Collection (konect.cc).

KONECT, curated by Jérôme Kunegis, describes every network in one fixed
vocabulary: its format (undirected, directed, bipartite), its edge type
(unweighted, multigraph, positive or signed weights), timestamps, and whether it
contains loops. The loop check is what no other source here reports, and it is
what a benchmark of how community detection treats self-links needs::

    from netgamut import konect

    konect.search(loops=True, max_nodes=10_000)
    konect.search(signed=True)
    konect.search(bipartite=True, domain="Economic")

The catalog is KONECT's downloadable networks minus its per-language Wikimedia
series (about a thousand edit, hyperlink and talk networks, one per language
edition), of which only the English editions are kept.

A few networks ship per-node attributes -- names, sexes, roles -- as ``ent.*``
files in their archive. No KONECT page says which, so the catalog build lists
every archive up to 100 MB; ``search(node_metadata=True)`` finds those that do,
and :meth:`Network.nodes` reads them. The larger archives are unlisted, and
answer ``None``.

Netzschleuder imported a third of KONECT, and some of it is SNAP's. Such a
network is still listed here, but the cross-source :func:`netgamut.search`,
:func:`netgamut.domains` and :func:`netgamut.load` use the other copy; see
:func:`duplicates`.

Please cite the networks you use; see ``Dataset.citations``.
"""

from __future__ import annotations

import itertools
import os
import tarfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .._core.cache import cache_dir, cache_info, clear_cache, file_lock, set_cache_dir
from .._core.cache import dataset_dir as _dataset_dir
from .._core.columns import Columns
from .._core.domains import DOMAINS
from .._core.download import DownloadError
from .._core.download import download as _download_file
from .._core.models import DataFile
from .._core.network import Network
from .._core.nodes import NodeTable
from . import _ent
from ._catalog import Catalog, build_catalog, load_catalog, refresh
from ._copies import Pair
from ._models import EDGE_TYPES, FORMATS, SITE_URL, Dataset, is_edge_list_header

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
    "Pair",
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
    "node_names",
    "node_table",
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
    node_metadata: bool | None = None,
    max_nodes: int | None = None,
    min_nodes: int | None = None,
    max_edges: int | None = None,
    min_edges: int | None = None,
) -> tuple[Dataset, ...]:
    """Find networks by domain, KONECT category, size or structural property.

    ``loops=True`` finds networks with at least one self-link, by KONECT's own
    check. ``node_metadata=True`` finds those whose archive ships per-node
    attributes, which :func:`node_table` reads. Networks with no answer -- no
    loop check reported, an archive too large to have been listed -- are
    excluded from either filter.
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
        node_metadata=node_metadata,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        max_edges=max_edges,
        min_edges=min_edges,
    )


# -- overlap with the other sources -----------------------------------------
_overlap: tuple[tuple[int, int, int], dict[str, Pair], dict[str, Dataset]] | None = None


def _compute_overlap() -> tuple[dict[str, Pair], dict[str, Dataset]]:
    """The catalog's copy verdicts whose other network is still listed.

    Returns KONECT name -> the pair cross-source queries use, and
    ``"<source>/<name>"`` -> the KONECT network with the same edge list.
    """
    global _overlap
    from .. import netzschleuder, snap

    nz, sn, kc = netzschleuder.catalog(), snap.catalog(), catalog()
    key = (id(nz), id(sn), id(kc))
    if _overlap is not None and _overlap[0] == key:
        return _overlap[1], _overlap[2]

    listed = {f"netzschleuder/{item.name}" for item in nz.datasets}
    listed.update(f"snap/{item.name}" for item in sn.datasets)

    def present(pair: Pair) -> bool:
        return pair.name in kc and pair.of in listed

    duplicate: dict[str, Pair] = {}
    twins: dict[str, Dataset] = {}
    for pair in kc.copies:
        if not pair.is_copy or not present(pair):
            continue
        # The first copy is the preferred one: Netzschleuder's before SNAP's.
        duplicate.setdefault(pair.name, pair)
        if pair.verdict == "same-edges":
            twins.setdefault(pair.of, kc[pair.name])

    _overlap = (key, duplicate, twins)
    return duplicate, twins


def duplicates() -> dict[str, str]:
    """KONECT name -> ``"<source>/<name>"`` of the copy cross-source queries use.

    A KONECT network is a copy only on evidence recorded when the catalog was
    built, never on its name or its size alone: KONECT's ``epinions`` and
    ``twitter`` are different networks from Netzschleuder's, and
    ``brunson_southern-women`` shares its 10 nodes and 14 edges with an
    unrelated one. The evidence is either of:

    * ``imported``: Netzschleuder's entry names the KONECT network as its
      upstream and holds that one network, not a family of subgraphs.
    * ``same-edges``: the two edge lists were downloaded and found equal, as they
      stand or up to a relabelling of the nodes.

    A pair the build has not compared, usually one above its size cap, stays
    two separate networks; so do near-copies -- a different snapshot, the
    largest component only. ``catalog().copies`` lists every pair considered,
    with its verdict and how it was reached.
    """
    return {name: pair.of for name, pair in _compute_overlap()[0].items()}


def loops_for(key: str) -> bool | None:
    """KONECT's loop check for another source's network, if KONECT holds it.

    *key* is ``"<source>/<name>"``. Only a pair whose edge lists were compared
    and found equal counts, so the check describes the same edge list.
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
    if dataset.node_attributes is None:
        lines.append("Node attrs  not known (archive too large to list)")
    else:
        lines.append(f"Node attrs  {', '.join(dataset.node_attributes) or 'none'}")
    if dataset.edge_attributes:
        lines.append(f"Edge attrs  {', '.join(dataset.edge_attributes)}")
    copy = _compute_overlap()[0].get(dataset.name)
    if copy is not None:
        lines.append(f"Also as     {copy.of} ({copy.verdict}; used by netgamut.search/load)")
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


def _columns(dataset: Dataset) -> Columns:
    """KONECT's fixed layout: ``source target [weight [time]]``.

    The weight is read only where Infomap accepts it -- not for signed or
    arbitrary-real edge types -- and the fourth field is the timestamp of a
    temporal network.
    """
    return Columns(
        weight=2 if dataset.infomap_weights else None,
        time=3 if dataset.temporal else None,
        names=("source", "target", "weight", "time") if dataset.temporal else (),
    )


def _write_edge_file(dataset: Dataset, archive_path: Path, target: Path) -> None:
    with file_lock(target):
        if target.is_file():
            return
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        try:
            # Stream mode: one pass over the compressed archive, flat memory.
            with tarfile.open(archive_path, "r|*") as archive:
                for member in archive:
                    # The edge list is known by its header, not its name: the
                    # older layout calls it rel.<relation>, beside per-edge
                    # attribute files that are also rel.* but have no header.
                    if not (
                        member.isfile() and Path(member.name).name.startswith(("out.", "rel."))
                    ):
                        continue
                    handle = archive.extractfile(member)
                    assert handle is not None
                    first = handle.readline()
                    if not is_edge_list_header(first):
                        continue
                    raw_lines = itertools.chain([first], handle)
                    lines = (raw.decode("utf-8", "replace") for raw in raw_lines)
                    rewritten = _normalise(
                        lines, bipartite=dataset.bipartite, left_nodes=dataset.left_nodes
                    )
                    with partial.open("w", encoding="utf-8") as out:
                        out.writelines(rewritten)
                    break
                else:
                    raise LookupError(
                        f"{archive_path.name} holds no edge list (no file opens with "
                        "a '% sym|asym|bip' header)."
                    )
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
        columns=_columns(dataset),
    )


def _entity_sides(directory: Path) -> list[str]:
    """The entity names from KONECT's meta file, left side first."""
    for meta in directory.rglob("meta.*"):
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "entity-names":
                return [name.strip() for name in value.split(",") if name.strip()]
    return []


def node_table(network: Network) -> NodeTable:
    """Read a KONECT network's per-node attributes into the shared shape.

    KONECT keeps them as ``ent.*`` files inside the archive :func:`load` already
    fetched, so this costs an extraction and no network access. The three file
    shapes it uses -- one attribute per file, a profile table, a list of labels --
    come out as named columns, the ones ``info(name).node_attributes`` lists. Node
    ids are the edge list's, right side of a bipartite network shifted included,
    so the table and :meth:`Network.edges` agree.

    Few networks ship such files; ``search(node_metadata=True)`` finds them.

    Raises:
        LookupError: The archive ships no per-node attributes.
    """
    dataset = network.dataset
    assert isinstance(dataset, Dataset)
    directory = network.extract()
    files = sorted(path for path in directory.rglob("ent.*") if path.is_file())
    if not files:
        raise LookupError(f"{network.name!r} ships no per-node attributes (no ent.* files).")
    values = _ent.read(
        files,
        sides=_entity_sides(directory),
        left_nodes=dataset.left_nodes,
        bipartite=dataset.bipartite,
    )
    return NodeTable(values=values, origin=f"{SOURCE}/{network.name}/ent.*")


def node_names(network: Network) -> dict[Any, str]:
    """Each node's name from the archive's ``ent.*`` files, or ``{}`` without one.

    :meth:`Network.to_pajek` writes these as the ``*Vertices`` labels. The column is
    the one :meth:`NodeTable.labels` picks.
    """
    try:
        return node_table(network).labels()
    except LookupError:
        return {}


def is_cached(name: str) -> bool:
    """Whether *name* is already downloaded, without fetching anything."""
    try:
        dataset = info(name)
    except KeyError:
        return False
    return (_dataset_dir(SOURCE, dataset.name) / dataset.primary_file.name).is_file()
