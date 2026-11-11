"""The handle returned by every source's ``load()``.

A :class:`Network` is a thin wrapper around a cached file. It does not read the
data into memory when you create it, which is what lets the same API serve
``ca-GrQc`` (14k edges) and ``com-Friendster`` (1.8 billion edges). Conversions
to networkx and Infomap are lazy imports, so neither is needed to install or
use the package.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from . import columns as _columns
from . import formats
from .cache import file_lock, write_atomic
from .columns import Columns
from .download import extract_gzip
from .models import DataFile, DatasetLike
from .nodes import NodeTable

if TYPE_CHECKING:
    from .in_memory import InMemoryNetwork
    from .multilayer import MultilayerNetwork

__all__ = ["Columns", "Network"]

_COMMENT_PREFIXES = ("#", "%")


class ArchiveError(ValueError):
    """A multi-file bundle was read as one network; extract it and open a member."""


# SNAP files are mostly tab- or space-separated, but its Bitcoin trust networks
# are `source,target,rating,time` and every Netzschleuder edge list is CSV.
# Splitting CSV on whitespace can silently yield an empty network, so the
# delimiter is detected rather than assumed.
_CANDIDATE_DELIMITERS = (",", ";")

# Enough lines to tell a real delimiter from an incidental character.
_SAMPLE_LINES = 25

_LEADING_NUMBER = _columns.LEADING_NUMBER


def _detect_delimiter(samples: Sequence[str]) -> str | None:
    """Return the delimiter used by *samples*, or ``None`` for whitespace.

    A candidate wins only if it splits *every* sampled line into the same number
    of fields, at least two. Consistency is what makes this reliable: a
    whitespace-separated file with commas inside labels splits inconsistently on
    comma and so falls through to whitespace, while a genuine CSV whose fields
    contain spaces (``openflights`` has an aircraft column reading ``DH8 DHT``)
    is still recognised -- which a whitespace-first guess gets wrong.
    """
    if not samples:
        return None
    for delimiter in _CANDIDATE_DELIMITERS:
        counts = {len(line.split(delimiter)) for line in samples}
        if len(counts) == 1 and counts.pop() >= 2:
            return delimiter
    return None


@dataclass(frozen=True)
class Network:
    """A downloaded network on local disk."""

    #: Full metadata for the dataset this file belongs to.
    dataset: DatasetLike
    #: The specific artifact that was downloaded.
    file: DataFile
    #: The downloaded artifact exactly as the source serves it, compression
    #: included.
    archive_path: Path
    #: For sources that ship a bundle, the member holding the edge list --
    #: Netzschleuder's ``csv.zip`` contains ``edges.csv`` alongside node and
    #: graph properties. ``None`` when the download is a single file.
    member: str | None = None
    #: Which ``netgamut.<source>`` produced this handle. :meth:`nodes` uses it to
    #: reach the reader that understands this source's metadata layout, since no
    #: two sources keep node attributes in the same place.
    source: str = ""
    #: For sources whose download needs rewriting before any tool can read it,
    #: the rewritten edge list. KONECT's files open with a ``%`` header Infomap
    #: rejects and number the two sides of a bipartite network independently, so
    #: ``netgamut.konect`` writes a clean copy and keeps the archive beside it.
    edge_file: Path | None = None
    #: What the fields beyond source and target hold -- weight, time, layer --
    #: as the source declares it from its metadata. Everything that reads links
    #: follows it, so a timestamp or a sign is never mistaken for a weight.
    #: Replace it with :func:`dataclasses.replace` to read a column differently.
    columns: Columns = Columns()
    #: The labels of the nodes a :meth:`largest_component` view keeps, as they
    #: appear in the file; ``None`` for the whole network.
    component: frozenset[str] | None = None

    @property
    def weighted_links(self) -> bool:
        """Whether links carry a weight column Infomap accepts."""
        return self.columns.weight is not None

    # -- paths -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def _stem(self) -> str:
        """The default name of what this handle writes or builds.

        A :meth:`largest_component` view adds ``-lcc``, so its files never take
        the whole network's name. :attr:`name` stays the dataset's, which the
        sources look datasets up by.
        """
        return self.name if self.component is None else f"{self.name}-lcc"

    @property
    def is_bundle(self) -> bool:
        """Whether the download is a multi-file archive on disk."""
        lowered = self.archive_path.name.lower()
        return lowered.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip"))

    @property
    def is_archive(self) -> bool:
        """Whether reading needs :meth:`extract` rather than :attr:`path`.

        False for a bundle with a designated *member*: that member is the
        network, so reading works directly even though the download is an
        archive. Use :attr:`is_bundle` to ask about the file itself.
        """
        return self.is_bundle and self.member is None and self.edge_file is None

    @property
    def path(self) -> Path:
        """A path you can hand to any tool that reads a text file.

        Gzipped files are decompressed once, next to the archive. A designated
        *member* is extracted from its bundle the same way. For a bundle with no
        member there is no single text file, so this returns the archive; use
        :meth:`extract` instead.
        """
        if self.edge_file is not None:
            return self.edge_file
        if self.member is not None:
            return self._extract_member()
        if self.is_archive:
            return self.archive_path
        if self.archive_path.name.lower().endswith(".gz"):
            target = self.archive_path.with_suffix("")
            return extract_gzip(self.archive_path, target)
        return self.archive_path

    def _extract_member(self) -> Path:
        """Pull the designated member out of its bundle, once.

        A member may itself be compressed -- OGB ships ``raw/edge.csv.gz`` inside a
        ``.zip`` -- so a gzipped member is decompressed on the way out and the ``.gz``
        dropped from the extracted name. Callers then get a plain text file whatever
        the source nested.
        """
        assert self.member is not None
        stem = self.archive_path.name.split(".")[0]
        name = Path(self.member).name
        gzipped = name.lower().endswith(".gz")
        target = self.archive_path.parent / f"{stem}-{name[:-3] if gzipped else name}"
        if target.is_file():
            return target
        with zipfile.ZipFile(self.archive_path) as archive:
            payload = archive.read(self.member)
        if gzipped:
            payload = gzip.decompress(payload)
        write_atomic(target, payload)
        return target

    @property
    def size_bytes(self) -> int:
        return self.archive_path.stat().st_size if self.archive_path.is_file() else 0

    def extract(self, destination: Path | None = None) -> Path:
        """Unpack a ``.tar.gz`` or ``.zip`` bundle and return the directory.

        Works even when a *member* has been designated as the network, so a
        Netzschleuder bundle's node and graph properties stay reachable.

        Extraction lands in a scratch directory and is published with a
        rename, so a crashed or interrupted run cannot leave a partial
        directory that a later call would mistake for the real thing.
        """
        if not self.is_bundle:
            raise ValueError(
                f"{self.file.name} is a single file, not an archive. Use .path instead."
            )
        target = (
            Path(destination)
            if destination
            else self.archive_path.parent / (self.archive_path.name.split(".")[0] + ".extracted")
        )
        if target.is_dir() and any(target.iterdir()):
            return target
        with file_lock(target):
            # Another process may have finished while we waited for the lock.
            if target.is_dir() and any(target.iterdir()):
                return target
            partial = target.with_name(f".{target.name}.{os.getpid()}.part")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            try:
                if zipfile.is_zipfile(self.archive_path):
                    with zipfile.ZipFile(self.archive_path) as zip_archive:
                        # ZipFile.extract already strips absolute paths and ".."
                        # components from member names.
                        zip_archive.extractall(partial)
                else:
                    with tarfile.open(self.archive_path) as tar_archive:
                        # filter="data" refuses members that would escape the target.
                        tar_archive.extractall(partial, filter="data")
                if target.is_dir():
                    target.rmdir()  # empty, or we would have returned above
                os.replace(partial, target)
            finally:
                shutil.rmtree(partial, ignore_errors=True)
        return target

    # -- reading ---------------------------------------------------------
    def open(self) -> IO[str]:
        """Open the network as a text stream, decompressing on the fly."""
        if self.member is not None or self.edge_file is not None:
            return self.path.open(encoding="utf-8", errors="replace")
        if self.is_archive:
            raise ArchiveError(
                f"{self.file.name} is an archive. Call .extract() and open a member."
            )
        if self.archive_path.name.lower().endswith(".gz"):
            return io.TextIOWrapper(
                gzip.open(self.archive_path, "rb"), encoding="utf-8", errors="replace"
            )
        return self.archive_path.open(encoding="utf-8", errors="replace")

    @property
    def header(self) -> tuple[str, ...]:
        """The leading comment lines, which SNAP uses to describe the file."""
        lines: list[str] = []
        with self.open() as stream:
            for line in stream:
                if not line.startswith(_COMMENT_PREFIXES):
                    break
                lines.append(line.lstrip("#% ").rstrip())
        return tuple(lines)

    @property
    def delimiter(self) -> str | None:
        """The field separator this file uses, or ``None`` for whitespace.

        The one :attr:`columns` declares, else one detected from the first lines.
        """
        if self.columns.delimiter is not None:
            return self.columns.delimiter
        return _detect_delimiter(self._samples())

    def _samples(self) -> list[str]:
        """The first data lines, for sniffing the delimiter and the row width."""
        samples: list[str] = []
        with self.open() as stream:
            for line in stream:
                stripped = line.strip()
                if stripped and not stripped.startswith(_COMMENT_PREFIXES):
                    samples.append(stripped)
                    if len(samples) >= _SAMPLE_LINES:
                        break
        return samples

    def rows(self, *, delimiter: str | None = None) -> Iterator[tuple[str, ...]]:
        """Yield every non-comment line, split into its fields.

        Use this for the datasets that carry more than two columns, such as
        timestamps in the temporal category or signs in the signed category.

        The separator is the one :attr:`columns` declares, or else detected
        from the first data lines, since a few SNAP files are CSV rather than
        whitespace-separated. Pass *delimiter* to override that. A header row
        that :attr:`columns` declares is left out.

        In a :meth:`largest_component` view, only the rows whose source node is
        in the component -- and so their target too.
        """
        rows = self._file_rows(delimiter)
        if self.component is None:
            yield from rows
            return
        keep = self.component
        for fields in rows:
            if fields[0] in keep:
                yield fields

    def _file_rows(self, delimiter: str | None) -> Iterator[tuple[str, ...]]:
        """Every row of the file, as :meth:`rows` describes, whatever the view."""
        sniffed = delimiter if delimiter is not None else self.delimiter
        skip = self.columns.header_row
        with self.open() as stream:
            if sniffed is None:
                for line in stream:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                        continue
                    fields = stripped.split()
                    if skip:
                        skip = False
                    elif fields:
                        yield tuple(fields)
                return
            # csv.reader honours quoting, so a label containing the delimiter,
            # or a quoted field running over a line break, stays one field.
            _columns.allow_long_fields()
            reader = csv.reader(stream, delimiter=sniffed, escapechar=self.columns.escapechar)
            for record in reader:
                if not record or record[0].lstrip().startswith(_COMMENT_PREFIXES):
                    continue
                fields = [field.strip() for field in record]
                if not any(fields):
                    continue
                if skip:
                    skip = False
                else:
                    yield tuple(fields)

    def nodes(self) -> NodeTable:
        """Read this network's per-node metadata, in the shape every source uses.

        Where the attributes live differs completely by source -- a separate SNAP
        artifact, a member of a Netzschleuder bundle, OGB's ``raw/`` and ``mapping/``
        folders, a materialised PyG table -- so the reading is delegated to the
        source. What comes back is always a :class:`~netgamut._core.nodes.NodeTable`::

            nodes = netgamut.load("ogbn-arxiv").nodes()
            nodes.columns                       # ('arxiv_category', 'year', 'paper')
            nodes["arxiv_category"]             # {node: 'arxiv cs na', ...}
            nodes.partition("arxiv_category")   # {node: 0..k-1}, for AMI/NMI/D

        Sources that keep metadata in a separate artifact download it here, on the
        same cache-once terms as :func:`load`.

        Node metadata is not ground truth for community structure: a partition that
        disagrees with an attribute is not thereby wrong. See
        :mod:`netgamut._core.nodes`.

        Raises:
            LookupError: The source publishes no per-node metadata, or this dataset
                ships none.
        """
        import importlib

        if not self.source:
            raise LookupError(
                f"{self.dataset.name!r} was built without a source tag, so its node "
                "metadata cannot be located."
            )
        module = importlib.import_module(f"netgamut.{self.source}")
        reader = getattr(module, "node_table", None)
        if reader is None:
            raise LookupError(f"netgamut.{self.source} publishes no per-node metadata.")
        table: NodeTable = reader(self)
        if self.component is None:
            return table
        keep = self.component
        return replace(
            table,
            values={
                column: {node: value for node, value in values.items() if str(node) in keep}
                for column, values in table.values.items()
            },
        )

    def communities(self, *, cast: Callable[[str], Any] | None = None) -> Iterator[tuple[Any, ...]]:
        """Stream communities from a SNAP ground-truth file.

        Each line of a ``*.cmty.txt`` file is one community, listed as
        tab-separated node ids. Communities overlap, so a node may appear in
        several. Use this as the reference partition when scoring a
        community-detection run.
        """
        for fields in self.rows():
            yield tuple(fields) if cast is None else tuple(cast(field) for field in fields)

    def edges(self, *, cast: Callable[[str], Any] | None = None) -> Iterator[tuple[Any, Any]]:
        """Stream ``(source, target)`` pairs.

        Values are strings unless you pass *cast*; ``cast=int`` is usual, since
        most SNAP files use integer node ids. Streaming means memory stays flat
        no matter how large the file is.
        """
        for fields in self.rows():
            if len(fields) < 2:
                continue
            if cast is None:
                yield fields[0], fields[1]
            else:
                yield cast(fields[0]), cast(fields[1])

    def largest_component(self) -> Network:
        """The largest connected component, as a view on the same cached file.

        One pass over the links finds the components, ignoring direction, so
        for a directed network this is the largest *weakly* connected component.
        Memory grows with the number of nodes, never the links. A component's
        size is its number of nodes; a tie goes to the one met first in the file.

        The view is a :class:`Network` like any other. :meth:`rows`,
        :meth:`edges`, every export, :meth:`snapshots` and :meth:`to_multilayer`
        see only the links within the component, and :meth:`nodes` only its
        nodes. Files it writes are named ``<name>-lcc``. :attr:`component` holds
        the node labels, to restrict a reference partition from elsewhere::

            lcc = netgamut.load("com-Amazon").largest_component()
            len(lcc.component)
            im = lcc.to_infomap()

        :attr:`path`, :meth:`open` and :attr:`header` still give the whole
        download, since that is what a tool reading the file would see. For a
        temporal or multilayer network the component is that of all links
        together, across times and layers.

        There is no strongly connected variant: finding those components needs
        every link in memory, and in a nearly acyclic network, such as one of
        citations, the largest is a sliver of the whole.

        Raises:
            ValueError: The network has no links.
        """
        return replace(self, component=_largest_component(self.edges(), self.name))

    # -- conversions -----------------------------------------------------
    def to_networkx(
        self, *, directed: bool | None = None, cast: Callable[[str], Any] | None = int
    ) -> Any:
        """Build a networkx graph.

        Direction defaults to what SNAP reports for the dataset. Requires the
        ``networkx`` extra::

            pip install "netgamut[networkx]"

        Be deliberate here: a networkx graph holds every node and edge in
        memory, so check ``dataset.edges`` before calling this on a large one.
        """
        try:
            import networkx
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                'to_networkx() needs networkx. Install it with `pip install "netgamut[networkx]"`.'
            ) from error

        if directed is None:
            directed = bool(self.dataset.directed)
        graph = networkx.DiGraph() if directed else networkx.Graph()
        graph.name = self._stem

        try:
            graph.add_edges_from(self.edges(cast=cast))
        except ValueError:
            # Node ids are not integers after all; keep them as strings.
            graph.clear()
            graph.add_edges_from(self.edges(cast=None))
        return graph

    @property
    def partition(self) -> None:
        """Always ``None``: a downloaded network has no planted partition.

        Present so code written for generated networks runs on downloaded ones
        too. Per-node metadata, which is not ground truth, is :meth:`nodes`.
        """
        return None

    def _links(self) -> Iterator[tuple[str, str, float | None]]:
        """Stream ``(source, target, weight)`` exactly as :meth:`to_infomap` reads them.

        The weight comes from :attr:`columns` alone -- never from whatever
        happens to be the third field -- read the way Infomap's parser reads a
        number, so a written file loads the same network.
        """
        column = self.columns.weight
        for fields in self.rows():
            if len(fields) < 2:
                continue
            weight = None if column is None else _columns.weight_of(fields, column)
            yield fields[0], fields[1], weight

    def _infomap_reads_directly(self) -> bool:
        """Whether Infomap's own parser reads this file as :attr:`columns` says.

        It splits on whitespace and takes any third field as the weight, so the
        file must be whitespace-separated and hold no field past the declared
        weight -- on every line, not only the sampled ones, or a stray third
        field deep in the file would be a weight here and nowhere else. The
        check is a plain pass over the lines, about a tenth of Infomap's read.
        """
        if self.columns.header_row or self.columns.layer is not None:
            return False
        if self.component is not None:  # the file holds links outside the view
            return False
        if self.delimiter is not None:
            return False
        width = 3 if self.columns.weight == 2 else 2
        return all(len(fields) <= width for fields in self.rows())

    def _typed_links(self, cast: Callable[[str], Any]) -> Iterator[formats.Link]:
        for source, target, weight in self._links():
            yield cast(source), cast(target), weight

    def to_pajek(
        self,
        directory: str | Path,
        *,
        name: str | None = None,
        directed: bool | None = None,
        names: Mapping[Any, str] | None = None,
    ) -> Path:
        """Write ``<directory>/<name>.net`` in Pajek format and return its path.

        *name* defaults to the dataset's name and *directed* to the direction
        the source reports. Integer node ids are kept; any other labels are
        numbered ``1..N`` and kept as ``*Vertices`` names. The file is
        streamed in two passes -- nodes, then links -- so memory holds the
        node set but never the links.

        *names* maps a node id to the name shown on its ``*Vertices`` line. By
        default it is the source's own names, where they come in the download
        already made -- Netzschleuder's ``nodes.csv``, KONECT's ``ent.*`` files;
        see :meth:`NodeTable.labels`. A node without one shows its id. Pass
        ``{}`` to label every node by its id.
        """
        if directed is None:
            directed = bool(self.dataset.directed)
        labels = dict.fromkeys(label for u, v, _ in self._links() for label in (u, v))
        cast: Callable[[str], Any] = (
            int if all(_columns.is_int_label(label) for label in labels) else str
        )
        if names is None:
            names = self._source_names(cast)
        target = formats.output_path(directory, name or self._stem, ".net")
        return formats.write_pajek(
            target,
            nodes=(cast(label) for label in labels),
            links=lambda: self._typed_links(cast),
            directed=directed,
            names=names,
            comments=formats.describe(self._stem, None),
        )

    def _source_names(self, cast: Callable[[str], Any]) -> dict[Any, str]:
        """The source's node names, keyed the way :meth:`to_pajek` keys nodes.

        Only sources whose names arrive with the links publish a ``node_names``
        reader, so this never starts a download of its own.
        """
        import importlib

        if not self.source:
            return {}
        reader = getattr(importlib.import_module(f"netgamut.{self.source}"), "node_names", None)
        if reader is None:
            return {}
        found: dict[Any, str] = reader(self)
        keep = self.component
        return {
            cast(str(node)): label
            for node, label in found.items()
            if (cast is str or _columns.is_int_label(str(node)))
            and (keep is None or str(node) in keep)
        }

    def to_link_list(
        self, directory: str | Path, *, name: str | None = None, directed: bool | None = None
    ) -> Path:
        """Write ``<directory>/<name>.txt`` as a link list and return its path.

        Streams in one pass. A ``# directed`` or ``# undirected`` comment
        records *directed*, which defaults to what the source reports.

        Raises:
            ValueError: A node id is not a non-negative integer; use
                :meth:`to_pajek`, which keeps labels as names.
        """
        if directed is None:
            directed = bool(self.dataset.directed)

        def cast(label: str) -> Any:
            return int(label) if _columns.is_int_label(label) else label

        target = formats.output_path(directory, name or self._stem, ".txt")
        return formats.write_link_list(
            target,
            links=self._typed_links(cast),
            directed=directed,
            comments=formats.describe(self._stem, None),
        )

    def to_infomap(self, *, directed: bool | None = None, **kwargs: Any) -> Any:
        """Build an :class:`infomap.Infomap` instance with this network loaded.

        Direction defaults to what SNAP reports. Extra keyword arguments are
        passed to the ``Infomap`` constructor. Requires the ``infomap`` extra::

            pip install "netgamut[infomap]"
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                'to_infomap() needs infomap. Install it with `pip install "netgamut[infomap]"`.'
            ) from error

        if directed is None:
            directed = bool(self.dataset.directed)
        kwargs.setdefault("directed", directed)
        instance = Infomap(**kwargs)

        if self._infomap_reads_directly():
            # Fast path: let Infomap's own parser read the file directly.
            instance.read_file(str(self.path))
        else:
            # CSV, or extra fields Infomap would misread -- a timestamp taken
            # for a weight, a sign it refuses -- so feed the links in, with the
            # weight from the column the source declared.
            instance.add_links(
                (int(u), int(v), 1.0 if w is None else w) for u, v, w in self._links()
            )
        return instance

    # -- time and layers ---------------------------------------------------
    def _require(self, role: str) -> int:
        column: int | None = getattr(self.columns, role)
        if column is None:
            raise LookupError(
                f"{self.name!r} has no {role} column ({self.columns}). If one of its "
                "fields is one, set it with dataclasses.replace(net, columns=Columns(...))."
            )
        return column

    @cached_property
    def times(self) -> Any:
        """The time of every link, as a numpy array in file order.

        One entry per link row, aligned with :meth:`edges`. Integers stay
        ``int64`` (Unix timestamps usually), fractional times are ``float64``
        and ISO dates ``datetime64[s]``. Read once and kept, so look at it
        before choosing how to bin::

            t = net.times
            t.min(), t.max()
            numpy.histogram(t, bins="auto")

        Needs numpy.

        Raises:
            LookupError: The source declares no time column for this file.
        """
        column = self._require("time")
        table = _columns.read_table(self.rows(), self.columns, time=column, nodes=False)
        return table.times

    def time_bins(self, bins: Any = 10, range: tuple[Any, Any] | None = None) -> Any:
        """The bin edges :meth:`snapshots` would use, from :attr:`times`.

        *bins* and *range* mean what they mean to
        :func:`numpy.histogram_bin_edges`: a number of equal-width bins, a
        sequence of edges, or an estimator such as ``"auto"`` or ``"fd"``.
        """
        edges, _ = _columns.assign_bins(self.times, bins, range)
        return edges

    def snapshots(
        self,
        bins: Any = 10,
        range: tuple[Any, Any] | None = None,
        *,
        directed: bool | None = None,
    ) -> Iterator[InMemoryNetwork]:
        """Yield one first-order network per time bin, built only when reached.

        Bins follow :meth:`time_bins`; bin ``i`` holds the links with
        ``edges[i] <= t < edges[i + 1]`` (the last bin also ``t == edges[-1]``),
        and links outside every bin are left out. Every bin yields, empty ones
        included, so the position in the iteration is the bin. A link's weight
        is how many times it occurs in the bin -- the sum of its weights if the
        file has a weight column. Each snapshot's ``info`` holds its ``bin``,
        ``start``, ``end`` and the number of ``events``.

        The file is read once, on the first ``next()``; each snapshot is
        assembled from its rows when it is reached.

        Raises:
            LookupError: The source declares no time column for this file.
        """
        from .in_memory import InMemoryNetwork

        column = self._require("time")
        if directed is None:
            directed = bool(self.dataset.directed)
        table = _columns.read_table(self.rows(), self.columns, time=column)
        edges, index = _columns.assign_bins(table.times, bins, range)
        for b, rows in _columns.groups(index, len(edges) - 1):
            links = _aggregate(table, rows, directed)
            yield InMemoryNetwork(
                name=f"{self._stem}-t{b}",
                nodes=tuple(dict.fromkeys(node for pair in links for node in pair)),
                links=links,
                directed=directed,
                weighted=True,
                info={
                    "source": self.name,
                    "bin": b,
                    "start": edges[b],
                    "end": edges[b + 1],
                    "events": len(rows),
                },
            )

    def to_multilayer(
        self,
        *,
        bins: Any = None,
        range: tuple[Any, Any] | None = None,
        layer: str | None = None,
        directed: bool | None = None,
    ) -> MultilayerNetwork:
        """Build a :class:`~netgamut.MultilayerNetwork` from a layer or time column.

        With *bins*, the layers are time bins, exactly the :meth:`snapshots`:
        layer ``i`` is bin ``i``, empty ones kept so layer ids stay time order,
        and ``info["bin_edges"]`` records the edges. Without, the layers come
        from the source's layer column, or from the column named *layer* -- any
        categorical field, e.g. a type of interaction. Link weights are counts
        of occurrences, summed weights where the file has a weight column.

        There are no inter-layer links: give Infomap a relax rate, and for time
        layers a relax limit so flow moves only between neighbouring bins::

            ml.to_infomap(multilayer_relax_rate=0.15, multilayer_relax_limit=1)

        The source's node names come along, as :meth:`to_pajek` writes them.

        Raises:
            LookupError: No time column (with *bins*), or no layer column.
            ValueError: Both *bins* and *layer* given.
        """
        from .multilayer import MultilayerNetwork

        if bins is not None and layer is not None:
            raise ValueError("give bins (time layers) or layer (a layer column), not both.")
        if directed is None:
            directed = bool(self.dataset.directed)
        intra: dict[Any, dict[tuple[Any, Any], float]] = {}
        if bins is not None:
            column = self._require("time")
            table = _columns.read_table(self.rows(), self.columns, time=column)
            edges, index = _columns.assign_bins(table.times, bins, range)
            for b, rows in _columns.groups(index, len(edges) - 1):
                intra[b] = _aggregate(table, rows, directed)
            info: dict[str, Any] = {"layers": "time", "bin_edges": edges}
        else:
            column = self.columns.find(layer) if layer is not None else self._require("layer")
            table = _columns.read_table(self.rows(), self.columns, layer=column)
            assert table.layers is not None
            by_layer: dict[Any, list[int]] = {}
            for row, label in enumerate(table.layers):
                if label is not None:  # a row without a layer belongs to none
                    by_layer.setdefault(label, []).append(row)
            for label in formats.ordered(by_layer):
                intra[label] = _aggregate(table, by_layer[label], directed)
            name = self.columns.names[column] if column < len(self.columns.names) else column
            info = {"layers": name}
        everyone = {node for links in intra.values() for pair in links for node in pair}
        cast: Callable[[str], Any] = int if all(isinstance(node, int) for node in everyone) else str
        return MultilayerNetwork(
            name=self._stem,
            intra=intra,
            directed=directed,
            weighted=any(w != 1.0 for links in intra.values() for w in links.values()),
            names=self._source_names(cast),
            info={"source": self.name, **info},
        )

    def __repr__(self) -> str:
        return f"<Network {self._stem!r} file={self.file.name!r} at {self.archive_path}>"

    def __str__(self) -> str:
        return f"{self.dataset} -> {self.path}"


def _aggregate(
    table: _columns.Table, rows: Iterable[int], directed: bool
) -> dict[tuple[Any, Any], float]:
    """Sum the links of *rows*: occurrences, or weights where there are any.

    An undirected link counts the same whichever way round a row lists it.
    """
    links: dict[tuple[Any, Any], float] = {}
    weights = table.weights
    for row in rows:
        u, v = table.sources[row], table.targets[row]
        key = (u, v)
        if not directed and key not in links and (v, u) in links:
            key = (v, u)
        links[key] = links.get(key, 0.0) + (1.0 if weights is None else float(weights[row]))
    return links


def _largest_component(links: Iterable[tuple[str, str]], name: str) -> frozenset[str]:
    """The node labels of the largest component of *links*, direction ignored.

    Union-find by size with path splitting, so memory holds a parent and a size
    per node and nothing per link. Of equally large components, the one whose
    first node comes first in *links* wins, so the answer does not depend on
    hashing.

    Raises:
        ValueError: *links* is empty; *name* says which network.
    """
    parent: dict[str, str] = {}
    size: dict[str, int] = {}  # roots only

    def find(node: str) -> str:
        if node not in parent:
            parent[node] = node
            size[node] = 1
            return node
        while (up := parent[node]) != node:
            parent[node] = parent[up]
            node = up
        return node

    for u, v in links:
        a, b = find(u), find(v)
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size.pop(b)
    if not size:
        raise ValueError(f"{name!r} has no links, so it has no largest component.")
    largest = max(size.values())
    # Path splitting only rewrites values, so iterating the keys while finding is safe.
    root = next(r for r in map(find, parent) if size[r] == largest)
    return frozenset(node for node in parent if find(node) == root)
