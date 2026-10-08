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
import re
import shutil
import tarfile
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from . import formats
from .cache import file_lock, write_atomic
from .download import extract_gzip
from .models import DataFile, DatasetLike
from .nodes import NodeTable

__all__ = ["Network"]

_COMMENT_PREFIXES = ("#", "%")

# SNAP files are mostly tab- or space-separated, but its Bitcoin trust networks
# are `source,target,rating,time` and every Netzschleuder edge list is CSV.
# Splitting CSV on whitespace can silently yield an empty network, so the
# delimiter is detected rather than assumed.
_CANDIDATE_DELIMITERS = (",", ";")

# Enough lines to tell a real delimiter from an incidental character.
_SAMPLE_LINES = 25

# The numeric prefix a C++ stream reads as a double -- what Infomap's own parser
# makes of a weight field -- so "1_000" is 1 here too, not Python's 1000.
_LEADING_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


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
    #: Which ``networks.<source>`` produced this handle. :meth:`nodes` uses it to
    #: reach the reader that understands this source's metadata layout, since no
    #: two sources keep node attributes in the same place.
    source: str = ""
    #: For sources whose download needs rewriting before any tool can read it,
    #: the rewritten edge list. KONECT's files open with a ``%`` header Infomap
    #: rejects and number the two sides of a bipartite network independently, so
    #: ``networks.konect`` writes a clean copy and keeps the archive beside it.
    edge_file: Path | None = None
    #: Whether a third column, where present, holds weights Infomap accepts.
    #: False for signed or arbitrary-real weights -- Infomap refuses a negative
    #: link weight -- so :meth:`to_infomap` reads the first two columns only.
    weighted_links: bool = True

    # -- paths -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.dataset.name

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
            raise ValueError(f"{self.file.name} is an archive. Call .extract() and open a member.")
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
        """The field separator this file uses, or ``None`` for whitespace."""
        samples: list[str] = []
        with self.open() as stream:
            for line in stream:
                stripped = line.strip()
                if stripped and not stripped.startswith(_COMMENT_PREFIXES):
                    samples.append(stripped)
                    if len(samples) >= _SAMPLE_LINES:
                        break
        return _detect_delimiter(samples)

    def rows(self, *, delimiter: str | None = None) -> Iterator[tuple[str, ...]]:
        """Yield every non-comment line, split into its fields.

        Use this for the datasets that carry more than two columns, such as
        timestamps in the temporal category or signs in the signed category.

        The separator is detected from the first data line, since a few SNAP
        files are CSV rather than whitespace-separated. Pass *delimiter* to
        override that.
        """
        sniffed = delimiter if delimiter is not None else self.delimiter
        with self.open() as stream:
            if sniffed is None:
                for line in stream:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                        continue
                    fields = stripped.split()
                    if fields:
                        yield tuple(fields)
                return
            # csv.reader honours quoting, so a label containing the delimiter
            # does not split into extra fields.
            for record in csv.reader(stream, delimiter=sniffed):
                if not record or record[0].lstrip().startswith(_COMMENT_PREFIXES):
                    continue
                fields = [field.strip() for field in record]
                if any(fields):
                    yield tuple(fields)

    def nodes(self) -> NodeTable:
        """Read this network's per-node metadata, in the shape every source uses.

        Where the attributes live differs completely by source -- a separate SNAP
        artifact, a member of a Netzschleuder bundle, OGB's ``raw/`` and ``mapping/``
        folders, a materialised PyG table -- so the reading is delegated to the
        source. What comes back is always a :class:`~networks._core.nodes.NodeTable`::

            nodes = networks.load("ogbn-arxiv").nodes()
            nodes.columns                       # ('arxiv_category', 'year', 'paper')
            nodes["arxiv_category"]             # {node: 'arxiv cs na', ...}
            nodes.partition("arxiv_category")   # {node: 0..k-1}, for AMI/NMI/D

        Sources that keep metadata in a separate artifact download it here, on the
        same cache-once terms as :func:`load`.

        Node metadata is not ground truth for community structure: a partition that
        disagrees with an attribute is not thereby wrong. See
        :mod:`networks._core.nodes`.

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
        module = importlib.import_module(f"networks.{self.source}")
        reader = getattr(module, "node_table", None)
        if reader is None:
            raise LookupError(f"networks.{self.source} publishes no per-node metadata.")
        table: NodeTable = reader(self)
        return table

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

    # -- conversions -----------------------------------------------------
    def to_networkx(
        self, *, directed: bool | None = None, cast: Callable[[str], Any] | None = int
    ) -> Any:
        """Build a networkx graph.

        Direction defaults to what SNAP reports for the dataset. Requires the
        ``networkx`` extra::

            pip install "mapequation-networks[networkx]"

        Be deliberate here: a networkx graph holds every node and edge in
        memory, so check ``dataset.edges`` before calling this on a large one.
        """
        try:
            import networkx
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_networkx() needs networkx. Install it with "
                '`pip install "mapequation-networks[networkx]"`.'
            ) from error

        if directed is None:
            directed = bool(self.dataset.directed)
        graph = networkx.DiGraph() if directed else networkx.Graph()
        graph.name = self.dataset.name

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

        A third column counts as a weight only where :meth:`to_infomap` lets
        Infomap parse the file itself -- whitespace-separated, and weights the
        source marks as valid -- so a written file loads the same network.
        """
        if not (self.delimiter is None and self.weighted_links):
            for source, target in self.edges():
                yield source, target, None
            return
        for fields in self.rows(delimiter=None):
            if len(fields) < 2:
                continue
            number = _LEADING_NUMBER.match(fields[2]) if len(fields) > 2 else None
            yield fields[0], fields[1], float(number.group()) if number else None

    def _typed_links(self, cast: Callable[[str], Any]) -> Iterator[formats.Link]:
        for source, target, weight in self._links():
            yield cast(source), cast(target), weight

    def to_pajek(
        self, directory: str | Path, *, name: str | None = None, directed: bool | None = None
    ) -> Path:
        """Write ``<directory>/<name>.net`` in Pajek format and return its path.

        *name* defaults to the dataset's name and *directed* to the direction
        the source reports. Integer node ids are kept; any other labels are
        numbered ``1..N`` and kept as ``*Vertices`` names. The file is
        streamed in two passes -- nodes, then links -- so memory holds the
        node set but never the links.
        """
        if directed is None:
            directed = bool(self.dataset.directed)
        labels = dict.fromkeys(label for u, v, _ in self._links() for label in (u, v))
        cast: Callable[[str], Any] = (
            int if all(label.isascii() and label.isdigit() for label in labels) else str
        )
        target = formats.output_path(directory, name or self.name, ".net")
        return formats.write_pajek(
            target,
            nodes=(cast(label) for label in labels),
            links=lambda: self._typed_links(cast),
            directed=directed,
            comments=formats.describe(self.name, None),
        )

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
            return int(label) if label.isascii() and label.isdigit() else label

        target = formats.output_path(directory, name or self.name, ".txt")
        return formats.write_link_list(
            target,
            links=self._typed_links(cast),
            directed=directed,
            comments=formats.describe(self.name, None),
        )

    def to_infomap(self, *, directed: bool | None = None, **kwargs: Any) -> Any:
        """Build an :class:`infomap.Infomap` instance with this network loaded.

        Direction defaults to what SNAP reports. Extra keyword arguments are
        passed to the ``Infomap`` constructor. Requires the ``infomap`` extra::

            pip install "mapequation-networks[infomap]"
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_infomap() needs infomap. Install it with "
                '`pip install "mapequation-networks[infomap]"`.'
            ) from error

        if directed is None:
            directed = bool(self.dataset.directed)
        kwargs.setdefault("directed", directed)
        instance = Infomap(**kwargs)

        if self.delimiter is None and self.weighted_links:
            # Fast path: let Infomap's own parser read the file directly.
            instance.read_file(str(self.path))
        else:
            # Infomap cannot parse CSV, nor a negative weight, so feed the links
            # in. Only the first two columns are used: the extra columns on
            # these files are ratings and timestamps, and signed ratings would
            # be invalid as Infomap link weights.
            instance.add_links(self.edges(cast=int))
        return instance

    def __repr__(self) -> str:
        return f"<Network {self.dataset.name!r} file={self.file.name!r} at {self.archive_path}>"

    def __str__(self) -> str:
        return f"{self.dataset} -> {self.path}"
