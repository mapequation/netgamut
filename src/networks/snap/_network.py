"""The handle returned by :func:`networks.snap.load`.

A :class:`Network` is a thin wrapper around a cached file. It does not read the
data into memory when you create it, which is what lets the same API serve
``ca-GrQc`` (14k edges) and ``com-Friendster`` (1.8 billion edges). Conversions
to networkx and Infomap are lazy imports, so neither is needed to install or
use the package.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ._download import extract_gzip
from ._models import DataFile, Dataset

__all__ = ["Network"]

_COMMENT_PREFIXES = ("#", "%")


@dataclass(frozen=True)
class Network:
    """A downloaded SNAP dataset on local disk."""

    #: Full metadata for the dataset this file belongs to.
    dataset: Dataset
    #: The specific artifact that was downloaded.
    file: DataFile
    #: The downloaded artifact exactly as SNAP serves it, compression included.
    archive_path: Path

    # -- paths -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def is_archive(self) -> bool:
        """True for multi-file bundles (``.tar.gz``, ``.zip``)."""
        lowered = self.archive_path.name.lower()
        return lowered.endswith((".tar.gz", ".tgz", ".tar", ".zip"))

    @property
    def path(self) -> Path:
        """A path you can hand to any tool that reads a text file.

        For a single gzipped file this decompresses once, next to the archive,
        and returns the plain file. For multi-file bundles there is no single
        text file, so this returns the archive; use :meth:`extract` instead.
        """
        if self.is_archive:
            return self.archive_path
        if self.archive_path.name.lower().endswith(".gz"):
            target = self.archive_path.with_suffix("")
            return extract_gzip(self.archive_path, target)
        return self.archive_path

    @property
    def size_bytes(self) -> int:
        return self.archive_path.stat().st_size if self.archive_path.is_file() else 0

    def extract(self, destination: Path | None = None) -> Path:
        """Unpack a ``.tar.gz`` or ``.zip`` bundle and return the directory."""
        if not self.is_archive:
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
        target.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(self.archive_path):
            with zipfile.ZipFile(self.archive_path) as zip_archive:
                # ZipFile.extract already strips absolute paths and ".."
                # components from member names.
                zip_archive.extractall(target)
        else:
            with tarfile.open(self.archive_path) as tar_archive:
                # filter="data" refuses members that would escape the target.
                tar_archive.extractall(target, filter="data")
        return target

    # -- reading ---------------------------------------------------------
    def open(self) -> IO[str]:
        """Open the network as a text stream, decompressing on the fly."""
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

    def rows(self) -> Iterator[tuple[str, ...]]:
        """Yield every non-comment line, split into its whitespace-separated fields.

        Use this for the datasets that carry more than two columns, such as
        timestamps in the temporal category or signs in the signed category.
        """
        with self.open() as stream:
            for line in stream:
                if not line or line.startswith(_COMMENT_PREFIXES):
                    continue
                fields = line.split()
                if fields:
                    yield tuple(fields)

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
        instance.read_file(str(self.path))
        return instance

    def __repr__(self) -> str:
        return f"<Network {self.dataset.name!r} file={self.file.name!r} at {self.archive_path}>"

    def __str__(self) -> str:
        return f"{self.dataset} -> {self.path}"
