"""The web mirror: networks a browser can load, from mapequation/netgamut-networks.

The upstream hosts send no CORS headers, so a web app such as the Network
Navigator cannot fetch their files. :func:`export_networks` converts the graph
sources' networks to gzipped Pajek, smallest first, until a size budget is
spent, and writes the ``index.json`` manifest that the catalog explorer and the
Navigator read. Every file opens with comment lines that say where the network
came from and how to cite it. A multilayer network is written in Infomap's
multilayer format, its manifest entry saying how many ``layers`` it has.

The mirror is committed to the repository's ``mirror`` branch and served by
raw.githubusercontent.com, so git's limits apply: GitHub rejects a file of
100 MiB, and a repository should stay under 5 GB.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import shutil
import tempfile
import time
import urllib.error
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import netgamut

from . import _KINDS, __version__
from ._core.download import DownloadError
from ._core.multilayer import MultilayerNetwork

#: The sources whose networks are first-order graphs the Navigator can load.
MIRRORED_SOURCES = ("netzschleuder", "konect", "snap")

DEFAULT_BUDGET = 4_000_000_000
MAX_FILE = 95_000_000
MAX_DOWNLOAD = 1_000_000_000

MANIFEST = "index.json"


def export_networks(
    output: str | Path,
    *,
    sources: Iterable[str] = MIRRORED_SOURCES,
    budget: int = DEFAULT_BUDGET,
    max_file: int = MAX_FILE,
    max_download: int = MAX_DOWNLOAD,
    force: bool = False,
    stop_after: float | None = None,
    clear_cache: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Write ``<output>/<source>/<name>.net.gz`` files and their ``index.json``.

    Networks are taken in order of edge count until their files would exceed
    *budget* bytes. Left out: KONECT's copies of another source's network,
    SNAP's collections that are not graphs, networks of unknown size, files
    over *max_file* bytes, downloads over *max_download*, and networks that
    read as having no nodes or no links, which would list a file with nothing
    in it.

    A network its source tags multilayer is written in Infomap's multilayer
    format (``*Vertices``, ``*Intra``) where its source lays out the layers,
    with ``layers`` in its manifest entry; one whose layers netgamut cannot
    find is written as one network, and says so in a comment.

    Run it again on the same *output* to update it. A network whose catalog
    entry is unchanged keeps its file without a download unless *force* is set;
    one that fails to build keeps its previous file; files no longer listed are
    deleted. *clear_cache* deletes each download once it is converted, so a full
    build holds one network on disk at a time.

    *stop_after* seconds into the run, no new download starts: the remaining
    networks keep the files they have and the rest are left for the next run,
    which continues where this one stopped. The manifest's ``complete`` is
    false until a run gets through every network. A run that changes nothing
    writes the manifest unchanged, timestamp included. Returns the manifest.
    """
    deadline = None if stop_after is None else time.monotonic() + stop_after
    complete = True
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    sources = tuple(sources)
    earlier = _read_manifest(folder)
    kept = {
        entry["key"]: entry
        for entry in earlier.get("datasets", [])
        # An empty file is never worth keeping: it is built again, or skipped.
        if (folder / entry["file"]).is_file() and entry["nodes"] and entry["edges"]
    }
    too_large = {
        entry["key"]: entry["fingerprint"]
        for entry in earlier.get("skipped", [])
        if entry["reason"].startswith("too large") and "fingerprint" in entry
    }

    # Another run's sources stay as they were.
    entries = [entry for key, entry in kept.items() if _source(key) not in sources]
    skipped = [
        entry for entry in earlier.get("skipped", []) if _source(entry["key"]) not in sources
    ]
    total = sum(entry["bytes"] for entry in entries)

    candidates = []
    for source in sources:
        for match in netgamut.search(source=source):
            if getattr(match.dataset, "duplicate_of", None) or not _is_graph(match):
                continue
            fingerprint = _fingerprint(match)
            if match.edges is None:
                skipped.append({"key": match.key, "reason": "size unknown"})
            # A gzipped Pajek link takes at least about two bytes.
            elif match.edges * 2 > max_file:
                skipped.append(
                    {"key": match.key, "reason": "too large", "fingerprint": fingerprint}
                )
            elif too_large.get(match.key) == fingerprint:
                # Rejected last time and unchanged since: not worth another download.
                reason = "too large on an earlier build"
                skipped.append({"key": match.key, "reason": reason, "fingerprint": fingerprint})
            else:
                candidates.append((match, fingerprint))
    candidates.sort(key=lambda pair: (pair[0].edges, pair[0].nodes or 0, pair[0].key))

    down: set[str] = set()  # sources whose host stopped answering in this run
    reserved = 0
    for position, (match, fingerprint) in enumerate(candidates):
        previous = kept.get(match.key)
        entry: dict[str, Any] | None
        reason = ""
        if previous is not None and previous.get("fingerprint") == fingerprint and not force:
            entry = previous
        elif deadline is not None and time.monotonic() >= deadline:
            complete = False
            if previous is None:
                skipped.append({"key": match.key, "reason": "not built yet"})
                continue
            entry = previous
        elif match.source in down:
            # Each attempt at a host that does not answer waits out three
            # timeouts, so the source waits for the next run instead.
            entry = previous
            reason = f"{match.source} unreachable in this run"
        else:
            try:
                entry = _convert(match, folder, fingerprint, max_file, max_download)
            except Exception as error:
                described = (
                    error
                    if isinstance(error, (_TooLarge, _Empty))
                    else f"{type(error).__name__}: {error}"
                )
                reason = " ".join(str(described).split())[:300]
                if _unreachable(error):
                    down.add(match.source)
                    log(f"{match.source} does not answer; its other networks wait for a later run")
                entry = previous
                if previous is not None:
                    log(f"{match.key}: {reason}; kept the previous file")
                elif match.source not in down:
                    skipped.append({"key": match.key, "reason": reason, "fingerprint": fingerprint})
                    log(f"{match.key}: skipped, {reason}")
                    continue
            finally:
                if clear_cache:
                    netgamut.clear_cache(match.name, source=match.source)
        size = entry["bytes"] if entry is not None else _estimate(match)
        if total + size > budget:
            skipped.extend(
                {"key": rest.key, "reason": "over budget"} for rest, _ in candidates[position:]
            )
            break
        total += size
        if entry is None:
            # Unreachable and never built: its estimated space is held, so that
            # building it in a later run pushes out no file already pushed.
            reserved += size
            skipped.append({"key": match.key, "reason": reason, "reserved": size})
            log(f"{match.key}: skipped, {reason}")
            continue
        # From the catalog, not the file, so a kept entry gains them without a
        # download; the Navigator filters on them as the explorer does.
        entries.append({**entry, **{name: getattr(match, name) for name in _KINDS}})
        log(f"{match.key}: {entry['nodes']:,} nodes, {entry['edges']:,} links")

    _prune(folder, sources, {entry["file"] for entry in entries})
    document = {
        "netgamut": __version__,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "complete": complete,
        "bytes": total - reserved,
        "reserved": reserved,
        "datasets": sorted(entries, key=lambda entry: str(entry["key"])),
        "skipped": sorted(skipped, key=lambda entry: str(entry["key"])),
    }
    # An idle run must not touch the file, or every one of them would commit.
    if {**earlier, "built": None} == {**document, "built": None}:
        document["built"] = earlier["built"]
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    (folder / MANIFEST).write_text(text + "\n", encoding="utf-8")
    return document


class _TooLarge(ValueError):
    """The gzipped file would be over the per-file limit."""


class _Empty(ValueError):
    """The network read as having no nodes or no links."""


def _unreachable(error: Exception) -> bool:
    """Whether the host did not answer, rather than refusing this one file."""
    if not isinstance(error, DownloadError):
        return False
    cause = error.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code >= 500
    return isinstance(cause, (urllib.error.URLError, TimeoutError, OSError))


def _estimate(match: Any) -> int:
    """A network's likely file size: gzipped Pajek measured about 3.9 bytes a link."""
    return int(match.edges) * 4


def _source(key: str) -> str:
    return key.split("/", 1)[0]


def _is_graph(match: Any) -> bool:
    """SNAP also lists reviews, tweets and edit histories; a graph's type names its direction."""
    kind = getattr(match.dataset, "type", None)
    return match.source != "snap" or bool(kind and re.search(r"irected|bipartite", kind, re.I))


#: Raised for a source when the files its networks convert to change, so the next
#: run converts them again. A file that comes out the same stays byte for byte the
#: same, so only the changed ones reach the mirror's history. 2: node names;
#: Netzschleuder 3: its edge lists read as CSV, quoted fields and all; SNAP 2:
#: the Reddit hyperlink TSVs read without their header row.
_FORMATS = {"netzschleuder": 3, "konect": 2, "snap": 2}
#: Raised when multilayer networks are written differently, which rebuilds
#: those alone. 1: in Infomap's multilayer format.
_LAYERED_FORMAT = 1


def _fingerprint(match: Any) -> str:
    """Changes whenever the source's catalog entry or file format does, so either rebuilds."""
    text = repr(match.dataset)
    version = _FORMATS.get(match.source, 1)
    if version > 1:  # version 1 left it out, and its fingerprints still hold
        text += f"\nformat {version}"
    if getattr(match, "multilayer", False):
        text += f"\nlayers {_LAYERED_FORMAT}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_manifest(folder: Path) -> dict[str, Any]:
    path = folder / MANIFEST
    if not path.is_file():
        return {}
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _convert(
    match: Any, folder: Path, fingerprint: str, max_file: int, max_download: int
) -> dict[str, Any]:
    """Download one network, write it as gzipped Pajek, and describe the file."""
    meta = match.to_dict()
    with tempfile.TemporaryDirectory() as scratch:
        layered = _layered(match, max_download)
        if layered is not None:
            pajek = layered.to_pajek(scratch, name=match.name)
            note = _layer_note(layered)
        else:
            network = match.load(progress=False, max_bytes=max_download)
            pajek = network.to_pajek(scratch, name=match.name)
            note = []
            if getattr(match, "multilayer", False):
                note = [
                    f"{match.source} tags this network multilayer, but netgamut finds no "
                    "layers in its data, so its links are written as one network."
                ]
        packed = Path(scratch) / f"{pajek.name}.gz"
        header = _attribution(meta) + note + _id_note(pajek)
        nodes, edges, directed = _compress(pajek, packed, header)
        if layered is not None:
            directed = layered.directed  # the multilayer format does not say
        if not nodes or not edges:
            raise _Empty(f"empty: {nodes:,} nodes and {edges:,} links read")
        size = packed.stat().st_size
        if size > max_file:
            raise _TooLarge(f"too large: {size:,} bytes gzipped, over {max_file:,}")
        relative = f"{match.source}/{packed.name}"
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(packed, folder / relative)
    entry = {
        "key": meta["key"],
        "title": meta["title"] or meta["name"],
        "file": relative,
        "bytes": size,
        "nodes": nodes,
        "edges": edges,
        "directed": directed,
        "url": meta["url"],
        "citations": meta["citations"],
        "fingerprint": fingerprint,
    }
    if layered is not None:
        entry["layers"] = layered.num_layers
    return entry


def _layered(match: Any, max_download: int) -> MultilayerNetwork | None:
    """The network in its layers, if its source tags it multilayer and can lay them out."""
    if not getattr(match, "multilayer", False):
        return None
    module = importlib.import_module(f"netgamut.{match.source}")
    loader = getattr(module, "load_multilayer", None)
    if loader is None:
        return None
    try:
        layered: MultilayerNetwork = loader(match.name, progress=False, max_bytes=max_download)
    except LookupError:
        return None
    return layered


def _layer_note(layered: MultilayerNetwork) -> list[str]:
    """How to read a multilayer file, which neither Pajek nor the format itself says."""
    direction = "directed" if layered.directed else "undirected"
    return [
        f"Multilayer: {layered.num_layers:,} layers in Infomap's multilayer format. "
        f"*Intra lines are layer, node, node, weight, and the links are {direction}. "
        "No link joins two layers, so couple them with a relax rate, as Infomap's "
        "--multilayer-relax-rate does."
    ]


def _attribution(meta: dict[str, Any]) -> list[str]:
    lines = [f"{meta['key']}: {meta['title'] or meta['name']}", f"From {meta['url']}"]
    lines += [f"Cite: {citation['text']} {citation['url']}" for citation in meta["citations"]]
    lines.append(
        f"Converted by netgamut {__version__}; see github.com/mapequation/netgamut-networks"
    )
    # One comment line each: a newline inside a citation would end the comment.
    return [" ".join(line.split()) for line in lines]


def _id_note(pajek: Path) -> list[str]:
    """Say where the node ids run, since they are the source's own.

    They are kept rather than renumbered ``1..N``, so a node keeps its id across
    the subgraphs of a dataset -- the snapshots of a temporal network, say --
    even where it has no links in some of them. A reader keys nodes by id, as
    Infomap does.
    """
    ids = []
    with pajek.open("rb") as lines:
        for line in lines:
            if line.startswith(b"*"):
                if not line.lower().startswith(b"*vertices"):
                    break  # the links follow the vertices
            elif line.strip() and not line.startswith(b"#"):
                ids.append(int(line.split(maxsplit=1)[0]))
    if not ids:
        return []
    return [
        "Node ids are the source's own, so they can start at 0 and skip numbers: "
        f"{len(ids):,} nodes, ids {min(ids)} to {max(ids)}."
    ]


def _compress(source: Path, target: Path, header: list[str]) -> tuple[int, int, bool]:
    """Gzip *source* behind *header* comments; return its node and link counts.

    The gzip header carries no name or time and every line ends in ``\\n``,
    whatever the platform wrote, so an unchanged network gives the same bytes
    on every build. Level 6 rather than 9: on a 1.5M-link network it wrote the
    same bytes six times faster.
    """
    nodes = links = 0
    directed = in_links = False
    with (
        source.open("rb") as lines,
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as packed,
    ):
        for comment in header:
            packed.write(f"# {comment}\n".encode())
        for line in lines:
            line = line.rstrip(b"\r\n") + b"\n"
            packed.write(line)
            if line.startswith(b"*"):
                keyword, *rest = line.split()
                heading = keyword.lower()
                if heading == b"*vertices":
                    nodes = int(rest[0])
                in_links = heading in (b"*edges", b"*arcs", b"*intra", b"*inter")
                directed = directed or heading == b"*arcs"
            elif in_links and line.strip() and not line.startswith(b"#"):
                links += 1
    return nodes, links, directed


def _prune(folder: Path, sources: tuple[str, ...], listed: set[str]) -> None:
    """Delete the files of networks this build no longer lists."""
    for source in sources:
        for path in (folder / source).glob("*.net.gz"):
            if path.relative_to(folder).as_posix() not in listed:
                path.unlink()
