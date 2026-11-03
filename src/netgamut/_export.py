"""The web mirror: networks a browser can load, from mapequation/netgamut-networks.

The upstream hosts send no CORS headers, so a web app such as the Network
Navigator cannot fetch their files. :func:`export_networks` converts the graph
sources' networks to gzipped Pajek, smallest first, until a size budget is
spent, and writes the ``index.json`` manifest that the catalog explorer and the
Navigator read. Every file opens with comment lines that say where the network
came from and how to cite it.

The mirror is committed to the repository's ``mirror`` branch and served by
raw.githubusercontent.com, so git's limits apply: GitHub rejects a file of
100 MiB, and a repository should stay under 5 GB.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import netgamut

from . import __version__

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
    clear_cache: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Write ``<output>/<source>/<name>.net.gz`` files and their ``index.json``.

    Networks are taken in order of edge count until their files would exceed
    *budget* bytes. Left out: KONECT's copies of another source's network,
    SNAP's collections that are not graphs, networks of unknown size, files
    over *max_file* bytes and downloads over *max_download*.

    Run it again on the same *output* to update it. A network whose catalog
    entry is unchanged keeps its file without a download unless *force* is set;
    one that fails to build keeps its previous file; files no longer listed are
    deleted. *clear_cache* deletes each download once it is converted, so a full
    build holds one network on disk at a time. Returns the manifest.
    """
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    sources = tuple(sources)
    earlier = _read_manifest(folder)
    kept = {
        entry["key"]: entry
        for entry in earlier.get("datasets", [])
        if (folder / entry["file"]).is_file()
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

    for position, (match, fingerprint) in enumerate(candidates):
        previous = kept.get(match.key)
        if previous is not None and previous.get("fingerprint") == fingerprint and not force:
            entry = previous
        else:
            try:
                entry = _convert(match, folder, fingerprint, max_file, max_download)
            except Exception as error:
                described = (
                    error if isinstance(error, _TooLarge) else f"{type(error).__name__}: {error}"
                )
                reason = " ".join(str(described).split())[:300]
                if previous is None:
                    skipped.append({"key": match.key, "reason": reason, "fingerprint": fingerprint})
                    log(f"{match.key}: skipped, {reason}")
                    continue
                entry = previous
                log(f"{match.key}: {reason}; kept the previous file")
            finally:
                if clear_cache:
                    netgamut.clear_cache(match.name, source=match.source)
        if total + entry["bytes"] > budget:
            skipped.extend(
                {"key": rest.key, "reason": "over budget"} for rest, _ in candidates[position:]
            )
            break
        total += entry["bytes"]
        entries.append(entry)
        log(f"{match.key}: {entry['nodes']:,} nodes, {entry['edges']:,} links")

    _prune(folder, sources, {entry["file"] for entry in entries})
    document = {
        "netgamut": __version__,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": total,
        "datasets": sorted(entries, key=lambda entry: str(entry["key"])),
        "skipped": sorted(skipped, key=lambda entry: str(entry["key"])),
    }
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    (folder / MANIFEST).write_text(text + "\n", encoding="utf-8")
    return document


class _TooLarge(ValueError):
    """The gzipped file would be over the per-file limit."""


def _source(key: str) -> str:
    return key.split("/", 1)[0]


def _is_graph(match: Any) -> bool:
    """SNAP also lists reviews, tweets and edit histories; a graph's type names its direction."""
    kind = getattr(match.dataset, "type", None)
    return match.source != "snap" or bool(kind and re.search(r"irected|bipartite", kind, re.I))


def _fingerprint(match: Any) -> str:
    """Changes whenever the source's catalog entry does, so an update rebuilds the file."""
    return hashlib.sha256(repr(match.dataset).encode()).hexdigest()[:16]


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
        network = match.load(progress=False, max_bytes=max_download)
        pajek = network.to_pajek(scratch, name=match.name)
        packed = Path(scratch) / f"{pajek.name}.gz"
        nodes, edges, directed = _compress(pajek, packed, _attribution(meta))
        size = packed.stat().st_size
        if size > max_file:
            raise _TooLarge(f"too large: {size:,} bytes gzipped, over {max_file:,}")
        relative = f"{match.source}/{packed.name}"
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(packed, folder / relative)
    return {
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


def _attribution(meta: dict[str, Any]) -> list[str]:
    lines = [f"{meta['key']}: {meta['title'] or meta['name']}", f"From {meta['url']}"]
    lines += [f"Cite: {citation['text']} {citation['url']}" for citation in meta["citations"]]
    lines.append(
        f"Converted by netgamut {__version__}; see github.com/mapequation/netgamut-networks"
    )
    # One comment line each: a newline inside a citation would end the comment.
    return [" ".join(line.split()) for line in lines]


def _compress(source: Path, target: Path, header: list[str]) -> tuple[int, int, bool]:
    """Gzip *source* behind *header* comments; return its node and link counts.

    The gzip header carries no name or time and every line ends in ``\\n``,
    whatever the platform wrote, so an unchanged network gives the same bytes
    on every build.
    """
    nodes = links = 0
    directed = in_links = False
    with (
        source.open("rb") as lines,
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as packed,
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
                in_links = heading in (b"*edges", b"*arcs")
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
