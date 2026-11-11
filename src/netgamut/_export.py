"""The web mirror: networks a browser can load, from mapequation/netgamut-networks.

The upstream hosts send no CORS headers, so a web app such as the Network
Navigator cannot fetch their files. :func:`export_networks` converts the graph
sources' networks to Parquet, smallest first, until a size budget is spent, and
writes the ``index.json`` manifest that the catalog explorer and the Navigator
read. Each network is a links file and, where the source names its nodes or has
node metadata, a nodes file (see :mod:`netgamut._parquet`); their metadata says
where the network came from and how to cite it. A multilayer network has a
``layer`` column, its manifest entry saying how many ``layers`` it has.

The mirror is committed to the repository's ``mirror`` branch and served by
raw.githubusercontent.com, so git's limits apply: GitHub rejects a file of
100 MiB, and a repository should stay under 5 GB.
"""

from __future__ import annotations

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

from . import _TRAITS, __version__, _parquet
from ._core.download import DownloadError
from ._core.network import ArchiveError

#: The sources whose networks are first-order graphs the Navigator can load.
MIRRORED_SOURCES = ("netzschleuder", "konect", "snap")

DEFAULT_BUDGET = 4_000_000_000
MAX_FILE = 95_000_000
MAX_DOWNLOAD = 1_000_000_000

MANIFEST = "index.json"
#: The mirror's file format, ``format`` in the manifest. 1, which a manifest
#: without it is, was gzipped Pajek; 2 is Parquet.
FORMAT = 2


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
    """Write ``<output>/<source>/<name>.links.parquet`` files and their ``index.json``.

    Networks are taken in order of edge count until their files would exceed
    *budget* bytes. Left out: KONECT's copies of another source's network,
    SNAP's collections that are not graphs, networks of unknown size, files
    over *max_file* bytes, downloads over *max_download*, and networks that
    read as having no nodes or no links, which would list a file with nothing
    in it.

    A network with a layer column, or one whose source keeps its layers in
    subgraphs, is written with a ``layer`` column and ``layers`` in its
    manifest entry; one its source tags multilayer whose layers netgamut
    cannot find is written as one network, and its entry's ``note`` says so.

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
    _parquet.require()  # before any download, rather than failing every network
    deadline = None if stop_after is None else time.monotonic() + stop_after
    complete = True
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    sources = tuple(sources)
    earlier = _read_manifest(folder)
    kept = {
        entry["key"]: entry
        for entry in earlier.get("datasets", [])
        # An empty file is never worth keeping: it is built again, or skipped. A
        # format-1 entry, with its one ``file``, is built again too.
        if "files" in entry
        and all((folder / path).is_file() for path in entry["files"].values())
        and entry["nodes"]
        and entry["links"]
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
    # A format-1 entry of another run's source waits for that source's run: the
    # manifest lists format 2 alone, and the mirror is not complete without it.
    stale = [
        entry
        for entry in earlier.get("datasets", [])
        if "files" not in entry and _source(entry["key"]) not in sources
    ]
    skipped.extend({"key": entry["key"], "reason": "not built yet"} for entry in stale)
    if any(entry["reason"] == "not built yet" for entry in skipped):
        complete = False
    # A failure stands until the network is built again.
    failed_before = {entry["key"]: entry for entry in earlier.get("failed", [])}
    failed = [entry for key, entry in failed_before.items() if _source(key) not in sources]
    total = sum(entry["bytes"] for entry in entries)

    candidates = []
    for source in sources:
        for match in netgamut.search(source=source):
            if getattr(match.dataset, "duplicate_of", None) or not _is_graph(match):
                continue
            fingerprint = _fingerprint(match)
            if match.edges is None:
                skipped.append({"key": match.key, "reason": "size unknown"})
            # A link takes at least about 1.25 bytes of Parquet: two of gzipped
            # Pajek, which Parquet measured 0.63 to 0.88 of.
            elif match.edges * 5 > max_file * 4:
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
            if match.key in failed_before:
                failed.append(failed_before[match.key])
            if previous is None:
                skipped.append({"key": match.key, "reason": "not built yet"})
                continue
            entry = previous
        elif match.source in down:
            # Each attempt at a host that does not answer waits out three
            # timeouts, so the source waits for the next run instead.
            entry = previous
            reason = f"{match.source} unreachable in this run"
            if match.key in failed_before:
                failed.append(failed_before[match.key])
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
                if not _upstream(error):
                    failed.append({"key": match.key, "reason": reason})
                    kept_file = "; kept the previous file" if previous is not None else ""
                    log(f"{match.key}: failed, {reason}{kept_file}")
                    if previous is None:
                        continue
                elif previous is not None:
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
        # From the catalog, as the files say them; the Navigator filters on them
        # as the explorer does.
        entries.append({**entry, **_traits(match)})
        log(f"{match.key}: {entry['nodes']:,} nodes, {entry['links']:,} links")

    _prune(folder, sources, {path for entry in entries for path in entry["files"].values()})
    for entry in stale:
        if "file" in entry:
            (folder / entry["file"]).unlink(missing_ok=True)
    document = {
        "format": FORMAT,
        "netgamut": __version__,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "complete": complete,
        "bytes": total - reserved,
        "reserved": reserved,
        "datasets": sorted(entries, key=lambda entry: str(entry["key"])),
        "skipped": sorted(skipped, key=lambda entry: str(entry["key"])),
        "failed": sorted(failed, key=lambda entry: str(entry["key"])),
    }
    # An idle run must not touch the file, or every one of them would commit.
    if {**earlier, "built": None} == {**document, "built": None}:
        document["built"] = earlier["built"]
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    (folder / MANIFEST).write_text(text + "\n", encoding="utf-8")
    return document


class _TooLarge(ValueError):
    """A file would be over the per-file limit."""


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


def _upstream(error: Exception) -> bool:
    """Whether *error* comes from the source, or a limit the mirror knows about.

    Those skip the network, or keep its previous file, and the build carries
    on. Anything else is a fault in netgamut, such as a missing module or a
    reader that cannot parse the file. It is listed under ``failed``, so the
    mirror's workflow turns red. A plain LookupError is a source saying it has
    no such file; KeyError and IndexError, its subclasses, are bugs.
    """
    if type(error) is LookupError:
        return True
    return isinstance(error, (DownloadError, _TooLarge, _Empty, ArchiveError))


def _estimate(match: Any) -> int:
    """A network's likely file size: Parquet measured about 2.5 to 3.4 bytes a link."""
    return int(match.edges) * 3


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
    # The files say the traits, so a change in what the catalog says of them,
    # its own or netgamut's reading of it, rebuilds them.
    text += f"\ntraits {_traits(match)}"
    # Every network is built again in a new mirror format.
    text += f"\nmirror {FORMAT}"
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
    """Download one network, write its Parquet files, and describe them."""
    meta = match.to_dict()
    parts, nodes, directed, note = _parts(match, max_download)
    with tempfile.TemporaryDirectory() as scratch:
        written = _parquet.write(
            parts, nodes, Path(scratch), match.name, _attribution(meta, directed, _traits(match))
        )
        if not written.nodes or not written.links:
            raise _Empty(f"empty: {written.nodes:,} nodes and {written.links:,} links read")
        sizes = {role: path.stat().st_size for role, path in written.files.items()}
        for role, size in sizes.items():
            if size > max_file:
                raise _TooLarge(
                    f"too large: {size:,} bytes of {role} in Parquet, over {max_file:,}"
                )
        files = {}
        for role, path in written.files.items():
            files[role] = f"{match.source}/{path.name}"
            (folder / files[role]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, folder / files[role])
    entry = {
        "key": meta["key"],
        "title": meta["title"] or meta["name"],
        "files": files,
        "bytes": sum(sizes.values()),
        "nodes": written.nodes,
        "links": written.links,
        "directed": directed,
        "url": meta["url"],
        "citations": meta["citations"],
        "fingerprint": fingerprint,
    }
    if written.layers is not None:
        entry["layers"] = written.layers
    if written.dropped:
        entry["dropped"] = written.dropped
    if note:
        entry["note"] = note
    return entry


def _parts(
    match: Any, max_download: int
) -> tuple[list[_parquet.Part], _parquet.Nodes | None, bool, str]:
    """The network's links, in parts; its nodes; whether it is directed; and a note.

    A network its source tags multilayer and keeps in one subgraph a layer, as
    some of Netzschleuder's do, is those subgraphs. Any other network is its
    own link list, its layer column the layers where it has one.
    """
    multilayer = bool(getattr(match, "multilayer", False))
    if multilayer:
        module = importlib.import_module(f"netgamut.{match.source}")
        subgraphs = getattr(module, "_subgraph_layers", None)
        if subgraphs is not None:
            try:
                layers, key = subgraphs(match.name, progress=False, max_bytes=max_download)
                parts, nodes = _parquet.stack(layers, key)
            except (KeyError, IndexError):
                raise  # a bug, not a missing table
            except LookupError:
                pass  # its layers are not subgraphs, or one has no nodes to match by
            else:
                directed = any(bool(network.dataset.directed) for network in layers.values())
                return parts, nodes, directed, ""
    network = match.load(progress=False, max_bytes=max_download)
    note = ""
    if multilayer and network.columns.layer is None:
        note = f"{match.source} tags this network multilayer, but its data has no layers"
    try:
        nodes = _parquet.node_rows(network.nodes())
    except (KeyError, IndexError):
        raise
    except LookupError:  # the source has no node table for it
        nodes = None
    return [_parquet.Part(network)], nodes, bool(network.dataset.directed), note


def _traits(match: Any) -> dict[str, bool | None]:
    """What the catalog says of each trait: yes, no, or ``None`` where it cannot say."""
    return {name: getattr(match, name) for name in _TRAITS}


def _attribution(
    meta: dict[str, Any], directed: bool, traits: dict[str, bool | None]
) -> dict[str, str]:
    """The files' key-value metadata, so that a downloaded file carries its attribution
    and says what its manifest entry says of the network."""
    citations = [f"{citation['text']} {citation['url']}" for citation in meta["citations"]]
    return {
        "netgamut:key": meta["key"],
        "netgamut:title": meta["title"] or meta["name"],
        "netgamut:url": meta["url"],
        # One line each: a newline inside a citation would read as two.
        "netgamut:citation": "\n".join(" ".join(citation.split()) for citation in citations),
        "netgamut:directed": "true" if directed else "false",
        # One key a trait, as for direction, where the catalog says yes or no.
        **{
            f"netgamut:{name}": "true" if value else "false"
            for name, value in traits.items()
            if value is not None
        },
        "netgamut:version": __version__,
    }


def _prune(folder: Path, sources: tuple[str, ...], listed: set[str]) -> None:
    """Delete the files of networks this build no longer lists, format 1's included."""
    for source in sources:
        for pattern in ("*.net.gz", "*.parquet"):
            for path in (folder / source).glob(pattern):
                if path.relative_to(folder).as_posix() not in listed:
                    path.unlink()
