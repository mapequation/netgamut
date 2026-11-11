"""Command line interface: ``netgamut <command>``.

Mirrors the Python API so you can explore both collections from a shell before
writing any code::

    netgamut domains
    netgamut list --domain Biological --max-nodes 1000
    netgamut info celegansneural
    netgamut get ca-GrQc
    netgamut cache
    netgamut export-index --output catalog.json
    netgamut export-networks --output mirror
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import netgamut

from . import __version__, snap
from ._export import DEFAULT_BUDGET, MAX_FILE, MIRRORED_SOURCES, export_networks

# The package's own registry, so the CLI cannot drift behind a new source.
_SOURCES = netgamut._SOURCES


def _count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "-"


def _print_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    if not rows:
        print("No matches.")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def _resolve(name: str, source: str | None) -> tuple[str, Any]:
    """Find which source has *name*, or fail with a clear message."""
    if source is not None:
        if source not in _SOURCES:
            raise KeyError(f"Unknown source {source!r}. Available: {', '.join(_SOURCES)}")
        return source, _SOURCES[source]
    found = netgamut._owners(name)
    if not found:
        raise KeyError(f"No dataset named {name!r} in any source.")
    if len(found) > 1:
        raise KeyError(f"{name!r} exists in {', '.join(found)}. Pass --source to choose.")
    return found[0], _SOURCES[found[0]]


def _cmd_domains(args: argparse.Namespace) -> int:
    counts = netgamut.domains()
    rows = []
    for domain in netgamut.DOMAINS:
        per_source = [str(len(netgamut.search(domain=domain, source=name))) for name in _SOURCES]
        rows.append([domain, str(counts[domain]), *per_source])
    _print_table(rows, ["Domain", "Total", *_SOURCES])
    return 0


def _cmd_categories(args: argparse.Namespace) -> int:
    rows = [
        [category.name, str(len(category.dataset_names)), category.description]
        for category in snap.categories()
    ]
    _print_table(rows, ["SNAP category", "N", "Description"])
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    if args.category:
        # A SNAP category has no cross-source meaning, so this narrows to SNAP.
        selected = snap.search(
            args.query or "",
            category=args.category,
            directed=args.directed,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
        )
        matches = tuple(netgamut.Match("snap", dataset) for dataset in selected)
    else:
        matches = netgamut.search(
            args.query or "",
            domain=args.domain,
            source=args.source,
            directed=args.directed,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
        )
    rows = [
        [
            match.source,
            match.name,
            "/".join(match.domains) or "-",
            "D" if match.directed else "U",
            _count(match.nodes),
            _count(match.edges),
        ]
        for match in matches
    ]
    _print_table(rows, ["Source", "Name", "Domains", "Dir", "Nodes", "Edges"])
    print(f"\n{len(rows)} dataset(s).")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    source, module = _resolve(args.name, args.source)
    print(f"[{source}]")
    print(module.describe(args.name))
    if module.is_cached(args.name):
        print("\nCached locally.")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    source, module = _resolve(args.name, args.source)
    kwargs = {"force": args.force, "progress": True, "max_bytes": args.max_bytes}
    if args.file:
        if source != "snap":
            raise ValueError(f"--file only applies to the snap source, not {source!r}.")
        kwargs["file"] = args.file
    if args.year is not None or args.quarter is not None:
        if source != "paths":
            raise ValueError(f"--year/--quarter only apply to the paths source, not {source!r}.")
        if args.year is not None:
            kwargs["year"] = args.year
        if args.quarter is not None:
            kwargs["quarter"] = args.quarter
    if args.revision is not None:
        if source != "ahorn":
            raise ValueError(f"--revision only applies to the ahorn source, not {source!r}.")
        kwargs["revision"] = args.revision
    handle = module.load(args.name, **kwargs)
    # A Hypergraph has no members to extract; its file is the dataset.
    print(handle.archive_path if getattr(handle, "is_archive", False) else handle.path)
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    for name in [args.source] if args.source else list(_SOURCES):
        print(f"Refreshing {name} ...", file=sys.stderr)
        try:
            catalog = _SOURCES[name].refresh()
        except ImportError as error:
            # A source whose rebuild needs a package this install lacks -- ahorn
            # needs pyyaml. Skipping keeps `netgamut refresh` useful for the rest.
            print(f"  skipped: {error}", file=sys.stderr)
            continue
        print(f"  {len(catalog.datasets)} datasets ({catalog.retrieved})")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        reclaimed = netgamut.clear_cache(args.name, source=args.source)
        print(f"Reclaimed {reclaimed / 1e6:.1f} MB.")
        return 0

    info = netgamut.cache_info()
    print(f"Location  {info.path}")
    print(f"Size      {info.bytes / 1e6:.1f} MB")
    refreshed = ", ".join(info.refreshed_catalogs) or "none (using bundled)"
    print(f"Refreshed {refreshed}")
    if info.datasets:
        print("\nCached datasets")
        for key, detail in info.datasets.items():
            print(f"  {key:<42} {detail.bytes / 1e6:8.1f} MB  {len(detail.files)} file(s)")
    else:
        print("\nNothing downloaded yet.")
    return 0


def _cmd_export_index(args: argparse.Namespace) -> int:
    # Every source's full listing, KONECT's copies included: each copy carries
    # duplicate_of, so a reader can hide it exactly as search() does.
    datasets = [match.to_dict() for name in _SOURCES for match in netgamut.search(source=name)]
    document = {
        "netgamut": __version__,
        "domains": list(netgamut.DOMAINS),
        "sources": [
            {
                "name": name,
                "summary": (module.__doc__ or "").strip().splitlines()[0],
                "retrieved": module.catalog().retrieved,
            }
            for name, module in _SOURCES.items()
        ],
        "datasets": datasets,
    }
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    if args.output is None:
        print(text)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"{len(datasets)} datasets written to {args.output}", file=sys.stderr)
    return 0


def _cmd_export_networks(args: argparse.Namespace) -> int:
    document = export_networks(
        args.output,
        sources=args.source or MIRRORED_SOURCES,
        budget=args.budget * 1_000_000,
        max_file=args.max_file * 1_000_000,
        force=args.force,
        stop_after=None if args.stop_after is None else args.stop_after * 60,
        clear_cache=args.clear_cache,
        log=lambda message: print(message, file=sys.stderr),
    )
    waiting = sum(entry["reason"] == "not built yet" for entry in document["skipped"])
    print(
        f"{len(document['datasets'])} networks ({document['bytes'] / 1e6:.0f} MB) written "
        f"to {args.output}; {len(document['skipped'])} skipped"
        + (f", {len(document['failed'])} failed" if document["failed"] else "")
        + ("" if document["complete"] else f", {waiting} not built yet"),
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netgamut",
        description=(
            "Browse and download network datasets from SNAP, Netzschleuder, OGB, "
            "PyG, the curated paths collection and AHORN."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    domains = subparsers.add_parser("domains", help="dataset counts per domain, per source")
    domains.set_defaults(func=_cmd_domains)

    categories = subparsers.add_parser("categories", help="SNAP's own category headings")
    categories.set_defaults(func=_cmd_categories)

    listing = subparsers.add_parser("list", help="list or search datasets across sources")
    listing.add_argument("query", nargs="?", help="substring of the name or description")
    listing.add_argument("-d", "--domain", choices=netgamut.DOMAINS)
    listing.add_argument("-s", "--source", choices=list(_SOURCES))
    listing.add_argument("-c", "--category", help="a SNAP category heading; implies --source snap")
    listing.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    listing.add_argument("--max-nodes", type=int)
    listing.add_argument("--max-edges", type=int)
    listing.set_defaults(func=_cmd_list)

    info = subparsers.add_parser("info", help="show metadata without downloading")
    info.add_argument("name")
    info.add_argument("-s", "--source", choices=list(_SOURCES))
    info.set_defaults(func=_cmd_info)

    get = subparsers.add_parser("get", help="download and print the cached path")
    get.add_argument("name")
    get.add_argument("-s", "--source", choices=list(_SOURCES))
    get.add_argument("-f", "--file", help="which artifact to fetch (SNAP only)")
    get.add_argument("--revision", type=int, help="which published revision to pin (AHORN only)")
    get.add_argument("--year", type=int, help="which year to fetch (quarterly paths data only)")
    get.add_argument("--quarter", type=int, help="which quarter to fetch, 1-4 (paths data only)")
    get.add_argument("--force", action="store_true")
    get.add_argument("--max-bytes", type=int, help="refuse downloads larger than this")
    get.set_defaults(func=_cmd_get)

    refresh = subparsers.add_parser("refresh", help="re-read a source into the local catalog")
    refresh.add_argument("-s", "--source", choices=list(_SOURCES))
    refresh.set_defaults(func=_cmd_refresh)

    cache = subparsers.add_parser("cache", help="show or clear the local cache")
    cache.add_argument("--clear", action="store_true")
    cache.add_argument("-s", "--source", choices=list(_SOURCES))
    cache.add_argument("name", nargs="?", help="limit --clear to one dataset")
    cache.set_defaults(func=_cmd_cache)

    export = subparsers.add_parser(
        "export-index", help="write every source's catalog as one JSON document"
    )
    export.add_argument("-o", "--output", help="file to write; prints to stdout if omitted")
    export.set_defaults(func=_cmd_export_index)

    mirror = subparsers.add_parser(
        "export-networks",
        help="write or update the web mirror: networks as gzipped Pajek, smallest first",
    )
    mirror.add_argument("-o", "--output", required=True, help="directory to write or update")
    mirror.add_argument(
        "-s", "--source", action="append", choices=MIRRORED_SOURCES, help="repeatable"
    )
    mirror.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET // 1_000_000,
        help="total size in MB (default: %(default)s)",
    )
    mirror.add_argument(
        "--max-file",
        type=int,
        default=MAX_FILE // 1_000_000,
        help="largest file in MB (default: %(default)s; GitHub rejects 100 MiB)",
    )
    mirror.add_argument(
        "--force", action="store_true", help="rebuild networks whose catalog entry is unchanged"
    )
    mirror.add_argument(
        "--stop-after",
        type=float,
        metavar="MINUTES",
        help="start no new download after this long; the next run continues",
    )
    mirror.add_argument(
        "--clear-cache", action="store_true", help="delete each download once converted"
    )
    mirror.set_defaults(func=_cmd_export_networks)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (LookupError, ValueError) as error:  # LookupError covers KeyError
        print(f"error: {error}", file=sys.stderr)
        return 1
    except netgamut.snap.DownloadError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
