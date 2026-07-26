"""Command line interface: ``networks <command>``.

Mirrors the Python API so you can explore both collections from a shell before
writing any code::

    networks domains
    networks list --domain Biological --max-nodes 1000
    networks info celegansneural
    networks get ca-GrQc
    networks cache
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import networks

from . import __version__, netzschleuder, snap

_SOURCES = {"snap": snap, "netzschleuder": netzschleuder}


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
    found = [key for key, module in _SOURCES.items() if name in module.catalog()]
    if not found:
        raise KeyError(f"No dataset named {name!r} in any source.")
    if len(found) > 1:
        raise KeyError(f"{name!r} exists in {', '.join(found)}. Pass --source to choose.")
    return found[0], _SOURCES[found[0]]


def _cmd_domains(args: argparse.Namespace) -> int:
    counts = networks.domains()
    rows = []
    for domain in networks.DOMAINS:
        per_source = [str(len(networks.search(domain=domain, source=name))) for name in _SOURCES]
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
        matches = tuple(networks.Match("snap", dataset) for dataset in selected)
    else:
        matches = networks.search(
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
    if source == "snap" and args.file:
        kwargs["file"] = args.file
    network = module.load(args.name, **kwargs)
    print(network.archive_path if network.is_archive else network.path)
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    for name in [args.source] if args.source else list(_SOURCES):
        print(f"Refreshing {name} ...", file=sys.stderr)
        catalog = _SOURCES[name].refresh()
        print(f"  {len(catalog.datasets)} datasets ({catalog.retrieved})")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        reclaimed = networks.clear_cache(args.name, source=args.source)
        print(f"Reclaimed {reclaimed / 1e6:.1f} MB.")
        return 0

    info = networks.cache_info()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="networks",
        description="Browse and download network datasets from SNAP and Netzschleuder.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    domains = subparsers.add_parser("domains", help="dataset counts per domain, per source")
    domains.set_defaults(func=_cmd_domains)

    categories = subparsers.add_parser("categories", help="SNAP's own category headings")
    categories.set_defaults(func=_cmd_categories)

    listing = subparsers.add_parser("list", help="list or search datasets across sources")
    listing.add_argument("query", nargs="?", help="substring of the name or description")
    listing.add_argument("-d", "--domain", choices=networks.DOMAINS)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (LookupError, ValueError) as error:  # LookupError covers KeyError
        print(f"error: {error}", file=sys.stderr)
        return 1
    except networks.snap.DownloadError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
