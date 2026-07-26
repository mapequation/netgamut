"""Command line interface: ``networks <command>``.

Mirrors the Python API so you can explore the collection from a shell before
writing any code::

    networks categories
    networks list --category "Collaboration networks"
    networks info ca-GrQc
    networks get ca-GrQc
    networks cache
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__, snap


def _format_count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "-"


def _print_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    if not rows:
        print("No matching datasets.")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    print(line.rstrip())
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())


def _cmd_categories(args: argparse.Namespace) -> int:
    rows = [
        [category.name, str(len(category.dataset_names)), category.description]
        for category in snap.categories()
    ]
    _print_table(rows, ["Category", "N", "Description"])
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    results = snap.search(
        args.query or "",
        category=args.category,
        directed=args.directed,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
    )
    rows = [
        [
            dataset.name,
            dataset.category_name,
            dataset.type or "-",
            _format_count(dataset.nodes),
            _format_count(dataset.edges),
        ]
        for dataset in results
    ]
    _print_table(rows, ["Name", "Category", "Type", "Nodes", "Edges"])
    print(f"\n{len(rows)} dataset(s).")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(snap.describe(args.name))
    if snap.is_cached(args.name):
        print("\nCached locally.")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    network = snap.load(
        args.name,
        file=args.file,
        force=args.force,
        progress=True,
        max_bytes=args.max_bytes,
    )
    print(network.path if not network.is_archive else network.archive_path)
    return 0


def _cmd_refresh(args: argparse.Namespace) -> int:
    print("Scraping snap.stanford.edu ...", file=sys.stderr)
    catalog = snap.refresh(details=not args.quick)
    print(
        f"Catalog refreshed: {len(catalog.datasets)} datasets in "
        f"{len(catalog.categories)} categories ({catalog.retrieved})."
    )
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.clear:
        reclaimed = snap.clear_cache(args.name)
        print(f"Reclaimed {reclaimed / 1e6:.1f} MB.")
        return 0

    info = snap.cache_info()
    print(f"Location  {info.path}")
    print(f"Size      {info.bytes / 1e6:.1f} MB")
    print(f"Catalog   {'refreshed copy in cache' if info.catalog_refreshed else 'bundled'}")
    if info.datasets:
        print("\nCached datasets")
        for name, detail in info.datasets.items():
            print(f"  {name:<32} {detail.bytes / 1e6:8.1f} MB  {len(detail.files)} file(s)")
    else:
        print("\nNothing downloaded yet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="networks",
        description="Browse and download network datasets from the SNAP collection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    categories = subparsers.add_parser("categories", help="list SNAP's categories")
    categories.set_defaults(func=_cmd_categories)

    listing = subparsers.add_parser("list", help="list or search datasets")
    listing.add_argument("query", nargs="?", help="substring of the name or description")
    listing.add_argument("-c", "--category")
    listing.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    listing.add_argument("--max-nodes", type=int)
    listing.add_argument("--max-edges", type=int)
    listing.set_defaults(func=_cmd_list)

    info = subparsers.add_parser("info", help="show a dataset's metadata without downloading")
    info.add_argument("name")
    info.set_defaults(func=_cmd_info)

    get = subparsers.add_parser("get", help="download a dataset and print its cached path")
    get.add_argument("name")
    get.add_argument("-f", "--file", help="which artifact to fetch")
    get.add_argument("--force", action="store_true", help="re-download if already cached")
    get.add_argument("--max-bytes", type=int, help="refuse downloads larger than this")
    get.set_defaults(func=_cmd_get)

    refresh = subparsers.add_parser("refresh", help="re-scrape SNAP into the local catalog")
    refresh.add_argument(
        "--quick", action="store_true", help="index page only, skip per-dataset detail pages"
    )
    refresh.set_defaults(func=_cmd_refresh)

    cache = subparsers.add_parser("cache", help="show or clear the local cache")
    cache.add_argument("--clear", action="store_true")
    cache.add_argument("name", nargs="?", help="limit --clear to one dataset")
    cache.set_defaults(func=_cmd_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        return int(result)
    except (LookupError, ValueError) as error:  # LookupError covers KeyError
        print(f"error: {error}", file=sys.stderr)
        return 1
    except snap.DownloadError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
