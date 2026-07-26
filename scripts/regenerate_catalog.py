#!/usr/bin/env python3
"""Rebuild the vendored SNAP catalog.

Scrapes snap.stanford.edu -- the index page plus every dataset's detail page --
and writes the result to ``src/networks/snap/catalog.json``. That file is what
the installed package reads, which is why metadata lookups need no network.

Run it after SNAP publishes new datasets:

    python scripts/regenerate_catalog.py

The GitHub Actions workflow ``refresh-catalog.yml`` runs this monthly and opens
a pull request when the output changes. Exits non-zero if the scrape looks
implausible, so a SNAP redesign fails the job instead of silently shipping an
empty catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from networks.snap._catalog import build_catalog  # noqa: E402

TARGET = ROOT / "src" / "networks" / "snap" / "catalog.json"

# Floors, not exact counts: the point is to catch a scrape that broke, not to
# pin the collection's current size.
MIN_CATEGORIES = 20
MIN_DATASETS = 100
MIN_WITH_FILES = 90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=TARGET)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the catalog is up to date without writing it",
    )
    args = parser.parse_args()

    print("Scraping the SNAP index and detail pages ...", file=sys.stderr)
    catalog = build_catalog(details=True, workers=args.workers)

    with_files = sum(1 for dataset in catalog.datasets if dataset.files)
    with_stats = sum(1 for dataset in catalog.datasets if dataset.statistics)
    print(
        f"  {len(catalog.categories)} categories\n"
        f"  {len(catalog.datasets)} datasets\n"
        f"  {with_files} with download links\n"
        f"  {with_stats} with statistics",
        file=sys.stderr,
    )

    problems = []
    if len(catalog.categories) < MIN_CATEGORIES:
        problems.append(f"only {len(catalog.categories)} categories (expected >= {MIN_CATEGORIES})")
    if len(catalog.datasets) < MIN_DATASETS:
        problems.append(f"only {len(catalog.datasets)} datasets (expected >= {MIN_DATASETS})")
    if with_files < MIN_WITH_FILES:
        problems.append(f"only {with_files} datasets have files (expected >= {MIN_WITH_FILES})")
    if problems:
        print("\nRefusing to write, the scrape looks wrong:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("SNAP may have changed its markup; check networks/snap/_scrape.py.", file=sys.stderr)
        return 1

    payload = catalog.to_dict()
    # The timestamp changes on every run and would make every diff non-empty.
    comparable = {key: value for key, value in payload.items() if key != "retrieved"}

    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if {k: v for k, v in existing.items() if k != "retrieved"} == comparable:
            print("Catalog is already up to date.", file=sys.stderr)
            return 0
        if args.check:
            print("Catalog is OUT OF DATE.", file=sys.stderr)
            return 1

    if args.check:
        print("Catalog is missing.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    size = args.output.stat().st_size
    print(f"Wrote {args.output.relative_to(ROOT)} ({size / 1024:.0f} KB).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
