#!/usr/bin/env python3
"""Rebuild the vendored AHORN catalog.

Reads the dataset records from the repository behind ahorn.rwth-aachen.de --
one ``src/datasets/<slug>.mdx`` per dataset, YAML frontmatter -- plus the site
index for per-revision file sizes, and writes
``src/netgamut/ahorn/catalog.json``. That file is what the installed package
reads, which is why metadata lookups need no network.

Run it after AHORN publishes new datasets or revisions:

    python scripts/regenerate_ahorn_catalog.py

Needs pyyaml, which the installed package does not:

    uv pip install pyyaml

Exits non-zero if the result looks implausible, so a repository reorganisation
fails the job instead of silently shipping an empty catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from netgamut.ahorn._catalog import build_catalog  # noqa: E402

TARGET = ROOT / "src" / "netgamut" / "ahorn" / "catalog.json"

# Floors, not exact counts: the point is to catch a build that broke, not to pin
# the collection's current size. AHORN held 83 datasets at the time of its
# announcing paper (arXiv:2608.03491) and is expected to grow.
MIN_DATASETS = 70
MIN_WITH_REVISIONS = 70
MIN_WITH_DESCRIPTION = 60
MIN_WITH_CITATIONS = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=TARGET)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the catalog is up to date without writing it",
    )
    parser.add_argument(
        "--no-sizes",
        action="store_true",
        help="skip the site index, leaving per-revision file sizes unknown",
    )
    args = parser.parse_args()

    print("Reading the AHORN dataset records ...", file=sys.stderr)
    catalog = build_catalog(sizes=not args.no_sizes)

    with_revisions = sum(1 for dataset in catalog.datasets if dataset.revisions)
    with_description = sum(1 for dataset in catalog.datasets if dataset.description)
    with_citations = sum(1 for dataset in catalog.datasets if dataset.citations)
    with_sizes = sum(
        1
        for dataset in catalog.datasets
        for revision in dataset.revisions
        if revision.size is not None
    )
    revisions = sum(len(dataset.revisions) for dataset in catalog.datasets)
    print(
        f"  {len(catalog.datasets)} datasets\n"
        f"  {revisions} revisions ({with_sizes} with a known size)\n"
        f"  {with_revisions} with a downloadable revision\n"
        f"  {with_description} with a description\n"
        f"  {with_citations} with a citation",
        file=sys.stderr,
    )

    if (
        len(catalog.datasets) < MIN_DATASETS
        or with_revisions < MIN_WITH_REVISIONS
        or with_description < MIN_WITH_DESCRIPTION
        or with_citations < MIN_WITH_CITATIONS
    ):
        print(
            "error: the result looks implausible; refusing to write it. "
            "Check whether the repository moved src/datasets or changed its "
            "frontmatter schema.",
            file=sys.stderr,
        )
        return 1

    payload = json.dumps(catalog.to_dict(), indent=2) + "\n"

    if args.check:
        current = args.output.read_text() if args.output.is_file() else ""
        # `retrieved` is a timestamp and always differs; compare the datasets.
        fresh = json.loads(payload).get("datasets")
        stale = json.loads(current).get("datasets") if current else None
        if fresh == stale:
            print("Catalog is up to date.", file=sys.stderr)
            return 0
        print("Catalog is out of date; rerun without --check.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"Wrote {args.output.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
