#!/usr/bin/env python3
"""Rebuild the vendored Netzschleuder catalog.

Reads networks.skewed.de's JSON API and writes
``src/netgamut/netzschleuder/catalog.json``, which is what the installed
package uses so that metadata lookups need no network.

    python scripts/regenerate_netzschleuder_catalog.py

Exits non-zero if the result looks implausible, so an API change fails the job
instead of shipping an empty catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from netgamut._core.domains import DOMAINS  # noqa: E402
from netgamut.netzschleuder._catalog import build_catalog  # noqa: E402

TARGET = ROOT / "src" / "netgamut" / "netzschleuder" / "catalog.json"

MIN_NETWORKS = 250
MIN_WITH_SIZE = 200
# Every domain must be represented, otherwise the taxonomy has broken.
MIN_PER_DOMAIN = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=TARGET)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print("Reading the Netzschleuder API ...", file=sys.stderr)
    catalog = build_catalog(workers=args.workers)

    with_size = sum(1 for item in catalog.datasets if item.nodes)
    counts = catalog.domains()
    print(f"  {len(catalog.datasets)} networks\n  {with_size} with reported size", file=sys.stderr)
    for domain in DOMAINS:
        print(f"  {domain:<16} {counts.get(domain, 0)}", file=sys.stderr)

    problems = []
    if len(catalog.datasets) < MIN_NETWORKS:
        problems.append(f"only {len(catalog.datasets)} networks (expected >= {MIN_NETWORKS})")
    if with_size < MIN_WITH_SIZE:
        problems.append(f"only {with_size} have sizes (expected >= {MIN_WITH_SIZE})")
    for domain in DOMAINS:
        if counts.get(domain, 0) < MIN_PER_DOMAIN:
            problems.append(f"domain {domain} has only {counts.get(domain, 0)} networks")
    if problems:
        print("\nRefusing to write, the result looks wrong:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    payload = catalog.to_dict()
    comparable = {key: value for key, value in payload.items() if key != "retrieved"}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if {k: v for k, v in existing.items() if k != "retrieved"} == comparable:
            print("Catalog is already up to date.", file=sys.stderr)
            return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output.relative_to(ROOT)} ({args.output.stat().st_size / 1024:.0f} KB).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
