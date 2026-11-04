#!/usr/bin/env python3
"""Rebuild the vendored KONECT catalog.

Reads konect.cc's network index and each network's page, and writes
``src/netgamut/konect/catalog.json``, which is what the installed
package uses so that metadata lookups need no network.

    python scripts/regenerate_konect_catalog.py

Archives up to 100 MB are listed to learn which ship node attributes; a
listing is reused from the existing catalog while its archive's size and date
are unchanged, so only the first build downloads them (2.4 GB).

It also judges which KONECT networks are copies of a Netzschleuder or SNAP
network, against the vendored catalogs of those two, so regenerate them first.
Equal node and edge counts only nominate a pair; see netgamut.konect._copies.
Comparing the edge lists is opt-in, as it downloads both:

    python scripts/regenerate_konect_catalog.py --verify-copies
    python scripts/regenerate_konect_catalog.py --copies-only --verify-copies \
        --max-verify-edges 10000000

Without --verify-copies, comparisons from the existing catalog are kept and new
pairs stay unverified, which keeps them apart. --copies-only skips konect.cc
and judges the pairs of the existing catalog. With networkx installed
(uv run --extra networkx ...), small pairs that match only up to relabelling
are also checked for an exact isomorphism.

Exits non-zero if the result looks implausible, so a site redesign fails the job
instead of shipping an empty catalog.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from netgamut._core.domains import DOMAINS  # noqa: E402
from netgamut.konect import _copies  # noqa: E402
from netgamut.konect._catalog import Catalog, build_catalog  # noqa: E402
from netgamut.netzschleuder._catalog import Catalog as NetzschleuderCatalog  # noqa: E402
from netgamut.snap._catalog import Catalog as SnapCatalog  # noqa: E402

TARGET = ROOT / "src" / "netgamut" / "konect" / "catalog.json"
PACKAGE = ROOT / "src" / "netgamut"
#: Pairs with more edges than this are left unverified unless asked for.
MAX_VERIFY_EDGES = 1_000_000

MIN_NETWORKS = 250
MIN_WITH_SIZE = 250
# Every domain must be represented, otherwise the taxonomy has broken.
MIN_PER_DOMAIN = 5
# Most pages state the format and edge type; if they stop parsing, fail.
MIN_WITH_FORMAT = 250
# Archives listed for their contents; fewer means listing broke.
MIN_LISTED = 250


def judge_copies(
    catalog: Catalog, previous: Catalog | None, *, verify: bool, max_edges: int
) -> tuple[_copies.Pair, ...]:
    """A verdict on every pair a KONECT network forms with a same-sized network elsewhere."""

    def vendored(source: str) -> dict[str, Any]:
        path = PACKAGE / source / "catalog.json"
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    netzschleuder = NetzschleuderCatalog.from_dict(vendored("netzschleuder")).datasets
    snap = SnapCatalog.from_dict(vendored("snap")).datasets
    candidates = _copies.nominate(catalog.datasets, netzschleuder, snap)
    others: dict[str, Any] = {f"netzschleuder/{item.name}": item for item in netzschleuder}
    others.update({f"snap/{item.name}": item for item in snap})
    konect = {item.name: item for item in catalog.datasets}

    checked = 0

    def check(candidate: _copies.Candidate) -> tuple[bool, str]:
        nonlocal checked
        checked += 1
        edges = konect[candidate.name].edges or 0
        print(
            f"  comparing {candidate.name} with {candidate.of} ({edges:,} edges)", file=sys.stderr
        )
        return _copies.check(candidate)

    pairs = _copies.judge(
        candidates,
        konect=konect,
        others=others,
        previous=previous.copies if previous else (),
        check=check if verify else None,
        max_edges=max_edges,
    )
    copies = {pair.name for pair in pairs if pair.is_copy}
    print(
        f"  {len(pairs)} candidate pairs, {len(copies)} KONECT networks copied elsewhere, "
        f"{checked} pairs compared now",
        file=sys.stderr,
    )
    for verdict, count in sorted(Counter(pair.verdict for pair in pairs).items()):
        print(f"    {verdict:<12} {count}", file=sys.stderr)
    return tuple(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output", type=Path, default=TARGET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verify-copies",
        action="store_true",
        help="download and compare the edge lists of pairs not compared before",
    )
    parser.add_argument(
        "--max-verify-edges",
        type=int,
        default=MAX_VERIFY_EDGES,
        help=f"compare only pairs up to this many edges (default {MAX_VERIFY_EDGES:,})",
    )
    parser.add_argument(
        "--copies-only",
        action="store_true",
        help="keep the existing catalog and only judge its copies anew",
    )
    args = parser.parse_args()

    previous = None
    if args.output.is_file():
        previous = Catalog.from_dict(json.loads(args.output.read_text(encoding="utf-8")))
    if args.copies_only:
        if previous is None:
            print(f"--copies-only needs an existing {args.output}", file=sys.stderr)
            return 1
        catalog = previous
    else:
        print("Reading konect.cc ...", file=sys.stderr)
        catalog = build_catalog(workers=args.workers, previous=previous)
    print("Judging copies ...", file=sys.stderr)
    copies = judge_copies(
        catalog, previous, verify=args.verify_copies, max_edges=args.max_verify_edges
    )
    catalog = dataclasses.replace(catalog, copies=copies)

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
    listed = [item for item in catalog.datasets if item.members is not None]
    with_nodes = [item.name for item in listed if item.node_metadata]
    no_edges = [item.name for item in listed if not item.edge_member]
    print(
        f"  {len(listed)} archives listed, {len(with_nodes)} with node attributes", file=sys.stderr
    )
    if no_edges:
        print(f"  warning: no edge list found in {', '.join(no_edges)}", file=sys.stderr)
    if len(listed) < MIN_LISTED:
        problems.append(f"only {len(listed)} archives listed (need {MIN_LISTED})")

    with_format = sum(1 for item in catalog.datasets if item.format and item.edge_type)
    if with_format < MIN_WITH_FORMAT:
        problems.append(f"only {with_format} have format and edge type (need {MIN_WITH_FORMAT})")
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
    output = args.output.resolve()
    shown = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(
        f"Wrote {shown} ({args.output.stat().st_size / 1024:.0f} KB).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
