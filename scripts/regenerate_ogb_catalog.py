#!/usr/bin/env python3
"""Rebuild the vendored OGB catalog.

Reads the ``master.csv`` files that OGB maintains as its dataset registry --
``ogb/nodeproppred/master.csv`` and ``ogb/linkproppred/master.csv`` -- and writes
``src/netgamut/ogb/catalog.json``, which is what the installed package uses so that
metadata lookups need no network::

    python scripts/regenerate_ogb_catalog.py
    python scripts/regenerate_ogb_catalog.py --with-stats   # also fill node/edge counts

``master.csv`` carries no node or edge counts, so they stay ``None`` unless
``--with-stats`` is given, which downloads each archive and reads the exact figures
out of ``raw/num-node-list.csv.gz`` and ``raw/num-edge-list.csv.gz``. That is slow and
runs to several gigabytes, but the numbers then come from the data rather than from a
web page someone transcribed. Already-cached archives are reused, and
``--max-bytes`` skips the ones too large to be worth it.

Exits non-zero if the registry looks implausible, so an OGB reorganisation fails the
job instead of shipping an empty catalog. A dataset with no entry in ``DOMAIN_MAP``
is also an error: a new OGB dataset needs a domain assigned by hand, and defaulting
to none would silently drop it out of every domain query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from netgamut.ogb import _catalog as ogb_catalog  # noqa: E402
from netgamut.ogb._models import DOMAIN_MAP  # noqa: E402

TARGET = ROOT / "src" / "netgamut" / "ogb" / "catalog.json"

#: Below this, the registry is not something we should ship.
MIN_DATASETS = 10


def fill_stats(catalog: ogb_catalog.Catalog, *, max_bytes: int | None) -> ogb_catalog.Catalog:
    """Download each loadable archive and read its exact node and edge counts."""
    import dataclasses
    import gzip
    import zipfile

    from netgamut import ogb

    filled = []
    for dataset in catalog.datasets:
        if not dataset.supported:
            filled.append(dataset)
            continue
        try:
            network = ogb.load(dataset.name, max_bytes=max_bytes, progress=True)
        except Exception as error:  # noqa: BLE001 - report and carry on
            print(f"  {dataset.name}: skipped ({error})", file=sys.stderr)
            filled.append(dataset)
            continue

        def count(member: str, archive: Path = network.archive_path) -> int | None:
            with zipfile.ZipFile(archive) as bundle:
                names = [n for n in bundle.namelist() if n.endswith(member)]
                if not names:
                    return None
                rows = gzip.decompress(bundle.read(names[0])).decode().split()
            return sum(int(row) for row in rows if row.strip())

        nodes = count("raw/num-node-list.csv.gz")
        edges = count("raw/num-edge-list.csv.gz")
        print(f"  {dataset.name}: {nodes} nodes, {edges} edges")
        filled.append(dataclasses.replace(dataset, nodes=nodes, edges=edges))
    return dataclasses.replace(catalog, datasets=tuple(filled))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--with-stats",
        action="store_true",
        help="download each archive to read exact node/edge counts",
    )
    parser.add_argument(
        "--max-bytes", type=int, default=None, help="with --stats, skip archives larger than this"
    )
    args = parser.parse_args(argv)

    catalog = ogb_catalog.build_catalog()
    print(f"read {len(catalog.datasets)} datasets from {', '.join(ogb_catalog.FAMILIES)}")

    if len(catalog.datasets) < MIN_DATASETS:
        print(
            f"\nRefusing to write: only {len(catalog.datasets)} datasets, expected at "
            f"least {MIN_DATASETS}. Has OGB moved master.csv?",
            file=sys.stderr,
        )
        return 2

    unmapped = [d.name for d in catalog.datasets if d.name not in DOMAIN_MAP]
    if unmapped:
        print(
            f"\nRefusing to write: no domain assigned for {', '.join(unmapped)}.", file=sys.stderr
        )
        print(
            "Add them to DOMAIN_MAP and DESCRIPTIONS in src/netgamut/ogb/_models.py.",
            file=sys.stderr,
        )
        return 2

    if args.with_stats:
        print("filling node/edge counts from the archives:")
        catalog = fill_stats(catalog, max_bytes=args.max_bytes)

    # Keep counts already vendored, so a run without --with-stats does not erase them.
    if TARGET.is_file() and not args.with_stats:
        import dataclasses

        previous = {
            item["name"]: item for item in json.loads(TARGET.read_text()).get("datasets", [])
        }
        catalog = dataclasses.replace(
            catalog,
            datasets=tuple(
                dataclasses.replace(
                    d,
                    nodes=previous.get(d.name, {}).get("nodes", d.nodes),
                    edges=previous.get(d.name, {}).get("edges", d.edges),
                )
                for d in catalog.datasets
            ),
        )

    TARGET.write_text(json.dumps(catalog.to_dict(), indent=2) + "\n")
    loadable = sum(1 for d in catalog.datasets if d.supported)
    print(
        f"\nWrote {TARGET.relative_to(ROOT)}: {len(catalog.datasets)} datasets, "
        f"{loadable} loadable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
