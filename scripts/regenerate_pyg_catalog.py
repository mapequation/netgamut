#!/usr/bin/env python3
"""Rebuild the vendored PyTorch Geometric catalog.

Reads ``torch_geometric/datasets/*.py`` from GitHub and parses the statistics tables
PyG keeps in its class docstrings -- the same source its documentation cheatsheet is
generated from -- and writes ``src/networks/pyg/catalog.json``::

    python scripts/regenerate_pyg_catalog.py
    python scripts/regenerate_pyg_catalog.py --show-omitted

Neither ``torch`` nor ``torch_geometric`` is needed: this parses source text, so
regenerating the catalog is a light job even though loading a dataset is not.

Only single-graph datasets are catalogued. A class whose statistics table reports more
than one graph per dataset is a collection -- thousands of molecules or meshes -- and
cannot be one :class:`Network`. A class with no statistics table at all is catalogued
only if ``DOMAIN_MAP`` in ``src/networks/pyg/_models.py`` vouches for it, since
otherwise there is no way to tell the two apart.

Exits non-zero if the result looks implausible, so a PyG reorganisation fails the job
instead of shipping an empty catalog. Datasets with no domain assigned are reported
but not fatal: PyG has 89 homogeneous classes and the map is filled in as they matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from networks.pyg import _catalog as pyg_catalog  # noqa: E402

TARGET = ROOT / "src" / "networks" / "pyg" / "catalog.json"

#: Below this, PyG has been reorganised and the parse has failed.
MIN_DATASETS = 40


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show-omitted", action="store_true",
                        help="list every class left out, and why")
    args = parser.parse_args(argv)

    catalog = pyg_catalog.build_catalog()
    classes = {d.cls for d in catalog.datasets}
    print(f"parsed {len(catalog.datasets)} single-graph datasets in {len(classes)} classes")
    print(f"omitted {len(catalog.omitted)} classes")

    if len(catalog.datasets) < MIN_DATASETS:
        print(f"\nRefusing to write: only {len(catalog.datasets)} datasets, expected at "
              f"least {MIN_DATASETS}. Has PyG restructured torch_geometric/datasets?",
              file=sys.stderr)
        return 2

    reasons = Counter(catalog.omitted.values())
    for reason, count in reasons.most_common():
        print(f"  {count:3d}  {reason}")
    if args.show_omitted:
        print()
        for cls, reason in catalog.omitted.items():
            print(f"  {cls:34s} {reason}")

    unmapped = sorted({d.cls for d in catalog.datasets if not d.domains})
    if unmapped:
        print(f"\n{len(unmapped)} classes have no domain and will not match domain "
              f"queries:\n  {', '.join(unmapped)}")
        print("Add them to DOMAIN_MAP in src/networks/pyg/_models.py.")

    unsized = sum(1 for d in catalog.datasets if d.nodes is None)
    print(f"\n{unsized} of {len(catalog.datasets)} datasets have no size published by PyG; "
          f"search(max_nodes=...) will not match those.")

    TARGET.write_text(json.dumps(catalog.to_dict(), indent=2) + "\n")
    print(f"Wrote {TARGET.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
