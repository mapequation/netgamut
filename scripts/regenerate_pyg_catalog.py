#!/usr/bin/env python3
"""Rebuild the vendored PyTorch Geometric catalog.

Reads ``torch_geometric/datasets/*.py`` from GitHub and parses the statistics tables
PyG keeps in its class docstrings -- the same source its documentation cheatsheet is
generated from -- and writes ``src/networks/pyg/catalog.json``::

    python scripts/regenerate_pyg_catalog.py
    python scripts/regenerate_pyg_catalog.py --show-omitted
    python scripts/regenerate_pyg_catalog.py --with-direction   # needs [pyg]

Neither ``torch`` nor ``torch_geometric`` is needed: this parses source text, so
regenerating the catalog is a light job even though loading a dataset is not.

The exception is ``--with-direction``. PyG's docstrings never say whether a network is
directed, so without it every entry has ``directed=None`` and ``search(directed=...)``
matches none of them. That pass materialises each dataset and reads the answer off its
``edge_index`` -- measured, not guessed -- filling in the sizes PyG leaves unpublished
along the way. It needs the ``[pyg]`` extra, downloads a lot, and is slow, so it is
opt-in and keeps whatever a previous run established for datasets it cannot reach.

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


def fill_direction(catalog: pyg_catalog.Catalog, *, max_edges: int | None) -> pyg_catalog.Catalog:
    """Materialise each dataset and record whether it is directed, and how big.

    PyG publishes neither, so both are read off the loaded graph. A dataset that
    cannot be reached -- a dead upstream mirror, a class needing arguments we do not
    pass -- keeps whatever the catalog already had rather than being reset to unknown.
    """
    import dataclasses

    from networks import pyg

    # The cap can only act on sizes PyG publishes. A dataset with no published size is
    # attempted whatever its true size, because that is the thing being measured -- so
    # a full pass can pull tens of gigabytes through the cache.
    filled, failed, skipped = [], [], 0
    for dataset in catalog.datasets:
        if max_edges is not None and dataset.edges is not None and dataset.edges > max_edges:
            skipped += 1
            filled.append(dataset)
            continue
        try:
            network = pyg.load(dataset.name)
        except (TypeError, AttributeError) as error:
            # Structural: PyG's constructor needs an argument we do not supply, or
            # refuses this name in the mode we use. That is a property of the dataset,
            # so record it -- an entry nobody can load should say so before download.
            reason = f"{type(error).__name__}: {str(error).strip()[:120]}"
            failed.append((dataset.name, reason + "  [recorded]"))
            filled.append(dataclasses.replace(dataset, load_error=reason))
            continue
        except Exception as error:  # noqa: BLE001
            # Environmental: an unreachable mirror, a missing optional dependency.
            # Says nothing about the dataset, so it must not be baked into the catalog
            # -- the same run on another machine would succeed.
            failed.append((dataset.name, f"{type(error).__name__}: {str(error)[:70]}"))
            filled.append(dataset)
            continue
        resolved = network.dataset
        print(f"  {dataset.name:46s} directed={str(resolved.directed):5s} "
              f"{resolved.nodes:>9,} nodes {resolved.edges:>10,} edges")
        filled.append(
            dataclasses.replace(
                dataset,
                directed=resolved.directed,
                nodes=resolved.nodes,
                edges=resolved.edges,
                load_error="",
            )
        )

    if skipped:
        print(f"\n{skipped} skipped as larger than --max-edges.")
    if failed:
        print(f"\n{len(failed)} could not be materialised, keeping what the catalog had:",
              file=sys.stderr)
        for name, reason in failed:
            print(f"  {name:46s} {reason}", file=sys.stderr)
    return dataclasses.replace(catalog, datasets=tuple(filled))


def carry_over(catalog: pyg_catalog.Catalog) -> pyg_catalog.Catalog:
    """Keep direction and sizes a previous --with-direction run established."""
    import dataclasses

    if not TARGET.is_file():
        return catalog
    previous = {item["name"]: item for item in json.loads(TARGET.read_text()).get("datasets", [])}
    return dataclasses.replace(
        catalog,
        datasets=tuple(
            dataclasses.replace(
                d,
                directed=previous.get(d.name, {}).get("directed", d.directed),
                load_error=previous.get(d.name, {}).get("load_error", d.load_error),
                nodes=d.nodes if d.nodes is not None else previous.get(d.name, {}).get("nodes"),
                edges=d.edges if d.edges is not None else previous.get(d.name, {}).get("edges"),
            )
            for d in catalog.datasets
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show-omitted", action="store_true",
                        help="list every class left out, and why")
    parser.add_argument("--with-direction", action="store_true",
                        help="materialise each dataset to record directedness and size")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="with --with-direction, skip datasets PyG already reports "
                             "as larger; the ones with no published size are attempted "
                             "regardless, since the size is only knowable by loading")
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

    catalog = carry_over(catalog)
    if args.with_direction:
        print("\nmaterialising to measure directedness:")
        catalog = fill_direction(catalog, max_edges=args.max_edges)

    unmapped = sorted({d.cls for d in catalog.datasets if not d.domains})
    if unmapped:
        print(f"\n{len(unmapped)} classes have no domain and will not match domain "
              f"queries:\n  {', '.join(unmapped)}")
        print("Add them to DOMAIN_MAP in src/networks/pyg/_models.py.")

    unsized = sum(1 for d in catalog.datasets if d.nodes is None)
    unknown = sum(1 for d in catalog.datasets if d.directed is None)
    total = len(catalog.datasets)
    print(f"\n{unsized} of {total} datasets have no size; search(max_nodes=...) skips those.")
    print(f"{unknown} of {total} have unknown directedness; search(directed=...) skips those."
          + ("  Run --with-direction to measure them." if unknown else ""))
    broken = [d.name for d in catalog.datasets if not d.supported]
    if broken:
        print(f"{len(broken)} cannot be constructed and are marked unloadable: "
              f"{', '.join(broken[:6])}{' ...' if len(broken) > 6 else ''}")

    TARGET.write_text(json.dumps(catalog.to_dict(), indent=2) + "\n")
    print(f"Wrote {TARGET.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
