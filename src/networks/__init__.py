"""Browse, download and cache network datasets.

Each source of networks lives in its own subpackage. Today that is
:mod:`networks.snap`, covering the Stanford Large Network Dataset Collection::

    from networks import snap

    print(snap.describe("ca-GrQc"))   # offline, instant
    net = snap.load("ca-GrQc")        # downloads once, then cached

Downloads land in a shared user-level cache, so importing this package from a
different project reuses whatever is already on the machine.
"""

from __future__ import annotations

from . import snap

__all__ = ["__version__", "snap"]

__version__ = "0.1.0"
