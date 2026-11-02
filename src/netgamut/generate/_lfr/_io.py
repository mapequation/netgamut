"""Reading what the LFR programs write.

The generators write plain text into their working directory, one-based throughout:

* ``network.dat`` -- ``u v`` per line, or ``u v w`` for the weighted variants.
  Undirected files list every edge twice, once per direction; directed files once.
* ``community.dat`` -- ``u c...``, one line per node, with several ``c`` when the node
  was made to overlap (``on``/``om``).
* ``community_first_level.dat`` / ``community_second_level.dat`` -- the hierarchical
  variant's micro and macro assignments, in the same format.
* ``network_layer_<i>`` / ``community_layer_<i>`` -- the multiplex variant's slices.

Everything here returns the file's own labels; the wrapper renumbers to zero-based
last, so a parse can still be compared against the raw ``.dat`` if something looks
wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "as_partition",
    "compact",
    "macro_of_micro",
    "read_memberships",
    "read_network",
    "relabel",
]


def read_network(path: Path, *, directed: bool, weighted: bool) -> Any:
    """Read an edge list into a networkx graph, keeping the file's node labels.

    Self-loops are dropped: the map equation has no use for them and LFR does not
    intend to plant any.
    """
    import networkx

    G = networkx.DiGraph() if directed else networkx.Graph()
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            u, v = int(fields[0]), int(fields[1])
            if u == v:
                continue
            if weighted:
                G.add_edge(u, v, weight=float(fields[2]))
            else:
                G.add_edge(u, v)
    return G


def read_memberships(path: Path) -> dict[int, tuple[int, ...]]:
    """Read ``u c...`` lines into ``{node: (community, ...)}``, labels as written."""
    memberships: dict[int, tuple[int, ...]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            memberships[int(fields[0])] = tuple(int(c) for c in fields[1:])
    return memberships


def compact(values: Iterable[int]) -> dict[int, int]:
    """Map the distinct *values* onto ``0..k-1``, in ascending order."""
    return {value: index for index, value in enumerate(sorted(set(values)))}


def macro_of_micro(
    micro: Mapping[int, tuple[int, ...]], macro: Mapping[int, tuple[int, ...]]
) -> dict[int, int]:
    """Which macro community each micro community sits in.

    The hierarchical generator numbers micro communities globally and nests each one
    entirely inside a single macro community, but says so only implicitly, through the
    two per-node files. Nodes with one membership at both levels pin the relation
    without ambiguity; overlapping nodes are then read through it rather than guessed
    at.
    """
    parent: dict[int, int] = {}
    for node, inner in micro.items():
        outer = macro.get(node, ())
        if len(inner) == 1 and len(outer) == 1:
            parent.setdefault(inner[0], outer[0])
    return parent


def as_partition(paths: Mapping[int, Sequence[tuple[int, ...]]]) -> dict[int, Any]:
    """Collapse per-node module paths to the shape the caller asked for.

    One path of depth one is a plain module index, one deeper path is a hierarchical
    path tuple, and several paths are a cover -- the node's memberships side by side.
    Which of the three a call produces follows from its arguments, so the caller always
    knows: a flat non-overlapping run gives ``int``, ``mu1``/``mu2`` give
    ``(macro, micro)``, and ``on > 0`` gives a tuple of memberships.
    """
    partition: dict[int, Any] = {}
    for node, node_paths in paths.items():
        flat = [p[0] if len(p) == 1 else tuple(p) for p in node_paths]
        partition[node] = flat[0] if len(flat) == 1 else tuple(flat)
    return partition


def relabel(G: Any, mapping: Mapping[int, int]) -> Any:
    """Renumber *G* through *mapping*, keeping edge data and graph type."""
    import networkx

    return networkx.relabel_nodes(G, dict(mapping), copy=True)
