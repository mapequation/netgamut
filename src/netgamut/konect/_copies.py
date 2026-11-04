"""Which KONECT networks are copies of a Netzschleuder or SNAP network.

A copy is decided on recorded evidence, never inferred at runtime: a false copy
hides a distinct network from cross-source queries, while a missed one only
shows a network twice. Equal node and edge counts, which unrelated small
networks share by chance, therefore only *nominate* a pair. The catalog build
then judges each pair and vendors the verdict; see :class:`Pair`.

Verdicts that make a copy:

``imported``
    Netzschleuder's entry names the KONECT network as its upstream, and holds
    that one network rather than a family of subgraphs.
``same-edges``
    The two edge lists were downloaded and compared: equal as they stand
    (KONECT's ids are Netzschleuder's plus one, SNAP's as they are), or equal up
    to a relabelling of the nodes.
``override``
    A pair recorded by hand in :data:`OVERRIDES`, for one too large to check.

Verdicts that do not:

``different``
    The edge lists differ, or the published statistics rule the pair out: one
    network is directed and the other is not, or KONECT's is bipartite and
    Netzschleuder's cannot be. Statistics only ever reject a pair, since the two
    sources define some of them differently.
``unverified``
    Not compared yet, usually because the pair is above the build's size cap.
"""

from __future__ import annotations

import importlib
from array import array
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COPY_VERDICTS",
    "OVERRIDES",
    "Candidate",
    "Pair",
    "compare_edges",
    "contradiction",
    "judge",
    "nominate",
]

#: The verdicts under which cross-source queries use the other copy.
COPY_VERDICTS = frozenset({"imported", "same-edges", "override"})
VERDICTS = COPY_VERDICTS | {"different", "unverified"}

#: Verdicts recorded by hand, ``(KONECT name, "<source>/<name>") -> (verdict,
#: why)``, for pairs the build cannot settle or settles too tersely: one too
#: large to download, say, or one whose edges match only once reversed.
OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    # Equal up to relabelling as undirected networks, but directed, and every
    # link points the other way: a different network for any flow on it.
    ("facebook-wosn-wall", "netzschleuder/facebook_wall"): (
        "different",
        "the same links, each pointing the other way",
    ),
}

#: Up to this many edges, a pair that matches up to relabelling is also checked
#: for an exact isomorphism, when networkx is installed.
ISOMORPHISM_CAP = 10_000


@dataclass(frozen=True, slots=True)
class Pair:
    """A KONECT network, a network elsewhere it may be a copy of, and the verdict."""

    #: KONECT's name for the network.
    name: str
    #: ``"<source>/<name>"`` of the other network.
    of: str
    #: One of ``imported``, ``same-edges``, ``override``, ``different`` and
    #: ``unverified``; see the module docstring.
    verdict: str
    #: How the verdict was reached, for a reader.
    note: str = ""

    @property
    def is_copy(self) -> bool:
        return self.verdict in COPY_VERDICTS

    def to_dict(self) -> dict[str, str]:
        record = {"name": self.name, "of": self.of, "verdict": self.verdict}
        if self.note:
            record["note"] = self.note
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pair:
        return cls(
            name=data["name"], of=data["of"], verdict=data["verdict"], note=data.get("note", "")
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A pair worth judging, and why it was nominated."""

    name: str
    of: str
    #: Netzschleuder's entry names this KONECT network as its upstream.
    imported: bool
    #: The two report the same node and edge count.
    same_size: bool


# -- nomination ----------------------------------------------------------------
def _imported(netzschleuder: Iterable[Any]) -> dict[str, str]:
    """KONECT name -> the Netzschleuder entry that names it as its upstream."""
    found: dict[str, str] = {}
    for item in netzschleuder:
        # An entry of many subgraphs (one per language, say) describes its
        # default one, which need not be the KONECT network it links to.
        if item.subgraph_count > 1:
            continue
        host, _, path = item.upstream_url.partition("://")[2].partition("/")
        if "konect" in host and path.startswith("networks/"):
            found[path.removeprefix("networks/").strip("/")] = f"netzschleuder/{item.name}"
    return found


def nominate(
    konect: Iterable[Any], netzschleuder: Sequence[Any], snap: Sequence[Any]
) -> list[Candidate]:
    """Every pair worth judging, in order of preference per KONECT network.

    A pair is nominated by provenance (Netzschleuder imported the network) or by
    an exact node and edge count match. Netzschleuder comes before SNAP, so it
    wins when both hold a verified copy.
    """
    imported = _imported(netzschleuder)
    by_size: dict[tuple[int, int], list[str]] = {}
    for prefix, items in (("netzschleuder", netzschleuder), ("snap", snap)):
        for other in items:
            if other.nodes and other.edges:
                by_size.setdefault((other.nodes, other.edges), []).append(f"{prefix}/{other.name}")

    candidates: list[Candidate] = []
    for dataset in konect:
        size = (dataset.nodes or 0, dataset.edges or 0)
        twins = by_size.get(size, []) if all(size) else []
        upstream = imported.get(dataset.name)
        keys = ([upstream] if upstream else []) + [key for key in twins if key != upstream]
        for key in keys:
            candidates.append(
                Candidate(dataset.name, key, imported=key == upstream, same_size=key in twins)
            )
    return candidates


def contradiction(dataset: Any, other: Any) -> str:
    """Why the published statistics rule a pair out, or ``""`` if they don't.

    Only properties both sources mean the same way are compared: direction, and
    KONECT's bipartite flag against Netzschleuder's 2-colourability (a
    non-bipartite network may still be 2-colourable, so only that direction
    rejects).
    """
    known = dataset.directed is not None and other.directed is not None
    if known and dataset.directed != other.directed:
        here = "directed" if dataset.directed else "undirected"
        there = "directed" if other.directed else "undirected"
        return f"{here} in KONECT, {there} in the other"
    if dataset.bipartite and getattr(other, "bipartite", None) is False:
        return "bipartite in KONECT, not 2-colourable in the other"
    return ""


# -- content -------------------------------------------------------------------
def _label(value: str) -> Hashable:
    """A node id as an int where it is one, so ids can be shifted."""
    return int(value) if value.isdigit() else value


@dataclass
class _Edges:
    """An edge list on dense ids, with the original labels kept for the ids."""

    labels: dict[Hashable, int]
    source: array[int]
    target: array[int]

    @classmethod
    def read(cls, pairs: Iterable[tuple[str, str]]) -> _Edges:
        labels: dict[Hashable, int] = {}
        source, target = array("q"), array("q")
        for left, right in pairs:
            source.append(labels.setdefault(_label(str(left)), len(labels)))
            target.append(labels.setdefault(_label(str(right)), len(labels)))
        return cls(labels, source, target)

    def keys(self, ids: Sequence[int], base: int, *, directed: bool, simple: bool) -> list[int]:
        """Each edge as one int ``u * base + v`` on *ids*, sorted."""
        if directed:
            found = [ids[u] * base + ids[v] for u, v in zip(self.source, self.target, strict=True)]
        else:
            found = [
                min(ids[u], ids[v]) * base + max(ids[u], ids[v])
                for u, v in zip(self.source, self.target, strict=True)
            ]
        return sorted(set(found)) if simple else sorted(found)


def _same_ids(left: _Edges, right: _Edges, shift: int, **how: bool) -> bool:
    """Whether the edge lists are equal once *right*'s ids are shifted by *shift*."""
    translate: list[int] = []
    for label in right.labels:
        moved = label + shift if isinstance(label, int) else label
        if moved not in left.labels:
            return False
        translate.append(left.labels[moved])
    base = len(left.labels) + 1
    identity = range(len(left.labels))
    return left.keys(identity, base, **how) == right.keys(translate, base, **how)


def _neighbours(edges: _Edges, *, directed: bool, simple: bool) -> list[tuple[list[int], ...]]:
    """Per dense id: its neighbours, or its successors and predecessors."""
    count = len(edges.labels)
    base = count + 1
    keys = edges.keys(range(count), base, directed=directed, simple=simple)
    if directed:
        out: list[list[int]] = [[] for _ in range(count)]
        into: list[list[int]] = [[] for _ in range(count)]
        for key in keys:
            u, v = divmod(key, base)
            out[u].append(v)
            into[v].append(u)
        return list(zip(out, into, strict=True))
    adjacent: list[list[int]] = [[] for _ in range(count)]
    for key in keys:
        u, v = divmod(key, base)
        adjacent[u].append(v)
        if u != v:
            adjacent[v].append(u)
    return [(row,) for row in adjacent]


def _refine(
    left: list[tuple[list[int], ...]], right: list[tuple[list[int], ...]], rounds: int = 50
) -> tuple[bool, list[int], list[int]]:
    """Weisfeiler--Lehman colour refinement of both graphs on one palette.

    Starts from the degrees and stops when the partition no longer splits. The
    graphs differ if a round's colour histograms do; otherwise the final colours
    are returned, to guide an exact isomorphism check.
    """

    def start(graph: list[tuple[list[int], ...]]) -> list[Hashable]:
        return [tuple(len(side) for side in row) for row in graph]

    palette: dict[Hashable, int] = {}
    colours = [
        [palette.setdefault(value, len(palette)) for value in start(graph)]
        for graph in (left, right)
    ]
    if Counter(colours[0]) != Counter(colours[1]):
        return False, colours[0], colours[1]
    classes = len(set(colours[0]))
    for _ in range(rounds):
        palette = {}
        colours = [
            [
                palette.setdefault(
                    (own[node], *(tuple(sorted(own[other] for other in side)) for side in row)),
                    len(palette),
                )
                for node, row in enumerate(graph)
            ]
            for graph, own in ((left, colours[0]), (right, colours[1]))
        ]
        if Counter(colours[0]) != Counter(colours[1]):
            return False, colours[0], colours[1]
        now = len(set(colours[0]))
        if now == classes:
            break
        classes = now
    return True, colours[0], colours[1]


def _isomorphic(
    left: _Edges,
    right: _Edges,
    colours: tuple[list[int], list[int]],
    *,
    directed: bool,
    simple: bool,
) -> bool:
    """An exact check with networkx's VF2, guided by the colours.

    Raises:
        ImportError: networkx is not installed.
    """
    try:
        nx: Any = importlib.import_module("networkx")
    except ImportError as error:
        raise ImportError(
            "comparing small networks up to relabelling needs networkx "
            "(uv run --extra networkx ...)"
        ) from error
    kind = {
        (False, True): nx.Graph,
        (True, True): nx.DiGraph,
        (False, False): nx.MultiGraph,
        (True, False): nx.MultiDiGraph,
    }[directed, simple]
    graphs = []
    for edges, colour in ((left, colours[0]), (right, colours[1])):
        graph = kind()
        graph.add_nodes_from((node, {"colour": value}) for node, value in enumerate(colour))
        graph.add_edges_from(zip(edges.source, edges.target, strict=True))
        graphs.append(graph)
    return bool(
        nx.is_isomorphic(graphs[0], graphs[1], node_match=lambda a, b: a["colour"] == b["colour"])
    )


def compare_edges(
    konect: Iterable[tuple[str, str]],
    other: Iterable[tuple[str, str]],
    *,
    directed: bool,
    simple: bool,
    shift: int = 0,
    isomorphism_cap: int = ISOMORPHISM_CAP,
) -> tuple[bool, str]:
    """Whether two edge lists describe the same network, and how that was settled.

    First directly, with KONECT's id equal to the other's plus *shift*: 1 for
    Netzschleuder, which counts from 0. Failing that, up to relabelling: the
    sorted degree sequences, then Weisfeiler--Lehman colour refinement, then,
    up to *isomorphism_cap* edges, an exact isomorphism check with networkx.
    *simple* compares the sets of edges, so that a file listing an undirected
    edge in both directions still matches; otherwise repeated edges count, as
    in a multigraph.

    Raises:
        ImportError: A small pair needs the exact check and networkx is missing.
    """
    how = {"directed": directed, "simple": simple}
    left, right = _Edges.read(konect), _Edges.read(other)
    if _same_ids(left, right, shift, **how):
        return True, "same ids" if shift == 0 else f"same ids, offset by {shift}"

    left_graph = _neighbours(left, **how)
    right_graph = _neighbours(right, **how)
    if len(left_graph) != len(right_graph):
        return (
            False,
            f"{len(left_graph)} nodes with links in KONECT, {len(right_graph)} in the other",
        )
    degrees = [
        sorted(tuple(len(side) for side in row) for row in graph)
        for graph in (left_graph, right_graph)
    ]
    if degrees[0] != degrees[1]:
        return False, "degree sequences differ"
    same, *colours = _refine(left_graph, right_graph)
    if not same:
        return False, "not equal up to relabelling (Weisfeiler-Lehman)"
    if len(left.source) > isomorphism_cap:
        return True, "equal up to relabelling (degree sequences, Weisfeiler-Lehman)"
    # Colour refinement cannot tell some small graphs apart -- a 6-cycle from
    # two triangles -- so a small pair also needs an exact isomorphism.
    if not _isomorphic(left, right, (colours[0], colours[1]), **how):
        return False, "not isomorphic"
    return True, "isomorphic after relabelling"


# -- judging -------------------------------------------------------------------
def judge(
    candidates: Iterable[Candidate],
    *,
    konect: Mapping[str, Any],
    others: Mapping[str, Any],
    previous: Iterable[Pair] = (),
    check: Callable[[Candidate], tuple[bool, str]] | None = None,
    max_edges: int | None = None,
    overrides: Mapping[tuple[str, str], tuple[str, str]] = OVERRIDES,
) -> list[Pair]:
    """A verdict for every candidate, in the candidates' order.

    *konect* maps a KONECT name, and *others* a ``"<source>/<name>"`` key, to its
    dataset. A content comparison from *previous* is reused for the same pair.
    *check* compares the two edge lists, for pairs of equal size up to
    *max_edges* edges; without it, pairs not settled otherwise stay unverified.
    A check that fails, a download that breaks, leaves its pair unverified.
    """
    # Only content comparisons are worth carrying over; the rest is cheap.
    settled = {
        (pair.name, pair.of): pair
        for pair in previous
        if pair.verdict in ("same-edges", "different")
    }
    verdicts: list[Pair] = []
    for candidate in candidates:
        key = (candidate.name, candidate.of)
        dataset, other = konect[candidate.name], others[candidate.of]
        fallback = "imported" if candidate.imported else "unverified"
        if key in overrides:
            verdict, note = overrides[key]
            verdicts.append(Pair(*key, verdict, note))
            continue
        reason = contradiction(dataset, other)
        if reason and not candidate.imported:
            verdicts.append(Pair(*key, "different", reason))
            continue
        if not candidate.same_size or reason:
            verdicts.append(Pair(*key, fallback, reason))
            continue
        if key in settled:
            verdicts.append(settled[key])
            continue
        edges = dataset.edges or 0
        if check is None or (max_edges is not None and edges > max_edges):
            verdicts.append(Pair(*key, fallback, ""))
            continue
        try:
            same, note = check(candidate)
        except Exception as error:  # one broken download must not sink the build
            verdicts.append(Pair(*key, fallback, f"check failed: {type(error).__name__}"))
            continue
        if same:
            verdicts.append(Pair(*key, "same-edges", note))
        else:
            verdicts.append(Pair(*key, "imported" if candidate.imported else "different", note))
    return verdicts


def check(candidate: Candidate) -> tuple[bool, str]:
    """Download both edge lists of *candidate* and compare them.

    Uses each source's :func:`load`, so the files land in the shared cache and a
    second build reads them from there.
    """
    from .. import netzschleuder, snap
    from . import info, load

    dataset = info(candidate.name)
    source, _, name = candidate.of.partition("/")
    if source == "netzschleuder":
        other, shift = netzschleuder.load(name, progress=False), 1
    elif source == "snap":
        other, shift = snap.load(name, progress=False), 0
    else:
        raise ValueError(f"no edge list to compare for {candidate.of!r}")
    return compare_edges(
        load(candidate.name, progress=False).edges(),
        other.edges(),
        directed=bool(dataset.directed),
        simple=not dataset.multigraph,
        shift=shift,
    )
