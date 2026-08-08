"""Higher-order state networks built from path data.

A first-order network forgets where flow came from: at node ``j`` every walker
looks the same. Path data -- flight itineraries, navigation trails, chained
trips -- says otherwise, and a *state network* keeps that memory. With Markov
order 2, every trigram ``i -> j -> k`` observed in the paths contributes a link
from state node ``(i, j)`` to state node ``(j, k)``, where state ``(i, j)``
means "at physical node ``j``, having arrived from ``i``". Infomap reads this
representation natively and reports modules that can overlap on physical nodes,
which is exactly what second-order flows produce (Rosvall et al., "Memory in
network flows and its effects on spreading dynamics and community detection",
Nature Communications 5, 4630, 2014).

The construction generalises: with order ``m``, a state is the window of the
last ``m`` physical nodes, and every ``(m+1)``-gram contributes one state link.
Order 1 recovers the plain directed network from the same paths, which makes
first-order/second-order comparisons a one-argument change.
"""

from __future__ import annotations

import numbers
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["StateNetwork"]

#: One observed path: a node sequence, optionally paired with a weight.
PathItem = Sequence[Hashable] | tuple[Sequence[Hashable], float]


def _is_node_sequence(value: Any) -> bool:
    """Whether *value* can be the node-sequence half of a weighted pair.

    Strings are node labels here, never node sequences. Beyond the standard
    sequences the sized-ordered-iterable test admits numpy arrays, which is
    how pandas-shaped path data usually arrives.
    """
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return True
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__iter__")
        and not isinstance(value, (Mapping, Set))
    )


def _coerce(item: PathItem) -> tuple[tuple[Hashable, ...], float]:
    """Split a path item into ``(nodes, weight)``.

    A weighted item is a 2-tuple of a node sequence and a real number --
    ``numbers.Real``, so numpy scalars count; ``bool`` deliberately does not.
    Anything else is taken as a bare node sequence with weight 1.

    One ambiguity is inherent: a bare *two-node* path passed as a tuple whose
    first label is itself a sequence and whose second is a number reads as a
    weighted pair. Pass such paths as lists to keep them unambiguous.
    """
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and _is_node_sequence(item[0])
        and isinstance(item[1], numbers.Real)
        and not isinstance(item[1], bool)
    ):
        return tuple(item[0]), float(item[1])
    return tuple(item), 1.0


@dataclass(frozen=True)
class StateNetwork:
    """A memory network in Infomap's state format.

    Build one from paths with :meth:`from_paths`, then :meth:`write` it as a
    ``*States`` file for the Infomap CLI or hand it to the Python API with
    :meth:`to_infomap`. Instances are frozen; the dicts are the construction
    result and must not be mutated.
    """

    #: Markov order: how many physical nodes one state remembers.
    order: int
    #: Physical node id -> label, ids assigned from 1 in first-seen order.
    physical_nodes: dict[int, str] = field(hash=False)
    #: State id -> window of physical ids; the current node is the last entry.
    states: dict[int, tuple[int, ...]] = field(hash=False)
    #: ``(source_state_id, target_state_id)`` -> accumulated weight.
    links: dict[tuple[int, int], float] = field(hash=False)
    #: How many paths were consumed, including the too-short ones.
    num_paths: int = 0
    #: Paths shorter than ``order + 1`` nodes: they contain no state
    #: transition and contribute nothing. Nonzero here is not an error,
    #: but a large fraction means the chosen order overreaches the data --
    #: or, if the paths were cut to a node subset, that cutting fragmented
    #: them.
    num_short_paths: int = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_paths(cls, paths: Iterable[PathItem], *, order: int = 2) -> StateNetwork:
        """Accumulate paths into a state network of the given Markov order.

        Each item is a sequence of node labels, or a ``(sequence, weight)``
        pair -- itineraries flown by many passengers count that many times. A
        sliding window turns every ``(order + 1)``-gram into one state link:
        with the default order 2, ``i -> j -> k`` adds weight to the link
        ``(i, j) -> (j, k)``. Repeated transitions accumulate.

        Labels can be any hashable value; they are compared as given and
        rendered with ``str`` in output. Ids are assigned in first-seen order,
        so the same paths in the same order produce identical output files.

        Raises:
            ValueError: ``order`` is smaller than 1.
        """
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}.")

        physical_ids: dict[Hashable, int] = {}
        state_ids: dict[tuple[int, ...], int] = {}
        links: dict[tuple[int, int], float] = {}
        num_paths = 0
        num_short = 0

        for item in paths:
            nodes, weight = _coerce(item)
            num_paths += 1
            if len(nodes) < order + 1:
                num_short += 1
                continue
            ids = [physical_ids.setdefault(node, len(physical_ids) + 1) for node in nodes]
            previous: int | None = None
            for start in range(len(ids) - order + 1):
                window = tuple(ids[start : start + order])
                state = state_ids.setdefault(window, len(state_ids) + 1)
                if previous is not None:
                    key = (previous, state)
                    links[key] = links.get(key, 0.0) + weight
                previous = state

        return cls(
            order=order,
            physical_nodes={pid: str(label) for label, pid in physical_ids.items()},
            states={sid: window for window, sid in state_ids.items()},
            links=links,
            num_paths=num_paths,
            num_short_paths=num_short,
        )

    # -- inspection --------------------------------------------------------
    @property
    def num_physical_nodes(self) -> int:
        return len(self.physical_nodes)

    @property
    def num_state_nodes(self) -> int:
        return len(self.states)

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def total_weight(self) -> float:
        return sum(self.links.values())

    def state_name(self, state_id: int) -> str:
        """A readable name for one state: its memory, e.g. ``"LAX -> JFK"``."""
        window = self.states[state_id]
        return " -> ".join(self.physical_nodes[pid] for pid in window)

    @staticmethod
    def _printable(name: str) -> str:
        """Make a label safe inside a quoted name field.

        Whitespace is collapsed so a label cannot inject newlines into the
        file, and double quotes -- the field delimiters -- become single.
        """
        return " ".join(name.split()).replace('"', "'")

    # -- output ------------------------------------------------------------
    def write(self, target: str | Path) -> Path:
        """Write the Infomap states format: ``*Vertices``, ``*States``, ``*Links``.

        The file works directly as CLI input::

            infomap network_states.net output/ --directed

        Output is sorted by id, so equal networks produce equal files.
        """
        target = Path(target)
        with target.open("w", encoding="utf-8") as out:
            out.write(
                f"# State network with Markov order {self.order} "
                f"built from {self.num_paths - self.num_short_paths} paths\n"
            )
            out.write(f"*Vertices {len(self.physical_nodes)}\n")
            for pid in sorted(self.physical_nodes):
                out.write(f'{pid} "{self._printable(self.physical_nodes[pid])}"\n')
            out.write("*States\n")
            out.write("# state_id physical_id name\n")
            for sid in sorted(self.states):
                window = self.states[sid]
                out.write(f'{sid} {window[-1]} "{self._printable(self.state_name(sid))}"\n')
            out.write("*Links\n")
            for source, dest in sorted(self.links):
                # repr is the shortest round-trip form, so the file carries
                # exactly the weights to_infomap() would feed the API.
                out.write(f"{source} {dest} {self.links[(source, dest)]!r}\n")
        return target

    def to_infomap(self, **kwargs: Any) -> Any:
        """Build an ``infomap.Infomap`` instance with this state network loaded.

        Directed flow is the default -- transitions are inherently directed.
        Pass ``flow_model="rawdir"`` to use the accumulated counts as raw flow
        instead of smoothing them with the default PageRank iteration; which
        is right depends on whether the paths sample a stationary process.
        Requires the ``infomap`` extra::

            pip install "mapequation-networks[infomap]"
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "to_infomap() needs infomap. Install it with "
                '`pip install "mapequation-networks[infomap]"`.'
            ) from error

        kwargs.setdefault("directed", True)
        instance = Infomap(**kwargs)
        if self.order == 1:
            # States and physical nodes coincide, so hand Infomap the plain
            # directed network instead of a degenerate higher-order one.
            for pid, name in self.physical_nodes.items():
                instance.set_name(pid, name)
            for (source, dest), weight in self.links.items():
                instance.add_link(self.states[source][-1], self.states[dest][-1], weight)
            return instance
        for sid, window in self.states.items():
            instance.add_state_node(sid, window[-1])
        for pid, name in self.physical_nodes.items():
            instance.set_name(pid, name)
        for (source, dest), weight in self.links.items():
            instance.add_link(source, dest, weight)
        return instance

    def __str__(self) -> str:
        return (
            f"StateNetwork(order={self.order}): "
            f"{self.num_physical_nodes:,} physical nodes, "
            f"{self.num_state_nodes:,} state nodes, "
            f"{self.num_links:,} links from "
            f"{self.num_paths - self.num_short_paths:,} paths"
        )
