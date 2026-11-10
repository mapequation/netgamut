"""Multilayer networks in the shape Infomap's multilayer model reads.

A multilayer network has physical nodes that appear in several layers. Each
appearance is a *state node* ``(node, layer)``; links inside a layer join state
nodes of that layer, and inter-layer links join state nodes across layers. A
multiplex in the strict sense has no inter-layer links in the data at all --
Infomap couples its layers with a relax rate at analysis time -- which is what
every generator in :mod:`netgamut.generate` produces: the dependence between
layers lives in the planted partition, not in links.

Partitions are keyed by state node, ``{(node, layer): module}``, which is how
Infomap reports a multilayer partition and how the samplers take one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import formats

if TYPE_CHECKING:
    from .network import Network

__all__ = ["MultilayerNetwork"]

#: A state node: ``(physical node, layer)``.
StateNode = tuple[Hashable, Hashable]


@dataclass(frozen=True)
class MultilayerNetwork:
    """Intra-layer links per layer, optional inter-layer links, planted partition.

    Instances are frozen; the containers are the generator's result and must
    not be mutated.
    """

    #: Used as the file name by :meth:`to_pajek`.
    name: str
    #: ``{layer: {(u, v): weight}}``, weight 1.0 throughout when unweighted.
    intra: dict[Hashable, dict[tuple[Hashable, Hashable], float]] = field(hash=False)
    #: ``{((node, layer), (node, layer)): weight}``, both ends on the same
    #: physical node -- Infomap's ``*Inter`` links, which it spreads to the
    #: node's neighbours in the target layer. Empty for a multiplex whose
    #: layers are coupled by the relax rate rather than by links.
    inter: dict[tuple[StateNode, StateNode], float] = field(default_factory=dict, hash=False)
    directed: bool = False
    #: Whether link weights are worth writing; multilayer files always carry a
    #: weight column, so this only affects :meth:`to_networkx`.
    weighted: bool = False
    #: The planted ground truth, ``{(node, layer): module}``.
    partition: dict[StateNode, Any] | None = field(default=None, hash=False)
    #: Display names for physical nodes, where they differ from the labels.
    names: dict[Hashable, str] = field(default_factory=dict, hash=False)
    #: ``"generator"`` and ``"parameters"`` say how this was made; the rest is
    #: the generator's own diagnostics.
    info: dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        crossing = [(a, b) for a, b in self.inter if a[0] != b[0]]
        if crossing:
            raise ValueError(
                f"inter-layer links must stay on one physical node, as Infomap's *Inter "
                f"links do; {crossing[0]} joins two different nodes."
            )

    @classmethod
    def from_networks(
        cls,
        layers: Mapping[Any, Network],
        *,
        key: str | None = None,
        name: str | None = None,
        directed: bool | None = None,
    ) -> MultilayerNetwork:
        """Stack downloaded networks as the layers of one multilayer network.

        Each network in *layers* becomes the layer its key names. Its links are
        summed as :meth:`Network.to_multilayer` sums them: occurrences, or
        weights where the file has a weight column. There are no inter-layer
        links, so Infomap couples the layers with its relax rate.

        Without *key*, a node is the same physical node in every layer where it
        has the same id. That fits files that share one id space. Where each
        network numbers its nodes on its own, as Netzschleuder's subgraphs do,
        *key* names the column of :meth:`Network.nodes` that identifies a node
        across them, such as copenhagen's ``id``. The source's node names come
        along, for :meth:`to_pajek`'s ``*Vertices`` labels.

        *name* defaults to the first network's, and *directed* to whether any
        of them is directed.

        Raises:
            ValueError: No layers, or under *key* a node with no value or two
                nodes of one layer with the same value.
            LookupError: A network has no node metadata, or no *key* column.
        """
        from .columns import is_int_label

        if not layers:
            raise ValueError("no networks to stack as layers.")
        if directed is None:
            directed = any(bool(network.dataset.directed) for network in layers.values())
        identities: dict[Hashable, dict[str, str] | None] = {}
        labels: dict[str, str] = {}
        for layer, network in layers.items():
            if key is None:
                identities[layer] = None
                continue
            table = network.nodes()
            values = {str(node): str(value) for node, value in table[key].items()}
            shared = [value for value, count in Counter(values.values()).items() if count > 1]
            if shared:
                raise ValueError(
                    f"{key!r} does not identify the nodes of layer {layer!r}: "
                    f"{len(shared):,} values, such as {shared[0]!r}, belong to more than one node."
                )
            identities[layer] = values
            for node, label in table.labels().items():
                labels.setdefault(values.get(str(node), ""), label)
        raw: dict[Hashable, dict[tuple[str, str], float]] = {}
        for layer, network in layers.items():
            identity = identities[layer]
            links: dict[tuple[str, str], float] = {}
            for u, v, weight in network._links():
                if identity is not None:
                    try:
                        u, v = identity[u], identity[v]
                    except KeyError as missing:
                        raise ValueError(
                            f"node {missing.args[0]} of layer {layer!r} has no {key!r}."
                        ) from None
                pair = (u, v)
                if not directed and pair not in links and (v, u) in links:
                    pair = (v, u)
                links[pair] = links.get(pair, 0.0) + (1.0 if weight is None else weight)
            raw[layer] = links
        everyone = {node for links in raw.values() for pair in links for node in pair}
        cast: Any = int if all(is_int_label(node) for node in everyone) else str
        intra = {}
        for layer in list(raw):  # popped as it goes, so a large network is held once
            intra[layer] = {(cast(u), cast(v)): w for (u, v), w in raw.pop(layer).items()}
        names = {cast(node): label for node, label in labels.items() if node and label != node}
        first = next(iter(layers.values()))
        return cls(
            name=name or first.name,
            intra=intra,
            directed=directed,
            weighted=any(w != 1.0 for links in intra.values() for w in links.values()),
            names=names,
            info={"source": name or first.name, "layers": "networks", "key": key},
        )

    # -- inspection --------------------------------------------------------
    @property
    def layers(self) -> list[Hashable]:
        """Every layer, integers in numeric order."""
        found: list[Hashable] = list(self.intra)
        found += [layer for pair in self.inter for _, layer in pair]
        if self.partition:
            found += [layer for _, layer in self.partition]
        return formats.ordered(found)

    def state_nodes(self) -> list[StateNode]:
        """Every ``(node, layer)``: link endpoints plus partitioned isolates."""
        seen: dict[StateNode, None] = {}
        if self.partition:
            seen.update(dict.fromkeys(self.partition))
        for layer, links in self.intra.items():
            for u, v in links:
                seen[(u, layer)] = None
                seen[(v, layer)] = None
        for a, b in self.inter:
            seen[a] = None
            seen[b] = None
        return list(seen)

    @property
    def physical_nodes(self) -> list[Hashable]:
        return list(dict.fromkeys(node for node, _ in self.state_nodes()))

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_state_nodes(self) -> int:
        return len(self.state_nodes())

    @property
    def num_links(self) -> int:
        return sum(len(links) for links in self.intra.values()) + len(self.inter)

    # -- conversions -------------------------------------------------------
    def to_networkx(
        self, *, layer: Hashable | None = None, partition_attribute: str | None = "module"
    ) -> Any:
        """Build a networkx graph of the whole network or of one layer.

        With *layer* ``None``, the supra-graph: nodes are ``(node, layer)``
        tuples carrying ``node`` and ``layer`` attributes, intra-layer edges
        join state nodes of one layer and inter-layer edges are marked
        ``interlayer=True``. With a *layer*, that layer alone over physical
        nodes -- what ``{layer: graph}`` used to hold. Either way each node's
        planted module lands under *partition_attribute* (``None`` to skip).
        """
        try:
            import networkx
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                'to_networkx() needs networkx. Install it with `pip install "netgamut[networkx]"`.'
            ) from error

        graph = networkx.DiGraph() if self.directed else networkx.Graph()
        partition = self.partition or {}
        if layer is not None:
            graph.name = f"{self.name} layer {layer}"
            graph.add_nodes_from(node for node, at in self.state_nodes() if at == layer)
            for (u, v), w in self.intra.get(layer, {}).items():
                graph.add_edge(u, v, **({"weight": w} if self.weighted else {}))
            if partition_attribute is not None:
                for node in graph:
                    if (node, layer) in partition:
                        graph.nodes[node][partition_attribute] = partition[(node, layer)]
            return graph

        graph.name = self.name
        for node, at in self.state_nodes():
            graph.add_node((node, at), node=node, layer=at)
            if partition_attribute is not None and (node, at) in partition:
                graph.nodes[(node, at)][partition_attribute] = partition[(node, at)]
        for at, links in self.intra.items():
            for (u, v), w in links.items():
                graph.add_edge((u, at), (v, at), **({"weight": w} if self.weighted else {}))
        for (a, b), w in self.inter.items():
            graph.add_edge(a, b, interlayer=True, **({"weight": w} if self.weighted else {}))
        return graph

    def to_infomap(self, **kwargs: Any) -> Any:
        """Build an :class:`infomap.Infomap` instance with this network loaded.

        Pass ``multilayer_relax_rate`` (and friends) to control how Infomap
        couples layers that have no inter-layer links. Ids follow
        :func:`netgamut._core.formats.node_ids`: kept when they are
        non-negative integers, renumbered otherwise.
        """
        try:
            from infomap import Infomap
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                'to_infomap() needs infomap. Install it with `pip install "netgamut[infomap]"`.'
            ) from error

        kwargs.setdefault("directed", self.directed)
        instance = Infomap(**kwargs)
        node_id = formats.node_ids(self.physical_nodes)
        layer_id = formats.node_ids(self.layers)
        for label, index in node_id.items():
            name = self.names.get(label, None if label == index else str(label))
            if name is not None:
                instance.set_name(index, name)
        for layer, links in self.intra.items():
            for (u, v), w in links.items():
                instance.add_multilayer_intra_link(layer_id[layer], node_id[u], node_id[v], w)
        for ((u, a), (_, b)), w in self.inter.items():
            instance.add_multilayer_inter_link(layer_id[a], node_id[u], layer_id[b], w)
        return instance

    # -- output ------------------------------------------------------------
    def to_pajek(self, directory: str | Path, *, name: str | None = None) -> Path:
        """Write ``<directory>/<name>.net`` in Infomap's multilayer format.

        ``*Vertices``, then ``*Intra``, and ``*Inter`` when there are
        inter-layer links. Returns the path.
        """
        target = formats.output_path(directory, name or self.name, ".net")
        return formats.write_multilayer(
            target,
            intra=self.intra,
            inter=self.inter,
            nodes=self.physical_nodes,
            layers=self.layers,
            names=self.names,
            comments=formats.describe(self.name, self.info),
        )

    def to_link_list(self, directory: str | Path, *, name: str | None = None) -> Path:
        """Not available: a link list cannot say which layer a link is in.

        Raises:
            TypeError: Always; use :meth:`to_pajek`, whose ``*Intra``/``*Inter``
                sections are the multilayer equivalent.
        """
        raise TypeError(
            "Infomap reads multilayer networks only in its sectioned format "
            "(*Intra/*Inter); use to_pajek()."
        )

    def edges(self) -> Iterator[tuple[StateNode, StateNode]]:
        """Yield every link as a ``((node, layer), (node, layer))`` pair."""
        for layer, links in self.intra.items():
            for u, v in links:
                yield (u, layer), (v, layer)
        yield from self.inter

    def __str__(self) -> str:
        planted = ", planted partition" if self.partition is not None else ""
        return (
            f"MultilayerNetwork {self.name!r}: {self.num_layers:,} layers, "
            f"{self.num_state_nodes:,} state nodes over {len(self.physical_nodes):,} "
            f"physical nodes, {self.num_links:,} links{planted}"
        )
