"""Dependency tensors (vendored from MultilayerGM, see package docstring)."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from itertools import accumulate, chain
from typing import Any

Node = tuple[int, ...]


class SubscriptIterator:
    """Iterate over all subscripts of a tensor of given shape in lexicographic order."""

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = shape
        self.state = [0 for _ in self.shape]
        if self.state:
            self.state[-1] = -1

    def __iter__(self) -> SubscriptIterator:
        return self

    def __next__(self) -> tuple[int, ...]:
        self._update_state(len(self.shape) - 1)
        return tuple(self.state)

    def _update_state(self, index: int) -> None:
        if index < 0:
            raise StopIteration
        if self.state[index] < self.shape[index] - 1:
            self.state[index] += 1
        else:
            self.state[index] = 0
            self._update_state(index - 1)


class DependencyTensor(ABC):
    """Base class: subclasses implement ``getrandneighbour`` (the interlayer dependencies)."""

    def __init__(
        self,
        shape: Sequence[int],
        aspect_types: str,
        state_nodes: Iterable[Node] | None = None,
        *,
        rng: random.Random,
    ) -> None:
        """``shape`` is ``[n, a_1, ..., a_d]``; ``aspect_types[i]`` is 'o'rdered or 'r'andom."""
        self.shape = shape
        self.aspect_types = aspect_types
        self.rng = rng
        if state_nodes is None:
            self._state_nodes = {
                lay: [(n, *lay) for n in range(shape[0])] for lay in SubscriptIterator(shape[1:])
            }
        else:
            self._state_nodes = {lay: [] for lay in SubscriptIterator(shape[1:])}
            for n in state_nodes:
                if all(mi > ni >= 0 for mi, ni in zip(self.shape, n, strict=False)):
                    self._state_nodes[n[1:]].append(n)
                else:
                    raise ValueError("Provided state nodes do not match `shape`")

    def state_nodes(self, layer: Node | None = None) -> Iterable[Node]:
        """All state nodes, or only those of ``layer``."""
        if layer is None:
            return chain.from_iterable(self._state_nodes.values())
        return self._state_nodes[layer]

    def ordered_aspect_layers(self) -> SubscriptIterator:
        """Layers over ordered aspects (random-aspect indices fixed at 0)."""
        shape = list(self.shape[1:])
        for i in range(len(shape)):
            if self.aspect_types[i] == "r":
                shape[i] = 1
        return SubscriptIterator(shape)

    def random_aspect_layers(self) -> SubscriptIterator:
        """Layers over random aspects (ordered-aspect indices fixed at 0)."""
        shape = list(self.shape[1:])
        for i in range(len(shape)):
            if self.aspect_types[i] == "o":
                shape[i] = 1
        return SubscriptIterator(shape)

    @abstractmethod
    def getrandneighbour(self, node: Node) -> Node:
        """Neighbour whose assignment ``node`` copies, or ``node`` itself to resample."""


class UniformMultiplex(DependencyTensor):
    """Layer-coupled, single random aspect, uniform interlayer dependencies."""

    def __init__(self, n_nodes: int, n_layers: int, p_hat: float, *, rng: random.Random) -> None:
        super().__init__((n_nodes, n_layers), "r", rng=rng)
        self.p_hat = p_hat

    def getrandneighbour(self, node: Node) -> Node:
        if self.rng.random() > self.p_hat:
            return node
        n = self.rng.randrange(0, self.shape[1] - 1)
        if n >= node[1]:
            n += 1
        return (node[0], n)


class Temporal(DependencyTensor):
    """Layer-coupled, single ordered aspect; a layer depends on the previous one."""

    def __init__(self, n_nodes: int, n_layers: int, p: Any, *, rng: random.Random) -> None:
        super().__init__((n_nodes, n_layers), "o", rng=rng)
        self.p: Sequence[float] = list(p) if isinstance(p, Iterable) else [p] * (n_layers - 1)

    def getrandneighbour(self, node: Node) -> Node:
        if node[1] == 0 or self.rng.random() > self.p[node[1] - 1]:
            return node
        return (node[0], node[1] - 1)


class MultiAspect(DependencyTensor):
    """Combine single-aspect tensors; tensor ``i`` is used with probability ``weights[i]``."""

    def __init__(
        self, tensors: Sequence[DependencyTensor], weights: Sequence[float], *, rng: random.Random
    ) -> None:
        nodes = tensors[0].shape[0]
        for t in tensors:
            if t.shape[0] != nodes:
                raise ValueError("All tensors need to have the same number of nodes")
            if len(t.shape) != 2:
                raise ValueError("All tensors need to have exactly one aspect")
        shape = [nodes]
        shape += [t.shape[1] for t in tensors]
        super().__init__(tuple(shape), "".join(t.aspect_types for t in tensors), rng=rng)
        self.calpha = list(accumulate(weights))
        a_sum = self.calpha[-1]
        if a_sum != 1:
            self.calpha[:] = [a / a_sum for a in self.calpha]
        self.alpha = [a / a_sum for a in weights]
        self.tensors = tensors

    def getrandneighbour(self, node: Node) -> Node:
        from ._partition import categorical  # deferred: _partition imports this module

        i = categorical(self.rng, self.calpha)
        n, a_i = self.tensors[i].getrandneighbour((node[0], node[i + 1]))
        out = list(node)
        out[0] = n
        out[i + 1] = a_i
        return tuple(out)


__all__ = ["DependencyTensor", "MultiAspect", "SubscriptIterator", "Temporal", "UniformMultiplex"]
