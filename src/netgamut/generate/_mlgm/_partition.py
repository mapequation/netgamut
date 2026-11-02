"""Partition sampling (vendored from MultilayerGM, see package docstring)."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from itertools import accumulate

from ._tensors import DependencyTensor, SubscriptIterator


def categorical(rng: random.Random, cum_weights: Sequence[float]) -> int:
    """Sample an index given cumulative probabilities (``cum_weights[-1] == 1``)."""
    p = rng.random()
    i = 0
    while p > cum_weights[i]:
        i += 1
    return i


def dirichlet(rng: random.Random, theta: float, n: int) -> list[float]:
    """Sample from a symmetric Dirichlet distribution with concentration ``theta``."""
    weights = [rng.gammavariate(theta, 1) for _ in range(n)]
    s = sum(weights)
    weights[:] = [w / s for w in weights]
    return weights


class DirichletNull:
    """Null distribution: per-layer categorical probabilities drawn from a Dirichlet."""

    def __init__(
        self, layers: Sequence[int], theta: float, n_sets: int, *, rng: random.Random
    ) -> None:
        """``layers`` is ``[a_1, ..., a_d]``, the number of layers for each aspect."""
        self.rng = rng
        self.weights: dict[tuple[int, ...], list[float]] = {}
        for layer in SubscriptIterator(layers):
            self.weights[layer] = list(accumulate(dirichlet(rng, theta, n_sets)))

    def __call__(self, node: tuple[int, ...]) -> int:
        return categorical(self.rng, self.weights[node[1:]])


def sample_partition(
    tensor: DependencyTensor,
    null: Callable[[tuple[int, ...]], int],
    *,
    updates: int = 100,
    initial_partition: Mapping[tuple[int, ...], int] | None = None,
    rng: random.Random,
) -> dict[tuple[int, ...], int]:
    """Sample a partition of state nodes ``(node, *layer)`` with the given dependencies."""
    if initial_partition is None:
        partition = {node: null(node) for node in tensor.state_nodes()}
    else:
        partition = {node: initial_partition[node] for node in tensor.state_nodes()}

    random_layers = list(tensor.random_aspect_layers())
    n_updates = 1 if len(random_layers) <= 1 else updates * len(random_layers)
    for ordered_layer in tensor.ordered_aspect_layers():
        for _ in range(n_updates):
            random_layer = rng.choice(random_layers)
            layer = tuple(o + r for o, r in zip(ordered_layer, random_layer, strict=False))
            for node in tensor.state_nodes(layer):
                update_node = tensor.getrandneighbour(node)
                if update_node == node:
                    partition[node] = null(node)
                else:
                    partition[node] = partition[update_node]
    return partition
