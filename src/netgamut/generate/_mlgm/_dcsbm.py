"""DCSBM edge sampler (vendored from MultilayerGM, see package docstring)."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Node = tuple[int, ...]


def power_law_sample(
    n: int, t: float, x_min: float, x_max: float, rng: np.random.RandomState
) -> np.ndarray:
    """Sample ``n`` values from a truncated power law with exponent ``t``."""
    y = rng.rand(n)
    if t != -1:
        return ((x_max ** (t + 1) - x_min ** (t + 1)) * y + x_min ** (t + 1)) ** (1 / (t + 1))
    return x_min * (x_max / x_min) ** y


def build_block_matrix(partition: Any, degrees: Any) -> np.ndarray:
    """Diagonal block matrix of summed expected degrees per group."""
    nc = max(partition) + 1
    block_matrix = np.zeros([nc, nc])
    for i, g in enumerate(partition):
        block_matrix[g, g] += degrees[i]
    return block_matrix


def block_model_sampler(
    block_matrix: np.ndarray,
    partition: Any,
    degrees: Any,
    neighbors: dict[Any, set[Any]],
    nodes: Sequence[Any],
    rng: np.random.RandomState,
    max_reject: int = 100,
) -> dict[Any, set[Any]]:
    """Add DCSBM edges to ``neighbors`` (node -> set of neighbours), in place."""
    partition = np.array(partition)
    degrees = np.array(degrees)
    n_nodes = len(partition)
    if n_nodes != len(degrees):
        raise ValueError("length mismatch between partition and node degrees")

    n_groups = max(partition) + 1
    partition_map: list[list[int]] = [[] for _ in range(n_groups)]
    for i, g in enumerate(partition):
        partition_map[g].append(i)
    group_sizes = [len(g) for g in partition_map]
    sigma = [degrees[g] / sum(degrees[g]) for g in partition_map]

    for g1 in range(n_groups):
        for g2 in range(g1, n_groups):
            w = block_matrix[g1, g2]
            if w > 0:
                m = rng.poisson(w)
                if g1 == g2:
                    dense = 2 * m > group_sizes[g1] * (group_sizes[g1] - 1)
                    m = int(np.ceil(m / 2))
                else:
                    dense = 2 * m > group_sizes[g1] * group_sizes[g2]

                if dense:
                    probabilities = w * np.outer(sigma[g1], sigma[g2])
                    probabilities[probabilities > 1] = 1
                    if g1 == g2:
                        for i in range(group_sizes[g1]):
                            ni = np.extract(
                                probabilities[i, i + 1 :] > rng.rand(1, group_sizes[g1] - i - 1),
                                np.arange(i + 1, group_sizes[g1]),
                            )
                            for j in ni:
                                neighbors[nodes[partition_map[g1][i]]].add(
                                    nodes[partition_map[g2][j]]
                                )
                                neighbors[nodes[partition_map[g2][j]]].add(
                                    nodes[partition_map[g1][i]]
                                )
                    else:
                        for i in range(group_sizes[g1]):
                            ni = np.flatnonzero(probabilities[i, :] > rng.rand(1, group_sizes[g2]))
                            for j in ni:
                                neighbors[nodes[partition_map[g1][i]]].add(
                                    nodes[partition_map[g2][j]]
                                )
                                neighbors[nodes[partition_map[g2][j]]].add(
                                    nodes[partition_map[g1][i]]
                                )
                else:
                    for _ in range(m):
                        is_neighbor = True
                        reject_count = 0
                        while is_neighbor and reject_count <= max_reject:
                            node1 = nodes[rng.choice(partition_map[g1], 1, p=sigma[g1])[0]]
                            node2 = nodes[rng.choice(partition_map[g2], 1, p=sigma[g2])[0]]
                            is_neighbor = node1 in neighbors[node2] or node1 == node2
                            reject_count += 1
                        if reject_count > max_reject:
                            warnings.warn(
                                "rejection threshold reached", RuntimeWarning, stacklevel=2
                            )
                            break
                        neighbors[node1].add(node2)
                        neighbors[node2].add(node1)
    return neighbors


def sample_edges(
    partition: Mapping[Node, int],
    *,
    mu: float = 0.1,
    k_min: float = 5,
    k_max: float = 70,
    t_k: float = -2,
    degrees: Mapping[Node, float] | None = None,
    rng: np.random.RandomState,
) -> list[tuple[Node, Node]]:
    """Intra-layer undirected edges of the multilayer DCSBM, as a sorted list of sorted pairs."""
    layer_partitions: dict[Node, dict[Node, int]] = defaultdict(dict)
    for node, p in partition.items():
        layer_partitions[tuple(node[1:])][tuple(node)] = p

    neighbors: dict[Node, set[Node]] = defaultdict(set)
    for lp in layer_partitions.values():
        n_nodes = len(lp)
        nodes_l = list(lp.keys())
        if degrees is None:
            deg = power_law_sample(n_nodes, t_k, k_min, k_max, rng)
        else:
            deg = np.array([degrees[node] for node in nodes_l])
        # random edges
        no_partition = np.zeros(n_nodes, dtype=int)
        block_matrix = build_block_matrix(no_partition, deg * mu)
        neighbors = block_model_sampler(
            block_matrix, no_partition, deg * mu, neighbors, nodes_l, rng
        )
        # community edges
        p_l = np.fromiter(lp.values(), dtype=int)
        block_matrix = build_block_matrix(p_l, deg * (1 - mu))
        neighbors = block_model_sampler(block_matrix, p_l, deg * (1 - mu), neighbors, nodes_l, rng)

    edges = {(a, b) if a <= b else (b, a) for a, adj in neighbors.items() for b in adj}
    return sorted(edges)
