"""Golden-value tests: the vendored MultilayerGM sampler matches upstream (pinned commit)."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from networks.generate._mlgm._dcsbm import sample_edges
from networks.generate._mlgm._partition import DirichletNull, sample_partition
from networks.generate._mlgm._tensors import (
    DependencyTensor,
    MultiAspect,
    Temporal,
    UniformMultiplex,
)

# (config, seed) -> (n_state_nodes, n_edges, partition_sha16, edges_sha16), recorded from
# upstream MultilayerGM-py 49a192f run after random.seed(s); numpy.random.seed(s).
GOLDEN = {
    ("uni", 1): (800, 3035, "64f5f127cd7fec34", "4a17e2d8074f95ca"),
    ("uni", 7): (800, 3168, "ed303644da510b1d", "69d423d601a7a957"),
    ("uni", 42): (800, 3032, "d06094fe6b117313", "89ed45fba8b32bcf"),
    ("temp", 1): (1200, 4607, "380947cded46949a", "76656316121f46b8"),
    ("temp", 7): (1200, 4530, "a1e6b8dbc4d60647", "98a62381d75cae2a"),
    ("temp", 42): (1200, 4431, "e7630ff2142331f7", "071a319e06b3ff4d"),
    ("multi", 1): (2400, 9283, "6051fa5f0dc7c568", "266cf29dccf462b2"),
    ("multi", 7): (2400, 9306, "ebb09249e0c45aeb", "f8b679ef061748e0"),
    ("multi", 42): (2400, 9437, "cd29d94e840439a6", "c77926ac267e2f77"),
    ("uni_deg", 1): (800, 4993, "64f5f127cd7fec34", "2d4ed0d8f17e8743"),
    ("uni_deg", 7): (800, 4966, "ed303644da510b1d", "baf4977601642b41"),
    ("uni_deg", 42): (800, 4997, "d06094fe6b117313", "c3668e2b1f8693e5"),
    ("dense", 1): (60, 406, "773861cf558a897a", "260f74a96fb9f52a"),
    ("dense", 7): (60, 297, "9938d1aae8259f24", "b030cb267fe6b131"),
    ("dense", 42): (60, 317, "7928e0ad4874b876", "6b1769e382c54a55"),
}


def _h(x: object) -> str:
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def _run(
    config: str, seed: int
) -> tuple[dict[tuple[int, ...], int], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    rr = random.Random(seed)
    rn = np.random.RandomState(seed)
    tensor: DependencyTensor
    layers: tuple[int, ...]
    if config in ("uni", "uni_deg"):
        tensor, layers = UniformMultiplex(200, 4, 0.9, rng=rr), (4,)
    elif config == "temp":
        tensor, layers = Temporal(200, 6, 0.7, rng=rr), (6,)
    elif config == "multi":
        tensor = MultiAspect(
            [UniformMultiplex(200, 3, 0.9, rng=rr), Temporal(200, 4, 0.7, rng=rr)], [1, 2], rng=rr
        )
        layers = (3, 4)
    else:  # dense
        tensor, layers = UniformMultiplex(30, 2, 0.5, rng=rr), (2,)
    null = DirichletNull(layers, 1, 5, rng=rr)
    part = sample_partition(tensor, null, rng=rr)
    degrees = None
    if config == "uni_deg":
        degrees = {k: 3 + (i % 20) for i, k in enumerate(part)}
    elif config == "dense":
        degrees = {k: 25 for k in part}
    edges = sample_edges(part, mu=0.2, k_min=3, k_max=30, t_k=-2, degrees=degrees, rng=rn)
    return part, edges


@pytest.mark.parametrize(("config", "seed"), sorted(GOLDEN))
def test_matches_upstream(config: str, seed: int) -> None:
    part, edges = _run(config, seed)
    got = (len(part), len(edges), _h(sorted(part.items())), _h(edges))
    assert got == GOLDEN[(config, seed)]


def test_same_seed_identical() -> None:
    assert _run("multi", 3) == _run("multi", 3)
    assert _run("multi", 3) != _run("multi", 4)


def test_edges_sorted_pairs() -> None:
    _, edges = _run("uni", 1)
    assert edges == sorted(edges)
    assert all(a < b for a, b in edges)


def test_global_rng_untouched() -> None:
    random.seed(123)
    np.random.seed(123)
    py_state, np_state = random.getstate(), np.random.get_state()
    _run("temp", 5)
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert (after[1] == np_state[1]).all()
    assert after[2:] == np_state[2:]
