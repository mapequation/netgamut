"""MultilayerGM's multilayer benchmarks behind one call.

Upstream (Bazzi, Jeub, Arenas, Howison & Porter) builds a benchmark in four
steps -- a dependency tensor, a Dirichlet null, a sampled partition, and a
degree-corrected SBM per layer. :func:`multilayer_gm` takes every parameter of
the four at once. Building the tensor and the null costs milliseconds; the
time goes into sampling partition and edges, so there is no generator object
to keep around. What is worth reusing is a *partition*: pass ``partition=``
to resample edges for a fixed planted structure, e.g. in a ``mu`` sweep.

The sampling code is vendored in :mod:`._mlgm`, with the seeding made local;
the same ``seed`` reproduces upstream's output exactly.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .._core.multilayer import MultilayerNetwork
from ._naming import run_name, settings

__all__ = ["multilayer_gm"]

_COUPLINGS = ("multiplex", "temporal")


def multilayer_gm(
    n: int,
    layers: int,
    *,
    p: float | Sequence[float],
    coupling: str = "multiplex",
    mu: float = 0.1,
    communities: int = 10,
    theta: float = 1.0,
    k_min: float = 5,
    k_max: float = 70,
    t_k: float = -2,
    degrees: Mapping[tuple[int, int], float] | None = None,
    updates: int = 100,
    partition: Mapping[tuple[int, int], int] | None = None,
    seed: int = 1,
) -> MultilayerNetwork:
    """Generate a multilayer benchmark with MultilayerGM's planted mesoscale structure.

    A partition of the ``n * layers`` state nodes is sampled first: each state
    node either copies the community of a state node of the same physical node
    in another layer (probability *p*) or draws a fresh one from a Dirichlet
    null over *communities* communities. Then each layer gets an undirected
    degree-corrected SBM with expected degrees from a truncated power law: a
    fraction *mu* of each node's edges are placed ignoring the communities,
    the rest inside its own.

    Args:
        n: Number of physical nodes; every one appears in every layer.
        layers: Number of layers.
        p: Copying probability -- how strongly a node's community in one layer
            depends on its others. For ``coupling="temporal"`` it may be a
            sequence of ``layers - 1`` probabilities, ``p[i - 1]`` for copying
            from layer ``i - 1`` into layer ``i``.
        coupling: ``"multiplex"`` -- copy from any other layer, uniformly
            (upstream's ``UniformMultiplex``) -- or ``"temporal"`` -- copy only
            from the previous layer (``Temporal``).
        mu: Fraction of edges placed independently of the communities. Not
            LFR's mixing parameter: a random edge can still land inside a
            community, so the realized fraction of edges leaving one is lower.
        communities: Number of communities the null distribution draws from
            (upstream's ``n_sets``).
        theta: Dirichlet concentration of the null; small values make some
            communities much larger than others in each layer.
        k_min: Lower cutoff of the expected-degree distribution.
        k_max: Upper cutoff of the expected-degree distribution.
        t_k: Exponent of that distribution -- the exponent itself, so ``-2``
            where LFR's ``t1`` would say ``2``.
        degrees: Expected degree per state node, ``{(node, layer): degree}``,
            replacing *k_min*, *k_max* and *t_k*.
        updates: Expected (pseudo-)Gibbs updates per state node when sampling
            the partition; no effect for ``"temporal"``, which is sampled in
            one ordered sweep.
        partition: A planted partition to reuse, ``{(node, layer): community}``
            -- typically ``earlier.partition``. Skips partition sampling, so
            *p*, *coupling*, *communities*, *theta* and *updates* have no
            effect; it must cover exactly ``n`` nodes in ``layers`` layers.
        seed: Seeds both random streams. The same seed gives the same network,
            identical to upstream's after ``random.seed(seed)`` and
            ``numpy.random.seed(seed)``; global random state is not touched.

    Returns:
        A :class:`~netgamut.MultilayerNetwork` with intra-layer links only --
        layers are coupled through the partition, so analyse it with a relax
        rate -- and ``partition`` ``{(node, layer): community}``.

    Raises:
        ValueError: Unknown *coupling*, or a *partition* that does not cover
            ``n`` nodes in ``layers`` layers.

    Example:
        >>> net = multilayer_gm(1000, 5, p=0.9, mu=0.2, k_min=3, k_max=150)  # doctest: +SKIP
        >>> net.to_pajek("benchmarks/")                                      # doctest: +SKIP
        PosixPath('benchmarks/mlgm-multiplex-n1000-L5-p0.9-mu0.2-seed1.net')
    """
    import numpy as np

    from ._mlgm._dcsbm import sample_edges
    from ._mlgm._partition import DirichletNull, sample_partition
    from ._mlgm._tensors import Temporal, UniformMultiplex

    if coupling not in _COUPLINGS:
        raise ValueError(f"coupling must be one of {_COUPLINGS}, not {coupling!r}")

    if partition is None:
        rng = random.Random(seed)
        tensor = (
            UniformMultiplex(n, layers, p, rng=rng)  # type: ignore[arg-type]
            if coupling == "multiplex"
            else Temporal(n, layers, p, rng=rng)
        )
        null = DirichletNull(tensor.shape[1:], theta, communities, rng=rng)
        planted = sample_partition(tensor, null, updates=updates, rng=rng)
    else:
        expected = {(node, layer) for layer in range(layers) for node in range(n)}
        if set(partition) != expected:
            raise ValueError(
                f"partition must have exactly the {n * layers:,} state nodes (node, layer) "
                f"with node < {n} and layer < {layers}."
            )
        # Upstream walks state nodes layer by layer; keep that order, since it
        # decides which random numbers each node's degree gets.
        planted = {
            (node, layer): partition[(node, layer)]
            for (node, layer) in sorted(expected, key=lambda state: (state[1], state[0]))
        }

    edges = sample_edges(
        planted,
        mu=mu,
        k_min=k_min,
        k_max=k_max,
        t_k=t_k,
        degrees=degrees,  # type: ignore[arg-type]
        rng=np.random.RandomState(seed),
    )
    intra: dict[Any, dict[tuple[Any, Any], float]] = {layer: {} for layer in range(layers)}
    for (u, layer), (v, _) in edges:
        intra[layer][(u, v)] = 1.0

    scalar_p = p if isinstance(p, (int, float)) else None
    recorded_p: Any = p if isinstance(p, (int, float)) else list(p)
    parameters = settings(
        {
            "n": n,
            "layers": layers,
            "coupling": coupling,
            "p": recorded_p,
            "mu": mu,
            "communities": communities,
            "theta": theta,
            "k_min": k_min,
            "k_max": k_max,
            "t_k": t_k,
            "degrees": "given" if degrees is not None else None,
            "updates": updates,
            "partition": "given" if partition is not None else None,
            "seed": seed,
        }
    )
    return MultilayerNetwork(
        name=run_name(f"mlgm-{coupling}", n=n, L=layers, p=scalar_p, mu=mu, seed=seed),
        intra=intra,
        partition={(state[0], state[1]): int(module) for state, module in planted.items()},
        info={"generator": "multilayer_gm", "parameters": parameters},
    )
