"""Private, seedable vendoring of the MultilayerGM sampling code.

Source: https://github.com/MultilayerGM/MultilayerGM-py (BSD-2-Clause, see ``LICENSE``),
pinned at commit 49a192f709931ead8a26f3100029b7c7e0b48f74, copyright (c) 2019 Lucas Jeub.
Cite the method and the code as upstream asks:

    M. Bazzi, L. G. S. Jeub, A. Arenas, S. D. Howison, and M. A. Porter. Generative
    benchmark models for mesoscale structure in multilayer networks. arXiv:1608.06196.

    L. G. S. Jeub. A Python framework for generating multilayer networks with planted
    mesoscale structure. https://github.com/MultilayerGM/MultilayerGM-py, 2019.

Why vendored: the package is not on PyPI, it depends on a git-only ``nxmultilayer``, its
``__init__`` imports ``pkg_resources`` (gone from modern setuptools), and it draws from the
global ``random`` / ``numpy.random`` state, which makes seeded, side-effect-free use
impossible.

Faithfulness: the algorithm and the exact sequence of RNG calls are kept, so that
``random.Random(s)`` + ``numpy.random.RandomState(s)`` reproduce the partition and edge set
of upstream run after ``random.seed(s); numpy.random.seed(s)``. Global RNG state is never
touched.

Deviations from upstream:

- Instance RNGs (``random.Random`` for tensors/partition, ``numpy.random.RandomState`` for
  edges) replace the module-level ``random`` / ``np.random``.
- ``print('rejection threshold reached')`` is a ``warnings.warn``.
- Upstream's dense off-diagonal branch called ``np.random.rand([1, n])`` (a bug, raises);
  it is ``rng.rand(1, n)`` here. That branch is unreachable (the block matrix is diagonal),
  so no reachable RNG call order changes.
- The abstract ``DependencyTensor.getrandneighbour`` example body (broken upstream) is
  dropped, and the unimplemented ``BlockMultiplex`` stub is omitted.
- Partitions must be dicts (upstream also accepted numpy arrays).
- ``sample_edges`` returns sorted ``(state, state)`` pairs instead of building an
  ``nxmultilayer`` graph, so there is no networkx dependency.
"""
