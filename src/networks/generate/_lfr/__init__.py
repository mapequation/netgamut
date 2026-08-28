"""The Lancichinetti--Fortunato LFR benchmarks, driven from Python.

:func:`lfr` runs the authors' own C++ generators -- fetched and compiled once per
machine, see :mod:`._sources` -- and hands back a networkx graph with its planted
communities.  Which of the six upstream programs runs follows from the arguments:

===========================  =========================================================
argument                     generator
===========================  =========================================================
``mu``                       ``unweighted_undirected`` (the classic binary benchmark)
``mu`` + ``directed=True``   ``unweighted_directed``
``muw`` (with ``mut``)       ``weighted_undirected``
``muw`` + ``directed=True``  ``weighted_directed``
``mu1``/``mu2``              ``hierarchical`` -- two planted levels
``layers``                   ``MultiplexBenchmark`` -- one graph sliced into layers
===========================  =========================================================

The programs are deterministic in their seed: they read it from ``time_seed.dat`` in
the working directory, so writing that file rather than letting them fall back to the
clock is what makes a run reproducible.  Same ``seed``, same network, byte for byte.

Node labels come back zero-based, like everything else in :mod:`networks.generate`,
not one-based as the ``.dat`` files write them.

Compared with :func:`~networks.generate.lfr_networkx`, which wraps networkx's
reimplementation: this is the reference implementation, it covers weighted, directed,
hierarchical and multiplex benchmarks rather than the undirected unweighted case
alone, it supports overlapping communities, and it does not fail to converge.  It
needs a C++ compiler and, once, a download.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._core.cache import cache_dir
from . import _io
from ._build import CompilerNotFound, binary, compiler
from ._sources import (
    COMMIT,
    REPOSITORY,
    SHA256,
    SOURCE_ENV_VAR,
    URL,
    VARIANTS,
    Variant,
)

__all__ = [
    "COMMIT",
    "URL",
    "VARIANTS",
    "CompilerNotFound",
    "LfrSource",
    "lfr",
    "lfr_available",
    "lfr_source",
]


def _resolve(
    *, directed: bool, weighted: bool, hierarchical: bool, multiplex: bool
) -> Variant:
    """Pick the upstream generator, refusing combinations upstream does not have."""
    if hierarchical and (directed or weighted or multiplex):
        raise ValueError(
            "mu1/mu2 select the hierarchical generator, which upstream provides only "
            "for undirected unweighted networks. Drop directed/muw/layers, or plant the "
            "hierarchy yourself with generate.benchmark on a hierarchical partition."
        )
    if multiplex and (directed or weighted):
        raise ValueError(
            "layers selects the multiplex generator, which slices the undirected "
            "unweighted benchmark. Drop directed/muw."
        )
    if hierarchical:
        return VARIANTS["hierarchical"]
    if multiplex:
        return VARIANTS["multiplex"]
    if weighted:
        return VARIANTS["weighted_directed" if directed else "weighted"]
    return VARIANTS["directed" if directed else "undirected"]


def _flags(**options: Any) -> list[str]:
    """Upstream's ``-flag value`` form, skipping anything left unset."""
    flags: list[str] = []
    for name, value in options.items():
        if value is None or value is False:
            continue
        flags.append(f"-{name}")
        if value is not True:
            flags.append(f"{value:.17g}" if isinstance(value, float) else str(value))
    return flags


def _run(
    variant: Variant,
    flags: Sequence[str],
    seed: int,
    timeout: float,
    progress: bool | None,
) -> Path:
    """Run *variant* in a scratch directory and return it, or raise with the log.

    The generators report infeasible parameters and **still exit 0**, leaving no
    output behind, so a run counts as successful only once every file it promised is
    on disk.
    """
    program = binary(variant, progress=progress)
    workdir = Path(tempfile.mkdtemp(prefix=f"lfr-{variant.key}-"))
    (workdir / "time_seed.dat").write_text(f"{int(seed)}\n")

    try:
        done = subprocess.run(
            [str(program), *flags],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        shutil.rmtree(workdir, ignore_errors=True)
        raise TimeoutError(
            f"The {variant.key} LFR generator ran longer than {timeout:g}s. A target "
            "clustering coefficient it cannot reach is the usual cause; raise timeout "
            "or drop `clustering`."
        ) from error

    missing = [name for name in variant.outputs if not (workdir / name).is_file()]
    if done.returncode != 0 or missing:
        log = (done.stdout or "") + (done.stderr or "")
        detail = "\n".join(line for line in log.splitlines() if line.strip())[-2000:]
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError(
            f"The {variant.key} LFR generator produced no benchmark "
            f"(exit {done.returncode}, missing {', '.join(missing) or 'nothing'}). "
            f"It reported:\n{detail}"
        )
    return workdir


@dataclass(frozen=True)
class LfrSource:
    """Which LFR revision is pinned, and whether this machine has it yet.

    Upstream publishes no releases, so the pin is a commit hash. Print this to see
    what a run would use; ``scripts/update_lfr_pin.py`` is what moves it.
    """

    repository: str
    commit: str
    sha256: str
    url: str
    directory: Path
    fetched: bool
    overridden: bool
    compiler: str | None
    variants: tuple[str, ...]

    def __str__(self) -> str:
        where = "from " + SOURCE_ENV_VAR if self.overridden else "pinned"
        state = "fetched" if self.fetched else "not fetched yet"
        toolchain = self.compiler or "no C++ compiler found"
        return (
            f"{self.repository}@{self.commit[:12]} ({where}, {state})\n"
            f"  url       {self.url}\n"
            f"  sha256    {self.sha256}\n"
            f"  sources   {self.directory}\n"
            f"  compiler  {toolchain}\n"
            f"  variants  {', '.join(self.variants)}"
        )


def lfr_source() -> LfrSource:
    """Report the pinned LFR revision and whether :func:`lfr` could run right now.

    Never downloads and never compiles, so it is safe to call for a status line::

        >>> print(generate.lfr_source())  # doctest: +SKIP
        andrealancichinetti/LFRbenchmarks@ec9a860282d8 (pinned, fetched)
          url       https://github.com/.../archive/ec9a8602....tar.gz
          sha256    ad13a354c2cf...
          sources   ~/.cache/networks/lfr/src/ec9a8602...
          compiler  Apple clang version 17.0.0
          variants  undirected, weighted, directed, ...

    To move the pin, run ``python scripts/update_lfr_pin.py`` in a checkout of this
    repository; to run one revision without moving it, set ``NETWORKS_LFR_DIR``.
    """
    override = os.environ.get(SOURCE_ENV_VAR)
    directory = Path(override).expanduser() if override else cache_dir() / "lfr" / "src" / COMMIT
    try:
        found: str | None = " ".join(compiler())
    except CompilerNotFound:
        found = None
    return LfrSource(
        repository=REPOSITORY,
        commit=COMMIT,
        sha256=SHA256,
        url=URL,
        directory=directory,
        fetched=directory.is_dir(),
        overridden=bool(override),
        compiler=found,
        variants=tuple(VARIANTS),
    )


def lfr_available() -> bool:
    """Whether :func:`lfr` can run here: sources on hand and a compiler to build them.

    Never downloads, so a ``False`` may just mean "not fetched yet". Use it to skip
    LFR work rather than to catch the failure.
    """
    source = lfr_source()
    return source.compiler is not None and source.fetched


def lfr(
    n: int,
    k: float,
    maxk: int,
    *,
    mu: float | None = None,
    mut: float | None = None,
    muw: float | None = None,
    beta: float | None = None,
    t1: float = 2.0,
    t2: float = 1.0,
    minc: int | None = None,
    maxc: int | None = None,
    on: int = 0,
    om: int = 0,
    clustering: float | None = None,
    directed: bool = False,
    mu1: float | None = None,
    mu2: float | None = None,
    min_macro: int | None = None,
    max_macro: int | None = None,
    layers: int | None = None,
    graphs: int = 1,
    bound: str | None = None,
    rand: bool = False,
    seed: int = 1,
    timeout: float = 600.0,
    progress: bool | None = None,
) -> tuple[Any, dict[Any, Any]]:
    """Generate an LFR benchmark with the authors' own generator.

    Args:
        n: Number of nodes (``-N``).
        k: Average degree (``-k``); the average *in*-degree when ``directed``.
        maxk: Maximum degree (``-maxk``).
        mu: Mixing parameter for the unweighted variants (``-mu``) -- the fraction of
            a node's links that leave its community.
        mut: Topological mixing for the weighted variants (``-mut``); upstream
            defaults it to ``muw``.
        muw: Weight mixing for the weighted variants (``-muw``). Giving ``muw``,
            ``mut`` or ``beta`` is what selects a weighted generator.
        beta: Exponent of the weight distribution (``-beta``, upstream default 1.5).
        t1: Minus exponent of the degree sequence (``-t1``).
        t2: Minus exponent of the community-size distribution (``-t2``).
        minc: Smallest community (``-minc``); upstream picks one from the degree
            extremes when omitted.
        maxc: Largest community (``-maxc``).
        on: How many nodes belong to several communities (``-on``).
        om: How many communities each of those belongs to (``-om``).
        clustering: Target average clustering coefficient (``-C``); the generator
            rewires towards it and gives up, with a warning, if it cannot reach it.
            Undirected variants only.
        directed: Generate a directed benchmark.
        mu1: Fraction of links leaving the macro community (``-mu1``). Selects the
            hierarchical generator.
        mu2: Fraction of links inside the macro but outside the micro community
            (``-mu2``). Selects the hierarchical generator.
        min_macro: Smallest macro community (``-minC``), hierarchical only.
        max_macro: Largest macro community (``-maxC``), hierarchical only.
        layers: Layers per generated graph (``-L``). Selects the multiplex generator.
        graphs: How many independent graphs to slice into layers (``-Orig``); the
            result has ``graphs * layers`` layers in all.
        bound: ``"sup"`` or ``"inf"`` to make the mixing parameter the maximum or the
            minimum of the external-degree ratio rather than its mean (``-sup``,
            ``-inf``).
        rand: Ask for a random network instead (``-rand``): mixing zero, one community.
        seed: Random seed, written to ``time_seed.dat``. Fully determines the output.
        timeout: Seconds before the generator is abandoned.
        progress: Show a download bar on the first call; defaults to on for a tty.

    Returns:
        ``(G, partition)``, both zero-based.

        * flat, no overlap: ``{node: module}``
        * hierarchical: ``{node: (macro, micro)}``, the path convention
          :func:`~networks.generate.flow_matrix` and friends take
        * overlapping (``on > 0``): ``{node: (module, ...)}``, a cover rather than a
          partition
        * multiplex: ``G`` is ``{layer: graph}`` and the partition is
          ``{(node, layer): module}``, as
          :func:`~networks.generate.multilayer_benchmark` expects; communities are
          numbered per generated graph, so layers sliced from different graphs never
          share a module.

    Raises:
        ValueError: Arguments select a generator upstream does not provide, or leave a
            required mixing parameter unset.
        CompilerNotFound: No C++ compiler to build the generator with.
        RuntimeError: The generator refused the parameters -- its own message is
            included, since it reports infeasible combinations rather than guessing.
        TimeoutError: The generator ran past *timeout*.

    Example:
        >>> G, partition = lfr(1000, 15, 50, mu=0.2, minc=20, maxc=50, seed=1)
        ... # doctest: +SKIP
    """
    weighted = muw is not None or mut is not None or beta is not None
    hierarchical = mu1 is not None or mu2 is not None
    multiplex = layers is not None
    variant = _resolve(
        directed=directed, weighted=weighted, hierarchical=hierarchical, multiplex=multiplex
    )

    if bound not in (None, "sup", "inf"):
        raise ValueError(f'bound must be None, "sup" or "inf", not {bound!r}')
    if clustering is not None and variant.directed:
        raise ValueError(
            "upstream's directed generators have no -C flag, so a target clustering "
            "coefficient cannot be requested for them."
        )
    if weighted and mu is not None:
        raise ValueError("the weighted generators take mut and muw, not mu.")
    if not weighted and not hierarchical and mu is None and not rand:
        raise ValueError("mu is required for the unweighted generators.")
    if weighted and muw is None:
        raise ValueError("muw is required for the weighted generators (mut defaults to it).")

    common = _flags(N=n, k=k, maxk=maxk, t1=t1, t2=t2, minc=minc, maxc=maxc, on=on, om=om)
    if hierarchical:
        specific = _flags(mu1=mu1, mu2=mu2, minC=min_macro, maxC=max_macro, C=clustering)
    elif weighted:
        specific = _flags(mut=mut, muw=muw, beta=beta, C=clustering)
    else:
        specific = _flags(
            mu=mu, C=clustering, L=layers, Orig=graphs if multiplex else None
        )
    extra = [f"-{bound}"] if bound else []
    if rand:
        extra.append("-rand")

    workdir = _run(variant, [*common, *specific, *extra], seed, timeout, progress)
    try:
        if variant.multiplex:
            return _read_multiplex(workdir, int(layers or 0), graphs)
        if variant.hierarchical:
            return _read_hierarchical(workdir)
        return _read_flat(workdir, variant)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _read_flat(workdir: Path, variant: Variant) -> tuple[Any, dict[Any, Any]]:
    """``network.dat`` + ``community.dat`` to a graph and a zero-based partition."""
    memberships = _io.read_memberships(workdir / "community.dat")
    G = _io.read_network(
        workdir / "network.dat", directed=variant.directed, weighted=variant.weighted
    )
    G.add_nodes_from(memberships)  # a node LFR left unlinked still belongs somewhere

    nodes = _io.compact(G.nodes())
    modules = _io.compact(c for ids in memberships.values() for c in ids)
    paths = {
        nodes[u]: [(modules[c],) for c in ids]
        for u, ids in memberships.items()
        if u in nodes
    }
    return _io.relabel(G, nodes), _io.as_partition(paths)


def _read_hierarchical(workdir: Path) -> tuple[Any, dict[Any, Any]]:
    """The two level files to ``{node: (macro, micro)}`` path tuples."""
    micro = _io.read_memberships(workdir / "community_first_level.dat")
    macro = _io.read_memberships(workdir / "community_second_level.dat")
    G = _io.read_network(workdir / "network.dat", directed=False, weighted=False)
    G.add_nodes_from(micro)

    parent = _io.macro_of_micro(micro, macro)
    nodes = _io.compact(G.nodes())
    outer = _io.compact(parent.values())
    inner = _io.compact(parent)
    paths = {
        nodes[u]: [(outer[parent[c]], inner[c]) for c in ids if c in parent]
        for u, ids in micro.items()
        if u in nodes
    }
    return _io.relabel(G, nodes), _io.as_partition(paths)


def _read_multiplex(workdir: Path, layers: int, graphs: int) -> tuple[Any, dict[Any, Any]]:
    """The per-layer files to ``({layer: graph}, {(node, layer): module})``.

    Layer ``i`` is a slice of generated graph ``i // layers`` and inherits its
    communities, so module ids are namespaced by that graph: two layers cut from
    different graphs never share a module even though upstream numbers both from one.
    """
    width = max(1, layers)
    per_layer: dict[int, tuple[Any, dict[int, tuple[int, ...]]]] = {}
    for index in range(width * max(1, graphs)):
        edges = workdir / f"network_layer_{index}"
        if not edges.is_file():
            break
        per_layer[index] = (
            _io.read_network(edges, directed=False, weighted=False),
            _io.read_memberships(workdir / f"community_layer_{index}"),
        )

    all_nodes = {u for G, members in per_layer.values() for u in (*G.nodes(), *members)}
    nodes = _io.compact(all_nodes)
    keys = sorted(
        {
            (index // width, c)
            for index, (_, members) in per_layer.items()
            for ids in members.values()
            for c in ids
        }
    )
    modules = {key: value for value, key in enumerate(keys)}

    sliced: dict[int, Any] = {}
    partition: dict[Any, Any] = {}
    for index, (G, members) in per_layer.items():
        G.add_nodes_from(members)
        sliced[index] = _io.relabel(G, nodes)
        origin = index // width
        for u, ids in members.items():
            paths = [(modules[(origin, c)],) for c in ids]
            partition[(nodes[u], index)] = _io.as_partition({0: paths})[0]
    return sliced, partition
