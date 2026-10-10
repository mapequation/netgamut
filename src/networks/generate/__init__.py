"""Generative benchmarks from the modular flow of real networks.

Community-detection benchmarks are usually synthetic: the ground truth is
planted, but the networks look nothing like the systems the methods are for.
This subpackage inverts the map equation's flow model so that any network in
the pool can serve as a *blueprint* -- read its modular flow (degrees, module
sizes, per-module mixing), then sample fresh networks with the same
statistics and a known planted partition::

    import networks
    from networks import generate

    G = networks.load("polbooks").to_networkx()
    partition = generate.detect(G)             # or bring your own
    net = generate.benchmark(G, partition)

Every generator returns a network object -- a :class:`~networks.InMemoryNetwork`,
a :class:`~networks.StateNetwork` or a :class:`~networks.MultilayerNetwork` --
with the same methods as a downloaded :class:`~networks.Network`. The planted
partition is ``net.partition``, the sampler's diagnostics are ``net.info``::

    H = net.to_networkx()              # the planted module is node attribute "module"
    net.to_pajek("benchmarks/")        # -> Path("benchmarks/polbooks-benchmark.net")

Three routes, increasingly detached from the blueprint:

* :func:`benchmark` and :func:`weighted_benchmark` resample a blueprint
  directly -- microcanonical stub matching that preserves the degree
  sequence, each module's mixing and connectivity, for directed and
  undirected, two-level and hierarchical blueprints alike; the weighted
  variant adds flow-model edge weights, which is what dense weighted
  blueprints need to stay recoverable.
* :func:`fit_families` and :func:`fit_mixing_beta` reduce a blueprint to a
  handful of fitted parameters, and :func:`parametric` generates from those
  alone -- a generalized LFR whose degree and community-size families are
  chosen by AIC instead of assumed to be power laws.
* :func:`lfr` runs the Lancichinetti--Fortunato LFR generators themselves --
  the authors' C++, fetched and compiled on first use -- covering the binary,
  weighted, directed, hierarchical and multiplex benchmarks, with overlapping
  communities and a seed that fully determines the network.
  :func:`lfr_networkx` keeps networkx's reimplementation of the binary case,
  with retries around its flaky convergence, for callers who would rather not
  have a compiler in the loop.
* :func:`multilayer_gm` runs MultilayerGM (Bazzi, Jeub et al.): a multilayer
  partition sampled with tunable dependence between layers, multiplex or
  temporal, then a degree-corrected SBM in every layer.

Higher-order blueprints keep the same flow model and change only the sampler:
:func:`memory_benchmark` and :func:`multilayer_benchmark` pair *typed* stubs,
so a memory link matches at its physical junction and a multiplex link stays
inside its layer.  :func:`overlapping_memory_benchmark` needs no blueprint: it plants
communities that overlap on the physical nodes and are disjoint on the states --
Lancichinetti's trigram sampler, seeded, and the one generator here that needs only the
standard library.

The flow model is public too -- :class:`ModuleTree`, :func:`flow_matrix`,
:func:`cross_R` and the flow-model distances -- for callers that want the
joint distribution ``P`` rather than a sample of it.  Comparing two partitions
*as* flow models is :func:`gmd` (the generative map distance) with its
similarity :func:`gms`, and the adjusted pair :func:`agmd` / :func:`agms`
calibrated against the one-module null partition.

Everything here needs numpy and networkx (Infomap only when no partition is
given, scipy only for :func:`fit_families`)::

    pip install "mapequation-networks[generate]"

The samplers implement the generative map equation manuscript and are kept
faithful to the code behind its published numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._naming import run_name, settings

if TYPE_CHECKING:
    from .._core.in_memory import InMemoryNetwork
    from .._core.multilayer import MultilayerNetwork
    from .._core.state_network import StateNetwork

__all__ = [
    # high-level generation
    "benchmark",
    "detect",
    "fit_clustering_profile",
    "fit_families",
    "fit_mixing_beta",
    "LfrSource",
    "lfr",
    "lfr_available",
    "lfr_networkx",
    "lfr_source",
    "memory_benchmark",
    "multilayer_benchmark",
    "multilayer_gm",
    "overlapping_memory_benchmark",
    "parametric",
    "weighted_benchmark",
    # the flow model itself
    "ModuleTree",
    "apportion",
    "as_paths",
    "common_prefix_len",
    "cross_R",
    "directed_mixing_stats",
    "flow_matrix",
    "mixing_stats",
    "node_flows",
    "stationary_flows",
    # flow-model distances
    "D_dist",
    "S_dist",
    "element_distances",
    "jsd",
    # partition distances built on the flow model (GMD/GMS and adjusted AGMD/AGMS)
    "D_adjusted",
    "D_blockwise",
    "D_null",
    "agmd",
    "agms",
    "gmd",
    "gms",
    "null_partition",
    # samplers and repair passes, for callers that drive them directly
    "connect_components",
    "pair_stubs",
    "raise_clustering",
    "sample_benchmark",
    "sample_multinomial",
    "sample_state_links",
    "sample_stub_matching",
    "sample_stub_matching_directed",
    # Infomap helpers and IO
    "infomap_partition",
    "read_pajek",
    "run_infomap",
]

#: Internal modules that need only the standard library, so no extras check.
_STDLIB_ONLY = {"_overlapping_memory"}

#: Names re-exported from the sampler internals, resolved on first use so that
#: ``import networks`` keeps working without the ``[generate]`` extras.
_REEXPORTS = {
    **{
        name: "_flowgen"
        for name in (
            "ModuleTree",
            "apportion",
            "as_paths",
            "common_prefix_len",
            "connect_components",
            "cross_R",
            "directed_mixing_stats",
            "flow_matrix",
            "infomap_partition",
            "mixing_stats",
            "node_flows",
            "raise_clustering",
            "read_pajek",
            "run_infomap",
            "sample_benchmark",
            "sample_multinomial",
            "sample_stub_matching",
            "sample_stub_matching_directed",
            "stationary_flows",
            "D_dist",
            "S_dist",
            "element_distances",
            "jsd",
        )
    },
    **{
        name: "_partdist"
        for name in (
            "D_adjusted",
            "D_blockwise",
            "D_null",
            "agmd",
            "agms",
            "gmd",
            "gms",
            "null_partition",
        )
    },
    **{name: "_lfr" for name in ("LfrSource", "lfr", "lfr_available", "lfr_source")},
    "sample_state_links": "_higher_order",
    "multilayer_gm": "_multilayer_gm",
    "overlapping_memory_benchmark": "_overlapping_memory",
    "pair_stubs": "_flowgen",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-exported sampler name on first use (PEP 562)."""
    module = _REEXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if module not in _STDLIB_ONLY:
        _modules()  # raises the friendly ImportError when the extras are missing
    import importlib

    target = importlib.import_module(f".{module}", __name__)
    value = getattr(target, "_pair_stubs" if name == "pair_stubs" else name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


_INSTALL = 'pip install "mapequation-networks[generate]"'


def _modules() -> tuple[Any, Any]:
    """Import the sampler internals, failing with install advice."""
    try:
        from . import _flowgen, _params
    except ImportError as error:  # pragma: no cover - depends on install
        raise ImportError(
            f"networks.generate needs numpy and networkx. Install them with `{_INSTALL}`."
        ) from error
    return _flowgen, _params


def _require_infomap(flowgen: Any, what: str) -> None:
    """Fail with install advice when Infomap is needed but absent."""
    if flowgen.infomap is None:  # pragma: no cover - depends on install
        raise ImportError(f"{what} needs infomap. Install it with `{_INSTALL}`.")


def _blueprint(G: Any) -> Any:
    """A networkx graph from *G*, which may also be any network of this package."""
    return G.to_networkx() if hasattr(G, "to_networkx") else G


def _synthetic(
    graph: Any,
    *,
    name: str,
    partition: Any,
    generator: str,
    parameters: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> InMemoryNetwork:
    """Wrap a sampled graph, its planted partition restricted to the nodes it has."""
    from .._core.in_memory import InMemoryNetwork

    planted = None if partition is None else {u: partition[u] for u in graph if u in partition}
    return InMemoryNetwork.from_networkx(
        graph,
        name=name,
        partition=planted,
        info={"generator": generator, "parameters": settings(parameters), **(info or {})},
    )


def _pick_family(fit: dict[str, Any]) -> dict[str, Any]:
    """Normalize a fit to the ``{"family", "params", "support_min"}`` shape."""
    if "family" in fit:
        return fit
    winner = fit.get("winner")
    if winner in (None, "insufficient") or winner not in fit:
        raise ValueError(
            "fit has no usable family; pass a fit_families() result with a winner, "
            'or an explicit {"family", "params", "support_min"} dict'
        )
    return {
        "family": winner,
        "params": fit[winner]["params"],
        "support_min": fit.get("support_min", 1),
    }


def benchmark(G: Any, partition: dict[Any, Any] | None = None, **kwargs: Any) -> InMemoryNetwork:
    """Sample one benchmark network from a blueprint's modular flow.

    The default sampler is microcanonical stub matching: degrees follow the
    blueprint's visit rates, each module keeps its observed mixing
    ``mu = q/p`` (exactly the LFR mixing parameter), and boundary stubs route
    through the module tree under the non-redundant constraint. The result is
    a simple connected graph with *partition* planted as ground truth. With
    power-law degrees, power-law module sizes and flat mixing this reduces to
    LFR; on real blueprints it keeps what LFR throws away.

    Args:
        G: Blueprint network, a networkx graph (weights honoured) or any
            network of this package. A directed blueprint gets the
            in-/out-degree-preserving directed sampler.
        partition: ``{node: module id or path tuple}``; hierarchical paths
            welcome. When omitted, Infomap detects one (needs the ``infomap``
            package) -- with the same ``num_trials`` and ``seed`` the sampler
            would use -- and that becomes the planted partition.
        **kwargs: Passed to the sampler -- ``target_edges``, ``mixing_scale``
            (> 1 blurs communities for detectability sweeps), ``method``
            (``"stub"`` or ``"multinomial"``), ``connect``, ``min_degree``,
            ``match_clustering``, ``rng``, ``seed``.

    Returns:
        A :class:`~networks.InMemoryNetwork` named ``<blueprint>-benchmark``,
        with the planted ``partition`` and the sampler's diagnostics in
        ``info``. ``seed`` seeds Infomap only; pass ``rng`` to make the
        sample itself reproducible.

    Raises:
        ImportError: numpy or networkx is missing, or Infomap is needed but
            not installed.
    """
    flowgen, _ = _modules()
    G = _blueprint(G)
    if partition is None:
        _require_infomap(flowgen, "benchmark() without a partition")
        directed = kwargs.get("directed")
        partition = flowgen.infomap_partition(
            G,
            two_level=False,
            num_trials=kwargs.get("num_trials", 20),
            seed=kwargs.get("seed", 1),
            directed=G.is_directed() if directed is None else directed,
        )
    H, info = flowgen.sample_benchmark(G, partition=partition, **kwargs)
    H = _labelled(H, G, kwargs.get("method"))
    return _synthetic(
        H,
        name=f"{G.name or 'network'}-benchmark",
        partition=partition,
        generator="benchmark",
        parameters={"blueprint": G.name or None, **_recorded(kwargs)},
        info=info,
    )


def _labelled(H: Any, G: Any, method: str | None) -> Any:
    """Give a multinomial sample the blueprint's node labels back.

    The multinomial sampler draws from the flow matrix, whose rows are
    ``sorted(G.nodes())``, and returns row indices as nodes; the planted
    partition is keyed by the blueprint's labels.
    """
    if method != "multinomial":
        return H
    import networkx

    nodes = sorted(G.nodes())
    return networkx.relabel_nodes(H, dict(enumerate(nodes)), copy=True)


def _recorded(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Sampler keyword arguments worth writing into a file header."""
    return {key: value for key, value in kwargs.items() if key != "rng"}


def weighted_benchmark(
    G: Any,
    partition: dict[Any, Any] | None = None,
    *,
    degree_target: str = "plain",
    num_trials: int = 20,
    seed: int = 1,
    **kwargs: Any,
) -> InMemoryNetwork:
    """Sample a benchmark topology and give its edges flow-model weights.

    Dense weighted blueprints carry their modular structure in the link
    weights, not the topology, so an unweighted sample of one can be
    unrecoverable even when the blueprint compresses perfectly. This sampler
    puts the structure back into the weights: each realized edge ``(u, v)``
    gets a weight proportional to the joint flow ``p_u p_v R(A_u, A_v)``,
    rescaled so the total edge weight matches the blueprint's.

    Works in the undirected frame throughout, like the paper's weighted
    variant: a directed blueprint is symmetrized (weights summed over both
    directions) before sampling.

    Args:
        G: Blueprint network (weights honoured; a DiGraph is symmetrized), a
            networkx graph or any network of this package.
        partition: ``{node: module id or path tuple}``. When omitted, Infomap
            detects a hierarchical partition of the symmetrized blueprint.
        degree_target: Which degree sequence the *topology* targets.
            ``"plain"`` (default) draws it from the blueprint's unweighted
            degree sequence, which is feasible as a simple graph by
            construction, and lets the weights carry all of the flow
            structure -- the mode to use for weighted blueprints, since
            multi-edges aggregate to link weights in the map equation.
            ``"flow"`` targets the flow-apportioned (strength-proportional)
            degrees instead; faithful to the flow model but infeasible when
            weights are very skewed, where it strands stubs on hub
            saturation and no assignment of weights can rescue a topology
            that was never realized.
        num_trials: Infomap trials when detecting the partition.
        seed: Infomap seed when detecting the partition.
        **kwargs: Passed to the topology sampler (see :func:`benchmark`).

    Returns:
        An undirected weighted :class:`~networks.InMemoryNetwork` named
        ``<blueprint>-weighted-benchmark``, with the planted ``partition`` and
        the diagnostics in ``info``; ``info["degree_target"]`` records the
        mode used.

    Raises:
        ImportError: numpy or networkx is missing, or Infomap is needed but
            not installed.
        ValueError: *degree_target* is not ``"plain"`` or ``"flow"``.

    .. warning::
       Called without ``partition`` this plants with the RECURSIVE search
       (``infomap_partition`` passes no ``extra``).  The generative-map-equation project
       runs the columnar engine (``-C``) on both sides of every plant/re-detect comparison,
       so a partition-less call there would silently make its recovery numbers measure
       cross-engine agreement instead.  Every call site in that project passes a partition
       today (checked 2026-09-19), and the default is left alone because this package is
       shared with other projects that may rely on it; pass ``partition=`` explicitly, or
       ``infomap_partition(..., extra="-C")``, when the engine matters.
    """
    if degree_target not in ("plain", "flow"):
        raise ValueError('degree_target must be "plain" or "flow"')
    flowgen, params = _modules()
    import networkx

    G = _blueprint(G)
    U = params._und(G)
    if partition is None:
        _require_infomap(flowgen, "weighted_benchmark() without a partition")
        partition = flowgen.infomap_partition(
            U, two_level=False, num_trials=num_trials, seed=seed, directed=False
        )
    if degree_target == "plain":
        topology_source = networkx.Graph()
        topology_source.add_nodes_from(U.nodes())
        topology_source.add_edges_from(U.edges())  # weights dropped on purpose
    else:
        topology_source = U
    H, info = flowgen.sample_benchmark(topology_source, partition=partition, **kwargs)
    for (u, v), w in params.flow_edge_weights(U, partition, H).items():
        H[u][v]["weight"] = w
    info["frame"] = "undirected"
    info["degree_target"] = degree_target
    return _synthetic(
        H,
        name=f"{G.name or 'network'}-weighted-benchmark",
        partition=partition,
        generator="weighted_benchmark",
        parameters={
            "blueprint": G.name or None,
            "degree_target": degree_target,
            **_recorded(kwargs),
        },
        info=info,
    )


def parametric(
    n: int,
    degree_fit: dict[str, Any],
    size_fit: dict[str, Any],
    mixing_fit: dict[str, Any],
    *,
    min_size: int = 2,
    connect: bool = True,
    k_mean: float | None = None,
    k_max: int | None = None,
    size_max: int | None = None,
    clustering_fit: dict[str, Any] | None = None,
    clustering_kwargs: dict[str, Any] | None = None,
    rng: Any = None,
    seed: int | None = None,
) -> InMemoryNetwork:
    """Generate a two-level benchmark purely from fitted parameters.

    The generalized-LFR generator: draw module sizes and node degrees from
    their fitted families, per-module mixing from the fitted Beta (or a
    delta), then stub-match exactly like the blueprint sampler. Mirroring
    LFR's ``(tau, <k>, k_max, max_community)`` parameterization, drawn
    degrees can be rescaled to *k_mean* and capped at *k_max*, and sizes
    capped at *size_max*, so different family choices stay density-matched.

    Args:
        n: Number of nodes (the realized count can differ by up to
            *min_size* because the last module is trimmed).
        degree_fit: A :func:`fit_families` result (its AIC winner is used) or
            an explicit ``{"family", "params", "support_min"}`` dict.
        size_fit: The same, for module sizes.
        mixing_fit: A :func:`fit_mixing_beta` result, or a flat LFR-style
            mixing as ``{"family": "delta", "mu": 0.15}``.
        min_size: Smallest allowed module.
        connect: Repair connectivity with degree-preserving swaps.
        clustering_fit: A :func:`fit_clustering_profile` result. Given, the
            sample runs the same clustering pass as the blueprint sampler,
            driven by the fitted profile ``c(k)`` instead of a blueprint's
            triangle counts (LFR's absence of clustering is ``c == 0``).
        clustering_kwargs: Extra arguments for that pass, e.g.
            ``{"time_cap": 120}``.
        k_mean: Rescale drawn degrees to this mean.
        k_max: Cap on node degree.
        size_max: Cap on module size.
        rng: A ``numpy.random.Generator``; takes precedence over *seed*.
        seed: Seed for a fresh generator when *rng* is omitted.

    Returns:
        A :class:`~networks.InMemoryNetwork` with the planted
        ``{node: module}`` partition and the diagnostics in ``info``.

    Raises:
        ImportError: numpy or networkx is missing.
        ValueError: A fit carries no usable family.
    """
    _, params = _modules()
    import numpy

    if rng is None:
        rng = numpy.random.default_rng(seed)
    degree_family = _pick_family(degree_fit)
    size_family = _pick_family(size_fit)
    H, partition, info = params.sample_from_params(
        n,
        degree_family,
        size_family,
        mixing_fit,
        rng,
        min_size=min_size,
        connect=connect,
        k_mean=k_mean,
        k_max=k_max,
        size_max=size_max,
        clustering_fit=clustering_fit,
        clustering_kwargs=clustering_kwargs,
    )
    return _synthetic(
        H,
        name=run_name("parametric", n=n, seed=seed),
        partition=partition,
        generator="parametric",
        parameters={
            "n": n,
            "degrees": degree_family["family"],
            "sizes": size_family["family"],
            "mixing": mixing_fit.get("family"),
            "min_size": min_size,
            "k_mean": k_mean,
            "k_max": k_max,
            "size_max": size_max,
            "clustering": "fitted" if clustering_fit is not None else None,
            "seed": seed,
        },
        info=info,
    )


def fit_clustering_profile(graph: Any, min_degree: int = 2) -> dict[str, Any]:
    """Fit the local-clustering profile ``c(k) = c0 k**-alpha`` of *graph*.

    The clustering component of a blueprint's fingerprint, so that
    :func:`parametric` can be given a clustering target without a blueprint
    at sampling time. Fit by least squares on ``log c`` over the nodes of
    degree at least *min_degree* with nonzero clustering; a graph with fewer
    than three such nodes gets a flat profile at its mean clustering.

    Args:
        graph: A networkx graph (directed input is read undirected).
        min_degree: Smallest degree entering the fit.

    Returns:
        ``{"c0", "alpha", "n_fit", "c_mean"}``, ready to pass to
        :func:`parametric` as *clustering_fit*.

    Raises:
        ImportError: networkx is missing.
    """
    _, params = _modules()
    result: dict[str, Any] = params.fit_clustering_profile(graph, min_degree=min_degree)
    return result


def fit_families(values: Any, support_min: int | None = None) -> dict[str, Any]:
    """Fit discrete families to *values* and pick a winner by AIC.

    Candidates: Zipf power law, geometric, truncated Poisson and discretized
    log-normal, all fit by maximum likelihood on the shifted variable
    ``k' = k - (support_min - 1) >= 1`` -- otherwise a family anchored at 1
    is structurally misfit to quantities on a shifted support, such as module
    sizes starting at 5.

    Args:
        values: Positive integers -- a degree sequence, module sizes, ...
        support_min: Smallest value of the support; defaults to
            ``min(values)``.

    Returns:
        ``{family: {"params", "loglik", "aic", "aic_weight"}, "winner",
        "support_min", "n_values"}``. Feed the whole dict to
        :func:`parametric`, which uses the winner.

    Raises:
        ImportError: numpy or scipy is missing.
    """
    _, params = _modules()
    try:
        import scipy  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on install
        raise ImportError(f"fit_families() needs scipy. Install it with `{_INSTALL}`.") from error
    result: dict[str, Any] = params.fit_families(values, support_min=support_min)
    return result


def fit_mixing_beta(mixing: Any, flows: Any) -> dict[str, Any]:
    """Fit a flow-weighted Beta (method of moments) to per-module mixing.

    Args:
        mixing: Per-module mixing values ``mu_A = q_A / p_A``, in [0, 1].
        flows: Per-module flows ``p_A``, used as weights.

    Returns:
        ``{"family": "beta", "alpha", "beta", "mu", "var"}``, or a
        ``{"family": "delta", "mu", "var"}`` fallback when the weighted
        variance is degenerate. Feed it to :func:`parametric`.

    Raises:
        ImportError: numpy is missing.
    """
    _, params = _modules()
    result: dict[str, Any] = params.fit_mixing_beta(mixing, flows)
    return result


def lfr_networkx(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    *,
    average_degree: float | None = None,
    min_degree: int | None = None,
    max_degree: int | None = None,
    min_community: int | None = None,
    max_community: int | None = None,
    seed: int = 1,
    max_iters: int = 800,
    retries: int = 5,
) -> InMemoryNetwork:
    """Generate a classic LFR benchmark with networkx's reimplementation.

    ``networkx.LFR_benchmark_graph`` fails to converge for perfectly
    reasonable parameters depending on the seed. This wrapper retries with
    escalating seeds and iteration budgets, strips self-loops, and returns
    the planted communities as a plain partition dict instead of node
    attribute sets.

    This is the undirected unweighted case only. :func:`lfr` runs the
    authors' own generator instead -- weighted, directed, hierarchical and
    multiplex too, and without the convergence failures -- at the cost of a
    C++ compiler and one download.

    Args:
        n: Number of nodes.
        tau1: Degree power-law exponent (> 1).
        tau2: Community-size power-law exponent (> 1).
        mu: Mixing parameter -- the fraction of a node's links that leave
            its community.
        average_degree: Mean degree (give this or *min_degree*).
        min_degree: Minimum degree, the alternative to *average_degree*.
        max_degree: Maximum degree.
        min_community: Smallest community size.
        max_community: Largest community size.
        seed: Base seed; retry *i* uses ``seed + 1000 * i``.
        max_iters: Base iteration budget; retry *i* uses
            ``max_iters * (i + 1)``.
        retries: Attempts before giving up.

    Returns:
        A :class:`~networks.InMemoryNetwork` with ``partition``
        ``{node: community index}``. ``info["seed"]`` is the seed of the
        attempt that converged.

    Raises:
        ImportError: networkx is missing.
        RuntimeError: No attempt converged.
    """
    _modules()
    import networkx

    last: Exception | None = None
    for attempt in range(retries):
        try:
            G = networkx.LFR_benchmark_graph(
                n,
                tau1,
                tau2,
                mu,
                average_degree=average_degree,
                min_degree=min_degree,
                max_degree=max_degree,
                min_community=min_community,
                max_community=max_community,
                max_iters=max_iters * (attempt + 1),
                seed=seed + 1000 * attempt,
            )
        except networkx.ExceededMaxIterations as error:
            last = error
            continue
        G.remove_edges_from(networkx.selfloop_edges(G))
        communities = sorted({frozenset(G.nodes[u]["community"]) for u in G}, key=min)
        partition = {u: index for index, members in enumerate(communities) for u in members}
        return _synthetic(
            G,
            name=run_name("lfr-nx", n=n, mu=mu, seed=seed),
            partition=partition,
            generator="lfr_networkx",
            parameters={
                "n": n,
                "tau1": tau1,
                "tau2": tau2,
                "mu": mu,
                "average_degree": average_degree,
                "min_degree": min_degree,
                "max_degree": max_degree,
                "min_community": min_community,
                "max_community": max_community,
                "seed": seed,
            },
            info={"seed": seed + 1000 * attempt},
        )
    raise RuntimeError(
        f"LFR did not converge in {retries} attempts (n={n}, tau1={tau1}, tau2={tau2}, "
        f"mu={mu}). Relax the constraints or raise max_iters."
    ) from last


def detect(
    G: Any,
    *,
    two_level: bool = False,
    num_trials: int = 20,
    seed: int = 1,
    directed: bool | None = None,
    extra: str = "",
) -> dict[Any, tuple[int, ...]]:
    """Detect a partition with Infomap, in the shape the samplers expect.

    Args:
        G: A networkx graph (weights honoured).
        two_level: Restrict the search to a two-level solution; the default
            is the full hierarchical search, which the samplers exploit.
        num_trials: Number of Infomap outer-loop trials.
        seed: Infomap seed.
        directed: Directed flow; defaults to ``G.is_directed()``.
        extra: Extra Infomap CLI arguments.

    Returns:
        ``{node: module path tuple}`` -- leaf modules as tuples, ready to
        pass as *partition* to the samplers.

    Raises:
        ImportError: infomap (or numpy/networkx) is missing.
    """
    flowgen, _ = _modules()
    _require_infomap(flowgen, "detect()")
    result: dict[Any, tuple[int, ...]] = flowgen.infomap_partition(
        G, two_level=two_level, num_trials=num_trials, seed=seed, extra=extra, directed=directed
    )
    return result


def memory_benchmark(
    links: dict[tuple[Any, Any], float],
    windows: dict[Any, tuple[Any, ...]],
    partition: dict[Any, Any],
    rng: Any = None,
    seed: int | None = None,
    topup: bool = True,
) -> StateNetwork:
    """Sample a memory-network benchmark from a state-level blueprint.

    Typed stub matching: a state link ``a -> b`` is realizable only where the
    windows overlap in ``m - 1`` nodes, so every sampled link is a valid
    memory transition; see :func:`~networks.generate._higher_order.memory_benchmark`
    for the construction.

    A :class:`~networks.StateNetwork` blueprint passes as
    ``memory_benchmark(net.links, net.states, partition)``.

    Args:
        links: ``{(state, state): weight}`` blueprint state links.
        windows: ``{state: tuple of physical nodes}``, oldest first.
        partition: ``{state: module}`` planted ground truth.
        rng, seed: ``numpy`` Generator, or a seed for a fresh one.
        topup: Place stubs the non-redundant rule could not.

    Returns:
        A :class:`~networks.StateNetwork` with the planted ``partition`` keyed
        by state id and the diagnostics in ``info``. State and physical ids
        are the blueprint's labels when those are non-negative integers, and
        numbered from 1 otherwise, the labels kept as names.
    """
    from .._core.formats import node_ids, ordered
    from .._core.state_network import StateNetwork
    from . import _higher_order

    _modules()
    sampler: Any = _higher_order.memory_benchmark
    sampled, info = sampler(links, windows, partition, rng=rng, seed=seed, topup=topup)
    physical = node_ids(ordered(node for window in windows.values() for node in window))
    order = len(next(iter(windows.values()))) if windows else 1
    if order == 1:
        # States are physical nodes here, and Infomap and the writers use the
        # physical ids, so the state ids -- and the partition -- must match them.
        state = {s: physical[w[0]] for s, w in windows.items()}
    else:
        state = node_ids(ordered(windows))
    return StateNetwork(
        order=order,
        physical_nodes={index: str(label) for label, index in physical.items()},
        states={state[s]: tuple(physical[node] for node in w) for s, w in windows.items()},
        links={(state[a], state[b]): float(w) for (a, b), w in sampled.items()},
        name=run_name("memory-benchmark", order=order, seed=seed),
        partition={state[s]: module for s, module in partition.items() if s in state},
        info={
            "generator": "memory_benchmark",
            "parameters": settings({"order": order, "topup": topup, "seed": seed}),
            **info,
        },
    )


def multilayer_benchmark(
    layers: Any,
    partition: dict[tuple[Any, Any], Any] | None = None,
    rng: Any = None,
    seed: int | None = None,
    **kwargs: Any,
) -> MultilayerNetwork:
    """Sample a multiplex benchmark layer by layer from a state-level blueprint.

    Intra-layer links pair within their layer, so each layer is resampled on
    its own with the state partition restricted to it; inter-layer coupling is
    left to the relax rate at analysis time. See
    :func:`~networks.generate._higher_order.multilayer_benchmark` for what is
    and is not preserved -- notably, a weighted layer keeps its link count but
    not its total weight.

    Args:
        layers: ``{layer: networkx graph}``, ``{layer: {(u, v): weight}}``,
            or a :class:`~networks.MultilayerNetwork` blueprint.
        partition: ``{(node, layer): module}`` planted ground truth. Defaults
            to the blueprint's own partition when *layers* is a
            ``MultilayerNetwork`` that has one.
        rng, seed: ``numpy`` Generator, or a seed for a fresh one.
        **kwargs: Passed to the per-layer sampler; ``connect`` defaults to
            ``False`` here.

    Returns:
        A :class:`~networks.MultilayerNetwork` with intra-layer links only, the
        planted ``partition`` and the diagnostics in ``info``.

    Raises:
        ValueError: No partition given and none on the blueprint.
    """
    from .._core.multilayer import MultilayerNetwork
    from . import _higher_order

    _modules()
    name = "network"
    if isinstance(layers, MultilayerNetwork):
        name = layers.name or name
        if partition is None:
            partition = layers.partition
        layers = {layer: layers.to_networkx(layer=layer) for layer in layers.intra}
    if partition is None:
        raise ValueError("multilayer_benchmark() needs a partition {(node, layer): module}.")
    directed = any(getattr(spec, "is_directed", lambda: False)() for spec in layers.values())
    sampler: Any = _higher_order.multilayer_benchmark
    sampled, info = sampler(layers, partition, rng=rng, seed=seed, **kwargs)

    def layer_nodes(spec: Any) -> list[Any]:
        if isinstance(spec, dict):
            return list(dict.fromkeys(node for pair in spec for node in pair))
        return list(spec.nodes())

    intra: dict[Any, dict[tuple[Any, Any], float]] = {}
    planted: dict[tuple[Any, Any], Any] = {}
    for layer, links in sampled.items():
        nodes = layer_nodes(layers[layer])
        if kwargs.get("method") == "multinomial":
            # Row indices into the layer's sorted nodes, as in benchmark().
            label = dict(enumerate(sorted(nodes)))
            links = {(label[u], label[v]): w for (u, v), w in links.items()}
        intra[layer] = dict(links)
        # The sampler puts a node the partition leaves out in module (0,).
        planted.update({(u, layer): partition.get((u, layer), (0,)) for u in nodes})
    return MultilayerNetwork(
        name=f"{name}-multilayer-benchmark",
        intra=intra,
        directed=directed,
        weighted=any(w != 1.0 for links in intra.values() for w in links.values()),
        partition=planted,
        info={
            "generator": "multilayer_benchmark",
            "parameters": settings({"blueprint": name, "seed": seed, **_recorded(kwargs)}),
            **info,
        },
    )
