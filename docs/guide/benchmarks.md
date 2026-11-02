# Benchmarks

## Benchmarking Infomap

Six SNAP datasets ship reference partitions, which makes them usable for
scoring community detection rather than just running it:

```python
suite = snap.search(ground_truth=True, max_edges=5_000_000)
# com-Youtube, com-DBLP, com-Amazon

for dataset in suite:
    net = snap.load(dataset.name)
    result = net.to_infomap(silent=True).run()

    truth = snap.load_ground_truth(dataset.name, variant="top5000")
    reference = [set(community) for community in truth.communities()]

    print(dataset.name, result.num_top_modules, "vs", len(reference), "ground-truth")
```

`variant="top5000"` gives SNAP's 5000 highest-quality communities — the usual
reference set; `"all"` gives the full list. Communities overlap, so a node can
appear in several.

Because every dataset is unique in the catalog and sizes are known up front,
a graded suite is just a filter — no risk of benchmarking the same network
several times because SNAP happens to list it under several headings.

### A domain-balanced set for publication

CI regression wants structural diversity; a paper wants *domain* diversity —
a different axis. SNAP alone cannot supply it (no biological networks, no
multilayer, transport limited to three road networks), so draw across sources:

```python
POOL = {
    "Social": [("nz", "karate"), ("nz", "football"), ("snap", "email-Eu-core")],
    "Informational": [("nz", "polbooks"), ("nz", "polblogs"), ("nz", "dblp_cite")],
    "Biological": [
        ("nz", "celegansneural"),
        ("nz", "celegans_metabolic"),
        ("nz", "foodweb_little_rock"),
        ("nz", "malaria_genes"),
    ],
    "Transportation": [
        ("nz", "london_transport"),
        ("nz", "openflights"),
        ("nz", "us_air_traffic"),
        ("snap", "roadNet-PA"),
    ],
    "Technological": [("nz", "power"), ("nz", "internet_as"), ("nz", "linux")],
    "Economic": [("nz", "fao_trade"), ("nz", "at_migrations")],
}
```

Reference partitions exist for `karate` (2 factions), `football` (12
conferences), `polbooks`/`polblogs` (political alignment), `malaria_genes`
(CysPoLV/UPS), `foodweb_little_rock` (trophic groups) and `email-Eu-core`
(42 departments).

Treat those with care: node metadata is **not** guaranteed to be the community
structure, a point reviewers will raise. Use synthetic benchmarks with planted
communities for the quantitative claim, and this pool for generality — reported
via codelength against baselines. Neither source provides synthetic generators —
that is what {mod}`netgamut.generate` (see {doc}`benchmarks`) is for.

## Generating benchmarks

The `generate` subpackage turns any network in the pool into a synthetic
benchmark with planted communities: it reads the blueprint's modular flow —
degrees, module sizes, per-module mixing — and samples fresh networks with the
same statistics, so ground truth is known while realism is kept:

```python
import netgamut
from netgamut import generate

G = netgamut.load("polbooks").to_networkx()
partition = generate.detect(G)                # Infomap, or bring your own
net = generate.benchmark(G, partition)        # degree- and mixing-preserving

net = generate.weighted_benchmark(G, partition)  # + flow-model weights
```

Every generator returns a network object with the same methods as a download —
a `InMemoryNetwork`, or a `StateNetwork` / `MultilayerNetwork` for the
higher-order ones — carrying the planted partition and the sampler's
diagnostics:

```python
net.partition                  # {node: module}, the ground truth
net.info                       # diagnostics, plus "generator" and "parameters"
H = net.to_networkx()          # planted module as node attribute "module"
net.to_pajek("benchmarks/")    # PosixPath('benchmarks/polbooks-benchmark.net')
```

Or drop the blueprint entirely and generate from fitted parameters — a
generalized LFR whose degree and community-size families are chosen by AIC
instead of assumed to be power laws:

```python
deg = generate.fit_families([k for _, k in G.degree()])
size = generate.fit_families([120, 80, 60, 45, 30])  # module sizes
mix = {"family": "delta", "mu": 0.15}                # flat LFR-style mixing
net = generate.parametric(1000, deg, size, mix, seed=1)
```

Directed blueprints keep both degree sequences and, like undirected ones,
their full module hierarchy; weighted blueprints need weighted samples, since
a dense weighted network carries its structure in the weights rather than the
topology:

```python
net = generate.weighted_benchmark(G, partition)  # topology + flow weights
```

Higher-order blueprints change only the sampler, never the flow model — a
memory link has to match at its physical junction, a multiplex link has to
stay inside its layer:

```python
memory = generate.memory_benchmark(links, windows, partition, seed=1)  # StateNetwork
multiplex = generate.multilayer_benchmark(layers, partition, seed=1)  # MultilayerNetwork
```

Without a blueprint, `overlapping_memory_benchmark` plants communities that
overlap on the physical nodes and are disjoint on the states — the regime a
first-order partition cannot represent. It is Andrea Lancichinetti's trigram
sampler, seeded, and needs only the standard library:

```python
memory = generate.overlapping_memory_benchmark(4, 50_000, seed=1)  # StateNetwork, om=4, E=50000
memory.to_pajek("out/")  # out/overlapping-memory-n256-om4-nc64-E50000-mu0.1-seed1.net
memory.partition         # {state id: planted community}, for `infomap -d --no-infomap -c`
```

Every node sits in `memberships` communities of `community_size` nodes
(defaults: 256 nodes, communities of 64, `mu=0.1`); `mu` of the trigrams cross
communities by stitching two existing pairs, so mixing adds links but no
states. A seeded network has the same states, links and planted partition as
the original script run with that seed.

The flow model itself is public as well (`generate.flow_matrix`,
`generate.ModuleTree`, `generate.cross_R`) for callers who want the joint
distribution rather than a sample of it. Everything here needs the extras:
`pip install "netgamut[generate]"`.

### The original LFR benchmarks

`generate.lfr(...)` runs the Lancichinetti–Fortunato generators themselves —
the authors' C++, not a reimplementation — so the classic baseline is one call
away, in all six of its variants:

```python
net = generate.lfr(1000, 15, 50, mu=0.2, minc=20, maxc=50, seed=1)
net = generate.lfr(1000, 15, 50, mut=0.2, muw=0.1, beta=1.5)   # weighted
net = generate.lfr(1000, 15, 50, mu=0.2, directed=True)        # directed
net = generate.lfr(5000, 20, 100, mu1=0.1, mu2=0.3,            # two planted
                   minc=10, maxc=100,                          #   levels
                   min_macro=200, max_macro=1000)
net = generate.lfr(1000, 20, 50, mu=0.3, layers=10, graphs=3)  # multiplex
net = generate.lfr(1000, 15, 50, mu=0.2, on=20, om=2)          # overlapping

net.to_pajek("benchmarks/")
# PosixPath('benchmarks/lfr-undirected-n1000-k15-maxk50-mu0.2-seed1.net')
```

Which program runs follows from the arguments, and so does the shape of
`net.partition`: `{node: module}` normally, `{node: (macro, micro)}` for the
hierarchical variant — the path convention the flow model reads —
`{node: (module, ...)}` for an overlapping cover, and `{(node, layer): module}`
for the multiplex one, which comes back as a `MultilayerNetwork`
(`net.to_networkx(layer=3)` is one layer) ready for
`generate.multilayer_benchmark`. Node labels are zero-based, unlike the
one-based `.dat` files upstream writes. The default name, and so the file name,
spells out the variant, the main parameters and the seed; pass `name=` to
`to_pajek` when a sweep varies something else.

**`seed` alone decides the network.** Left to themselves the generators seed
from the clock; the wrapper writes `time_seed.dat` instead, so the same seed
gives the same network byte for byte, on any machine, with the pinned upstream
revision.

The C++ is fetched rather than vendored, and compiled where it runs. That
keeps one wheel valid on every platform and needs no compiler at install time;
the price is a download and a build on the first call, both cached next to the
datasets:

```text
~/.cache/netgamut/lfr/src/<commit>/     the sources, fetched once
~/.cache/netgamut/lfr/bin/<key>/        one binary per variant and toolchain
```

The first call takes about four seconds — Xcode command line tools on macOS,
`build-essential` on Linux — and every call after it is milliseconds.
`generate.lfr_available()` reports whether the sources and a compiler are both
in place, without downloading anything.

If you would rather not have a compiler in the loop, `generate.lfr_networkx(...)`
keeps networkx's reimplementation of the undirected unweighted case, with
retries around its flaky convergence.

### Multilayer benchmarks with MultilayerGM

`generate.multilayer_gm(...)` runs
[MultilayerGM](https://github.com/MultilayerGM/MultilayerGM-py) (Bazzi, Jeub,
Arenas, Howison & Porter): a partition of the state nodes is sampled with a
tunable dependence between layers, then every layer gets a degree-corrected
SBM. Upstream's four steps — dependency tensor, null distribution, partition,
network — are one call:

```python
net = generate.multilayer_gm(1000, 5, p=0.9, mu=0.2, communities=10,
                             theta=1, k_min=3, k_max=150, t_k=-2, seed=1)
net.to_pajek("benchmarks/")   # *Intra only: couple layers with Infomap's relax rate
# PosixPath('benchmarks/mlgm-multiplex-n1000-L5-p0.9-mu0.2-seed1.net')

net = generate.multilayer_gm(1000, 20, p=0.95, coupling="temporal")  # layer i copies i-1
```

`p` is how likely a node keeps its community from another layer — any other
layer for `coupling="multiplex"`, the previous one for `"temporal"`, where it
can also be a list of `layers - 1` probabilities. `mu` is the fraction of edges
placed ignoring communities, which is *not* LFR's mixing parameter: a random
edge can still land inside a community.

There is no generator object to keep around: building the dependency tensor
and null takes milliseconds, and the time goes into sampling edges. What is
worth reusing is the partition — `partition=net.partition` resamples only the
edges, for a `mu` sweep over one planted structure.

The sampling code is vendored, at a pinned upstream commit, because upstream is
not on PyPI and seeds only through global random state; the vendored copy takes
local random streams instead. The same `seed` reproduces upstream's network
exactly, and never touches the caller's `random` or `numpy.random`.

#### Which revision, and moving it

Upstream publishes no releases — it is a repository of six source trees, last
touched in June 2024 — so the pin is a commit hash plus the SHA-256 of its
archive. `generate.lfr_source()` shows what a run would use:

```python
print(generate.lfr_source())
# andrealancichinetti/LFRbenchmarks@ec9a860282d8 (pinned, fetched)
#   url       https://github.com/.../archive/ec9a8602....tar.gz
#   sha256    ad13a354c2cfbb1de299dcb13caadd5801243b9041cd0e7e89852b9aeb384f76
#   sources   ~/.cache/netgamut/lfr/src/ec9a8602...
#   compiler  Apple clang version 17.0.0
#   variants  undirected, weighted, directed, weighted_directed, hierarchical, multiplex
```

To move it, in a checkout of this repository:

```bash
python scripts/update_lfr_pin.py --check   # has upstream moved? exit 1 if so
python scripts/update_lfr_pin.py           # download, verify, rewrite the pin
python scripts/update_lfr_pin.py --commit <sha>   # pin something specific
```

The script asks GitHub what the default branch points at, downloads that
archive, refuses to pin one that is missing any of the six generators, and
rewrites `COMMIT` and `SHA256` in `src/netgamut/generate/_lfr/_sources.py`.
Bumping the pin is a real change rather than housekeeping: these generators are
random, so a different revision can give a different network for the same seed.
Run `pytest tests/test_lfr.py` after, and say so in the commit message.

To try a revision without moving the pin, point `NETGAMUT_LFR_DIR` at a
checkout — the wrapper builds from there and skips the download entirely.

### Comparing partitions as flow models

The same flow model compares two partitions of one network: each becomes a
joint distribution over node pairs, and the **generative map distance** is the
Jensen–Shannon distance between them — a metric in `[0, 1]` that scores a
disagreement by how much flow it misroutes, not by how many labels it moves:

```python
D = generate.gmd(G, partA, partB)         # distance, 0 when flow-equivalent
S = generate.gms(G, partA, partB)         # similarity, 1 - D
D, per_node = generate.gmd(G, partA, partB, element=True)   # D^2 = sum_i p_i d_i^2
```

`generate.agmd` / `generate.agms` are the adjusted pair, a Steinhaus transform
against the one-module null partition (`generate.null_partition`) that puts the
trivial partition at distance 1 from anything carrying flow structure — the
zero-at-trivial convention of chance-corrected label measures, without a random
model. Its base point is network-specific, so adjusted values compare partitions
of one network, not partitions across networks.

Evaluation is blockwise over the meet cells of the two partitions,
`O(cells^2 + n + m^2)`, so it never materialises the `n x n` matrices. Either
partition may be hierarchical (`{node: path tuple}`), directed or weighted, and
nodes carrying no flow carry no weight: two partitions that differ only in where
they put isolated nodes are flow-equivalent, at `D = 0` exactly.
