# networks

Browse, download and cache network datasets from Python.

Five sources are indexed, each in its own subpackage:

| Source | Datasets | Strength |
|---|---|---|
| [`networks.snap`](https://snap.stanford.edu/data/index.html) | 109 | large, well-known social / web / technological networks |
| [`networks.netzschleuder`](https://networks.skewed.de) | 286 | domain-tagged, incl. biological, transport and multilayer |
| [`networks.ogb`](https://ogb.stanford.edu) | 12 | graph-learning benchmarks, with **named** node categories |
| [`networks.pyg`](https://pytorch-geometric.readthedocs.io) | 99 | the graph-learning corpus: citation, web, social, heterophilous |
| `networks.paths` | 2 | curated path data (flight itineraries, navigation trails) for memory networks |

Metadata for all five **ships inside the package**, so you can explore every
dataset — domains, sizes, statistics, citations — without downloading anything.
When you do ask for data it lands in a shared user-level cache, so a dataset is
fetched once per machine no matter how many projects, notebooks or virtualenvs
use it.

```bash
pip install mapequation-networks
```

The distribution is `mapequation-networks`; the import name is `networks`.
For a runnable tour — browsing, loading, and building memory networks from
path data — see [`examples/tutorial.ipynb`](examples/tutorial.ipynb).

## One domain vocabulary across both sources

The two sites label things differently — SNAP groups by provenance
("Autonomous systems graphs"), Netzschleuder tags by domain — so both are mapped
onto one vocabulary. That makes it possible to compare methods across *kinds of
system* rather than kinds of provenance:

```python
import networks

networks.domains()
```

```
{'Social': 212, 'Informational': 71, 'Biological': 36,
 'Technological': 30, 'Transportation': 17, 'Economic': 39}
```

```python
for match in networks.search(domain="Biological", max_nodes=500):
    print(match)
# netzschleuder/celegansneural [Biological] (297 nodes, 2,359 edges)
# netzschleuder/foodweb_little_rock [Biological] (183 nodes, 2,494 edges)
# ...

net = networks.load("celegans_metabolic")
```

Note `Biological: 36` — all of it from Netzschleuder. SNAP has none, which is
why the second source exists. The mapping is deliberately visible in
`networks/_core/domains.py` rather than buried per source, so it can be argued
with in one place.

For anything source-specific, use that source's own `search()`:

```python
from networks import netzschleuder as nz
from networks import snap

nz.search(multilayer=True)  # 17 multiplex networks; SNAP has none
nz.search(subtype="Food web")
snap.search(ground_truth=True)  # reference partitions
```

## Look before you download (SNAP)

```python
from networks import snap

for category in snap.categories():
    print(category)
```

```
Social networks (24 datasets)
Networks with ground-truth communities (9 datasets)
Communication networks (4 datasets)
Citation networks (3 datasets)
Collaboration networks (5 datasets)
...
```

The categories are exactly SNAP's own — same names, same grouping, same order
as the index page: 23 categories over 109 datasets.

SNAP cross-lists some datasets, so a dataset carries *all* the categories that
list it and appears once overall:

```python
snap.info("soc-RedditHyperlinks").category_names
# ('Social networks', 'Signed networks', 'Temporal networks', 'Online communities')

len(snap.datasets())  # 109 unique datasets, no repeats
len(snap.datasets("Signed networks"))
```

```python
print(snap.describe("ca-GrQc"))
```

```
ca-GrQc
=======
Category    Collaboration networks
Summary     Collaboration network of Arxiv General Relativity
Type        Undirected
Page        https://snap.stanford.edu/data/ca-GrQc.html

Reported by SNAP
  Type                         Undirected
  Nodes                        5,242
  Edges                        14,496

Dataset statistics
  Nodes                        5242
  Edges                        14496
  Nodes in largest WCC         4158 (0.793)
  Average clustering coefficient  0.5296
  Diameter (longest shortest path)  17
  ...

Files
  * ca-GrQc.txt.gz
  (* downloaded by default)

Source
  J. Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: ...
```

None of that touched the network.

## Find datasets that fit

```python
# Undirected networks small enough to hold in memory
small = snap.search(directed=False, max_edges=1_000_000)

# Everything in one category
snap.datasets("Collaboration networks")

# Free-text over names and descriptions
snap.search("bitcoin")
```

Each result is a `Dataset` with `.nodes`, `.edges`, `.directed`, `.description`,
`.statistics`, `.files` and `.citations`.

Categories report different columns — temporal networks have *Temporal Edges*
and *Static Edges*, review datasets have *Number of items* — so `.columns`
preserves each table's own headers verbatim rather than flattening them into a
schema SNAP never used.

## Download once, reuse everywhere

```python
net = snap.load("ca-GrQc")  # downloads on first use
net = snap.load("ca-GrQc")  # instant, from cache
```

`load` returns a `Network` handle, not a parsed graph — so the same call works
for a 14k-edge collaboration network and a billion-edge social graph.

```python
net.path  # decompressed file on disk, ready for any tool
net.dataset  # the metadata from above
net.header  # SNAP's own descriptive comment lines
net.size_bytes

for source, target in net.edges(cast=int):  # streamed, flat memory
    ...

for fields in net.rows():  # full records: timestamps, signs, weights
    ...
```

Conversions are lazy and optional:

```python
graph = net.to_networkx()  # pip install "mapequation-networks[networkx]"
im = net.to_infomap()  # pip install "mapequation-networks[infomap]"
im.run()
```

Both default to the direction SNAP reports for the dataset.

Some entries are multi-file bundles:

```python
net = snap.load("ego-Facebook", file="facebook.tar.gz")
directory = net.extract()
```

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
that is what [`networks.generate`](#generating-benchmarks) is for.

## Netzschleuder

```python
from networks import netzschleuder as nz

nz.domains()
# {'Social': 148, 'Informational': 53, 'Biological': 36,
#  'Technological': 14, 'Transportation': 13, 'Economic': 25}

print(nz.describe("celegans_metabolic"))
```

Every network carries the site's precomputed measures, so sizes, clustering,
assortativity and diameter are all available before downloading. Filters go
beyond size:

```python
nz.search(domain="Transportation", subtype="Airport")
nz.search(multilayer=True)  # 17 multiplex networks
nz.search(weighted=True, domain="Economic")
nz.search(bipartite=True)
```

Downloads use the site's CSV bundle, so **graph-tool is not required** despite
being the site's native format:

```python
net = nz.load("celegansneural")
net.path  # extracted edges.csv
list(net.edges(cast=int))
net.extract()  # nodes.csv and gprops.csv too
```

Metadata is read from a JSON API rather than scraped, so this subpackage is
markedly less fragile than the SNAP one.

## Node metadata, in one shape

Every source stores node attributes somewhere different — a separate SNAP artifact,
a member of a Netzschleuder bundle, OGB's `raw/` and `mapping/` folders, a
materialised PyG table. `Network.nodes()` reads all of them the same way:

```python
nodes = networks.load("ogbn-arxiv").nodes()

nodes.columns                       # ('arxiv_category', 'year', 'paper')
nodes["arxiv_category"]             # {0: 'arxiv cs cr', ...}  decoded where named
nodes.partition("arxiv_category")   # {0: 6, ...}  contiguous 0..k-1, for AMI/NMI/D
nodes.categories("arxiv_category")  # ('arxiv cs ai', ...)  categories[i] names module i
```

`partition` and `categories` are two halves of one answer: the first is what you score
a detected partition against, the second is what lets you say *which* category a module
lines up with. They always agree —
`categories(c)[partition(c)[node]]` is that node's value.

The same four calls, whatever the source:

```python
networks.load("email-Eu-core").nodes()["department"]        # SNAP
networks.load("karate", source="netzschleuder").nodes()["groups"]
networks.load("ogbn-arxiv").nodes()["arxiv_category"]       # OGB
networks.load("Planetoid/Cora", source="pyg").nodes()["label"]
```

Column names come from the source, so a figure legend can use them as they are. Values
are decoded to names **where the source ships names** — OGB does, PyG does not:
`Data.y` is an integer tensor and PyG's API has no `class_names` anywhere, so
`categories("label")` returns the codes and their meaning lives in the originating
paper. Nodes with no value are absent from `partition` rather than pooled into a
module of their own, which would invent a community the source never claimed.

Overlapping ground-truth communities are a different thing and stay where they were —
`snap.load_ground_truth(...)` and `Network.communities()` — because a cover is not a
column.

> **Node metadata is not ground truth for community structure.** A partition that
> disagrees with an attribute is not thereby wrong; scoring against `partition(col)`
> measures agreement with an attribute, which is a finding about the attribute as much
> as about the algorithm. See Peel, Larremore & Clauset, *Science Advances* **3**,
> e1602548 (2017).

## Graph-learning benchmarks

`networks.ogb` and `networks.pyg` index the datasets the graph-neural-network
literature is benchmarked on — which is where most empirical networks *with node
labels* now live.

**OGB** needs no extra dependency at all. Its archives are plain zips of `csv.gz`, and
it is the only source that ships names for its codes:

```python
ogb.describe("ogbn-arxiv")
ogb.search(supported=True, domain="Biological")
```

Not everything it publishes is loadable, and the catalog says so rather than omitting
it: the 15 `ogbg-*` molecule collections are not indexed at all (a `Network` is one
graph), while heterogeneous datasets and the two stored as `data.npz` are listed and
refused with their reason. Node and edge counts are read out of each archive's own
`num-node-list.csv.gz`, so the ones too large to download stay `None` rather than
being guessed.

**PyG** catalogues 99 single-graph datasets across 38 classes; the collections of
thousands of molecules or meshes are omitted, with `pyg.omitted()` giving the reason
for every class left out. A further 9 are listed but marked unloadable — PyG's
constructor wants an argument this loader does not supply — so they refuse with the
reason instead of failing deep inside PyG. Browsing needs nothing installed:

```python
pyg.search(domain="Informational", max_nodes=50_000)
pyg.describe("WikipediaNetwork/chameleon")
```

Loading materialises once: the first call needs `pip install
"mapequation-networks[pyg]"`, runs PyG's own downloader, and writes `edges.csv.gz` and
`nodes.csv.gz` into the cache. **Every call after that reads those files and never
imports torch** — the heavy dependency is a one-time cost, and the result is a plain,
archivable network like any other. The dense feature matrix is deliberately not
written: Cora's raw PyG directory is ~16 MB, almost all of it the 1,433-dimensional
bag-of-words matrix, against ~44 KB materialised. Use `pyg.from_pyg(data)` when you
want a `Data` object you already have — including one this package does not catalogue.

PyG is **not** an undirected-only collection, which is easy to assume because its
docstrings never mention direction. Of the 58 measured so far, 24 are directed —
WebKB, Actor, `WikipediaNetwork/chameleon`, the Airports networks, PolBlogs,
EmailEUCore and the JODIE interaction graphs — against undirected Planetoid, Amazon
and the heterophilous set.

Because PyG publishes nothing about it, direction is *measured* rather than assumed:
`--with-direction` materialises each dataset and reads the answer off its
`edge_index`, filling in unpublished sizes at the same time. Until a dataset has been
measured its direction is `None` and `search(directed=...)` skips it. That the
measurement is worth making is clear from Cora, which is **undirected through
`Planetoid` and directed through `AttributedGraphDataset`** — the preprocessing
decides, not the network, so no lookup table keyed on a name could get it right.

Both catalogs are regenerated from upstream's own registry:

```bash
python scripts/regenerate_ogb_catalog.py [--with-stats]
python scripts/regenerate_pyg_catalog.py [--show-omitted]
python scripts/regenerate_pyg_catalog.py --with-direction --max-edges 3000000
```

Neither needs torch for the catalog itself: OGB publishes `master.csv`, and PyG's
statistics live in its class docstrings, which is what its documentation cheatsheet is
generated from. Only `--with-direction` does, because measuring means loading. Both
`--with-stats` and `--with-direction` keep whatever a previous run established for
datasets they cannot reach, so a partial pass never resets known values to unknown.

## Generating benchmarks

The `generate` subpackage turns any network in the pool into a synthetic
benchmark with planted communities: it reads the blueprint's modular flow —
degrees, module sizes, per-module mixing — and samples fresh networks with the
same statistics, so ground truth is known while realism is kept:

```python
import networks
from networks import generate

G = networks.load("polbooks").to_networkx()
partition = generate.detect(G)              # Infomap, or bring your own
H, info = generate.benchmark(G, partition)  # degree- and mixing-preserving

H, info = generate.weighted_benchmark(G, partition)  # + flow-model weights
```

Or drop the blueprint entirely and generate from fitted parameters — a
generalized LFR whose degree and community-size families are chosen by AIC
instead of assumed to be power laws:

```python
deg = generate.fit_families([k for _, k in G.degree()])
size = generate.fit_families([120, 80, 60, 45, 30])  # module sizes
mix = {"family": "delta", "mu": 0.15}                # flat LFR-style mixing
H, partition, info = generate.parametric(1000, deg, size, mix, seed=1)
```

Directed blueprints keep both degree sequences and, like undirected ones,
their full module hierarchy; weighted blueprints need weighted samples, since
a dense weighted network carries its structure in the weights rather than the
topology:

```python
H, info = generate.weighted_benchmark(G, partition)  # topology + flow weights
```

Higher-order blueprints change only the sampler, never the flow model — a
memory link has to match at its physical junction, a multiplex link has to
stay inside its layer:

```python
sampled, info = generate.memory_benchmark(links, windows, partition, seed=1)
sampled, info = generate.multilayer_benchmark(layers, partition, seed=1)
```

The flow model itself is public as well (`generate.flow_matrix`,
`generate.ModuleTree`, `generate.cross_R`) for callers who want the joint
distribution rather than a sample of it. Everything here needs the extras:
`pip install "mapequation-networks[generate]"`.

### The original LFR benchmarks

`generate.lfr(...)` runs the Lancichinetti–Fortunato generators themselves —
the authors' C++, not a reimplementation — so the classic baseline is one call
away, in all six of its variants:

```python
G, part = generate.lfr(1000, 15, 50, mu=0.2, minc=20, maxc=50, seed=1)
G, part = generate.lfr(1000, 15, 50, mut=0.2, muw=0.1, beta=1.5)   # weighted
G, part = generate.lfr(1000, 15, 50, mu=0.2, directed=True)        # directed
G, part = generate.lfr(5000, 20, 100, mu1=0.1, mu2=0.3,            # two planted
                       minc=10, maxc=100,                          #   levels
                       min_macro=200, max_macro=1000)
layers, part = generate.lfr(1000, 20, 50, mu=0.3, layers=10, graphs=3)
G, cover = generate.lfr(1000, 15, 50, mu=0.2, on=20, om=2)         # overlapping
```

Which program runs follows from the arguments, and so does the shape that comes
back: `{node: module}` normally, `{node: (macro, micro)}` for the hierarchical
variant — the path convention the flow model reads — `{node: (module, ...)}` for
an overlapping cover, and `({layer: graph}, {(node, layer): module})` for the
multiplex one, ready for `generate.multilayer_benchmark`. Node labels are
zero-based, unlike the one-based `.dat` files upstream writes.

**`seed` alone decides the network.** Left to themselves the generators seed
from the clock; the wrapper writes `time_seed.dat` instead, so the same seed
gives the same network byte for byte, on any machine, with the pinned upstream
revision.

The C++ is fetched rather than vendored, and compiled where it runs. That
keeps one wheel valid on every platform and needs no compiler at install time;
the price is a download and a build on the first call, both cached next to the
datasets:

```
~/.cache/networks/lfr/src/<commit>/     the sources, fetched once
~/.cache/networks/lfr/bin/<key>/        one binary per variant and toolchain
```

The first call takes about four seconds — Xcode command line tools on macOS,
`build-essential` on Linux — and every call after it is milliseconds.
`generate.lfr_available()` reports whether the sources and a compiler are both
in place, without downloading anything.

If you would rather not have a compiler in the loop, `generate.lfr_networkx(...)`
keeps networkx's reimplementation of the undirected unweighted case, with
retries around its flaky convergence.

#### Which revision, and moving it

Upstream publishes no releases — it is a repository of six source trees, last
touched in June 2024 — so the pin is a commit hash plus the SHA-256 of its
archive. `generate.lfr_source()` shows what a run would use:

```python
print(generate.lfr_source())
# andrealancichinetti/LFRbenchmarks@ec9a860282d8 (pinned, fetched)
#   url       https://github.com/.../archive/ec9a8602....tar.gz
#   sha256    ad13a354c2cfbb1de299dcb13caadd5801243b9041cd0e7e89852b9aeb384f76
#   sources   ~/.cache/networks/lfr/src/ec9a8602...
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
rewrites `COMMIT` and `SHA256` in `src/networks/generate/_lfr/_sources.py`.
Bumping the pin is a real change rather than housekeeping: these generators are
random, so a different revision can give a different network for the same seed.
Run `pytest tests/test_lfr.py` after, and say so in the commit message.

To try a revision without moving the pin, point `NETWORKS_LFR_DIR` at a
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

## Path data and memory networks

Edge lists forget where flow came from; path data does not have to. The
`paths` source curates datasets whose unit of observation is a whole
trajectory, and converts them into **second-order state networks**: a sliding
window turns every observed trigram *i → j → k* into a link from state node
*(i, j)* to state node *(j, k)*, where *(i, j)* means "at physical node *j*,
having arrived from *i*". Infomap reads this representation natively and
finds the overlapping, memory-aware modules described in Rosvall et al.,
*Memory in network flows and its effects on spreading dynamics and community
detection*, Nat. Commun. 5, 4630 (2014).

```python
from networks import paths

data = paths.load("wikispeedia")      # 9.5 MB, 51,318 finished navigation trails
memory = data.to_state_network(order=2)
print(memory)
# StateNetwork(order=2): 4,169 physical nodes, 58,944 state nodes,
# 154,895 links from 50,509 paths

memory.write("wikispeedia_states.net")  # *Vertices / *States / *Links file
im = memory.to_infomap(silent=True)     # or straight into Infomap, directed flow
result = im.run()
```

The `order` parameter generalises the window: `order=1` recovers the plain
directed network from the same trajectories, so first-order versus
second-order comparisons — the codelength case for memory — are a
one-argument change. `StateNetwork.from_paths(...)` accepts your own
sequences (optionally `(nodes, weight)` pairs) with the same machinery.

For demonstrations and small benchmarks, subset the trajectories first:
`data.to_state_network(order=2, max_paths=10_000, max_nodes=300)` keeps the
first 10,000 paths and the 300 most-visited nodes. Trajectories are cut at
removed nodes, never bridged across them, so no transition is fabricated.

Flight itineraries work the same way, at scale:

```python
data = paths.load("db1b-coupon", year=2019, quarter=1)  # ~202 MB zip from BTS
for airports, passengers in data.paths():               # ('OAK', 'SLC', 'LAX', ...)
    ...
memory = data.to_state_network(order=2)
```

DB1B is the US airline itinerary survey (a 10% ticket sample, public domain);
each itinerary is a sequence of airports weighted by its sampled passengers.
The raw quarterly CSV scatters an itinerary's coupons across ~2 GB, so the
first `paths()` call groups them once through SQLite and caches a derived
path file next to the download; later calls stream it directly. From the
shell:

```bash
networks get db1b-coupon --year 2023 --quarter 3
```

Two parsing choices to be aware of, both deliberate: Wikispeedia back clicks
(`<`, in ~17% of trails) are resolved against the browsing history, so paths
record the articles actually visited; and DB1B itineraries keep their full
airport sequence including the return legs, since that is the memory the
model is for.

Note that `wikispeedia` now exists in two sources — the navigation *paths*
here, the article *hyperlink graph* in SNAP — so the top-level
`networks.load("wikispeedia")` asks you to pass `source="paths"` or
`source="snap"` rather than guessing.


## Edge cases worth knowing

### Datasets SNAP has withdrawn

Five entries are still listed on the index page but their downloads have been
disabled (`musae-facebook`, `twitter7`, `web-BeerAdvocate`, `web-RateBeer`,
`web-CellarTracker`). Their metadata is kept so the catalog mirrors the site,
and `load` fails with an explicit message rather than a confusing 404:

```python
snap.load("web-BeerAdvocate")
# LookupError: 'web-BeerAdvocate' has no downloadable files. SNAP has
# withdrawn a handful of datasets while keeping them listed; see ...

[d.name for d in snap.datasets() if not d.files]  # check up front
```

### Guarding against the very large ones

SNAP includes datasets in the tens of gigabytes. `max_bytes` refuses anything
bigger before the transfer starts:

```python
net = snap.load("com-Friendster", max_bytes=2_000_000_000)
# DownloadError: ... is 30.1 GB, which exceeds the 1.9 GB limit.
```

## The cache

```python
snap.cache_dir()  # ~/Library/Caches/networks on macOS
# ~/.cache/networks on Linux
snap.is_cached("ca-GrQc")
snap.clear_cache("ca-GrQc")

info = snap.cache_info()
info.bytes, info.datasets["ca-GrQc"].files
```

Point it somewhere else with the `NETWORKS_CACHE_DIR` environment variable, or
`snap.set_cache_dir(...)` at runtime — useful for a shared cache on a cluster.

Downloads are published with an atomic rename and coordinated by a lock file,
so concurrent processes asking for the same dataset cannot corrupt it or each
other. An interrupted download leaves no partial file behind.

`lfr/` sits beside the dataset sources and holds the LFR generators — the
fetched C++ and the binaries built from it. Deleting it costs one download and
a few seconds of compilation.

## Keeping the catalog current

The bundled catalog is a snapshot. To pick up datasets SNAP has published
since this release:

```python
snap.refresh()  # re-scrapes, stores in the cache, takes precedence
```

A refreshed catalog in the cache always wins over the bundled one; if it is
ever unreadable the package silently falls back to the snapshot rather than
breaking.

## Command line

```bash
networks categories
networks list --category "Collaboration networks"
networks list --directed --max-edges 100000
networks info ca-GrQc
networks get ca-GrQc            # prints the cached path
networks cache
networks refresh
```

## Development

```bash
uv venv && uv pip install -e ".[all]" --group dev
pytest                       # offline tests only
pytest -m network            # also hits snap.stanford.edu
python scripts/regenerate_catalog.py
```

`scripts/regenerate_catalog.py` rebuilds the vendored snapshot and refuses to
write an implausible result, so a SNAP redesign fails loudly instead of
shipping an empty catalog. CI runs it monthly and opens a pull request when the
collection changes.

## Attribution

The data belongs to the Stanford Network Analysis Project. This package only
indexes and fetches it. Please cite the datasets you use — `Dataset.citations`
carries the reference SNAP asks for — and cite the collection itself:

> Jure Leskovec and Andrej Krevl. *SNAP Datasets: Stanford Large Network
> Dataset Collection.* <https://snap.stanford.edu/data>, 2014.

OGB and PyG are other people's collections too. Cite them if you use them:

> Weihua Hu et al. *Open Graph Benchmark: Datasets for Machine Learning on Graphs.*
> NeurIPS 2020. — and the paper behind each individual dataset.

> Matthias Fey and Jan Eric Lenssen. *Fast Graph Representation Learning with PyTorch
> Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
`generate.lfr` runs Andrea Lancichinetti and Santo Fortunato's own programs,
downloaded on demand from
[their MIT-licensed repository](https://github.com/andrealancichinetti/LFRbenchmarks).
If you publish LFR results, cite them:

> Andrea Lancichinetti, Santo Fortunato and Filippo Radicchi. *Benchmark graphs
> for testing community detection algorithms.* Phys. Rev. E 78, 046110 (2008).

> Andrea Lancichinetti and Santo Fortunato. *Benchmarks for testing community
> detection algorithms on directed and weighted graphs with overlapping
> communities.* Phys. Rev. E 80, 016118 (2009).

## License

MIT. Applies to this package, not to the datasets it downloads.
