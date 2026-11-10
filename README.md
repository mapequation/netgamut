# netgamut

Browse, download and cache network datasets from Python.

Seven sources are indexed, each in its own subpackage:

| Source | Datasets | Strength |
|---|---|---|
| [`netgamut.snap`](https://snap.stanford.edu/data/index.html) | 109 | large, well-known social / web / technological networks |
| [`netgamut.netzschleuder`](https://networks.skewed.de) | 286 | domain-tagged, incl. biological, transport and multilayer |
| [`netgamut.konect`](http://konect.cc) | 300 | one fixed vocabulary of format, edge type and **loops**; signed and temporal |
| [`netgamut.ogb`](https://ogb.stanford.edu) | 12 | graph-learning benchmarks, with **named** node categories |
| [`netgamut.pyg`](https://pytorch-geometric.readthedocs.io) | 99 | the graph-learning corpus: citation, web, social, heterophilous |
| `netgamut.paths` | 3 | curated path data (flight itineraries, navigation trails) for memory networks |
| [`netgamut.ahorn`](https://ahorn.rwth-aachen.de) | 83 | higher-order data: hypergraphs, simplicial complexes, revision-pinned |

Metadata for all seven **ships inside the package**, so you can explore every
dataset — domains, sizes, statistics, citations — without downloading anything.
When you do ask for data it lands in a shared user-level cache, so a dataset is
fetched once per machine no matter how many projects, notebooks or virtualenvs
use it.

```bash
pip install netgamut
```

For a runnable tour — browsing, loading, and building memory networks from
path data — see [`examples/tutorial.ipynb`](examples/tutorial.ipynb).
Upgrading from 0.2? Generators now return network objects rather than tuples,
and some downloads load with different weights; [`CHANGELOG.md`](CHANGELOG.md)
has the call-by-call migration.

## One domain vocabulary across every source

The sites label things differently — SNAP groups by provenance ("Autonomous
systems graphs"), Netzschleuder tags by domain, AHORN by a vocabulary of its
own, KONECT by kind of relationship, OGB and PyG by nothing at all — so all seven
are mapped onto one vocabulary.
That makes it possible to compare methods across *kinds of system* rather than
kinds of provenance:

```python
import netgamut

netgamut.domains()
```

```
{'Social': 366, 'Informational': 159, 'Biological': 51,
 'Technological': 32, 'Transportation': 42, 'Economic': 91}
```

```python
for match in netgamut.search(domain="Biological", max_nodes=500):
    print(match)
# netzschleuder/blumenau_drug [Biological] (75 nodes, 181 edges)
# netzschleuder/celegans_2019 [Biological] (453 nodes, 4,879 edges)
# netzschleuder/celegans_metabolic [Biological] (453 nodes, 4,596 edges)
# ...

net = netgamut.load("celegans_metabolic")
```

Note `Biological: 51` — 36 from Netzschleuder, 5 each from KONECT and OGB, 4
from AHORN, 1 from PyG, and none at all from SNAP. That imbalance is why the other sources exist.
The mapping is deliberately visible in
`netgamut/_core/domains.py` rather than buried per source, so it can be argued
with in one place.

For anything source-specific, use that source's own `search()`:

```python
from netgamut import netzschleuder as nz
from netgamut import snap

nz.search(multilayer=True)  # 17 multiplex networks; SNAP has none
nz.search(subtype="Food web")
snap.search(ground_truth=True)  # reference partitions
```

## Look before you download (SNAP)

```python
from netgamut import snap

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
graph = net.to_networkx()  # pip install "netgamut[networkx]"
im = net.to_infomap()  # pip install "netgamut[infomap]"
im.run()
```

Both default to the direction SNAP reports for the dataset.

### Writing files

Every network — downloaded or generated, first-order, memory or multilayer —
writes itself into a directory and returns the path, named after the network
unless you pass `name=`:

```python
net.to_pajek("out/")                  # PosixPath('out/ca-GrQc.net')
net.to_link_list("out/", name="grqc") # PosixPath('out/grqc.txt')
```

| Network | `to_pajek` | `to_link_list` |
|---|---|---|
| first-order (`Network`, `InMemoryNetwork`) | `*Vertices` + `*Edges`/`*Arcs` | `source target [weight]` |
| `StateNetwork` | `*Vertices` + `*States` + `*Links` | order 1 only |
| `MultilayerNetwork` | `*Vertices` + `*Intra` (+ `*Inter`, or `*Multilayer`) | — |

Integer node ids are written as they are, so a generated network's file lines up
with its planted partition; other labels are numbered from 1 and kept as Pajek
names. A link list has nowhere to put names, so it refuses non-integer labels
rather than renumbering them silently. A written file loads the same network
Infomap gets from `to_infomap()`, weights included. Downloads are streamed, so
writing one never holds its links in memory.

### Weights, times and layers

A third column is a weight in one file, a Unix timestamp in the next and a
sign or rating in a third, so nothing here guesses. Each source declares what
the extra columns of the file it loaded hold — Netzschleuder from the file's
own header, KONECT from its fixed `source target weight time` layout, SNAP from
a table of its temporal and signed files — and `to_infomap()`, the writers and
everything below follow it:

```python
net = snap.load("CollegeMsg")
print(net.columns)           # source target time (time)
net.weighted_links           # False: the timestamp is not a weight
```

Temporal networks expose their times as a numpy array, read once, so you can
look before you bin:

```python
t = net.times                        # int64 Unix seconds, in file order
t.min(), t.max()
net.time_bins("auto")                # numpy.histogram_bin_edges(t, "auto")

for snapshot in net.snapshots(bins=12):   # lazy: one network per bin
    print(snapshot.info["start"], snapshot.info["end"], snapshot)
```

`bins` and `range` mean what they mean to `numpy.histogram_bin_edges`: a
number of equal-width bins, explicit edges, or an estimator name. Bin `i` is
`[edges[i], edges[i+1])`, the last one closed; every bin yields, empty ones
too, and a link's weight in a snapshot is how often it occurs there (summed
weights where the file has a weight column). ISO dates come back as
`datetime64`.

The same bins, or a layer column, make a multilayer network: see
[Multilayer networks](#multilayer-networks).

If a source has a column wrong, or you know better, declare it yourself:
`dataclasses.replace(net, columns=netgamut.Columns(time=2))`. `Columns` also
says how a row splits: `delimiter=","` reads a CSV whose quoted fields hold
commas or line breaks, `delimiter="\t"` a TSV whose fields hold spaces, and
`header_row=True` leaves out a first line that names the fields. Netzschleuder
declares its CSV, and SNAP its Reddit hyperlink TSVs.

Some entries are multi-file bundles:

```python
net = snap.load("ego-Facebook", file="facebook.tar.gz")
directory = net.extract()
```

### Multilayer networks

A multilayer network has nodes that appear in several layers: the lines of a
transport network, the kinds of tie between the same people. netgamut builds
one as a `MultilayerNetwork`, the shape Infomap's multilayer model reads, from
wherever a source keeps its layers. Each layer's links are summed, as counts or
as weights where the file has a weight column, and the source's node names come
along.

A time column gives a layer per time bin, the same bins as `snapshots`:

```python
ml = net.to_multilayer(bins=12)          # layer i = bin i; ml.info["bin_edges"]
ml.to_infomap(multilayer_relax_rate=0.15, multilayer_relax_limit=1)  # couple adjacent bins
```

A layer column gives a layer per value, and any categorical column can serve:

```python
netgamut.load("london_transport").to_multilayer()             # its layer column
snap.load("higgs-twitter", file="higgs-activity_time.txt.gz").to_multilayer(
    layer="interaction"                                       # retweet, mention, reply
)
```

Netzschleuder keeps the layers of its multilayer entries in different places.
`netzschleuder.load_multilayer` knows where:

```python
from netgamut import netzschleuder as nz

nz.load_multilayer("london_transport")         # its layer column
nz.load_multilayer("kujala_public_transport")  # route_type: tram, rail, bus...
nz.load_multilayer("copenhagen")               # subgraphs sms, fb_friends, calls and bt
```

copenhagen, malaria_genes, twitter_higgs, ugandan_village and
youtube_interactions keep each layer in a subgraph of its own. Each subgraph
numbers its nodes `0..n-1` by itself, so the stacked layers match a node by a
`nodes.csv` column: copenhagen's `id`, the `name` of the others, and the row
number for youtube_interactions, whose subgraphs list the same users in the same
order. Name other subgraphs to stack, and the column that matches their nodes,
yourself:

```python
nz.load_multilayer("ugandan_village", ["friendship-5", "health-advice_5"])  # village 5
nz.load_multilayer("copenhagen", ["calls", "sms"], key="id")
```

For column-layered entries with several subgraphs, load the subgraph you want
and lay it out: `nz.load("kujala_public_transport", subgraph="berlin").to_multilayer()`.

Three entries are tagged multilayer, but their files hold no layers:
physician_trust merges its advice, discussion and friendship ties; iwdb is a
collection of separate studies; and in paris_transportation's current upstream
file, every link has one end on a node without attributes. `load_multilayer`
raises `LookupError` for them.

Networks you loaded yourself stack with `MultilayerNetwork.from_networks`. Files
that share one id space, as SNAP's higgs-twitter files do, need no `key`:

```python
layers = {
    kind: snap.load("higgs-twitter", file=f"{kind}_network.edgelist.gz")
    for kind in ("retweet", "reply", "mention")
}
ml = netgamut.MultilayerNetwork.from_networks(layers, name="higgs")
```

None of these layerings links one layer to another: Infomap couples them with
its relax rate. `ml.to_pajek(directory)` writes Infomap's multilayer format
(`*Vertices`, then `*Intra` lines of layer, node, node and weight), and
`ml.to_infomap()` builds the same network in Infomap. The web mirror writes
multilayer networks in this format, with their number of `layers` in
`index.json`.

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
that is what [`netgamut.generate`](#generating-benchmarks) is for.

## Netzschleuder

```python
from netgamut import netzschleuder as nz

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

## KONECT

KONECT describes every network in one fixed vocabulary — a format (`sym`,
`asym`, `bip`), an edge type (`unweighted`, `positive` for multigraphs,
`posweighted`, `signed`, ...) and structural checks — and opens every data file
with the same two codes, `% asym multisigned`. That makes it the one source here
that says, for every network, whether it contains **loops**:

```python
from netgamut import konect

konect.search(loops=True, max_nodes=10_000)  # self-links, by KONECT's own check
konect.search(signed=True, temporal=True)
konect.search(bipartite=True, domain="Economic")
print(konect.describe("chess"))
```

The loop check also reaches the cross-source search, `netgamut.search(loops=True)`,
including for a Netzschleuder or SNAP network whenever KONECT holds the identical
edge list. Networks no source has checked are left out of either filter.

The catalog is KONECT's downloadable networks minus its per-language Wikimedia
series — about a thousand edit, hyperlink and talk networks, one per language
edition — of which only the English editions are kept.

**Overlap.** Netzschleuder imported a third of KONECT, and some of KONECT is
SNAP's. Such a network stays in `konect.search()`, but the cross-source
`netgamut.search`, `netgamut.domains` and `netgamut.load` use the other copy, so
nothing is counted twice. A duplicate is decided on evidence, never on name —
KONECT's `twitter` is Netzschleuder's `twitter_social`, and Netzschleuder's
`twitter` is KONECT's `munmun_twitter_social`:

```python
konect.info("moreno_lesmis").duplicate_of  # 'netzschleuder/lesmis'
konect.duplicates()  # KONECT name -> the copy used elsewhere
```

The evidence is either Netzschleuder naming the KONECT network as its upstream,
or another entry with exactly the same node and edge count. Near-copies — a
different snapshot, the largest component only — stay separate networks.

**Loading.** The archive is kept as KONECT serves it, and the edge list is
rewritten once beside it so every tool reads it the same way:

```python
net = konect.load("brunson_southern-women")
net.path  # <name>.tsv: '%' comments become '#', which Infomap requires
net.rows()  # every column kept: weights, signs, timestamps
net.extract()  # KONECT's README and meta files
```

KONECT numbers the two sides of a bipartite network independently, so the right
side is shifted past the left: ids are unique, and right-side nodes start at
`info(name).left_nodes + 1`. Signed and rating weights stay in the file but are
not handed to Infomap, which accepts only non-negative weights; `to_infomap()`
then reads the links unweighted, summing repeated ones.

**Node metadata.** 27 networks ship per-node attributes as `ent.*` files — people's
names and sexes in `moreno_crime`, a whole profile per pet in the petster networks,
coauthor names in `pajek-erdos`. No KONECT page says which, so the catalog build
lists every archive up to 100 MB (268 of 300) and records the columns each will
yield:

```python
konect.search(node_metadata=True)
konect.info("petster-hamster").node_attributes  # ('name', 'joined', 'species', ...)
konect.load("moreno_crime").nodes()["name"]  # {1: 'AbelDennis', ...}
```

Ids are the edge list's, so in a bipartite network a right-side entity's rows land
on the shifted ids. The 32 larger archives were not listed and answer `None`, which
keeps them out of either filter. A refresh re-lists an archive only when its size
or date has changed; KONECT's last changed in 2017–2018.

## Node metadata, in one shape

Every source stores node attributes somewhere different — a separate SNAP artifact,
a member of a Netzschleuder bundle, KONECT's `ent.*` files, OGB's `raw/` and
`mapping/` folders, a materialised PyG table. `Network.nodes()` reads all of them the same way:

```python
nodes = netgamut.load("ogbn-arxiv").nodes()

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
netgamut.load("email-Eu-core").nodes()["department"]        # SNAP
netgamut.load("karate", source="netzschleuder").nodes()["groups"]
netgamut.load("ogbn-arxiv").nodes()["arxiv_category"]       # OGB
netgamut.load("Planetoid/Cora", source="pyg").nodes()["label"]
```

To find datasets that have node metadata before downloading anything, ask
across sources — by presence, or by column name:

```python
netgamut.search(node_metadata=True)  # 240 datasets known to carry some
netgamut.search(node_attribute="gender")  # something to score a partition against
netgamut.search("karate")[0].node_attributes  # ('name', 'groups')
```

Each source answers from its own catalog: Netzschleuder from the site's analyses,
KONECT from its archive listings, SNAP from its label files, OGB and PyG from their
class labels. Netzschleuder, KONECT and SNAP name every column. OGB names only its
extra files (`year`, `species`) and PyG none, since the rest of their column names
come with the download. Datasets a catalog cannot answer for — all of AHORN, whose
node lines live inside the dataset file — are left out of either filter.

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

`netgamut.ogb` and `netgamut.pyg` index the datasets the graph-neural-network
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
"netgamut[pyg]"`, runs PyG's own downloader, and writes `edges.csv.gz` and
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

```
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
from netgamut import paths

data = paths.load("wikispeedia")      # 9.5 MB, 51,318 finished navigation trails
memory = data.to_state_network(order=2)
print(memory)
# StateNetwork(order=2): 4,169 physical nodes, 58,944 state nodes,
# 154,895 links from 50,509 paths

memory.to_pajek("out/")  # out/wikispeedia.net: *Vertices / *States / *Links
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
netgamut get db1b-coupon --year 2023 --quarter 3
```

The same itineraries exist at the level of metropolitan areas, and airports
have coordinates for plotting:

```python
flights = paths.load("db1b-coupon")
where = flights.coordinates()          # {'LAX': (33.94, -118.41), ...} from OpenFlights
cities = paths.load("db1b-city")       # same archive, shared cache, no second download
for markets, passengers in cities.paths():  # ('31703', '30977', ...) BTS CityMarketIDs
    ...
cities.names()["31703"]                # 'New York City, NY (Metropolitan Area)'
```

`db1b-city` names each node by its BTS `CityMarketID`, so JFK, LGA and EWR
are one node and an airport change inside a city collapses to one visit; the
labels stay ids (stable, and joinable with BTS's own tables) and `names()`
fetches the `L_CITY_MARKET_ID` lookup for the descriptions. Both side tables
are small and land in the cache next to the download on first use.

Two parsing choices to be aware of, both deliberate: Wikispeedia back clicks
(`<`, in ~17% of trails) are resolved against the browsing history, so paths
record the articles actually visited; and DB1B itineraries keep their full
airport sequence including the return legs, since that is the memory the
model is for.

Note that `wikispeedia` now exists in two sources — the navigation *paths*
here, the article *hyperlink graph* in SNAP — so the top-level
`netgamut.load("wikispeedia")` asks you to pass `source="paths"` or
`source="snap"` rather than guessing.


## Higher-order data: hypergraphs and complexes

An edge list says two things met. A hypergraph says *how many* met at once, and
that is not recoverable afterwards — three authors on one paper and three
authors on three pairwise papers project onto the same triangle. The
[`ahorn`](https://ahorn.rwth-aachen.de) source indexes the Aachen Higher-Order
Repository of Networks: 83 datasets whose unit of observation is a **set** of
nodes, so `load` returns a `Hypergraph`, not a `Network`.

```python
from netgamut import ahorn

ahorn.search(network_type="simplicial-complex", max_nodes=1000)
ahorn.search(tag="benson")          # everything from the Benson collection
ahorn.search(temporal=True)         # 16 with timestamped interactions

h = ahorn.load("algebra-questions")
print(h)
# algebra-questions revision 2: 423 nodes, 1,268 interactions

for members, attributes in h.interactions(cast=int):
    ...   # ((2, 12, 20), {'category': 'lie-algebras'})
```

Flattening is a choice you make, never something that happens behind your back:

```python
h.to_networkx(projection="clique")  # every pair inside an interaction; lossy
h.to_networkx(projection="star")    # interactions as nodes; lossless
h.to_network()                      # a plain Network over the clique projection
```

`to_infomap()` uses the bipartite representation, which Infomap reads natively,
so the interaction sizes are still in the model when the modules are found —
the clique projection would have thrown them away first:

```python
im = h.to_infomap(silent=True)
result = im.run()   # modules span both nodes and interactions
```

Check `max_interaction_size` before projecting a wide one: an interaction of
*k* members becomes *k(k-1)/2* edges, and `algebra-questions` has one of 107.

### Revisions are pinnable, and only here

AHORN archives each revision as its own Zenodo record, so a pinned call keeps
returning the same bytes after a correction lands:

```python
ahorn.load("algebra-questions", revision=1)   # reproducible
ahorn.load("algebra-questions")               # latest, and not
```

That matters more than it sounds: revision 2 of that dataset renamed the
attribute revision 1 called `genre`. `describe()` prints the changelog, and
revisions cache side by side so pinning one does not evict another.

### Node and interaction attributes

Node attributes arrive as the same `NodeTable` every other source produces, and
interaction attributes — which are the interesting reference labels here — come
off the interactions:

```python
ahorn.load("senate-committees").nodes().partition("party")
# {'1': 0, '2': 0, ...}   Democrat / Republican

h.interaction_labels("category")   # one label per interaction, in file order
```

The same caveat as everywhere: an attribute is what someone recorded, not a
claim about flow.

### Multi-network files

Two datasets hold many networks in one file — `MANTRA-2-manifolds` is 43,214
triangulated manifolds, each with its own Betti numbers and orientability:

```python
m = ahorn.load("MANTRA-2-manifolds")
m.networks()[0]
# {'id': 'manifold_2_4_5_1', 'betti_numbers': [1, 0, 1], 'name': 'S^2', ...}
m.interactions(network="manifold_2_4_5_1")
```

Three versions of AHORN's exchange format are in circulation — 0.1, 0.2 and the
0.3 the specification documents — and they disagree about whether the reserved
metadata keys are underscore-prefixed. The parser reads all three;
`h.format_version` says which one a file used, and `h.file_revision` is `None`
for the 0.1 files, which recorded no revision at all.

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
snap.cache_dir()  # ~/Library/Caches/netgamut on macOS
# ~/.cache/netgamut on Linux
snap.is_cached("ca-GrQc")
snap.clear_cache("ca-GrQc")

info = snap.cache_info()
info.bytes, info.datasets["ca-GrQc"].files
```

Point it somewhere else with the `NETGAMUT_CACHE_DIR` environment variable, or
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
netgamut categories
netgamut list --category "Collaboration networks"
netgamut list --directed --max-edges 100000
netgamut info ca-GrQc
netgamut get ca-GrQc            # prints the cached path
netgamut get algebra-questions --revision 1   # pin an AHORN revision
netgamut cache
netgamut refresh
```

## Development

```bash
uv venv && uv pip install -e ".[all]" --group dev
pytest                       # offline tests only
pytest -m network            # also hits snap.stanford.edu
python scripts/regenerate_catalog.py
python scripts/regenerate_ahorn_catalog.py
```

Each `scripts/regenerate_*_catalog.py` rebuilds one vendored snapshot and
refuses to write an implausible result, so an upstream redesign fails loudly
instead of shipping an empty catalog. `refresh-catalogs.yml` runs the SNAP,
Netzschleuder and AHORN ones monthly and opens a pull request when a collection
changes; the OGB and PyG scripts are run by hand.

The AHORN one reads YAML frontmatter out of the repository behind the site, so
it needs `pyyaml` — which is in the dev group and *not* a dependency of the
package, because only rebuilding needs it. `ahorn.refresh()` says so if it is
missing.

### Releasing

Pushing a `v*` tag publishes that version to PyPI. `publish.yml` checks that
the tag matches `__version__`, runs the whole CI workflow, and uploads the
wheel and sdist that CI's build job made.

1. On a branch, set `__version__` in `src/netgamut/__init__.py` and rename the
   changelog's `## Unreleased` section to that version. Merge it to main.
2. Tag the merge commit on main and push the tag:

```bash
git tag -a v0.4.0 -m "0.4.0: <one-line summary>; see CHANGELOG.md"
git push origin v0.4.0
```

PyPI never takes the same version twice, so a bad release is fixed with a new
version, not a re-upload. If the workflow fails before the upload step, delete
the tag (`git push origin :v0.4.0`), fix main, and tag again.

There is no token to keep: PyPI trusts `publish.yml` in this repository when it
runs in the `pypi` environment, as set up under the project's Publishing
settings on pypi.org.

## Attribution

The data belongs to the Stanford Network Analysis Project. This package only
indexes and fetches it. Please cite the datasets you use — `Dataset.citations`
carries the reference SNAP asks for — and cite the collection itself:

> Jure Leskovec and Andrej Krevl. *SNAP Datasets: Stanford Large Network
> Dataset Collection.* <https://snap.stanford.edu/data>, 2014.

KONECT asks you to cite the collection alongside each network's own reference
(both are in `Dataset.citations`):

> Jérôme Kunegis. *KONECT – The Koblenz Network Collection.* Proc. Int. Conf. on
> World Wide Web Companion, 1343–1350, 2013.

OGB and PyG are other people's collections too. Cite them if you use them:

> Weihua Hu et al. *Open Graph Benchmark: Datasets for Machine Learning on Graphs.*
> NeurIPS 2020. — and the paper behind each individual dataset.

> Matthias Fey and Jan Eric Lenssen. *Fast Graph Representation Learning with PyTorch
> Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
AHORN is a repository in its own right, and it asks you to cite two things
separately: the original source publication for a dataset — the BibTeX in
`Dataset.citations`, plus the Zenodo record when the exact revision matters —
and the repository itself, in `ahorn.CITATION`:

> Florian Frantzen and Michael T. Schaub. *A repository for discovery and reuse
> of higher-order network datasets.* arXiv:2608.03491 (2026).

Licences on AHORN records are the upstream source's, and grant nothing beyond
what that source already allowed.

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
