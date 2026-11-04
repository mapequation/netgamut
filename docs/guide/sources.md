# The sources

## Look before you download (SNAP)

```python
from netgamut import snap

for category in snap.categories():
    print(category)
```

```text
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

```text
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

The evidence is recorded when the catalog is built: either Netzschleuder names
the KONECT network as its upstream (`imported`), or the two edge lists were
downloaded and found equal, as they stand or up to a relabelling of the nodes
(`same-edges`). Equal node and edge counts only nominate a pair: KONECT's
`brunson_southern-women` has the same 10 nodes and 14 edges as Netzschleuder's
`global_suicide_attacks`, and is a different network. A pair not compared yet,
usually because it is too large, stays two networks, as do near-copies — a
different snapshot, the largest component only. `konect.catalog().copies` lists
every pair considered, with its verdict.

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
