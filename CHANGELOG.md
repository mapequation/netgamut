# Changelog

## Unreleased

### Fixed

- Netzschleuder subgraph names are percent-encoded in download URLs, so a
  name with a space no longer fails with `InvalidURL`
  ([#16](https://github.com/mapequation/netgamut/issues/16)).
- Netzschleuder edge lists are read as CSV, quoting and all, instead of
  detecting the delimiter from split lines. A quoted field holding a comma
  (`wiki_rfa`'s dates) or a line break (`product_space`'s colours) made the
  detection fall back to whitespace, which garbled every row or read none
  ([#9](https://github.com/mapequation/netgamut/issues/9),
  [#29](https://github.com/mapequation/netgamut/issues/29)).
- SNAP's `soc-RedditHyperlinks` files are read as tab-separated, without their
  header row, and with `TIMESTAMP` as the time column
  ([#9](https://github.com/mapequation/netgamut/issues/9)).
- `export-networks` skips a network that reads as having no nodes or no links,
  with the reason under `skipped`, instead of listing an empty file; an empty
  file an earlier build listed is built again
  ([#29](https://github.com/mapequation/netgamut/issues/29)). Netzschleuder
  and SNAP networks are converted again on the next build.

- `Network.to_multilayer` keeps the source's node names, as `to_pajek` does.
- `Network.to_multilayer` reads a layer column without numpy. Only time
  layers need it, and the web mirror builds without it, so its first layered
  build left the 28 Netzschleuder networks with a layer column flat.
- Fields longer than csv's 128 KiB default, such as in Netzschleuder's
  `littlesis`, no longer stop a Netzschleuder edge list or `nodes.csv` from
  being read.

### Added

- `Columns.delimiter` and `Columns.header_row`, for a source to declare how a
  row splits and whether the first line names the fields.
- `MultilayerNetwork.from_networks({layer: network}, key=...)` stacks
  downloaded networks as layers, matching nodes by a node-metadata column, or
  by id where the networks share one id space
  ([#8](https://github.com/mapequation/netgamut/issues/8)).
- `netzschleuder.load_multilayer(name, subgraphs=None, key=None)` returns an
  entry as a multilayer network from wherever it keeps its layers: a layer
  column (`layer`, or `route_type` and `transmitter` for the entries that name
  it so), or one subgraph per layer for copenhagen, malaria_genes,
  twitter_higgs, ugandan_village and youtube_interactions
  ([#8](https://github.com/mapequation/netgamut/issues/8)).
- `export-networks` writes multilayer networks in Infomap's multilayer format
  (`*Vertices`, `*Intra`), with their number of `layers` in `index.json`. A
  network tagged multilayer whose layers are not found is written as one
  network, with a comment saying so
  ([#28](https://github.com/mapequation/netgamut/issues/28)).
- A "Multilayer networks" section in the guide.

## 0.4.0

The first release on PyPI: `pip install netgamut`.

### Renamed to netgamut

The package, its import name, its command and its repository are now all
`netgamut`. Nothing else changed, so migrating is a search and replace:

| Before | Now |
| --- | --- |
| `pip install mapequation-networks` | `pip install netgamut` |
| `import networks`, `from networks.snap import ...` | `import netgamut`, `from netgamut.snap import ...` |
| `networks list ...` on the command line | `netgamut list ...` |
| `NETWORKS_CACHE_DIR`, `NETWORKS_LFR_DIR` | `NETGAMUT_CACHE_DIR`, `NETGAMUT_LFR_DIR` |
| `github.com/mapequation/networks` | `github.com/mapequation/netgamut` |

The default cache moves from `.../networks` to `.../netgamut` the first time
the new package uses it, so no dataset is downloaded again. A cache you
pointed elsewhere with `NETWORKS_CACHE_DIR` stays where it is; set
`NETGAMUT_CACHE_DIR` to the same path to keep using it.

### Added

- `generate.overlapping_memory_benchmark`: second-order memory networks with
  communities that overlap on the physical nodes and are disjoint on the states,
  from Andrea Lancichinetti's trigram sampler, seeded. Returns a `StateNetwork`
  with the planted state partition. Needs only the standard library.
- `Match.to_dict()`: a JSON-ready summary of a dataset, the same shape for every
  source -- the fields `search()` filters on, plus description, citations and
  links. `subgraph_count` says how many networks the name holds; a
  Netzschleuder entry can hold several, the first few named in `subgraphs`.
- `netgamut export-index [--output FILE]`: every source's catalog as one JSON
  document of those summaries. KONECT's copies of other sources' networks are
  included and name the network they defer to in `duplicate_of`.
- A documentation site, with a catalog explorer for every dataset, built from
  `docs/` and `site/explorer/` and published to GitHub Pages. The explorer
  renders descriptions as Markdown and, for a Netzschleuder entry holding
  several networks, lets you pick the subgraph its load snippet fetches.
- `netgamut export-networks --output DIR`: the web mirror. Netzschleuder, KONECT
  and SNAP networks as gzipped Pajek files a browser can load, smallest first
  within a size budget, each opening with its citation, plus an `index.json`
  manifest. Run again on the same directory, it rebuilds only networks whose
  catalog entry changed. `--stop-after MINUTES` splits a long build into runs
  that each continue where the last stopped. The mirror lives on the `mirror`
  branch of `mapequation/netgamut-networks`.
- The explorer links every mirrored network to the Network Navigator, which
  opens it and finds its modules with Infomap in the browser. An "In Navigator"
  filter keeps or leaves out those networks.
- `to_pajek` names a downloaded network's nodes by the source's own names:
  Netzschleuder's `name`, `label` or `nodeLabel` column, KONECT's `ent.*` files.
  A node without a name still shows its id, and `names={}` asks for ids only.
  `NodeTable.labels()` gives the same names. Before, every node showed its id,
  so mirror files lost names like euroroad's cities.
- `netgamut.search()` filters on the kind of network: `weighted`, `bipartite`,
  `temporal`, `multilayer`, `signed` and `multigraph`, from each source's own
  catalog, leaving out datasets whose source cannot say. `Match` has the same
  properties, and the explorer's index and the web mirror's manifest carry them,
  so the explorer and the Navigator can filter on them too.
- `netgamut.search(unknown=True)` keeps the datasets whose source cannot answer
  a filter, where a size bound, a yes or no, or a node attribute leaves them out
  by default. The explorer has the same choice.
- `generate.two_level_constants(p_mod, exits, enters=None)`, returning
  `(diag, up, down)`, and `generate.two_level_table(...)`, the dense K x K
  table: the two-level flow-model constants as arrays, from the one derivation
  the blockwise distance uses too.

### Changed

- The vendored Netzschleuder catalog is refreshed to 2026-11-03: 793 networks,
  up from 286. Sixteen entries were withdrawn upstream and no longer load; most
  live on as a subgraph of a collection:

  | Removed | Now |
  | --- | --- |
  | `bison`, `cattle` | `dom`, subgraphs `Lott_1979`, `ScheinFohrman_1955` |
  | `eu_procurements_alt` | `eu_procurements`, subgraphs `AT_2008` ... |
  | `flickr_groups` | `mislove_osn`, subgraph `flickr_groups` |
  | `foodweb_baywet` | `foodweb_ulanowicz`, subgraph `baywet` |
  | `football_tsevans` | `football` |
  | `foursquare_friendships` | `foursquare_global`, subgraphs `friendship_old`, `friendship_new` |
  | `fresh_webs`, `plant_pol_kato`, `plant_pol_robertson`, `plant_pol_vazquez` | `iwdb`, subgraphs `food_web_thompson_*`, `pollination_*` |
  | `jdk`, `jung` | `software_dependencies`, subgraphs `jdk`, `jung` |
  | `movielens_100k` | `movielens_ratings`, subgraph `100k` |
  | `dblp_coauthor_snap`, `revolution` | none; SNAP's `com-DBLP` and KONECT's `brunson_revolution` remain |

- `generate.element_distances` is vectorized over rows, fast enough to call in
  a sampling loop. A row with no flow scores 0 directly rather than through a
  nan.
- `generate.cross_R` returns 0 where a denominator is degenerate, instead of
  raising `ZeroDivisionError`. That is the exact value, and what the blockwise
  path always scored there, so the two now agree on degenerate input too.

### Fixed

- On directed networks, `generate.D_blockwise`, `gmd`, `gms`, `agmd` and
  `agms` priced the step down into a module by exit rates instead of enter
  rates, so they disagreed with the dense
  `D_dist(flow_matrix(...), flow_matrix(...))`: 0.3425747 instead of 0.3427566
  on one directed hierarchy. They now use enter rates and match the dense
  model. Undirected results are unchanged.

- A KONECT network is no longer taken for a copy of a Netzschleuder or SNAP
  network because both have the same node and edge count. KONECT's
  `brunson_southern-women` (bipartite, 10 nodes, 14 edges) was hidden from
  `netgamut.search()` and the explorer as a copy of Netzschleuder's unrelated
  `global_suicide_attacks`. Equal sizes now only nominate a pair; the catalog
  build compares the two edge lists and vendors a verdict per pair in
  `konect.catalog().copies`, which `konect.duplicates()` and the cross-source
  loop check read. A pair not compared yet stays two networks. Of the 165
  pairs the old rule called copies, 126 still are: 111 compared edge by edge
  and 15 on Netzschleuder's record of importing them. Three are distinct:
  `brunson_southern-women`, `contact` (undirected in KONECT, directed in
  Netzschleuder) and `facebook-wosn-wall` (the same links, each pointing the
  other way). The other 36 are listed in their own right until compared: 34
  are over the million-edge limit of the comparison, and Netzschleuder refuses
  the download of two (`sister_cities` answers 404, `movielens_ratings` 401).

## 0.3.0

Every network -- downloaded, generated, built from paths or binned from time --
is now an object that writes Infomap's files and builds networkx and Infomap
from the same links. Generators return those objects instead of tuples, and
link weights, times and layers come only from the columns each source declares.

### Migrating from 0.2

Breaking changes, in the order you are likely to hit them. Each shows the 0.2
call and its 0.3 replacement.

#### 1. Generators return a network object, not a tuple

The planted partition is `net.partition`, the diagnostics are `net.info` (with
`"generator"` and `"parameters"` added), and `net.to_networkx()` is the graph,
with each node's planted module in the `module` attribute
(`net.to_networkx(partition_attribute=None)` leaves it out).

| 0.2 | 0.3 |
| --- | --- |
| `H, info = generate.benchmark(G, partition)` | `net = generate.benchmark(G, partition)`; `H, info = net.to_networkx(), net.info` |
| `H, info = generate.weighted_benchmark(G, partition)` | same as `benchmark` |
| `H, partition, info = generate.parametric(...)` | `net = generate.parametric(...)`; `net.to_networkx(), net.partition, net.info` |
| `G, partition = generate.lfr_networkx(...)` | `net = generate.lfr_networkx(...)`; `net.to_networkx(), net.partition` |
| `G, part = generate.lfr(...)` | `net = generate.lfr(...)`; `net.to_networkx(), net.partition` |
| `G, cover = generate.lfr(..., on=20, om=2)` | same; `net.partition` is the cover `{node: (module, ...)}` as before |
| `layers, part = generate.lfr(..., layers=10)` | `net = generate.lfr(..., layers=10)`, a `MultilayerNetwork`; `{layer: net.to_networkx(layer=layer) for layer in net.layers}`, `net.partition` |
| `sampled, info = generate.multilayer_benchmark(layers, partition)` | `net = generate.multilayer_benchmark(layers, partition)`, a `MultilayerNetwork`; `sampled` is `net.intra`, `info` is `net.info` |
| `sampled, info = generate.memory_benchmark(links, windows, partition)` | `net = generate.memory_benchmark(links, windows, partition)`, a `StateNetwork`; `sampled` is `net.links`, `info` is `net.info` |

Partition shapes are unchanged: `{node: module}`, `{node: (macro, micro)}` for
hierarchical LFR, `{(node, layer): module}` for multilayer networks.

`memory_benchmark` keys `net.links` and `net.partition` by state id. When the
blueprint's state labels are non-negative integers the ids are those labels
(at order 1, the physical node ids); otherwise states are numbered from 1 and
`net.states[id]` gives each state's window.

`multilayer_benchmark` also takes a `MultilayerNetwork` as the blueprint and
then defaults to its planted partition, so `generate.lfr(..., layers=10)` feeds
straight in.

#### 2. `Network(weighted_links=...)` is gone; declare columns instead

Code that builds a `Network` by hand:

```python
# 0.2
Network(dataset=..., file=..., archive_path=..., weighted_links=True)
# 0.3 -- say which field is the weight; field 0 is the source, 1 the target
from networks import Columns

Network(dataset=..., file=..., archive_path=..., columns=Columns(weight=2))
```

`net.weighted_links` still reads, as `net.columns.weight is not None`. To
change the columns of a loaded network, replace them:

```python
import dataclasses
from networks import Columns

net = dataclasses.replace(net, columns=Columns(weight=2))  # weighted
net = dataclasses.replace(net, columns=dataclasses.replace(net.columns, weight=None))  # unweighted
```

#### 3. Some downloaded networks load with different weights

No code change is needed, but `to_infomap()` results change for these:

- **SNAP files with a third column are unweighted** unless that column is a
  weight. In 0.2 Infomap read any third column as a weight, so the
  timestamps of the temporal datasets (`sx-*`, `wiki-talk-temporal`,
  `CollegeMsg`, `email-Eu-core-temporal*`, `higgs-activity_time`) were taken
  as link weights, and the signed ones (`soc-sign-*`) failed on negative
  weights. Only the Higgs `retweet`/`reply`/`mention` networks keep a weight.
  Times are now in `net.times`.
- **Netzschleuder networks with a numeric `weight` or `count` edge column are
  weighted** by it; 0.2 read every Netzschleuder network unweighted. Datasets
  tagged `Signed` stay unweighted. `print(net.columns)` shows what was
  declared; the snippet in section 2 restores the 0.2 reading.
- KONECT is unchanged: weighted when the dataset's weights are ones Infomap
  can use.

### Updating a project that depends on this package

Projects that take the package from git, as with uv:

```toml
[tool.uv.sources]
mapequation-networks = { git = "https://github.com/mapequation/networks" }
```

update with `uv lock --upgrade-package mapequation-networks && uv sync`. To
stay on 0.2 until the code is migrated, pin the tag:
`{ git = "https://github.com/mapequation/networks", tag = "v0.2.0" }`.

To find what needs changing, search the project for the 0.2 call shapes:

```sh
grep -rnE "\b(benchmark|weighted_benchmark|parametric|lfr|lfr_networkx|memory_benchmark|multilayer_benchmark)\(|weighted_links" --include='*.py' --include='*.ipynb' .
```

Every match that unpacks a tuple needs the change from section 1, and every
`weighted_links=` the one from section 2. Then rerun anything whose Infomap
results come from the downloads in section 3.

### Added

- `to_pajek(directory, *, name=None)` and `to_link_list(...)` on every
  network -- downloads, generated, multilayer, state and path data -- returning
  the written `Path`. Multilayer networks write `*Vertices`/`*Intra`/`*Inter`,
  state networks `*Vertices`/`*States`/`*Links`. Files are written atomically,
  with a header saying where the network came from.
- `InMemoryNetwork` (generated networks and time snapshots) and
  `MultilayerNetwork`, both with `to_networkx()`, `to_infomap()` and the
  writers; `StateNetwork.to_networkx()`.
- `generate.multilayer_gm`: MultilayerGM's multiplex and temporal multilayer
  benchmarks in one call, from a vendored copy of MultilayerGM-py.
- `Columns`: which field of each row is the weight, time and layer, declared
  by every source.
- Times: `net.times` (a numpy array), `net.time_bins(bins, range)` with
  `numpy.histogram_bin_edges` semantics, `net.snapshots(bins)` iterating one
  network per bin lazily, and `net.to_multilayer(bins=...)` with one layer per
  bin. `net.to_multilayer(layer=...)` builds layers from a layer column.

## 0.2.0

The release before the network objects; see the git history.
