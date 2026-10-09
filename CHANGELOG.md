# Changelog

## Unreleased

- `generate.overlapping_memory_benchmark`: second-order memory networks with
  communities that overlap on the physical nodes and are disjoint on the states,
  from Andrea Lancichinetti's trigram sampler, seeded. Returns a `StateNetwork`
  with the planted state partition. Needs only the standard library.

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
