# Working with networks

## Writing files

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

## Weights, times and layers

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

The same bins, or a layer column, make a multilayer network:

```python
ml = net.to_multilayer(bins=12)          # layer i = bin i; ml.info["bin_edges"]
ml.to_infomap(multilayer_relax_rate=0.15, multilayer_relax_limit=1)  # couple adjacent bins

netgamut.load("london_transport").to_multilayer()             # its layer column
snap.load("higgs-twitter").to_multilayer(layer="interaction") # any categorical column
```

If a source has a column wrong, or you know better, declare it yourself:
`dataclasses.replace(net, columns=netgamut.Columns(time=2))`.

Some entries are multi-file bundles:

```python
net = snap.load("ego-Facebook", file="facebook.tar.gz")
directory = net.extract()
```

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
