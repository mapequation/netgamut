# Getting started

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
path data — see the {doc}`../tutorial`.
Upgrading from 0.2? Generators now return network objects rather than tuples,
and some downloads load with different weights; [`CHANGELOG.md`](https://github.com/mapequation/netgamut/blob/main/CHANGELOG.md)
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

```text
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

For anything source-specific, use that source's own `search()` (see {doc}`sources`):

```python
from netgamut import netzschleuder as nz
from netgamut import snap

nz.search(multilayer=True)  # 17 multiplex networks; SNAP has none
nz.search(subtype="Food web")
snap.search(ground_truth=True)  # reference partitions
```

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

{func}`netgamut.snap.load` returns a {class}`netgamut.Network` handle, not a parsed graph — so the same call works
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
