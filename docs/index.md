# netgamut

Browse, download and cache network datasets from Python.

Seven sources are indexed, each in its own subpackage, and mapped onto one
domain vocabulary so a comparison can span kinds of system rather than kinds of
provenance:

| Source | Datasets | Strength |
|---|---|---|
| {mod}`netgamut.snap` | 109 | large, well-known social / web / technological networks |
| {mod}`netgamut.netzschleuder` | 286 | domain-tagged, incl. biological, transport and multilayer |
| {mod}`netgamut.konect` | 300 | one fixed vocabulary of format, edge type and **loops**; signed and temporal |
| {mod}`netgamut.ogb` | 12 | graph-learning benchmarks, with **named** node categories |
| {mod}`netgamut.pyg` | 99 | the graph-learning corpus: citation, web, social, heterophilous |
| {mod}`netgamut.paths` | 3 | curated path data (flight itineraries, navigation trails) for memory networks |
| {mod}`netgamut.ahorn` | 83 | higher-order data: hypergraphs, simplicial complexes, revision-pinned |

Metadata for all seven ships inside the package, so every dataset can be
explored without downloading anything; the [catalog explorer](explorer.md) shows
them all in your browser. A dataset you do load lands in a shared user-level
cache, so it is fetched once per machine.

```bash
pip install netgamut
```

```python
import netgamut

netgamut.domains()
# {'Social': 201, 'Informational': 68, 'Biological': 36, ...}

for dataset in netgamut.search(domain="Biological", max_nodes=1000):
    print(dataset)

net = netgamut.load("karate", source="netzschleuder")
im = net.to_infomap(silent=True)
```

```{toctree}
:hidden:

explorer
```

```{toctree}
:caption: Guide
:maxdepth: 2

guide/quickstart
guide/networks
guide/sources
guide/benchmarks
guide/operations
tutorial
guide/attribution
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
```

```{toctree}
:caption: Links

mapequation.org <https://www.mapequation.org>
GitHub <https://github.com/mapequation/netgamut>
```
