# networks

Browse, download and cache network datasets from Python.

Two repositories are indexed, each in its own subpackage:

| Source | Datasets | Strength |
|---|---|---|
| [`networks.snap`](https://snap.stanford.edu/data/index.html) | 109 | large, well-known social / web / technological networks |
| [`networks.netzschleuder`](https://networks.skewed.de) | 286 | domain-tagged, incl. biological, transport and multilayer |

Metadata for both **ships inside the package**, so you can explore every
dataset — domains, sizes, statistics, citations — without downloading anything.
When you do ask for data it lands in a shared user-level cache, so a dataset is
fetched once per machine no matter how many projects, notebooks or virtualenvs
use it.

```bash
pip install mapequation-networks
```

The distribution is `mapequation-networks`; the import name is `networks`.

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
{'Social': 212, 'Informational': 70, 'Biological': 36,
 'Technological': 30, 'Transportation': 16, 'Economic': 39}
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

nz.search(multilayer=True)              # 17 multiplex networks; SNAP has none
nz.search(subtype="Food web")
snap.search(ground_truth=True)          # reference partitions
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
    "Social":         [("nz", "karate"), ("nz", "football"), ("snap", "email-Eu-core")],
    "Informational":  [("nz", "polbooks"), ("nz", "polblogs"), ("nz", "dblp_cite")],
    "Biological":     [("nz", "celegansneural"), ("nz", "celegans_metabolic"),
                       ("nz", "foodweb_little_rock"), ("nz", "malaria_genes")],
    "Transportation": [("nz", "london_transport"), ("nz", "openflights"),
                       ("nz", "us_air_traffic"), ("snap", "roadNet-PA")],
    "Technological":  [("nz", "power"), ("nz", "internet_as"), ("nz", "linux")],
    "Economic":       [("nz", "fao_trade"), ("nz", "at_migrations")],
}
```

Reference partitions exist for `karate` (2 factions), `football` (12
conferences), `polbooks`/`polblogs` (political alignment), `malaria_genes`
(CysPoLV/UPS), `foodweb_little_rock` (trophic groups) and `email-Eu-core`
(42 departments).

Treat those with care: node metadata is **not** guaranteed to be the community
structure, a point reviewers will raise. Use synthetic benchmarks with planted
communities for the quantitative claim, and this pool for generality — reported
via codelength against baselines. Neither source provides synthetic generators.

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
nz.search(multilayer=True)           # 17 multiplex networks
nz.search(weighted=True, domain="Economic")
nz.search(bipartite=True)
```

Downloads use the site's CSV bundle, so **graph-tool is not required** despite
being the site's native format:

```python
net = nz.load("celegansneural")
net.path                     # extracted edges.csv
list(net.edges(cast=int))
net.extract()                # nodes.csv and gprops.csv too
```

Metadata is read from a JSON API rather than scraped, so this subpackage is
markedly less fragile than the SNAP one.


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

## License

MIT. Applies to this package, not to the datasets it downloads.
