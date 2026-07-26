# networks

Browse, download and cache network datasets from Python.

Metadata for the whole [Stanford SNAP collection](https://snap.stanford.edu/data/index.html)
ships inside the package, so you can explore every category, compare dataset
sizes and read the published statistics **without downloading anything**. When
you do ask for data, it lands in a shared user-level cache — so a dataset is
fetched from Stanford once per machine, no matter how many projects, notebooks
or virtualenvs use it.

```bash
pip install mapequation-networks
```

The distribution is `mapequation-networks`; the import name is `networks`.

## Look before you download

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
