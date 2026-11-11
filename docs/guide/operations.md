# Cache, CLI and edge cases

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
netgamut export-index --output catalog.json
netgamut export-networks --output mirror
```

`export-index` writes every source's catalog as one JSON document. Each dataset
is the output of `Match.to_dict()`, and KONECT copies of networks held by another
source carry `duplicate_of`. The {doc}`explorer <../explorer>` reads this file.

`export-networks` builds the web mirror. The upstream hosts don't let a web page
read their files, so it converts Netzschleuder, KONECT and SNAP networks to
Parquet, smallest first, until `--budget` (4,000 MB) is spent. No file may be
over `--max-file` (95 MB, below GitHub's 100 MiB limit). Writing Parquet needs
pyarrow: `pip install "netgamut[export]"`. Each network is a links file and,
where the source names its nodes or has node metadata, a nodes file; next to
them, `index.json` lists every network with its files, size, counts and
citations, and says why each other one was left out.

Run it again on the same directory to update it: a network whose catalog entry
is unchanged keeps its files without a download. A network that fails through
its source (a download error, a file over the limits, an empty network, an
archive, a file the source has withdrawn) keeps its previous files, or is listed
under `skipped`. One that fails through a fault in netgamut, such as a missing
module or a file its reader cannot parse, is listed under `failed` and keeps its
previous files; the mirror's workflow turns red on any. A change to how a
source's networks are written rebuilds that source's files, and those that come
out the same stay byte for byte the same. With `--stop-after MINUTES`, it starts
no new download once that time is up and leaves the rest for the next run,
which continues where it stopped. The manifest's `complete` stays false until a
run gets through every network, and a run that changes nothing leaves
`index.json` as it was. When a source's host stops answering (a timeout, a
failed connection or a 5xx), the rest of that source waits for a later run
instead of three timeouts each. Its unbuilt networks keep their estimated size
reserved in the budget (`reserved` in the manifest), so building them later
deletes no file already published. The mirror is committed to the `mirror`
branch of
[mapequation/netgamut-networks](https://github.com/mapequation/netgamut-networks)
and served from raw.githubusercontent.com, which answers range requests, so a
browser can read a file's footer and only the columns it needs. The explorer
links each of its networks to the
[Network Navigator](https://mapequation.org/navigator/).

### The mirror's files

`index.json` says `"format": 2`; a manifest without `format` is the earlier
gzipped Pajek. Each entry in `datasets` has `key`, `title`, `url`, `citations`,
`fingerprint`, `directed`, the catalog's kinds (`bipartite`, `multigraph`,
`multilayer`, `signed`, `temporal`, `weighted`) and:

- `files`: `{"links": "<source>/<name>.links.parquet", "nodes":
  "<source>/<name>.nodes.parquet"}`, `nodes` only where there is a nodes file;
- `links`, the rows in the links file, and `nodes`, the distinct ids across
  both files; `bytes`, both files together; `layers`, where it has layers;
- `dropped`: `{"nodes": [...], "links": [...]}`, the columns left out, and
  `note`, a sentence such as why a network tagged multilayer has no layers,
  each only where there is one.

`skipped` and `failed` list `{key, reason}`.

**`<name>.links.parquet`** holds one row per row of the source's link list, in
its order: a repeated link or a timed event stays its own row.

| column | when | type |
|---|---|---|
| `source`, `target` | always | int32 if every id is an integer that fits, else int64 if every id is an integer, else string |
| `weight` | the source declares a weight | float64 |
| `time` | the source declares a time | int64 if every value is an integer, float64 if numeric, timestamp (milliseconds, UTC) for dates, else string |
| `layer` | the network has layers | int32 if every label is an integer, else string |

Ids are the source's own, so a node keeps its id across a dataset's snapshots.
An undirected link is one row, either way round. Layers come from the source's
layer column, or, for a Netzschleuder entry whose layers are its subgraphs, are
the subgraphs' names, with nodes matched across them by the key
`netzschleuder.load_multilayer` uses. A network its source tags multilayer
whose layers netgamut does not find has no `layer` column and says so in
`note`. No link joins two layers: Infomap couples them with its relax rate.

**`<name>.nodes.parquet`** holds one row per node the source's node table lists,
nodes without links included; a node that appears only in the links has no row.
`id` has the same type as `source` and `target`. `name`, the source's node name,
is cut to 100 characters, with any line break turned into a space. A column that only repeats `id` or `name`, such as
Netzschleuder's `index` or the column the names came from, is not written.
Every other node column follows in the source's order: int64 if every present value is an integer, float64 if numeric, else
string, with missing values null and values decoded where the source names its
codes. A source column called `id` or `name` that is not the name column becomes
`id_` or `name_`. A string column with any value over 100 characters or a line
break is free text (Unicode's line and paragraph separators count as line breaks), such as littlesis's biographies: it is left out and listed
under `dropped`. So are Netzschleuder's internal properties, whose names start
with `_`, such as `_pos`, graph-tool's layout, in either file. A subgraph
stack's nodes file has `id` and `name` only.

Both files are zstd-compressed in row groups of at most 1,000,000 rows, and
carry their attribution as key-value metadata: `netgamut:key`, `netgamut:title`,
`netgamut:url`, `netgamut:citation` (citations joined by newlines),
`netgamut:directed` (`true` or `false`) and `netgamut:version`.

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

### Building the docs

The {doc}`explorer <../explorer>` reads the exported catalog, so generate that
first, then build its bundle in `site/explorer`, then run Sphinx from the
repository root:

```bash
uv run netgamut export-index -o site/explorer/public/catalog.json
(cd site/explorer && npm ci && npm run build)
uv run --group docs sphinx-build -W docs docs/_build/html
```
