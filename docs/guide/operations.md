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
gzipped Pajek, smallest first, until `--budget` (4,000 MB) is spent. No file
may be over `--max-file` (95 MB, below GitHub's 100 MiB limit). Each file opens
with comment lines that name its source and citations. Node ids are the
source's own, as `to_pajek` writes them, so a node keeps its id across a
dataset's snapshots. They can start at 0 and skip numbers; a comment line gives
their range. Next to the files,
`index.json` lists every network with its file, size, counts and citations,
and says why each other one was left out. Run it again on the same directory to
update it: a network whose catalog entry is unchanged keeps its file without a
download, and one that fails keeps its previous file. With `--stop-after
MINUTES`, it starts no new download once that time is up and leaves the rest
for the next run, which continues where it stopped. The manifest's `complete`
stays false until a run gets through every network, and a run that changes
nothing leaves `index.json` as it was. When a source's host stops answering
(a timeout, a failed connection or a 5xx), the rest of that source waits for a
later run instead of three timeouts each. Its unbuilt networks keep their
estimated size reserved in the budget (`reserved` in the manifest), so building
them later deletes no file already published. The mirror is committed to the `mirror`
branch of
[mapequation/netgamut-networks](https://github.com/mapequation/netgamut-networks).
The explorer links each of its networks to the
[Network Navigator](https://mapequation.org/navigator/).

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
