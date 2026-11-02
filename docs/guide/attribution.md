# Attribution and license

## Attribution

The data belongs to the Stanford Network Analysis Project. This package only
indexes and fetches it. Please cite the datasets you use — `Dataset.citations`
carries the reference SNAP asks for — and cite the collection itself:

> Jure Leskovec and Andrej Krevl. *SNAP Datasets: Stanford Large Network
> Dataset Collection.* <https://snap.stanford.edu/data>, 2014.

KONECT asks you to cite the collection alongside each network's own reference
(both are in `Dataset.citations`):

> Jérôme Kunegis. *KONECT – The Koblenz Network Collection.* Proc. Int. Conf. on
> World Wide Web Companion, 1343–1350, 2013.

OGB and PyG are other people's collections too. Cite them if you use them:

> Weihua Hu et al. *Open Graph Benchmark: Datasets for Machine Learning on Graphs.*
> NeurIPS 2020. — and the paper behind each individual dataset.

> Matthias Fey and Jan Eric Lenssen. *Fast Graph Representation Learning with PyTorch
> Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

AHORN is a repository in its own right, and it asks you to cite two things
separately: the original source publication for a dataset — the BibTeX in
`Dataset.citations`, plus the Zenodo record when the exact revision matters —
and the repository itself, in `ahorn.CITATION`:

> Florian Frantzen and Michael T. Schaub. *A repository for discovery and reuse
> of higher-order network datasets.* arXiv:2608.03491 (2026).

Licences on AHORN records are the upstream source's, and grant nothing beyond
what that source already allowed.

`generate.lfr` runs Andrea Lancichinetti and Santo Fortunato's own programs,
downloaded on demand from
[their MIT-licensed repository](https://github.com/andrealancichinetti/LFRbenchmarks).
If you publish LFR results, cite them:

> Andrea Lancichinetti, Santo Fortunato and Filippo Radicchi. *Benchmark graphs
> for testing community detection algorithms.* Phys. Rev. E 78, 046110 (2008).

> Andrea Lancichinetti and Santo Fortunato. *Benchmarks for testing community
> detection algorithms on directed and weighted graphs with overlapping
> communities.* Phys. Rev. E 80, 016118 (2009).

## License

MIT (see [`LICENSE`](https://github.com/mapequation/netgamut/blob/main/LICENSE)). Applies to this package, not to the datasets it downloads.
