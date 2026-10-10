"""Where the LFR generators come from, and which variant is which.

These are the Lancichinetti--Fortunato programs themselves, not a reimplementation:
the C++ that produced the benchmarks in *Benchmark graphs for testing community
detection algorithms* (2008) and *Directed, weighted and overlapping benchmark graphs*
(2009), MIT licensed and published by the first author.

They are fetched rather than vendored, and compiled on the machine that runs them.
That keeps one wheel valid everywhere -- no per-platform builds, no compiler at
install time, nothing to rebuild when a user's toolchain changes -- and it reuses the
download/lock/atomic-rename machinery the dataset cache already has.  The cost is a
compiler and one download, both on the first call only.

Upstream publishes no releases, so the pin is a commit:

* :func:`~networks.generate.lfr_source` prints what is pinned, where it came from and
  whether it is on this machine yet.
* ``python scripts/update_lfr_pin.py`` moves the pin to upstream's current head,
  rewriting :data:`COMMIT` and :data:`SHA256` below; ``--check`` only reports.
* ``NETWORKS_LFR_DIR`` points at an existing checkout, skipping the download -- the
  way to try a revision without touching the pin.

Upstream: https://github.com/andrealancichinetti/LFRbenchmarks
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from ..._core.cache import cache_dir, file_lock, read_sidecar
from ..._core.download import DownloadError, download

__all__ = [
    "COMMIT",
    "REPOSITORY",
    "SHA256",
    "URL",
    "VARIANTS",
    "Variant",
    "sources_dir",
    "variant_dir",
]

#: The pinned upstream revision. Bumping it is a deliberate change: the generators are
#: random, so a different commit can mean different networks for the same seed.
#: ``scripts/update_lfr_pin.py`` rewrites the two constants below.
REPOSITORY = "andrealancichinetti/LFRbenchmarks"
COMMIT = "ec9a860282d8fc9d52e9e08fea9ce93ddeb473af"
SHA256 = "ad13a354c2cfbb1de299dcb13caadd5801243b9041cd0e7e89852b9aeb384f76"
URL = f"https://github.com/{REPOSITORY}/archive/{COMMIT}.tar.gz"

#: Point at an unpacked upstream tree to skip the download.
SOURCE_ENV_VAR = "NETWORKS_LFR_DIR"

# The archive is a few hundred kilobytes; anything near a megabyte is not it.
_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Variant:
    """One upstream generator: where its sources live and what it writes.

    ``outputs`` is what a successful run must leave behind. The programs report
    infeasible parameters on stdout and still **exit 0**, writing nothing, so the
    files are the only reliable success signal.
    """

    key: str
    directory: str
    main: str
    directed: bool
    weighted: bool
    hierarchical: bool
    multiplex: bool
    outputs: tuple[str, ...]

    @property
    def source(self) -> str:
        """Path of the translation unit, relative to the upstream tree."""
        return f"{self.directory}/{self.main}"


_FLAT = ("network.dat", "community.dat")
_HIER = ("network.dat", "community_first_level.dat", "community_second_level.dat")
_MULTI = ("network_layer_0", "community_layer_0")

#: The six generators, keyed by the name the wrapper resolves arguments to.
VARIANTS: dict[str, Variant] = {
    v.key: v
    for v in (
        Variant(
            "undirected",
            "unweighted_undirected",
            "Sources/benchm.cpp",
            False,
            False,
            False,
            False,
            _FLAT,
        ),
        Variant(
            "weighted",
            "weighted_undirected",
            "Sources/benchm.cpp",
            False,
            True,
            False,
            False,
            _FLAT,
        ),
        Variant(
            "directed",
            "unweighted_directed",
            "Sources/benchm.cpp",
            True,
            False,
            False,
            False,
            _FLAT,
        ),
        Variant(
            "weighted_directed",
            "weighted_directed",
            "Sources/benchm.cpp",
            True,
            True,
            False,
            False,
            _FLAT,
        ),
        Variant(
            "hierarchical",
            "hierarchical",
            "hier_ben_Sources/benchm.cpp",
            False,
            False,
            True,
            False,
            _HIER,
        ),
        Variant(
            "multiplex",
            "MultiplexNMI-Benchmark/MultiplexBenchmark",
            "Sources/benchm.cpp",
            False,
            False,
            False,
            True,
            _MULTI,
        ),
    )
}


def _lfr_dir() -> Path:
    return cache_dir() / "lfr"


def _checksum(target: Path) -> str:
    """The archive's SHA-256, from the download sidecar or by re-reading it."""
    recorded = read_sidecar(target).get("sha256")
    if isinstance(recorded, str):
        return recorded
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sources_dir(*, progress: bool | None = None) -> Path:
    """Return the unpacked upstream tree, downloading it once if necessary.

    Honours ``NETWORKS_LFR_DIR``. Otherwise fetches :data:`URL` into
    ``<cache>/lfr/``, checks it against :data:`SHA256`, and unpacks it to
    ``<cache>/lfr/src/<commit>/``. Concurrent callers serialise on a lock and the
    result is published with an atomic rename, so a killed process cannot leave a
    half-unpacked tree behind.

    Raises:
        DownloadError: The archive did not match the pinned checksum.
    """
    override = os.environ.get(SOURCE_ENV_VAR)
    if override:
        root = Path(override).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"{SOURCE_ENV_VAR} points at {root}, which is not a directory")
        return root

    unpacked = _lfr_dir() / "src" / COMMIT
    if unpacked.is_dir():
        return unpacked

    archive = _lfr_dir() / f"LFRbenchmarks-{COMMIT[:12]}.tar.gz"
    download(URL, archive, progress=progress, max_bytes=_MAX_BYTES)

    digest = _checksum(archive)
    if digest != SHA256:
        archive.unlink(missing_ok=True)
        raise DownloadError(
            f"{URL} hashed to {digest}, not the pinned {SHA256}. "
            "The download was corrupted, or upstream rewrote the commit."
        )

    with file_lock(unpacked, timeout=600.0):
        if unpacked.is_dir():
            return unpacked
        staging = unpacked.with_name(f".{unpacked.name}.{os.getpid()}.part")
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.mkdir(parents=True)
            with tarfile.open(archive, "r:gz") as tar:
                # data_filter refuses absolute paths, links out of the tree and device
                # nodes. Present since 3.11.4; older 3.11 point releases just extract.
                extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tar.extractall(staging, **extra)  # type: ignore[arg-type]
            (top,) = list(staging.iterdir())  # GitHub wraps everything in one directory
            os.replace(top, unpacked)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    return unpacked


def variant_dir(variant: Variant, *, progress: bool | None = None) -> Path:
    """Return the directory holding *variant*'s sources, fetching them if needed."""
    root = sources_dir(progress=progress) / variant.directory
    if not (root / variant.main).is_file():
        raise FileNotFoundError(
            f"{variant.key}: {root / variant.main} is missing. "
            f"If {SOURCE_ENV_VAR} is set, it must point at an LFRbenchmarks checkout."
        )
    return root
