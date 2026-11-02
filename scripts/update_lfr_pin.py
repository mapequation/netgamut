#!/usr/bin/env python3
"""Move the pinned LFR revision to upstream's current head.

The LFR generators are fetched, not vendored, from
https://github.com/andrealancichinetti/LFRbenchmarks. That repository publishes no
releases, so ``netgamut.generate`` pins a commit hash and its archive's SHA-256. This
script looks up what upstream's default branch points at today, downloads that
archive, checks it holds all six generators, and rewrites the two constants in
``src/netgamut/generate/_lfr/_sources.py``::

    python scripts/update_lfr_pin.py            # move the pin, if it has moved
    python scripts/update_lfr_pin.py --check    # report only; exit 1 if outdated
    python scripts/update_lfr_pin.py --commit <sha>   # pin something specific

Exits non-zero if the archive does not look like LFRbenchmarks, so an upstream
reorganisation fails the job instead of pinning something that cannot build.

Bumping the pin is a real change, not housekeeping: these generators are random, and
a different revision can produce a different network for the same seed. Run the LFR
tests afterwards (``pytest tests/test_lfr.py``) and say so in the commit message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINFILE = ROOT / "src" / "netgamut" / "generate" / "_lfr" / "_sources.py"

REPOSITORY = "andrealancichinetti/LFRbenchmarks"
API = f"https://api.github.com/repos/{REPOSITORY}"
TIMEOUT = 120

#: Every generator the wrapper drives, as ``directory/main`` inside the archive. If a
#: revision is missing one of these, it is not a revision we can pin.
REQUIRED = (
    "unweighted_undirected/Sources/benchm.cpp",
    "weighted_undirected/Sources/benchm.cpp",
    "unweighted_directed/Sources/benchm.cpp",
    "weighted_directed/Sources/benchm.cpp",
    "hierarchical/hier_ben_Sources/benchm.cpp",
    "MultiplexNMI-Benchmark/MultiplexBenchmark/Sources/benchm.cpp",
)


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "netgamut"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
        data: bytes = response.read()
    return data


def head_commit() -> tuple[str, str]:
    """The default branch's current head: ``(sha, committed date)``."""
    repository = json.loads(_fetch(API))
    branch = repository.get("default_branch", "main")
    commit = json.loads(_fetch(f"{API}/commits/{branch}"))
    date = commit.get("commit", {}).get("committer", {}).get("date", "unknown date")
    return str(commit["sha"]), str(date)


def archive(commit: str) -> tuple[str, list[str]]:
    """Download the archive for *commit*: its SHA-256 and the paths it contains."""
    payload = _fetch(f"https://github.com/{REPOSITORY}/archive/{commit}.tar.gz")
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "lfr.tar.gz"
        path.write_bytes(payload)
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
    # GitHub wraps everything in one <repo>-<sha> directory; drop it from the paths.
    return digest, ["/".join(name.split("/")[1:]) for name in names]


def current_pin() -> tuple[str, str]:
    text = PINFILE.read_text()
    commit = re.search(r'^COMMIT = "([0-9a-f]{40})"$', text, re.M)
    sha256 = re.search(r'^SHA256 = "([0-9a-f]{64})"$', text, re.M)
    if not commit or not sha256:
        sys.exit(f"Could not find COMMIT/SHA256 in {PINFILE.relative_to(ROOT)}")
    return commit.group(1), sha256.group(1)


def rewrite_pin(commit: str, sha256: str) -> None:
    text = PINFILE.read_text()
    text = re.sub(r'^COMMIT = "[0-9a-f]{40}"$', f'COMMIT = "{commit}"', text, flags=re.M)
    text = re.sub(r'^SHA256 = "[0-9a-f]{64}"$', f'SHA256 = "{sha256}"', text, flags=re.M)
    PINFILE.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, never write")
    parser.add_argument("--commit", help="pin this revision instead of upstream's head")
    args = parser.parse_args(argv)

    pinned, pinned_sha = current_pin()
    if args.commit:
        target, when = args.commit, "explicitly requested"
    else:
        target, when = head_commit()

    print(f"pinned    {pinned}")
    print(f"upstream  {target}  ({when})")

    if target == pinned:
        print("\nAlready current.")
        return 0

    digest, names = archive(target)
    missing = [path for path in REQUIRED if path not in names]
    if missing:
        print(f"\nRefusing to pin {target[:12]}: the archive is missing", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "Upstream may have reorganised; update REQUIRED and the variant table "
            "in _sources.py together.",
            file=sys.stderr,
        )
        return 2

    if args.check:
        print(f"\nOutdated. Run without --check to pin {target[:12]} (sha256 {digest[:12]}...).")
        return 1

    rewrite_pin(target, digest)
    print(f"\nPinned {target[:12]}, sha256 {digest[:12]}...")
    print(f"Rewrote {PINFILE.relative_to(ROOT)} (was {pinned[:12]}/{pinned_sha[:12]}...).")
    print(
        "The generators are random: run `pytest tests/test_lfr.py` and note in the "
        "commit that seeds may now give different networks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
