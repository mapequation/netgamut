"""Compiling the LFR generators, once per machine.

Each generator is a single translation unit -- ``benchm.cpp`` ``#include``\\ s the rest
of its package -- so a build is one compiler invocation and takes about a second.  The
result lands in ``<cache>/lfr/bin/<key>/<variant>``, where *key* fingerprints the
pinned commit, the compiler and the flags: change any of them and the next call
rebuilds instead of running a stale binary.

The sources are C++ from 2009--2011.  They compile clean under clang and gcc with
warnings off; ``-w`` is not hiding anything we intend to fix, since the code is
upstream's, not ours.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ..._core.cache import cache_dir, file_lock
from ._sources import COMMIT, Variant, variant_dir

__all__ = ["CompilerNotFound", "binary", "compiler"]

#: Upstream's own flags, from the makefiles.
FLAGS = ("-O3", "-funroll-loops", "-w")

_CANDIDATES = ("c++", "g++", "clang++")


class CompilerNotFound(RuntimeError):
    """No C++ compiler was found on PATH."""


def compiler() -> list[str]:
    """Return the C++ compiler command, honouring ``CXX``.

    Raises:
        CompilerNotFound: Neither ``CXX`` nor any of ``c++``, ``g++``, ``clang++``
            resolves to an executable.
    """
    override = os.environ.get("CXX")
    if override:
        parts = shlex.split(override)
        if parts and shutil.which(parts[0]):
            return parts
        raise CompilerNotFound(f"CXX is set to {override!r}, which is not executable.")

    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return [found]

    raise CompilerNotFound(
        "The LFR generators are C++ and need a compiler to build. Install the Xcode "
        "command line tools (`xcode-select --install`) on macOS, or a C++ toolchain "
        "(`build-essential`) on Linux, or set CXX to one."
    )


def _version(command: list[str]) -> str:
    """The compiler's identity, so a toolchain upgrade invalidates the cache."""
    try:
        done = subprocess.run(
            [*command, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"
    return done.stdout.strip().splitlines()[0] if done.stdout.strip() else "unknown"


def _key(command: list[str]) -> str:
    fingerprint = "\0".join([COMMIT, *command, _version(command), *FLAGS])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def binary(variant: Variant, *, progress: bool | None = None) -> Path:
    """Return the compiled generator for *variant*, building it on first use.

    Raises:
        CompilerNotFound: No C++ compiler is available.
        RuntimeError: The compiler rejected the sources.
    """
    command = compiler()
    target = cache_dir() / "lfr" / "bin" / _key(command) / variant.key
    if target.is_file() and os.access(target, os.X_OK):
        return target

    source = variant_dir(variant, progress=progress)

    with file_lock(target, timeout=600.0):
        if target.is_file() and os.access(target, os.X_OK):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        try:
            done = subprocess.run(
                [*command, *FLAGS, "-o", str(partial), variant.main],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            if done.returncode != 0 or not partial.is_file():
                detail = (done.stderr or done.stdout or "").strip()
                raise RuntimeError(
                    f"Compiling the {variant.key} LFR generator failed "
                    f"({' '.join(command)} exited {done.returncode}):\n{detail[-2000:]}"
                )
            partial.chmod(0o755)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)

    return target
