"""Default names for generated networks, which become their file names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def run_name(prefix: str, **parts: Any) -> str:
    """``prefix-n1000-mu0.2-seed1``: the parameters that tell two runs apart.

    ``None`` and ``False`` are skipped, so optional settings appear only when
    given; anything that is not a plain number or string is left out too,
    since it would not make a readable file name.
    """
    tokens = [prefix]
    for key, value in parts.items():
        if value is None or value is False:
            continue
        if isinstance(value, float):
            tokens.append(f"{key}{value:g}")
        elif isinstance(value, (int, str)):
            tokens.append(f"{key}{value}")
    return "-".join(tokens)


def settings(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The parameters worth recording: everything set, nothing left at ``None``."""
    return {key: value for key, value in parameters.items() if value is not None}
