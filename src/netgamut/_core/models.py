"""Shapes shared by every data source.

Each source (``netgamut.snap``, ``netgamut.netzschleuder``) has its own
``Dataset`` carrying whatever metadata that site publishes. What they agree on
lives here, so the cache, downloader and :class:`~netgamut.Network`
handle can work with any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["DataFile", "DatasetLike"]


@dataclass(frozen=True, slots=True)
class DataFile:
    """One downloadable artifact."""

    name: str
    url: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "description": self.description}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataFile:
        return cls(
            name=data["name"],
            url=data["url"],
            description=data.get("description", ""),
        )


@runtime_checkable
class DatasetLike(Protocol):
    """The minimum a source's dataset record must provide.

    Structural, not inherited: each source defines its own dataclass and
    satisfies this by having the right attributes.
    """

    @property
    def name(self) -> str: ...

    @property
    def directed(self) -> bool | None: ...

    @property
    def domains(self) -> tuple[str, ...]: ...
