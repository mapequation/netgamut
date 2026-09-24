"""Parsing KONECT's HTML: the network index and each network's page.

KONECT has no API, but its pages are generated from one template, so the parsing
keys on that template's fixed labels (``Network format``, ``Loops``) and icon
filenames (``icon-format-asym.png``) rather than on layout. The icons carry
KONECT's own codes, which is why they are read in preference to the text.
"""

from __future__ import annotations

import html
import re
from typing import Any

__all__ = ["is_series", "parse_index", "parse_page"]

# KONECT generates one network per language edition of each Wikimedia project:
# edits, hyperlinks, their growth, and user-talk exchanges. They are about 1,000
# of its ~1,300 entries and near-identical in kind, so they would swamp any
# search or domain count. Each series keeps its English edition, the one the
# literature uses; Wikipedia networks that are not series (wikispeedia, elec,
# wikipedia-growth) are unaffected.
_SERIES = re.compile(r"^(edit-|wikipedia_link_|link-dynamic-|wiki_talk_)")
_KEPT = re.compile(r"^(edit-enwiki|wikipedia_link_en|link-dynamic-enwiki|wiki_talk_en)$")

_ROW = re.compile(r"^<TR><TD><CODE>", re.MULTILINE)
_TAG = re.compile(r"<[^>]+>")


def is_series(name: str) -> bool:
    """Whether *name* belongs to a per-language Wikimedia series."""
    return bool(_SERIES.match(name)) and not _KEPT.match(name)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", fragment)).split())


def _cell(fragment: str) -> str:
    """The text of a table cell cut out at ``<TD``, which starts mid-tag."""
    return _text(fragment.split(">", 1)[-1])


def _number(value: str) -> int | None:
    digits = value.replace(",", "").replace(" ", "").replace(" ", "")
    return int(digits) if digits.isdigit() else None


def parse_index(page: str) -> list[dict[str, Any]]:
    """Every row of ``/networks/``: name, title, availability and size."""
    rows: list[dict[str, Any]] = []
    for line in page.splitlines():
        if not _ROW.match(line):
            continue
        link = re.search(r'<A href="([^"/]+)/">([^<]*)</A>', line)
        if link is None:
            continue
        titles = re.findall(r'title="([^"]+)"', line)
        sizes = re.findall(r'<TD align="right">([\d,]+)', line)
        rows.append(
            {
                "name": link.group(1),
                "title": html.unescape(link.group(2)).strip(),
                "available": "Dataset is available for download" in titles,
                "temporal": "Edges are annotated with timestamps" in titles,
                "nodes": _number(sizes[0]) if sizes else None,
                "edges": _number(sizes[1]) if len(sizes) > 1 else None,
            }
        )
    return rows


def _section(page: str, heading: str) -> str:
    parts = page.split(f"<H2>{heading}</H2>", 1)
    if len(parts) < 2:
        return ""
    return re.split(r"<H2>|</TABLE>", parts[1], maxsplit=1)[0]


def _metadata(page: str) -> dict[str, tuple[str, str]]:
    """Metadata label -> (icon filename, text) for each row of the table."""
    rows: dict[str, tuple[str, str]] = {}
    for row in _section(page, "Metadata").split("<TR>")[1:]:
        cells = row.split("<TD")
        if len(cells) < 3:
            continue
        label = _cell(cells[1])
        icon = re.search(r'src="/ic/([^"]+)"', row)
        value = _cell(cells[-1])
        rows[label] = (icon.group(1) if icon else "", value)
    return rows


def _statistics(page: str) -> dict[str, str]:
    """Statistic label -> value, as the page shows it."""
    stats: dict[str, str] = {}
    for row in _section(page, "Statistics").split("<TR>")[1:]:
        label = re.search(r"<A[^>]*>([^<]+)</A>", row)
        cells = row.split("<TD>")
        if label is None or len(cells) < 2:
            continue
        stats[html.unescape(label.group(1)).strip()] = _cell(cells[-1]).replace(" ", "")
    return stats


def _references(page: str) -> tuple[str, ...]:
    found = re.findall(r'<td class="bibtexitem">(.*?)</td>', page, re.S)
    cleaned = []
    for item in found:
        # Drop the "[ http ]" link KONECT appends, then the gaps <em> leaves.
        item = re.sub(r"\[(?:&nbsp;|\s)*<a[^>]*>[^<]*</a>(?:&nbsp;|\s)*\]", "", item)
        cleaned.append(re.sub(r"\s+([,.;:])", r"\1", _text(item)))
    return tuple(item for item in cleaned if item)


def parse_page(page: str) -> dict[str, Any]:
    """The fields of one ``/networks/<name>/`` page that the catalog keeps."""
    meta = _metadata(page)
    stats = _statistics(page)

    def icon_code(label: str, prefix: str) -> str:
        icon = meta.get(label, ("", ""))[0]
        return icon[len(prefix) : -len(".png")] if icon.startswith(prefix) else ""

    fmt = icon_code("Network format", "icon-format-")
    loops_text = meta.get("Loops", ("", ""))[1]
    if loops_text.startswith("Contains loops"):
        loops: bool | None = True
    elif loops_text.startswith("Does not contain loops") or fmt == "bip":
        loops = False
    else:
        loops = None

    description = ""
    intro = re.search(r"</H1>\s*<P>(.*?)</P>", page, re.S)
    if intro:
        description = _text(intro.group(1))

    return {
        "description": description,
        "category": meta.get("Category", ("", ""))[1].removesuffix(" network"),
        "format": fmt,
        "edge_type": icon_code("Edge type", "icon-weights-"),
        "loops": loops,
        "temporal": "Temporal data" in meta,
        "left_nodes": _number(stats.get("Left size", "")),
        "node_meaning": meta.get("Node meaning", ("", ""))[1],
        "edge_meaning": meta.get("Edge meaning", ("", ""))[1],
        "upstream_url": meta.get("Data source", ("", ""))[1],
        "statistics": stats,
        "citations": _references(page),
    }
