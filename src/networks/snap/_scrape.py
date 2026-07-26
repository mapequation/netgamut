"""Parsers for the SNAP website.

This module is deliberately pure: every function takes HTML text and returns
data. Nothing here opens a socket, which keeps the parsing logic testable
against saved fixtures and keeps the runtime import free of network code.

Only the standard library's :mod:`html.parser` is used, so the installed
package needs no HTML dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

from ._models import Category, DataFile, Dataset

__all__ = [
    "INDEX_URL",
    "SNAP_BASE_URL",
    "DatasetDetails",
    "parse_dataset_page",
    "parse_index",
]

SNAP_BASE_URL = "https://snap.stanford.edu/data/"
INDEX_URL = urljoin(SNAP_BASE_URL, "index.html")

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})
_CELL_TAGS = frozenset({"td", "th"})
# Tags that imply a visual break; collapsing them to a space stops adjacent
# words from being glued together when we flatten a cell to text.
_SPACING_TAGS = frozenset({"br", "p", "div", "li", "tr", "td", "th"})

_WHITESPACE = re.compile(r"\s+")
# Anything that looks like a downloadable artifact rather than another page.
_DATA_SUFFIXES = (
    ".txt.gz",
    ".csv.gz",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".gz",
    ".bz2",
    ".txt",
    ".csv",
    ".json",
    ".tsv",
    ".xlsx",
)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


@dataclass
class _Cell:
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Table:
    heading: str = ""
    anchor: str = ""
    rows: list[list[_Cell]] = field(default_factory=list)

    @property
    def header(self) -> list[str]:
        return [cell.text for cell in self.rows[0]] if self.rows else []


@dataclass
class _ListItem:
    heading: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetDetails:
    """What a dataset's detail page adds to its row in the index."""

    statistics: dict[str, str] = field(default_factory=dict)
    files: tuple[DataFile, ...] = ()
    citations: tuple[str, ...] = ()


class _PageParser(HTMLParser):
    """Flattens a SNAP page into headings, tables and list items.

    SNAP's markup is hand-written HTML from the early 2000s: unclosed ``<tr>``
    tags, inconsistent quoting, stray ``<br>``. A tolerant single pass that
    accumulates text into whichever container is currently open handles it far
    more reliably than trying to build a strict tree.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_Table] = []
        self.items: list[_ListItem] = []
        #: Every ``(href, label)`` on the page, wherever it appears. Several
        #: SNAP pages list their downloads outside any table.
        self.links: list[tuple[str, str]] = []

        self._heading = ""
        self._pending_anchor = ""
        self._heading_anchor = ""

        self._in_heading = False
        self._heading_buffer = ""

        self._table_depth = 0
        self._table: _Table | None = None
        self._row: list[_Cell] | None = None
        self._cell: _Cell | None = None

        self._item: _ListItem | None = None

        # The <a href="..."> we are currently inside, and its accumulating
        # label text. Non-None only between the start and end tag.
        self._link_href: str | None = None
        self._link_buffer: str | None = None

    # -- text routing ----------------------------------------------------
    def _append(self, text: str) -> None:
        if self._link_buffer is not None:
            self._link_buffer += text
        if self._cell is not None:
            self._cell.text += text
        elif self._item is not None:
            self._item.text += text
        elif self._in_heading:
            self._heading_buffer += text

    # -- HTMLParser hooks ------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrib = {key: (value or "") for key, value in attrs}

        if tag in _SPACING_TAGS:
            self._append(" ")

        if tag == "a":
            name = attrib.get("name")
            if name and self._cell is None and self._item is None:
                # A bare <a name="x"> anchor immediately precedes the section
                # heading it labels.
                self._pending_anchor = name
            href = attrib.get("href")
            if href is not None:
                self._link_href = href
                self._link_buffer = ""
            return

        if tag in _HEADING_TAGS:
            self._in_heading = True
            self._heading_buffer = ""
            return

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table = _Table(heading=self._heading, anchor=self._heading_anchor)
            return

        if tag == "tr" and self._table_depth == 1:
            self._flush_row()
            self._row = []
            return

        if tag in _CELL_TAGS and self._table_depth == 1:
            if self._row is None:
                self._row = []
            self._flush_cell()
            self._cell = _Cell()
            return

        if tag == "li" and self._table_depth == 0:
            self._flush_item()
            self._item = _ListItem(heading=self._heading)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            if self._link_href is not None:
                label = _clean(self._link_buffer or "")
                self.links.append((self._link_href, label))
                target = self._cell or self._item
                if target is not None:
                    target.links.append((self._link_href, label))
                self._link_href = None
                self._link_buffer = None
            return

        if tag in _HEADING_TAGS:
            self._in_heading = False
            self._heading = _clean(self._heading_buffer)
            self._heading_anchor = self._pending_anchor
            self._pending_anchor = ""
            self._heading_buffer = ""
            return

        if tag in _CELL_TAGS and self._table_depth == 1:
            self._flush_cell()
            return

        if tag == "tr" and self._table_depth == 1:
            self._flush_row()
            return

        if tag == "table":
            if self._table_depth == 1:
                self._flush_row()
                if self._table is not None and self._table.rows:
                    self.tables.append(self._table)
                self._table = None
            self._table_depth = max(0, self._table_depth - 1)
            return

        if tag == "li" and self._table_depth == 0:
            self._flush_item()

    def handle_data(self, data: str) -> None:
        self._append(data)

    # -- flushing --------------------------------------------------------
    def _flush_cell(self) -> None:
        if self._cell is None:
            return
        self._cell.text = _clean(self._cell.text)
        if self._row is None:
            self._row = []
        self._row.append(self._cell)
        self._cell = None

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._row:
            if self._table is None:
                self._table = _Table(heading=self._heading, anchor=self._heading_anchor)
            self._table.rows.append(self._row)
        self._row = None

    def _flush_item(self) -> None:
        if self._item is None:
            return
        self._item.text = _clean(self._item.text)
        if self._item.text or self._item.links:
            self.items.append(self._item)
        self._item = None

    def close(self) -> None:
        super().close()
        self._flush_row()
        self._flush_item()
        if self._table is not None and self._table.rows:
            self.tables.append(self._table)
            self._table = None


def _parse_page(html: str) -> _PageParser:
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return parser


def _is_dataset_row(cells: list[_Cell]) -> bool:
    """A dataset row's first cell links to that dataset's detail page."""
    if not cells or not cells[0].links:
        return False
    href = cells[0].links[0][0]
    return href.endswith(".html") and not href.startswith(("http://", "https://", "#"))


def parse_index(html: str, base_url: str = SNAP_BASE_URL) -> tuple[list[Category], list[Dataset]]:
    """Parse the SNAP index page into its categories and dataset rows.

    Categories come from the page's own section headings, so the result mirrors
    exactly how SNAP groups its data. Sections that SNAP has commented out are
    invisible to the parser and therefore correctly skipped.

    Each dataset's ``columns`` preserves that category's table headers verbatim,
    because the tables genuinely differ: temporal categories report *Temporal
    Edges* and *Static Edges*, review categories report *Number of items*.
    """
    page = _parse_page(html)

    # The table of contents supplies the one-line gloss for each category.
    toc: dict[str, tuple[str, str]] = {}
    for item in page.items:
        for href, label in item.links:
            if not href.startswith("#") or len(href) < 2:
                continue
            anchor = href[1:]
            _, _, tail = item.text.partition(":")
            toc.setdefault(anchor, (label, _clean(tail)))
            break

    categories: list[Category] = []
    # Keyed by casefolded name: SNAP lists some datasets under several
    # sections, and those rows describe one dataset, not several.
    datasets: dict[str, Dataset] = {}
    seen_categories: set[str] = set()

    for table in page.tables:
        rows = [row for row in table.rows if _is_dataset_row(row)]
        if not rows or not table.heading:
            continue

        anchor = table.anchor or table.heading.lower().replace(" ", "-")
        header = table.header
        # A malformed header row would otherwise be silently treated as data.
        if not header or _is_dataset_row(table.rows[0]):
            header = []

        toc_name, toc_description = toc.get(anchor, ("", ""))
        names: list[str] = []

        for row in rows:
            href, label = row[0].links[0]
            name = label or row[0].text
            columns = {
                header[index] if index < len(header) else f"column{index}": cell.text
                for index, cell in enumerate(row)
                if index > 0
            }
            key = name.casefold()
            existing = datasets.get(key)
            if existing is None:
                datasets[key] = Dataset(
                    name=name,
                    categories=(anchor,),
                    category_names=(table.heading,),
                    description=row[-1].text if len(row) > 1 else "",
                    type=columns.get("Type") or None,
                    url=urljoin(base_url, href),
                    columns=columns,
                )
            else:
                # Same dataset seen in another section: record the extra
                # category and fold in any columns that section reports and
                # the first one did not.
                merged = dict(columns)
                merged.update(existing.columns)
                datasets[key] = Dataset(
                    name=existing.name,
                    categories=existing.categories + (anchor,),
                    category_names=existing.category_names + (table.heading,),
                    description=existing.description,
                    type=existing.type,
                    url=existing.url,
                    columns=merged,
                )
            names.append(name)

        if anchor in seen_categories:
            # SNAP occasionally splits one section across two tables.
            previous = next(item for item in categories if item.id == anchor)
            categories[categories.index(previous)] = Category(
                id=previous.id,
                name=previous.name,
                description=previous.description,
                toc_name=previous.toc_name,
                dataset_names=previous.dataset_names + tuple(names),
            )
            continue

        seen_categories.add(anchor)
        categories.append(
            Category(
                id=anchor,
                name=table.heading,
                description=toc_description,
                toc_name=toc_name or table.heading,
                dataset_names=tuple(names),
            )
        )

    return categories, list(datasets.values())


def _looks_like_data_file(href: str) -> bool:
    return href.lower().split("?")[0].endswith(_DATA_SUFFIXES)


def parse_dataset_page(html: str, page_url: str) -> DatasetDetails:
    """Parse a dataset detail page.

    Returns the *Dataset statistics* table, the downloadable files with their
    descriptions, and the citation lines under *Source (citation)*.
    """
    page = _parse_page(html)

    statistics: dict[str, str] = {}
    files: list[DataFile] = []
    seen_files: set[str] = set()

    for table in page.tables:
        flat = [cell.text for row in table.rows for cell in row]
        is_statistics = any(text.lower().startswith("dataset statistic") for text in flat)

        for row in table.rows:
            if is_statistics and len(row) == 2:
                key, value = row[0].text, row[1].text
                if key and value and not key.lower().startswith("dataset statistic"):
                    statistics[key] = value

            for cell_index, cell in enumerate(row):
                for href, label in cell.links:
                    if not _looks_like_data_file(href):
                        continue
                    name = label or href.rsplit("/", 1)[-1]
                    if name in seen_files:
                        continue
                    seen_files.add(name)
                    # The Files table pairs each link with a description in the
                    # following cell.
                    description = row[cell_index + 1].text if cell_index + 1 < len(row) else ""
                    files.append(
                        DataFile(
                            name=name,
                            url=urljoin(page_url, href),
                            description=description,
                        )
                    )

    # Not every dataset uses a Files table -- some list downloads in a bullet
    # list or a bare paragraph -- so sweep the remaining links too. These have
    # no adjacent description cell, hence the empty description.
    for href, label in page.links:
        if not _looks_like_data_file(href):
            continue
        name = label or href.rsplit("/", 1)[-1]
        if name in seen_files:
            continue
        seen_files.add(name)
        files.append(DataFile(name=name, url=urljoin(page_url, href), description=""))

    citations = tuple(
        item.text for item in page.items if item.heading.lower().startswith("source") and item.text
    )

    return DatasetDetails(statistics=statistics, files=tuple(files), citations=citations)
