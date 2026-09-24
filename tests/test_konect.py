"""The KONECT subpackage: page parsing, the edge-list rewrite, and overlap."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import networks
from networks import konect
from networks._core.domains import DOMAINS
from networks.konect import _scrape
from networks.konect._models import Dataset

INDEX_ROW = (
    '<TR><TD><CODE>MB</CODE><TD><A href="moreno_bison/">Bison</A><TD>'
    '<IMG class="icon" src="/ic/icon-availability-matrix.png" '
    'title="Dataset is available for download"> '
    '<IMG class="icon" src="/ic/icon-timestamps.png" '
    'title="Edges are annotated with timestamps">'
    '<TD align="right">26<TD align="right">1,314\n'
)

PAGE = """<H1>Bison</H1>
<P>
This directed network contains <I>dominance</I> between bison.
</P>
<H2>Metadata</H2>
<TABLE>
<TR><TD>Data source<TD>
<TD><A href="http://moreno.ss.uci.edu/data.html#bison">http://moreno.ss.uci.edu/data.html#bison</A>
<TR><TD><A href="/categories/">Category</A><TD class="padleft">
<A href="/categories/Animal/"><DIV class="category-circle" title="Animal network"></DIV></A>
<TD><A href="/categories/Animal/">Animal network</A>
<TR><TD>Network format<TD class="padleft"><IMG class="icon" src="/ic/icon-format-asym.png">
<TD>Unipartite, directed
<TR><TD>Edge type<TD class="padleft"><IMG class="icon" src="/ic/icon-weights-posweighted.png">
<TD>Positive weights, no multiple edges
<TR><TD>Loops<TD class="padleft"><IMG class="icon" src="/ic/icon-no.png">
<TD>Contains loops
</TABLE>
<H2>Statistics</H2>
<TABLE>
<TR><TD><A href="../../statistics/size/">Size</A>\t<TD class="padleft" align="right">n =<TD>26
<TR><TD><A href="../../statistics/fill/">Fill</A>\t<TD align="right">p =<TD>0.483&nbsp;077
</TABLE>
<H2>References</H2>
<table><tr valign="top"><td class="bibtexitem">
Dale&nbsp;F. Lott.
 Dominance relations.
 <em>Z. f&uuml;r Tierpsychol.</em>, 49(4), 1979.
[&nbsp;<a href="http://example.org">http</a>&nbsp;]
</td></tr></table>
"""


def make(name: str = "toy", **overrides) -> Dataset:
    return Dataset(name=name, **overrides)


# -- parsing ----------------------------------------------------------------
def test_index_rows_carry_availability_timestamps_and_size() -> None:
    (row,) = _scrape.parse_index(INDEX_ROW)
    assert row["name"] == "moreno_bison"
    assert row["available"] and row["temporal"]
    assert (row["nodes"], row["edges"]) == (26, 1314)


def test_page_codes_are_read_from_the_icons() -> None:
    page = _scrape.parse_page(PAGE)
    assert page["format"] == "asym"
    assert page["edge_type"] == "posweighted"
    assert page["category"] == "Animal"
    assert page["loops"] is True
    assert page["upstream_url"] == "http://moreno.ss.uci.edu/data.html#bison"
    assert page["description"].startswith("This directed network contains dominance")


def test_page_statistics_and_references_are_cleaned() -> None:
    page = _scrape.parse_page(PAGE)
    assert page["statistics"] == {"Size": "26", "Fill": "0.483077"}
    (citation,) = page["citations"]
    assert citation == "Dale F. Lott. Dominance relations. Z. für Tierpsychol., 49(4), 1979."


def test_a_bipartite_page_without_a_loops_row_has_no_loops() -> None:
    page = _scrape.parse_page(
        PAGE.replace("icon-format-asym", "icon-format-bip").replace(
            "<TR><TD>Loops", "<TR><TD>Other"
        )
    )
    assert page["loops"] is False


def test_wikimedia_series_are_excluded_except_english() -> None:
    assert _scrape.is_series("edit-frwiki")
    assert _scrape.is_series("wikipedia_link_de")
    assert _scrape.is_series("wiki_talk_sv")
    assert not _scrape.is_series("edit-enwiki")
    assert not _scrape.is_series("wiki_talk_en")
    assert not _scrape.is_series("elec")


# -- records ----------------------------------------------------------------
def test_format_and_edge_type_drive_the_properties() -> None:
    signed = make(format="asym", edge_type="multisigned")
    assert signed.directed is True
    assert signed.signed and signed.multigraph and signed.weighted
    assert signed.infomap_weights is False
    multi = make(format="bip", edge_type="positive")
    assert multi.directed is False and multi.bipartite
    assert multi.multigraph and not multi.weighted
    assert multi.infomap_weights is True


def test_categories_map_to_domains_with_overrides() -> None:
    assert make(category="Metabolic").domains == ("Biological",)
    assert make(category="Animal").domains == ("Social",)
    assert make("opsahl-powergrid", category="Infrastructure").domains == ("Technological",)
    assert make(category="Miscellaneous").domains == ()


def test_round_trip() -> None:
    dataset = make(format="bip", edge_type="unweighted", loops=False, left_nodes=5)
    assert Dataset.from_dict(dataset.to_dict()) == dataset


# -- the vendored catalog ---------------------------------------------------
def test_vendored_catalog_is_complete() -> None:
    items = konect.datasets()
    assert len(items) >= 250
    assert all(item.format in konect.FORMATS for item in items)
    assert all(item.edge_type in konect.EDGE_TYPES for item in items)
    assert all(item.domains for item in items)
    assert all(item.left_nodes for item in items if item.bipartite)
    assert set(konect.domains()) == set(DOMAINS)


def test_duplicates_rest_on_evidence_not_names() -> None:
    found = konect.duplicates()
    # Imported by Netzschleuder, under another name.
    assert found["moreno_lesmis"] == "netzschleuder/lesmis"
    # Names cross over: Netzschleuder's "epinions" and "twitter" are other
    # networks, while these are held under other names.
    assert found["epinions"] == "snap/soc-sign-epinions"
    assert found["twitter"] == "netzschleuder/twitter_social"
    assert found["munmun_twitter_social"] == "netzschleuder/twitter"
    # Same name as a SNAP or Netzschleuder entry, different size: kept.
    assert "ego-twitter" not in found
    assert "openflights" not in found
    # Netzschleuder's wikipedia_link links here but defaults to another language.
    assert "wikipedia_link_en" not in found


def test_cross_source_queries_use_the_other_copy() -> None:
    keys = [match.key for match in networks.search("dolphins")]
    assert keys == ["netzschleuder/dolphins"]
    assert networks.search("dolphins", source="konect")
    shadowed = sum(1 for item in konect.datasets() if item.duplicate_of)
    total = sum(len(module.datasets()) for module in networks._SOURCES.values())
    assert len(networks.search()) == total - shadowed


def test_loops_filter_spans_sources() -> None:
    with_loops = networks.search(loops=True)
    assert with_loops
    assert all(match.loops is True for match in with_loops)
    # A Netzschleuder network inherits KONECT's check through an exact twin.
    assert any(match.source == "netzschleuder" for match in with_loops)


# -- loading ----------------------------------------------------------------
def _archive(path: Path, name: str, body: str) -> Path:
    payload = body.encode()
    with tarfile.open(path, "w:bz2") as archive:
        for member, data in ((f"{name}/README.{name}", b"readme"), (f"{name}/out.{name}", payload)):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def offline_konect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    bodies: dict[str, str] = {}

    def fake_download(url, target, **_kwargs):
        name = url.split("download.tsv.", 1)[1].removesuffix(".tar.bz2")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            _archive(target, name, bodies[name])
        return target

    monkeypatch.setattr(konect, "_download_file", fake_download)
    return bodies


def test_bipartite_right_side_is_shifted_past_the_left(offline_konect) -> None:
    dataset = next(item for item in konect.datasets() if item.bipartite)
    offline_konect[dataset.name] = "% bip unweighted\n% 3 2 2\n1 1\n1 2\n2 2\n"
    net = konect.load(dataset.name)
    left = dataset.left_nodes
    assert list(net.edges(cast=int)) == [(1, 1 + left), (1, 2 + left), (2, 2 + left)]
    assert net.header == ("bip unweighted", "3 2 2")
    assert "README." + dataset.name in {p.name for p in net.extract().rglob("*")}


def test_unipartite_size_line_does_not_shift_ids(offline_konect) -> None:
    dataset = konect.info("moreno_bison")
    offline_konect[dataset.name] = "% asym posweighted\n% 2 3 3\n1 2 8\n2 3 1\n"
    net = konect.load(dataset.name)
    assert list(net.rows()) == [("1", "2", "8"), ("2", "3", "1")]
    assert net.weighted_links is True


def test_signed_weights_are_kept_but_not_given_to_infomap(offline_konect) -> None:
    pytest.importorskip("infomap")
    dataset = konect.info("chess")
    offline_konect[dataset.name] = "% asym multisigned\n1 2 -1 100\n2 1 1 200\n2 3 0 300\n"
    net = konect.load(dataset.name)
    assert net.weighted_links is False
    assert list(net.rows())[0] == ("1", "2", "-1", "100")
    instance = net.to_infomap(silent=True)
    assert instance.num_links == 3


@pytest.mark.network
def test_really_downloads_a_small_network() -> None:
    net = konect.load("moreno_bison")
    assert sum(1 for _ in net.edges()) == konect.info("moreno_bison").edges


@pytest.mark.network
def test_live_page_still_parses() -> None:
    from networks._core.download import fetch_text

    page = _scrape.parse_page(fetch_text(f"{konect.SITE_URL}/networks/moreno_bison/"))
    assert (page["format"], page["edge_type"], page["loops"]) == ("asym", "posweighted", False)
