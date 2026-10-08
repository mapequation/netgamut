"""The KONECT subpackage: page parsing, the edge-list rewrite, and overlap."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import networks
from networks import konect
from networks._core.domains import DOMAINS
from networks.konect import _catalog, _ent, _scrape
from networks.konect._models import Dataset, attribute_name, is_edge_list_header

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
    listed = make(
        members=("out.toy", "ent.toy.person.name"), edge_member="out.toy", node_attributes=("name",)
    )
    assert Dataset.from_dict(listed.to_dict()) == listed


# -- archive contents -------------------------------------------------------
def test_the_edge_list_is_known_by_its_header() -> None:
    assert is_edge_list_header("% sym unweighted\n")
    assert is_edge_list_header(b"% bip weighted\n")
    assert not is_edge_list_header("% 14 5 5\n")
    assert not is_edge_list_header(b"Suspect\n")


def test_attribute_names_drop_the_kind_and_network_token() -> None:
    assert attribute_name("ent.moreno_crime_crime.person.name") == "person.name"
    assert attribute_name("rel.moreno_crime_crime.person.role") == "person.role"


def test_unlisted_archives_answer_unknown() -> None:
    dataset = make()
    assert dataset.node_attributes is None
    assert dataset.edge_attributes is None
    assert dataset.node_metadata is None


def test_listed_archives_separate_node_and_edge_attributes() -> None:
    dataset = make(
        members=("meta.x", "rel.x.person.role", "ent.x.person.name", "out.x"),
        edge_member="out.x",
        node_attributes=("name",),
    )
    assert dataset.edge_attributes == ("person.role",)
    assert dataset.node_metadata is True
    # The older layout's edge list is itself a rel.* file.
    old = make(members=("rel.rating", "meta.m"), edge_member="rel.rating", node_attributes=())
    assert old.edge_attributes == ()
    assert old.node_metadata is False


def test_columns_cover_the_three_file_shapes() -> None:
    # Name order, whatever order the archive stores them in.
    one_entity = [("ent.x.person.sex", []), ("ent.x.person.name", [])]
    assert _ent.columns(one_entity) == ("name", "sex")
    two_entities = [("ent.x.genre.name", []), ("ent.x.artist.name", [])]
    assert _ent.columns(two_entities) == ("artist.name", "genre.name")
    table = [("ent.pets", ["%", "%", "% ent dat.name dat.species"])]
    assert _ent.columns(table) == ("name", "species")
    assert _ent.columns([("ent.erdos", [])]) == (_ent.LABEL,)


# -- the vendored catalog ---------------------------------------------------
def test_vendored_catalog_is_complete() -> None:
    items = konect.datasets()
    assert len(items) >= 250
    assert all(item.format in konect.FORMATS for item in items)
    assert all(item.edge_type in konect.EDGE_TYPES for item in items)
    assert all(item.domains for item in items)
    assert all(item.left_nodes for item in items if item.bipartite)
    assert set(konect.domains()) == set(DOMAINS)


def test_vendored_catalog_lists_archive_contents() -> None:
    items = konect.datasets()
    listed = [item for item in items if item.members is not None]
    assert len(listed) >= 250
    assert all(item.edge_member for item in listed)
    assert konect.info("movielens-100k_rating").edge_member == "rel.rating"
    assert konect.info("moreno_crime").node_attributes == ("name", "sex")
    assert "species" in (konect.info("petster-hamster").node_attributes or ())
    found = {item.name for item in konect.search(node_metadata=True)}
    assert {"moreno_crime", "pajek-erdos"} <= found
    # Unlisted archives answer neither way.
    unknown = {item.name for item in items if item.members is None}
    assert all(konect.info(name).node_metadata is None for name in unknown)
    assert not found & unknown


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
def _archive(path: Path, name: str, files: dict[str, str]) -> Path:
    """A KONECT-style archive: every file under ``<name>/``, plus a README."""
    with tarfile.open(path, "w:bz2") as archive:
        for member, text in {f"README.{name}": "readme", **files}.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}/{member}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def offline_konect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # name -> the edge list's text, or {member: text} for a full layout.
    bodies: dict[str, str | dict[str, str]] = {}

    def fake_download(url, target, **_kwargs):
        name = url.split("download.tsv.", 1)[1].removesuffix(".tar.bz2")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            body = bodies[name]
            _archive(target, name, {f"out.{name}": body} if isinstance(body, str) else body)
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
    assert net.columns.time == 3  # chess is temporal: u v w t
    assert list(net.rows())[0] == ("1", "2", "-1", "100")
    instance = net.to_infomap(silent=True)
    assert instance.num_links == 3


def test_the_older_layout_loads_from_its_rel_file(offline_konect) -> None:
    """movielens-100k_rating has no out.* file; its edge list is rel.rating."""
    dataset = konect.info("movielens-100k_rating")
    offline_konect[dataset.name] = {
        # A per-edge attribute file sorts first and has no header: not the edges.
        "rel.aaa.user.tag": "funny\nsad\n",
        "rel.rating": "% bip weighted\n% 2 943 1682\n% ent.user ent.movie\n1 1 5\n2 3 4\n",
    }
    net = konect.load(dataset.name)
    assert list(net.rows()) == [("1", "944", "5"), ("2", "946", "4")]


def test_an_archive_with_no_edge_list_says_so(offline_konect) -> None:
    offline_konect["moreno_bison"] = {"rel.x.person.role": "Suspect\n"}
    with pytest.raises(LookupError, match="no edge list"):
        konect.load("moreno_bison")


def test_node_table_reads_ent_files_onto_the_edge_list_ids(offline_konect) -> None:
    dataset = konect.info("moreno_crime")
    left, right = dataset.left_nodes, dataset.nodes - dataset.left_nodes
    names = "".join(f"p{i}\n" for i in range(1, left + 1))
    offline_konect[dataset.name] = {
        "meta.moreno_crime_crime": "entity-names: person,crime\n",
        "ent.moreno_crime_crime.person.name": names,
        "ent.moreno_crime_crime.crime.kind": "".join(f"c{i}\n" for i in range(1, right + 1)),
        "out.moreno_crime_crime": "% bip unweighted\n1 1\n",
    }
    net = konect.load(dataset.name)
    table = net.nodes()
    # Two entities, so columns carry the entity; the right side is shifted.
    assert set(table.columns) == {"person.name", "crime.kind"}
    assert table["person.name"][1] == "p1"
    assert table["crime.kind"][left + 1] == "c1"
    assert list(net.edges(cast=int)) == [(1, left + 1)]


def test_node_table_names_a_single_entity_by_attribute(offline_konect) -> None:
    offline_konect["moreno_bison"] = {
        "ent.moreno_bison_bison.bison.name": "Ann\nBo\n",
        "out.moreno_bison_bison": "% asym posweighted\n1 2 3\n",
    }
    table = konect.load("moreno_bison").nodes()
    assert dict(table["name"]) == {1: "Ann", 2: "Bo"}


def test_node_table_reads_a_profile_table_by_node_id(offline_konect) -> None:
    """The petster layout: quoted rows keyed by node id, a few rows missing."""
    offline_konect["moreno_bison"] = {
        "ent.moreno_bison": (
            '%\n% ent dat.name dat.home\n"1" "Dexter" "Vernon Hills, IL"\n"3" "Bo" ""\n'
        ),
        "out.moreno_bison_bison": "% asym posweighted\n1 2 3\n",
    }
    table = konect.load("moreno_bison").nodes()
    assert table.columns == ("name", "home")
    assert dict(table["name"]) == {1: "Dexter", 3: "Bo"}
    assert dict(table["home"]) == {1: "Vernon Hills, IL"}


def test_node_table_reads_a_label_list(offline_konect) -> None:
    offline_konect["moreno_bison"] = {
        "ent.moreno_bison": "ABBOTT, HARVEY L.\nACZEL, JANOS D.\n",
        "out.moreno_bison_bison": "% asym posweighted\n1 2 3\n",
    }
    table = konect.load("moreno_bison").nodes()
    assert dict(table[_ent.LABEL]) == {1: "ABBOTT, HARVEY L.", 2: "ACZEL, JANOS D."}


def test_node_table_refuses_archives_without_ent_files(offline_konect) -> None:
    offline_konect["moreno_bison"] = "% asym posweighted\n1 2 3\n"
    with pytest.raises(LookupError, match="no per-node attributes"):
        konect.load("moreno_bison").nodes()


# -- listing during the catalog build ---------------------------------------
def test_listing_streams_the_archive_and_finds_the_edge_list(monkeypatch, tmp_path) -> None:
    path = _archive(
        tmp_path / "m.tar.bz2",
        "m",
        {
            "rel.m.person.role": "Suspect\n",
            "ent.m": '%\n% ent dat.name dat.age\n"1" "Ann" "3"\n',
            "rel.rating": "% bip weighted\n1 1 5\n",
        },
    )
    monkeypatch.setattr(_catalog, "open_stream", lambda url, **_: path.open("rb"))
    members, edge, columns = _catalog.list_archive("http://example/m.tar.bz2")
    assert members == ("README.m", "rel.m.person.role", "ent.m", "rel.rating")
    assert edge == "rel.rating"
    assert columns == ("name", "age")


def _stamp(monkeypatch, size: int, modified: str) -> list[str]:
    listed: list[str] = []
    monkeypatch.setattr(_catalog, "head", lambda url: {"bytes": size, "last_modified": modified})

    def fake_list(url):
        listed.append(url)
        return ("out.toy", "ent.toy.a.b"), "out.toy", ("b",)

    monkeypatch.setattr(_catalog, "list_archive", fake_list)
    return listed


def test_an_unchanged_archive_reuses_its_listing(monkeypatch) -> None:
    listed = _stamp(monkeypatch, 10, "Mon")
    old = make(
        archive_bytes=10,
        archive_modified="Mon",
        members=("out.toy",),
        edge_member="out.toy",
        node_attributes=(),
    )
    previous = _catalog.Catalog(datasets=(old,))
    result = _catalog._with_listing(make(), previous, cap=100)
    assert (result.members, result.node_attributes) == (("out.toy",), ())
    assert listed == []


def test_a_changed_archive_is_listed_again(monkeypatch) -> None:
    listed = _stamp(monkeypatch, 11, "Tue")
    old = make(
        archive_bytes=10, archive_modified="Mon", members=("out.toy",), edge_member="out.toy"
    )
    result = _catalog._with_listing(make(), _catalog.Catalog(datasets=(old,)), cap=100)
    assert result.members == ("out.toy", "ent.toy.a.b")
    assert result.node_attributes == ("b",)
    assert (result.archive_bytes, result.archive_modified) == (11, "Tue")
    assert len(listed) == 1


def test_an_archive_over_the_cap_stays_unlisted(monkeypatch) -> None:
    listed = _stamp(monkeypatch, 500, "Mon")
    result = _catalog._with_listing(make(), None, cap=100)
    assert result.members is None and result.node_metadata is None
    assert result.archive_bytes == 500
    assert listed == []


@pytest.mark.network
def test_really_downloads_a_small_network() -> None:
    net = konect.load("moreno_bison")
    assert sum(1 for _ in net.edges()) == konect.info("moreno_bison").edges


@pytest.mark.network
def test_live_page_still_parses() -> None:
    from networks._core.download import fetch_text

    page = _scrape.parse_page(fetch_text(f"{konect.SITE_URL}/networks/moreno_bison/"))
    assert (page["format"], page["edge_type"], page["loops"]) == ("asym", "posweighted", False)
