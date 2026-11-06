"""The KONECT subpackage: page parsing, the edge-list rewrite, and overlap."""

from __future__ import annotations

import io
import random
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import netgamut
from netgamut import konect
from netgamut._core.domains import DOMAINS
from netgamut.konect import _catalog, _copies, _ent, _scrape
from netgamut.konect._models import Dataset, attribute_name, is_edge_list_header

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
    # Held by Netzschleuder under another name.
    assert found["moreno_lesmis"] == "netzschleuder/lesmis"
    # Names cross over: Netzschleuder's "twitter" is this one, while KONECT's
    # "twitter" is another network.
    assert found["munmun_twitter_social"] == "netzschleuder/twitter"
    assert found.get("twitter") != "netzschleuder/twitter"
    # Same name as a SNAP or Netzschleuder entry, different size: kept.
    assert "ego-twitter" not in found
    assert "openflights" not in found
    # Netzschleuder's wikipedia_link links here but defaults to another language.
    assert "wikipedia_link_en" not in found


# -- copies -------------------------------------------------------------------
# Two networks of 10 nodes and 14 edges: a bipartite one, five women and five
# events with the events numbered 6-10 as KONECT's loaded file has them, and
# one with triangles. Equal counts, unrelated networks.
BIPARTITE_10_14 = [
    (1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 9), (3, 7), (3, 8),
    (3, 10), (4, 8), (4, 9), (5, 9), (5, 10), (4, 10),
]  # fmt: skip
TRIANGLES_10_14 = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6),
    (4, 6), (6, 7), (1, 3), (0, 3), (8, 9), (7, 8),
]  # fmt: skip


def _text(edges: list[tuple[int, int]]) -> list[tuple[str, str]]:
    return [(str(u), str(v)) for u, v in edges]


def _relabel(edges: list[tuple[int, int]], seed: int = 1) -> list[tuple[str, str]]:
    nodes = sorted({node for edge in edges for node in edge})
    shuffled = nodes[:]
    random.Random(seed).shuffle(shuffled)
    new = dict(zip(nodes, shuffled, strict=True))
    rows = [(new[u], new[v]) for u, v in edges]
    random.Random(seed).shuffle(rows)
    return [(str(u + 100), str(v + 100)) for u, v in rows]


def test_same_size_networks_are_told_apart_by_their_edges() -> None:
    same, note = _copies.compare_edges(
        _text(BIPARTITE_10_14), _text(TRIANGLES_10_14), directed=False, simple=True, shift=1
    )
    assert not same
    assert note == "degree sequences differ"


def test_a_copy_matches_with_netzschleuder_ids_one_lower() -> None:
    zero_based = [(u - 1, v - 1) for u, v in BIPARTITE_10_14]
    assert _copies.compare_edges(
        _text(BIPARTITE_10_14), _text(zero_based), directed=False, simple=True, shift=1
    ) == (True, "same ids, offset by 1")
    # Undirected edges may be listed either way round, or both.
    both_ways = [(v, u) for u, v in BIPARTITE_10_14] + BIPARTITE_10_14
    assert _copies.compare_edges(
        _text(BIPARTITE_10_14), _text(both_ways), directed=False, simple=True
    ) == (True, "same ids")
    # Directed, a link's direction counts: a star pointing out is not one
    # pointing in.
    out_star = [(1, 2), (1, 3), (1, 4)]
    in_star = [(v, u) for u, v in out_star]
    assert _copies.compare_edges(_text(out_star), _text(in_star), directed=True, simple=True) == (
        False,
        "degree sequences differ",
    )
    assert _copies.compare_edges(_text(out_star), _text(in_star), directed=False, simple=True)[0]


def test_a_copy_matches_up_to_relabelling() -> None:
    same, note = _copies.compare_edges(
        _text(TRIANGLES_10_14),
        _relabel(TRIANGLES_10_14),
        directed=False,
        simple=True,
        isomorphism_cap=0,
    )
    assert same
    assert note.startswith("equal up to relabelling")


def test_a_multigraph_counts_its_repeated_edges() -> None:
    doubled_first, doubled_last = [(1, 2), (1, 2), (2, 3)], [(1, 2), (2, 3), (2, 3)]
    # As sets of edges the two are one network ...
    assert _copies.compare_edges(
        _text(doubled_first), _text(doubled_last), directed=False, simple=True
    ) == (True, "same ids")
    # ... as multigraphs, only up to swapping nodes 1 and 3.
    same, note = _copies.compare_edges(
        _text(doubled_first), _text(doubled_last), directed=False, simple=False, isomorphism_cap=0
    )
    assert same
    assert note.startswith("equal up to relabelling")
    triangle = [(1, 2), (2, 3), (3, 1)]
    assert not _copies.compare_edges(
        _text(doubled_first), _text(triangle), directed=False, simple=False
    )[0]


def test_small_pairs_need_an_exact_isomorphism() -> None:
    pytest.importorskip("networkx")
    cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    # Colour refinement alone takes these for one network.
    loose = _copies.compare_edges(
        _text(cycle), _text(triangles), directed=False, simple=True, isomorphism_cap=0
    )
    assert loose[0]
    assert _copies.compare_edges(_text(cycle), _text(triangles), directed=False, simple=True) == (
        False,
        "not isomorphic",
    )
    assert _copies.compare_edges(
        _text(TRIANGLES_10_14), _relabel(TRIANGLES_10_14), directed=False, simple=True
    ) == (True, "isomorphic after relabelling")


def _fake(name: str, nodes: int, edges: int, **flags: object) -> SimpleNamespace:
    defaults = {"directed": False, "bipartite": False, "subgraph_count": 1, "upstream_url": ""}
    return SimpleNamespace(name=name, nodes=nodes, edges=edges, **{**defaults, **flags})


def test_judge_needs_evidence_for_a_copy() -> None:
    konect_side = {
        "women": _fake("women", 10, 14, bipartite=True),
        "lesmis": _fake("lesmis", 77, 254),
        "karate": _fake("karate", 34, 78),
        "huge": _fake("huge", 10**6, 10**8),
    }
    netzschleuder_side = [
        _fake("attacks", 10, 14, bipartite=False),
        _fake("les_mis", 77, 254),
        _fake("imported", 70, 250, upstream_url="http://konect.cc/networks/karate/"),
        _fake("big", 10**6, 10**8),
    ]
    snap_side = [_fake("zachary", 34, 78, bipartite=None)]
    candidates = _copies.nominate(konect_side.values(), netzschleuder_side, snap_side)
    assert [(c.name, c.of, c.imported, c.same_size) for c in candidates] == [
        ("women", "netzschleuder/attacks", False, True),
        ("lesmis", "netzschleuder/les_mis", False, True),
        ("karate", "netzschleuder/imported", True, False),
        ("karate", "snap/zachary", False, True),
        ("huge", "netzschleuder/big", False, True),
    ]
    others = {f"netzschleuder/{item.name}": item for item in netzschleuder_side}
    others["snap/zachary"] = snap_side[0]
    compared: list[str] = []

    def check(candidate: _copies.Candidate) -> tuple[bool, str]:
        compared.append(candidate.of)
        return candidate.of == "netzschleuder/les_mis", "by hand"

    def verdicts(**options: object) -> dict[str, str]:
        found = _copies.judge(candidates, konect=konect_side, others=others, **options)
        return {pair.of: pair.verdict for pair in found}

    # Without comparing edge lists, only provenance makes a copy.
    assert verdicts() == {
        "netzschleuder/attacks": "different",  # bipartite against triangles
        "netzschleuder/les_mis": "unverified",
        "netzschleuder/imported": "imported",
        "snap/zachary": "unverified",
        "netzschleuder/big": "unverified",
    }
    found = verdicts(check=check, max_edges=1000)
    assert found["netzschleuder/les_mis"] == "same-edges"
    assert found["snap/zachary"] == "different"
    assert found["netzschleuder/big"] == "unverified"  # above the cap
    # Statistics rejected the first pair without a download, and the cap the last.
    assert compared == ["netzschleuder/les_mis", "snap/zachary"]

    # A comparison is reused; a recorded override wins.
    previous = [_copies.Pair("lesmis", "netzschleuder/les_mis", "same-edges", "by hand")]
    override = {("huge", "netzschleuder/big"): ("override", "checked once by hand")}
    compared.clear()
    found = verdicts(check=check, max_edges=1000, previous=previous, overrides=override)
    assert found["netzschleuder/les_mis"] == "same-edges"
    assert found["netzschleuder/big"] == "override"
    assert compared == ["snap/zachary"]

    def broken(candidate: _copies.Candidate) -> tuple[bool, str]:
        raise OSError("connection reset")

    assert verdicts(check=broken)["netzschleuder/les_mis"] == "unverified"


def test_duplicates_rest_on_evidence_not_size() -> None:
    found = konect.duplicates()
    # Same 10 nodes and 14 edges, but bipartite here and full of triangles there.
    assert "brunson_southern-women" not in found
    # Compared edge by edge.
    assert found["opsahl-southernwomen"] == "netzschleuder/davis_southern_women"
    pairs = {(pair.name, pair.of): pair for pair in konect.catalog().copies}
    assert pairs["opsahl-southernwomen", "netzschleuder/davis_southern_women"].verdict == (
        "same-edges"
    )
    assert pairs["brunson_southern-women", "netzschleuder/global_suicide_attacks"].verdict == (
        "different"
    )
    # Every import Netzschleuder records is a copy, compared or not.
    imported = [pair for pair in konect.catalog().copies if pair.verdict == "imported"]
    assert all(found[pair.name] == pair.of for pair in imported)
    # Only verdicts make copies.
    assert set(found) == {pair.name for pair in konect.catalog().copies if pair.is_copy}


def test_loops_carry_over_only_to_a_compared_copy() -> None:
    copies = konect.catalog().copies
    compared = {pair.of for pair in copies if pair.verdict == "same-edges"}
    assert compared
    for pair in copies:
        if pair.of not in compared:
            assert konect.loops_for(pair.of) is None, pair
    assert konect.loops_for("netzschleuder/global_suicide_attacks") is None
    assert konect.loops_for("netzschleuder/davis_southern_women") is False


def test_cross_source_queries_use_the_other_copy() -> None:
    keys = [match.key for match in netgamut.search("southern")]
    # Davis's women and events, compared edge by edge, are listed once ...
    assert "netzschleuder/davis_southern_women" in keys
    assert "konect/opsahl-southernwomen" not in keys
    # ... and Brunson's smaller network, which only shares a size with
    # another, is listed in its own right.
    assert "konect/brunson_southern-women" in keys
    assert netgamut.search("opsahl-southernwomen", source="konect")
    shadowed = sum(1 for item in konect.datasets() if item.duplicate_of)
    total = sum(len(module.datasets()) for module in netgamut._SOURCES.values())
    assert len(netgamut.search()) == total - shadowed


def test_loops_filter_spans_sources() -> None:
    with_loops = netgamut.search(loops=True)
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


def test_to_pajek_names_nodes_from_the_ent_files(offline_konect, tmp_path) -> None:
    offline_konect["moreno_bison"] = {
        "ent.moreno_bison_bison.bison.name": "Ann\nBo\n",
        "out.moreno_bison_bison": "% asym posweighted\n1 2 3\n",
    }
    text = konect.load("moreno_bison").to_pajek(tmp_path).read_text()
    assert '1 "Ann"\n2 "Bo"\n' in text


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
    from netgamut._core.download import fetch_text

    page = _scrape.parse_page(fetch_text(f"{konect.SITE_URL}/networks/moreno_bison/"))
    assert (page["format"], page["edge_type"], page["loops"]) == ("asym", "posweighted", False)
