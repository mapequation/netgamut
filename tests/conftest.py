"""Shared fixtures.

The HTML fixtures are hand-written rather than copies of the real pages. They
are small enough to read at a glance and they pin the specific quirks SNAP's
markup actually has: cross-listed datasets, per-category column schemas,
sections and download links that have been commented out, and files listed
outside any table.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from networks._core import cache as _cache
from networks.ahorn import _catalog as _ahorn_catalog
from networks.netzschleuder import _catalog as _nz_catalog
from networks.paths import _catalog as _paths_catalog
from networks.snap import _catalog

INDEX_HTML = """
<html><body>
  <h1>Stanford Large Network Dataset Collection</h1>
  <ul>
    <li><a href="#socnets">Social networks</a> : online social networks</li>
    <li><a href="#temporal">Temporal networks</a> : networks where edges have timestamps</li>
    <li><a href="#amazon">Amazon networks</a> : co-purchasing products</li>
    <li><a href="#reviews">Online reviews</a> : review systems</li>
  </ul>

  <a name="socnets"></a><h3>Social networks</h3>
  <table>
    <tr><th>Name</th><th>Type</th><th>Nodes</th><th>Edges</th><th>Description</th></tr>
    <tr><td><a href="ego-Facebook.html">ego-Facebook</a></td><td>Undirected</td>
        <td>4,039</td><td>88,234</td><td>Social circles from Facebook</td></tr>
    <tr><td><a href="soc-Reddit.html">soc-Reddit</a></td><td>Directed</td>
        <td>55,863</td><td>858,490</td><td>Reddit hyperlinks</td></tr>
  </table>

  <a name="temporal"></a><h3>Temporal networks</h3>
  <table>
    <tr><th>Name</th><th>Type</th><th>Nodes</th><th>Temporal Edges</th>
        <th>Static Edges</th><th>Description</th></tr>
    <tr><td><a href="soc-Reddit.html">soc-Reddit</a></td><td>Directed</td>
        <td>55,863</td><td>858,490</td><td>571,927</td><td>Reddit hyperlinks</td></tr>
  </table>

  <a name="amazon"></a><h3>Product co-purchasing networks</h3>
  <table>
    <tr><th>Name</th><th>Type</th><th>Nodes</th><th>Edges</th><th>Description</th></tr>
    <tr><td><a href="amazon0302.html">amazon0302</a></td><td>Directed</td>
        <td>262,111</td><td>1,234,877</td><td>Amazon product network</td></tr>
  </table>

  <a name="reviews"></a><h3>Online reviews</h3>
  <table>
    <tr><th>Name</th><th>Type</th><th>Number of items</th><th>Description</th></tr>
    <tr><td><a href="web-Reviews.html">web-Reviews</a></td><td>Reviews</td>
        <td>1,586,614</td><td>Review text</td></tr>
  </table>

  <!--
  <a name="hidden"></a><h3>Retired category</h3>
  <table>
    <tr><th>Name</th><th>Type</th><th>Nodes</th><th>Edges</th><th>Description</th></tr>
    <tr><td><a href="gone.html">gone</a></td><td>Directed</td>
        <td>1</td><td>1</td><td>Should never be parsed</td></tr>
  </table>
  -->

  <h3>Network types</h3>
  <table>
    <tr><th>Type</th><th>Meaning</th></tr>
    <tr><td>Directed</td><td>Edges have a direction</td></tr>
  </table>
</body></html>
"""

DETAIL_HTML = """
<html><body>
  <h1>SNAP: ego-Facebook</h1>
  <h3>Dataset statistics</h3>
  <table>
    <tr><th colspan="2">Dataset statistics</th></tr>
    <tr><td>Nodes</td><td>4039</td></tr>
    <tr><td>Edges</td><td>88234</td></tr>
    <tr><td>Average clustering coefficient</td><td>0.6055</td></tr>
  </table>
  <h3>Source (citation)</h3>
  <ul>
    <li>J. McAuley and J. Leskovec. Learning to Discover Social Circles. NIPS, 2012.</li>
  </ul>
  <h3>Files</h3>
  <table>
    <tr><th>File</th><th>Description</th></tr>
    <tr><td><a href="facebook.tar.gz">facebook.tar.gz</a></td>
        <td>Facebook data (10 networks)</td></tr>
    <tr><td><a href="facebook_combined.txt.gz">facebook_combined.txt.gz</a></td>
        <td>Edges from all egonets combined</td></tr>
    <tr><td><a href="readme-Ego.txt">readme-Ego.txt</a></td><td>Description of files</td></tr>
    <tr><td>Not available <!--<a href="withdrawn.txt.gz">withdrawn.txt.gz</a>--></td>
        <td>Withdrawn by SNAP</td></tr>
  </table>
  <p>Bonus download outside any table:
     <a href="bigdata/extra-2009.txt.gz">extra-2009.txt.gz</a></p>
</body></html>
"""

GROUND_TRUTH_HTML = """
<html><body>
  <h1>SNAP: com-DBLP</h1>
  <h3>Files</h3>
  <table>
    <tr><th>File</th><th>Description</th></tr>
    <tr><td><a href="com-dblp.ungraph.txt.gz">com-dblp.ungraph.txt.gz</a></td>
        <td>Undirected network</td></tr>
    <tr><td><a href="com-dblp.all.cmty.txt.gz">com-dblp.all.cmty.txt.gz</a></td>
        <td>All communities</td></tr>
    <tr><td><a href="com-dblp.top5000.cmty.txt.gz">com-dblp.top5000.cmty.txt.gz</a></td>
        <td>Top 5000 communities</td></tr>
  </table>
</body></html>
"""


@pytest.fixture
def index_html() -> str:
    return INDEX_HTML


@pytest.fixture
def detail_html() -> str:
    return DETAIL_HTML


@pytest.fixture
def ground_truth_html() -> str:
    return GROUND_TRUTH_HTML


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the cache into tmp_path and reset memoised state.

    Without this, tests would read and write the developer's real cache.
    """
    root = tmp_path / "cache"
    monkeypatch.setenv(_cache.CACHE_ENV_VAR, str(root))
    monkeypatch.setattr(_cache, "_override", None, raising=False)
    monkeypatch.setattr(_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_nz_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_paths_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_ahorn_catalog, "_cached", None, raising=False)
    yield root
    monkeypatch.setattr(_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_nz_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_paths_catalog, "_cached", None, raising=False)
    monkeypatch.setattr(_ahorn_catalog, "_cached", None, raising=False)


@pytest.fixture
def gzipped_edges(tmp_path: Path) -> Path:
    """A small gzipped edge list shaped like a real SNAP file."""
    path = tmp_path / "toy.txt.gz"
    body = (
        "# Undirected graph: toy.txt\n"
        "# A toy network\n"
        "# Nodes: 4 Edges: 4\n"
        "# FromNodeId\tToNodeId\n"
        "1\t2\n"
        "2\t3\n"
        "3\t4\n"
        "4\t1\n"
    )
    path.write_bytes(gzip.compress(body.encode()))
    return path


@pytest.fixture
def gzipped_communities(tmp_path: Path) -> Path:
    path = tmp_path / "toy.cmty.txt.gz"
    path.write_bytes(gzip.compress(b"1\t2\t3\n3\t4\n"))
    return path


@pytest.fixture
def csv_edges(tmp_path: Path) -> Path:
    """A comma-separated edge list, as the Bitcoin trust networks use.

    Splitting these on whitespace yields one field per line, which would look
    like an empty network rather than an error.
    """
    path = tmp_path / "trust.csv.gz"
    path.write_bytes(gzip.compress(b"6,2,4,1289241911.72836\n6,5,2,1289241941.53000\n"))
    return path


@pytest.fixture
def gzipped_labels(tmp_path: Path) -> Path:
    """Node-label ground truth, as email-Eu-core uses."""
    path = tmp_path / "toy-department-labels.txt.gz"
    path.write_bytes(gzip.compress(b"1 0\n2 0\n3 1\n4 1\n"))
    return path


@pytest.fixture
def csv_edges_with_spaces(tmp_path: Path) -> Path:
    """CSV whose later fields contain spaces, as openflights' edges.csv does.

    A whitespace-first guess splits these into two fields and wrongly concludes
    the file is whitespace-separated.
    """
    path = tmp_path / "flights.csv"
    path.write_text("0,2,124.52,CG,1308,0,DH8 DHT,0\n0,3,88.1,CG,1309,0,AT7 ATR,0\n")
    return path


@pytest.fixture
def whitespace_edges_with_commas(tmp_path: Path) -> Path:
    """A whitespace-separated file whose labels contain commas.

    Splitting on comma gives an inconsistent field count, so whitespace wins.
    """
    path = tmp_path / "named.txt"
    path.write_text('"Smith, John"\t"Doe, Jane"\n"Wu, Li"\t"Smith, John"\n"a"\t"b"\n')
    return path


@pytest.fixture
def quoted_csv_edges(tmp_path: Path) -> Path:
    """CSV with a quoted field containing the delimiter."""
    path = tmp_path / "quoted.csv"
    path.write_text('1,2,"Air Lines, Inc."\n2,3,"Rail, Ltd."\n')
    return path
