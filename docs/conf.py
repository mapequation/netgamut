"""Sphinx configuration for the netgamut documentation.

Build with ``uv run --group docs sphinx-build -W docs docs/_build/html``. The
catalog explorer page needs its bundle first; see ``site/explorer/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from markupsafe import Markup
from sphinx.application import Sphinx
from sphinx.util import logging

import netgamut

project = "netgamut"
author = "The MapEquation team"
copyright = "the MapEquation team"
release = netgamut.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

exclude_patterns = ["_build"]

# The docstrings use Google-style sections (Args:, Returns:, Raises:).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

# The tutorial downloads data, so it is shown with the outputs it was saved with.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = f"netgamut {release}"
html_static_path = ["_static"]
html_css_files = ["netgamut.css"]
templates_path = ["_templates"]
html_theme_options = {
    "source_repository": "https://github.com/mapequation/netgamut",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/mapequation/netgamut",
            "html": (Path(__file__).parent / "_static" / "github-icon.svg").read_text(),
            "class": "",
        },
    ],
}

# Written by `npm run build` in site/explorer, together with the exported index.
_EXPLORER = Path(__file__).parent / "_static" / "explorer"
_logger = logging.getLogger(__name__)


def _explorer_assets(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Load the explorer bundle on its own page only."""
    if pagename != "explorer":
        return
    if not (_EXPLORER / "explorer.js").is_file():
        _logger.warning("explorer bundle missing; run `npm run build` in site/explorer")
        return
    app.add_js_file("explorer/explorer.js", type="module")
    app.add_css_file("explorer/explorer.css")


def _rebuild_explorer(app: Sphinx, env: Any, *changes: set[str]) -> list[str]:
    """Re-render the explorer page on every build: its bundle is not a Sphinx source."""
    return ["explorer"]


def _link_copyright(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Link the team named in the footer to mapequation.org.

    Furo escapes the copyright string, which leaves a ``Markup`` value as it is.
    """
    context["copyright"] = Markup('the <a href="https://www.mapequation.org">MapEquation team</a>')


def setup(app: Sphinx) -> None:
    app.connect("html-page-context", _explorer_assets)
    app.connect("html-page-context", _link_copyright)
    app.connect("env-get-outdated", _rebuild_explorer)
