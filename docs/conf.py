"""Sphinx configuration for the netgamut documentation.

Build with ``uv run --group docs sphinx-build -W docs docs/_build/html``. The
catalog explorer page needs its bundle first; see ``site/explorer/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
html_theme_options = {
    "source_repository": "https://github.com/mapequation/netgamut",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/mapequation/netgamut",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07'
                ".55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82"
                "-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52"
                ".28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08"
                "-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82"
                " 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29"
                ".25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0"
                '-4.42-3.58-8-8-8z"></path></svg>'
            ),
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


def setup(app: Sphinx) -> None:
    app.connect("html-page-context", _explorer_assets)
    app.connect("env-get-outdated", _rebuild_explorer)
