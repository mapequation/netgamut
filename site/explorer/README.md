# Catalog explorer

The explorer page of the netgamut documentation: React, Vite and Tailwind,
reading the JSON index that `netgamut export-index` writes. Every cross-source
decision -- the shared domains, which KONECT entries are copies, self-loops --
is made in Python and arrives in the index; the app only filters, sorts and
draws it.

```bash
npm ci
npm run index   # uv run netgamut export-index --output public/catalog.json
npm run dev     # http://localhost:5173, with stand-ins for the docs theme
```

`npm run build` writes `explorer.js`, `explorer.css` and the index into
`docs/_static/explorer/`, where `docs/conf.py` adds them to the explorer page.
Then build the docs from the repository root:

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
```

`.github/workflows/pages.yml` runs the same steps and deploys the result.

The explorer also reads the web mirror's manifest, which `netgamut
export-networks` writes and the `mirror` branch of
`mapequation/netgamut-networks` serves. Every network listed there gets a link
to the Network Navigator, and the "In Navigator" filter keeps or leaves out
those networks. The page's `data-mirror` attribute overrides the manifest's
URL. If the manifest fails to load, the links and the filter don't appear and
nothing else changes.

The page renders inside the furo theme, so the stylesheet has no reset and its
colours are furo's own variables; the categorical colours for the seven sources
are validated for colour-vision deficiency in light and dark mode.
