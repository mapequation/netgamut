import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// The web mirror's manifest, on the netgamut-networks repository's mirror branch.
const MIRROR =
  "https://raw.githubusercontent.com/mapequation/netgamut-networks/mirror/index.json";

const root = document.getElementById("netgamut-explorer");
if (root) {
  // Relative to the page, so the same bundle works under any Pages base path.
  const index = new URL(root.dataset.index ?? "catalog.json", document.baseURI)
    .href;
  createRoot(root).render(
    <StrictMode>
      <App indexUrl={index} mirrorUrl={root.dataset.mirror ?? MIRROR} />
    </StrictMode>,
  );
}
