import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `vite` serves index.html for development. `vite build` writes one script and
// one stylesheet with fixed names into the Sphinx static directory, where
// docs/conf.py adds them to the explorer page; public/catalog.json, the
// exported index, is copied alongside.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../../docs/_static/explorer",
    emptyOutDir: true,
    rolldownOptions: {
      input: "src/main.tsx",
      output: {
        entryFileNames: "explorer.js",
        assetFileNames: "explorer[extname]",
      },
    },
  },
});
