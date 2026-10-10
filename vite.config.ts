import { defaultClientConditions, defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { assistantPlugin } from "./server/assistant.ts";

export default defineConfig({
  plugins: [assistantPlugin(), svelte()],
  server: { host: "127.0.0.1", port: 8767, strictPort: true },
  preview: { host: "127.0.0.1", port: 8768, strictPort: true },
  // Relative base so a production build also works when served from a
  // subpath (a Hugging Face Space, GitHub Pages, a static folder).
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  resolve: {
    // Without this, onnxruntime-web resolves to its "bundle" build, which
    // base64-inlines a 27 MB .wasm into the JS. The app serves that wasm from
    // public/vendor/ort/ instead (see scripts/prepare-assets.mjs).
    //
    // Setting `conditions` REPLACES Vite's defaults rather than adding to
    // them, and dropping `browser` silently resolves packages to their server
    // builds (Svelte pulled in node:async_hooks that way) — so spread them.
    conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
  },
  // The MuJoCo runtime is loaded as a static file from public/vendor/, so
  // keep the dep optimizer from trying to pre-bundle it.
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
});
