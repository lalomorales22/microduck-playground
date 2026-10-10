import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

// vitePreprocess is what lets <script lang="ts"> work in .svelte files, for
// both the dev/build pipeline and svelte-check.
export default { preprocess: vitePreprocess() };
