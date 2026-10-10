/**
 * Resolve a path under the app's `public/` root to an absolute URL.
 *
 * The app is built with a relative base (`base: "./"`) so the bundle works from
 * any subpath — a Hugging Face Space, GitHub Pages, a plain folder. That makes
 * `import.meta.env.BASE_URL` the string `"./"`, which a dynamic `import()` or a
 * `fetch()` inside a bundled chunk resolves against *the chunk's* URL
 * (`/assets/index-….js`) rather than the page. Resolving against
 * `document.baseURI` instead pins everything to the document, which is where
 * `public/` actually lands.
 *
 * Workers have no `document`, and their own `self.location` is the worker
 * chunk's URL — the exact wrong base. So a worker must be told the page's base
 * explicitly via `setAssetBase()` before it touches any asset.
 */

let base: string | null = typeof document === "undefined" ? null : document.baseURI;

/** Give a worker the page's base URL. Send it in the worker's init message. */
export function setAssetBase(url: string): void {
  base = url;
}

/** The base a worker should be handed. */
export function assetBase(): string {
  if (base === null) throw new Error("asset base not set; call setAssetBase() first");
  return base;
}

export const assetUrl = (path: string): string => new URL(path, assetBase()).href;
