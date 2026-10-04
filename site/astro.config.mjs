import { defineConfig } from 'astro/config';

// Static only: no server, no API routes, no adapter. Every page is a file in
// dist/, and the search index is built from those files by Pagefind.
//
// `site` is the canonical origin. `base` is the subpath: this site is served
// from https://phantomic12.github.io/inference-research/, not from a domain
// root. Every absolute asset path, the sitemap, and Pagefind's basePath are
// derived from these two values — a wrong `base` 404s every asset.
//
// If the repo ever gets a custom domain, set `site` to that domain and remove
// `base` (or set it to '/'). Nothing else needs to change.
export default defineConfig({
  output: 'static',
  site: 'https://phantomic12.github.io/inference-research',
  base: '/inference-research/',
  trailingSlash: 'always',
  build: {
    // 2,049 record pages plus the search index; the default 500-page inline
    // threshold would inline every page's script tag into other pages.
    inlineStylesheets: 'never',
  },
  vite: {
    build: {
      // Per-record JSON is imported one page at a time; no chunk splitting needed.
      assetsInlineLimit: 4096,
    },
  },
});