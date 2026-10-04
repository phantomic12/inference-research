import { defineConfig } from 'astro/config';

// Static only: no server, no API routes, no adapter. Every page is a file in
// dist/, and the search index is built from those files by Pagefind.
export default defineConfig({
  output: 'static',
  site: 'https://inference-research.example',
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