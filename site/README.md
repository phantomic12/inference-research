# inference-research — browsable site

A static Astro site over the record corpus in `../data/`. Every page is a file
in `dist/`; there is no server, no API and no database.

    cd site
    npm install
    npm run build          # data payload -> Astro build -> Pagefind index
    npm run preview        # or: npx serve dist

## Where the data comes from

Nothing in this site is hand-written. `npm run prebuild` runs
`../tools/build_site_data.py`, which walks `../data/**/*.json`, resolves every
bare-slug cross-reference, and writes a payload under `src/data/` plus some
runtime fetches under `public/`. That script is the only reader of `data/`, and
it never writes to `data/`, `schemas/`, or `docs/00-index.md`.

Because records are written concurrently, the generated payload is gitignored —
it is rebuilt on every `npm run build`, so it cannot drift from the records.

## Pages

| Route | Source |
|---|---|
| `/` | counts by type, status distribution, decision docs |
| `/<type>/` | one index per record type, sortable and filterable |
| `/r/<dir>/<id>/` | one page per record: every field, resolved sources, back-references |
| `/docs/<slug>/` | the synthesis docs, markdown rendered, `[[type/id]]` links resolved |
| `/compare/` | 2–4 records of one type side by side |
| `/graph/` | record neighbourhood, plus a whole-corpus force map |
| `/search/` | full-text over every record and doc |

Routes for record pages are generated from `src/data/records/**`, so a record
appears on the site exactly when its JSON exists.

## Cross-references

Records reference each other by bare slug (`accelerator_ids: ["nvidia-h100-sxm"]`).
`build_site_data.py` resolves those to qualified ids at build time and stores
them as outgoing refs plus a reverse index, so a detail page can show both
"this record points at" and "referenced by these N records", and every one is a
working link. Names come from `names.json` so a reference reads
"NVIDIA H100 SXM" rather than `nvidia-h100-sxm`.

References that name a record which does not exist are reported by
`npm run check:data` and rendered as plain text, never as broken links.

## Search

Pagefind indexes the rendered HTML after `astro build`, so it covers the notes
prose, every field and the doc pages, and it works offline from the filesystem.
Type and status filters are resolved client-side from `public/facets.json` plus
the record id in the URL, because the Windows Pagefind CLI in this toolchain
exposes no flag to build its own filter index.

## Status and confidence

`contested` and `draft` are never hidden. A record page gets a status-coloured
left rail, a coloured status badge, a banded confidence reading (`low · 0.60`)
and — for contested and draft — a banner above the content saying the numbers
are not settled. Type indexes sort unresolved records first by default, so a
contested record cannot be scrolled past unnoticed.

## Verification

    npm run verify        # dist/ HTML: real names, zero broken links sitewide
    npm run verify:search # Pagefind runtime against the built index
    npm run verify:browser# real Chrome: search renders, filters narrow
    npm run verify:all
    npm run shots         # PNGs of 15 key pages -> ../.site-shots/

`verify` checks all internal links resolve and that no unresolved `[[wikilink]]`
survives into a rendered doc.