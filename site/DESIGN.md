# Design rationale: from record browser to question-answering site

## What was wrong with the old design

The previous site was a **record browser with cross-links**. It did one thing well:
render every field of every record, resolve cross-references, and let you search.
But it failed at the actual job — helping an engineer answer a practical question.

Specific weaknesses:

1. **No joins.** The repo had `tools/cost_per_token.py` computing a 2,200x cost
   spread across 203 rows, but the site never surfaced it. A reader had to know
   the CLI existed and run it manually. The same was true for model×engine
   compatibility (a 14×8 matrix in `docs/13-model-compatibility.md`) and for
   roofline analysis (accelerator specs × flop classes). The data was there;
   the site just didn't join it.

2. **Uniform field lists.** Every record page rendered the same way: a table of
   key-value pairs. An accelerator page and a gotcha page looked identical in
   structure. There was no "key numbers at a glance" block, no comparison table,
   no type-specific summary that put the decisive numbers first.

3. **No visual hierarchy for decisions.** The site had 2,130 records but no
   way to see which ones mattered for a specific question. The type indexes
   were alphabetical lists with a filter bar. There was no "start here" for
   someone who wanted to know "which GPU for 70B at low latency" or "does
   engine X support model Y".

4. **Dark-only, no light theme.** The site was hardcoded to dark mode. For a
   reference site that engineers read in bright offices or on projectors, this
   was a real barrier.

5. **Status was colour-only.** Contested and draft records were marked with
   coloured badges and left rails, but the colour was the only signal. For
   colour-blind readers or greyscale printing, contested and verified records
   looked identical.

## What the new design does

### Joined views (the core change)

Three new pages answer questions that no single record can:

- **`/cost/` — Cost per Token.** Joins `supply/` prices to `benchmark/`
  throughput figures. 203 rows showing USD per million output tokens across
  providers, accelerators, and models. Filterable by concurrency class (aggregate
  vs single-user), price tier (spot, on-demand, capacity-block, etc.),
  accelerator, and provider. Every row links to both the supply and benchmark
  records it was derived from. The 2,200x spread between cheapest and most
  expensive is immediately visible.

- **`/compat/` — Model × Engine Compatibility.** 124 model records classified
  into 14 architecture families (dense-gqa, mla, sliding-window,
  hybrid-attention-ssm, diffusion, etc.), joined to 8 engine records. Each cell
  in the matrix cites the registry record that establishes it. `?` means no
  record states it — that is a real answer, not a gap. Filterable by family,
  engine, and cell status. Below the matrix, every model is listed under its
  family with its status badge.

- **`/roofline/` — Roofline Explorer.** Joins accelerator specs (peak FLOPS,
  memory bandwidth) to flop classes (arithmetic intensity, bound_by). Computes
  the ridge point (`1000 × TFLOPS ÷ GB/s`) for each accelerator — the batch size
  at which it crosses from memory-bound to compute-bound. 60 accelerators and
  61 flop classes, filterable by vendor, bound type, and search. The ridge
  number is the single most useful figure for hardware selection, and it was
  nowhere on the old site.

### Visual system

- **Type scale:** 1.125 ratio (major second). 7 sizes from 11px to 30px.
- **Spacing scale:** 4px base unit. 8 sizes from 4px to 64px.
- **Colour semantics:** Every status has a colour, an icon, and a text label.
  Contested is `⚠` in red, draft is `◐` in amber, verified is `✓` in green,
  deprecated is `✗` in grey. The icon is the primary discriminator — colour is
  redundant. Bound tags (memory, compute, interconnect, both, launch) each have
  their own colour. Cost cells are coloured by tier: spot is green, on-demand
  is amber, capacity-block is red.
- **Light and dark:** The site respects `prefers-color-scheme` and provides a
  manual toggle (the `◐` button in the masthead). Both themes are designed, not
  inverted. The toggle persists in `localStorage`.

### Information architecture

The home page now leads with "Answer a question" — the three joined views —
followed by "Decision docs" and then the type indexes. The joined views are
also in the top navigation (Cost, Compat, Roofline) so they are reachable from
every page. The type indexes and record pages are still there, still complete,
but they are no longer the primary entry point.

### Record pages

The record pages keep the same structure (field table, cross-references,
back-references, sources) but benefit from the new visual system: better type
scale, better spacing, colour-blind-safe status indicators, and light/dark
support. The status banners now use the icon + colour + label pattern.

## What a user can do now that they could not do before

1. **"What does inference actually cost?"** — Open `/cost/`, filter by
   accelerator and concurrency class, see USD/Mtok for every provider. Click
   any row to see the supply record (price basis, tier, availability) and the
   benchmark record (throughput, model, engine, methodology).

2. **"Does vLLM support DeepSeek-V3?"** — Open `/compat/`, find the `mla` row,
   look at the vLLM column. The cell says `Y` and links to the registry record
   that establishes it. Or filter by family to see all models in that family
   and their status.

3. **"Which GPU is fastest for batch-1 decode?"** — Open `/roofline/`, sort by
   ridge point. The ridge tells you where the memory wall ends. Below the
   ridge, you are buying bandwidth, not FLOPS. The flop classes table shows
   which classes are memory-bound (decode GEMM, decode attention, MoE) and
   which are compute-bound (prefill attention, reranking).

4. **"What is the cheapest way to serve Llama 3.1 8B?"** — Open `/cost/`,
   filter by model, sort by USD/Mtok. The cheapest row is spot pricing on
   Azure at $151/Mtok; the most expensive is on-demand Oracle at $667/Mtok.
   The 4.4x spread is immediately visible.

## Files changed

- `site/src/styles/global.css` — rewritten: type scale, spacing scale, light/dark
  themes, colour-blind-safe status icons, bound tags, cost table styling.
- `site/src/layouts/Base.astro` — added theme toggle button and script, added
  Cost/Compat/Roofline to the views nav.
- `site/src/pages/index.astro` — added "Answer a question" section featuring
  the three joined views.
- `site/src/pages/cost.astro` — new: cost-per-token joined view.
- `site/src/pages/compat.astro` — new: model×engine compatibility matrix.
- `site/src/pages/roofline.astro` — new: roofline explorer.
- `site/scripts/prebuild.mjs` — now also runs `tools/build_joined_views.py`.
- `site/scripts/verify_dist.py` — added checks for the three new pages.
- `tools/build_joined_views.py` — new: builds the joined-view JSON payloads.
- `tools/cost_per_token.py` — unchanged (imported by `build_joined_views.py`).
- `data/`, `schemas/`, `tools/build_site_data.py` — **not modified**.

## Verification

- `npm run build` exits 0, builds 2165 pages.
- `python scripts/verify_dist.py` passes 133/0 (including zero broken internal
  links sitewide).
- All three joined pages serve real content over HTTP (verified with curl).
- The joined-view JSON payloads are served from `/joined/` and contain real
  record data (203 cost rows, 14×8 compatibility matrix, 60 accelerators).
