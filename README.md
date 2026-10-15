# inference-research

A structured, machine-readable knowledge base about LLM inference: accelerators,
the FLOP classes that dominate them, inference engines, quantization formats,
interconnect, measured benchmarks, known traps, models, papers, compilers,
metric exposures, and where to buy or rent the hardware.

Everything here is a JSON record with provenance. A new fact is one file in
`data/`, not a paragraph in a wiki page. Prose lives in `docs/` and cites
record ids.

## Layout

    data/
      accelerators/    one file per chip: specs, memory, FLOPS by precision, maturity
      flops/           one file per operation class: what it costs, what it is bound by
      engines/         one file per inference engine: hardware support, features, tradeoffs
      quantization/    one file per format: bits, kernels, quality, hardware support
      interconnect/    one file per link technology: bandwidth, fabric, topologies
      benchmarks/      one file per measurement: tok/s, TTFT, methodology
      gotchas/         one file per known trap: driver, kernel, framework, config
      models/          one file per model: architecture, parameter count, context
      papers/          one file per paper: venue, claims, the records it informs
      compilers/       one file per compiler / kernel DSL: targets, IR, maturity
      metric-exposures/ one file per metric an engine exports: name, unit, semantics
      supply/          one file per offer: purchase, used-market or per-GPU-hour price
      sources/         one file per source document, referenced by every other record
    docs/              synthesized prose, every claim cites record ids (start at docs/README.md)
    schemas/           JSON Schema per record type
    site/              Astro site built from data/ and docs/
    tools/             validate.py, new_record.py, query.py, index.py, cost_per_token.py,
                       build_site_data.py, test_tools.py, crawler.sh

## Record shape

Every record is a single JSON object with these common fields:

    id            stable slug, unique across the repo
    type          one of: accelerator, flop, engine, quantization, interconnect, benchmark,
                  gotcha, model, paper, compiler, metric_exposure, supply, source
    name          human readable
    status        verified | draft | contested | deprecated
    confidence    0.0 - 1.0
    updated       YYYY-MM-DD
    sources       [ "<source-id>", ... ]   -> resolves against data/sources/
    notes         free text, caveats, contradictions

`schemas/<type>.schema.json` defines the rest. Validate with:

    python tools/validate.py
    python tools/test_tools.py

`validate.py` also enforces two ratcheting budgets: dangling cross-record
references (`tools/ref_budget.json`) and unresolvable `docs/` citations
(`tools/docs_ref_budget.json`). Lower a budget in the same change that fixes a
reference; never raise one to go green.

## Ingesting new knowledge

1. Find the source. `data/sources/<slug>.json` records it: url, publisher,
   date, accessed date, kind (spec-sheet, paper, benchmark, forum, vendor-blog).
   Never cite a source that is not in `data/sources/`.
2. Create or extend a record. `python tools/new_record.py accelerator --name "..."`
   writes a skeleton with the right fields; fill it in, add sources.
3. Validate. `python tools/validate.py` must pass before commit.
4. Note conflicts. If a new source contradicts an existing record, do not
   overwrite it silently: set `status: contested`, add the contradicting source,
   and write the disagreement into `notes`. Resolution goes in
   `data/gotchas/` when it is not just a number disagreement.
5. Regenerate the index. `python tools/index.py`

Full contributor instructions in [AGENTS.md](AGENTS.md).
Field-by-field reference in [SCHEMA.md](SCHEMA.md).

## Rules

- One entity per file. No lists of hardware inside a single document.
- Numbers carry units and precision. `"memory_bandwidth_gbps": 936` is useless;
  `"memory_bandwidth_gbps": 936` with `"memory_bandwidth_basis": "GDDR6 28Gbps x 384-bit"`
  is auditable. Basis fields are mandatory for derived numbers.
- Vendor claims are marked `vendor_claim: true` unless independently measured.
- Dates in `updated` and `accessed` are the access date, not the publish date.
- Flat files, kebab-case ids, no directories deeper than `data/<type>/`.

## Current state

- `python tools/query.py stats` prints live record counts by type and status;
  `python tools/query.py gaps` lists coverage holes.
- [docs/README.md](docs/README.md) is the reading guide for the synthesized docs.
- [docs/00-index.md](docs/00-index.md) is the generated catalogue of every record
  (regenerate with `python tools/index.py`; CI fails if it is stale).
- [docs/26-known-defects.md](docs/26-known-defects.md) is the standing ledger of
  known, unfixed defects; check it before reporting one.
