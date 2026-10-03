# Ingesting knowledge into this repo

Read this before adding anything. The repo is only useful if every number can be
traced and contradicted later.

## The contract

A record is valid when:

1. It matches `schemas/<type>.schema.json` — `python tools/validate.py` passes.
2. Every claim has at least one `sources` entry, and every source entry exists in
   `data/sources/`.
3. Vendor-marketing numbers set `vendor_claim: true`.
4. Numbers derived from other numbers carry a `*_basis` field explaining the derivation.
5. The `updated` field is the date the record was last verified against its sources,
   not the date the file was written.

## Adding a fact

    # 1. record the source first
    python tools/new_record.py source --name "NVIDIA H100 datasheet" --url https://...

    # 2. skeleton for the entity, with sources already wired
    python tools/new_record.py accelerator --name "NVIDIA H100 SXM" --source nvidia-h100-sxm-datasheet

    # 3. fill fields, validate, index
    python tools/validate.py && python tools/index.py

## What counts as a new source

Anything that states a number or a capability: official spec sheets, architecture
whitepapers, arXiv papers, benchmark suites, GitHub release notes, reproducible
measurements, and vendor forums where a staff engineer gives a concrete answer.
A blog post that cites no primary source is still a source, but mark it
`kind: blog` and expect lower confidence.

Do not cite: SEO listicles, aggregator spec-comparison sites that copy Wikipedia,
AI-generated summaries, and second-hand paraphrase without attribution.

## Handling disagreement

Never resolve a numeric conflict by picking the number you like.

- Two sources, same quantity, different values: keep both. Set
  `status: contested`, put the values and their sources in `notes`, and add a
  `gotchas` record if the reason for the split is knowable (clock boost vs
  base, dense vs sparse, marketing vs measured).
- One source newer than the other: prefer the newer, but record the old value in
  `notes` with its source id so a reader can see what changed.
- Unresolvable: leave `status: contested` and move on. A wrong confident record
  is worse than an open question.

## Extending the schema

When a fact class has no home, add the record type before adding many instances.
New type = new `schemas/<x>.schema.json`, a new `data/<x>/` directory, a case in
`tools/validate.py`, and a row in SCHEMA.md. Adding fields to an existing type is
cheap and backward compatible; changing an existing field's meaning is not — bump
nothing, just fix the records and note the change in the type's `notes`.

## Writing docs/

`docs/` is the human view. Every quantitative claim in a doc cites record ids
inline, like `[[accelerators/nvidia-h100-sxm]]`. Docs never introduce numbers that
are not in a record. A doc that has no cited records belongs in a gotcha or nowhere.

## Adding a benchmark record

A benchmark is only useful with its methodology. Required:

- exact engine build, model, quantization, batch size, prompt and output lengths
- hardware: accelerator ids plus memory size and interconnect
- what was measured: decode tok/s, prefill tok/s, TTFT, or TPS — and whether the
  number is per-request or aggregate across concurrent requests
- whether the number came from the vendor, a third party, or your own run