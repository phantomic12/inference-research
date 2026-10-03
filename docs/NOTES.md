# Working notes

Scratch space for research in progress. Nothing here is authoritative; once a
fact is verified it moves into `data/` as a record.

## Conventions

- Record ids are kebab-case slugs, globally unique.
- Source ids are bare slugs; `sources` in any record must resolve in `data/sources/`.
- Today's date is used for `accessed` and `updated`.
- Vendor spec numbers are `vendor_claim: true` unless independently measured.
- Dense vs 2:4-sparse must be explicit on every FLOPS figure.
- Link bandwidth is recorded per-direction; if the vendor quoted bidirectional
  aggregate, the conversion goes in `bandwidth_basis`.

## Open questions

- Which parts of the consumer-GPU "AI TOPS" figures can be traced back to a
  documented dense fp8/fp16 number, and which are int4+sparse marketing only?
- Is there a defensible roofline-balance TFLOPS figure for the FLOP classes, or
  does it have to stay qualitative per part?
- How much of the ROCm-on-consumer-Radeon story has actually been fixed upstream?