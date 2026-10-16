# Reading guide

`docs/` is the synthesized layer over `data/`. Every quantitative claim cites a record
(`[[type/id]]`); `tools/index.py` regenerates the catalogue of every record (`docs/00-index.md`, generated in CI and not stored in the repo).
Numbers in a doc are a snapshot of the corpus at writing time — the cited record is the
source of truth.

## Start here

1. [01-hardware-selection.md](01-hardware-selection.md) — which accelerator for which workload.
2. [03-engine-selection.md](03-engine-selection.md) — which engine on that hardware.
3. [23-quantization-selection.md](23-quantization-selection.md) — which format, through which
   kernel (read [04-quant-selection.md](04-quant-selection.md) first for the W4A16 vs W4A4 regimes).
4. [09-model-memory-envelope.md](09-model-memory-envelope.md) — whether it fits.
5. [05-known-traps.md](05-known-traps.md) — what silently goes wrong.

## By question

| question | doc |
|---|---|
| what operations dominate, and what bounds them? | [02-flop-map.md](02-flop-map.md) |
| which kernels / compilers / DSLs generate the code? | [15-compilers-and-kernel-dsls.md](15-compilers-and-kernel-dsls.md) |
| does this model run on this engine? | [13-model-compatibility.md](13-model-compatibility.md) |
| is tensor parallel worth it over this fabric? | [11-interconnect-decision.md](11-interconnect-decision.md) |
| single GPU, multi-GPU or multi-node? | [17-deployment-shape.md](17-deployment-shape.md) |
| gateway, router, control plane, cold start? | [21-deployment-topology.md](21-deployment-topology.md) |
| where does serving time go? | [12-serving-workflows.md](12-serving-workflows.md) |
| what does a token cost? | [08-cost-per-token.md](08-cost-per-token.md) |
| rent or own, and what is actually rankable? | [28-value-rent-vs-own.md](28-value-rent-vs-own.md) |
| is a new GPU generation worth it? | [16-migration-upgrade.md](16-migration-upgrade.md) |
| what to run, alert on, and what breaks in production? | [19-production-operations.md](19-production-operations.md), [18-reliability.md](18-reliability.md) |
| which metrics to trust, and which engines export them? | [20-observability.md](20-observability.md), then [24-metrics-surface-selection.md](24-metrics-surface-selection.md) |
| what does the serving system own for safety? | [22-safety-and-security.md](22-safety-and-security.md) |
| which papers matter for a topic? | [27-paper-reading-map.md](27-paper-reading-map.md) |

## Method and integrity

- [06-glossary.md](06-glossary.md) — vocabulary.
- [07-benchmarking-methodology.md](07-benchmarking-methodology.md) — how a performance number
  must be measured; [10-testing-methodology.md](10-testing-methodology.md) — how correctness
  is tested.
- [25-citation-integrity.md](25-citation-integrity.md) — when a source does not contain the
  number it is cited for.
- [14-audit-findings.md](14-audit-findings.md) — the corpus audit and its fix order.
- [26-known-defects.md](26-known-defects.md) — the standing ledger of known, unfixed defects.
- [NOTES.md](NOTES.md) — open questions.

## Numbering

Docs are numbered in the order they were written, not by topic, and the numbers are part of
every link, so they are never renumbered. 15 was freed when the cost doc became 08 and is now
the compilers doc.
