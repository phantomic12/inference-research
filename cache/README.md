# Cache provenance

`cache/` is not general scratch space. The files under
`cache/agentic-backfill/` are the audit trail for the Phase-3 agentic benchmark
backfill: they show which models were in scope, what the read-only research
subagents returned, and which source each candidate row cited. The applied data
lives in `data/models/` and `data/sources/`; these files explain where it came
from.

## What is tracked

- `agentic-backfill/queue.json` is the original 116-model scope snapshot. It is
  generated once, but it is also the input that
  `tools/apply_agentic_backfill.py` reads to validate every returned slug, so it
  is tracked rather than treated as disposable scaffolding.
- `agentic-backfill/results/batch-00.json` through `batch-10.json`, plus
  `retry-both.json`, are the subagents' returned payloads. They contain 116 model
  blocks and 141 candidate benchmark rows, including empty benchmark lists and
  the negative-search notes that explain them. The task records the applied
  backfill as 136 rows after the applier's validation and deduplication rules.

These payloads are **immutable evidence**. Do not rewrite a returned note, score,
source, or empty answer to make it agree with a later correction. Preserve the
response exactly as received; record a correction in the data (and, when useful,
a new dated result artifact) instead.

The backfill was applied by:

- `af8ebd801bc8256010bd692e5e6b224259c74a9c` — 134 agentic rows for 49 models;
- `bc6a01c3d2010e9b1548580999eec1c51f0062b0` — the retry shard's two scored
  models.

Later canonicalization and data corrections are part of the same lineage:
`ee08cf1aac2786b9f28e6b9f131fc1206e4eb855`,
`e1f3b989c07c75b73fbc74e82ca6c6224806f525`, and
`709d9c0da71e58a75e6e4d8e10facdf97db9c73b`.

## What is ignored

`agentic-backfill/shard-*.json`, `task-*.txt`, and `retry-*.txt` are generated
fan-out scaffolding derived from the queue and the research prompt. They are not
what a subagent returned and are not needed to audit an applied row. They are
ignored by `.gitignore`, so `git clean -fdx` may remove them without deleting the
provenance. The tracked queue and result payloads survive that clean.

To replay the applier's checks without writing data, run:

```sh
python3 tools/apply_agentic_backfill.py cache/agentic-backfill/results
```

Do not add new research output here as scratch. Add a new, clearly named result
artifact only when it is intended to become permanent provenance, and update
this README when the applied lineage changes.
