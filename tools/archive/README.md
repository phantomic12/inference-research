# `tools/archive/one-shot/` — retired historical data-injection scripts

These files are **not part of the live pipeline and are never run.**

Each is a one-shot script from the October-2026 "Phase A" / "Phase C" ingest
passes. They wrote `benchmark`, `source`, `supply`, `gotcha`, and
`accelerator` JSON records **directly into `data/`** with write-once
("SKIP (exists)") semantics, then were left in `tools/`. Their work is already
committed to `data/`; running them again is a no-op at best and, because most
hardcode a stale `ROOT = "/opt/data/projects/inference-research/repo"` path,
would write to the wrong place at worst.

## Why they are dead (audited before archiving)

- **Nothing imports them** — neither statically (`import phase_c_write`) nor
  dynamically (`importlib.import_module`, `__import__`, `runpy`, `exec`,
  `spec_from_file_location`). An AST + string-constant audit over every `.py` in
  the repo found zero references to any of the 17 module names.
- **No workflow invokes them** — `.github/workflows/*.yml` and
  `site/scripts/prebuild.mjs` run an explicit, fixed list of tools
  (`index.py`, `rank_models.py`, `build_joined_views.py`,
  `decision_ranking.py`, `intelligence_density.py`, `agentic_ranking.py`,
  `quant_retention.py`, `mine_derivatives.py`, `build_site_data.py`,
  `validate.py`, `verify_scores.py`); none of these scripts appear.
- **No test references them** — `tools/test_tools.py` loads only the live tools
  it tests (`rank_models.py`, `build_joined_views.py`, `read_figures.py`,
  `validate.py`, …) by explicit path.
- **Not a documented procedure** — `README.md`, `AGENTS.md`, `docs/`, and the
  workflow files contain no instruction to run any `phase_*.py`.
- **No reference list names them** — `tools/registry.py` maps record types, not
  scripts; `tools/validate.py` scans `data/` and `docs/` only, never `tools/`.

## What to use instead

New records go through `tools/intake.py` (or `tools/new_record.py`) and are
validated by `tools/validate.py` and `tools/test_tools.py`. Do not resurrect
these scripts; if a record is missing, re-derive it from the cited source
through the live intake path.

## Contents

| File | Original purpose |
| --- | --- |
| `phase_a_oct15_fill.py` | Phase A gap-fill ingest (Oct-15 tick) |
| `phase_c_write.py` | Phase C record writer |
| `phase_c2_write.py` | Phase C second-pass record writer |
| `phase_c3_write.py` | Phase C third-pass record writer |
| `phase_c4_write.py` | Phase C pass-4 record writer |
| `phase_c_p5_write.py` | Phase C pass-5 writer |
| `phase_c6_write.py` | Phase C pass-6 record writer |
| `phase_c8_write.py` | Phase C pass-8 record writer |
| `phase_c_mi300a_write.py` | Phase C MI300A pass |
| `phase_c_oct15_write.py` | Phase C Oct-15 tick |
| `phase_c_oct15_p3_write.py` | Phase C Oct-15 pass-3 |
| `phase_c_oct15_p8_write.py` | Phase C Oct-15 pass-8 |
| `phase_c_oct16_p2_write.py` | Phase C Oct-15/16 pass-2 |
| `phase_c_oct17_p7_write.py` | Phase C Oct-17 pass-7 (Database Mart) |
| `phase_c_oct17_p8_write.py` | Phase C Oct-17 pass-8 (Database Mart) |
| `phase_c_oct17_p25_write.py` | Phase C Oct-17 pass-25 (M5 Max context sweep) |
| `phase_c_oct18_p37_write.py` | Phase C Oct-18 pass-37 (Apple Silicon base/Pro) |

Archived verbatim; history preserved. Do not edit or re-run.