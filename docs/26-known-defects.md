# Known defects: the scope-boundary ledger

**What this document is.** A single standing record of every defect a wave has
found and could *not* fix because it lived outside that wave's write scope.
Waves 4, 5 and 6 each reported problems "outside my scope", and the same four
items were named three times by three different agents before anyone wrote them
down. This document exists so that the fifth naming of an item is a lookup, not
a discovery.

**Status as of 2026-10-05**, against a corpus of **3873 records, 0 errors, 30
dangling references against a budget of 30** (`python tools/validate.py`).

**How to use it.** If you are about to report a defect, check here first. If it
is listed and still open, do not re-report it — either fix it if it is genuinely
in your scope, or leave it and say nothing. If it is listed and now fixed, that
is a bug in this ledger: retire the entry per the rules in Part 2 and say so in
your PR. If it is not listed, add it, with the same discipline.

---

## Part 1 — the ledger

### The tooling layer (`tools/`)

#### D1 — 18 `gaps` rows point at an `EXCLUDED` dict that does not contain them

| | |
|---|---|
| **What** | `python tools/cost_per_token.py gaps` prints `-> see EXCLUDED in tools/cost_per_token.py` for **18** benchmark rows. `EXCLUDED` in that same file holds **9** keys, and **none of the 18** is among them. Every one of the 18 is a row whose accelerator *is* priced, so it fell through to the generic fallback instead of naming its real blocker. |
| **Where** | `tools/cost_per_token.py` — the fallback `reason = EXCLUDED.get(bid, "see EXCLUDED in tools/cost_per_token.py")`. |
| **Wave** | 7 (`w7-scope-ledger`), while auditing D2. |
| **Why not then** | `tools/` is outside a docs-and-gotchas scope. |
| **Fixed now?** | **No.** And the message is actively misleading: it sends a reader to a 9-entry dict looking for 18 ids. The correct message names the real blocker, which for all 18 is "no declared GPU count — see `BENCH_JOIN`". |

The two real blockers, correctly stated: `[[benchmarks/mi325x-mlperf-v5-0-mangoboost-llama2-70b-offline]]`
and the other 17 are throughput rows on priced hardware that never declared how
many devices the aggregate covers.

#### D2 — AMD Instinct throughput rows are priced but still unpriceable

| | |
|---|---|
| **What** | Oracle publishes per-GPU-hour rates for MI300X ($6.00) and MI355X ($8.60) — recorded in [[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]], and `tools/cost_per_token.py`'s `PRICE_QUOTES` now carries both. The price side is closed **and it has reached the tool**: of **19** AMD MI3xx throughput rows, **11** are priced-but-unpriceable (10 `amd-instinct-mi355x`, 1 `amd-instinct-mi300x`) plus the 2 head-to-head rows that are correctly unpriceable. Each of the 11 needs a GPU count read from its own submission before an aggregate rate can be divided into a per-device rate. Only 2 AMD rows are joined: [[benchmarks/bench-mlperf-v6-1-amd-mi355x-llama2-70b-offline]] and [[benchmarks/llama-cpp-mi300x-deepseek-v3-671b-q4-decode-tok-s]]. |
| **Where** | `tools/cost_per_token.py` `BENCH_JOIN`; the gate is `check_evidence()`, which requires the GPU-count string to appear **verbatim** in the record's own `unit` or `methodology`. |
| **Wave** | 6 named it; 7 measured it exactly. **This entry was re-measured for the 2026-10-08 docs-citation pass and its numbers have drifted — see the status row.** |
| **Why not then** | `BENCH_JOIN` is a hand-maintained table in `tools/`. No docs agent can edit it. |
| **Fixed now?** | **No, and the measurement is now more precise than the one this entry was written from.** Measured 2026-10-08, of the **11** unpriceable rows: **6** already carry their GPU count verbatim in `unit`/`methodology` (`bench-mlperf-v6-0-dell-mangoboost-mi355x-powercap-1000w-offline`, `bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-offline` — its methodology reads *"Hardware: 1 node, 8x AMD Instinct MI355X 288GB HBM3e, Dell PowerEdge XE9785L"* — `bench-mlperf-v6-1-oracle-mi355x-llama3-1-8b-server`, `mi325x-mlperf-v5-0-mangoboost-llama2-70b-offline`, `mi355x-mlperf-v6-0-gpt-oss-120b-offline-tokens`, `mi355x-mlperf-v6-0-llama2-70b-87gpu-offline-tokens`) and need only a `BENCH_JOIN` entry; **5** state it only in the record `name` (e.g. `AMD 8xMI355X` in [[benchmarks/bench-mlperf-v6-1-amd-mi355x-llama2-70b-server]]), which `check_evidence()` cannot see, so those 5 additionally need a one-line methodology edit in `data/`; and the 2 head-to-head rows ([[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-output-throughput]] and its 70B twin) name two platforms at once and are correctly unpriceable as single rows. [[benchmarks/mi355x-mlperf-v6-0-llama2-70b-87gpu-offline-tokens]] carries **87** GPUs, not 8 — the one row that proves the divisor cannot be inferred from a sibling. **The 5/6 split is not the 7/4 this entry first published and not a stable proportion — it is whatever the records say on the day it is re-read, which is exactly why the re-measurement is recorded here rather than the old numbers being silently updated.** |

**The rule this entry exists to protect.** Guessing a divisor from a model name
or from a sibling row is the exact error the tool was written to prevent. A
price is not enough: an aggregate rate over an unknown number of devices is not
a per-device rate, and 87 ≠ 8 is what that looks like here.

#### D3 — `query.py refs` crashes with `KeyError: None`

| | |
|---|---|
| **What** | `python tools/query.py refs vllm` — the type argument is positional and optional, so omitting it reaches `DIRS[args.type]` with `args.type is None`. Traceback reproduced 2026-10-05 at `tools/query.py:139`. |
| **Where** | `tools/query.py`, `cmd_refs`. |
| **Wave** | 7 (`w7-scope-ledger`). |
| **Why not then** | `tools/` is out of scope. |
| **Fixed now?** | **No.** The fix is one guard: infer the type by scanning record types for the id, which is the same lookup `build_bare` already does, or fail with a message naming the 14 valid types. |

#### D4 — `tools/index.py` does **not** emit bare-id citations, and the 10 bare ids all resolve

| | |
|---|---|
| **What (as reported)** | "10 citations in `docs/00-index.md` cannot resolve because `tools/index.py` emits bare record ids." |
| **Where it lives** | Nowhere. **Disproven 2026-10-05, recorded so it stops being re-reported.** |
| **Why not then** | — |
|| **Fixed now?** | **Not applicable — the premise was wrong.** Three independent measurements: (a) `card()` in `tools/index.py` emits the id as `` `{rid}` `` in **backticks**, and the file contains no double-bracket citation sequence anywhere, so it emits no wikilinks at all; (b) the 10 bare wikilinks in `docs/00-index.md` are *copied through from record prose*, and every one resolves against a unique record — e.g. `sglang` → `engines/sglang`, `mlx-lm` → `engines/mlx-lm`, `flop-nvidia-h100-specs` → `sources/flop-nvidia-h100-specs`; (c) `python tools/build_site_data.py --check` reports `docs 26`, `data problems 0`, and resolves them via the bare-slug lookup. |

There **is** a real, smaller defect adjacent to this one, and it is worth
stating so it is not rediscovered either: an unresolvable doc wikilink degrades
to inline code rather than failing anything. `rewrite_docs()` returns
`` `label` `` for an unknown target, so a genuinely broken docs citation is
**invisible to every check in the repo** — `validate.py` never reads `docs/`.
A citation in `docs/` that rots will not fail CI. That is a tooling gap, not a
data defect, and it is unfixed.

### The data layer (`data/`)

#### D5 — the MPT benchmark trio carries a stale pointer inside its own methodology

| | |
|---|---|
| **What** | Four records state that no 40 GB A100 record exists in this repo, and that `accelerator_ids` is therefore deliberately empty. [[accelerators/nvidia-a100-40gb-sxm4]] now exists — verified, 40 GB, 1555 GB/s, form factor `sxm`. |
| **Where** | `methodology` in [[benchmarks/mpt7b-a100-bs1-ttft-ms]], [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]], [[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]], [[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]]. |
| **Wave** | Flagged by the `w6-benchmarks` accelerator-pointer audit, recorded inside the record's own `notes`. |
| **Why not then** | It is a four-record wording change in `data/`, and the four are a set: editing one alone makes the trio inconsistent. An agent holding only one of them could not do it safely. |
| **Fixed now?** | **No.** |

**The reasoning still holds; the justification does not.** An empty pointer is
still better than a wrong one, and the 40 GB SXM4 is still not the part Databricks
measured. But a reader who trusts the sentence will go looking for a missing
record and find this one instead — the failure mode is a confident reader
concluding the repo is missing a part. Fix the wording across all four; do not
populate `accelerator_ids`.

#### D6 — 17 `nvidia-a100-sxm` dangling references, and the other 13

| | |
|---|---|
| **What** | 30 dangling references total. **17** are `hardware_relevance: "nvidia-a100-sxm"` on paper records. The remaining **13**: `gfx950` (4), `gfx942` (3), `mlx-quantized-4bit` (3), `gfx1151` (1), `2bit` (1), `amd-ryzen-strix-point` (1). |
| **Where** | `hardware_relevance` on [[papers/w4p-megascale-infer-disaggregated-expert-parallelism]] and its peers; `affected_by_hardware` on the AMD FLOP records; `format_id` on the MLX and Radeon rows. |
| **Wave** | A100 disambiguation pass fixed 11 of 29; the rest survived into waves 6 and 7. |
| **Why not then** | **Cannot be fixed by a rename, and that is the finding.** `nvidia-a100-sxm` names an id that splits into [[accelerators/nvidia-a100-40gb-sxm4]] and [[accelerators/nvidia-a100-80gb-sxm4]]. The papers cite "A100" and do not state a capacity, so picking one variant would be inventing the hardware. The 4-row-remaining MDNA targets (`gfx942`, `gfx950`, `gfx1151`) are **architecture targets, not parts** — there is no record to point them at. The `format_id` values are bit-width and scheme names, not record ids. |
| **Fixed now?** | **No, and it should not be "fixed" by picking.** This is budgeted debt against `max_dangling_refs: 30`, which ratchets: 110 → 54 → 43 → 30. Lower it in the same change that reduces the count; never raise it to go green. |

#### D7 — two energy-schema members deliberately left unused

| | |
|---|---|
| **What** | `metric: joules_per_token` and `metric: memory_gb` have **zero** records each (measured 2026-10-05 across all of `data/benchmarks/`). |
| **Where** | The `metric` enum and `energy_basis` in `schemas/benchmark.schema.json`. |
| **Wave** | The wave-5 schema pass that added both members. |
| **Why not then** | Deliberate, and documented in the schema's own description rather than left silent. One benchmark record carries one `(metric, value)` pair, while every joules figure in this repo is measured **alongside** a throughput on the same run. Re-labelling a throughput record would destroy the throughput. The correct shape is a second record per run, which was not minted because that means asserting derived energy figures as new ids. `memory_gb` has no candidate at all: no benchmark here measures peak resident footprint as its primary quantity. |
| **Fixed now?** | **No, and the current state is correct.** 8 records carry `energy_basis` with the sample window and derivation written out, and `power_w`/`power_scope` hold the watts underneath. The related finding is [[gotchas/vendors-dont-publish-joules-per-token-estimation-error]]. Leave unused. |

#### D8 — MegaScale-Infer's preprint title differs from its published title

| | |
|---|---|
| **What** | [[papers/w4p-megascale-infer-disaggregated-expert-parallelism]] is titled "Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism" (arXiv 2504.02263), while the SIGCOMM record's title reads "Efficient Mixture-of-Experts Model Serving with Disaggregated Expert Parallelism" — the pre-print title. The identification therefore rests on the shared system name, the identical mechanism, and author-group overlap, **not** on title equality. |
| **Where** | That record's `notes`. |
| **Wave** | Wave 4 papers pass. |
| **Why not then** | At the time `sigcomm` was **not in the venue enum**, so the venue had to be recorded as `other` with the real one in `notes`. |
| **Fixed now?** | **The schema half is fixed; the title divergence is permanent.** `venue` is now `sigcomm` — the enum has 29 members and includes it. But the record's `notes` **still open with "VENUE IS 'other' BECAUSE SIGCOMM IS NOT IN THE SCHEMA ENUM"**, which is now false. That sentence is a stale pointer of the same shape as D5. The title divergence itself is a real-world fact and must stay documented. |

### Documentation layer (`docs/`)

#### D9 — `docs/08`'s AMD coverage claim: corrected, but the tooling gap it exposed is open

| | |
|---|---|
| **What** | `docs/08-cost-per-token.md` asserted "Every AMD MI300X/MI325X/MI355X benchmark is unpriceable … no supply record prices AMD Instinct per GPU-hour … This is the highest-value gap in the repo." That was **false**: the claim generalised from *no supply record in this repo* to *AMD is unpriceable*. The four records it cited as evidence are all correct — they are OEM and reseller channels, which do not price per-GPU-hour because that is not a unit an OEM system sale is quoted in. |
| **Where** | The doc is fixed. The claim's error is preserved as [[gotchas/w5s-a-negative-about-one-channel-is-not-a-negative-about-the-market]]. |
| **Wave** | Named by three separate agents across waves 4, 5 and 6 before this repo had anywhere to record it. |
| **Fixed now?** | **Yes — the doc.** Superseded by [[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]], which found the per-GPU-hour price by two independent methods. **No — the consequence.** The tooling gap it created is D2 and is still open. |

This entry stays because the *pattern* is the durable finding: a negative
asserted on one channel, generalised across a boundary it was never tested at.

#### D10 — verified negatives and open questions that waves recorded but cannot close

These are not defects. They are negative results and live questions that a later
wave must not re-derive from scratch. They are listed here because the brief that
produced this ledger asked for them, and because "we looked and found nothing"
is the most expensive thing in this repo to rediscover.

| Finding | Record |
|---|---|
| Four venue indexes (ACL, Crossref, DBLP, OpenReview) all miss a venue — **a negative across four corpora is not a negative** | [[gotchas/w6p-acl-or-crossref-index-misses-venue-a-negative-across-four-conference-corpora-is-not-negative]] |
| An ACL ARR submission is not a negative venue finding | [[gotchas/w6p-negative-acl-arr-submission-is-not-a-negative-venue-finding]] |
| Affiliation fields invented from plausibility — "Rice University" appears twice on one author | [[gotchas/w6p-affiliation-fields-invented-from-plausibility-rice-university-appears-twice]] |
| One paper, two slugs, sixteen times — the field that caught it | [[gotchas/w6p-one-paper-two-slugs-sixteen-times-and-the-field-that-exposed-it]] |
| A citation goes dead-looking three ways (deleted / moved / wrong repo); one probe cannot tell them apart | [[gotchas/w6s-citation-rot-three-instrument-false-deaths]] |
| A stale repo-path citation looks first-party | [[gotchas/w6s-stale-repo-path-citation-looks-first-party]] |
| No paper exists for llama.cpp and for TensorRT-LLM — do not invent one | [[gotchas/meta-llama-cpp-has-no-paper-do-not-invent-one]], [[gotchas/meta-tensorrt-llm-has-no-paper-do-not-invent-one]] |
| vLLM has no sub-8-bit KV cache on NVIDIA | [[gotchas/w4q-vllm-no-sub-8bit-kv-cache-on-nvidia]] |
| NVFP4 native support is sm100/sm120 at CUDA 12.8 only | [[gotchas/w4q-nvfp4-native-only-sm100-sm120-cuda128]] |
| vLLM has no MXFP4 native kernel on sm120 | [[gotchas/w4q-vllm-mxfp4-no-native-kernel-sm120]] |
| SGLang's utilisation metric is dead; TGI has no TTFT metric | [[gotchas/obs-sglang-utilization-metric-is-dead]], [[gotchas/obs-tgi-has-no-ttft-metric]] |
| A GitHub open-issue count includes pull requests | [[gotchas/ev-github-open-issues-count-includes-pull-requests]] |
| ipEX is archived, unsupported | [[gotchas/xpu-ipex-archived-no-support]] |
| A spec table's column mixing is the defect | [[gotchas/aa-gotcha-spec-table-column-mixing]] |
| A derived FLOP row with no `basis` field | [[gotchas/flops-row-had-no-basis-field]] |
| Vendors do not publish joules per token; estimation error is the whole story | [[gotchas/vendors-dont-publish-joules-per-token-estimation-error]] |
| Community quant quality floor is opinion, not measurement | [[gotchas/comm-community-quant-quality-floor-opinion]] |
| Community splits on buy-local vs rent | [[gotchas/sent-community-split-buy-local-versus-rent]] |
| Reproducibility metadata is missing, not zero | [[gotchas/meth-reproducibility-missing-methodology]] |

Open questions left standing by the accelerator audit, each recorded in its own
record rather than resolved: Huawei Ascend 910D's interconnect, Biren BR106's
position in the Chinese accelerator set, Tenstorrent Wormhole N150's node count
(108 or 120, unsettled), Cambricon MLU370-X8 and Iluvatar Bi-V100's
vendor-stated-but-unresolved interface figures.

---

## Part 2 — the discipline

The failure mode this discipline exists to prevent is recorded as
[[gotchas/w7l-a-defect-out-of-scope-gets-renamed-every-wave]].

A ledger nobody prunes becomes noise, and noise is worse than the absence of a
ledger: it trains readers to skim, and the item that mattered gets skimmed past.
This is exactly how the dangling-ref budget became meaningless before it was
ratcheted. So the rules are part of the document, not commentary on it.

### What belongs here

1. **A defect found by a wave and not fixed by that wave**, because the fix lives
   outside its write scope. The defining property is the *boundary*, not the
   severity.
2. **A verified negative or open question** that a wave established and could not
   close, and that would cost another wave real money to re-derive. These are
   D10. A negative is a result; it is not a defect, but it is not in any report
   either.
3. **A claim that was reported and then disproven** (D4). These stay. The point
   of the ledger is to stop an item being re-litigated, and that includes
   stopping a *wrong* item from being re-reported.

### What does NOT belong here

1. **A defect fixed in the same wave that found it.** That is what the wave was
   for. If it is fixed, it is in the commit message and the record itself; putting
   it here adds a row that can only ever be noise.
2. **A finding with no owner and no next action** — not because such findings do
   not exist, but because without an owner they cannot be retired and will
   outlive their own relevance. If a finding cannot be phrased as "X is outside
   scope Y", it does not go in this file.
3. **Anything restatable as a gotcha record.** When a failure mode is a
   *property of the domain* — a hardware trap, a measurement trap, a citation
   trap — it belongs in `data/gotchas/` and is cited here. This file is for
   defects in *this repo*. The distinction matters: a gotcha record teaches
   something about GPUs; a ledger entry teaches something about our process, and
   it goes stale when we fix it.
4. **Budgeted debt that is already visible and already ratcheting** (D6). That is
   `tools/ref_budget.json`'s job, and it is enforced by CI on every run. Listing
   it here would create a second place to forget to update.

### How an entry is retired

An entry is retired by **deleting the row**, in the same change that fixes the
defect, and only when the fix is **verified by running the check** — not by
reading that the change looks right. A retirement that is not backed by a command
output is a guess.

Three acceptable endings:

- **Fixed.** Delete the row. If the fix is partial, keep the row and rewrite the
  status line to say exactly what remains — partial states are where a ledger
  becomes a lie.
- **Not a defect after all.** Do not delete it. Move the row to **Part 1's D4**
  treatment: keep it, marked disproven, with the measurement that disproved it
  and the date. A disproven entry is worth more than a deleted one, because it
  is the only thing that stops the next wave re-reporting it.
- **Superseded by a gotcha record.** Delete the row and leave the citation in the
  D10 table, if it was a negative or an open question.

### The standing rules

- **Never raise a budget to go green.** `max_dangling_refs` moves down in the
  same change that reduces the count. Its history is in the file: 110 → 54 → 43 →
  30.
- **Cite nothing you have not resolved.** Every `dir/id` citation in this document
  was checked against the record set on 2026-10-05 — 37 unique, all resolving. An
  unresolvable citation is worse than no citation, because it looks sourced.
- **A wrong number in this ledger is a defect in this ledger.** Several numbers
  in D1, D2, D4 and D6 arrived in the brief that commissioned this document and
  did not survive measurement. They are recorded at their measured values, with
  the discrepancy visible, because a ledger that smooths over its own errors
  cannot be trusted on the ones it hides.
- **If you fix something on this list, retire it in the same PR.** A ledger with a
  growing fixed-row backlog is a ledger nobody reads.
