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

#### D1 — ~~18 `gaps` rows point at an `EXCLUDED` dict that does not contain them~~ **FIXED 2026-10-08**

| | |
|---|---|
| **Was** | `python tools/cost_per_token.py gaps` printed `-> see EXCLUDED in tools/cost_per_token.py` for **18** benchmark rows. `EXCLUDED` in that same file held **9** keys, and **none of the 18** was among them. Every one of the 18 was a row whose accelerator *is* priced, so it fell through to the generic fallback instead of naming its real blocker. |
| **Where** | `tools/cost_per_token.py` — `cmd_gaps`. |
| **Wave** | 7 (`w7-scope-ledger`) while auditing D2; fixed by `w8-costtool`. |
| **Fixed now?** | **Yes.** The blanket fallback is gone; `cmd_gaps` now splits every gap into one of four categories and names the real blocker. Zero rows print "see EXCLUDED". A `GPU_COUNT_HINT` regex picks the wording only — it is never used as a divisor. The 18 broke down as: 6 rows state a GPU count but lack a `BENCH_JOIN` entry; 10 declare no GPU count at all; 1 names several platforms at once; 2 carried explicit `EXCLUDED` reasons that now outrank the pattern match. |

#### D2 — 13 AMD Instinct throughput rows are priced but still unpriceable **MOSTLY FIXED 2026-10-08**

| | |
|---|---|
| **What** | Oracle publishes per-GPU-hour rates for MI300X ($6.00) and MI355X ($8.60) — recorded in [[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]]. The price side is closed. 13 throughput rows on those accelerators still did not price, because each needed a GPU count read from its own submission before an aggregate rate could be divided into a per-device rate. |
| **Where** | `tools/cost_per_token.py` `BENCH_JOIN`; the gate is `check_evidence()`, which requires the GPU-count string to appear **verbatim** in the record's own `unit` or `methodology`. |
| **Wave** | 6 named it; 7 measured it exactly; `w8-costtool` fixed 11 of 13. |
| **Fixed now?** | **Yes for 11 of 13.** `w8-costtool` re-ran the measurement and the ledger's 7/4/2 split was **wrong in both directions**: 5 rows already carried their count verbatim (the ledger filed them as needing data edits), and one it filed as name-only (`bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-offline`) did carry its count, so the initially-made edit was reverted. 4 rows genuinely needed a one-line methodology insertion quoting the MLPerf system id, and the `BENCH_JOIN` entry quotes that same string back so `check_evidence()` proves it. AMD cost rows in the tool's output went **2 → 12**. |
| **Still unpriceable, by design** | (a) the 2 head-to-head rows naming two platforms at once — a rate divided by a device count means nothing when the devices differ; (b) [[benchmarks/mi325x-mlperf-v5-0-mangoboost-llama2-70b-offline]], whose methodology says 4 nodes but never states GPUs per node. Its `8xMI325X` string is a **corroborating context sentence about a different submission** and must not be borrowed as a divisor. Now in `EXCLUDED` with that reason. |

**The rule this entry exists to protect, unchanged:** guessing a divisor from
a model name or from a sibling row is the exact error the tool was written to
prevent. `[[benchmarks/mi355x-mlperf-v6-0-llama2-70b-87gpu-offline-tokens]]`
carries **87**, not 8 — and it now prices correctly at 87 GPUs ($717.97/Mtok,
11978.28 tok/s per GPU), which is the proof that the gate works.

**Extends past AMD.** The same fallback was also serving 7 NVIDIA MLPerf rows
(Cisco H200, CoreWeave GB300, NVIDIA B300, GB300-NVL72 slices). Their GPU counts
are stated but absent from `BENCH_JOIN`; each is a one-line entry away and its
verbatim evidence string is already verified present.

#### D3 — `query.py get`/`refs` crashed with `KeyError: None` **FIXED 2026-10-08**

| | |
|---|---|
| **Was** | `python tools/query.py refs vllm` — the type argument is positional and optional, so omitting it reached `DIRS[args.type]` with `args.type is None`. Traceback reproduced 2026-10-05 at `tools/query.py:139`. |
| **Where** | `tools/query.py`, `cmd_refs` and `cmd_get`. |
| **Wave** | 7 (`w7-scope-ledger`); fixed by `w8-querytool` (PR #29). |
| **Fixed now?** | **Yes.** Both commands now route through one shared `resolve_key()` helper: with no type it infers the type via a two-pass bare-stem lookup across every type directory, then falls back to the record's own `id` field — the same lookup `build_site_data.build_bare()` uses, so the two tools cannot disagree on which record a bare slug names. A genuine miss prints `no such record: <id>` plus the 13 valid types and exits 1. Verified: bare-id output is **byte-identical** to explicit-type output for 39 ids across all 13 types, both commands, 0 mismatches. |

#### D3b — `interconnect.schema.json` described `bandwidth_gbps` wrong, and could not express a verified negative **FIXED 2026-10-08**

| | |
|---|---|
| **Was** | The field was documented "per link, unidirectional", which is wrong or meaningless for most of the 71 interconnect records. The schema also had no `verified_negatively`, so four distinct causes of a null (definitional, sourcing, verified-negative, vendor-silent) lived only in prose. |
| **Where** | `schemas/interconnect.schema.json`. |
| **Wave** | 7 (`w7-hw-fabrics` reported it as an open item needing a coordinator call); fixed by `w8-ic-schema` (PR #34). |
| **Fixed now?** | **Yes, schema side.** The description now states the field holds the **vendor's own published figure**, with granularity named by `bandwidth_basis` — and it enumerates the records holding a bidirectional aggregate against those holding the per-direction half. Two optional fields added: `way` (enum `each`/`both`/`unknown`, default null) and `verified_negatively` (`["boolean","null"]`, matching engine and metric_exposure schemas verbatim). All 3,979 records validate with **zero data edits**. |
| **Data work this unlocks, deliberately not done** | `way` is unpopulated on all 71 records. The next `data/interconnect/` wave should set it on the 59 records carrying a bandwidth value, `way: unknown` on the vendor-silent ones (metaxlink, moore-threads-mtlink, cambricon-mlu-link), and `verified_negatively: true` on `iluvatar-bi-v150-and-bi-v250-fabric` and `enflame-s60-fabric`. The BI-V250 half must **not** be marked — its URL returns an application-error page, and an error is not evidence of absence. |

Naming note: `way` was chosen over `direction`/`bandwidth_direction` because
`test_tools.py` has a SCHEMA.md-sync test that fails on a schema field absent
from SCHEMA.md verbatim, and SCHEMA.md was out of scope. A later wave that can
edit SCHEMA.md may rename freely.

#### D4 — `tools/index.py` does **not** emit bare-id citations, and the 10 bare ids all resolve

| | |
|---|---|
| **What (as reported)** | "10 citations in `docs/00-index.md` cannot resolve because `tools/index.py` emits bare record ids." |
| **Where it lives** | Nowhere. **Disproven 2026-10-05, recorded so it stops being re-reported.** |
| **Why not then** | — |
| **Fixed now?** | **Not applicable — the premise was wrong.** Three independent measurements: (a) `card()` in `tools/index.py` emits the id as `` `{rid}` `` in **backticks**, and the file contains no double-bracket citation sequence anywhere, so it emits no wikilinks at all; (b) the 10 bare wikilinks in `docs/00-index.md` are *copied through from record prose*, and every one resolves against a unique record — e.g. `sglang` → `engines/sglang`, `mlx-lm` → `engines/mlx-lm`, `flop-nvidia-h100-specs` → `sources/flop-nvidia-h100-specs`; (c) `python tools/build_site_data.py --check` reports `docs 25`, `data problems 0`, and resolves them via the bare-slug lookup. |

There **was** a real, smaller defect adjacent to this one, and it is worth
stating so it is not rediscovered either: an unresolvable doc wikilink degrades
to inline code rather than failing anything. `rewrite_docs()` returns
`` `label` `` for an unknown target, so a genuinely broken docs citation is
**invisible to every check in the repo** — `validate.py` never reads `docs/`.
A citation in `docs/` that rots will not fail CI.

**That gap is now closed (2026-10-08, `w8-validate`, PR #35).** `validate.py`
resolves every `[[dir/id]]` citation in every `docs/*.md` and fails on a new rot, backed
by a ratcheting budget in `tools/docs_ref_budget.json` (8 → 5, history in the
file). Injection-tested: a planted `[[accelerators/does-not-exist]]` turns CI
red and `--max-docs-dangling-refs 7` fails against the real count of 8. The
remaining 5 are the `[label](../SCHEMA.md)` / `[label](../AGENTS.md)` class the
renderer's grammar cannot express — real files, not rot. 131 tests pass.

#### D4b — the CXL family broke the PCIe direction convention it copied from **FIXED 2026-10-08**

| | |
|---|---|
| **What** | `cxl-3-x` stored **236.0** and `cxl-4-0` stored **472.0** while each record's own `bandwidth_basis` asserted *"The recorded value now equals pcie-gen6 exactly"* — and `pcie-gen6` was 121.0. Both records stated an identity their own value violated, by exactly 2x. Only `cxl-2-0` (63.0 == `pcie-gen5`) obeyed it. |
| **Where** | `data/interconnect/cxl-3-x.json`, `data/interconnect/cxl-4-0.json`. |
| **Wave** | 8 (`w8-reverify`, PR #33) found it by reading each CXL record's prose against the `pcie-gen*` record it names; it could not edit `data/interconnect/` because a sibling owned the directory, so it recorded the gotcha with the exact repair. |
| **Fixed now?** | **Yes, 2026-10-08, in the wave-8 merge.** CXL 3.x and 4.0 run on the PCIe 6.0/7.0 physical layers, so per direction they equal that generation's per-direction figure: `cxl-3-x` 236.0 → **121.0**, `cxl-4-0` 472.0 → **242.0**. The CXL family was corrected once before, on 2026-10-03, in exactly the opposite direction — an earlier pass had halved a per-direction figure into an aggregate. The pattern is that this family has been wrong twice with a correction in between, which is why it is checked against its own stated identity rather than by memory. |

The `cxl-4-0` repair materially changes that record's recommendation: at 472
per direction its own notes claim CXL sat "roughly 4x below NVLink 4", while at
the corrected 242 per direction CXL 4.0 is essentially **at parity** with NVLink
4's 450 GB/s per direction per GPU.

### The data layer (`data/`)

#### D11 — a literal `%s` printf placeholder sat inside 34 records' notes text **FIXED 2026-10-08**

| | |
|---|---|
| **What** | 34 accelerator records carried a literal `%s` in their `notes`/`memory_basis` — e.g. *"(2026-10-04). %s NO FIGURE WAS BACK-COMPUTED"*. Introduced by wave-3 commit `9968769`. Invisible to every check in the repo because `validate.py` never reads `notes`. |
| **Wave** | 8 (`w8-reverify`, PR #33), while re-verifying the non-GPU nulling. |
| **Fixed now?** | **Yes, 2026-10-08, in the wave-8 merge.** Safe to repair only because `memory_bus_bit` is null on all 34 (explicitly verified before touching any), so the honest filler is the one the 7 uncorrupted records in the same family already carry. Every repair is annotated and dated. The **24 remaining `%s` occurrences are correct** — they are verbatim Python log-format strings and llama.cpp `%%s` router keys quoted from source code, and were audited rather than swept. |

#### D12 — `cerebras-wse-3t` was the last Cerebras record returning a bandwidth to a numeric query **FIXED 2026-10-08**

| | |
|---|---|
| **What** | `cerebras-wse-3t` stored `memory_bandwidth_gbps: 43200` (43.2 PB/s) — the on-wafer SRAM figure — while its own `memory_bandwidth_basis` already flagged it as *"THE SAME FALSE FRIEND AS THE WSE-3 RECORD"*. Its sibling `cerebras-wse-3` had been nulled for exactly this reason two waves earlier. |
| **Wave** | 8 (`w8-reverify`, PR #33) found it while confirming the WSE-3 null was correct. |
| **Fixed now?** | **Yes, 2026-10-08, in the wave-8 merge.** Nulled, with the existing basis text left in place as the explanation. No Cerebras record now returns a bandwidth figure to a numeric query, which is the correct state for a part with no DRAM tier and no memory bus. |

#### D5 — the MPT benchmark trio carries a stale pointer inside its own methodology

| | |
|---|---|
| **What** | Four records state that no 40 GB A100 record exists in this repo, and that `accelerator_ids` is therefore deliberately empty. [[accelerators/nvidia-a100-40gb-sxm4]] now exists — verified, 40 GB, 1555 GB/s, form factor `sxm`. |
| **Where** | `methodology` in [[benchmarks/mpt7b-a100-bs1-ttft-ms]], [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]], [[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]], [[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]]. |
| **Wave** | Flagged by the `w6-benchmarks` accelerator-pointer audit, recorded inside the record's own `notes`. |
| **Why not then** | It is a four-record wording change in `data/`, and the four are a set: editing one alone makes the trio inconsistent. An agent holding only one of them could not do it safely. |
| **Fixed now?** | **Yes for the wording, 2026-10-08, in the wave-8 merge.** All four records now state that [[accelerators/nvidia-a100-40gb-sxm4]] exists but is a different part (PCIe-attached Gen4 instance memory, not SXM4), so `accelerator_ids` is still deliberately empty. **Do not populate `accelerator_ids`.** |

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
| **Fixed now?** | **Yes for the notes, 2026-10-08, in the wave-8 merge.** The record's notes no longer open with the now-false "VENUE IS 'other' BECAUSE SIGCOMM IS NOT IN THE SCHEMA ENUM" — `venue` is `sigcomm` and the enum has 29 members. **The title divergence itself is a real-world fact and stays documented.** |

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
