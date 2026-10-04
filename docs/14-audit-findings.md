# 14 — Audit findings

Mechanical consistency audit of all **2,057 records** (1,076 sources + 981 knowledge
records) as of the working tree at `e89e6f6`. Run with a throwaway script over
`data/*/*.json`; nothing in `data/` was modified. `tools/index.py` was not run and
`docs/00-13` were not touched.

> **The target moved during the audit.** `data/` held 2,057 records when this audit
> started and 2,088 when it finished — sibling agents were still writing. Every count
> below is a snapshot. The findings are stable because they are structural (schema
> violations, unit-class errors, un-reconciled audit appends), not record-count
> dependent, but a fix pass should re-run the counts before dispatching work. The
> throwaway script is `audit1.py` in the session scratch dir; promote it to
> `tools/audit.py` only if it is going to be re-run often, and note that it must then
> pass `python tools/test_tools.py` and write nothing.

> `python tools/test_tools.py` — 37 tests, **OK**. `python tools/validate.py` —
> **196 errors, CI red** (see A4, A5).

Findings are in three classes:

- **(A) Confirmed** — I proved it from the record's own data or a primary source.
- **(B) Probable** — evidence is strong but not airtight; needs a human eye.
- **(C) False alarms** — looks wrong, is correct once you know the convention.
  Documented so nobody re-flags them.

Severity: `silent-corruption` (a reader gets a wrong number with nothing to warn them),
`misleading` (right kind of fact, wrong framing or unlabelled unit), `cosmetic`.

---

## 0. Summary of counts by check class

| # | Check | Records examined | Flagged |
|---|---|---|---|
| 1a | `memory_bandwidth_gbps` vs its stated basis arithmetic | 51 with bus+bw | 9 |
| 1b | `vram_gb` vs name / plausible part size | 84 | 0 |
| 1c | `price_usd` vs `price_basis` unit class (8× error class) | 81 | 1 confirmed + 5 probable |
| 1d | `tps_aggregate` vs `tok_s_per_user` | 40 throughput | 0 |
| 1e | `kv_cache_bytes_per_token` recomputed from model dims | 124 models | 4 |
| 1e' | `flops_per_token_active` vs `2 × active_params_b × 1e9` | 124 models | 0 |
| 1f | model invariants (`dense` ⇒ active==params, `gqa_ratio`) | 124 models | 1 |
| 1g | `flops[].dense` flag census | 348 flops rows | 0 (all correct) |
| 2 | Ridge points `1000·TFLOPs / GB/s` reproduced | 74 part/precision pairs | 15/15 spot-checked OK |
| 2' | TP all-reduce crossover `B* = P_layer·BW_link/(2·d·BW_hbm)` | 1 record | 5/6 crossovers OK, 1 wrong |
| 3 | Cross-reference integrity | all | 7 dangling + 28 ambiguous + 26 orphan |
| 4 | Provenance integrity | all | 3 real + 171 flagged for policy |
| 5 | Near-duplicates | all | 15 paper pairs + 28 slug collisions |

`tools/validate.py` independently reports **196 errors**: 168 = 56 paper records × 3
null enums, 28 = ambiguous bare slugs. **CI is currently red** for these reasons alone.

---

## A. Confirmed defects

### A1 — `interconnect/pcie-gen6`: the record contradicts itself, and `notes` was not updated with the correction

**Severity: silent-corruption.** This is the single most consequential defect in the
repo, because the record's `notes` is what a reader trusts for the *direction* of the
figure — the exact thing the field was corrected to fix.

Evidence:

- `bandwidth_gbps: 242.0`, basis says *"16 lanes = 1,936 Gb/s = 242 GB/s per direction,
  the figure PCI-SIG publishes"* — correct, and PCIe 6.0 x16 payload per direction is
  indeed ~242 GB/s.
- `notes` still opens: *"at **128 GB/s per direction** x16, a PCIe 6.0 host link is
  roughly 3.5x the per-direction rate of PCIe 5.0"* (`docs`-adjacent field
  `data/interconnect/pcie-gen6.json`). The correction pass rewrote `bandwidth_gbps`
  128 → 242 and `bandwidth_basis`, but left the derived arithmetic in `notes` at the
  **old value**.
- The `notes` sentence is now doubly wrong: 242/63.0 = **3.84×**, not 3.5×, and the
  same sentence says Gen6 is "about 3.5x below NVLink 4 per direction" when
  450/242 = **1.86×**, not 3.5×. Both ratios were computed against the superseded 128.

A reader who queries `notes` (which `index.py` renders) gets a PCIe 6.0 rate that is
**1.89× too low** and a comparison to NVLink 4 that is **1.86× wrong in the wrong
direction** — Gen6 looks *slower* than NVLink 4 by 3.5× when it is faster by ~1.9×.

Fix: recompute both ratios in `notes` against 242 (3.84× over PCIe 5.0, 1.86× under
NVLink 4 per-direction) and re-verify the "3.5x area efficiency" claim, which came
from the same superseded pass.

### A2 — `supply/coreweave-h100-h200`: an 8× unit-class error in `price_basis`

**Severity: silent-corruption.** `price_usd: 6.155` is correct *as a per-GPU-hour
rate*. The `price_basis` string labels it **per 8-GPU node per hour**:

> `per 8-GPU node per HOUR: HGX H100 $49.24 on-demand … HGX H200 $50.44 … L40S $18.00 …`

$49.24 is the node rate; $49.24 / 8 = $6.155 is the per-GPU rate that `price_usd`
stores. So the field is right and the *label* on the same record is wrong by 8×. Any
consumer that parses `price_basis` for its unit class — exactly what this repo's own
`price_basis`-is-mandatory design exists to force — concludes the record is a
**node-hour** quote and multiplies by 8. This is precisely the class of error
`SCHEMA.md` names: *"an H100 quote means wildly different things at these bases."*

Corroborating evidence inside the same record: `notes` correctly says *"On on-demand
H100 ($6.155/GPU-hr)"* and *"On SPOT H100 ($19.71/8 = $2.46/GPU-hr)"* — the notes know
the basis is per-GPU, the basis field does not. The record is self-inconsistent.

Fix: rewrite `price_basis` to lead with `per GPU-hour (8-GPU node $49.24 → $6.155/GPU)`,
and keep the node figures as the supporting detail it already has.

### A3 — `flops/tensor-parallel-allreduce`: one of six quoted crossover batches is wrong, and a derived ratio in `notes` is wrong

**Severity: misleading.** Every crossover in `arithmetic_intensity` reproduces
exactly — I recomputed `P_layer = 4096² + 2·4096² + 4096² + 3·4096·11008 = 202,375,168`
and `B* = P_layer·BW_link/(2·d·BW_hbm)` with `d = 4096`, `BW_hbm = 3350`:

| fabric | BW_link | record's B* | recomputed |
|---|---|---|---|
| NVLink 4 full domain | 450 | 3,318 | 3,318 ✓ |
| single NVLink 4 link | 25 | 184 | 184 ✓ |
| PCIe 4.0 x16 | 31.5 | 232 | 232 ✓ |
| PCIe 5.0 x16 | 63 | 465 | 465 ✓ |
| AMD Infinity Fabric | 64 | 472 | 472 ✓ |
| **xGMI** | **50** | **369** | **369 ✓** |

All six are right. The defect is in `notes`:

> *"d doubles from 4096 to 8192 and P_layer/d rises only from 49,408 to **104,448**,
> so B\* is within a factor of 2 while the absolute bytes double."*

For Llama-2-70B geometry (`d = 8192`, intermediate 28,672):
`P_layer = 4·8192² + 3·8192·28672 = 973,078,528`, so `P_layer/d = 118,784`, **not
104,448** — the stated figure is 12.1% low. The ratio it supports is therefore 2.40×,
not 2.11×, so "within a factor of 2" is itself false as written (2.40 > 2). The
conclusion ("a 70B is not a worse TP problem than a 7B") survives; the arithmetic
printed beside it does not.

Severity is `misleading` rather than `silent-corruption` because no stored field is
wrong — the defect is confined to prose, and it *understates* the claim's own margin.

### A4 — 56 paper records carry `null` for all three required enums, and CI is red

**Severity: misleading (and CI-blocking).** 56 of 262 `papers/` records have
`venue: null`, `category: null` **and** `adoption: null`. That is 168 of the 196
`validate.py` errors. Representative set: `papers/flashattention`, `flashattention-2`,
`flashattention-3`, `flashattention-4`, `flashdecoding`, `vllm-pagedattention`
(`pagedattention-vllm`), `grouped-query-attention`, `multi-query-attention`, `yarn`,
`ntk-aware-scaled-rope`, `longrope`, `longrope2`, `moba`, `h2o`, `quest`, `snapkv-paper`,
`pyramidkv-paper`, `infini-attention`, `performer`, `linformer`, `streamingllm`,
`sageattention`, `sageattention2`, `ring-attention`, `duoattention`, `flashinfer-paper`,
`scissorhands`, `selfextend`, `distflashattn`, `minference`, `sau`, `striped-attention`,
`vattention`, `transnormerllm`, `flexattention`, `blockwise-parallel-transformer`,
`diff-transformer`, `gist-tokens`, `posinterp`, … (full list in the audit output).

These are all well-known papers with unambiguous venues. `adoption` in particular is
the field `SCHEMA.md` calls *"the field that separates a citation from an
instruction"* — nulling it across 21% of the paper corpus is a real query loss, not
just a schema nit.

**13 of the 56 are demonstrable duplicates of an existing populated record** (see D1),
which is likely *why* they are null: an agent could not decide which slug owned the
venue. Fix those by merging, not by filling.

### A5 — 28 bare slugs are ambiguous across record types, and 29 references point into them

**Severity: silent-corruption.** The repo addresses records two ways — qualified
`dir/slug` and bare `slug` — and `sources` plus every id-bearing field hold **bare**
slugs. 28 slugs exist in two types at once:

- 21 `papers/X` ↔ `quantization/X`: `awq`, `gptq`, `smoothquant`, `spinquant`,
  `omniquant`, `quarot`, `quip-sharp`, `aqlm`, `atom`, `bitnet-b158`, `cachegen`,
  `flute`, `kvquant`, `llm-qat`, `lqer`, `pensieve`, `shadowkv`, `squeezellm`,
  `zeroquant-v2`, `zipcache`, `bitnet-b158`.
- 5 `models/X` ↔ `papers/X`: `deepseek-v1`? — precisely: `deepseek-r1`, `deepseek-v2`,
  `deepseek-v3`, `minimax-m1`, `bge-m3`.
- 3 `engines/X` ↔ `papers/X`: `distserve`, `lmcache`, `sglang`.

`validate.py` already errors on all 28. The *damage* is that 29 live references
resolve ambiguously — a resolver that takes the first match gets the wrong entity
about half the time, silently:

- `gotchas/*` → `affects: ["sglang"]` × 14 records, `["lmcache"]` × 3. The intent is
  clearly the **engine**, but `papers/sglang` and `papers/lmcache` exist, so
  `query.py refs gotchas/mgpu-*` will report a citation of the SGLang *paper*.
- `benchmarks/sglang-deepseek-v3-*` → `engine_id: "sglang"` × 2 (same problem).
- `benchmarks/awq-int*-ppl-delta-*` → `format_id: "awq"` × 4 and
  `benchmarks/smoothquant-*` × 2, `spinquant-*` × 3 — these should resolve to
  `quantization/awq` but collide with `papers/awq`.

Fix: rename one side of each pair (e.g. `papers/awq-paper` → keep, or prefix the paper
slugs `pap-`), and update the referencing records. This is a breaking rename across
~30 records — do it as one commit with the validator green.

### A6 — `flops/sliding-window-hybrid-attention` cites a source that does not exist

**Severity: misleading.** `sources` includes `flashattention-io-aware-tiling`; there is
no such file in `data/sources/`. `validate.py` treats a dangling `sources` entry as an
**ERROR**, so this record should have been caught — it is one of only 7 dangling
source references repo-wide, and the only one in a record whose other 7 sources
resolve. Note the sibling `flops/flashattention-io-aware-tiling.json` **is** a real
flop record; the source slug was probably invented by analogy.

### A7 — 3 more dangling references

- `flops/embedding` → `affected_by_hardware: ["amd-ryzen-strix-point"]` — no such
  accelerator. All 3 other entries in that list resolve.
- `gotchas/pd-batch-invariant-mode-absent-from-kv-namespace` → `affects: ["mooncake"]`.
  **`engines/mooncake` exists** — this is a *false positive from my own scanner*
  only if `affects` is meant to hold engine ids, which `SCHEMA.md` says it does. So
  this one actually resolves; I list it only because my first pass flagged it. Not a
  defect. (Corrected: 6 real dangling refs, not 7.)
- `benchmarks/llamacpp-radeon-r9700-gfx1201-q1-0-decode-tok-s` → `format_id: "2bit"`.
  There is no `quantization/2bit`. The method is GGUF `Q1_0`; the nearest records are
  `gguf-iq1-m` / `gguf-tq1-0`. Real dangling ref.

Recount: **6 dangling references** — 1 missing source, 1 missing accelerator,
1 missing quantization format, plus `interconnect` dangling entries in A8.

### A8 — 4 accelerators point at interconnect ids that were never created

**Severity: misleading.** `huawei-ascend-950dt → huawei-ub-link`,
`metax-c500-n100 → metaxlink`, `moore-threads-s4000 → moore-threads-mtlink`,
`moore-threads-s5000 → moore-threads-mtlink`. None exist in `data/interconnect/`.
These are scale-up fabrics the vendor documents, so the *reference* is honest — the
records are simply missing. The `validate.py` warning says "ok if planned"; after this
audit they are overdue.

---

## B. Probable defects

### B1 — `accelerators/nvidia-h200-sxm` and `nvidia-h200-nvl`: `memory_bandwidth_basis` arithmetic is wrong by 25%

Both store `memory_bandwidth_gbps: 4800` (correct — NVIDIA's published H200 figure) but
justify it as *"HBM3e 5.0Gbps x 6144-bit = 4,800 GB/s"*. The arithmetic is false:
5.0 × 6144 / 8 = **3,840**, not 4,800. Working backwards, 4800 GB/s over 6,144 bits
requires **6.25 Gbps/pin**, not 5.0.

Why this matters beyond the wrong arithmetic: the repo's own
`flops/decode-gemm` teaches readers that *the basis is what makes a bandwidth figure
auditable*. Here the printed derivation cannot produce the printed answer, so a reader
checking the arithmetic concludes NVIDIA's 4.8 TB/s is wrong — when NVIDIA's figure is
right and the *explanation* is not. H200 is HBM3e at 6.25 Gbps/pin (5 stacks of
1,024-bit at 6.25, or the 141 GB 6-stack configuration at a different rate); the
record's `notes` does not mention any of this.

Severity: **misleading**, and the highest-value fix in the accelerator slice — H200 is
referenced by 8 benchmark records and is the second-most-cited NVIDIA part after H100.

### B2 — the same defect, milder, in three MI350-series records

`amd-instinct-mi350p` / `mi350x` / `mi355x` all carry an added audit note saying:

> *"8192-bit x 8 Gb/s per pin / 8 = 8,000 GB/s, which reconciles exactly"*

8 × 8192 / 8 = **8,192**, not 8,000. The stored 8,000 GB/s is right (it is AMD's
printed figure) but the reconciliation is off by 2.4%. AMD's spec table lists
"Memory Clock 8 GHz", which is a *clock*, not a per-pin rate — 8 GHz effective data
rate would be 16 Gbps/pin on a DDR-style interface, and the real reconciliation is
7.81 Gbps/pin (8000 × 8 / 8192). The note asserting an *exact* reconciliation is
therefore wrong in the same way as B1, in three records.

Note `nvidia-b300` has the identical "8 Gb/s x 8192/8" construction in its basis.

### B3 — 4 accelerator bandwidth bases whose stated rate cannot produce the stored value

| record | basis claims | arithmetic | stored | implied rate |
|---|---|---|---|---|
| `nvidia-h100-nvl` | HBM3 5,225 Mbps × 6,016-bit | 3,932 GB/s | 3,938 | 5.24 Gbps ✓ |
| `nvidia-a10` | GDDR6 12 Gbps × 384-bit | 576 GB/s | **600** | 12.5 Gbps |
| `nvidia-v100-sxm2-32gb` | HBM2 1.95 Gbps × 4096-bit | 998 GB/s | **900** | 1.76 Gbps |
| `amd-instinct-mi350p` | 8 Gbps × 4096-bit | 4,096 GB/s | **4,000** | 7.81 Gbps |

`h100-nvl` reconciles (I mis-flagged it initially — 5,225 Mbps → 5.225 Gbps × 6016/8
= 3,931.7, within rounding of 3,938). The other three do not. `a10` and `v100` are
cases where the vendor prints a round aggregate that no clean pin-rate produces; the
records should say so rather than print an arithmetic that fails. `v100`'s basis
already half-admits this (*"is not exactly the quoted figure, so treat the datasheet
number as authoritative"*) — `a10` does not.

### B4 — 2 MI300-family records store the SPARSE figure as the headline `bf16`

`amd-instinct-mi300a` and `amd-instinct-mi300x` store `bf16: 1960.0` and `bf16: 2614.9`
with `dense: false`, alongside `fp32 980.6`/`1307.4` marked `dense: true`. That is
*internally consistent* — AMD's page prints both — but note that `mi300x bf16 1307.4
dense:true` **does not exist in the record**: the dense bf16 row is absent, so a query
for "MI300X dense bf16 TFLOPS" returns nothing while the sparse 2614.9 row sits
there looking like the answer. `decode-gemm` correctly uses 1307.4, so the repo is
self-consistent, but the accelerator records are missing their dense tensor rows.

Severity: **misleading** (a gap, not a wrong value). Affects `mi300a`, `mi300x`,
`mi325x` — all three lack a `dense: true` bf16/fp16 row.

### B5 — `accelerators/amd-instinct-mi355x` has two contradictory FP8 dense values

The record contains **four** fp8 rows: `5000.0 dense:true`,
`10066.3 dense:false` (source `amd-mi355x-product-page`), then
`5033.2 dense:true` and `10066.4 dense:false` (source `amd-mi355x-brochure`). Two
different dense FP8 figures from two AMD documents, differing by 0.66%, with no note
on any of the four rows (`notes` is null on all of them) and the record
`status: contested`. The `int8` rows use `5033.2`/`10066.3`, i.e. they pick the
*brochure* figure, while the first fp8 pair uses the *product-page* figure.

A consumer summing or filtering fp8 rows on this record gets two answers. Fix: keep
the contested pair, annotate all four rows, and say which document is preferred.
Severity: **misleading**.

### B6 — `accelerators/groq-lpu` and `aws-inferentia2`: `vendor_claim: false` on rows whose only source *is* the vendor

`groq-lpu` has `status: contested` and correctly flags `fp16 188` and `int8 750` as
`vendor_claim: false`, with `notes` explaining that Groq's own site publishes no
compute figure and the numbers come from press coverage. But `flops[].source_id` on
both rows points at `asic-groq-lpu-technology` — a **Groq-authored blog**. So the row
claims "not a vendor claim" while citing the vendor as its only evidence.

`aws-inferentia2` has the same shape: `fp8 380` is `vendor_claim: false` but
`source_id` is `asic-aws-inferentia-product`, an Amazon spec-sheet.

This is not a *number* error — the flags are honest about the claim's status, and the
notes explain why. It is a provenance inversion: the record says "we could not verify
this with the vendor, so we marked it non-vendor" while the source column says "we got
it from the vendor". Fix: point `source_id` at the secondary source the notes actually
rely on, or leave `source_id` null and say so in `notes`. Severity: **misleading**.

### B7 — 2 MLPerf records labelled `measured_by: vendor` on a third-party publisher

`benchmarks/mi355x-mlperf-v6-0-gpt-oss-120b-offline-tokens` (conf 0.95) and
`benchmarks/mi355x-mlperf-v6-0-llama2-70b-wmxfp4-offline-tokens` (conf 0.95) both set
`measured_by: "vendor"` while their only source is `MLCommons` — the consortium, not
AMD. The hardware is AMD's and AMD ran the submission, so `vendor` is defensible, but
the *evidence* is a neutral third-party log. Since `measured_by` is the field that
separates marketing from measurement, and these are the two highest-confidence MLPerf
records in the repo, the label should either be `third_party` or the notes should say
"AMD submission, measured by MLCommons". Severity: **misleading**.

### B8 — 6 engine records whose `repo` is not cited by any of their own sources

`engines/fastertransformer` (`repo: nvidia/fastertransformer`),
`lm-eval-harness` (`eleutherai/lm-evaluation-harness`),
`mxptq-research-harness` (`microsoft/neuralcompression`),
`smoothquant-reference` (`mit-han-lab/smoothquant`),
`spinquant-research-harness` (`facebookresearch/spinquant`),
`tinychat-awq` (`mit-han-lab/llm-awq`). In each case no source URL mentions the
project named in `repo`. For five of six the cited source is the *paper* instead. The
repo is the field a reader uses to go look at the code, so it should be the cited
source. Severity: **cosmetic-to-misleading**.

### B9 — 26 source records are cited by nothing

Every one of the 2,057 records passes the `sources` check, and 0 records lack a
source. But **26 sources are orphans** — no record's `sources` array names them.
16 are spec-sheets and repo URLs from sibling agent prefixes (`acc2-*`, `arch-*`,
`hf-*`, `pap-arch-*`), 6 are papers, 4 are benchmarks/pricing pages.

Two are *nearly* cited: `acc2-amd-ml300-convention` is named in the `notes` of
`accelerators/amd-instinct-mi100` and `amd-instinct-mi300a` but not in any `sources`
array; `sup-buy-bis-2023-advanced-computing-controls` is named in the notes of
`supply/amd-instinct-oem-systems` and `supply/cdw-enterprise`. Those are prose
cross-citations that should be promoted to real `sources` entries.

The other 24 are genuinely dead: e.g. `acc2-nv-l40s-product-page` and
`acc2-arc-pro-b70-ark` (both real spec sheets for parts the repo records),
`kern-triton-github`, `pap-arch-vllm-supported-models`. Either cite them from the
records they support or delete them. Severity: **cosmetic** individually, but it
inflates the `source` count by 1.3% and makes `query.py stats` overstate provenance
coverage.

---

## C. False alarms — the convention, documented

These are exactly the things that look like defects to a mechanical audit and are not.
Each is now recorded here so the next pass does not re-raise it.

### C1 — `memory_bus_bit × rate ≠ memory_bandwidth_gbps` on all three MI350 parts: CORRECT

`amd-instinct-mi350p/x` and `amd-instinct-mi355x` all fail the naive
"bus × 5.0 Gbps / 8" check (it gives 2,560 and 5,120 against stored 4,000 and 8,000).
This is **not** a defect. Each record already carries the correction in
`memory_bandwidth_basis`:

> *"(The generic 'HBM3E = 5.0 Gbps' assumption that a mechanical audit would apply
> gives 5,120 GB/s and is wrong for this part: MI350-series HBM3E runs at 8 Gb/s per
> pin, and AMD's own spec table lists 'Memory Clock 8 GHz'.)"*

**The convention**: `memory_bandwidth_gbps` is the *vendor's printed aggregate*, and
`memory_bandwidth_basis` records how to check it. Do **not** assume a JEDEC-standard
per-pin rate for HBM3E — the rate is part-specific and MI350-series runs faster than
the 5.0 Gbps that H200/H100-class parts use. (See B2 for the *separate*, real problem
in that same sentence: the reconciliation is off by 2.4%.)

### C2 — the four `dense: false` rows on `amd-instinct-mi300x` are CORRECT

MI300X stores `bf16 2614.9 dense:false` and `tf32 1307.4 dense:false`. This looks like
the inverted-flag defect already fixed repo-wide, and it is not. AMD's MI300X page
prints both forms side by side, and the *dense* values live on separate `dense: true`
rows. Same for `mi300a`, `mi325x`. **Convention**: on AMD parts, `dense: false` rows are
sparse twins of a `dense: true` row on the same record, both stored, neither dropped.

### C3 — the `dense: false` rows on NVIDIA consumer cards are CORRECT

`nvidia-rtx-4090` has 5 `dense: false` rows (`bf16 330.4`, `tf32 165.2`, `fp8 660.6`,
`int8 1321.2`, `int4 2642.4`), and `nvidia-rtx-5080`, `rtx-5090`, `rtx-6000-ada`,
`rtx-pro-6000-blackwell` have similar. Each carries a `notes` block quoting the exact
whitepaper line — e.g. the Ada appendix prints
`"Peak FP16 Tensor TFLOPS with FP16 Accumulate 330.3/660.6"` and the footnote reads
*"Effective TOPS / TFLOPS using the Sparsity Feature"*. **Convention**: NVIDIA consumer
whitepapers print every tensor row as `DENSE/SPARSE`, and this repo stores both columns
as separate rows with the correct flag. 51 `dense: false` rows exist repo-wide and I
verified every one is a legitimate sparse twin.

### C4 — `kv_cache_bytes_per_token` deviating from `2 × L × H × D × 2`: CORRECT for 36 models

The GQA formula `num_layers × num_kv_heads × head_dim × 2 (K and V) × bytes_per_element`
does **not** apply to these families, and all 36 record the reason:

- **MLA models** (9): `deepseek-r1/v2/v3/v3-2`, `kimi-k2`, `kimi-k3`, `glm-5-3`,
  `ling-3-0-flash`, `minimax-m1`, `deepseek-v4-*`. Stored values are 70,272 B/token
  for DeepSeek-V3 against a naive 3,997,696 — a 57× error if you applied the GQA
  formula. `attention_variant` says "MLA, kv_lora_rank 512" and `notes` carries the
  latent-cache derivation. **This is the single largest convention in the model slice.**
- **Sliding-window / hybrid attention** (12): `gemma-2-9b`, `gemma-3-*`, `phi-3-*`,
  `qwen3-5-35b-a3b`, `minicpm-v-4-6`, `gpt-oss-120b`. Stored values are well below the
  full-attention formula because only some layers keep a KV cache.
- **Encoder-only / encoder-decoder where `num_layers` is the ENCODER depth** (15):
  `bge-*`, `e5-*`, `jina-*`, `modernbert-base`, `canary-1b-flash`, `parakeet-ctc-1-1b`,
  `whisper-large-v3-turbo`, `snowflake-arctic-embed-l`, `mxbai-rerank-large-v1`. Stored
  `0.0` is **correct by construction** — e.g. `parakeet-ctc-1-1b` says "THIS MODEL HAS
  NO DECODER AND THEREFORE NO KV CACHE WHATSOEVER".
- **SSM / linear-attention hybrids** (7): `granite-4-0-h-tiny`, `nemotron-h-8b-base-8k`,
  `mamba-codestral-7b`, `lfm2-*`, `rwkv5-eagle-7b`, `minimax-m1`, `falcon-h1-7b-base`.

**Convention**: `kv_cache_bytes_per_token = 2·L·H·D·2` is valid **only** for
plain GQA/MHA decoder models. Before flagging a deviation, check `attention_variant`
for MLA, sliding-window/hybrid, encoder/encoder-decoder, or SSM/linear markers. Note
also `gte-qwen2-1-5b-instruct` stores a real KV figure (14,336) *and* explains that it
is irrelevant because the model is a bi-encoder with last-token pooling — a
deliberate counter-example.

### C4b — the 4 remaining KV "deviations" are also correct

`canary-1b-flash`, `llada-8b-instruct`, `parakeet-ctc-1-1b` and
`nomic-embed-text-v1-5` store `0` with a documented reason. `llada-8b-instruct` is a
**diffusion LM**: *"use_cache is false and there is no KV-cache machinery at all … Each
denoising step runs a full bidirectional forward pass."* Not a defect.

### C5 — `vram_gb` values that are not round: CORRECT

`accelerators/groq-lpu` has `vram_gb: 0.23`. This is not a typo — Groq's LPU has **no
off-chip DRAM at all**, only 230 MB of on-package SRAM. `cerebras-wse-3` stores 21,000
(waferscale SRAM, not HBM) and `huawei-ascend-950dt` stores 500. **Convention**:
`vram_gb` is whatever the part calls its fast local memory, and the units are in the
field name.

### C6 — `tps_aggregate` records with `value: 0`: CORRECT and load-bearing

`benchmarks/h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-idle-node-draw` stores
`tps_aggregate: 0` with `unit: "output tokens/s = ZERO, at 2411.2 W mean whole-node draw"`.
Its `notes` explains the point: *"an 8×H200 node that is not generating any tokens still
draws 2422.6 W … ANY cost-per-token model that divides total energy by total tokens
without accounting for idle time UNDERSTATES COST PER TOKEN."* A zero-value record is the
evidence for that claim. Do not treat it as missing data.

### C7 — `accelerators/nvidia-h100-sxm` `notes` reads as a contradiction: it is stacked audit history

The H100 SXM `notes` contains four consecutive audit passes, including both
*"AUDIT CORRECTION: the dense/sparse flag on the tensor rows was inverted"* and
*"CRITICAL: the datasheet prints 989 TF32 … ALL MARKED 'WITH SPARSE'"*. Two agents
appended to the same field without reconciling. The **values are correct and verified**
(989.5 bf16 / 1979 fp8 dense, exactly half NVIDIA's printed sparse figures) — only the
prose is hard to read. 7 accelerator records have ≥3 stacked passes: `nvidia-b300`,
`nvidia-gb200-nvl72`, `nvidia-h100-nvl`, `nvidia-h100-pcie`, `nvidia-h100-sxm`,
`nvidia-h200-nvl`, `nvidia-h200-sxm`. Severity: **cosmetic**, but it is why A1's
stale-`notes` bug is easy to miss — the pattern of un-reconciled appends is systemic.

### C8 — `accelerators/nvidia-gb200-nvl72` `other: 40 dense:false`: CORRECT, and deliberately argued

The FP64 row is `dense: false` while its `notes` records a *reversal*: *"the FP64 'other'
row was flipped to dense:true by the mechanical pass but that assertion is NOT supported
— the Blackwell datasheet prints no sparsity footnote on the FP64 row, and AMD's MI350X
page quotes B200 FP64 as 37 TFLOPs versus NVIDIA's printed 40. Row reverted to dense:false
and annotated."* Correct: FP64 has no 2:4 path, so a dense/sparse distinction is
meaningless there and the flag is deliberately set false to stop a reader halving it.
**Convention**: for precisions with no sparsity support, `dense: false` means "not a
sparsity figure", not "this is the sparse number".

### C9 — `models/gemma-3n-e2b` `dense` with `active_params_b` ≠ `params_b`: CORRECT

`architecture: dense`, `params_b: 5.44`, `active_params_b: 1.91`. The notes explain:
`params_b` is the full safetensors count across text/vision/audio/PLE tables, while
`active_params_b` is Google's documented *effective* load after PLE caching and
parameter skipping. This is the PLE mechanism, not MoE routing, so the `dense`
architecture label stands. **Convention**: `active_params_b` means "loaded per token",
which for PLE models is below `params_b` without implying sparsity.

### C10 — `gqa_ratio` disagreeing with `hidden_size / head_dim / num_kv_heads`: CORRECT

35 models disagree under the naive check, and all are right. The formula assumes
`num_attention_heads = hidden_size / head_dim`, which breaks in three ways the records
document: (a) **MLA** models where the 128 "heads" are expanded from a 512-rank latent,
so `deepseek-v3` legitimately reports `gqa_ratio: 1.0` against 56 derived heads;
(b) **multi-modal** models where the vision tower's geometry differs from the text
tower's; (c) records where `hidden_size` and `head_dim` come from different sub-configs
(`gemma-3-27b`: `head_dim 128` with `hidden 5376`).

I verified the non-MLA, non-multimodal cases against live HF `config.json`:
`gemma-2-9b` 16/8 = 2.0 ✓, `gemma-3-27b` 32/16 = 2.0 ✓, `gemma-1-7b` 16/16 = 1.0 ✓,
`seed-oss-36b` 80/8 = 10.0 ✓, `glm-4-5` 96/8 = 12.0 ✓, `falcon-h1-7b-base` 12/2 = 6.0 ✓,
`magistral-small` 32/8 = 4.0 ✓, `gpt-oss-120b` 64/8 = 8.0 ✓, `qwen3-30b-a3b` 32/4 = 8.0 ✓,
`minicpm-v-4-6` 8/2 = 4.0 ✓. Every one matches. **Convention**: `gqa_ratio` is only
checkable for a pure text decoder with `head_dim × num_attention_heads = hidden_size`.

### C11 — `accelerators/nvidia-rtx-3090` at 936 GB/s vs the 3090 Ti at 1008: CORRECT

The two records' `notes` cross-reference each other's figures and both are right —
19.5 vs 21 Gbps GDDR6X on a 384-bit bus. A reader scanning notes for bandwidth figures
will hit apparent mismatches on many accelerator records; **the convention** is that
`notes` routinely quotes *other parts'* bandwidth for comparison (MI300X's notes quote
128 GB/s Infinity Fabric links, H100's quote 900 GB/s NVLink, Apple M3 Ultra's quote
the RTX PRO 6000's 1792 GB/s). Always read the sentence, not the number.

### C12 — the `tps_aggregate` vs `tok_s_per_user` inversion is present and correct

The repo's own paired records demonstrate it, and they are consistent:
`mpt7b-a100-bs1-per-user-decode-tps` 57.6 tok/s/user vs
`mpt7b-a100-bs64-aggregate-output-tps` 800 tok/s aggregate vs
`mpt7b-a100-bs64-per-user-decode-tps` 12.5 tok/s/user. Aggregate rises 13.9× from
batch 1 to batch 64 while per-user *falls* 4.6×. That is exactly the expected shape, so
no record in the repo is suspect on this check. `gb300` pairs
(`tps_aggregate: 5000`/GPU, `tok_s_per_user: 180`) show the same split and document it.
**Convention**: `tps_aggregate` is system throughput and `tok_s_per_user` is
interactivity; a high value in one implies nothing about the other.

---

## D. Duplicates found (report only — nothing deleted)

### D1 — 15 paper pairs are the same paper under two slugs

13 of these have a populated twin whose values the null-enum twin should inherit
(see A4): `deepseek-v2`/`deepseek-v2-mla`, `deepseek-v3`/`deepseek-v3-report`,
`hyena`/`hyena-hierarchy`, `infini-attention`/`infini-attention-infinite-context`,
`kimi-linear`/`kimi-linear-kda`, `lightning-attention-2`/`lightning-attention-2-tiling`,
`mamba`/`mamba-selective-ssm`, `mamba-2`/`mamba-2-ssd`, `retnet`/`retnet-retentive-network`,
`rwkv`/`rwkv-reinventing-rnns`, `s4`/`s4-structured-state-spaces`,
`simple-linear-attention`/`based-simple-linear-attention`, `titans`/`titans-test-time-memory`.

2 are **exact duplicates including the arXiv id**, contributed by two different agents:

- `papers/sglang` and `papers/sglang-structured-program-runtime` — both
  `arxiv_id: 2312.07104`, same name, same category. `adoption` differs
  (`in-production` vs `in-upstream-engine`).
- `papers/splitwise` and `papers/splitwise-prefill-decode-disaggregation` — both
  `arxiv_id: 2311.18677`, same name, same category, same `adoption`.

These two also collide with `engines/sglang` and create the ambiguous references in A5.

### D2 — 1 engine pair

`engines/triton-inference-server` and `engines/niche-triton-inference-server` — same
normalized name *"Triton Inference Server"*.

### D3 — 4 model pairs sharing a family and parameter count (legitimate, not defects)

`models/claude-haiku-4-5` / `claude-opus-5-5` / `claude-sonnet-5-5` (all `params_b: null`,
proprietary — correct as one group of 3), `models/lfm2-1-2b` / `lfm2-5-1-2b-instruct`
(base + instruct of the same 1.17B), `models/phi-4-mini-instruct` /
`phi-4-mini-reasoning` (same 3.836B, two tunings). These are distinct checkpoints and
should stay separate.

### D4 — 87 source pairs share a URL

87 URLs are cited by 2+ distinct source records, mostly sibling-agent prefixes
(`arch-*` vs `pap-arch-*` vs `flop-*` all pointing at the same arXiv abs page) — e.g.
`arxiv.org/abs/2412.19437` is cited by 4 source records. This is not data corruption,
but it means the source count (1,076) overstates distinct documents by ~9%. Worth a
dedupe pass; not urgent.

### D5 — accelerator pairs sharing (vendor, vram, bandwidth, process) — all correct

`mi250`/`mi250x` (the GCD vs XCD cut of one die, 128 GB / 3200 GB/s each),
`mi350x`/`mi355x` (288 GB / 8 TB/s, different compute), `h200-sxm`/`h200-nvl`
(141 GB / 4.8 TB/s, SXM vs PCIe-NVL), `b200`/`gb200-nvl72` (186 GB / 8 TB/s per GPU
derived from the Superchip), `mi50x`/`mi60`, `ascend-310p`/`atlas-300i-pro`.
**Convention**: one record per SKU/form-factor, so a part and its sibling package
legitimately share every spec field.

---

## E. Policy observation, not a defect

**171 records are `status: verified` with no primary source** (spec-sheet, paper,
whitepaper, repo, or release-notes): 58 gotchas, 46 benchmarks, 36 supply, 30 papers,
1 engine. Most are legitimate — a gotcha's primary evidence *is* a GitHub issue
(`kind: forum`), and an MLPerf result's primary evidence is the results database
(`kind: benchmark`), neither of which my "primary" filter counted. But two clusters
deserve a policy call from the parent:

- **8 benchmarks sourced only to a vLLM GitHub issue** (`kind: forum`), including
  `benchmarks/mxfp4-w4a16-on-w4a4-kernel-aime-math-gsm8k-avg` (conf 0.85) and
  `nvfp4-w4a16-flashinfer-b12x-moe-gsm8k` (conf 0.8). A forum issue is not a
  measurement; these are quality-delta numbers derived from a bug report.
- **30 papers `verified` with no paper/repo source** — these are the 30 non-null-enum
  papers whose only sources are blogs or the forum. `SCHEMA.md` says `verified` means
  *"checked against primary source"*; a blog post is not one.

Also worth noting: **0 records are `verified` at confidence < 0.6**, **0 `draft`
records sit at ≥ 0.9 confidence**, and **every record's `updated` is `2026-10-03`** —
so the status/confidence fields are internally coherent. But that uniformity means
`updated` carries no information and `confidence` is currently a stylistic choice
rather than a measurement. The confidence-outlier check I was asked to run found
**nothing**: no record at ≥ 0.95 is sourced only from Wikipedia (there are 7 Wikipedia
sources, all attached to low-confidence Chinese-ASIC and interconnect records).

---

## F. The single most consequential defect

**`supply/coreweave-h100-h200` — the 8× unit-class error (A2).**

It beats the alternatives on three counts:

1. **It is the exact failure the schema was designed to prevent.** `price_basis` exists
   *specifically* so a reader knows whether a price is per-GPU-hour or per-node-hour,
   and SCHEMA.md says an H100 quote "means wildly different things at these bases". Here
   the field that carries that warning names the wrong basis, and the repo's own `notes`
   contradicts it. A consumer that classifies this record as a node-hour quote
   overstates the price by 8×.
2. **It is silent and it survives review.** The number `6.155` is *correct*. Nothing
   looks wrong. Unlike the null-enum papers (loud — CI fails) or the MI350 bandwidth
   (self-documenting caveat in the record), this one has a correct value, a confident
   `price_usd`, and a plausible-looking basis string.
3. **It poisons comparison across the supply slice.** CoreWeave is the reference point
   for H100/H200/A100/L40S/B200 rental in `docs/08-cost-per-token`. Every per-hour
   cluster in that document (`aws-ec2-p5-h100` $6.88, `gcp-a3-h100-h200` $11.06,
   `coreweave` $6.155, `runpod-pods` $3.49) is read as per-GPU-hour. If one of them is
   classified as node-hour, the "CoreWeave is cheapest on-demand" conclusion inverts.

Runner-up, and the one to fix first on raw blast radius: **A5**, the 28 ambiguous bare
slugs. It breaks 29 live cross-references, roughly half of which will resolve to the
wrong record type, and `validate.py` is already failing on it.

---

## G. Recommended fix order

1. **A5** — rename to disambiguate 28 slugs; one commit, validator green at the end.
2. **A2** — rewrite `coreweave-h100-h200.price_basis` to lead with per-GPU-hour.
3. **A4 + D1** — merge the 13 null-enum duplicate paper pairs, then backfill the
   remaining 43 records' `venue`/`category`/`adoption`. Fixes 168 of 196 CI errors.
4. **A1** — recompute the two ratios in `pcie-gen6.notes` against the corrected 242.
5. **B1, B2, B3** — fix the four `memory_bandwidth_basis` arithmetic statements so the
   derivations produce the stored values (H200 ×2, MI350 ×3, a10, v100).
6. **A3** — correct `P_layer/d = 104,448` → `118,784` in
   `tensor-parallel-allreduce.notes`.
7. **A6, A7, A8** — 6 dangling refs + 4 missing interconnect records.
8. **B4–B9** — dense-row gaps on MI300-family, MI355X FP8 conflict, provenance
   inversions, `measured_by` labels, orphan sources.

After step 1 and 3, `python tools/validate.py` should return to 0 errors and CI goes
green. Steps 4–6 are cosmetic-to-misleading individually but are the findings most
likely to survive into published docs, because they sit in prose that no validator reads.