# Schema reference

JSON Schema per record type lives in `schemas/`. This is the human version.

## Common fields (all types)

| field | type | meaning |
|---|---|---|
| `id` | string | stable kebab-case slug, globally unique, matches filename |
| `type` | enum | accelerator, flop, engine, quantization, interconnect, benchmark, gotcha, supply, model, paper, compiler, metric_exposure, source |
| `name` | string | display name |
| `status` | enum | verified (checked against primary source), draft, contested (sources disagree), deprecated |
| `confidence` | number | 0.0-1.0 |
| `updated` | date | when the record was last verified, not when it was written |
| `sources` | [string] | source ids, must resolve in data/sources/ |
| `notes` | string | caveats, contradictions, derivations |

## accelerator

    vendor                 nvidia | amd | intel | qualcomm | apple | broadcom |
                          google | amazon | microsoft | meta | cerebras | groq |
                          tenstorrent | cambricon | samba | asic-other | other
                          (broadcom, microsoft, meta, samba and other have zero
                          records today; asic-other, 26 records, is the working
                          bucket for parts with no first-party vendor page)
    architecture           e.g. hopper, blackwell, gfx1100, arc-b-series, a100, m-series
    release_year           int
    process_nm             int, die node
    form_factors           [string]   pcie | sxm | oam | mdu | mcm | socs
    vram_gb                number     per-device or per-package, unit in field name
    memory_type            string     hbm3e | gddr6x | lpddr5x, ...
    memory_bus_bit         int
    memory_bandwidth_gbps   number
    memory_bandwidth_basis  string     how the bandwidth number was arrived at
    onchip_cache           object|null  { kind, l2_kb, llc_mb, sram_mb, per_scope,
                               basis, source_note } - see below
    flops                  [ { precision: fp32|tf32|bf16|fp16|fp64|fp8|fp6|fp4|
                                               fp4_block_scaled|int8|int16|int4|other,
                                   tflops: number, dense: bool, vendor_claim: bool,
                                   source_id: string|null,
                                   basis: vendor-spec-sheet|vendor-whitepaper|
                                          vendor-docs|secondary-aggregator|derived|
                                          unverified,
                                   basis_detail: string,
                                   unit: string|null } ]
    tdp_w                  number     per device or per accelerator in a package
    interconnect           [string]   interconnect ids this device exposes
    unified_memory         bool
    consumer               bool
    notes

`flops` is a list because precision matters more than the headline number. Always
record whether a figure is dense or sparse — sparse counts are not comparable and
marketing conflates them.

`basis` is REQUIRED on every `flops` row (added 2026-10-04). A row states which kind
of statement its number is, and `basis_detail` carries the derivation with the
numbers substituted, so a reader can re-run it with their own inputs. Without this a
row could carry a TFLOPS value justified only in free-text notes, which is how eight
generated values once survived review — see gotcha `flops-row-had-no-basis-field`.
`unit` is only for precisions with several distinct hardware paths on one part
(e.g. `fp64-vector` vs `fp64-tensor-core`), so two rows at one precision are never
silently summed.

`fp64` and `int16` are enum members added 2026-10-04. Before them every FP64 row was
filed as `other`, indistinguishable from a row whose precision is unknown; `other` now
has zero rows in the repo.

`onchip_cache` was added 2026-10-05: on-chip cache capacity is not `vram_gb`, and it was
reachable only through free-text notes — which is what a wave-4 crawler was working around
when it put an L2 size into a cross-reference field (see `flop.hardware_parameters` for
that incident). Where the figure belongs is on the part that has it.

**Scope is deliberately narrow and every sub-field is nullable, because the on-chip
hierarchy is not comparable across vendors.** `kind` says what the number actually *is*,
and it is the field that stops a reader treating these as interchangeable:

| kind | meaning | example |
|---|---|---|
| `l2` | a true last-level cache, in `l2_kb` | RTX 5090: 98304 KB (96 MB) |
| `llc-in-front-of-hbm` | package-level LLC sitting *in front of* HBM, in `llc_mb` | MI300X / MI355X: 256 MB; MI350P: 128 MB |
| `sram-feed-tier` | on-die SRAM acting as an L2-like feed tier, in `sram_mb` | Tenstorrent Wormhole n150: 108 MB |

All three kinds are in use (1 / 3 / 1 records). A fourth, `weight-resident-sram`, was drafted
for Qualcomm AI100/AI200 and then **removed unused** — those vendor pages could not be
re-verified first-party in this pass, and a kind no record could honestly hold is the exact
failure mode this repo documents. Re-add it alongside a verified record, never before.
`unknown` and `null` have zero rows and exist only so a skeleton and a record whose
hierarchy could not be identified still validate, matching `paper.venue_track` and
`metric_exposure.metric_type`.

A hit in an `llc-in-front-of-hbm` tier is served at HBM-class latency, but on a miss the
data has still crossed HBM — so it is a **hit-rate multiplier, not a bandwidth multiplier**,
the opposite of what the number looks like. Never sum or compare these across vendors
without reading `kind` first.

`per_scope` is one of `per-package`, `per-die`, `per-card`, `per-xcd`, `null`, and it is
the easiest field on this record to get wrong. AMD's MI300X and MI355X datasheets and
brochures both print "4 MB shared L2 cache shared across CUs" *and* "256 MB Infinity Cache
shared across 8 XCDs" — the first is **per compute die**, the second **per 8-die package**.
They are different scopes and must never be added or compared as tiers of one cache;
recording the die figure as the package one is a 64x error.

`basis` accepts only first-party provenance (`vendor-spec-sheet`, `vendor-whitepaper`,
`vendor-docs`). A secondary aggregator is not recorded, because a cache size copied from an
aggregator is indistinguishable from one copied from the wrong column. **A null here means
"not verified first-party", never "zero."** That is why only 5 of 108 accelerator records
carry an `onchip_cache` row: AMD MI300X / MI350P / MI355X, the RTX 5090 and Tenstorrent
Wormhole were each verified against a spec sheet or whitepaper, and every other candidate was
left null rather than filled from a comparison table.

The traps are recorded per record in `source_note`, because each is a way to get a
plausible wrong number:

- **RTX 5090: 98304 KB, not 131072 KB.** NVIDIA's Blackwell whitepaper prints 96 MB in the
  per-SKU table and 128 MB for the full GB202 die in its appendix. The per-SKU figure is
  correct; the full-die row overstates the card's cache by 33%.
- **Wormhole: 108 MB vs 120 MB, unresolved.** The vendor's card table says 108 MB per n150;
  its current PCIe-cards page says "SRAM: 120 MB (1.5 MB per Tensix Core)" against 80 cores.
  80 × 1.5 = 120, so the two pages genuinely disagree and 120 does not reconcile with the
  arithmetic this repo's own flops rows use. **108 MB is kept**, because this record's flops
  rows and notes derive from it and swapping it here would silently desynchronise them. The
  conflict is recorded as open, not resolved by picking the newer page.
- **GB203 full-die vs SKU.** The same whitepaper prints 65536 KB for a full GB203 die while
  the 5070 Ti's per-SKU figure is 49152 KB. Those appendix tables do not bind a column to a
  SKU unambiguously enough to record without guessing, so `nvidia-rtx-5070-ti` and
  `nvidia-rtx-5080` stay null.

## flop

One file per operation class, e.g. `flops/prefill-attention.md`.

    class                 prefill_attention | decode_gemm | decode_attention |
                         moe_routing | moe_experts | norm_softmax | sampling |
                         embedding | speculative_draft | speculative_verify |
                         kv_transfer | quantization_overhead | custom_kernel
    arithmetic_intensity  string    description of bytes moved per flop at batch 1
    bound_by              enum      memory | compute | both | interconnect |
                         launch_overhead
    scales_with           string    which knob changes cost: batch | context_length | width | ...
    affected_by_hardware  [string]  accelerator ids where this class behaves unusually.
                         PRODUCTS, not parameters — this is a cross-reference field
    hardware_parameters   [ { parameter: enum, effect: string,
                              values_for: [string]|absent,
                              magnitude: string|null,
                              basis: recorded | derived | unverified | null } ]
    workarounds           [string]  short-form mechanisms that reduce this class
    notes

`affected_by_hardware` holds accelerator **record ids**. `hardware_parameters`, added
2026-10-05, holds the hardware **quantities** that move the class — and it exists because a
wave-4 crawler put `['memory_bandwidth', 'l2_cache_size', 'tensor_core_count']` into
`affected_by_hardware`, which is a cross-reference field. Those could never resolve, and
worse they silently consumed the dangling-reference budget that exists to catch new bad
references. A later pass pushed them into free-text notes, which made them unqueryable
prose; `qhw-decode-gemm-memory-bound` still carries the whole incident in its notes with
`affected_by_hardware` left correctly empty. A part-independent dependency belongs in
`hardware_parameters`, and that is where it went.

`parameter` is one of `memory_bandwidth`, `memory_capacity`, `l2_cache_size`,
`tensor_core_count`, `shared_memory_capacity`, `tmem_capacity`, `peak_flops`. **All seven
are in use**: 36 / 4 / 7 / 25 / 9 / 10 / 10 rows respectively, **101 rows across 65
records**, none empty.

A row is a **claim that this parameter moves this class's cost**, not a measurement of any
part, and two rules keep it honest:

- `magnitude` reproduces figures the record **already quoted**, taken from the sentence that
  makes the claim. Only 15 of 101 rows carry one. An earlier attempt harvested figures from a
  wider window and misattributed them — one row picked up a tensor-core rate as a
  shared-memory figure, another a PFLOPS number for a bandwidth claim — which is the
  plausible-but-wrong value this repo treats as worse than an absent one.
- `basis` is `recorded` where the record cites figures or names a concrete
  architecture / ISA / vendor statement, and `unverified` where it asserts the effect with
  nothing behind it. Current split: **39 recorded, 62 unverified**. The unverified majority
  is deliberate and is the honest reading — most of these records gesture at a hardware
  parameter in passing without sourcing it, and inflating that number would be inventing
  provenance.

Never add a row to fill a parameter member. A parameter with zero rows is a finding, not a gap.

## engine

    repo                  owner/name
    languages             [string]
    license               string
    first_release_year    int
    design_goal           [string]  e.g. throughput-first | low-latency | minimal-deps | multi-model
    scheduling            [string]  continuous-batching | paged-kv | radix-cache | none
    supported_backends    [string]  cuda | rocm | hip | metal | vulkan | cpu-avx512 | opencl | tpu | xnnpack
    supported_formats     [string]  gguf | safetensors | awq | gptq | exl2 | onnx | mxfp4 | ...
    notable_features      [string]
    hardware_caveats      [string]  free text: "no ROCm on gfx1100", "CUDA-only MoE path"
    best_for              [string]  the workloads it wins, in plain language
    avoid_for             [string]
    tracing_support       enum      none | basic | per-request | per-phase |
                       unknown | null. How far tracing goes. per-phase is the
                       only level that can attribute a value to prefill vs
                       decode, which is what explains a TTFT-vs-ITL regression
    verified_negatively   bool|null true only when the ABSENCE of a capability
                       was actively confirmed against a primary source. Without
                       it, 'looked and it is not there' and 'never looked' are
                       the same null, and a verified negative gets recorded as
                       prose instead of data
    docs_url
    notes

Per-metric detail lives in `metric_exposure` records, one per (engine, metric),
so a verbatim metric name is queryable instead of buried in `notable_features`.

## quantization

    scheme                68 members, listed in full in
                         schemas/quantization.schema.json; 63 are in use and 5 are
                         empty (gguf-fp16, quanto, gguf-iq3-xs, gguf-iq3-m,
                         gguf-tq2-0). Families:
                         weight-only: awq | gptq | exl2 | quarx | spinquant |
                         smoothquant | hqq | ternary | wq4a4 | llm-fp4 | aqlm
                         gguf k-quants: gguf-q2_k | gguf-q3_k_m | gguf-q4_0 |
                         gguf-q4_k_m | gguf-q4_k_s | gguf-q5_k_m | gguf-q6_k |
                         gguf-q8_0 | gguf-imatrix
                         gguf i/t-quants: gguf-iq1-s | gguf-iq1-m | gguf-iq2-xxs |
                         gguf-iq2-xs | gguf-iq2-s | gguf-iq2-m | gguf-iq3-xxs |
                         gguf-iq3-s | gguf-iq4-xs | gguf-iq4-nl | gguf-tq1-0
                         gguf legacy: gguf-q4-1 | gguf-q5-1
                         fp8 family: fp8-e4m3 | fp8-e5m2 | fp8-blockwise | mxfp8
                         int8 family: int8 | int8-channelwise
                         nvidia: nvfp4 | mxfp4 | mxfp6 | modelopt
                         research: bitnet-b158 | flute | lqer | quip | atom |
                         squeezellm | quarot | omniquant | zeroquant | qat
                         runtimes: bitsandbytes-nf4 | bitsandbytes-int8 |
                         optimum-quanto | quanto | ort-matmulnbits-int4 |
                         llmcompressor-compressed-tensors | mlc-q4f16
                         gguf unlisted members: gguf-fp16
                         kv-side: kv-cache-quant | kv-eviction | kv-sparsity | w4a8kv4

`scheme` is the one enum in this repo that is deliberately NOT enumerated inline
in full: the authoritative list is `schemas/quantization.schema.json` and the
families above are a reading aid, not a second source of truth. The families
cover every member as of 2026-10-05; `gguf-fp16` is called out separately because
the gguf rows above are deliberately k-quant/i-quant shaped and fp16 is the
plain-storage outlier.
    bits                  string    "4", "3.2 avg", "w4a16"
    weight_group_size     number|null
    activation_scheme     string|null
    native_support        [accelerator ids]  ids where the format has a hardware path
    emulated_support      [accelerator ids]  ids where it runs via dequant-to-fp16
    kernels               [string]  kernel names where known
    quality_delta         string    measured perplexity or benchmark delta vs fp16
    notes

## interconnect

    kind                  nvlink | nvswitch | infinity-band | xgmi | uefi |
                          infiniband | ethernet | roce | pcie | cxl | hstx |
                          uvm | sxm-c2c | xcd | waa | other
                          (the doc previously read `roe`; the schema member is
                          `roce` — RDMA over Converged Ethernet — and the typo had
                          been drifting since the member was added)
    version               string
    bandwidth_gbps        number     per link, unidirectional
    bandwidth_basis       string     how the bandwidth figure was arrived at
    null_bandwidth        enum|null  definitional | sourcing | verified-negative
    link_count            number|null
    topology              string
    scale_up              bool
    scale_out             bool
    switching             string
    notes

## benchmark

    engine_id             string
    accelerator_ids       [string]
    interconnect_ids      [string]
    model                 string
    format_id             string|null
    metric                enum      decode_tok_s | prefill_tok_s | ttft_ms | tps_aggregate |
                               itl_ms | memory_gb | tok_s_per_user | quality |
                               joules_per_token | speedup_ratio | dimensionless_ratio
    value                 number
    unit                  string
    methodology           string    batch size, prompt len, output len, concurrency
    measured_by           enum      vendor | third_party | self
    reproducible          bool
    energy_basis          string|null   the measurement window behind a joules
                               figure: 'post-ramp core of 3634 1Hz samples, first
                               5% discarded'
    power_w               number|null   mean watts underlying the joules figure
    power_scope           enum      node | device | rack | null. An 8-GPU NODE
                               figure is NOT a per-GPU figure. Always state which.
                               node: 6 records, device: 2. `rack` has ZERO and that
                               is a VERIFIED NEGATIVE — every rack-scale result here
                               (CoreWeave GB200/GB300 NVL72, v6.1) has_power=false,
                               so no rack draw exists to record
    power_cap             number|null   the W cap ENFORCED during the run. NOT the
                               datasheet TDP, and NOT an MLPerf 'MaxQ' result (see
                               gotcha mlperf-maxq-is-not-a-watt-cap)
    clock_lock_mhz        number|null   the -lgc value if clocks were locked; null if
                               unlocked. A locked-clock number is not a stock-boost
                               number
    thermal_state         enum      steady | unknown | null. 'steady' only if the run
                               held a thermal plateau — a short run measures the ramp
                               and biases energy LOW
    notes

`joules_per_token`, `energy_basis`, `power_w`, `power_scope`, `power_cap`, `clock_lock_mhz`
and `thermal_state` were all drafted and shipped **unused** — zero records set them — while
watts and joules travelled as prose inside `unit`. They were populated on 2026-10-05 from
facts the records already documented: `energy_basis`, `power_scope` and `thermal_state` now
appear on **8 records each**; `power_w` and `clock_lock_mhz` on **4** (the four MLPerf MaxQ
node-scope runs, all locked at 1000 MHz); `power_cap` on **2**.

`power_cap` is deliberately null on the H200 records: 700 W is the factory TGP, a static
datasheet limit, not a measured cap, and filling it would assert the opposite of what those
records' own notes argue.

`power_w` is null wherever no mean draw was integrated. A throughput-under-a-cap submission
declares a cap but publishes no trace, so recording that cap in `power_w` would claim the
part drew exactly its limit — the conflation `mlperf-maxq-is-not-a-watt-cap` exists to
prevent. A null here means "not measured", never "zero".

`metric` gained `speedup_ratio` and `dimensionless_ratio` on 2026-10-05. The enum enumerates
rates and absolute quantities plus one catch-all, so a **dimensionless ratio had nowhere to
live** and was filed under whichever rate was the thing being compared. That is not
cosmetic — it is selected by exactly the filter a capacity planner uses.
`sd-medusa-batch32-degradation` carried value 0 as `decode_tok_s` (0 reads as a failed run
when it is the floor of a documented degradation curve), and
`ev-llamacpp-spec-bench-replay-inflates` carried 13.4 as `tps_aggregate` — an inflation factor
that reads as a respectable per-GPU figure and would be *quoted* as one. **5 records now use
`speedup_ratio` and 10 use `dimensionless_ratio`**, every stored value unchanged.

**The two ratio members are not interchangeable.** `speedup_ratio` means "N times faster than
a *named* baseline" and **must name that baseline in `unit`** — a bare ratio with no baseline
is the single most misleading number in this domain, the same rule `paper.speedup_reported`
follows. `dimensionless_ratio` covers ratios with no faster/slower reading: PUE (1.145 means
the facility draws 14.5% more than its IT load) and idle/maximum power dynamic range. PUE 2.5
is not a 2.5x speedup, and routing it to `speedup_ratio` would have replaced one wrong answer
with another — which is why they are separate members and why the ten PUE and power records
went to `dimensionless_ratio`.

`quality` remains the weakest key in this enum: it is still a catch-all spanning accuracy
deltas in percentage points, perplexity deltas and acceptance lengths. That is a separate,
still-open gap.

Two `metric` members remain unused and are documented rather than filled.
`joules_per_token` has **zero records**, and the reason is a modelling limitation rather than
missing data: one benchmark record carries one `(metric, value)` pair, while every joules
figure here is measured *alongside* a throughput on the same run — the four MLPerf MaxQ
records and the two Apple powermetrics records each state both. Re-labelling one of those
would destroy the independently measured throughput that the throughput-at-a-cap comparison
set depends on. The correct fix is a **second record per run** carrying the joules, and that
was deliberately not done in a schema pass, because it means minting new record ids
asserting derived energy figures. Until then the joules live in `energy_basis` (8 records,
with the sample window and derivation written out) on the record that measured them.
`memory_gb` has **zero records**: nothing here measures a peak resident footprint as its
primary quantity — memory figures sit in `methodology` and `notes` on KV-cache records
instead. Both were empty before this pass and are empty after it; what changed is that the
reason is written down where the next author will read it.

## paper

A research paper, recorded because this repo is an inference-research repo and
most of the field's progress is published as papers long before it ships in an
engine. Distinct from `source`: a `source` is a document you cited, a `paper` is
a contribution with a method, a claim, and an adoption status.

    arxiv_id             string|null   e.g. 2408.11743
    venue                arxiv-preprint | neurips | icml | iclr | acl | emnlp |
                         naacl | cvpr | iccv | eccv | osdi | sosp | nsdi | atc |
                         eurosys | asplos | micro | isca | mlsys | vldb | sigmod |
                         kdd | www | interspeech | icassp | colm | sigcomm |
                         cais | sigir | cikm | ppopp | sc22 | facct | enlsp |
                         tmlr | taslp | jmlr | acm-tos | nature | other
    venue_kind           conference | journal | null
                         distinguishes conference from journal publication venue
    year                 int|null
    category             attention | kv-cache | quantization | moe | ssm |
                         speculative-decoding | serving-systems | scheduling |
                         distillation | long-context | position-encoding |
                         sparsity | pruning | routing | inference-time-compute |
                         agentic | multimodal | speech | embedding | retrieval |
                         structured-output | security | cluster | interconnect |
                         training-inference-bridge | other
    authors              [string]
    affiliations         [string]
    problem              string|null   the bottleneck it attacks, one sentence
    mechanism            string|null   how it works, concrete enough to implement
    hardware_relevance   [string]      bare accelerator ids where the benefit is
                         hardware-dependent
    speedup_reported     string|null   the paper's OWN claim, always with baseline
                         and settings. Never a bare number — a "3x speedup" with
                         no baseline is worse than null
    venue_track          main | findings | workshop | industry | unknown | null
                         which track WITHIN the venue. Exists because ACL-family
                         Findings and EACL were under-claimed: a Findings paper
                         has no honest value in `venue`, and `other` cannot
                         separate 'second-tier venue' from 'venue not yet known'.
                         Additive — `venue` is untouched, so no existing paper
                         record can break
    adoption             in-production | in-upstream-engine | research-only
                         abandoned | unknown
    code_url             string|null
    open_weights         bool|null
    notes

`adoption` is the field that separates a citation from an instruction. A method
that is research-only and a method merged into vLLM differ enormously in what you
should do, and the paper's own framing hides that. `speedup_reported` must
always carry its baseline — self-reported speedups without a stated baseline are
the single most misleading number in this domain.

`sigcomm` and `cais` were added 2026-10-05. Before them a SIGCOMM or ACM CAIS paper had no
honest `venue` value, and its record was forced to `other` with the real venue in free-text
notes. Both were verified against the publisher's own Crossref deposit before being added,
never against an arXiv comment string: SIGCOMM '25 via DOI 10.1145/3718958.3750506 (pages
592-608, published-print 2025-09-08) and CAIS '26 via DOI 10.1145/3786335.3813124 (pages
1009-1022, 2026-05-26). `sigcomm` is a conference and is **not** interchangeable with the
existing `sigmod` member, which is a different ACM Special Interest Group. Three records
were reclassified in the same commit: `w4p-megascale-infer` (SIGCOMM '25), `w4p-xgrammar2`
(CAIS '26) and `paper-cachegen` (SIGCOMM '24, DOI 10.1145/3651890.3672274).

`other` is not one state. It still holds 18 papers spanning at least nine distinct venues —
SIGIR, CIKM, TMLR, SC22, TASLP, JMLR, FAccT, PPoPP, EACL/ENLSP, and one journal-of-record
rather than a conference — so a reader filtering `venue=other` still cannot separate
"published somewhere I could not name" from "venue not yet known". The survey, with a
per-record provenance and an explicit verified / not-verified verdict for each, is recorded
in `schemas/paper.schema.json` under `x-notes-other-venue-survey-2026-10-05` so the next
pass does not re-derive it. The shortest version of the lesson: `other` was mostly a
**provenance** problem, not a vocabulary one. Ten of those eighteen papers name a real,
verifiable venue that simply had no enum member, and several of those deposits are already
sitting in `data/sources/`. The two best-evidenced next members are `sigir` (ColBERT and
SPLADE, both Crossref-confirmed) and `eacl` (MTEB, crossref-confirmed) — but a journal member
(`tmlr`, `taslp`, `jmlr`) would need a decision about journals versus conferences first,
since every current member is a conference. One record (`mooncake-kimi`) is a
journal-of-record at ACM Transactions on Storage, not a systems conference, and the
`osdi` '25 paper sharing its name is a different system by different authors.

## model

A specific released model, recorded for what it does to the FLOP mix. The
fields here exist because architecture decides which cost dominates — a dense
GQA model, a DeepSeek-style MoE with MLA, and a hybrid Mamba model stress
completely different hardware.

    family              string|null   release family: llama, qwen, deepseek, ...
    vendor              string|null
    release_year        int|null
    architecture        dense | moe | hybrid-mamba | hybrid-attention-ssm |
                       ssm | recurrent | diffusion | other
    params_b            number|null   total parameters, billions
    active_params_b     number|null   activated per token; equals params_b for dense
    num_experts         int|null
    experts_per_token   int|null
    shared_expert       bool|null     always-on shared expert adds to per-token active
    hidden_size         int|null
    num_layers          int|null
    num_kv_heads        int|null
    head_dim            int|null
    kv_lora_rank        int|null    MLA latent KV rank (DeepSeek-V2/V3: 512). The
                       width actually cached, in place of num_kv_heads * head_dim.
                       Null for non-MLA models. This is the field that makes
                       "DeepSeek vs GLM-4.5 KV cache" a query instead of a note
    qk_rope_head_dim    int|null    MLA decoupled rotary sub-head (64). Added to
                       kv_lora_rank for the per-token KV size
    qk_nope_head_dim    int|null    MLA non-positional QK sub-head (128)
    v_head_dim          int|null    MLA value sub-head. Equals qk_nope_head_dim in
                       DeepSeek-V2/V3, which is why MLA caches
                       (kv_lora_rank + qk_rope_head_dim) and NOT 2 * v_head_dim
    kv_compression      string|null per-layer KV compression when non-uniform.
                       V4-style CSA gives selected layers a lower ratio than the
                       HCA baseline, so one average hides the real peak
    attention_layer_indices [int|null]  which layers are full attention, for
                       hybrids. Only these layers carry a KV cache
    layer_types         [string|null] per-layer type in order; length should
                       equal num_layers when present
    gqa_ratio           number|null    attention heads / kv heads
    max_position_embeddings int|null
    embedding_dims       int|null    output embedding dimensionality; null for
                         autoregressive LMs, which have no fixed output vector
    embedding_pooling    cls | mean | last | max | none | null
                         pooling that produces the embedding vector. Getting this
                         wrong silently degrades retrieval quality with no error.
                         16 records set embedding_dims; `embedding_pooling` is STILL
                         UNUSED across all 155 model records — an open gap, since
                         dims alone does not tell a reader whether vectors are CLS
                         or mean pooled
    context_scaling     string|null   free text, NEVER an enum: 'verified
                       native', 'mrope', 'llama3 rope scaling' and more are in
                       100+ existing records and an enum here would break every
                       one of them at once
    context_scaling_method  enum|null   native | none-verified | pi | ntk-aware |
                       yarn | self-extend | llama3 | mrope | linear | other |
                       unknown | null. The queryable CLASS; keep the detail in
                       context_scaling. null = never filled in; unknown = looked
                       and could not establish
    activation          string|null   silu, gelu, swiglu, geglu
    attention_variant   string|null    standard, MLA, GQA, MQA, sliding-window/
                       hybrid, NSA, lightning attention
    kv_cache_bytes_per_token   number|null   the single most decision-relevant
                       serving number; show the derivation in notes
    flops_per_token_active     number|null   approx 2 * active_params_b * 1e9
    open_weights        bool|null
    license             string|null
    notes

Two rules that matter here. First, `params_b` and `active_params_b` must not be
conflated for MoE — a 671B DeepSeek serving 37B active per token is a different
serving problem from a dense 70B, and only one of them is the memory bottleneck.
Second, `kv_cache_bytes_per_token` is what determines whether a model fits at
all: at 128k context a 70B GQA model and a 671B MoE can differ by an order of
magnitude in resident KV despite the MoE having fewer active params. Derive it
from `num_layers * num_kv_heads * head_dim * 2 (K and V) * bytes_per_element`
and put that arithmetic in `notes`.

That GQA formula does NOT apply to MLA. For an MLA model the per-token KV is
`num_layers * (kv_lora_rank + qk_rope_head_dim) * bytes_per_element`, which is
why `kv_lora_rank` and `qk_rope_head_dim` exist as fields: with only
`head_dim` recorded, the two families cannot be told apart and a reader
recomputing the figure gets an answer several times too large. When a model has
per-layer compression, `kv_compression` overrides the uniform formula and the
derivation in `notes` must show which layers.

`benchmark.model` predates this type and stays free text. Where a benchmark
names a model that has a record here, note the pairing rather than rewriting
the benchmark.

## supply

Where a given accelerator can actually be obtained. One record per vendor or
channel, not per SKU — the same integrator may carry many parts, and a record
listing them is still queryable.

    kind                  oem-direct | enterprise-distributor | used-market |
                         auction | cloud | integrator | regional-reseller |
                         broker | self-build | unknown
    vendor                string|null   who ultimately supplies the part
    accelerator_ids       [string]      bare record ids obtainable here
    region                string|null   geographic market served
    channels              [string]      direct-quote | reseller | marketplace |
                         auction | rental | on-prem-cloud | contact-sales
    price_usd             number|null
    price_basis           string|null    what the price covers: bare card, SXM
                         module, full 8-GPU server, annual rental, per-hour
                         cloud, incl. tax/shipping. MANDATORY whenever
                         price_usd is set — an H100 quote means wildly
                         different things at these bases.
    availability          in-stock | lead-time | backorder | allocation-only |
                         discontinued | unknown
    lead_time_weeks       number|null
    lead_time_source      enum      published | midpoint-inference |
                         structural-na | unknown | null. Provenance of the
                         number above, which is otherwise unreadable when null.
                         structural-na means the number does not apply because
                         the channel is structurally instant (on-demand cloud,
                         peer-to-peer marketplace, spot capacity) — NOT the same
                         as 'nobody publishes it'
    lead_time_basis       string|null    why the figure is what it is, or why it
                         is null: measurement window, quote-date range, and for
                         structural-na the reason no number applies
    export_controlled     bool|null      restricted by region/entity. The boolean
                         ALONE is not actionable — see the two fields below
    export_control_regime string|null   the actual citation: 'EAR 3A090.a',
                         'EAR 3E001', 'EAR 744.23 (Entity List)'. Free text on
                         purpose: these are not a closed set and a wrong enum
                         value here is a compliance error. Cite the paragraph,
                         not a country list
    export_control_note   string|null    the MECHANISM: which licence or
                         exception, self-declared or verified at sale, whether
                         destination and end-user screening happens, who signs.
                         This is what separates 'formally restricted, sold to
                         anyone with a credit card' from 'genuinely screened'
    notes

Supply records are the most perishable data in this repo. A price without an
`updated` date is noise. Re-verify before trusting any figure older than a
quarter.

## gotcha

    class                driver | kernel | framework | config | hardware | format
                         toolchain | measurement | build | operations |
                         security | privacy | compliance
                         operations = day-to-day running/scaling/housekeeping
                         (an OOM that only appears at 90% utilisation);
                         security = attack surface or vulnerability; privacy =
                         data handling, redaction, what the engine logs about
                         prompts; compliance = licence, export, regulatory
    affects              [string]   record ids only: engine, accelerator,
                                    quantization or interconnect ids
    concepts             [string]   free-text scope markers that are NOT record
                                    ids: decoding/scheduling concepts (kv-cache,
                                    prefix-caching, speculative-decoding,
                                    tensor-parallel), software stacks (rocm,
                                    sycl, oneapi, vulkan, cuda, triton, pytorch),
                                    model names, architecture codenames
                                    (hopper-sm90, gfx1100). Not id-resolved.
    symptom              string
    root_cause           string
    workaround           string
    severity             enum      blocker | major | minor
    notes

## compiler

A compiler, DSL, or code-generation stack for inference — the layer between a model definition and the GPU kernel.

    repo                  owner/name
    language              string    primary implementation language
    license               string
    first_release_year    int
    category              dsl | compiler-framework | template-library | graph-optimizer |
                          kernel-dsl | runtime
    target_backends       [string]  cuda | rocm | hip | metal | vulkan | cpu-avx512 |
                          cpu-avx2 | opencl | tpu | xnnpack | spirv | webgpu
    input_languages       [string]  python | jax | pytorch | tensorflow | onnx |
                          tflite | cutlass-cpp | triton | mlir | xla-hlo
    output_artifacts      [string]  ptx | amdgcn | spirv | air | metal | c | llvm-ir |
                          triton-ir | cubin | plan
    compilation_strategy  jit | aot | template-instantiation | graph-rewrite | mixed
    autotuning            bool
    production_ready      bool
    production_evidence   [string]  engine names or projects that use this in production
    strengths            [string]
    weaknesses            [string]
    learning_curve        low | moderate | high | very-high
    compile_time_seconds  number|null  typical compile time for a single kernel
    notes

## metric_exposure

One record per (engine, metric): how that engine exposes one metric and how to
turn it on. This type exists because verbatim metric names used to live in
`engine.notable_features` prose, which made them unqueryable — the single reason
the observability slice had to write metric inventories as sentences.

    engine_id            string     bare engine record id this metric belongs to
    metric_name          string     EXACTLY as emitted, namespace and unit suffix
                       included (vllm:time_to_first_token_seconds). A paraphrase
                       makes the record useless for detection work
    metric_type          enum      counter | gauge | histogram | summary |
                       log-only | unknown | null
    unit                 string|null   seconds, bytes, requests, tokens, 1 (ratio)
    exposure             enum      prometheus | opentelemetry | statsd |
                       http-json | otlp-endpoint | python-api | stdout-log |
                       callback-hook | file | cli | unknown | null. http-json and
                       stdout-log are the two a scraper matches by regex and
                       breaks on a version bump
    endpoint             string|null   '/metrics', ':9464/metrics'
    histogram_buckets    [number]|null  bucket upper bounds in the metric's own
                       unit. A histogram with unknown buckets gives quantiles
                       that are wrong rather than absent
    labels               [string]      label keys. Cardinality decides whether
                       the metric works per-model or only fleet-wide
    enabled_by_default   bool|null     false means someone has to turn it on —
                       the most common reason an 'it does not report X' claim
                       is wrong
    phase_scoped         bool|null     attributable to prefill or decode
                       separately rather than to the whole request
    verified_negatively  bool|null     the ABSENCE was actively confirmed at the
                       engine version named in notes. Without it, 'checked' and
                       'never checked' are the same record
    notes

`metric_type` and `exposure` are required, but nullable, so a generated skeleton
validates immediately. `new_record.py` seeds both with `unknown`.

## source

    url                  string
    publisher            string
    kind                 enum      spec-sheet | whitepaper | paper | benchmark | repo |
                              release-notes | forum | blog | interview | review |
                              database
    published            date|null
    accessed             date
    archived_url         string|null
    vendor_silence       enum|null export-control | paywall | aggregator-only | no-config.json
    notes

`database` (156 records) was missing from this doc until 2026-10-05 — SCHEMA.md had drifted
from the schema, the failure this repo has already hit once. It matters because a `database`
source is the highest-grade venue evidence available: it is how Crossref, OpenReview and DBLP
deposits get cited, and both `paper.venue` members added in 2026-10-05 (`sigcomm`, `cais`)
were verified against one rather than against an arXiv comment string. `interview` still has
zero records.