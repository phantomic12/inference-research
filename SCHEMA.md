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
                          tenstorrent | cambricon | other
    architecture           e.g. hopper, blackwell, gfx1100, arc-b-series, a100, m-series
    release_year           int
    process_nm             int, die node
    form_factors           [string]   pcie | sxm | oam | mdu | mcm | socs
    vram_gb                number     per-device or per-package, unit in field name
    memory_type            string     hbm3e | gddr6x | lpddr5x | ...
    memory_bus_bit         int
    memory_bandwidth_gbps   number
    memory_bandwidth_basis  string     how the bandwidth number was arrived at
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

## flop

One file per operation class, e.g. `flops/prefill-attention.md`.

    class                 prefill_attention | decode_gemm | decode_attention |
                         moe_routing | moe_experts | norm_softmax | sampling |
                         embedding | speculative_draft | speculative_verify |
                         kv_transfer | quantization_overhead | custom_kernel
    arithmetic_intensity  string    description of bytes moved per flop at batch 1
    bound_by              enum      memory | compute | both | interconnect
    scales_with           string    which knob changes cost: batch | context_length | width | ...
    affected_by_hardware   [string]  accelerator ids where this class behaves unusually
    workarounds           [string]  short-form mechanisms that reduce this class
    notes

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

    scheme                awq | gptq | exl2 | gguf-q4_k_m | fp8 | nvfp4 | mxfp4 | int8 |
                         bitsandbytes | smoothquant | hqq | quarx | spinquant | ternary | 2bit
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
                          infiniband | ethernet | roe | pcie | cxl | hstx | uvm
    version               string
    bandwidth_gbps        number     per link, unidirectional
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
    metric                enum      decode_tok_s | prefill_tok_s | ttft_ms | tps_aggregate | itl_ms | memory_gb
    value                 number
    unit                  string
    methodology           string    batch size, prompt len, output len, concurrency
    measured_by           enum      vendor | third_party | self
    reproducible          bool
    notes

## paper

A research paper, recorded because this repo is an inference-research repo and
most of the field's progress is published as papers long before it ships in an
engine. Distinct from `source`: a `source` is a document you cited, a `paper` is
a contribution with a method, a claim, and an adoption status.

    arxiv_id             string|null   e.g. 2408.11743
    venue                arxiv-preprint | neurips | icml | iclr | acl | emnlp |
                         naacl | cvpr | iccv | eccv | osdi | sosp | nsdi | atc |
                         eurosys | asplos | micro | isca | mlsys | vldb | sigmod |
                         kdd | www | interspeech | icassp | colm | other
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
                              release-notes | forum | blog | interview | review
    published            date|null
    accessed             date
    archived_url         string|null
    notes