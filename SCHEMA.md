# Schema reference

JSON Schema per record type lives in `schemas/`. This is the human version.

## Common fields (all types)

| field | type | meaning |
|---|---|---|
| `id` | string | stable kebab-case slug, globally unique, matches filename |
| `type` | enum | accelerator, flop, engine, quantization, interconnect, benchmark, gotcha, source |
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
    flops                  [ { precision: fp32|bf16|fp16|tf32|fp8|fp6|fp4|int8|int4,
                                tflops: number, dense: bool, source_id: string } ]
    tdp_w                  number     per device or per accelerator in a package
    interconnect           [string]   interconnect ids this device exposes
    unified_memory         bool
    consumer                bool
    notes

`flops` is a list because precision matters more than the headline number. Always
record whether a figure is dense or sparse — sparse counts are not comparable and
marketing conflates them.

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
    docs_url
    notes

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
    export_controlled     bool|null      restricted by region/entity; name the
                         regime in notes
    notes

Supply records are the most perishable data in this repo. A price without an
`updated` date is noise. Re-verify before trusting any figure older than a
quarter.

## gotcha

    class                driver | kernel | framework | config | hardware | format |
                         toolchain | measurement
    affects              [string]   engine ids, accelerator ids, or format ids
    symptom              string
    root_cause           string
    workaround           string
    severity             enum      blocker | major | minor
    notes

## source

    url                  string
    publisher            string
    kind                 enum      spec-sheet | whitepaper | paper | benchmark | repo |
                              release-notes | forum | blog | interview | review
    published            date|null
    accessed             date
    archived_url         string|null
    notes