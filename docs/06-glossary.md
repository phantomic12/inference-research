# Glossary

One line per term, defined the way this repo uses it, with the records where the
repo actually discusses it. Every id below resolves
(`python tools/query.py get <id>`); no number appears here without a record
carrying it.

This is a practitioner vocabulary, not a textbook. Where the industry uses a term
loosely, the definition given is the one the repo's records are consistent with,
and the record list is where the definition comes from.

## Serving metrics

**Prefill** — the prompt-processing phase that runs the whole input through the
model once before any token is emitted; compute-bound, and its rate is what sets
time to first token. [[flops/prefill-attention]] [[flops/decode-gemm]]

**Decode** — the token-generation phase, run one step at a time; memory-bound at
small batch, which is why it is the phase that rewards batching and KV-cache
engineering. [[flops/decode-gemm]] [[flops/decode-attention]]

**TTFT (time to first token)** — wall-clock latency from request submission to
the first output token. Dominated by prefill, and the one metric tensor
parallelism reliably improves. [[benchmarks/mpt7b-a100-bs1-ttft-ms]]

**ITL (inter-token latency)** — the interval between consecutive output tokens
after the first. In this repo some `itl_ms` records hold end-to-end time-to-last-
token instead; read the `unit` field, not the metric name.
[[benchmarks/mi300x-vllm-llama31-70b-fp8-tp8-ttlt-batch1]]

**TPOT (time per output token)** — mean per-token generation time, i.e. steady-
state ITL. See "terms with no coverage yet".

**TPS** — tokens per second. Meaningless unqualified: it is either *per user* or
*aggregate across all in-flight requests*, and the two differ by the concurrency.
This repo has one hardware configuration measured both ways — aggregate 13.9x
higher at batch 64 while per-user falls 4.6x.
[[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]]
[[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]]
[[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]]

**TTLT (time to last token)** — end-to-end request latency, i.e. TTFT plus all
decode steps. [[benchmarks/mi300x-vllm-llama31-70b-fp8-tp8-ttlt-batch1]]

**Goodput** — throughput measured only for requests meeting a latency SLO, which
is the metric disaggregated serving is designed to raise. Discussed in
[[engines/distserve]]; no benchmark record reports a measurement.

## Scheduling and cache management

**Paged KV / PagedAttention** — storing the KV cache in fixed-size blocks that
map to non-contiguous device memory, so allocation is on demand and sequence
length does not have to be reserved in advance. [[flops/kv-transfer]]

**Radix cache / prefix cache** — a prefix tree over token sequences so a request
sharing a system prompt or few-shot prefix with a previous one reuses that KV
instead of recomputing it. [[flops/prefix-cache-hit-miss]]

**Prefix-cache hit rate** — the fraction of a request's KV that came from cache
rather than recomputation. Speculative decoding measurably erodes it.
[[gotchas/eagle-prefix-cache-last-block-drop]]

**Chunked prefill** — splitting a long prompt into pieces that interleave with
decode steps, so a large prefill cannot monopolize the batch and starve running
sequences. [[flops/chunked-prefill-interleaving-scheduling]]

**Continuous batching** — adding and removing requests from the running batch at
every step, rather than waiting for a fixed batch to drain. The single largest
throughput lever in a serving engine. [[flops/continuous-batching-scheduling]]

**Disaggregated serving (prefill/decode disaggregation)** — running prefill and
decode on separate pools of GPUs so each phase uses hardware suited to it and KV
is transferred between them. Raises throughput, adds KV-transfer cost.
[[flops/prefill-decode-disaggregation-kv-handoff]] [[engines/distserve]]

**KV-cache quantization** — storing K and V in fewer bits than the weights.
Independent of weight quantization and often applied to the same model.
[[quantization/llm-qat-kv-cache]]

**KV eviction** — discarding cached KV under pressure, keeping a predicted-useful
subset. [[quantization/h2o-kv-eviction]]

## Attention variants

**MHA** — multi-head attention, one KV head per query head; largest KV footprint.

**GQA (grouped-query attention)** — several query heads share one KV head,
cutting KV cache proportionally. [[flops/modern-attention-variants]]
[[models/gpt-oss-20b]] (8 KV heads across 24 layers)

**MQA (multi-query attention)** — all query heads share a single KV head; the
extreme case of GQA, smallest KV cache.
[[flops/flashdecoding-split-kv-decode]]

**MLA (multi-head latent attention)** — compressing KV into a low-rank latent
space and caching that instead of the full K/V, so KV cache shrinks far more than
GQA can at comparable quality. [[flops/mla-latent-attention]]
[[models/deepseek-v3]] (kv_lora_rank 512, 128 heads expanded from latent)

**Sliding-window attention** — each token attends only to the previous W tokens,
so KV cost is independent of context length. Usually interleaved with a few full-
attention layers, which is what makes a hybrid model necessary rather than merely
optimized. [[flops/sliding-window-hybrid-attention]]

**Cross-attention KV sharing** — in encoder-decoder and multimodal models,
sharing or caching the encoder-side KV across decoder steps so it is computed once.
[[models/whisper-large-v3]] [[models/parakeet-ctc-1-1b]]

**Split-KV decode / FlashDecoding** — splitting the KV dimension across SMs in
decode so the bandwidth-bound phase uses more of the machine.
[[flops/flashdecoding-split-kv-decode]]

**Attention sink** — the tendency of a small set of early positions to absorb
disproportionate attention; the basis of most KV-sparsity methods.
[[quantization/duoattention-kv-sparsity]] [[quantization/h2o-kv-eviction]]

## AMD multi-die terms

These are the terms AMD's own architecture notes use, and they appear throughout
the MI2xx/MI3xx records.

**GCD (GPU Complex Die)** — the accelerator compute tiles on an XCD.

**XCD (eXpandable Complex Die)** — one GCD plus its HBM stacks and I/O; an
individual accelerator complex. [[accelerators/amd-instinct-mi300x]]
[[accelerators/amd-instinct-mi300a]]

**GIL (GCD-to-Infinity Link)** — the die-to-die interconnect *within* a package,
distinct from XGMI which runs *between* packages. [[accelerators/amd-instinct-mi300x]]
[[accelerators/amd-instinct-mi355x]]

## Sparsity and interconnect

**Dense vs 2:4-sparse** — 2:4 keeps 2 of every 4 adjacent weights in a fixed
pattern the tensor cores can skip; a 2:4 figure and a dense figure are not
comparable and vendor material conflates them. [[accelerators/nvidia-b200]]
[[accelerators/amd-instinct-mi210]]

**Structured sparsity** — a sparsity pattern with hardware-visible regularity
(2:4 blocks, block-sparse rows), as opposed to unstructured pruning a general
GEMM cannot exploit. [[accelerators/amd-instinct-mi210]]
[[accelerators/nvidia-rtx-5090]]

**NVLink / NVSwitch** — NVIDIA's scale-up domain; NVLink is the link, NVSwitch the
switch that makes many links act as one. [[interconnect/nvlink-3]]
[[interconnect/nvlink-4]] [[interconnect/nvlink-5]] [[interconnect/nvlink-6]]
[[accelerators/nvidia-gb200-nvl72]]

**XGMI** — AMD's between-package die-to-die/coherent interconnect.
[[interconnect/amd-xgmi]] [[accelerators/amd-instinct-mi300x]]

**HCCS** — Huawei's chip-to-chip scale-up interconnect. [[interconnect/huawei-hccs]]

**ICI** — Google TPU's interconnect fabric, its scale-up and scale-out domain.
[[interconnect/google-tpu-ici]]

**PCIe** — the host-attached link; much slower than a scale-up fabric, which is
why form factor (SXM vs PCIe) changes performance and not just price.
[[interconnect/pcie-gen5]] [[accelerators/nvidia-h100-pcie]]

## Quantization notation

**W4A16** — 4-bit weights, 16-bit activations: weight-only, the usual choice
because it costs no accuracy and needs no activation calibration.
[[quantization/awq]] [[quantization/gptq]]

**W4A4** — 4-bit weights *and* activations; much harder, and mostly research
kernel work rather than a silicon path. [[quantization/gguf-q4-k-m]]

**W8A8** — 8-bit both ways, via outlier migration rather than naive rounding.
[[quantization/smoothquant]]

**MXFP4 / NVFP4** — microscaling 4-bit float formats with per-block scales;
MXFP4 shares one scale across 32 elements, NVFP4 uses 16 plus a per-tensor scale.
[[quantization/mxfp4]] [[quantization/nvfp4]]

**GGUF** — llama.cpp's container and quantization format; the "K-quants" are
per-tensor recipes, not global bpw targets. [[quantization/gguf-q4-k-m]]

**imatrix (importance matrix)** — a calibration matrix of per-input-channel
activation magnitudes used by llama.cpp to decide which weights matter most, so
higher importance can get more bits. [[quantization/gguf-imatrix]]

## The roofline model

**Roofline** — attainable performance = min(peak FLOPs, memory bandwidth x
arithmetic intensity). [[sources/flop-roofline-williams-2008]]
[[sources/flop-llm-inference-roofline-survey]]

**Arithmetic intensity** — FLOPs per byte moved; the x-axis position of a kernel
on the roofline. [[flops/prefill-attention]]

**Ridge point / balance point** — the intensity at which a kernel becomes
compute-bound rather than bandwidth-bound; set by the machine's FLOPs-to-bytes
ratio. [[flops/decode-gemm]] [[accelerators/nvidia-h100-sxm]]

**Memory-bound vs compute-bound** — whether a kernel is limited by bytes or by
FLOPs. Decode is memory-bound at small batch; prefill is compute-bound.
[[flops/decode-gemm]] [[flops/prefill-attention]]

## Memory

**HBM** — high-bandwidth memory on-package with the GPU; the reason datacenter
accelerators have both high FLOPs and high bandwidth.
[[accelerators/nvidia-h100-sxm]] (HBM3, 80GB, 3350 GB/s)
[[accelerators/amd-instinct-mi300x]] (hbm3, 192GB, 5300 GB/s)

**GDDR** — graphics DRAM, cheaper and lower-bandwidth than HBM; the consumer and
lower-end accelerator path. [[accelerators/intel-arc-pro-b60]]

**LPDDR** — low-power DRAM, for APUs, Apple silicon and Grace-class memory.
[[accelerators/apple-m4-max]]

**Unified memory / UMA** — one address space for CPU and accelerator, removing
an explicit copy at the cost of bandwidth. [[accelerators/apple-m4-max]]
[[accelerators/amd-instinct-mi250x]]

**MIG** — Multi-Instance GPU, hardware partitioning of one GPU into isolated
smaller instances with their own memory and compute. [[accelerators/nvidia-h100-sxm]]

## Parallelism

**Tensor parallelism (TP)** — splitting individual matrix multiplications across
GPUs; requires all-reduce per layer, so it is interconnect-bound.
[[flops/tensor-parallel-allreduce]]

**Pipeline parallelism (PP)** — splitting layers across GPUs; needs a
micro-batch in flight to avoid bubbles. [[engines/vllm]]

**All-reduce** — the collective every tensor-parallel layer ends with; on the
critical path once per layer. [[flops/tensor-parallel-allreduce]]

**Expert parallelism (EP)** — the MoE analogue of TP, distributing experts across
GPUs. [[engines/sglang]]

## MoE

**MoE (mixture of experts)** — a model with many parallel expert FFNs of which
only a few run per token. [[models/mixtral-8x22b]]

**Active parameters** — the parameters actually computed per token. For a 671B
DeepSeek-style model it is ~37B, so the serving problem is KV and expert
placement, not dense FLOPs. [[models/deepseek-v3]]

**Expert routing** — selecting which experts a token goes to; top-k selection is
the standard mechanism. [[flops/moe-routing]]

**Shared expert** — an always-on expert applied to every token, which adds to
per-token active parameters and is not optional. [[flops/moe-experts]]

**MTP (multi-token prediction)** — training the model to predict several future
tokens per forward pass, used as a zero-draft speculative scheme.
[[gotchas/mtp-plus-prefix-caching-accuracy-loss]] [[engines/sglang]]

**Speculative decoding** — drafting several tokens cheaply then verifying them in
one forward pass; a bandwidth win that is not always a win, and is not lossless
on a quantized target. [[gotchas/spec-decode-greedy-diverges-on-quantized-target]]

## Execution-level terms

**CUDA graph** — capturing a launch sequence so the CPU submits it once instead of
per-step, removing launch overhead. Degrading to PIECEWISE silently costs ~16%
decode. [[gotchas/spec-decode-silently-downgrades-cudagraphs]]

**Kernel launch overhead** — per-launch CPU cost; material at small batch where
the GPU is not the bottleneck. [[flops/custom-kernel-launch-overhead]]

**Warp specialization** — dedicating warps to producer vs consumer roles so TMA
copies overlap with math. [[flops/flashattention-3-hopper-warp-specialization]]

**Arithmetic intensity of attention** — the S x S score-matrix traffic that
tiling exists to remove. [[flops/flashattention-io-aware-tiling]]

## Terms with no coverage yet

Used in real benchmark methodology, discussed nowhere in this repo. Do not cite a
record for these.

- **TPOT** as a distinct metric — the repo reports `itl_ms` and
  `tok_s_per_user` instead; no record defines or measures TPOT on its own.
- **Goodput** as a measurement — the term appears in the DistServe material, but
  no benchmark record reports goodput against an SLO.
- **P99 / tail latency** as a recorded metric — the phrase appears inside
  interconnect records and NVFP4 gotchas, never as a benchmark `metric` value.
- **Long-context quality benchmarks** (RULER, needle-in-a-haystack) — long-context
  behavior is covered from the performance side
  ([[gotchas/gated-deltanet-decode-collapse-long-context]]) but no
  retrieval-quality-at-length record exists.
- **Prefix-cache hit rate** as a benchmark record — the concept is documented in a
  gotcha, but hit rate itself is never a measured record.
- **TPOT-style streaming/chunking artifacts** — the effect of `--stream-interval`
  on measured ITL is recorded inside
  [[benchmarks/vllm-qwen35-nvfp4-gb200-nvl72-tps-per-gpu]] rather than as its own
  term.

## Related

- [docs/07-benchmarking-methodology.md](07-benchmarking-methodology.md) — how to
  record a measurement without misleading someone.
- [SCHEMA.md](../SCHEMA.md) — field-by-field reference.
- [AGENTS.md](../AGENTS.md) — the ingest contract.