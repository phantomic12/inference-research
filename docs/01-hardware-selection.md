# 01 — Hardware selection by workload

<!-- generated against a snapshot of 1,087 records (74 accelerators, 45 flop classes, 40 engines, 52 quantization formats, 33 interconnects, 60 benchmarks, 34 gotchas, 60 supply records, 97 models, 590 sources). data/ grew to 1,281 records / 47 gotchas / 64 engines / 82 quantization formats while this was written; the later gotchas were folded into doc 05 and the later engines into doc 03. Counts were a snapshot; data/ is written concurrently. -->

**How to read this document.** It is organised by *workload*, not by part. For each
workload it names (a) the binding constraint, (b) which accelerator records fit that
constraint, (c) which gotcha records will bite you there, and (d) — where it matters —
what the records cannot answer.

The single structural fact underneath every recommendation here is that **batch-1
decode is memory-bound and prefill is compute-bound**, with the crossover between them
quantified per part in [02-flop-map.md](02-flop-map.md). A "fast GPU" is a fast
compute part or a fast memory part; they are rarely the same part, and the workload
decides which one you need.

Read [[accelerators/nvidia-h100-sxm]] alongside this document for the worked example:
80 GB HBM3 at 3,350 GB/s and 989.5 dense bf16 TFLOPS gives a dense-bf16 roofline
balance point of **295.4 flop/byte**, so a batch-1 decode GEMM (~1 flop/byte) sits roughly
**295x below the ridge**. That arithmetic, not the headline TOPS figure, is what each
recommendation below turns on.

**One warning from the records that governs how the whole table set is read:**
[[flops/decode-gemm]] notes that "an RTX 4090 has a **LOWER** bf16 ridge than an H100
(163.9 vs 295.4) and **is still dramatically worse at decode**. The ridge tells you **WHERE
the memory wall ends, not HOW WIDE it is.** Wall width is what matters at batch 1, and there
the ranking is set by **absolute bandwidth** (MI300X 5.3 TB/s > H100 3.35 TB/s > RTX 4090
1.008 TB/s) with **no reference to FLOPS at all.**" Do not rank decode candidates by ridge
point.

---

## 1. Single-user local chat (laptop or desktop, one stream)

**Binding constraint: memory capacity, then memory bandwidth. Not FLOPS.**

A single stream is a batch-1 decode loop. The records are unambiguous that this is
bandwidth-bound at every batch size: decode GEMM has arithmetic intensity exactly equal
to the batch size [[flops/decode-gemm]], so at batch 1 it is ~1 flop/byte, and the
hardware consequence stated in that record is that "memory bandwidth, not peak TOPS,
sets single-stream decode throughput, so a part with high FLOPS and modest bandwidth is
a throughput part, not a latency part." At batch 1 there is also a second, non-bandwidth
tax: per-kernel launch overhead [[flops/custom-kernel-launch-overhead]], which sets a
floor on inter-token latency that no FLOPS improvement moves.

**Fits, and why:**

| Record | Capacity | Bandwidth | Why it fits |
|---|---|---|---|
| [[accelerators/nvidia-rtx-5090]] | 32 GB GDDR7 | 1792 GB/s | "the strongest consumer memory subsystem here" per its own record, and it is what "lets a 30B-class model run in bf16/fp8 on a single card" |
| [[accelerators/nvidia-rtx-4090]] | 24 GB GDDR6X | 1008 GB/s | the 24 GB reference point; 165.2 dense bf16 TFLOPS |
| [[accelerators/nvidia-rtx-3090]] | 24 GB GDDR6X | 936 GB/s | capacity-equivalent but 936 against [[accelerators/nvidia-rtx-3090-ti]]'s 1008; the record explicitly warns that quoting "24 GB RTX 3090" from a Ti benchmark overstates this card |
| [[accelerators/nvidia-rtx-pro-6000-blackwell]] | 96 GB GDDR7 ECC | 1792 GB/s | "the single-GPU capacity leader in this slice: 96 GB ECC GDDR7 at 1792 GB/s — same bandwidth as a 32 GB RTX 5090 with 3x the memory" |
| [[accelerators/apple-m3-ultra]] | 512 GB unified | 819 GB/s | if capacity, not speed, is the constraint — see the warning below |
| [[accelerators/intel-arc-b580]] | 12 GB GDDR6 | 456 GB/s | "Capacity is the binding constraint by a wide margin" — only for small models |

**Apple capacity-vs-bandwidth warning, stated by the record itself.** The M3 Ultra record
refuses to collapse these into one ranking: 512 GB at 819 GB/s versus the RTX PRO 6000
Blackwell's 96 GB at 1792 GB/s gives "5.3x capacity, 0.46x bandwidth". For a model that
fits on both, the record's conclusion is that the discrete card "is roughly 2x faster
per stream". Apple publishes no GPU TFLOPS for any of these parts — the flops array is
empty by choice, not by gap, and "any 'M-series TFLOPS' number in circulation is a
third-party die-level extrapolation, not an Apple figure."

**Gotchas that bite here:**

- [[gotchas/nvfp4-marlin-bf16-garbled-output]] — **blocker, silent wrong output.** Serving
  any NVFP4 model with `--dtype bfloat16` on SM < 100 (RTX 4090 SM89, V100 SM70) "loads,
  starts and serves, but every response is garbled." Workaround: `--dtype float16` on
  pre-Blackwell-datacenter parts, because the FP16 path maps the 5-bit exponent directly.
- [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]] — **blocker, silent wrong output**, and
  the corrupted path is "~2.5% FASTER end-to-end, so throughput benchmarking selects for
  the bug." Affects NVIDIA GB10 (DGX Spark, sm_121a).
- [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — **major.** On gfx1151 (Strix
  Halo APU, ROCm 7.2.4) "the HIP backend is wrong and Vulkan is right", verified same
  machine, same build, byte-identical flags. Treat Vulkan as the reference there.
- [[gotchas/vulkan-decode-cliff-hidden-size-4096]] — **major.** On gfx1201, a hidden-size
  cliff in the Vulkan decode path; the record's advice is to compute effective decode
  bandwidth from file size before concluding a card is slow.
- [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] — **major, silent.** llama-perplexity
  on ROCm/gfx1151 returns a PPL jump "of two orders of magnitude" vs the same model on
  another backend. Treat a PPL jump of two orders of magnitude as a hard stop.
- [[gotchas/gated-deltanet-decode-collapse-long-context]] — **major.** Hybrid GDN GGUF
  models on this stack: decode falls from ~33 t/s at 68K KV position to 1.4 t/s at 91K,
  *even with the model fully resident on GPU*. Cap `-c` below roughly 80K for these models.
- [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]] — **major.** The one
  Apple-Silicon trap in the repo: `mlx_lm.server` passes `/health` then the first request
  raises `NotImplementedError: RotatingKVCache Quantization NYI` on hybrid-attention models
  (Gemma 4 uses sliding-window attention on 35 of 42 layers at 26B-A4B).

**What the records cannot answer:** whether a given consumer card's llama.cpp or
MLX/Vulkan path is fast for a specific model, and any per-part tok/s ranking. The
benchmark records that do exist are heavily model-and-backend specific and mostly
`draft` status (e.g. [[benchmarks/llamacpp-radeon-r9700-gfx1201-q1-0-decode-tok-s]] is
draft, [[benchmarks/mlx-m3-pro-qwen3-4b-4bit-decode-tok-s]] is draft). There is no
controlled consumer-card decode comparison in the repo.

---

## 2. A 24 GB consumer card — the specific 24 GB decision

**Binding constraint: capacity at exactly 24 GB, which is why this deserves its own
section.** Four records in the repo carry 24 GB and they are not interchangeable.

| Record | Memory type | Bandwidth | TGP / TDP |
|---|---|---|---|
| [[accelerators/nvidia-rtx-3090]] | GDDR6X | 936 GB/s | 350 W |
| [[accelerators/nvidia-rtx-3090-ti]] | GDDR6X | 1008 GB/s | 450 W |
| [[accelerators/nvidia-rtx-4090]] | GDDR6X | 1008 GB/s | 450 W |
| [[accelerators/nvidia-a10]] | 24 GB GDDR6 | 600 GB/s | 150 W |

The 3090's record states the trap explicitly: "The two 24 GB cards are not
interchangeable on bandwidth; recommendations that quote '24 GB RTX 3090' from a Ti
benchmark silently overstate this card." The 3090 Ti is also the **only** GeForce in the
repo with NVLink (2-way SLI was the only consumer multi-GPU path on Ampere), and the 3090
family carries the NVLink bridge on the product page.

The **A10** is the memory-cheap datacenter option: 150 W and 600 GB/s for the same 24 GB.
Its record notes 125 dense bf16 TFLOPS, which is "roughly a tenth" of H100 SXM's FLOPS
for roughly a fifth of the bandwidth — it is a capacity-and-power part.

**Cross-tier comparison from the [[accelerators/nvidia-rtx-4080]] record, which states the
16 GB position explicitly:** "16 GB at 716.8 GB/s is ~71% of the 4090's 1008 GB/s. But the binding
constraint at 16 GB is capacity, not bandwidth." So within the consumer line, below ~24 GB
capacity binds first and above ~24 GB bandwidth binds — but see [[accelerators/nvidia-rtx-5090]]
(32 GB, 1792 GB/s) for where the crossover actually lands today.

**What the records cannot answer:** whether 24 GB is enough for a specific model at a
specific context length. That calculation needs the KV-cache figures from
[[flops/kv-transfer]] and [[flops/mla-latent-attention]] and is model-specific; the repo
has model records but no per-model capacity planner.

---

## 3. Long-context serving (32k+ context, KV cache is the budget)

**Binding constraint: memory capacity for the KV cache, and memory bandwidth to stream
it. Interconnect matters only if you shard.** [[flops/kv-transfer]] states this most
directly: "KV movement is the class that decides whether a long-context service is
viable at all, and it is the one that most rewards hardware with fast local memory and
low-latency scale-up links rather than fast FLOPs."

Two structural facts from the flop records that should drive the architecture choice:

- Decode attention has AI ~1 flop/byte **at every context length and every batch size**
  ([[flops/decode-attention]]): flops = 4*L*d and bytes = 4*L*d, both linear in L, so
  "growing the context does NOT raise arithmetic intensity, and batching does NOT raise
  it either." Cost per token grows linearly with context and no kernel fixes that.
- Prefill attention crosses from memory-bound to compute-bound at roughly **L ~ 300
  tokens** on H100 SXM dense bf16, and is "about 27x the ridge point" at L=8k
  ([[flops/prefill-attention]]). So long-context serving is a *two-constraint* workload:
  a compute-bound prefill phase and a bandwidth-bound decode phase over a capacity-hungry
  cache.

**Architecture choices that change the KV budget** — all from flop records, not speculation:

- [[flops/mla-latent-attention]]: DeepSeek-V3's MLA cache is 68.6 KiB/token at bf16
  versus 5,856 KiB/token for the MHA equivalent — "an 85.3x reduction, or 1.17% of the
  MHA cache" — and it is rated *stronger* than MHA in the same table, unlike GQA/MQA.
- [[flops/sliding-window-hybrid-attention]]: Gemma 3 at w=1024 with 5 local layers per
  1 global; local-layer cache is bounded at w and "does not grow with context at all."
- [[flops/ssm-recurrent-state-update]]: SSM/recurrent layers have **no KV cache at all**;
  the state "is CONSTANT in context length, so unlike KV attention its byte count never
  grows." Nemotron-H-8B is 48 Mamba / 4 attention.
- [[flops/modern-attention-variants]]: YOCO stores the global cache once instead of per
  layer; NSA/DSA reduce the work set but "still pay a full-context scoring pass."
- [[flops/flashmla-decode-fp8-fp4-kv]]: FlashMLA V4.1 stores 528 bytes/token at fp8;
  the fp4 variant 288 bytes/token.

**Fits, and why:**

| Record | Capacity | Bandwidth | Why |
|---|---|---|---|
| [[accelerators/nvidia-h200-sxm]] | 141 GB HBM3E | 4800 GB/s | "purely a memory-capacity and bandwidth uplift" over H100 SXM — same compute, same 700 W. Its record: "it wins wherever KV cache or weights do not fit, and loses on nothing" |
| [[accelerators/amd-instinct-mi325x]] | 256 GB HBM3E | 6000 GB/s | "the capacity sweet spot of the CDNA3 generation: 256 GB per OAM at 1000W" |
| [[accelerators/nvidia-h100-nvl]] | 94 GB HBM3 | 3938 GB/s | "MORE bandwidth (3,938 GB/s) than the 80 GB HBM3 SXM (3,350) while computing LESS" — the deliberate bandwidth-over-FLOPS position |
| [[accelerators/apple-m3-ultra]] | 512 GB unified | 819 GB/s | the only record in the repo where a 400B-class model fits at all; the M3 Ultra record states "a 400B-class model fits on the Mac and nowhere else in this slice" |
| [[accelerators/amd-instinct-mi350x]] | 288 GB HBM3E | 8000 GB/s | same CDNA4 silicon as MI355X at 2200 MHz / 1000 W; 8 TB/s at 1000 W is the best bandwidth-per-watt in the AMD line |

**The MI300X/MI250 trap that affects capacity planning:** MI250X's 128 GB is "the WHOLE
OAM module, i.e. 2 x 64 GB HBM2e" and you must "NOT read '128 GB' as one 128GB device,
and do NOT read it as two 64GB devices either" [[accelerators/amd-instinct-mi250x]]. The
MI300X record makes the same point for 192 GB across 8 XCDs, and MI325X for 256 GB. This
is a shared address space across compute dies, not a device count.

**Gotchas that bite here:**

- [[gotchas/vllm-kv-cache-block-budget]] — **major.** "No available memory for the cache
  blocks." The KV budget is a first-class constraint and the error message names the fix
  (`--gpu_memory_utilization`, `--max-num-batched-tokens`, `--max-model-len`).
- [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — **blocker, silent
  corruption.** 8x A800-80GB (SM80, no native FP8 tensor cores) with
  `--kv-cache-dtype fp8_e5m2`, TP=8: the server runs and returns wrong output.
- [[gotchas/eagle-prefix-cache-last-block-drop]] — **major.** Multi-round and
  repeated-prompt workloads lose ~6-10 points of prefix-cache hit rate (97.5% → 86.9-91.7%)
  when speculative decoding is on. Measure hit rate with and without `speculative_config`.
- [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] — **blocker.** On a finetuned Qwen3.6
  35B-A3B with MTP (num_speculative_tokens=2) plus `--enable-prefix-caching`, internal
  classification accuracy drops ~20% versus the identical setup without.
- [[gotchas/vllm-v1-memory-footprint-growth]] — **major.** Same model and flags: vLLM 0.6.x
  ran at 12K context on 4x RTX 3070; 0.7.0 with V1 enabled cannot exceed ~3K. Re-tune for
  the V1 engine rather than carrying V0 values.
- [[gotchas/rdna-triton-paged-attn-decode-cliff]] — **major.** Qwen3.6-27B on gfx1100:
  12.1 tok/s at 518 tokens of context but 4.2 tok/s at 32K. Only the 16 full-attention
  layers degrade. Check head_dim against 128 and the measured KV block size against 16.

**What the records cannot answer:** the KV-cache budget for any specific model at a
specific context length, as a number you can plug into a sizing decision. The ingredients
exist ([[flops/kv-transfer]] gives the per-token formula, [[flops/mla-latent-attention]]
and [[flops/sliding-window-hybrid-attention]] give architecture multipliers) but no
record composes them into a table. This is the single highest-value gap in the repo for
this workload.

---

## 4. High-concurrency multi-user server

**Binding constraint: it moves. At low concurrency, bandwidth and KV capacity. At high
concurrency, FLOPS — because batching is the only lever that raises decode arithmetic
intensity, and its ceiling is the roofline ridge point.**

The mechanism, in the records' own terms: decode GEMM AI = B, so "the entire throughput
story of an inference service is 'what is the average effective B'", and continuous
batching is the mechanism that keeps B high ([[flops/decode-gemm]],
[[flops/continuous-batching-scheduling]]). The crossover the records give you: you need
batch on the order of **150-300** for bf16 to be compute-bound on H100 SXM, and roughly
half that in fp8. For an MoE the bar is dramatically higher — see below.

**The MoE number is the one that should drive a multi-user MoE server decision.**
[[flops/moe-experts]]: with E experts and top-k, AI = B*(k/E) flop/byte, because you must
stream every expert weight to do k/E of the work. Worked against DeepSeek-V3's own config
(256 routed experts, 8 activated, k/E = 1/32): "an MoE decode step needs a batch of about
32*295 = 9,440 to reach compute-bound where the same-width dense model needs ~295." The
record's design consequence: "a MoE model with the same parameter count as a dense model
is not faster to serve — it is slower."

**Fits, and why:**

| Record | Capacity | Bandwidth | Dense bf16 | Why |
|---|---|---|---|---|
| [[accelerators/nvidia-b200]] | 186 GB HBM3E | 8000 GB/s | 2500 TFLOPS | the only record with a genuine 8000 GB/s *and* 2500 dense bf16 pairing — the ridge point moved, so the batch needed to hit compute-bound is much lower |
| [[accelerators/nvidia-gb200-nvl72]] | 186 GB HBM3E | 8000 GB/s | 2500 TFLOPS | same silicon plus NVSwitch 5 (900 GB/s per GPU per direction, [[interconnect/nvswitch-5]]) |
| [[accelerators/amd-instinct-mi350x]] / [[accelerators/amd-instinct-mi355x]] | 288 GB HBM3E | 8000 GB/s | 2300 / 2500 TFLOPS | MI355X is the best bandwidth-per-FLOP in the AMD line; MI350X differs only in clock and power (2200 vs 2400 MHz, 1000 vs 1400 W) |
| [[accelerators/nvidia-h100-sxm]] | 80 GB HBM3 | 3350 GB/s | 989.5 TFLOPS | the reference part every crossover figure in [[flops/decode-gemm]] is quoted against |
| [[accelerators/nvidia-a100-80gb-sxm4]] | 80 GB HBM2e | 2039 GB/s | 312 TFLOPS | the older-generation option: 312 against H100 SXM's 989.5 dense bf16 (a 3.2x gap), at 2039 against 3350 GB/s |

**Scale-up fabric matters here, and the records disagree about what it costs.**
[[accelerators/amd-instinct-mi300x]] and MI325X carry 8 Infinity Fabric links at 128 GB/s;
the interconnect record is **contested** on direction — AMD's MI300X page states 128 GB/s
"with no direction stated", and the record retains both readings (64 GB/s per-direction
bidirectional, or 128 GB/s per-direction), noting a 64 GB/s per-direction figure is "roughly
7x below NVLink 4 per direction" ([[interconnect/amd-infinity-fabric-link]]). By
contrast [[accelerators/nvidia-h100-sxm]] exposes NVLink 4 at 900 GB/s bidirectional per
GPU and [[interconnect/nvswitch-4]] 900 GB/s between GPUs. **Both figures are retained;
neither record resolves the direction question.** Treat "MI300X multi-GPU bandwidth
headroom is tighter than H100's" as contested, not settled.

**Where the measured numbers actually land** (benchmark records, with the methodology the
schema demands): [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]] 57.6 tok/s at batch 1
versus [[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]] 12.5 tok/s per user at batch 64
with aggregate 800 tok/s — the cleanest illustration in the repo that aggregate throughput
and per-user latency trade directly.

**AMD can beat NVIDIA on aggregate throughput — with caveats you must quote.** On Llama 3.1
405B FP8 at TP=8, [[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-output-throughput]]
records 3171 output tok/s aggregate for MI300X against
[[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-h100]] 1957 for H100 — and the record's
own note says MI300X "wins all four" 405B shapes in that table (1.25x, 1.62x, 1.13x, 1.28x;
geomean 1.31x). **But on 70B the same table's record says the opposite**:
[[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]] 15105 against
[[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-h100]] 15810, i.e. 0.96x, geomean 0.94x
across four shapes. So the honest summary is: **MI300X is at parity or slightly behind on 70B
and clearly ahead on 405B**, and the record attributes the crossover to "the
bandwidth-to-capacity balance at 8-way TP for large weights," not to CDNA versus Hopper.

Two caveats the records insist on, which must travel with any quote of these numbers: the
NVIDIA half was "measured by AMD, not by NVIDIA" and the two sides used **different
engines** (vLLM on AMD, TensorRT-LLM on NVIDIA), so "the delta blends silicon and software";
and the comparison is **not memory-capacity-neutral** (MI300X has 192 GB against H100's
80 GB). Batch size, KV-cache dtype, GEMM tuning state and driver versions were all omitted
by the source.

**Gotchas that bite here:**

- [[gotchas/fa4-num-splits-ignored-sm90]] — **major.** Decode on H100 (SM90) is 48% slower
  per token than on v0.26.0 at batch 1 and 36% slower at batch 4, while TTFT is flat. The
  fix is backend selection, not hardware: `attention_backend=TRITON_ATTN` recovers ~70%.
- [[gotchas/spec-decode-silently-downgrades-cudagraphs]] — **major.** A single
  `logger.warning` announces a CUDA-graph downgrade; decode throughput 47.5 tok/s on
  FLASHINFER vs 55.2 on FLASH_ATTN (+16%), changing only `--attention-backend`.
- [[gotchas/spec-decode-greedy-diverges-on-quantized-target]] — **major.** Under greedy
  sampling, `--spec-type draft-dspark` or `draft-mtp` produces *different text* than the
  same server without it, when the target is quantized. n-gram speculation is verified
  lossless.
- [[gotchas/cc-mode-uva-view-garbage-output]] — **blocker.** On H100 NVL in CC mode inside
  a TDX guest, vLLM 0.29.0 with the default V2 model runner serves and every response is
  garbage (token id 0 repeated). `VLLM_USE_V2_MODEL_RUNNER=0` fixes it.
- [[gotchas/vllm-rocm-fp8-cold-start-timeout]] — **major.** vLLM on gfx1100 with an FP8
  checkpoint aborts during startup on the engine-ready timeout (default 600 s); measured
  first init far exceeds it. Set `VLLM_ENGINE_READY_TIMEOUT_S=1800` for the first cold
  start; once `~/.triton/cache` is warm, init drops to ~49 s.
- [[gotchas/aiter-gate-skips-rdna3-gfx1100]] — **major.** On gfx1100 Radeon, vLLM logs
  "AITER is not found or not supported on the current platform" and falls back to
  emulation. (Note this gotcha's `affects` list names `amd-radeon-pro-w7900` and
  `amd-radeon-rx-7900-xtx`, which have **no accelerator records** — a dangling reference.)

**What the records cannot answer:** sustained tokens-per-dollar. The supply records give
per-GPU-hour prices but no benchmark ties a specific price to a specific measured
tokens/s on the same part with the same engine, so any "$/Mtok" figure would be a
fabrication. See [06](01-hardware-selection.md#what-the-records-cannot-answer-workload-6)
below and the cost section.

---

## 5. Batch / offline throughput

**Binding constraint: FLOPS, at high enough batch that decode becomes compute-bound.
This is the *only* workload where peak dense TFLOPS is the right metric — and it is
exactly the workload where the dense-vs-sparse trap is most expensive.**

**The dense/sparse trap is the first thing to check on any part you are buying for batch
throughput.** NVIDIA's own printed figures are sparse-marked and dense is one half. From
[[accelerators/nvidia-b200]]: "NVIDIA's Blackwell datasheet states 'Specifications in
sparse. Dense is one-half of the sparse spec shown'" — so a '10 PFLOPS FP8' claim for
B200 is sparse, and dense FP8 is 5 PFLOPS. From [[accelerators/nvidia-h100-sxm]]: NVIDIA
prints 989 TF32 / 1,979 BF16+FP16 / 3,958 FP8 all "WITH SPARITY", and dense is exactly
half — and the record notes "anyone reading the sheet without the footnote halves it
twice." The repo's stored dense values: B200 2500 bf16 / 5000 fp8 dense; H100 SXM 989.5 /
1979 dense.

Two cases where the convention *differs*, both retained:
- [[accelerators/nvidia-l40s]] prints explicit `dense|sparse` pairs (FP16 "362.05 | 733*"),
  so the **left** column is dense — and the record notes "the marketing headline for L40S is
  733 'FP8 TFLOPS' and that is already the DENSE number, while other vendors' 733 figures
  are sparse."
- [[accelerators/nvidia-b300]] is **contested** (confidence 0.6): the Blackwell Ultra
  datasheet prints FP4 as '20 PFLOPS | 15 PFLOPS' under a footnote reading "Specification in
  Sparse | Dense" — a **sparse|dense pair, the opposite convention from the other rows** — so
  dense FP4 is 15 PFLOPS, not 10. The record also flags the INT8 row as a probable broken
  text-layer extraction (330/307 TOPS against ~10 PFLOPS implied) and does not record INT8.
- [[accelerators/intel-arc-pro-b60]] is the counterexample where labelling is good: the
  [[accelerators/intel-arc-b580]] record notes Intel defines its 233 TOPS as "the peak
  throughput when running XMX workloads with INT8 datatype and DENSE models."

**Fits, and why:**

| Record | Dense bf16 | Dense fp8 | Why |
|---|---|---|---|
| [[accelerators/nvidia-b300]] | 2500 TFLOPS | 4500 TFLOPS (contested record) | highest recorded dense FP8 among NVIDIA; caveat above |
| [[accelerators/nvidia-b200]] / [[accelerators/nvidia-gb200-nvl72]] | 2500 | 5000 | 10,000 dense FP4 — the only records with a native FP4 tensor path |
| [[accelerators/amd-instinct-mi355x]] | 2500 (matrix) / 157.3 (vector) | 5033.2 | **contested record.** Product page and datasheet publish *different* numbers for the same datatypes (5.0 vs 5.0332 PFLOPs OCP-FP8); both retained, ~0.06% spread. CDNA4 is the first AMD part where fp16 vector and fp16 matrix differ ~16x, so a record listing only 2500 "loses real information" |
| [[accelerators/amd-instinct-mi350x]] | 2300 | 4600 | the air-cooled sibling; the 2300/2500 = 0.92 ratio matches the 2200/2400 = 0.917 clock ratio exactly |
| [[accelerators/intel-gaudi3]] | 1678 | 1678 | "The published BF16 and FP8 MME TFLOPS are IDENTICAL at 1678, which is unusual and worth not misreading as a typo" |
| [[accelerators/amd-instinct-mi300x]] | 1307.4 | 2614.9 | CDNA3, and the benchmark that beats H100 on aggregate 8-GPU throughput |

**Two accuracy warnings that must travel with any batch-throughput number:**
- [[accelerators/amd-instinct-mi350x]] / [[accelerators/amd-instinct-mi355x]]: TF32 is
  **software-emulated** on CDNA4 (AMD's own footnote). "Do not carry a MI300X TF32 figure
  forward to MI355X" — MI300X has 653.7 dense TF32.
- [[accelerators/nvidia-h20]] is **draft, confidence 0.45**, and is the lowest-confidence
  record in the accelerator set: NVIDIA has never published an H20 datasheet, every figure
  is from a third-party aggregator with `vendor_claim: false`. Its shape is the
  interesting part — 148 dense FP16 TFLOPS against 96 GB at 4.0 TB/s, i.e. "roughly 2/3 of
  H100 SXM's dense 989 TFLOPS but a HIGHER bandwidth-per-FLOP ratio, so H20 is a
  memory-rich, compute-poor Hopper." Do not quote it as verified.

**Where measured throughput lands** — MLPerf-family benchmark records, all with full
methodology: [[benchmarks/mi355x-mlperf-v6-0-llama2-70b-wmxfp4-offline-tokens]] 103,480
output tok/s aggregate on WMXFP4 weights with FP8 KV cache;
[[benchmarks/mi355x-mlperf-v6-0-gpt-oss-120b-offline-tokens]] 95,004 on a MoE;
[[benchmarks/sglang-deepseek-v3-96xh100-output-tps-per-node]] 22,300 on 96x H100.
**These are not comparable to each other** — different models, formats, node counts and
concurrency. Cite them only with their model and node count attached.

**Gotchas that bite here:**
- [[gotchas/triton-w4a16-gptq-qzeros-assert]] — **major.** Serving GPTQ-Int4 fails on the
  Triton w4a16 path with a `qzeros.shape` assertion on ROCm.
- [[flops/marlin-weight-only-int4-gemm]]'s stated limitation: Marlin "is not yet
  optimized for Hopper" — on Hopper Marlin is the fallback, not the fast path.
- [[gotchas/flashinfer-rejects-large-head-dim]] — **major.** Engine init aborts with
  "head_size not supported" for large head dims on RTX PRO 6000 Blackwell; the same model
  runs under TRITON_ATTN.

---

## 6. Cost-per-token-sensitive

**Binding constraint: capacity-to-bandwidth ratio per dollar, and — more often than not
— the fact that the records give you prices and throughputs in *different* units, so the
join has to be done by hand and is not yet done anywhere in the repo.**

The supply records are careful about this, and their own warnings should be carried
forward:

- **Rental bands** (all observed 2026-10-03, per-GPU-hour, verified hosts):
  RTX 3090 median $0.1498, RTX 4090 $0.4289, RTX 5090 $0.4896
  ([[supply/rental-consumer-4090-3090-5090-vast-index]]); L40S $0.7085 and RTX 6000 Ada
  $0.6289 ([[supply/rental-datacenter-l40s-6000ada-vast-index]]); H100 SXM $2.4704
  ([[supply/rental-h100-sxm-vast-index]]); H200 SXM $5.0005 ([[supply/rental-h200-sxm-vast-index]]);
  B200 $7.9695 ([[supply/rental-b200-vast-index]]).
- **Rental and purchase prices do not move together.** The consumer rental record states
  the ratio explicitly: the RTX 3090 rents at ~0.35x the RTX 4090 rate while its asking
  purchase price is ~0.46x. "Neither number can stand in for the other."
- **The cheapest credible accelerator hour in the slice is a TPU, not a GPU**, and it is
  not a drop-in substitute: v5e at $1.20 per *chip*-hour, less than AWS g6's L4 at $1.67,
  but "XLA/JAX only" and quota-gated rather than payment-gated
  ([[supply/gcp-cloud-tpu-v4-v5e-v5p]]).
- **Hyperscaler Hopper is mostly allocation-only.** [[supply/aws-ec2-p5-h100]] is the
  "scarcity record for this slice" — AWS's own price list carries p5.48xlarge at an
  explicit $0.00/hour alongside the real $55.04. Same sentinel on p4d and p5en.
  [[supply/azure-nd-h200-v5]] records a **0% spot discount** on H200 v5 — "$84.80,
  byte-identical to the on-demand rate... A zero-discount spot price is what a sold-out
  fleet looks in a price API."
- **A stale belief this repo explicitly corrects:** [[supply/oracle-bm-gpu-h100-h200]]
  records that the historical "OCI is dramatically cheaper for GPU" claim "no longer
  holds" for Hopper — $10.00/GPU-hour against AWS p5's $6.88. What Oracle still has is the
  bare-metal shape with no vCPU/RAM bundling.
- **The 48 GB Ada tier is the value tier**, per the rental record: L40S and RTX 6000 Ada
  "rent within ~13% of each other... while sitting ~6.5x below H100 SXM and ~1.6x above
  RTX 4090."
- **Refurbished consumer is the cheapest 24 GB per dollar** in the whole slice: a
  Micro Center refurbished RTX 3090 24GB at $1,099.99 is called "the most cost-effective
  24GB-per-dollar option in this entire slice" ([[supply/microcenter-refurb-rtx-3090]]).
  The channel is store-inventory only — "it will not ship."
- **Do not buy used Intel silicon.** [[supply/newegg-refreshed-arc-b580]] records refreshed
  B580 at $199.99-$259.99 versus used $290-$499 and new $329.99-$590: "Buying USED Intel
  silicon is strictly dominated."

**Fits:** for a cost-sensitive *single-user or small-batch* deployment the value ordering
the records support is RTX 3090 (rental $0.1498/h, [[supply/rental-consumer-4090-3090-5090-vast-index]];
refurb purchase $1,099.99, [[supply/microcenter-refurb-rtx-3090]]) → L40S at $0.7085/h or
RTX 6000 Ada at $0.6289/h, both 48 GB ([[supply/rental-datacenter-l40s-6000ada-vast-index]])
→ H100 SXM ($2.4704/h rental, [[supply/rental-h100-sxm-vast-index]]; $6.88/h AWS on-demand
when you can get it at all, [[supply/aws-ec2-p5-h100]]). For cost-sensitive
*high-concurrency* serving, the cheapest H100 hour in the slice is Runpod at $3.49/GPU-hour
([[supply/runpod-pods-h100-h200]]), below [[supply/lambda-gpu-cloud-h100]]'s $3.99 and well
below AWS's $6.88.

**Gotchas that bite here:** none of the gotcha records are about price. The price traps
are in the supply records themselves and are quoted above.

**What the records cannot answer — and this is the biggest honest gap in this document:**
**cost per token.** There is no record that joins a per-GPU-hour price to a measured
tokens/s for the same part, engine, model and precision. Every figure above is either a
price or a throughput, never both. A reader wanting $/Mtok must either run the benchmark
themselves or accept a hand-built ratio whose error bars nobody has written down.
[[supply/market-gpu-price-bands-2026-10]] holds the aggregate bands and is
deliberately `price_usd: null` on the grounds that "one point estimate for 'the GPU market'
would be a fabrication" — that same discipline is why no cost-per-token record exists.

---

## 7. Embedding / reranking workloads (compute-bound, the inverse of everything above)

**Binding constraint: FLOPS, at batch 1. This is the one workload class where a bandwidth-bound
consumer card is the wrong purchase.**

The records are unanimous and give the arithmetic. [[flops/cross-encoder-reranking]]: "Reranking
N passages is N INDEPENDENT forward passes with no KV-cache reuse, so it is **prefill-shaped
and compute-bound even at batch 1** — the exact opposite of the generative decode regime."
Its AI is **~t flop/byte** from weight amortization alone, "t times HIGHER than the ~1
flop/byte of a decode GEMM."

[[flops/embedding-batch-encoding]] gives the threshold explicitly: **AI = B·t**, so compute-bound
once **B·t ≥ 295.4** on H100 SXM — **B ≥ 3 at t = 128**. Corpus work pushes B far above that.

**Corpus cost shape decides the bill:**
- **Embedding: linear** in corpus size — "each document is encoded exactly once and the vector
  is cached."
- **Reranking: linear in N but N is top-k of first-stage retrieval**, so "rerank top-20 vs
  top-100 is a **5x difference in a bill that nothing else can reduce**."

**Fits:** the same dense-FLOPS parts as batch serving —
[[accelerators/nvidia-b200]] (2500 dense bf16) and
[[accelerators/amd-instinct-mi355x]] (2500 dense bf16 matrix) lead, with
[[accelerators/nvidia-h100-sxm]] (989.5) as the reference. Notably the **48 GB Ada tier is
well-matched here** rather than capacity-constrained: [[accelerators/nvidia-l40s]] at 362 dense
bf16 TFLOPS and [[accelerators/nvidia-rtx-6000-ada]] at 48 GB.

**The ColBERT inversion is worth knowing.** [[engines/colbert-reference]]: late interaction
"INVERTS the usual arithmetic-intensity intuition for a GPU... Hardware selection therefore
needs **BOTH**: a high-FLOPS part for encoding the corpus, and high-bandwidth/high-capacity
RAM or storage for the index, which is **the opposite mix from what an LLM decode part
wants**." Index size is on the order of ~100x a single-vector index.

**What the records cannot answer:** encoder throughput per GPU-second by part. The records
say the engine matters more ([[engines/hf-text-embeddings-inference]] "the reference
implementation most teams reach for first, and it is what Hugging Face itself deploys";
[[engines/vllm-pooling-models]]'s own docs disclaim a performance advantage) but there is **no
benchmark record for any encoder on any accelerator**. Only two Apple decode benchmarks exist
at all, and neither is an encoder.

---

## What the records cannot answer, by workload

Honest negatives, collected in one place because they are the next research priorities:

1. **Cost per token** (all workloads). No price↔throughput join exists. Highest value.
2. **KV-cache budget per model per context length** (§3). The formula exists in
   [[flops/kv-transfer]], the architecture multipliers exist, the composition does not.
3. **A controlled consumer-card decode comparison** (§1). Existing benchmarks are
   backend- and model-specific and mostly `draft`; there is no apples-to-apples ranking.
4. **Sustained multi-user throughput per dollar for AMD vs NVIDIA at equal model and
   precision** (§4). [[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]]
   is one such comparison and is valuable precisely because there are so few.
5. **Anything about Huawei, Biren, Enflame, MetaX, Horizon or Moore Threads as a
   purchase.** [[accelerators/huawei-ascend-910b]] is `contested` at confidence 0.45 with
   "no primary source at all"; [[accelerators/biren-bili-166]], [[accelerators/enflame-s60]],
   [[accelerators/metax-c500-n100]], [[accelerators/horizon-journey-6p]],
   [[accelerators/moore-threads-s4000]] and [[accelerators/moore-threads-s5000]] are all
   `draft`, several with no memory or compute figure at all. [[accelerators/huawei-ascend-910d]]
   is "DELIBERATELY ALMOST EMPTY" because only *existence* is substantiable, not shipping.
   [[accelerators/horizon-journey-6p]] records an explicit **negative finding** that it is
   not a plausible source of LLM inference hardware.
6. **Volta and anything below sm75 as a 2026 purchase.** [[engines/vllm]] requires compute
   capability 7.5+, [[engines/hf-text-embeddings-inference]] excludes V100 and Titan V
   explicitly, and [[engines/flashinfer]] is unsupported below sm75.
7. **Non-NVIDIA, non-Apple parts as first-class multi-user serving.** The ROCm caveats in
   [[engines/vllm]] and [[engines/sglang]] are specific and severe enough that "AMD works"
   is not a supportable statement; see [03-engine-selection.md](03-engine-selection.md).

---

## Related documents

- [02-flop-map.md](02-flop-map.md) — why each workload above binds the way it does.
- [03-engine-selection.md](03-engine-selection.md) — which engine can actually run on these parts.
- [04-quant-selection.md](04-quant-selection.md) — the lever that changes the capacity column in every table above.
- [05-known-traps.md](05-known-traps.md) — every gotcha record, grouped by severity.