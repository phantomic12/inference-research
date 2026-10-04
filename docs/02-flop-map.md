# 02 — FLOP map: which operations dominate, and what that means for hardware

<!-- generated against a snapshot of 1,087 records (45 flop classes of 45; counts were a snapshot at authoring time). -->

This is the document that answers the underlying question — *why is my inference slow, and
which hardware property would fix it* — directly, without going through a part catalogue.

Every flop record carries a `bound_by` and an `arithmetic_intensity` derivation. This
document reorganises all 45 by **what property of the machine the class actually
demands**, so a hardware decision can be read straight off it. Crossover points are
quoted from the records; where a record does not give one, this document says so rather
than deriving a new number.

---

## 1. The one-paragraph version

Arithmetic intensity for LLM inference is not a property of the model, it is a property of
**the phase**. Batch-1 decode is ~1 flop/byte and is therefore purely a question of
**memory bandwidth**. Prefill crosses from memory-bound to compute-bound within the first few
hundred tokens — **591 on H100 for MHA, 148 for GQA-4, 74 for GQA-8** — and is therefore a
question of **tensor-core FLOPS** beyond that. Decode attention sits at **AI = g, the GQA
group ratio** (1 for MHA, 4 for Llama-3-8B, 8 for Llama-2-70B, 32 for MQA), and that value is
**independent of both context length and batch size**, so long-context serving is a question of
**memory capacity** (the cache) and **bandwidth** (streaming it) with FLOPS almost
irrelevant. Elementwise work (norm, softmax, embedding, sampling) is
~0.5-2 flop/byte and is pure overhead that fusion removes. MoE is the worst of all: AI =
B·(k/E), so a MoE decode step needs a batch of ~9,440 to reach compute-bound where a dense
model of the same width needs ~295.

The practical consequence: **"which GPU is fastest" is not a well-formed question until
you say which phase and which batch size.** At batch 1 decode you are buying bandwidth; at
batch 256 you are buying FLOPS; the crossover for your model is computable and it is *not*
the same part either side of it. And the workload type flips the answer outright: **encoders
are compute-bound at batch 1** — reranking is prefill-shaped at AI ~ t flop/byte
([[flops/cross-encoder-reranking]]) and batch encoding crosses the ridge at **B ≥ 3** for
128-token documents ([[flops/embedding-batch-encoding]]) — so the same card is bought for
FLOPS to embed and for bandwidth to generate.

---

## 2. The complete map

| flop record | bound_by | knob that moves it | hardware property that therefore matters |
|---|---|---|---|
| [[flops/decode-gemm]] | memory | batch size (AI = B) | **memory bandwidth**, not FLOPS |
| [[flops/decode-attention]] | memory | **GQA group ratio g** — AI = g, flat in context AND batch | **memory bandwidth** + **capacity**; FLOPS never helps |
| [[flops/kv-transfer]] | interconnect | context length, batch × live sequences, TP degree, prefix-sharing rate | **scale-up fabric**, then **capacity** |
| [[flops/moe-experts]] | memory | batch size (AI = B·k/E), num_experts, top_k | **memory bandwidth**; capacity sets the roof |
| [[flops/moe-routing]] | interconnect | num_experts, top_k, expert_parallel_degree | **all-to-all latency** over scale-up fabric |
| [[flops/norm-softmax]] | memory | hidden_width, batch (AI unchanged) | **bandwidth**; and *fusion quality* in software |
| [[flops/embedding]] | memory | vocab_size, hidden_width, batch | **bandwidth**, plus **latency** for the gather |
| [[flops/sampling]] | memory | vocab_size, batch (no batching benefit) | **bandwidth**; costs are latency *variance* |
| [[flops/speculative-draft]] | memory | draft_k (AI = k) | **bandwidth** + capacity for both models |
| [[flops/speculative-verify]] | memory | draft_k (AI = k), context (k·L flops) | **bandwidth**; capacity |
| [[flops/ssm-recurrent-state-update]] | memory | batch (state does *not* amortise), NOT context | **bandwidth** on a constant-size state |
| [[flops/mla-latent-attention]] | memory | num_kv_heads, d_c, layers | **capacity** (it is a 85.3x cache reduction) |
| [[flops/mla-absorbed-upprojection]] | memory | context length, num_layers | **bandwidth** — this is a pure weight-stream tax |
| [[flops/paged-attention-block-table-kernel]] | memory | context, batch, page size | **capacity** (its payoff is capacity, not bytes) |
| [[flops/flashdecoding-split-kv-decode]] | memory | context (more KV to split), batch, num_SMs | **SM count / occupancy**, not bytes or FLOPs |
| [[flops/flashmla-decode-fp8-fp4-kv]] | memory | context, KV precision | **bandwidth** (528 or 288 bytes/token) |
| [[flops/structured-2-4-sparsity-mma-sp]] | memory | sparsity pattern (only exactly 2:4) | **bandwidth** — it halves bytes by dropping zeros |
| [[flops/llamacpp-mmq-int8-dp4a-gemm]] | memory | batch, weight quant type | **broad arch reach**, *no* tensor cores used |
| [[flops/lmcache-kv-tier-outside-attention]] | memory | prefix sharing rate, HBM capacity | **capacity** — it is a cache, not a kernel |
| [[flops/flashinfer-backend-selection]] | memory | batch, context, prefix-sharing rate | which kernels exist on your arch |
| [[flops/tma-cp-async-bulk-tensor]] | memory | tile size it enables | register file → tile size → FLOPs per byte |
| [[flops/prefill-attention]] | **compute** | context length (AI = L) | **tensor-core FLOPS** beyond L≈300 |
| [[flops/flashattention-io-aware-tiling]] | **compute** | context (bytes saved scale L²) | **FLOPS**, because it removes the memory round trip |
| [[flops/sliding-window-hybrid-attention]] | **compute** | window w, local:global ratio | **FLOPS**; capacity for the global layers only |
| [[flops/quantization-overhead]] | both | bits, weight_group_size, batch (the crossover knob) | **native** low-precision path or not |
| [[flops/marlin-weight-only-int4-gemm]] | compute | batch (target 4-8x larger than prior kernels) | **tensor cores**; **compute capability ≥ 8.0** |
| [[flops/machete-weight-only-hopper-gemm]] | compute | batch (M dimension) | **Hopper wgmma** specifically (sm90, CUDA only) |
| [[flops/flashattention-2-work-partitioning]] | **compute** | num_SMs, seqlen, batch | **SM count** — it converts achieved→implied FLOPS |
| [[flops/flashattention-3-hopper-warp-specialization]] | both | head_dim, precision | **SFU count vs tensor-core count** (the ratio) |
| [[flops/flashattention-4-blackwell-tmem]] | both | head_dim, tile size | **shared-memory bandwidth** (backward binds there) |
| [[flops/block-scaled-tensor-core-fp4-fp8]] | compute | scale block size, precision, arch generation | **tensor cores at fp8/fp4** |
| [[flops/cutlass-collective-gemm-tensorop-tiles]] | both | tile size, batch, num_SMs | tile size bounded by **shared memory** (228 KB Hopper / 328 KB Blackwell) |
| [[flops/continuous-batching-scheduling]] | both | batch, arrival rate, KV pressure | whatever the current B makes it — it *chooses the regime* |
| [[flops/deepep-expert-parallel-dispatch-combine]] | interconnect | num_experts, top_k, link bandwidth | **NVLink or RDMA**; requires Hopper+ |
| [[flops/deepgemm-grouped-masked-moe-gemm]] | both | num_experts, tokens per expert, precision | **tensor cores**; SM90 vs SM100 scale layout |
| [[flops/deepgemm-mqa-lightning-indexer-scoring]] | both | context, num_heads, precision | **Blackwell** (SM100) for the packed-scale path |
| [[flops/diffusion-lm-denoising-steps]] | **compute** | denoising steps S, generation length Lg | **FLOPS** — it inverts the cost model entirely |
| [[flops/ssm-sequential-scan-tensor-core-mismatch]] | both | sequence length at prefill, chunk size | **tensor cores**, but only via a reformulation |
| [[flops/tensor-core-instruction-generations-wgmma-tcgen05]] | compute | architecture generation | **sm_90a vs sm_100a targets** — a compile-time gate |
| [[flops/tensor-memory-tmem-allocation]] | both | TMEM column budget, MMAs in flight | **512 TMEM columns/CTA** on sm_100a |
| [[flops/sm120-vs-sm100a-kernel-binary-incompatibility]] | compute | compute capability of the target | **the exact capability number, not the arch name** |
| [[flops/modern-attention-variants]] | both | context, sparse granularity, ratio | varies by variant — see §5 |
| [[flops/flashmla-fused-norm-rope-attn-rope-cast]] | both | hidden_width, batch (fewer launches at small batch) | software: **fusion**, and an offline layout cost |
| [[flops/tilelang-tile-dsl-moe-routing-quant]] | both | hardware backend, op, precision | a portability claim, not a speed claim |
| [[flops/custom-kernel-launch-overhead]] | **launch_overhead** | batch (inverse), kernels per layer, experts, host CPU | **host CPU and driver maturity** — CPU-bound cost |
| [[flops/quantized-decode-high-batch]] | both | bits (every halving halves the tolerable batch), group size | **native** low-precision path, or dequant is a separate pass |
| [[flops/tensor-parallel-allreduce]] | interconnect | batch B, link bandwidth per rank, TP degree | **scale-up link per rank** — an 18x spread across fabrics |
| [[flops/prefix-cache-hit-miss]] | both | **hit rate** (dominant), new tokens, cached prefix length | the profile *flips* between miss and hit — measure both |
| [[flops/long-context-rag-prefill]] | **compute** | context length, model width per layer, retrieved docs | **FLOPS** — the densest step in the stack |
| [[flops/cross-encoder-reranking]] | **compute** | num passages N (linear), pair length t (AI = t) | **FLOPS** — prefill-shaped even at batch 1 |
| [[flops/embedding-batch-encoding]] | **compute** | batch B (AI = B·t), chunk length, corpus size | **FLOPS** — the inverse of a decode part |
| [[flops/constrained-decoding-grammar-mask]] | memory | vocab V, grammar stack depth | **bandwidth** — O(depth·V) per token |
| [[flops/low-entropy-greedy-decode]] | memory | vocab V, **not** entropy | **bandwidth** — and see §3.10 |

---

## 3. The crossovers the records actually give you

These are the only numbers in this document that let you compute a decision. All are
quoted from the record named.

### 3.1 The ridge points, per part

**Dense-bf16 balance point on H100 SXM ≈ 295.4 flop/byte** (989 dense TFLOPS ÷ 3.35 TB/s) —
the reference every other figure is quoted against, stated in both
[[flops/decode-gemm]] and [[flops/prefill-attention]]. In **fp8 (1979 dense TFLOPS) the H100
ridge is ≈ 590.7**, so every crossover roughly doubles.

The records now compute the ridge for **several parts**, not just H100
([[flops/decode-gemm]] and [[flops/prefill-attention]], both from the accelerator records'
own dense figures, `ridge = 1000·TFLOPs / GB_per_s`):

| Part | bf16 ridge | fp8 ridge | B_ridge (decode) | L_ridge (prefill, MHA) |
|---|---|---|---|---|
| [[accelerators/nvidia-h100-sxm]] | 295.4 | 590.7 | 295 | 591 tokens |
| [[accelerators/amd-instinct-mi300x]] | 246.7 | 493.4 | 247 | 493 tokens |
| [[accelerators/nvidia-rtx-4090]] | 163.9 | 327.7 | 164 | 328 tokens |
| [[accelerators/google-tpu-v5e]] | 240.5 | — | 241 | 481 tokens |
| [[accelerators/nvidia-a100-80gb-sxm4]] | 153.0 | — | 153 | — |
| [[accelerators/nvidia-b200]] | 312.5 | — | 313 | — |
| [[accelerators/nvidia-l4]] | 403.3 | — | — | 807 tokens |

Two readings that change how you use this table:

**The ridge tells you where the memory wall ends, not how wide it is.** The record's own
warning: "an RTX 4090 has a LOWER bf16 ridge than an H100 (163.9 vs 295.4) and **is still
dramically worse at decode**... The ridge tells you WHERE the memory wall ends, not HOW WIDE
it is. Wall width is what matters at batch 1, and there the ranking is set by **absolute
bandwidth** (MI300X 5.3 TB/s > H100 3.35 TB/s > RTX 4090 1.008 TB/s) with **no reference to
FLOPS at all.**"

**Every data-center part crosses in the same band.** [[flops/decode-gemm]]: "every
data-center part here crosses at B of roughly **150-600 in bf16**, so **no single-stream
serving configuration is ever compute-bound** on the weight GEMMs, and the parts differ in
HOW FAR below the ridge they sit at B=1 (a 4090 is 164x below, an H100 295x below) but not
in whether they are."

> **One caution the record issues against its own numbers:** because AI rises exactly linearly
> in B and the ridge is a fixed property of the part, "the crossover batch is a fully
> determined number rather than a range." That is true of the *roofline*; the earlier caveat
> below about real kernels crossing earlier still applies.

### 3.2 Prefill attention: AI = L flop/byte

[[flops/prefill-attention]]: flops = 4·L²·d, bytes = 4·L·d, so **AI = L**. Consequences
the record states:
- Crosses memory→compute on H100 at **L = 591 tokens** for MHA (g=1), **148** for GQA-4 and
  **74** for GQA-8 — "GQA makes it happen **sooner**, not later, which surprises people who
  read GQA as a memory optimization only." In fp8 the ridge doubles, so every L_ridge
  doubles too.
- At **L = 8k on H100 with GQA-4, AI = 16,384 flop/byte — 55x the ridge**. "Long-context
  prefill is a tensor-core problem, not a bandwidth problem."
- Per-part MHA crossings: **328 tokens on an RTX 4090, 481 on TPU v5e, 807 on an L4**. The
  record's hardware conclusion: "the 4090 reaches compute-bound attention over 1.8x sooner
  and then has **6x less compute to do it with**".
- The record's warning for hardware selection is the important one: "a part whose peak
  FLOPS is weak relative to its bandwidth will look fine on short prompts and fall over on
  long ones. Hardware that is bandwidth-rich and FLOPS-poor... will pass a short-prompt
  benchmark and fail a 32k-context one."

### 3.3 Decode GEMM: AI = B flop/byte

[[flops/decode-gemm]]: "**a batch-1 decode GEMM sits ~295x below the ridge**," and the
crossover is now a determined per-part number rather than a range (§3.1 table): **H100 SXM
B_ridge = 295** bf16 / **591** fp8; **MI300X 247 / 493**; **RTX 4090 164 / 328**; **B200 313**;
**A100 SXM 153**; **TPU v5e 241**. The record's own aggregate reading: "every data-center part
here crosses at B of roughly **150-600 in bf16**, so **no single-stream serving configuration
is ever compute-bound** on the weight GEMMs."

And the crucial caveat: "Real kernels hit their crossover **earlier** than the ideal roofline,
because the memory-bound path reaches a higher fraction of peak bandwidth than the
compute-bound path reaches of peak FLOPS, so measured crossovers for bf16 GEMV-style decode
land well below the 295 batch figure." Real kernels cross *below* the roofline number — do not
use 295 as a floor.

Width does not help: "hidden_width / intermediate_size (bytes and flops both scale with it,
so AI is unchanged)."

### 3.4 MoE experts: AI = B·(k/E) — the worst crossover in the repo

[[flops/moe-experts]]: with E experts and top-k, **AI = B·(k/E) flop/byte**, because you
must stream *every* expert weight to do k/E of the work — "a batch large enough to touch
most experts does not reduce the bytes, because every expert is still needed by somebody."

Worked against DeepSeek-V3 (256 routed experts, 8 activated, k/E = 1/32): "an MoE decode
step needs a batch of about **32·295 = 9,440** to reach compute-bound where the same-width
dense model needs ~295."

Published k/E ratios, all from the same record: DeepSeek-V3 1/32, DeepSeek-V2 1/26.7,
Qwen3-Next **1/51.2** (the most aggressive in the repo, 512 experts / 10 per token),
OLMoE 1/8.

Also: only `weight_dtype` reduces the bytes at all, "since the flop count is not the
problem."

### 3.5 Decode attention: AI = g, the **GQA group ratio** — and always below the ridge

[[flops/decode-attention]] carries an explicit **CORRECTION** to its own earlier text, and
this is the single most important correction in the flop set:

> "it stated AI = 1 flop/byte unconditionally. **That is true only for MHA (g=1).** For
> Llama-3-8B (32 query / 8 KV heads) **AI = 4**; for Llama-2-70B (64 query / 8 KV heads)
> **AI = 8**; for MQA with 32 query heads **AI = 32**. ... AI = n_q/n_kv = g, the GQA group
> ratio, **independent of B AND independent of L**."

Derivation: flops = 4·L·d_h·n_q, bytes = 4·L·d_h·n_kv, so **AI = n_q/n_kv = g**. With
int8/fp8 KV the bytes halve and **AI doubles to 2g**.

**The gap to the ridge, in the record's own table** (against H100's 295.4):

| Configuration | AI | Gap below ridge |
|---|---|---|
| MHA, bf16 KV | 1 | **295x** |
| GQA-8, bf16 KV | 8 | 37x |
| GQA-8, int8 KV | 16 | 18x |
| MQA with 64 query heads, int8 KV | 128 | 2.3x |

The record's conclusion: **"NO standard attention configuration reaches the compute roof, at
any context length, at any batch size, at any dtype."**

What that rules out is the important part. Continuous batching is the standard cure for
memory-bound decode — and here it "does nothing," because "**batch size does not appear in the
AI expression at all**." So "long-context ITL degrades linearly in L for every sequence in the
batch at once, and **there is no batch size at which the wall recedes**."

Independent corroboration, quoted in the record: KIVI's abstract makes the two-part failure
statement — as batch and context grow the KV cache "becomes the new bottleneck in speed and
memory usage," and "the loading of the KV cache causes the **computational core to be idle**,
which limits the inference speed." **An idle compute core is this result in plain words.**

What does work, in the record's order of how much AI each buys: int8/fp8 KV (**x2**),
GQA/MQA at the model's own group ratio (**xg**, a model-selection decision not a serving one),
then the structural family — MLA, sliding window, sparse/hardware-aligned attention (NSA), or
replacing attention layers entirely.

### 3.6 The MLA absorption break-even: 36,400 tokens

[[flops/mla-absorbed-upprojection]] is the one flop record in the repo that gives a
*token-count* crossover rather than a batch or arithmetic-intensity one. If you decline
MLA's absorption trick and up-project the cached latent every step, DeepSeek-V3's shapes
cost **20,971,520 elements = 41.9 MB of bf16 weights read per layer per step** at AI = 1
flop/byte, "and this repeats for all 61 layers: **~2.56 GB of weight traffic per decode
step** that the absorbed form does not pay."

**Break-even context length: 36,400 tokens.** Below it, absorption is strictly correct.
Above it, "an engine may legitimately choose to spend bandwidth on recomputation to buy
batch size, and this is a real dial rather than an implementation detail."

This is genuinely rare — [[flops/mla-latent-attention]] calls it "the rare serving
optimization that ADDS arithmetic to REMOVE memory."

### 3.7 Quantization batch crossovers

[[flops/quantization-overhead]]: a 4-bit weight format "cuts weight bytes 4x versus bf16, so
decode GEMM AI goes from ~B to ~4B flop/byte and the effective ridge crossing batch drops by
roughly the compression factor." The measured crossovers, from [[flops/marlin-weight-only-int4-gemm]]:
- **batch up to 16-32**: still close to the maximum ~4x quantization speedup.
- **batch up to 64-128**: gradually decreasing but still significant acceleration.
- **up to 2.8x end-to-end** integrated with vLLM.

And the engineering conclusion in [[flops/quantization-overhead]]: "The reason
quantization speedups are quoted as 'up to 4x' is that they are **batch-conditional, and
the condition is a batch number**. An engine that quantizes for a latency target (batch 1)
gets nearly the full win; an engine that quantizes to serve high concurrency gets a
decaying fraction of it, because batching already raised AI and the weight bytes stopped
being the binding constraint."

A related threshold from [[flops/llamacpp-mmq-int8-dp4a-gemm]]: llama.cpp uses DP4A
"only while fp16 tensor cores are unavailable or the batch is small
(MMQ_DP4A_MAX_BATCH_SIZE 64)."

### 3.7b The 4-bit crossover, computed: **B_ridge ≈ 73.8** on H100 SXM

[[flops/quantized-decode-high-batch]] is the record that closes the loop left open by
[[flops/quantization-overhead]], and it is the most decision-relevant number in this
document. Because a 4-bit format cuts bytes 4x while leaving flops unchanged, AI becomes
**4·B rather than B**, which moves the ridge-crossing batch **down by the same factor**:

| Weight format | AI | B_ridge on H100 SXM (dense bf16 ridge 295.4 flop/byte) |
|---|---|---|
| bf16 | B | **295.4** |
| fp8 / int8 | 2·B | **147.7** |
| 4-bit | 4·B | **73.8** |

**The record's own claim is that this derivation predicts published behaviour:** the ideal
4-bit crossover of 295.4/4 = 73.8 "falls inside the **16-128 band MARLIN actually measures**"
(§3.7 above), so the two-line derivation AI = bits_factor · B holds against measurement
rather than being a roofline armchair result.

Two practical corollaries:
- **Every halving of bits halves the tolerable batch.** You can serve far more concurrent
  users per card at 4 bits than at 8.
- **Group size is a bandwidth number, not just a quality knob.** The record: "g=16 gives
  **20% overhead**, g=128 gives **2.8%**" — scale bytes as a fraction of weight bytes.

And the boundary condition: a **native** fp8/int8 tensor-core path "fuses dequant into the
load; an emulated dequant-to-bf16 path pays a separate pass that **batching cannot
amortize**." That is the hardware-native-vs-emulated distinction from
[04-quant-selection.md](04-quant-selection.md) arriving as a performance fact.

### 3.7c Tensor parallelism gets *harder* as you batch

[[flops/tensor-parallel-allreduce]] is interconnect-bound (AI = 0) and its cost model is
counter-intuitive: **payload is 4·B·d bytes per rank per layer — linear in B — while the
weight read it competes with is constant.** "Which is exactly why **TP gets harder as you
batch more**."

The dominant multiplier is per-rank link bandwidth, and the spread is enormous on identical
silicon: the record gives B* "from **184 on a single 25 GB/s NVLink 4 link to 3,318 on a
full 450 GB/s per-GPU domain — an 18x spread**." Two structural notes: cost per rank is
roughly flat in N (ring volume 2(N-1)/N saturates near N≥4) "so the win is that **P is
divided by N**"; and **only row-parallel layers need the all-reduce** — column-parallel needs
an all-gather of *t* tokens instead, "and in PREFILL t ≫ B, which **inverts the
comparison**."

### 3.8 The SFU bottleneck: a generational asymmetry, not a roofline one

[[flops/flashattention-3-hopper-warp-specialization]] gives the numbers that decide whether
attention is tensor-core-bound at all. On H100 SXM5: "the tensor cores do 989 TFLOPS FP16
but only **3.9 TFLOPS of special functions** (16 ops/SM/clock · 132 SM · 1.83 GHz), and at
head_dim 128 there are ~**512x more matmul than exponential FLOPs** — so without overlap,
exp can take **50% of the time of the matmuls**, and in FP8 it gets worse because matmul
doubles while exp does not."

[[flops/flashattention-4-blackwell-tmem]] then shows the bottleneck *moving* off tensor
cores entirely. Per B200 SM at M=N=D=128:
- **Forward: 1024 tensor-core cycles vs 1024 exp cycles vs 768 SMEM cycles** — bound by
  compute *and* exponential simultaneously.
- **Backward (5 chained MMAs): 2560 tensor-core, 1024 exp, 3328 SMEM** — "backward is bound
  by **shared-memory bandwidth**, not FLOPs."

The paper's argument, in the record's words: "tensor core throughput grows far faster than
SFU count and shared-memory bandwidth (BF16 tensor cores go from **1 to 2.25 PFLOPs
H100→B200** while SFU count and SMEM bandwidth are unchanged)." This is a generational
scaling asymmetry every Blackwell kernel author has to work around, and it means **FLOPs
per byte stops being the right metric for attention on Blackwell**.

### 3.9a Long-context RAG ingest is the **densest step in the serving stack**, and it is compute-bound

[[flops/long-context-rag-prefill]] exists to correct a naming trap: "'prefill_attention' as a
class name invites the reading that long context means attention-bound, and **the arithmetic
says the opposite**." Worked on Llama-3-8B from its own config (d=4096, 32 layers, 32 query /
8 KV heads, head_dim 128, vocab 128256), total non-embedding parameters come to 6.98B plus
1.05B untied embeddings, and the whole 32k ingest runs at **~124,600 flop/byte** — "firmly
compute-bound" and roughly 420x the H100 ridge point.

The crossover that decides it is **L\* = P_layer/(n_q·dh)**: **53k tokens for an 8B-class
model and 104k for a 70B-class one**. Below L\*, prefill GEMMs dominate; above it, attention
dominates. The counterintuitive consequence the record draws out: **a 70B model at 32k is
LESS attention-bound than a 7B model at 32k**, because a wider MLP per layer pushes L\*
further out. So "long context" does not map to one hardware requirement — it maps to a
threshold that moves with model width.

### 3.9b Reranking and batch encoding are compute-bound at batch 1 — the inverse of a decode part

[[flops/cross-encoder-reranking]]: "Reranking N passages is N INDEPENDENT forward passes with
no KV-cache reuse, so it is **prefill-shaped and compute-bound even at batch 1** — the exact
opposite of the generative decode regime." AI is **~t flop/byte** from weight amortization
alone, "t times HIGHER than the ~1 flop/byte of a decode GEMM." Cost knobs: N (linear — so
rerank top-20 vs top-100 "is a 5x difference in a bill that nothing else can reduce") and
pair length t.

[[flops/embedding-batch-encoding]] is structurally the same but **linear** in corpus size
(each document encoded once and cached), with the same batch threshold logic:
**AI = B·t**, so compute-bound when **B·t ≥ 295.4** on H100 — i.e. **B ≥ 3 at t = 128**.
Corpus work pushes B far above that, which is why embedding ingestion wants FLOPS and decode
wants bandwidth.

### 3.9c Prefix caching changes the profile, not just the latency

[[flops/prefix-cache-hit-miss]] is "a single workload with two genuinely different
arithmetic-intensity profiles, and conflating them is the single most common error in prefix-cache
benchmarks." The record quotes vLLM's own documentation for the asymmetry practitioners get
wrong: prefix caching "**only** reduces the time of processing the queries (the prefilling
phase) and **does not** reduce the time of generating new tokens (the decoding phase)."

And the deployment insight: "system prompts and few-shot blocks are shared across **ALL**
requests, so a deployed multi-tenant service can have a high steady-state hit rate that a
**single-user benchmark will never show**." This is directly relevant to the gotcha
[[gotchas/eagle-prefix-cache-last-block-drop]] (6-10 points of hit rate lost to spec decode).

### 3.9d Constrained decoding and low-entropy decode: both are vocab-sized, and entropy is not a cost driver

[[flops/constrained-decoding-grammar-mask]]: grammar masking is "the LAST operation applied
to the logits before the token is chosen, it consumes the same V-wide tensor the sampler
consumes, it runs exactly once per generated token." Naive CFG evaluation is O(stack_depth · V)
per token; the bitmask formulation is **15.66 KiB/token at V=128256, 3.9 KiB at V=32000**. The
real risk is stated plainly: "a mask kernel **ADDED to an already launch-bound decode step**"
(§4, [[flops/custom-kernel-launch-overhead]]).

[[flops/low-entropy-greedy-decode]] answers the obvious question with a negative: "**kernels can
exploit low-entropy decode much less than intuition suggests**, and the reason is ordering."
For Llama-3-8B (V=128,256, d=4096) the logits are **256.5 KiB/token** in bf16 against a step
reading **16.06 GB** of weights — 0.0016% of the step's HBM traffic. Entropy is **NOT** a cost
driver because "bytes must be read before the entropy is known, so a greedy token costs the
same as a uniform one." The one knob that does reduce work: a top-256 candidate set reads
**2.10 MB instead of 1.051 GB** of head — "a **500x** reduction of that term."

### 3.9 KV movement, not KV arithmetic, is the long-context cost

[[flops/kv-transfer]]: AI = 0 — "no flops at all, so every definition of this class is a
bandwidth or interconnect problem." Volume per token is
2 × num_kv_heads × head_dim × layers × kv_dtype_bytes; "for a 32-layer, hidden-4096,
GQA-8-head, bf16 model that is on the order of hundreds of KB per token." At 32k context
that is "several GB of KV per sequence, which is why PagedAttention's near-zero-waste block
management was worth **2-4x end-to-end throughput** at the same latency vs Orca and
FasterTransformer, more so at longer sequences and larger models."

Two multipliers worth carrying:
- **Tensor parallel**: "a KV block sharded across N devices must be gathered before
  attention, adding **N-1 transfers per block read**."
- **Prefix sharing**: the dominant copy is not writing new KV but "the movement implied by
  preemption, rescheduling, prefix sharing, and multi-node sharding."

[[flops/lmcache-kv-tier-outside-attention]] adds the biggest single lever that lives
*outside* the kernel: "if a shared prefix KV is resident in GPU memory because a cache tier
kept it there, every request reading that prefix does **zero HBM traffic** for it. So the
practical AI of decode attention is set by **cache hit rate**, not only by the kernel's
tiling."

---

## 4. Memory-bound classes ranked by how badly they behave

All at bf16 unless stated. Ranked by how far below the ~295 flop/byte ridge they sit.

| Class | AI | The knob that would move it |
|---|---|---|
| [[flops/norm-softmax]] | **~0.5 flop/byte** | fusion only — "half a flop per byte, the worst ratio of any class here" |
| [[flops/embedding]] | **~0** for the input gather; ~1 for the LM head | vocab_size; the gather's cost is *latency*, not bandwidth |
| [[flops/sampling]] | **1-2 flop/byte, independent of batch** | fusion; and it costs *variance*, not average time |
| [[flops/decode-attention]] | **1** | fp8 KV (→2), GQA/MQA, MLA |
| [[flops/decode-gemm]] | **= B** | batch, or fewer bytes (quantization) |
| [[flops/speculative-draft]] / [[flops/speculative-verify]] | **= k** | draft_k |
| [[flops/ssm-recurrent-state-update]] | **~1** on a *constant-size* state | batch — "the inverse of every other decode class" |
| [[flops/kv-transfer]] | **0** | prefix reuse, GQA, KV dtype, TP degree |
| [[flops/moe-experts]] | **= B·(k/E)** | batch (E/k times harder than dense) |
| [[flops/decode-gemm]] in fp4 weights | **≈ 4B** | batch, now moved 4x closer to the ridge |

Three notes that change how you read this table:

**Elementwise is not small.** [[flops/norm-softmax]] is "a few percent of the flops and can
be a **double-digit percentage of the time** in a bandwidth-starved decode step, because
decode is exactly the regime where bandwidth is the wall." Any engine that leaves norms,
activations and residual adds as separate kernels "is paying an avoidable HBM round trip per
layer per token."

**Sampling is a variance problem, not a throughput problem.** [[flops/sampling]]: its cost
per token "is roughly fixed regardless of batch, so at high batch it becomes a small share
of step time; at batch 1 it is a real share of it." Its design consequence: "its cost shows
up as **inter-token latency variance** — a latency spike at the end of every step." And the
catch-all most people forget: on large-vocab models "the LM head weight matrix... for big
vocabs is a **second full decode GEMM**."

**SSM inverts the whole model.** [[flops/ssm-recurrent-state-update]]: "this class has no
KV cache at all, which inverts the entire decode cost model." The state is per-sequence and
**does not amortise with batch** — "the inverse of every other decode class" — but it is
**constant in context**, so long-context cost stops growing. The two records together make
the honest summary: [[flops/ssm-sequential-scan-tensor-core-mismatch]] records that the
"linear-time, constant-state claims are real and verified; the implication that they are
therefore FASTER is not, because the arithmetic that replaces attention is exactly the
shape tensor cores are worst at." Nemotron-H claims "up to 3x faster at inference" — 3x,
not 10x, despite removing 92% of attention layers.

---

## 5. Compute-bound classes, and what they cost you

**Prefill** is the one place peak dense FLOPS is the right metric. Beyond L≈300 tokens it
is a tensor-core problem (§3.2), and the scaling mechanisms all attack memory traffic so
that FLOPS can be reached:

- [[flops/flashattention-io-aware-tiling]] — the base mechanism, and the claim is precise:
  it "reduces HBM traffic by removing the S x S materialization", **not** "reduces attention
  FLOPs". Before it, "a part with weak FLOPs but strong bandwidth could not hide the S x S
  round trip, so long-context attention failed on memory traffic alone; after it the same
  part is limited by tensor-core throughput."
- [[flops/flashattention-2-work-partitioning]] — a pure scheduling fix, changing MMA
  *utilisation*, not arithmetic intensity. FA-1 reached "only 25-40% of peak FLOPS/s";
  FA-2's target was roughly 2x.
- [[flops/sliding-window-hybrid-attention]] — architectural, and the cheapest lever found.
  Gemma 3 at w=1024 with 5 local per 1 global: model-wide prefill attention cost is
  (5/6)·L·w + (1/6)·L·L. "Gemma 3 shows perplexity is insensitive down to 7:1 and the cache
  saving continues, so this is **close to free quality**," and the window itself "can be
  reduced significantly without impacting perplexity" (tested 512/1024/2048/4096/8192).
  Published ratios in the repo: Gemma 3 5:1, Gemma 2 1:1, GPT-OSS 12:12 at w=128,
  Ministral-3 1:4, Kimi Linear 3:1, Samba w=2048 interleaved with Mamba.
- **The flop crossover, which is more decisive than the ridge crossing.** Attention flops grow
  as L² while GEMM flops grow as L, so the two trade at **L\* = P_layer/d** (for standard
  models where n_q·d_h = hidden_size, i.e. roughly the feed-forward expansion ratio).
  Computed from the models' own configs in [[flops/prefill-attention]]: **Llama-2-7B
  L\* = 49,408**, **Llama-2-70B L\* = 104,448**, **Llama-3-8B L\* = 53,248**. The resulting
  share of prefill flops spent in attention for Llama-2-7B is **1.0% at L=512, 4.0% at 2k,
  14.2% at 8k, 39.9% at 32k, 72.6% at 128k**; for the much wider Llama-2-70B the same shares
  are **0.5%, 1.9%, 7.3%, 23.9%, 55.7%**. The record draws the lesson out: "**a WIDER model
  defers the attention crossover further out**, so a short-prompt benchmark and a 32k-prompt
  benchmark on different models **disagree about which part is fast for reasons that are
  entirely in the config**."
- [[flops/modern-attention-variants]] — the taxonomy matters and the record groups them:
  **YOCO** reduces cache *size* by sharing across layers with no math change; **NSA/DSA**
  reduce the *work set* but "add a full-context index pass and a non-contiguous gather";
  **the linear/retention family** removes the cache entirely and moves you into the SSM
  regime. NSA quantifies the prize: "attention computation with softmax architectures
  accounts for **70-80% of total latency** when decoding 64k-length contexts."
- [[flops/diffusion-lm-denoising-steps]] — inverts everything. "In an AR decode loop the
  per-token flops, the per-token bytes and the per-token latency are all constant and known
  in advance... In diffusion LM generation, the cost per emitted token is a function of two
  user-facing hyperparameters (S and Lg)." Flops/token scale linearly in denoising steps S
  and inversely with generation length Lg. This is the one class where **"buy FLOPS" is
  unambiguously right and "buy bandwidth" is not.**

**Quantization FLOP removal** — [[flops/marlin-weight-only-int4-gemm]] and
[[flops/machete-weight-only-hopper-gemm]] do not change the FLOP count at all; they
"remove the dequantization tax by moving it offline". Marlin's design target is stated as a
ratio: "as long as we perform **less than 25-50 tensor-core multiply-accumulates per 4-bit
quantized weight**, the FLOP:byte ratio of the GPU permits near-ideal 4x speedup", and it
"should extend to batch sizes 4-8x larger than prior kernels achieved."

The contrast that matters for hardware choice is
[[flops/llamacpp-mmq-int8-dp4a-gemm]] vs those two: MMQ "performs the dot product in INT8
accumulating to INT32, using DP4A where available. So the multiply-accumulates are integer
SIMT instructions, not tensor-core MMAs — the FLOP:byte ratio the tensor cores exploit is
not available at all." Its per-arch configs cover "Ampere, Blackwell, CDNA, GCN, RDNA2/3/4/5
and Pascal", so "on a consumer or datacenter GPU, llama.cpp reaches 4-bit throughput
through integer SIMT dot products with a **broad arch reach**... while Marlin and Machete
reach it through tensor cores on a **narrower arch list**. Neither is strictly better."

Note Machete's `bound_by: compute` is not an accident: it keeps the Hopper wgmma mainloop and
changes only the B *operand layout*, so "this is a tensor-core bf16/fp16 GEMM over int4
weights dequantized into the MMA fragments on the fly, **NOT a hardware int4 tensor-core
path**."

**Block-scaled tensor cores** are the only mechanism that turns precision reduction into
FLOPs rather than a dequantization tax ([[flops/block-scaled-tensor-core-fp4-fp8]]): "the
scale is part of the **MMA OPERAND**, not a separate kernel, so a 32-element scale block is
applied inside the tensor-core pipeline and the activation quantization no longer needs a
fused dequant pass." It scales with "precision (fp8 → fp4 doubles then quadruples
tensor-core throughput)" — but it has a portability tax recorded in
[[flops/deepgemm-grouped-masked-moe-gemm]]: "SM90 requires scaling factors in FP32 while
SM100 requires them packed in UE8M0 with four packed into a single int, and the LHS scale
must have a TMA-aligned transposed layout for both. So switching an FP8 model from H100 to
B200 changes the memory layout the engine must produce, not just the kernel it calls."

---

## 6. Interconnect-bound classes

Three classes are bound by links rather than by local memory, and all three are MoE or
multi-node:

- [[flops/moe-routing]] — "routing is a network problem wearing an arithmetic costume."
  The router GEMM itself is negligible (DeepSeek-V3's router weight matrix is 1.83M
  elements "against 671B"). The expensive part is "the permutation that moves each token to
  its k selected experts' devices, a network transfer whose cost is
  **k·t·(hidden bytes)/link bandwidth**". Its bounded-ness comes from device-limited routing:
  the all-to-all fan-out per token is bounded by **M=4 for DeepSeek-V3, not by k=8**.
- [[flops/deepep-expert-parallel-dispatch-combine]] — hardware scope is explicit and worth
  quoting: "EP dispatch and combine **require GPU SMs**; zero-SM RDMA EP is not supported",
  requires **NVIDIA Hopper or newer**, NVLink intranode plus RDMA internode. So on a part
  without NVLink, "DeepEP is not the path, and expert parallelism becomes bandwidth-bound on
  the slowest transport rather than hidden behind MMA."
- [[flops/kv-transfer]] — TP degree adds N-1 transfers per block read (§3.9).

The hardware consequence is the same in all three: **judge MoE hardware on all-to-all
latency and on whether expert parallelism can be laid out to keep most experts resident on
one device**, not on router FLOPS or even on HBM bandwidth.

---

## 7. Where the records contradict each other on FLOPs

Both contradictions are preserved, not resolved, per the repo's stated policy.

**1. NVIDIA's sparsity convention is not self-consistent across datasheets.**
[[accelerators/nvidia-b300]]'s Blackwell Ultra sheet prints FP4 as '20 PFLOPS | 15 PFLOPS'
under footnote "Specification in Sparse | Dense" — a sparse|dense pair, i.e. **the opposite
convention** from the "dense is one-half of sparse" used on the [[accelerators/nvidia-b200]]
and [[accelerators/nvidia-h100-sxm]] sheets. So dense FP4 on B300 is 15 PFLOPS, not 10. The
record is `contested` (confidence 0.6), partly because of this and partly because its INT8
row reads 330/307 TOPS against ~10 PFLOPS implied by its own other rows — flagged as a
probable broken text-layer extraction, with INT8 therefore not recorded at all. **Both
conventions are in the data; neither is declared correct.**

**2. AMD publishes two different peak numbers for the same datatype on the same part.**
[[accelerators/amd-instinct-mi355x]] (also `contested`): the product page gives OCP-FP8 as
5.0 PFLOPs dense / 10.1 sparse and MXFP4 10.1, while datasheet GD-176 gives OCP-FP8
5.0332/10.0664, MXFP4 10.0663. The record keeps both and says why: "the ~0.06% spread is
rounding to 4 significant figures vs 2... **Recorded both because they are published as
different figures.**"

**3. A subtler one, relevant to any roofline reasoning:** fp16 on CDNA4 has two different
values depending on which path you mean. [[accelerators/amd-instinct-mi355x]] deliberately
records 157.3 TFLOPS (vector) and 2500 TFLOPS (matrix) — "CDNA4 is the first AMD part where
fp16 vector and fp16 matrix differ by ~16x on the same die. **A record that only lists 2500
loses real information.**" Any comparison that does not say which path it means is
meaningless.

**4. And one that affects MoE planning:** [[accelerators/amd-instinct-mi300x]] lists
tf32 at 653.7 dense TFLOPS, but [[accelerators/amd-instinct-mi350x]] and MI355X have **no
native TF32 matrix path at all** — AMD's footnote says "TF32 support through software
emulation". The record's instruction is unambiguous: "**Do not carry a MI300X TF32 figure
forward to MI355X.**"

---

## 8. What the records cannot answer

1. **Roofline balance points for any part other than H100 SXM.** The single most-requested
   number for hardware selection, and the repo does not have it. `docs/NOTES.md` lists this
   as an open question. Everything in §3 that says "295 flop/byte" is H100 SXM; on H200 or
   B200 or MI355X the ridge sits somewhere the records do not state.
2. **No measured crossover batch for any real model on any real part.** The crossovers in
   §3 are analytic (roofline-derived), and [[flops/decode-gemm]] warns real kernels cross
   *earlier* than the analytic figure. No record measures where it actually happens.
3. **No attribution of step time to flop classes.** The records give per-class AI and
   crossovers; nothing decomposes an observed decode step into GEMM / attention / norm /
   sampling shares. So the relative importance ranking in §4 is structural, not measured.
4. **Sparse-mode performance is almost entirely unrecorded.** [[flops/structured-2-4-sparsity-mma-sp]]
   notes "Ampere's sparsity support was largely dropped from the datacenter line after a
   sparsity-format deprecation, so a sparsity number on an A100 datasheet is not a promise
   about a Hopper or Blackwell part; **this record deliberately does not assert that
   deprecation**" — and the record's own confidence is 0.75, the lowest in the flop set.
5. **Roofline crossovers exist for exactly one part.** [[flops/quantized-decode-high-batch]]
   gives B_ridge = 295.4 / 147.7 / 73.8 for bf16 / fp8 / 4-bit, and
   [[flops/tensor-parallel-allreduce]] gives B* from 184 to 3,318 across fabrics — but every
   one is anchored to H100 SXM's 295.4 flop/byte ridge.
6. **No attribution of step time to flop classes** (see 3 above) — still true, and now more
   pointed, because the newer records show the *shape* of the profile changes with workload
   type (miss vs hit, prefill vs decode, generate vs embed) even though the split within a
   step is still unmeasured.
7. **No roofline for the SSM/recurrent path.** [[flops/ssm-sequential-scan-tensor-core-mismatch]]
   explains qualitatively that the sequential scan bypasses tensor cores and that chunkwise
   reformulation is the fix, but gives no intensity figure for either form.

---

## Related documents

- [01-hardware-selection.md](01-hardware-selection.md) — which part to buy for each of these regimes.
- [03-engine-selection.md](03-engine-selection.md) — which engine can reach these crossovers on your part.
- [04-quant-selection.md](04-quant-selection.md) — the one lever that changes AI without touching the model.
- [05-known-traps.md](05-known-traps.md) — the software failures behind §3's crossovers not materialising.