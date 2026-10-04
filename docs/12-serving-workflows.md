# 12 — Serving workflows: where the time actually goes, and why it moves

<!-- Authored against the flop, benchmark, accelerator and model records present on 2026-10-03.
     Every number below cites the record it came from. Where no record exists, this document says
     "no record" rather than estimating. This document does NOT close the open question
     docs/02-flop-map.md §8 item 3 raises (measured attribution of step time to flop classes); it
     walks the attribution analytically and says which measurements exist and which do not. -->

This is the document for the question the FLOP map leaves open. `docs/02-flop-map.md` gives each
class an arithmetic intensity and a crossover, and then says in §8 item 3: *"No attribution of
step time to flop classes... nothing decomposes an observed decode step into GEMM / attention /
norm / sampling shares."* This document does the walk end to end, four times, in four regimes.

**The thesis, and it is the reason to read this before buying anything:** the bottleneck is not a
property of the model, it is a property of the *regime*, and it moves. For Llama-3-8B in bf16 the
weight read alone is **2 × 8.03e9 = 16.06 GB per decode step** ([[flops/decode-gemm]] with
[[models/llama-3-1-8b]]), and *every other class in the taxonomy is a rounding error against it in the same
regime*. But that same model in a 32-slot server on [[accelerators/nvidia-rtx-5090]] spends
**41.2% of its time with the GPU idle** waiting on a CPU sampler
([[flops/host-side-sampling-loop]]), and a sub-1B model on an H100 can have its GPU outrun its own
scheduler ([[flops/custom-kernel-launch-overhead]]). Same model maths, three different dominant
classes.

---

## 1. The reference model and the numbers everything else divides into

Worked Llama-3-8B (Llama-3.1 config: `params_b` 8.03, hidden 4096, 32 layers, 32 query / 8 KV
heads, head_dim 128, vocab 128256 — [[models/llama-3-1-8b]], geometry cross-checked against
[[sources/flop2-llama3-8b-config]]). All figures below are derived from those records; the derivations
are shown so they can be re-checked.

**Decode step bytes (bf16, batch B=1).**

| term | bytes | record |
|---|---|---|
| weight GEMMs (2·P) | **16.06 GB** | [[flops/decode-gemm]] |
| decode attention at L=2k | 256.0 MiB | [[flops/decode-attention]] |
| decode attention at L=32k | 4.00 GiB | [[flops/decode-attention]] |
| KV bytes/token | 128 KiB | [[models/llama-3-1-8b]] |
| logits row (fp32) | 501.0 KiB | [[flops/sampling]] |
| grammar bitmask (V/8) | 15.66 KiB | [[flops/constrained-decoding-grammar-mask]] |
| RMSNorm/softmax activations | ~0.5 flop/byte, per t×d read+write | [[flops/norm-softmax]] |

At L=2k the attention term is **1.7%** of the step; at 32k it is **27%**. That single ratio is the
long-context story in one number, and it is why §5 exists.

**Prefill flops (32k ingest).** GEMM = 2·P·L = **526 TFLOP**; causal attention = 2·n_q·d_h·L² =
**8.80 TFLOP**, i.e. attention is **1.7%** of prefill flops — the GEMMs dominate by ~60x
([[flops/long-context-rag-prefill]]). Whole-step AI ≈ **124,600 flop/byte**, roughly **420x** the
H100 ridge of 295.4. Prefill is the only phase in the entire stack that wants FLOPS.

---

## 2. The four regimes, at a glance

The table is the document. Each cell is the class that *dominates*, and the reason follows in the
section named.

| | **(a) high-concurrency batched server** | **(b) single-user local session** | **(c) long-context RAG** | **(d) small model, fast GPU** |
|---|---|---|---|---|
| **Dominant class** | `host-side-sampling-loop` / `logits-processors-cpu-pipeline` | `decode-gemm` (memory bandwidth) | `long-context-rag-prefill` (FLOPS), then `decode-attention` (capacity+bandwidth) | `custom-kernel-launch-overhead` (per-kernel **and** per-step scheduler) |
| **bound_by** | launch_overhead | memory | compute → memory | launch_overhead |
| **What it scales with** | batch × vocab (host), *not* batch (device) | batch (AI = B) | ingested token count, then context length | params, concurrency, kernels/step, host CPU |
| **What would fix it** | move sampling/processors on device; fewer host round trips | raise batch (continuous batching) or cut weight bytes (quantization) | more FLOPS (prefill); smaller KV (fp8 KV, GQA/MLA, cache tier) | CUDA Graphs + fusion; a control plane that is not Python |
| **Arithmetic intensity** | ~1 flop/byte device-side, no roofline host-side | 1 flop/byte at B=1, 295x below H100 ridge | ~124,600 flop/byte in prefill | fixed µs per launch/step, independent of work |
| **Published evidence** | 32 slots: 705.9 → 1045.8 tok/s, TPOT 41.21 → 25.39 ms; sampler 58.6% → 1.0% of in-gap CPU ([[flops/host-side-sampling-loop]]) | batch 1 → 64 on MPT-7B: aggregate 800 vs 57.6 tok/s per user, a 13.9x gain per card ([[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]], [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]]) | no measured TTFT for a 32k RAG ingest: **no record**. Analytic: 526 TFLOP prefill, 4.00 GiB KV write ([[flops/long-context-rag-prefill]]) | scheduler CPU 11.7% of wall; user reports 4 ms CPU / 1.4 ms GPU per step ([[flops/custom-kernel-launch-overhead]]) |

Read the table as a prediction tool, not a summary: **which row are you in?** is answerable from
four questions — how many concurrent requests, how big the model is, how long the context, and is
the sampler on the host. Those four answers determine the dominant class; the rest of this document
is the derivation.

---

## 3. Regime (b): single-user local session — bandwidth, and only bandwidth

This is the regime the FLOP map describes best and the one people benchmark most often.

**Prefill.** For a 512-token prompt the whole prompt is processed in one batched step at AI = t
flop/byte from weight amortisation alone, which for t=512 is **512x the ~1 flop/byte of a decode
step** ([[flops/prefill-attention]]). Prefill is compute-dense; it is over in one step.

**Decode.** Weights are read once per step and reused across all B rows, so **AI = B exactly**, and
at B=1 that is 1 flop/byte — **~295x below** the H100 SXM ridge of 295.4, i.e. batch-1 decode GEMMs
are never compute-bound on any data-center part in the repo ([[flops/decode-gemm]]). The honest
statement of the floor: the step must move 16.06 GB, which is **4.79 ms at 3350 GB/s
(H100)** and **3.03 ms at 5300 GB/s (MI300X)** — and real kernels reach a *higher* fraction of peak
bandwidth than of peak FLOPS, so measured decode lands below the analytic figure. What this regime
buys is **bandwidth, not FLOPS**: at batch 1 the ranking of parts is set by absolute HBM bandwidth
with no reference to tensor cores at all, which is why a 4090 (1008 GB/s) loses to an MI300X
(5300 GB/s) at batch 1 despite the ridge numbers being closer (163.9 vs 246.7)
([[flops/decode-gemm]]).

**Everything else is a rounding error, and the arithmetic is worth doing once.** Against the
16.06 GB weight read:

| class | bytes/token | share of step |
|---|---|---|
| decode attention @2k | 256.0 MiB | **1.7%** |
| RMSNorm/softmax activations | ~0.5 flop/byte, read+write per t×d | small and fusable ([[flops/norm-softmax]]) |
| logits, fp32 | 501.0 KiB | **0.0031%** |
| grammar bitmask | 15.66 KiB | **0.0001%** |

So sampling "feels expensive" for a reason that is **not bytes**. The record says it directly:
sampling's cost "shows up as inter-token latency **variance** — a latency spike at the end of every
step," because it is small and batch-invariant ([[flops/sampling]]). At B=1 that spike is a visible
share of a 4.79 ms step; at B=256 it is invisible. Same code, different regime, different
importance — which is the whole point of this document.

**The measured anchor for this regime** exists and is directly interpretable: MPT-7B on one A100
40GB, 512 in / 64 out, static batching — **TTFT 46 ms** at batch 1
([[benchmarks/mpt7b-a100-bs1-ttft-ms]]) and **57.6 tok/s per user** at batch 1
([[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]]), i.e. ~17.4 ms per inter-token interval, with
TTFT ~2.6x one ITL. Two things follow directly from those records and the arithmetic above. First,
**prefill is not the bottleneck for a short prompt** — 46 ms of TTFT is ~43% of a 512-in/64-out
request, so for anything with a long answer decode dominates outright. Second, ITL of 17.4 ms at
batch 1 versus 4.79 ms of pure HBM time for an 8B is the *bandwidth efficiency gap*: real serving
achieves a fraction of the analytic floor, and that gap is where launch overhead, unfused
elementwise passes and kernel-count costs live. The repo does not decompose that gap — §8 says so
explicitly — so do not attribute it to any single class without measuring.

---

## 4. Regime (a): high-concurrency batched server — the host, not the GPU

Same model, same part, different answer. The reason is one structural fact: **decode_gemm gets
cheaper per token as B grows** (AI = B, so the 16.06 GB weight read is amortised over B rows), while
the per-token host work outside the model **multiplies**. So batching improves the class you cannot
avoid and worsens the classes you did not think about.

**What is measured.** llama.cpp's `llama-server`, RTX 5090 (1792 GB/s), Qwen2.5-7B-Instruct F16
(n_vocab 152064), 512 prompt + 128 generated tokens, continuous batching, 32 slots
([[flops/host-side-sampling-loop]]):

| | 1 slot | 32 slots |
|---|---|---|
| throughput | 100.1 tok/s | 705.9 tok/s |
| TPOT | 9.65 ms | 41.21 ms |
| D2H memcpy per copy | **608.3 KB** = exactly 152064 × 4 B, one fp32 logits row | **16,157.4 KB** |
| GPU busy | **88.5%** | **41.2%** |
| sampler's share of in-gap CPU | 5.1% (of a 0.185 s / 10 s idle budget) | **58.6%** |

That table *is* the regime shift, in one column transition: aggregate throughput rises 7.1x while
GPU utilisation **falls** from 88.5% to 41.2%, because the per-step work that does not scale is on
the host. The 608.3 KB figure is the mechanism made checkable — it is exactly one fp32 logits row
for that model's vocab, moved once per decode step.

**Logit processors add a second, discontinuous host cost.** With penalties enabled, vLLM
v0.6.2–v0.7.2 on A100 with Llama-3-8B-Instruct regressed from batch ~20 upward: TPOT 0.0356 → 0.24 s
at 30 users, 0.115 → 0.314 s at 100, 0.208 → 0.544 s at 190; generated TPM 33.3K → 5.24K at 30
users; CPU 1.43 → ~4 cores (the pod cap); GPU utilisation 81–100% → 0–25%
([[flops/logits-processors-cpu-pipeline]]). Same penalty path is **free below batch ~20**. The cost
model: every step, per-request `output_token_ids` Python lists are padded into a pinned CPU int64
tensor and re-uploaded — payload `max_output_len × batch × 8 B`, **growing as sequences lengthen**,
which is the worst possible scaling for a per-step cost. The device half of the same features is
irrelevant by comparison: `min_p` alone is four to five full-width passes over `[B, V]` at fp32,
which is **0.012–0.016%** of a batch-1 dense step's bytes.

**The cures are all host-side**, which is the operational lesson: device sampling (`-bs`) took the
same 32-slot workload to **1045.8 tok/s, TPOT 25.39 ms, sampler 1.0% of in-gap CPU**, with
byte-identical greedy output at temperature 0 ([[flops/host-side-sampling-loop]]). Control
experiments in the same report each moved throughput <10% — FlashAttention on/off, 64 slots at 64K
context, doubled HTTP threads — while removing the host round trip moved it **48%**.

**MoE changes this regime's shape completely.** With AI = B·(k/E), DeepSeek-V3 (256 routed experts,
8 per token, k/E = 1/32) needs batch ≈ **9,440** to reach compute-bound where the dense equivalent
needs ~295 ([[flops/moe-experts]]), so a high-concurrency server is *mandatory* for MoE, not
optional. And dispatch/combine is interconnect-latency work, not FLOPs: at B=32, k=8, d=7168 it is
B·k·d·2 B = 3.67 MB each way per layer, ≈ 220 MB per step across DeepSeek-V3's 61 layers, which is
~0.3% of the 2 × 37e9 = 74 GB of active expert weights streamed in the same step — but it sits on a
different resource and appears as bubbles and jitter
([[flops/moe-routing]]). The single highest-leverage inference-side decision is DeepEP's
`do_cpu_sync=False` decode path, which allocates to configured capacity and exposes
`psum_num_recv_tokens_per_expert` so the grouped GEMM finds valid ranges **without a host round
trip** ([[flops/deepep-expert-parallel-dispatch-combine]]).

**How to tell you are in this regime:** GPU utilisation *drops* as concurrency rises, TPOT degrades
super-linearly in slot count, throughput per card stops scaling past some slot count, and a profiler
shows large GPU-idle gaps. Any one of these is sufficient.

---

## 5. Regime (c): long-context RAG — prefill wants FLOPS, then KV wants capacity

Two different bottlenecks in sequence, and confusing them is the standard error.

**Phase 1 — prefill is the densest step in the entire stack.** For a 32k ingest:
**526 TFLOP** of GEMM against **8.80 TFLOP** of causal attention, i.e. attention is **1.7%** of
prefill flops and the GEMMs dominate ~60x ([[flops/long-context-rag-prefill]]). Whole-step AI
≈ **124,600 flop/byte**, ~420x the H100 ridge. The crossover that decides when this stops being true
is **L\* = P_layer/(n_q·d_h)** = **53,248 tokens** for an 8B-class model and **104,448** for a
70B-class one — and the counterintuitive consequence is that **a 70B model at 32k is *less*
attention-bound than a 7B at 32k**, because a wider MLP per layer pushes L\* further out. So "long
context" does not name one hardware requirement; it names a threshold that moves with model width.
Prefill also has to *write* the cache: **4.00 GiB** of KV at 32k for Llama-3-8B, a pure store at
AI = 0, costing **1.28–5.24 ms** depending on part. Prefill attention separately crosses from
memory- to compute-bound at **L_ridge = 2·ridge/g**: **591 tokens** for MHA on H100, **148** for
GQA-4, **74** for GQA-8 ([[flops/prefill-attention]]) — i.e. for any GQA model, prefill attention is
compute-bound almost immediately.

**Phase 2 — decode at long context is bandwidth and capacity, never FLOPS.** Decode attention has
**AI = g**, the GQA group ratio — **4** for Llama-3-8B, **8** for Llama-2-70B, 32 for MQA —
*independent of both batch and context* ([[flops/decode-attention]]). Two consequences that decide
the design: continuous batching, the standard cure for memory-bound decode, **does nothing here**,
because batch does not appear in the expression; and *"NO standard attention configuration reaches
the compute roof, at any context length, at any batch size, at any dtype."* At 32k the KV read is
**4.00 GiB per token per sequence** against 16.06 GB of weights — **27%** of the step, and it
grows linearly forever with no regime change.

**What actually moves it**, in the order the record gives: fp8/int8 KV (**×2 AI**), GQA/MQA at the
model's own group ratio (×g — a model-selection decision), then the structural family — MLA,
sliding window, sparse attention, or replacing attention layers. Cache-tier residency is the biggest
lever that lives *outside* the kernel: if a shared prefix is resident in GPU memory, every request
reading it does **zero HBM traffic** for it, so effective decode-attention AI is set by **cache hit
rate** ([[flops/lmcache-kv-tier-outside-attention]]). Prefix caching has the same asymmetry worth
stating explicitly: it *"only reduces the time of processing the queries (the prefilling phase) and
does not reduce the time of generating new tokens"*, so it gives no benefit when the answer is long
([[flops/prefix-cache-hit-miss]]).

**The served-workspace trap for this regime** is [[flops/detokenization-stop-string-scan]]: with a
warm 20k-token prefix, priming the detokeniser's DecodeStream with the *entire* prompt costs
**~12 ms of native detokenisation** (0.5–1.5 µs/token, tokenizers 0.22.2), landing verbatim in
TTFT *and* head-of-line blocking the SSE deltas of every other stream on the same event loop.
Bounded-tail priming (last 32 ids) reduces it to ~7 µs. The cost hits hardest exactly where GPU-side
TTFT is otherwise best — long warm-prefix agent and RAG requests behind a prefix cache.

**Measured anchor:** **no record.** The repo has TTFT and ITL measurements for short prompts
(e.g. [[benchmarks/mpt7b-a100-bs1-ttft-ms]] at 512 tokens) and per-token prefill throughput
([[benchmarks/mlx-m4-max-qwen3-4b-fp16-prefill-tok-s]], 1780.63 tok/s on a 2048-token prompt at
batch 1 on M4 Max), but **no end-to-end TTFT measurement for a 32k-context RAG request on any part
in any engine**. The 526 TFLOP / 4.00 GiB figures above are analytic, not measured. If you need the
measured number, this document cannot give it to you.

---

## 6. Regime (d): small model on a fast GPU — the host outruns the device

The regime nobody budgets for, and it is the one where buying a faster GPU is *least* effective.

**The crossover is computable.** A decode step's GPU time is ≈ 2·P/BW_device. For a 0.6B model in
bf16 that is 1.2 GB per step, or **358 µs at 3350 GB/s**. Compare the step's *fixed* CPU cost:
[[flops/custom-kernel-launch-overhead]] now records **two** distinct overheads, and the distinction
matters because they have different cures:

- **Per-kernel launch** — a fixed µs cost per kernel. Measured directly by NVIDIA on a V100: a
  2.9 µs kernel costs **9.6 µs** per launch with per-kernel synchronisation, **3.8 µs** with
  overlapping launches, and **3.4 µs** inside a captured CUDA Graph, at a graph instantiation cost
  of ~400 µs ([[sources/flop-nvidia-cuda-graphs]]). For a large model, "a few hundred to a few
  thousand kernels" per step is the regime CUDA Graphs were built for.
- **Per-step scheduler (control plane)** — Python work that CUDA Graphs **cannot** remove, because
  it is not a kernel. Measured on Qwen3-0.6B / H100 / `enforce_eager=True` / `num_seqs=256` over a
  12.68 s wall: `Scheduler.schedule()` **907 ms = 7.2%** and `update_from_output()` **572 ms = 4.8%**,
  **11.7% total** scheduler CPU — and vLLM's own code flags that loop as a bottleneck
  (`scheduler.py:1340`). Users report **4 ms CPU / 1.4 ms GPU per step**, i.e. the CPU plane ~3x the
  GPU work ([[flops/custom-kernel-launch-overhead]]). The same source states the fraction is much
  larger on exactly these workloads: small models, multimodal/TTS, higher batch, lower-end GPUs.

**So the regime test is: per-step GPU time vs per-step CPU time.** At 358 µs of GPU time for a 0.6B
model, any per-step host cost in the hundreds of µs makes the step host-bound — and the step time
stops responding to faster silicon entirely. That is the definition of "the fast GPU is not the
bottleneck."

**The per-token non-matmul work is worst here, for a specific reason.** It is *not* that its byte
count is larger (it is not — 0.0031% of the step for the logits, per [[flops/sampling]]); it is that
each of these classes carries a **fixed per-step or per-token cost that does not shrink when the
model does**. So as parameters → 0, the fixed costs become an unbounded share:

| class | why it hurts *more* for a small model |
|---|---|
| `custom-kernel-launch-overhead` | kernel count per step depends on **depth × ops per layer**, not on width; and the scheduler's per-step Python cost is independent of P |
| `detokenization-stop-string-scan` | per-token host text work, independent of P — a native per-token vocab lookup (the `len(tokenizer)` → `get_added_vocab()` case) starved the event loop until `/health` hung |
| `logits-processors-cpu-pipeline` | per-step host work with an `output_token_ids` payload that grows with output length; published TPOT regression is 4–7x above batch ~20 |
| `host-side-sampling-loop` | full `batch × vocab × 4 B` D2H per step; the *byte count* is model-independent, so its share of a small model's step is proportionally larger |
| `sampling` (device) | visible only when V/P is large — a 0.5B model with a 128k vocab gives 2·V/P ≈ 5% of step bytes, versus 0.0032% for Llama-3-8B |

**Two published production failure modes in this regime**, both with the GPU visibly idle:
a structured-output reasoning-end predicate that re-detokenised the whole sequence every step for
every running request decayed throughput **260 → 170 tok/s at concurrency 50**, with the GPU at
~150 W of a 300 W limit and EngineCore pinned below one CPU core, ~4.9 ms per decode step at 16k
reasoning length ([[flops/detokenization-stop-string-scan]]); and a batch-invariance/determinism
workload where vLLM default 26 s vs unoptimised deterministic 55 s (2.11x) and an improved-kernel
variant 42 s (1.62x) on 1000 sequences of 90–110 tokens on one GPU
([[benchmarks/vllm-batch-invariant-determinism-throughput-cost-1-6x]]) — per-step CPU cost is
exactly what batch-invariant kernels add back.

**Cures, in order of leverage for this regime:** CUDA Graphs / kernel fusion (per-kernel); a control
plane that is not Python (per-step, currently an open PoC rather than a shipped fix); bounded-tail
and O(delta) host predicates; device sampling; and only then a faster GPU. Note the last one is
listed last on purpose.

---

## 7. Where the taxonomy does not fit, and what that costs

Three of the classes in this document do not have a home in the flop `class` enum, and a reader
querying by class will get misleading answers. Recorded here and in each record's `notes`:

1. **Detokenisation / stop-string detection** (`[[flops/detokenization-stop-string-scan]]`) has no
   class. It sits *after* the token id exists, on the host, with a host round trip between it and
   sampling — neither of which the `sampling` label conveys. **Requested enum addition: class
   `detokenization`.** Filed under `sampling` as the nearest existing value.
2. **Host-side per-step work** — `host-side-sampling-loop`, `logits-processors-cpu-pipeline`, and the
   per-step scheduler half of `custom-kernel-launch-overhead` — is bound by `launch_overhead`, which
   is the least-bad existing value but describes *kernel* launches. A `host_pipeline` value, or a
   provenance field distinguishing device passes from host passes, would make the family queryable.
3. **Logit processors** are folded into `sampling`, which conflates "post-LM-head arithmetic on the
   logit tensor" with "per-request state marshalled through Python every step". The two have
   opposite scaling behaviour (device: batch-invariant, negligible; host: multiplies with batch and
   output length) and the published regression lives entirely in the second.

## 8. What this document cannot tell you

- **No measured step-time attribution exists.** `docs/02-flop-map.md` §8 item 3 and item 6 both say
  so. Everything in §1's tables is derived from record arithmetic; it is a *model* of the step, not
  a decomposition of a measured one. The single most-requested missing measurement is the one that
  would let you rank these classes by observed share rather than by derived bytes.
- **No measured TTFT for long-context RAG** (see §5). 526 TFLOP and 4.00 GiB are analytic.
- **No per-token microsecond measurement of detokenisation, stop-string scanning, or CPU sampling**
  in this repo. The 0.5–1.5 µs/token figure is for *full-prompt DecodeStream priming*, a different
  operation from incremental detokenisation; the 4.9 ms/step figure is one predicate in one
  deployment. Where a per-token cost is needed, say **no record**.
- **The three published host-overhead failure modes are single-engine reports** (vLLM and
  llama.cpp), with fixes verified as "CPU utilisation returned to normal" rather than as published
  tok/s deltas. The mechanisms are the durable claim; the magnitudes are indicative.
- **The 32-slot GPU-utilisation figures are quoted for the default path only.** With device sampling
  enabled, `cudaGraphLaunch` calls rose 144 → 502 in the same window, so the node-level trace does
  not cover both configurations equally; throughput, TPOT and memcpy volume are direct measurements
  in both.