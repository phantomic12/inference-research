# 10 — Testing methodology: differential oracles that actually catch silent corruption

This document exists to close one specific gap, stated in
[05-known-traps.md](05-known-traps.md) §"What the records cannot answer about traps", item 1:

> No systematic fuzzing or differential-test record. Every trap here came from a user
> report. There is no record of an automated cross-backend differential harness, so the
> absence of a record for (engine, hardware, format) means **untested, not safe**.

The repo now has ~103 gotchas. Almost all of them are *silent* corruption: no crash, no
exception, no error in the log, plausible-looking output. A test suite that only checks
"did it return 200" and "is the text fluent" passes on every single one of them. What
follows is the set of oracles that would have caught them, each tied to the specific
existing record that demonstrates the need for it.

Read the oracles in the order below. They are ordered by **signal-to-cost**: the cheapest
oracle that catches a whole class of bugs comes first, and the most expensive (and most
conclusive) comes last.

---

## The core problem, stated precisely

Two facts govern everything in this document.

**1. The reference is a model, not an assertion.** There is no closed form for "the
correct logits" of a 70B transformer. So the question is never "is this output right?"
but "does this output *change* when it should not?" Every useful oracle here is a
**differential**: two runs that a correct implementation must make agree, and that a
broken one may not.

**2. Bit-exactness and numerical accuracy are different tests, and a build can fail
either one independently.** vLLM issue
[[sources/nrg2-gh-vllm-55131-tf32-breaks-invariance|55131]] reports that `tl.dot` uses TF32 by
default for fp32 inputs on Ampere/Hopper, which is wrong for precision-sensitive modules
(the DeepSeek Sparse Attention Indexer, the MoE router). A TF32 matmul can be perfectly
batch-invariant and still be numerically wrong. Conversely, a loose tolerance like
`rtol=1e-2` will happily pass a build whose output text visibly diverges. **Run both a
bit-exactness check and a tolerance-based accuracy check; never substitute one for the
other.**

---

## Oracle 1 — Batch invariance (the strongest single check)

**The check.** Take one fixed prompt. Run it (a) alone at batch size 1, then (b) inside
batches of 8, 64, 256, at several different *positions* within the batch. Assert the
**logits are bit-identical** every time.

**Why it is so strong.** It requires no reference model and no golden output. It is a
*self-consistency* property that any correct implementation must satisfy by construction:
the arithmetic for a given token does not depend on how many unrelated tokens share its
batch. A single flipped ULP anywhere in prefill or decode breaks it. This is why it is
the primary oracle: it converts "is the model correct?" — which is unanswerable — into
"is the engine self-consistent?", which is checkable on every request in a smoke test.

**The evidence it is needed.** The same prompt at temperature 0, sampled 1000 times on
Qwen3-235B-A22B, produced **80 unique completions**
([[benchmarks/qwen3-235b-a22b-greedy-decode-80-unique-completions-of-1000]]). All 1000 agreed
for the first 102 tokens and first diverged at token 103. With batch-invariant kernels all
1000 became identical.

**The mechanism, so you know what you are testing.** It is *not* floating-point
non-associativity in general — the same matmul run 1000× on identical data is bitwise
identical. It is **batch-dependent reduction strategy**. Thinking Machines identify
exactly which strategies break invariance: split-K matmul, stream-K (which additionally
breaks batch-*position* invariance by splitting K differently per output tile),
split-KV/FlashDecoding with a per-call split count, FlashInfer's "balanced scheduling"
which picks the largest split size that saturates the cores, and small-batch split
reductions in RMSNorm. A kernel library that picks any of those *per call, based on
load*, is non-invariant by construction.

**Which existing gotchas this would have caught — and note that most of them are *not*
batch-size bugs at all:**

- [[gotchas/pd-cache-hit-silently-differs-from-cold-run]] — "the cold path and the hit path run
  different kernel tilings, so the KV for the same tokens is not bit-identical depending
  on how the block was computed." Batch-invariance testing at several sizes exercises the
  tiling switch directly.
- [[gotchas/pd-packed-kv-layout-change-silently-breaks-external-caches]] — a layout change makes
  restores wrong; a cold-vs-warm comparison catches it, and invariance over batch sizes
  catches the tiling half.
- [[gotchas/eagle-prefix-cache-last-block-drop]] — 6–10 points of hit-rate loss, correctness
  preserved but cache accounting wrong. This is a *hit-rate* regression, so the oracle is
  Oracle 4 (cold-vs-warm) plus a hit-rate assertion, not invariance.
- [[gotchas/rocm-flash-attention-kv-type-graph-split-explosion]], [[gotchas/vulkan-rebar-off-collapses-amd-decode]],
  [[gotchas/rdna-triton-paged-attn-decode-cliff]] — all change the reduction/tiling strategy
  under different shapes.

**How to run it upstream.** vLLM ships this as a **beta** feature with a per-op test
suite at `tests/v1/determinism/` — 10 files covering batch invariance for VLM, CUTLASS
matmul, NVFP4, RMSNorm, XPU, plus an online end-to-end test
([[sources/nrg2-vllm-docs-batch-invariance]]). Run the per-op suite, not a single end-to-end
check.

**Two limits you must plan around.**

1. **Support is bounded to NVIDIA cc8.0+ and Intel XPU with the Triton backend.** There is
   no ROCm/HIP, Metal, TPU or CPU claim
   ([[benchmarks/vllm-batch-invariance-hardware-support-scope-cc80-and-xpu-triton]]). For the
   AMD, Apple, Gaudi, TPU and Ascend records in this repo there is *no flag to test
   against* — which is exactly why Oracles 3, 4 and 5 below exist.
2. **The flag is not a guarantee.**
   [[gotchas/nrg2-batch-invariance-is-not-universal-across-kernel-paths]] — with
   `VLLM_BATCH_INVARIANT=1` set, the guarantee silently does not hold for fused-MoE
   experts (issue #57016) or for AWQ on sm8x in default mode (#59086), and TF32 breaks the
   matmul invariant (#55131). No warning in any case. Assert invariance **on the exact
   (architecture × quantization × kernel-family) config you ship**; a pass on one config
   is no evidence about another.

**Cost.** Determinism is not free: 26 s → 55 s (2.11×) → 42 s (1.62× with an improved
attention kernel) for 1000 sequences of 90–110 tokens on one GPU with Qwen-3-8B
([[benchmarks/vllm-batch-invariant-determinism-throughput-cost-1-6x]]). **So: run this as a
test oracle on a small config, not as a production mode.**

---

## Oracle 2 — Cross-backend agreement (ROCm/HIP vs Vulkan, and friends)

**The check.** Same model file, same quantization, **byte-identical flags** (verify with
`diff`), on the same machine — run it on two backends and compare.

**Why it works here specifically.** On a single backend you have no ground truth. On two
backends written by different code paths, a *shared* bug is unlikely while an
*independent* one shows up immediately. This is the repo's most-productive oracle by
far: both of the gotchas below were found this way, by users, not by CI.

**Which existing gotchas demand this oracle — this is the pair the task refers to:**

- [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — on gfx1151 (Strix Halo, ROCm 7.2.4),
  same machine, same build, byte-identical flags: **HIP is wrong, Vulkan is right.**
  Llama-3.1-8B Q4_K_M is broken outright at every depth (0/16 at 30,516 prompt tokens);
  Qwen3.8-27B UD-Q8_K_XL is correct shallow and fails past ~29k tokens with a large tool
  array, where it confidently mis-states its own tool list. This record is the direct
  justification for treating **Vulkan as the reference backend on gfx1151**.
- [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] — perplexity as the smoke test. The
  numbers in the record above are the evidence: WikiText-2 PPL HIP vs Vulkan, same 2-item
  set — BF16 84.7476 vs 13.8167; Q8_0 110.2795 vs 14.4350; Q4_K_M 92.7499 vs 12.9609;
  Q4_0 112.7851 vs 12.1284. A 6–9× PPL gap between backends on the *same file* is
  unambiguous.

**Two disciplines these records teach, both non-obvious:**

1. **Read the backend from the running process, not from the launcher.** The reporter of
   [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] found an earlier "Vulkan is correct"
   result was actually the HIP server still answering after a boot script silently
   failed. Check `/proc/<pid>/exe` on every run. A differential harness that records which
   backend it *believes* ran is worse than useless — it will confidently mis-attribute.
2. **Two SHAs agreeing is not corroboration.** In that same record, reproduction on
   `b21e4de74567f` and `3a653fea932e` proves nothing, because no commits between them
   touch the relevant source. Pin and diff the actual backend directories.

**Cost.** One command. `llama-bench` on the same GGUF files across two backends. This is
the best value in the document.

**Bonus use — performance, not just correctness.**
[[gotchas/vulkan-decode-cliff-hidden-size-4096]] was diagnosed by converting tok/s into *effective
bandwidth*: Qwen3-8B Q4_K_M on Vulkan gives 21.0 t/s ≈ 98 GB/s on a 640 GB/s card, while
HIP gives 98.8 t/s ≈ 464 GB/s. That conversion turns "Vulkan is slow" into a diagnosable
shape problem (hidden 2560 fast; 4096 and 5120 slow) and is worth doing on every
cross-backend measurement.

---

## Oracle 3 — Perplexity / likelihood on a fixed corpus

**The check.** Score a fixed text corpus (e.g. WikiText-2, a fixed 2-item set) and compare
PPL across configurations, backends and builds.

**Why.** Generation hides corruption behind fluency; PPL does not. It is scalar,
deterministic, and needs no reference model — only itself, compared across time.

**Which existing gotchas demand it:** [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] (the
canonical case), [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] (ppl 93.86 on HIP vs 7.40
on Vulkan for 31B — "which reads as a weak model rather than a broken backend"), and
[[gotchas/mtp-plus-prefix-caching-accuracy-loss]].

**Caveat that makes PPL a necessary but insufficient oracle:** PPL is an *aggregate*. It
can be unchanged while specific long-context or long-prompt requests are badly broken —
which is exactly the split in [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]], where
BF16 is affected too (ruling out the quantized-matmul kernel and making it a
scheduling/batching bug) and the default `n_ubatch` of 512 means "almost every real
prompt" is affected. PPL over short contexts would miss it. **Pair PPL with Oracle 4.**

---

## Oracle 4 — KV-cache hit path vs cold path equivalence

**The check.** Run a set of requests twice against the *same warm server*:

- **run 1**: fresh server, no prefix cache populated → output A
- **runs 2..N**: same requests, cache now warm → output B

Assert A == B bitwise, and assert the cache hit-rate is non-zero in run 2 (otherwise the
test is vacuous — you proved nothing, because nothing was cached).

**Why this is a distinct oracle.** It is the only check that covers the *stateful* half of
the system: cache insertion, block layout, eviction, connector serialization, and
cross-instance transfer. It is the highest-yield oracle in this repo by bug count.

**Which existing gotchas demand it — the largest cluster in the repo:**

- [[gotchas/pd-cache-hit-silently-differs-from-cold-run]] — the archetype. "with prefix caching
  on, the same request at temperature=0 returns output A on a freshly started server and a
  different, stable output B on runs 2..N, and restarting returns to A."
- [[gotchas/pd-packed-kv-layout-change-silently-breaks-external-caches]] — PR #44455 changed the
  KV block layout from `[num_blocks, 2, block_size, num_heads, head_size]` to a packed
  `[num_blocks, num_heads, block_size, 2*head_size]`; LMCache's connectors did not track
  it, so restores were silently wrong. A pure engine-side cold-vs-warm test would *miss*
  this — the engine's own cache is fine; the external tier is stale.
- [[gotchas/pd-lora-adapter-swap-reuses-stale-prefix-blocks]] — cache blocks keyed without the
  adapter identity.
- [[gotchas/pd-batch-invariant-mode-absent-from-kv-namespace]] — the sharpest one, and it links
  Oracle 1 to Oracle 4: the offload namespace carries model name, dtype, parallel sizes
  and `inference_engine`, but **no batch-invariance flag and no attention backend**, so an
  instance can load blocks produced under different numerics with nothing signalling it.
  The reporter is explicit that they have not reproduced a divergence. Two oracles that
  are individually sufficient are jointly insufficient here — you need cold-vs-warm *and*
  invariance *and* a namespace that binds the numerical config.
- [[gotchas/eagle-prefix-cache-last-block-drop]] — assert hit rate, not just correctness.
- [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]] — a *deferred*-validation
  failure: startup passes and `/health` returns 200, then the first request raises.
  Therefore: **a cold-vs-warm test must actually issue requests.** A health check is not a
  test.

**What "identical" must mean here.** Bitwise, at temperature 0, on the *same* request.
Note from Oracle 2's evidence that temperature 0 alone is not sufficient — see Oracle 6.

---

## Oracle 5 — One-parameter differential diagnosis

**The check.** Change **exactly one** parameter, observe one output.

**Why.** When several things are suspect, bisecting one at a time isolates the layer
without a debugger. It is the cheapest diagnostic in the repo and it is already the
documented instinct in three records.

**Which existing gotchas demonstrate it — all three, verbatim from [05-known-traps.md](05-known-traps.md)
Pattern 4:**

- [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] — "change exactly one batching parameter
  and diff the logits". Its attached minimal reproducer decodes one prompt twice using two
  contexts differing in **exactly one** parameter (`n_ubatch`), with identical token ids,
  and reports argmax disagreement rate, mean and worst KL divergence per position, and
  largest absolute logit difference. This is the template: a correct backend *must* return
  the same distribution both times.
- [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — `--enforce-eager` good + CUDA
  graphs garbage. One flag, and the fault moves.
- [[gotchas/rdna-triton-paged-attn-decode-cliff]] — prompt processing flat + token generation
  collapsed. One stage of the pipeline, isolated by metric.

**Note the direction of the last one:** sometimes the differential is between *metrics*,
not configurations. Keep prefill and decode numbers separate; a single aggregate tok/s
hides a 5× cliff.

---

## Oracle 6 — What NOT to trust (the false-pass list)

Explicit, because these are the checks people reach for first:

| Tempting check | Why it fails |
|---|---|
| "Temperature 0 makes it deterministic" | **80 unique completions from 1000 runs** at temperature 0, first diverging at token 103, all fluent. See [[gotchas/nrg2-temperature-zero-greedy-decode-is-not-a-determinism-oracle]]. |
| "Compare against a stored golden string" | Inherits the 7.9% minority-branch rate above; the divergent branch reads as correct to a human. Compare across *batch sizes and positions*, not to one golden. |
| "Is the output fluent English?" | Every one of the 80 divergent completions is coherent. This is why all the corrupt-output records pass a fluency check. |
| "It returns HTTP 200 and /health is green" | [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]] passes startup and health, then raises on first request. |
| "It works under transformers" | [[gotchas/flashinfer-rejects-large-head-dim]] — same model, fine under a different attention backend. Engine-level clearance requires engine-level tests. |
| "Set the determinism flag and ship" | [[gotchas/nrg2-batch-invariance-is-not-universal-across-kernel-paths]] — silently non-invariant for fused-MoE and AWQ on sm8x. |
| "A passing end-to-end test on one model clears the engine" | The upstream determinism suite is *per-op* (RMSNorm, CUTLASS matmul, NVFP4, attention) precisely because invariance does not transfer across kernel families. |

---

## The minimum viable suite

For a given (engine, hardware, quantization) tuple, in this order. Anything less leaves a
documented class of bug uncaught.

1. **Oracle 1** — batch invariance at bs ∈ {1, 8, 64, 256} × several in-batch positions,
   bitwise on logits. *(NVIDIA cc8.0+ / XPU-Triton only; elsewhere mark as UNAVAILABLE
   rather than passed.)*
2. **Oracle 4** — cold vs warm, ≥2 runs, bitwise on output, **plus** a non-zero hit-rate
   assertion. Issue real requests.
3. **Oracle 2** — cross-backend, byte-identical flags, verified process identity.
4. **Oracle 3** — PPL on a fixed corpus, compared against a stored baseline.
5. **Oracle 5** — one-parameter bisection on any failure above.

Then record the tuple as **tested** or **untested**. The point of the exercise is to make
"untested, not safe" ([05-known-traps.md](05-known-traps.md) gap 1) a *distinction the repo can actually
represent** — because an untested tuple and a clean one currently look identical.

## Known coverage holes in this methodology

- **No accelerator-specific idle ratio** exists for MI300X, L40S, A100 or TPU; only the
  general volume-server dynamic range
  ([[benchmarks/us-volume-server-idle-dynamic-range-model-0-44-floor]]). Power-side testing has the
  same untested-tuple problem as numerics.
- **Gaudi, TPU, Ascend, Inferentia, Trainium: zero gotcha records** ([05-known-traps.md](05-known-traps.md)
  gap 4) and no batch-invariance support. Every oracle above is untested there.
- **No fuzzing harness.** This document is a set of oracles, not a fuzzer. Corpus-driven
  fuzzing over prompt lengths, batch shapes and cache states is the obvious next step and
  would cover the shape-dependent failures ([[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]],
  [[gotchas/vulkan-decode-cliff-hidden-size-4096]]) that fixed-shape tests miss.
- **Bit-exactness across *versions* is not tested.** Oracle 1 catches a regression within
  one build; it does not tell you the new build is as correct as the old one. The
  cross-version case needs Oracle 2 or 3 as the bridge.
