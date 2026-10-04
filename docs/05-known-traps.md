# 05 — Known traps, grouped by severity

<!-- generated against a snapshot of 1,087 records (34 gotchas: 9 blocker, 20 major, 2 minor). data/ grew to 47 gotchas / 1,281 records while this was being written; the 13 later gotchas were folded in, see §Post-snapshot additions. Counts were a snapshot; data/ is written concurrently. -->

This is the document to read **before** deploying, not after. Every entry is a record in
`data/gotchas/`, and the ordering below is deliberate: **the blocker-severity
silent-corruption cases come first**, because they are the only category that produces
confident, plausible, wrong answers rather than a crash.

A crash is a nuisance. A model that serves clean-looking tokens at temperature 0 while
running the wrong kernel is a liability, and several of the records below were found only
because someone ran a cross-check that a deployment would not run.

---

## Tier 0 — Silent wrong output at `blocker` severity

*(Numbering below — 0.1, 1.4, 2.13 — is this document's own, not a field in the records.
The records carry `severity`, `class` and `confidence`; there is no index field.)*

**These nine records share one property: the server starts, serves every request, emits no
error, no warning and no NaN, and returns confident wrong text or wrong logits.** Several
are *faster* when broken, so throughput-based A/B testing selects for the bug.

### 0.1 [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]] — the wrong path is faster

`severity: blocker`, `class: kernel`, confidence 0.85.

On NVIDIA GB10 (DGX Spark, **sm_121a**) serving a WNA16 INT4 MoE
(Qwen3.5-122B-A10B, AutoRound/GPTQ-style, group_size 128) with
`VLLM_MARLIN_INPUT_DTYPE=fp8`: the server starts normally, "no crash and no NaN assert, and
at temperature=0 every reasoning prompt degenerates into a repeated `</think>` loop until
max_tokens." The same binary with the env var unset produces coherent chain-of-thought.

**The line that should stop any benchmark-driven workflow:** "The corrupted path is also
**~2.5% FASTER** end-to-end, so throughput benchmarking selects for the bug."

The record's own confidence statement is precise about what is and is not established:
"Not yet root-caused by a maintainer — confidence is in the reproduction (single env var
flips it on the same binary...), not in the mechanism." Root cause is a code reading: the
capability gate `get_marlin_input_dtype` tests `is_device_capability(89)` or
`is_device_capability_family(120)`, and since the family test is `(cap.to_int() // 10) == 12`,
`sm_121 -> 121//10 == 12` **passes and sm_121a is opted in with no sm_120-vs-sm_121
distinction**.

**Workaround:** do not set the var on sm_121a; leave it unset for the verified-coherent
W4A16 path. **And note the record's own warning:** "The reporter explicitly did **NOT** test
`VLLM_MARLIN_INPUT_DTYPE=int8` (W4A8-INT8) on this hardware, so **do not assume int8 is
safe either**."

### 0.2 [[gotchas/cc-mode-uva-view-garbage-output]] — wrong under confidential computing only

`blocker`, `class: framework`, confidence 0.9.

On H100 NVL in CC mode inside a TDX guest, vLLM 0.29.0 with the default V2 GPU model runner
"starts and serves, but **every response is garbage** — `!!!!!!!!` (token id 0 repeated)".
Workaround: `VLLM_USE_V2_MODEL_RUNNER=0` falls back to the V1 runner, which produces correct
output on the same setup.

**Diagnostic ordering matters here** and the record says so: "the same guest runs
transformers, CuPy and plain torch correctly, so **'works under transformers' does NOT rule
out the GPU or CC configuration**. Test vLLM's runner version before suspecting hardware."

### 0.3 [[gotchas/nvfp4-marlin-bf16-garbled-output]] — a dtype flag that silently destroys the output

`blocker`, `class: format`, confidence 0.95 — the highest-confidence gotcha in the repo.

`vllm serve` of any NVFP4 model with `--dtype bfloat16` on SM < 100 (RTX 4090 SM89, V100
SM70) "loads, starts and serves, but **every response is garbled**."

**Workaround:** serve with `--dtype float16` on pre-Blackwell-datacenter parts — "the FP16
path maps the 5-bit exponent directly onto FP16's 5-bit exponent and is unaffected."
Otherwise use a GPU with native FP4 (SM100+). This is directly relevant to consumer hardware:
[[accelerators/nvidia-rtx-4090]] is in scope, [[accelerators/nvidia-rtx-pro-6000-blackwell]]
(SM120) is not.

### 0.4 [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] — two correct features that are not independently safe

`blocker`, `class: kernel`, confidence 0.9.

1x B200 (SM100), vLLM 0.23.1rc1.dev843, flashinfer-python 0.6.14, serving
RedHatAI/gemma-4-31B-it-NVFP4 (50 of 60 layers SWA, window 1024) with
`--attention-backend FLASHINFER --kv-cache-dtype fp8_e4m3 --speculative-config ...`: output
corrupted.

**The record contains the cleanest attribution matrix in the repo**, all temp-0 on the same
NVFP4 checkpoint: fp8-Q + spec → **clean**; bf16-Q + no spec → **clean**; bf16-Q + spec →
**corrupted**, and corrupted under both `-O3` cudagraphs and `--enforce-eager`. So each
feature is individually safe and the combination is not.

**Workaround:** drop `--attention-config disable_flashinfer_q_quantization`, or drop
`--speculative-config`, or drop quantized KV — any one of the three.

Also a good example of "a bug that hides behind the common `_q_scale=_k_scale=_v_scale=1.0`
assumption."

### 0.5 [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — the decisive experiment to copy

`blocker`, `class: framework`, confidence 0.9.

8x A800-80GB (SM80, **no native FP8 tensor cores**), TP=8, MiniMax-M3-AWQ-INT4 (weight-only
int4), launched with `--kv-cache-dtype fp8_e5m2 --max-model-len 1048576 --attention-backend T...`:
corrupted output.

**The diagnostic is the reusable part:** "`--enforce-eager` good + CUDA graphs garbage
isolates the cause to a **graph-unsafe write**, not to precision."

**Workaround (in the record):** decorate `_insert_kv` with `@eager_break_during_capture` (the
codebase's own idiom), route dense-layer KV writes through the CUDA `reshape_and_cache_flash`
with a software fp8 convert, and relax the SM89 guard.

Two things worth noting: fixing this "also unblocks the whole 1M-context class on SM80, which
needs fp8 KV for capacity" — and the side note "issue #39137's over-broad e5m2 gate was
already fixed on main by #45040."

### 0.6 [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] — silent accuracy loss, not corruption

`blocker`, `class: config`, confidence 0.85. Same severity tier but a different shape: no
garbled text, just **wrong answers**.

On a finetuned Qwen3.6 35B-A3B (MoE) served with MTP (`num_speculative_tokens=2`) plus
`--enable-prefix-caching`, "internal classification accuracy drops **~20%** versus the
identical setup."

**Workaround:** "Do not combine MTP/EAGLE speculative decoding with prefix caching on models
where output quality matters until this is fixed. Serve with one or the other, or fall back
to plain decoding, and validate..." This is the only blocker-severity record whose failure
mode is entirely invisible in the output.

### 0.7 [[gotchas/sm120-sparse-mla-block-size]] — no workaround at all

`blocker`, `class: kernel`, confidence 0.9.

8x RTX PRO 6000 Blackwell (SM120) with TP8: `deepseek-ai/DeepSeek-V4.1-Flash` fails to start
with `ValueError: SM120 sparse-MLA has no decode kernel for this shape: num_tokens=8, ...`

**"None available on SM120 with this checkpoint and FlashInfer 0.7.0 — the mixed-ratio
architecture is incompatible with the current kernel set."** Run the model on SM100 datacenter
Blackwell.

The record also establishes that this is **one of a cluster, not one bug**: "the reporter
walked the other SM120 gates first: `indexer_kv_dtype='mxfp4'` is rejected on SM120
('requires Blackwell datacenter GPUs sm_10x') and had to be switched to 'fp8'. So SM120
consumer/server parts hit a cluster of **independent kernel-set gaps** for this model family."
Compare [[flops/sm120-vs-sm100a-kernel-binary-incompatibility]] and
[[engines/vllm-cpp]]'s candid "several architectures are token-gated but not speed-gated".

### 0.8 [[gotchas/torch-source-build-abi-mismatch]] — a build failure that presents as a runtime one

`blocker`, `class: toolchain`, confidence 0.9. Affects [[engines/vllm]].

`import vllm` raises `ImportError: .../vllm/_C.abi3.so: undefined symbol:
_ZN5torch3jit17parseSchemaOrNameERKSsb`. "The `.so` file exists and vllm is installed;
nothing about the install looks wrong."

**Workaround:** "Install the official PyTorch wheels rather than source builds, or rebuild
vLLM against exactly the torch you have and check `torch._C._GLIBCXX_USE_CXX11_ABI` before
building."

### 0.9 [[gotchas/sycl-build-unrecognized-fsycl]] — Intel SYCL build abort

`blocker`, `class: build`, confidence 0.85. Affects [[engines/llama-cpp]].

After installing `intel-oneapi-base-toolkit` from Intel's apt repository (2025.2/2025.3
packages, including `intel-oneapi-compiler-dpcpp-cpp`), a `GGML_SYCL=ON` configure/build
aborts on an unrecognised flag.

**Workaround:** "Configure the build inside the oneAPI environment so CMake picks up `icpx`,
and verify which compiler CMake resolves before building. If the flag is still rejected,
the base toolkit alone is not sufficient."

The record is careful about its own confidence: "Blocked on issue, not confirmed resolved;
the specific missing-component diagnosis is the reporter's environment rather than a
maintainer-verified root cause."


### 0.10 [[gotchas/windows-hip-multi-gpu-silent-output-corruption]] — healthy-looking throughput, garbage text

`blocker`, `class: driver`, confidence 0.9.

On **Windows 11** with a gfx1100 + gfx1201 pair, the HIP backend splits a model across both
cards and "returns fluent-but-nonsensical word salad instead of an answer, with no error, no
warning and **healthy-looking throughput** (llama-bench: pp ~197, tg ~26). Corrupted samples
begin plausibly and then degrade into nonsense." All controls are clean.

**Workaround:** build llama.cpp with `-DGGML_CUDA_NO_PEER_COPY=ON` for Windows multi-GPU HIP
— "a source build, so the official Windows ROCm release zip does not help" — or use Windows
Vulkan, which the reporter runs in production at about 37 t/s. Note this is the *same shape*
as [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] and
[[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]]: on this stack, Vulkan is the
reference and HIP is not.

### 0.11 [[gotchas/vulkan-rebar-on-causes-amd-device-lost]] — the crash twin of a slowdown

`blocker`, `class: driver`, confidence 0.8.

With SAM/ReBAR **enabled**, every decode on an AMD Radeon AI PRO R9700 fails with
`vk::Queue::submit: ErrorDeviceLost`; llama-server returns HTTP 500 and exits. Disabling
ReBAR in BIOS makes the identical driver, binary and command line succeed.

**Workaround:** "Disable SAM/ReBAR in BIOS on this platform — that is the only confirmed fix
in the thread, and it **re-introduces the slowdown recorded in
[[gotchas/vulkan-rebar-off-collapses-amd-decode]]**, so on this class of board you are
choosing between two bad options and should test both."

Read together, the two ReBAR records are a trap in both directions: the fix for one
reproduces the other.

### 0.12 [[gotchas/sycl-device-memory-query-abort-blocks-model-load]] — SYCL cannot even load a model

`blocker`, `class: toolchain`, confidence 0.9.

`llama-server` and `llama-cli` **abort at model load** with `GGML_ABORT` in
`ggml_backend_sycl_device_get_memory`: "failed to get device memory size". Confirmed on two
very different parts, including a Sparkle Arc Pro B60 24GB on Ubuntu 26.04 with the xe
driver.

**Workaround:** patch `ggml_backend_sycl_device_get_memory` to fall back to system RAM instead
of aborting — verified to restore normal performance (Qwen3.5-9B Q4_K_M: pp512 502 t/s, tg64
18.2 t/s). Failing that, use the Vulkan backend or a build before the regression.

This is the third distinct **blocker** in the Intel SYCL path alongside
[[gotchas/sycl-build-unrecognized-fsycl]] (build fails) and
[[gotchas/sycl-multi-gpu-arc-loses-p2p-and-crashes]] / [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]]
(runtime stalls or crashes). **Treat the SYCL path on Intel Arc as not production-ready.**

---

## Tier 1 — `major`: silent corruption

These do not stop the server. They produce wrong output, wrong logits, or a wrong number,
and in most cases no diagnostic is emitted at all.

### 1.1 [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] — the canonical eval-harness case

`major`, `class: measurement`, confidence 0.9.

On Ryzen AI Max+ 395 (iGPU Radeon 8060S, gfx1151), ROCm 7.2.4, Llama-3.2-3B-Instruct Q4_0,
wikitext-2-raw, `-c 32768 -ngl 999 --chunks 1`: `llama-perplexity` returns a **PPL two
orders of magnitude** above the same model on another backend.

**The record states the methodological lesson exactly:** "an output-quality regression of
this magnitude produces no error, no warning and no NaN — the server answers every request —
and the regression is **390x in perplexity**. It also shows the right triage order: a PPL this
bad is a **backend bug**, so check another backend on the same file before touching the
quantization settings."

**Workaround:** "Run llama-perplexity as a smoke test on the same model, context and file
across the backends you intend to ship, and treat a PPL jump of two orders of magnitude as a
hard stop."

### 1.2 [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — same machine, both backends, one is wrong

`major`, `class: kernel`, confidence 0.8.

On a gfx1151 Strix Halo APU (ROCm 7.2.4), "same machine, same build, byte-identical server
flags (verified with diff), **the HIP backend is wrong and Vulkan is right**."

**Workaround:** "On gfx1151, treat Vulkan as the reference backend and cross-check before
trusting HIP output, especially for dense architectures and for long-context tool-array
requests. Read the backend from `/proc/...`" — the record's own methodological point is "verify
which backend actually served the request **from the process**, not from the script you
launched."

Two severity levels are recorded deliberately: word salad, and "a much nastier shape than
word salad because **the output stays plausible**" — head-of-array tool-list loss at long
context. The record also warns against over-reading the diagnosis: "dense-vs-MoE is a live
axis here but the reporter's evidence is one observation and is labelled as such."

### 1.3 [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] — plausible numbers, no warning

`major`, `class: kernel`, confidence 0.85.

"Any prompt longer than `n_ubatch` produces badly wrong logits on HIP/ROCm on gfx1151 — **no
error, no warning, no crash, and the numbers look plausible.**" Default `n_ubatch` is 512.

**What makes this record worth separating from every quantized-kernel gotcha in the repo**,
in its own words: "The **BF16-also-affected** fact is what makes this worth separating... **do
not reach for a quantization explanation here.**" It reproduces on gemma-4-E2B-it in BF16,
Q8_0, Q4_K_M and Q4_0, plus Qwen3.5-0.8B Q4_K_M.

**Workaround:** "Check perplexity across the two backends on the same file before interpreting
any HIP/gfx1151 quality number." And a general technique the record names: "The n_ubatch-only
reproducer is a good general technique for this whole class — **change exactly one batching
parameter and diff the logits**."

### 1.4 [[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] — the most dangerous shape, with a measured cost

`major`, `class: format`, confidence 0.85.

A compressed-tensors MXFP4 checkpoint with `input_activations: null` (i.e. **MXFP4 weights,
group_size 32, W4A16**) served on B200 with `--dtype bfloat16` routes to a **W4A4** kernel.

**The three benchmark rows are the same model on the same harness:**

| Path | AIME/MATH/GSM8K avg | Cite |
|---|---|---|
| Unquantized BF16 reference | **82.36** | [[benchmarks/qwen3-8b-bf16-aime-math-gsm8k-avg-reference]] |
| W4A16 on MarlinMxfp4Kernel (correct) | **78.16** | [[benchmarks/mxfp4-w4a16-on-w4a16-marlin-aime-math-gsm8k-avg]] |
| W4A16 on the W4A4 kernel (bug) | **70.58** | [[benchmarks/mxfp4-w4a16-on-w4a4-kernel-aime-math-gsm8k-avg]] |

**A 7.58-point drop caused purely by kernel misrouting**, on identical weights.

The record's own framing is the sharpest in the repo: "The most dangerous shape in this repo:
the corrupted number (70.58% avg) **is still a plausible score**, and it sits between the
Marlin result and badly enough below BF16 that it reads as 'MXFP4 costs ~12 points'. W4A16
weight-only quantization is expected to be near-lossless, so an 11.8-point drop invites
**exactly the wrong conclusion.**"

**Workaround:** "Select MarlinMxfp4Kernel for MXFP4 weight-only checkpoints (vLLM PR #56803
fixes the dispatch to do this). **Check the startup log line naming the MXFP4 GEMM kernel
before trusting any MXFP4 quality number.**"

### 1.5 [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]] — the same bug class, reported twice

`major`, `class: config`, confidence 0.85.

Qwen3.6-35B-A3B-NVFP4 (`quant_algo W4A16_NVFP4` for `mlp.experts`) served with
`--moe-backend flashinfer_b12x` runs the **FP4xFP4** micro-kernel.

**Measured cost:** GSM8K **0.9037** (b12x) vs **0.9136** (Marlin reference) — 1192 vs 1205
correct of the same set, and latency 1080 s vs 833 s. Benchmark records:
[[benchmarks/nvfp4-w4a16-flashinfer-b12x-moe-gsm8k]],
[[benchmarks/nvfp4-w4a16-marlin-moe-gsm8k]]. Both `verified`.

**Workaround:** "Use the Marlin backend (the reference W4A16 path), or set
`FLASHINFER_B12X_FORCE_MOE_W4A16=1`... **Do not accept a b12x trace showing an FP4 kernel for
a W4A16 checkpoint.**"

The record explicitly notes this is "**Same family of bug as
[[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] — two independent reports of the same
class.**" That makes it a pattern, not an incident.

### 1.6 [[gotchas/nvfp4-fused-moe-w13-global-scale-mismatch]] — a zero-GPU check

`major`, `class: format`, confidence 0.9.

`ModelOptNvFp4FusedMoE.process_weights_after_loading` "fuses gate_proj and up_proj into w13
and keeps only the gate per-tensor global scale. When a checkpoint ships different scale_2
values..." the model serves with wrong quality.

**Workaround — and this is the pattern to steal:** "**Grep the startup log for the
`w1_weight_scale_2`/`w3_weight_scale_2` warning before trusting an NVFP4 MoE quality number.**
The check is cheap and **needs no GPU**: read the ...experts.N.gate_proj.weight_scale_2..."

### 1.7 [[gotchas/nvfp4-gguf-conversion-drops-block-scales]] — conversion, not serving

`major`, `class: toolchain`, confidence 0.85.

`RedHatAI/Muse-Glimmer-30B-NVFP4` converted with `convert_hf_to_gguf.py --outtype q8_0`
"produces a GGUF that does not load: `check_tensor_dims: tensor blk.0.attn_q_norm.weight not
found`."

**Workaround:** "Treat an NVFP4-to-GGUF conversion as **unverified until you check it**:
confirm every quantized projection actually receives its scale in the graph, and sanity-test
with a factual prompt whose answer is a known value."

### 1.8 [[gotchas/gated-deltanet-decode-collapse-long-context]] — fast prefill, collapsed decode

`major`, `class: kernel`, confidence 0.85.

With the model fully resident on GPU (zero "assigned to device CPU" lines in the log),
decode falls from **~33 t/s at 68K KV position to 1.4 t/s (714 ms/token) at 91K position**.

**Workaround:** "Cap `-c` below roughly 80K KV position for these models on this stack, or use
q8_0 KV with FA off at the smaller context. **Treat as an open upper bound on usable context
for hybrid GDN GGUF models.**"

The record's confidence statement matters: "the symptom is well reproduced with a clean
position sweep, but **the mechanism is not identified, so the workaround is a bound rather
than a fix**." And the diagnostic it teaches: "The pp/tg split is the key diagnostic — fast pp
with collapsed tg at long context **rules out memory-capacity and offload explanations**."

### 1.9 [[gotchas/rdna-triton-paged-attn-decode-cliff]] — misattribution risk named explicitly

`major`, `class: kernel`, confidence 0.9.

Qwen3.6-27B on gfx1100: 12.1 tok/s at 518 tokens of context, 4.2 tok/s at 32K. Linear-attention
layers keep constant cost; only the 16 full-attention layers degrade.

**Workaround:** "Check head_dim against 128 and the measured KV block size against 16 before
assuming an RDNA card is slow. The reporter verified the upstream split-KV decode kernel fixes
it on gfx1100, gfx1151 and gfx..."

And the anti-pattern the record warns against: "**Do not misattribute this to the
linear/SSM layers** — in the reported profile their decode kernels cost the same at 1K and 32K
(0.95x), which is what the architecture promises. Verify on the version you run before
concluding the fix landed."

### 1.10 [[gotchas/spec-decode-greedy-diverges-on-quantized-target]] — non-reproducible output

`major`, `class: kernel`, confidence 0.85. Affects [[engines/llama-cpp]].

Under greedy sampling (`temperature=0, top_k=1`), `llama-server` with `--spec-type
draft-dspark` or `draft-mtp` produces **different text** than the same server without `-md`,
when the target is quantized.

**Workaround:** "Do not assume spec decode is lossless on a quantized target. If greedy
reproducibility against a non-speculative run matters, either use a **bf16** target, or use
**n-gram speculation (verified lossless)**."

---

## Tier 2 — `major`: performance and availability

Not correctness — but each one will be measured as correctness-by-proxy when someone
reports "the new version is slower."

### 2.1 [[gotchas/fa4-num-splits-ignored-sm90]] — 48% decode regression, flat TTFT

`major`, `class: kernel`, confidence 0.95 — affects [[accelerators/nvidia-h100-sxm]] and
[[accelerators/nvidia-h200-sxm]]. Recorded in [[gotchas/fa4-num-splits-ignored-sm90]].

Decode on H100 (SM90) is **48% slower per token than on v0.26.0 at batch 1** and 36% slower at
batch 4 for `google/gemma-4-26B-A4B-it` at ~5.9k context, **while TTFT is flat at 61 ms** on
both.

**Workaround:** "Set `attention_backend=TRITON_ATTN` (recovers ~70% of the gap), or pin
`vllm_flash_attn` to 6d11b3c or later." This is the clearest case in the repo of why TTFT and
ITL must be measured separately — see [[flops/decode-attention]].

### 2.2 [[gotchas/spec-decode-silently-downgrades-cudagraphs]] — an optimization silently disabling another

`major`, `class: config`, confidence 0.85.

"A single `logger.warning` announces that `cudagraph_mode` was downgraded; decode throughput
is **47.5 tok/s on FLASHINFER vs 55.2 tok/s on FLASH_ATTN (+16%)**, changing only
`--attention-backend`."

**Workaround:** "Use FLASH_ATTN (FA2/FA3) instead of FLASHINFER with speculative decoding to
keep the full decode graph, or drop speculative decoding. **Check the resolved
`cudagraph_mode` in the startup log** rather than trusting the default."

The record generalises it: "the general lesson is the measurement trap, not this specific
backend: a performance optimization silently disabled by another optimization, announced only
by a warning that does not quantify the cost. **Same shape as
[[gotchas/mtp-plus-prefix-caching-accuracy-loss]] but on the speed axis instead of the
accuracy axis.**"

### 2.3 [[gotchas/vllm-rocm-fp8-cold-start-timeout]] — first start only

`major`, `class: config`, confidence 0.95.

vLLM on a gfx1100 Radeon with an FP8 checkpoint aborts during startup on the engine-ready
timeout after `VLLM_ENGINE_READY_TIMEOUT_S` (default 600 s).

**Workaround:** set it to 1800 for the first cold start. "Once `~/.triton/cache` is warm,
engine init drops to **~49 s** and later starts take ~2 min; the cost is **one-time per host
until that cache is invalidated**."

### 2.4 [[gotchas/aiter-gate-skips-rdna3-gfx1100]] — confirmed permanent, not a bug to wait on

`major`, `class: config`, confidence 0.95 — the highest-confidence gotcha in the repo.

On gfx1100 Radeon (Radeon PRO W7900 or 7900 XTX) vLLM logs "AITER is not found or not
supported on the current platform, QuarkOCP_MX will fall back to emulation" and selects the
slow path.

**The record's closing note is the actionable part:** "A vLLM maintainer (vllmellm) confirmed
**RDNA support is not on the vLLM roadmap** and redirected to ROCm/aiter#4604. So this gate
is unlikely to widen on its own; the issue remains open."

**Workaround:** "Run on the Triton/emulation path deliberately, or apply the four-part patch in
issue #51136."

> **Data integrity note:** this gotcha's `affects` list names `amd-radeon-pro-w7900` and
> `amd-radeon-rx-7900-xtx`, **neither of which has an accelerator record in this repo**. The
> same is true of two benchmark records ([[benchmarks/llamacpp-radeon-r9700-gfx1201-q1-0-decode-tok-s]]
> names `amd-radeon-ai-pro-r9700`;
> [[benchmarks/llamacpp-rx7900xtx-gfx1100-qwen3-27b-q4-batch1-decode-tok-s]] and
> [[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]] likewise). These are dangling
> cross-references, and **the Radeon consumer parts that matter most for consumer-Radeon
> advice are the least represented in `data/accelerators/`.** Flagged, not fixed — `data/` is
> not this document's to change.

### 2.5 [[gotchas/vulkan-flash-attn-slower]] — FlashAttention can be a 2.5x loss on Vulkan

`major`, `class: kernel`, confidence 0.7.

On Strix Halo (RADV, warp size 64) `llama-bench` shows FA on costing more than it saves:
`llama-2-7b.Q4_0` pp512 drops **1252.63 → 508.36 tok/s** and tg128 drops **49.91 → 25.80
tok/s**.

**Workaround:** "Benchmark with and without `-fa` on your own GPU and quantization before
enabling it; on this system leave FA off. **Do not assume FA is a free win on Vulkan the way
it is on CUDA.**"

Cross-record connection: [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]] reports the
same FA-on pp drop on a different vendor's stack, and its note observes that "suggests
**FA dispatch is the thing to A/B first on any non-CUDA backend**."

### 2.6 [[gotchas/vulkan-decode-cliff-hidden-size-4096]] — a measurement-discipline record

`major`, `class: kernel`, confidence 0.85.

`llama-bench` on an RX 9070 XT 16 GB (gfx1201, RDNA4, Windows 11, AMD proprietary driver
25.20.42.14) at `-ngl 99`: Qwen3-4B Q4_K_M (hidden 2560) gives tg 183.0 t/s (~424 GB/s
effective) while larger hidden sizes fall off a cliff.

**Workaround:** "On gfx1201 use the HIP build for dense models with hidden_size >= 4096, or at
minimum **compute effective decode bandwidth from file size before concluding a card or a
quant is slow**. Cross-backend llama-bench."

The record's note is the transferable part: "the useful move is converting tok/s into
**effective bandwidth**, which turns 'Vulkan is slow' into 'Vulkan achieves 98 GB/s on a 640
GB/s card, and the same file gets 464 GB/s on HIP'. That immediately **localizes it to the
backend and away from the quantization**. Do not generalize the hidden_size >= 4096 threshold
to other architectures or drivers — it is one card, one driver, Windows."

### 2.7 [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]] — no reliable workaround

`major`, `class: hardware`, confidence 0.75 (the lowest confidence among the `major`
records — flagged in its own note as "one user's hardware-specific report with no maintainer
diagnosis").

On a Ryzen 5600g with Intel Arc Pro B50, agentic coding with Devstral 2 small 24b or GPT-OSS
120b stalls in the first few agent steps; prompt processing drops to single digits.

**"No reliable workaround reported. Intel dGPU support in llama.cpp should be treated as
experimental for long-context or agentic workloads; validate on your exact SKU before
committing to it."**

### 2.8 [[gotchas/flashinfer-rejects-large-head-dim]] — engine init abort

`major`, `class: kernel`, confidence 0.9.

Engine init aborts with `ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not
valid for this configuration. Reason: ['head_size not supported']`.

**Workaround:** "Drop the `attention_config` override and let vLLM choose; the same model loads
and runs correctly under TRITON_ATTN."

### 2.9 [[gotchas/eagle-prefix-cache-last-block-drop]] — prefix cache hit rate

`major`, `class: config`, confidence 0.85.

Multi-round and repeated-prompt workloads lose **~6-10 points of prefix-cache hit rate** vs an
identical no-spec boot (**97.5% → 86.9-91.7%**). On one hybrid Qwen3.8 GDN layout each hybrid
block...

**Workaround:** "**Measure prefix-cache hit rate with and without `speculative_config` before
enabling spec decoding on prefix-heavy workloads.**"

### 2.10 [[gotchas/vllm-v1-memory-footprint-growth]] — engine generation moves the budget

`major`, `class: framework`, confidence 0.8.

Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 ran at **12K context** on 4x RTX 3070 (32 GB) under vLLM
0.6.x. The same model, same flags, under 0.7.0 + `VLLM_USE_V1=1`, "cannot exceed ~3K context."

**Workaround:** "Compare against `VLLM_USE_V1=0` explicitly when a context length regresses
across a version bump, and re-tune `--max-model-len` and `--max-num-batched-tokens` for the V1
engine."

**The record's own caution is important:** "Closed 2025-03-14 after maintainer discussion
**without a single canonical per-version memory overhead figure**, so treat the 12K-to-3K ratio
as one configuration on one machine, not a general ratio. What generalizes is the habit:
**engine-generation changes move the memory budget even when the model and flags are
identical.**"

### 2.11 [[gotchas/vllm-kv-cache-block-budget]] — the config error that is really a capacity error

`major`, `class: config`, confidence 0.8.

`ValueError: No available memory for the cache blocks.`

**Workaround:** "Raise `--gpu_memory_utilization` (the error names this), and/or lower
`--max-num-batched-tokens` and `--max-model-len` so each request's KV working set is smaller.
**Check the startup log's reported KV cache** ..." This is the practical face of
[[flops/kv-transfer]]'s claim that "KV footprint is a first-class constraint, not a secondary
one".

### 2.12 [[gotchas/triton-w4a16-gptq-qzeros-assert]] — group-size layout mismatch

`major`, `class: format`, confidence 0.85.

Serving GPTQ-Int4 Qwen3.6-27B fails at `triton_w4a16.py:201` with an assertion on
`qzeros.shape == (K // group_size, N // 8)`. Unquantized models run fine, and two independent
reporters reproduce it.

**Workaround:** "Select a different quantization kernel path on ROCm instead of `TRITON_ATTN`,
or use a GPTQ export whose qzeros layout matches the kernel's expectation. **Verify the
checkpoint's qzeros shape against the kernel's** ..."

### 2.13 [[gotchas/vulkan-q4km-speed-regression]] — a bisected regression, scope deliberately narrow

`minor`, `class: kernel`, confidence 0.8.

Bisected to first bad commit `adc5dd92e8aea98f5e7ac84f6e1bc15de35130b5` on Windows 11 with dual
Radeon PRO W7800, Vulkan SDK 1.3.283, Qwen2.5-14B-Instruct-Q4_K_M at `-ngl 99`. Decode
regressed.

**Workaround:** "Stay on a known-good commit or bisect around `adc5dd92` for your own build;
verify with llama-bench comparing identical seeds and flags across commits rather than
comparing to numbers from another machine."

The record's methodology note: "pp flat + tg regressed localizes the change to the
**decode-specific kernel**, not the general backend." And its scope limit: "Reported under
label bug-unconfirmed / stale upstream, so treat the exact scope as **narrow evidence rather
than a general Vulkan-k-quant verdict**."


### 2.14 [[gotchas/rocm-backend-silently-serves-weights-from-host-ram-on-gfx1201]] — reports VRAM, allocates none

`major`, `class: driver`, confidence 0.8.

On an 8x RADEON AI Pro 9700 XT (gfx1201) host, llama.cpp's ROCm backend "reports the full
VRAM size in `--list-devices`, shows both GPUs at normal compute activity, and then
allocates nothing on the GPU": `nvtop` shows zero VRAM in use and a 30 GB model at 256k
context consumes over **256 GB of system RAM**. The same host on Vulkan allocates correctly.

**Workaround:** "**None found.** Use the Vulkan backend on gfx1201 if you hit this."
This is the mirror image of [[gotchas/windows-hip-multi-gpu-silent-output-corruption]] —
wrong answers there, catastrophic slowness here, and both are invisible without an
independent check (a `nvtop` reading, or a cross-backend run).

### 2.15 [[gotchas/vulkan-rebar-off-collapses-amd-decode]] — the slowdown twin of a crash

`major`, `class: driver`, confidence 0.85.

With ReBAR/SAM **off**, llama.cpp's Vulkan backend "runs decode at a fraction of the card's
bandwidth": Qwen3.8-27B Q4_K_XL gives 13.89 t/s for 128-token generation and 25.28 t/s at
depth 4096, against 37.09 t/s when `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` is set.

**Workaround:** set `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, or better, enable Resizable BAR
/ SAM in BIOS — accepting [[gotchas/vulkan-rebar-on-causes-amd-device-lost]].

### 2.16 [[gotchas/vulkan-suballoc-fragmentation-decode-cliff]] — a context-length cliff with an env-var fix

`major`, `class: measurement`, confidence 0.85.

Vulkan decode on an RX 7900 XTX "falls off a cliff between 98K and 131K context":
Qwen3.8-27B Q4_K_XL with q8_0 KV and FA on measures **40.4 t/s at ctx 65536, 40.8 t/s at
98304, and 8.9 t/s at 131072**. Raising `GGML_VK_SUBALLOCATION_BLOCK_SIZE` to 4294967296
returns 131072 to **40.3 t/s**.

**Workaround:** export `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` (or higher) before
launching llama-server or llama-bench, with the record's stated trade-off: "larger blocks
raise host-memory usage when the pool does not fit in VRAM."

### 2.17 [[gotchas/rocm-flash-attention-kv-type-graph-split-explosion]] — a KV dtype that costs 68% of prefill

`major`, `class: kernel`, confidence 0.85.

With flash attention on and a **q5_0** KV cache (either K or V), the ROCm backend fragments
the compute graph from 2 splits into **34** and pushes ~550 MiB of compute work onto the CPU
host buffer. On an RX 7900 XTX with Qwen3.8-27B Q4_K_XL: pp512 falls from **957.11 → 300.10
t/s** (68% regression), decode 37.53 → 27.96.

**Workaround:** "On ROCm use **q8_0 for both K and V, or q4_0 for both. Do not mix.** If you
need a KV type ROCm's FA cannot take, either disable FA or run that configuration on Vulkan."

### 2.18 [[gotchas/sycl-quant-reorder-gap-makes-some-quants-4x-slower]] — a 4x spread between formats on one card

`major`, `class: kernel`, confidence 0.9.

On Intel Arc Pro B70, Q8_0 and IQ4_NL decode "at a fraction of the speed of Q4-family quants
despite moving comparable or less data". Full sweep on Qwen3.5-27B, one GPU, 608 GB/s
theoretical: Q4_0 23.67 t/s (57% of peak), Q4_K_S 23.05, Q4_K_M 20.56, IQ4_XS 17.52, Q6_K
13.83, Q5_K_M 13.78, IQ4_NL 5.x.

**Workaround:** "Use Q4_0, Q4_K_S, Q4_K_M, Q4_K_L or Q6_K on Battlemage." Fix for Q8_0 landed
as PR #21527.

This record is the sharpest illustration in the repo of why
[[gotchas/same-named-quant-is-not-a-controlled-comparison]] matters on hardware: **the format
you choose changes your speed by 4x on the same card in the same session**, so "Arc is slow"
is usually the wrong diagnosis.

### 2.19 [[gotchas/sycl-hybrid-linear-attention-arch-crash-intel-arc]] — architecture-specific SYCL crash

`major`, `class: kernel`, confidence 0.85.

On 2x Intel Arc Pro B60, models with a `qwen3next` or `qwen35` architecture — Qwen3-Coder-Next
at Q3_K_XL (33.79 GiB) and Q4_K_M (48.18 GiB) — "either crash inside `ggml_sycl_op_mul_mat`
or return empty/gibberish output." Build 8477 (2026-03-23) works on both; later builds do not.

**Workaround:** pin to a build at or before b8477 for these architectures on Arc, or use
Vulkan. The record's own advice generalises: "hybrid-architecture problems recur across
unrelated backends and models, so check whether the model is hybrid **before** you debug the
backend."

### 2.20 [[gotchas/sycl-multi-gpu-arc-loses-p2p-and-crashes]] — no P2P on the OpenCL adapter

`major`, `class: hardware`, confidence 0.85.

Multi-card Intel Arc dGPU fails in the peer-copy path three ways: (1) the OpenCL runtime has
no P2P at all — `ur_die` with "Experimental P2P feature is not implemented for OpenCL
adapter"; (2) under `--split-mode tensor` on dual Arc Pro B70 with P2P working,
llama.cpp crashes in `dev2dev_memcpy` with DEVICE_LOST; (3) a third report.

**Workaround:** "Use **Level Zero rather than the OpenCL adapter** for multi-card Arc. Avoid
speculative decoding on multi-card Arc until the TDR is resolved. **Test multi-card Arc on
Linux before deploying**: several reports show the same hardware working on Linux and failing
on Windows."

### 2.21 [[gotchas/vllm-arc-b60-no-pcie-p2p-kv-transfer]] — Arc cannot do PD-disaggregation

`major`, `class: framework`, confidence 0.85.

Prefill/decode disaggregation on Arc Pro B60 falls back off PCIe peer-to-peer for KV
transfer. "The blocker is upstream of vLLM: Intel's own engineer on the issue states that the
`ze_ipc` support in UCX (openucx/ucx PR #11218) is **not merged**, so there is no supported P2P
path for KV transfer on B60 today."

**Workaround:** none in vLLM. Either build against the unmerged UCX commit, or "run Arc cards
as **single-card workers rather than as a disaggregated pair**."

### 2.22 [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]] — MLX, the only Apple coverage

`major`, `class: config`, confidence 0.9. Affects [[engines/mlx-lm]].

`mlx_lm.server` starts cleanly and `/health` returns 200, then the first chat completion
raises `NotImplementedError: RotatingKVCache Quantization NYI`. Affects models whose attention
schedule is mixed — Gemma 4 uses sliding-window attention on 35 of 42 layers in the 26B-A4B
configuration, because sliding-window layers use `RotatingKVCache`.

**Workaround:** the fix landed in PR #1584 (`RotatingQuantizedKVCache` /
`BatchRotatingQuantizedKVCache`), verified on an M5 Max with mlx 0.32.0 using that branch;
until released, monkeypatch `make_prompt_cache`.

This is the **only** Apple-Silicon trap record in the repo — see §What the records cannot
answer, item 3, which is now only partly true.

### 2.23 [[gotchas/vllm-rdna4-rowwise-fp8-kernel-auto-selected]] — throughput decided by a kernel selector

`minor`, `class: framework`, confidence 0.85.

On RDNA4 (Radeon AI PRO R9700, gfx1201), serving FP8 compressed-tensors checkpoints with
`--kv-cache-dtype fp8` and `TRITON_ATTN`, "decode throughput depends on a **kernel selector**
rather than on the hardware": v0.27.1 picks `ChannelWiseTorchFP8ScaledMMLinear` and gets
35.45 t/s on gemma-4-12B-it-FP8-Dynamic; v0.28.0 picks `RowWiseTorchFP8ScaledMM`.

**Workaround:** force the ChannelWise kernel until PR #57861 lands, or pin v0.27.1 for FP8
serving on RDNA4.

---

## Tier 3 — `minor`: measurement hygiene

### 3.1 [[gotchas/same-named-quant-is-not-a-controlled-comparison]]

`minor`, `class: measurement`, confidence 0.85.

"Standard llama.cpp quantization applies hardcoded rules ('use Q4_K_M, except bump some
tensors up/down, except fall back if incompatible, except keep some tensors unquantized')" —
so a quant *name* is not a bit budget.

**Workaround:** "Normalize comparisons to a matched **bpw budget** rather than a matched quant
name" — llama.cpp's `--target-bpw` (PR #15550) "solves a constrained optimization to minimize
estimated per-tensor error subject to..." This pairs directly with
[[quantization/gguf-q4-k-m]]'s finding that Q4_K_M is a *mixture*, and with the `contested`
status on five K-quant records whose measured bpw does not match the struct comments.

### 3.2 [[gotchas/nvfp4-marlin-warning-blames-gpu-for-weight-only-checkpoint]]

`minor`, `class: measurement`, confidence 0.9.

`prepare_nvfp4_moe_layer_for_marlin()` emits "Your GPU does not have native support for FP4
computation but FP4 quantization is being used. Weight-only FP4 compression will be used..."

**Workaround:** "**Do not read this warning as a hardware capability statement.** Establish
FP4 support directly with `torch.ops._C.cutlass_scaled_mm_supports_fp4(<capability>)` — that
is the same predicate the NVFP4 linear path uses."

---

## Cross-cutting patterns

The individual records cluster into a small number of shapes, and recognising the shape is
more useful than memorising the record.

**Pattern 1 — a W4A16 checkpoint dispatched to a W4A4 kernel.** Two independent reports,
[[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] and
[[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]]. Both are silent, both cost accuracy
(7.58 points and ~1 point respectively), and **both are caught by reading one startup log
line** naming the GEMM kernel. This is a class of bug, not two incidents.

**Pattern 2 — a correctness/performance feature pair that is individually safe and jointly
unsafe.** [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] (quantized KV × spec
decode), [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] (MTP × prefix caching),
[[gotchas/spec-decode-silently-downgrades-cudagraphs]] (spec decode × attention backend ×
CUDA graphs). All three announce themselves with at most one warning that does not quantify
the cost.

**Pattern 3 — "works under transformers" does not clear the engine.** See
[[gotchas/cc-mode-uva-view-garbage-output]]'s diagnostic ordering, and
[[gotchas/flashinfer-rejects-large-head-dim]] where the same model runs fine under a
different attention backend.

**Pattern 4 — one-parameter differential diagnosis.** Three records hand you the same
technique: [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] ("change exactly one
batching parameter and diff the logits"),
[[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] (`--enforce-eager` good +
CUDA graphs garbage), and [[gotchas/rdna-triton-paged-attn-decode-cliff]] (pp flat + tg
collapsed). Each isolates a layer without a debugger.

**Pattern 5 — cross-backend agreement as the only correctness test available.** The records
that catch silent corruption all catch it the same way: run the *same model, same file, same
flags* on a second backend and compare. [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]]
makes Vulkan the reference on gfx1151; [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]]
makes perplexity the smoke test; [[gotchas/spec-decode-greedy-diverges-on-quantized-target]]
makes greedy decode the differential.

**Pattern 6 — arch-generation gates, not bugs.** [[gotchas/sm120-sparse-mla-block-size]] has
no workaround and is not waiting for one; [[gotchas/aiter-gate-skips-rdna3-gfx1100]] was
confirmed off-roadmap by a maintainer; [[gotchas/sycl-build-unrecognized-fsycl]] is blocked on
an issue. Read the confidence and `notes` fields before spending engineering time.

---

## Deployment checklist derived from the records

Not a substitute for reading the records, but every line traces to one:

1. **Cross-check every backend you plan to ship on**, same model, same file — because the
   silent cases are found only this way. ([[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]])
2. **Run `llama-perplexity` as a startup smoke test**; a two-order-of-magnitude jump is a hard
   stop. ([[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]])
3. **Validate at temperature 0 on your own prompts** before enabling any activation-quantized
   path. ([[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]])
4. **Read the startup log line naming the GEMM kernel** for every quantized checkpoint.
   ([[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]], [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]])
5. **Grep the startup log for the NVFP4 MoE scale warning** — it costs no GPU time.
   ([[gotchas/nvfp4-fused-moe-w13-global-scale-mismatch]])
6. **Check `indexer_kv_dtype` and sparse-MLA shape support before choosing SM120** for a
   DeepSeek-family MoE. ([[gotchas/sm120-sparse-mla-block-size]])
7. **Never combine speculative decoding with quantized KV**, and never with prefix caching on
   a model where output quality matters. ([[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]],
   [[gotchas/mtp-plus-prefix-caching-accuracy-loss]])
8. **Use `--dtype float16`, not `bfloat16`, for NVFP4 below SM100.**
   ([[gotchas/nvfp4-marlin-bf16-garbled-output]])
9. **Measure TTFT and ITL separately**, and measure prefix-cache hit rate with and without
   spec decode. ([[gotchas/fa4-num-splits-ignored-sm90]], [[gotchas/eagle-prefix-cache-last-block-drop]])
10. **Convert tok/s into effective bandwidth** before blaming a card or a quant.
    ([[gotchas/vulkan-decode-cliff-hidden-size-4096]])
11. **A/B FlashAttention on any non-CUDA backend** before enabling it.
    ([[gotchas/vulkan-flash-attn-slower]], [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]])
12. **On Intel dGPU, validate on your exact SKU** before committing to a long-context or
    agentic workload. ([[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]])
13. **Re-tune `--max-model-len` and `--max-num-batched-tokens` on every engine version bump**,
    and compare against `VLLM_USE_V1=0` when context regresses.
    ([[gotchas/vllm-v1-memory-footprint-growth]])
14. **Compare quantizations at matched bpw, not matched quant name.**
    ([[gotchas/same-named-quant-is-not-a-controlled-comparison]])
15. **On AMD multi-GPU Windows, verify the backend actually allocated to VRAM** — check
    `nvtop`, not `--list-devices`.
    ([[gotchas/rocm-backend-silently-serves-weights-from-host-ram-on-gfx1201]])
16. **Match K and V KV dtypes on ROCm** and prefer q8_0 or q4_0; never mix.
    ([[gotchas/rocm-flash-attention-kv-type-graph-split-explosion]])
17. **Set `GGML_VK_SUBALLOCATION_BLOCK_SIZE` before any long-context Vulkan run** — the
    cliff appears between 98K and 131K.
    ([[gotchas/vulkan-suballoc-fragmentation-decode-cliff]])
18. **On AMD Vulkan, A/B the ReBAR setting** — the two failure modes are a crash and a 4x
    slowdown, and which one you get is a BIOS toggle.
    ([[gotchas/vulkan-rebar-off-collapses-amd-decode]],
    [[gotchas/vulkan-rebar-on-causes-amd-device-lost]])
19. **On Arc Pro B60, do not plan on PD-disaggregation or multi-card** — no supported P2P
    path exists today.
    ([[gotchas/vllm-arc-b60-no-pcie-p2p-kv-transfer]],
    [[gotchas/sycl-multi-gpu-arc-loses-p2p-and-crashes]])

---

## What the records cannot answer about traps

1. **No systematic fuzzing or differential-test record.** Every trap here came from a user
   report. There is no record of an automated cross-backend differential harness, so the
   absence of a record for (engine, hardware, format) means **untested, not safe**.
2. **No fix-tracking field.** Records carry `status` and `confidence`, not "fixed in version
   X." [[gotchas/fa4-num-splits-ignored-sm90]] names the pinning commit
   `6d11b3c`, and [[gotchas/vulkan-q4km-speed-regression]] the bad commit `adc5dd92`, but
   there is no schema field for it.
3. **Apple Silicon coverage is a single record.** One record now affects [[engines/mlx-lm]]
   ([[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]], `major`, MLX KV
   quantization on hybrid-attention models) and **zero** affect [[engines/vllm-metal]] or
   [[engines/llama-cpp]] on Metal. Against ~20 CUDA and ~10 ROCm records, one Apple record is
   a gap, not a clean bill of health.
4. **No coverage for Gaudi, TPU, Ascend, Inferentia or Trainium** in the gotcha set — one
   partial exception is [[engines/vllm-pooling-models]] mentioning Gaudi, which is an engine
   record, not a gotcha.
5. **Dangling accelerator references.** Four records name accelerator ids with no accelerator
   record (§2.4 above). The consumer-Radeon parts most in need of trap coverage are the least
   represented.
6. **One blocker is not root-caused:** [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]]
   ("Not yet root-caused by a maintainer"), and its untested sibling
   (`VLLM_MARLIN_INPUT_DTYPE=int8`) is explicitly flagged as unsafe-by-omission.
9. **Severities are the reporter's, not a calibrated scale.** The field is a string enum with
   no definition in `SCHEMA.md`, and the same failure shape appears as `blocker`
   ([[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]]) and `major`
   ([[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]]) without an obvious principled
   difference. Read `severity` as an ordering hint, not a risk model.

---

## Related documents

- [01-hardware-selection.md](01-hardware-selection.md) — which part exposes which of these.
- [02-flop-map.md](02-flop-map.md) — why the crossovers behind several traps exist.
- [03-engine-selection.md](03-engine-selection.md) — the support matrix whose holes these traps sit in.
- [04-quant-selection.md](04-quant-selection.md) — the precision regimes the format traps belong to.