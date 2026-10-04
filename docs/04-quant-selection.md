# 04 — Quantization: weight-only-4bit vs W4A4 are different regimes

<!-- generated against a snapshot of 1,087 records (52 quantization formats, 74 accelerators, 40 engines). Counts were a snapshot at authoring time. -->

The single most consequential distinction in this repo is that **"4-bit" describes two
regimes that share no arithmetic, no hardware path, no accuracy profile and no failure
modes.** Conflating them is, in the words of [[quantization/wq4a4]], "the single most
common error in quantization blog posts."

| | **Weight-only 4-bit (W4A16)** | **W4A4** |
|---|---|---|
| What is quantized | weights only | weights **and** activations |
| Compute dtype | fp16/bf16/int8 | fp4 (Blackwell) or simulated |
| Hardware 4-bit path | **none needed** — dequant-to-fp16 into tensor cores | **Blackwell only** (fp4 block-scaled MMA) |
| What it is for | **memory capacity** | **FLOPs and bytes, simultaneously** |
| Records | [[quantization/awq]], [[quantization/gptq]], [[quantization/exl2]], [[quantization/bitsandbytes-nf4]], every GGUF k-quant, [[quantization/spinquant]] | [[quantization/nvfp4]], [[quantization/mxfp4]], [[quantization/wq4a4]] |

The category record [[quantization/wq4a4]] splits "W4A4" into three further things, and the
distinctions matter more than the headline:

> "(1) **Integer W4A4 with INT4 tensor cores — does not exist** on shipping NVIDIA or AMD
> parts; Hopper and older have no INT4 path at all and Ampere-era 'INT4' marketing was
> sparsity, not 4-bit multiply. (2) **FP4 W4A4 — this IS native**, on Blackwell, and that is
> exactly what [[quantization/nvfp4]] is. (3) **Simulated/fake W4A4 in research papers** —
> SpinQuant, QuaRot, QuIP# — which measure quality but buy no speed without custom kernels."

And the honest state of the art, in TensorRT-LLM's own naming: "it ships W4A16 and W4A8
variants of AWQ and GPTQ and **no W4A4 row at all**."

---

## 1. Regime A — weight-only 4-bit (W4A16)

### 1.1 What it actually is

[[quantization/awq]] states the definition crisply: "AWQ is w4a16. Weights are 4-bit;
activations and the KV cache stay 16-bit, so there is **no hardware 4-bit-activation path
and nothing on an INT4 tensor core is being used**. Group size 128 is the scheme's defining
choice and the file layout - it is not a tunable." And the runtime mechanism: "Runtime
support is entirely software dequant-to-fp16 inside a fused GEMM."

[[quantization/gptq]] adds the crucial framing: "GPTQ's contribution is the rounding
criterion (approximate inverse Hessian from calibration), **not a new number format** - a
GPTQ file is INT4/INT3 with per-group scales, the same layout AWQ produces, **which is why
the two share every serving kernel**."

That shared-kernel fact is why group size dominates every "GPTQ is slow" report:
[[quantization/gptq]] notes "Group size 128 is the conventional default; 32/64/1024 are
legal and change both quality and kernel eligibility, which is a frequent source of 'GPTQ is
slow' reports." There is a gotcha record for exactly this —
[[gotchas/triton-w4a16-gptq-qzeros-assert]] (**major**): serving GPTQ-Int4 fails on the
ROCm Triton w4a16 path with a `qzeros.shape == (K // group_size, N // 8)` assertion.

### 1.2 Why it works at all: the two kernels, and the arch split

This is the most decision-relevant piece of the whole regime, because it determines which
NVIDIA part you want:

- **[[flops/marlin-weight-only-int4-gemm]]** — "removes the dequantization tax by moving it
  offline and restructuring the kernel so dequant and MMA are ordered so neither pipeline
  stalls." Design target as a ratio: "as long as we perform **less than 25-50 tensor-core
  multiply-accumulates per 4-bit quantized weight**, the FLOP:byte ratio of the GPU permits
  near-ideal 4x speedup, and the theoretical benefit of weight-only quantization should
  **extend to batch sizes 4-8x larger than prior kernels achieved**."
  **Hardware floor, verbatim:** "requires CUDA >= 11.8 and 'NVIDIA GPU with compute
  capability >= 8.0 (Ampere or Ada, **Marlin is not yet optimized for Hopper**)'."
  So: Ampere/Ada → Marlin is the fast path. Hopper → Marlin is the *fallback*.
- **[[flops/machete-weight-only-hopper-gemm]]** — the Hopper answer. It "keeps CUTLASS's
  Hopper WGMMA tensor-core mainloop and changes the B OPERAND LAYOUT instead."
  **And it is emphatic about what this is not:** "this is a tensor-core bf16/fp16 GEMM over
  int4 weights dequantized into the MMA fragments on the fly, **NOT a hardware int4
  tensor-core path** - there is no int4 MMA instruction being used."
  **Floor, hard-coded in vLLM:** `get_min_capability()` returns 90 and `can_implement()`
  returns False with "Machete requires compute capability of 90 (Hopper)" plus "Machete only
  supported on CUDA". So "a Hopper box gets Machete, an Ampere or Ada box falls through to
  Marlin, a consumer Blackwell (SM120) is not SM90 so also falls through, and **ROCm is
  excluded outright**."
- **[[flops/llamacpp-mmq-int8-dp4a-gemm]]** — the third answer, and it is a different
  machine entirely: MMQ "quantizes the ACTIVATIONS to q8_1 blocks of 32 elements and
  performs the dot product in INT8 accumulating to INT32, using DP4A where available. So the
  multiply-accumulates are **integer SIMT instructions, not tensor-core MMAs** - the
  FLOP:byte ratio the tensor cores exploit is not available at all." Its arch reach is the
  widest: "per-arch config headers exist for Ampere, Blackwell, CDNA, GCN, RDNA2/3/4/5 and
  Pascal (DP4A)."

The record states the honest comparison directly: "on a consumer or datacenter GPU,
llama.cpp reaches 4-bit throughput through integer SIMT dot products with a **broad arch
reach** (including RDNA and CDNA), while Marlin and Machete reach it through tensor cores
on a **narrower arch list**. Neither is strictly better - MMQ works on more parts, Machete
reaches much higher tensor-core utilization where it applies - and **any comparison must
state which path was taken**."

**Three W4A16 arch paths, one per part generation. Any advice that does not name the
kernel is incomplete advice.**

### 1.3 Quality, where the records have controlled numbers

| Record | Model | Delta vs FP16 | Cite |
|---|---|---|---|
| [[quantization/gguf-q4-k-m]] | Llama-3.1-8B-Instruct, WikiText-2 | ppl **7.56 vs 7.32** F16 (+0.24) — "the smallest perplexity cost of any 4-bit option measured"; GSM8K 69.41 vs 77.63; IFEval 72.35 vs 78.93; unweighted mean 69.15 vs 69.47 (−0.46%); size 4685 MiB (−69.41%) | benchmark: [[benchmarks/gguf-q4-k-m-wikitext2-ppl-delta-llama3-1-8b]] |
| [[quantization/awq]] | OPT-6.7B INT3-g128 | WikiText ppl **43.2 (plain RTN) vs 13.0** (AWQ's salient-1%-in-fp16 comparison) — a demonstration of mechanism, not a W4A16 delta | benchmark: [[benchmarks/awq-int4g128-wikitext2-ppl-delta-llama2-7b]] +0.13, [[benchmarks/awq-int4g128-wikitext2-ppl-delta-llama2-13b]] +0.09, [[benchmarks/awq-int4g128-wikitext2-ppl-delta-mistral-7b]] +0.16, [[benchmarks/awq-int3g128-wikitext2-ppl-delta-opt-6-7b]] +0.53 |
| [[quantization/gguf-q3-k-m]] (record is `contested`) | Llama-3.1-8B | GSM8K **−9.32 pp** | benchmark: [[benchmarks/gguf-q3-k-s-gsm8k-accuracy-delta-llama3-1-8b]] |
| [[quantization/smoothquant]] (W8A8, not W4A16) | OPT-175B | ppl **+0.18** | benchmark: [[benchmarks/smoothquant-w8a8-wikitext-ppl-delta-opt-175b]] |

**The Q4_K_M correction that matters more than any of the numbers above.** The record's
notes: "IMPORTANT CORRECTION to the common 'Q4_K_M mixes 6-bit and 4-bit sub-blocks' claim:
**it does not**. All 256 weights in block_q4_K are uniform 4-bit. What is mixed is the
METADATA: the 8 sub-blocks of 32 elements each have a 6-bit scale and a 6-bit min." The
quality advantage over legacy q4_0 therefore "comes from 6-bit per-32 sub-block scales plus
per-32 mins on an fp16 super-block scale, **not from wider weights**."

And "Q4_K_M IS A MIXTURE, not a type" — per `src/llama-quant.cpp`, base q4_K with attn_v and
ffn_down promoted to Q6_K. The plain `'Q4_K'` CLI name is an **alias for Q4_K_M**.

**Which also means quant names are not comparable across models.** See
[[gotchas/same-named-quant-is-not-a-controlled-comparison]] (**minor**): "Standard llama.cpp
quantization applies hardcoded rules ('use Q4_K_M, except bump some tensors up/down, except
fall back if incompatible, except keep some tensors unquantized')" — so "normalize
comparisons to a matched bpw budget rather than a matched quant name", which llama.cpp's
`--target-bpw` now solves.

### 1.4 The lever nobody records: imatrix

[[quantization/gguf-imatrix]] is "not a format - a per-tensor activation-energy file", and
its note calls it "THE MAJOR PRACTICAL LEVER NOBODY RECORDS". Mechanism: "a per-tensor table
of sum-of-squared activations (Sigma(Act^2))... fed back into llama-quantize as
`--imatrix file_name` to weight the rounding-error objective per weight, so weights with
high activation energy get their bits spent more carefully." It applies **across formats** —
"every ggml type, every bit width, legacy and k-quant and i-quant alike." Its own value: "its
effect is to **REMOVE part of the quantization delta** of the format it is applied to," with
official per-type ppl strings Q4_K_M +0.1754 and Q5_K_M +0.056.

### 1.5 Format family roundup with real bits-per-weight numbers

All K-quants share `QK_K = 256` per block with 12 bytes of packed sub-block scales
([[quantization/gguf-q4-k-m]]). Measured average bpw on Llama-3.1-8B, from the same record
family:

| bpw | format records |
|---|---|
| 1.6875 (TQ1_0) / 2.0625 (TQ2_0) | [[quantization/gguf-tq1-0]] — ternarized, base-3 packing, 3⁵=243 ≤ 256 is the packing constraint |
| 2.0042 – 2.1460 | [[quantization/gguf-iq1-s]], [[quantization/gguf-iq1-m]] (codebook i-quants) |
| 2.3824 – 2.9294 | [[quantization/gguf-iq2-xxs]], [[quantization/gguf-iq2-s]], [[quantization/gguf-iq2-m]] |
| 3.2548 – 3.6606 | [[quantization/gguf-iq3-xxs]], [[quantization/gguf-iq3-s]] |
| 3.9960 | [[quantization/gguf-q3-k-m]] (`contested`) |
| 4.4597 – 4.8944 | [[quantization/gguf-iq4-xs]], [[quantization/gguf-iq4-nl]], [[quantization/gguf-q4-0]], [[quantization/gguf-q4-k-s]], [[quantization/gguf-q4-k-m]] (`contested`) |
| 5.7036 | [[quantization/gguf-q5-k-m]] (`contested`) |
| 6.5633 | [[quantization/gguf-q6-k]] (`contested`) |
| 8.5008 | [[quantization/gguf-q8-0]] |

Five K-quant records are **`contested`** (gguf-q3-k-m, gguf-q4-k-m, gguf-q4-k-s,
gguf-q5-k-m, gguf-q6-k) — because the measured bpw of the *_M mixture types does not match
the pure-block struct comments. Both numbers are retained per record.

### 1.6 Two low-bit families that are not post-training quantization at all

- **Ternary / BitNet b1.58** — [[quantization/ternary]], [[quantization/bitnet-b158]]. "The
  framing that matters: ternary LLM quantization is a **TRAINING-TIME recipe, not a
  post-training one.** There is no such thing as PTQ-ing an existing fp16 LLM to 1.58 bits
  and keeping its quality." The paper's "matches fp16" means "a ternary model trained from
  scratch on the same token budget matches an fp16 model trained from scratch on the same
  token budget" — a statement about the ternary set's expressiveness, **not** about
  quantization loss on a pretrained checkpoint. `native_support` is empty on both records.
- **EXL2** — [[quantization/exl2]] (`draft`, 0.6). "EXL2 uses the same optimization method as
  GPTQ... but **mixes quantization levels within a model** - and, per the upstream README,
  can apply multiple quantization levels to each individual linear layer - to hit any target
  average bitrate between 2 and 8 bpw." Its `quality_delta` is empty and the record says why:
  the README publishes throughput (Llama2-7B EXL2 4.0 bpw at 185 tok/s on a 3090 Ti, 211 on
  a 4090) but **no perplexity or benchmark deltas vs fp16**, and none was found elsewhere.

---

## 2. Regime B — W4A4, and the only genuine hardware-native 4-bit path

### 2.1 NVFP4: the one native W4A4

[[quantization/nvfp4]] is the record that "most needs the two distinctions kept separate":

**(1) It is genuinely w4a4.** "the element format is E2M1 (1 sign, 2 exp, 1 mantissa,
values ~-6..6) and **BOTH weights and activations are quantized to it**, with a Blackwell
hardware path - unlike MXFP4 and unlike every w4a16 scheme above."

**(2) It is NOT the OCP microscaling format**, despite the identical element layout:
- scale granularity: NVFP4 **16** values, MX **32** values
- scale format: NVFP4 uses a "*power-of-two-free* **E4M3 FP8** scale"; MX uses an
  "**E8M0 power-of-two** scale"
- NVFP4 adds a **second per-tensor FP32 level**: "s_global = global_amax/(448*6),
  s_block = (block_amax/6)/s_global"

The vendor's own quantitative reason for the E4M3 scale: "encoding the scale itself in E4M3
rather than E8M0 **drops the scale-encoding MSE from 0.72 to 0.08**."

**Native on Blackwell only.** "TensorRT-LLM's hardware matrix marks NVFP4 Y on Blackwell
sm120, sm100/103 and Rubin, and **no on Hopper/Ada/Ampere**." `native_support`:
[[accelerators/nvidia-b200]], [[accelerators/nvidia-b300]], [[accelerators/nvidia-gb200-nvl72]].
`emulated_support`: [[accelerators/nvidia-h100-sxm]], [[accelerators/nvidia-a100-80gb-sxm4]],
[[accelerators/amd-radeon-pro-vii]].

**Quality — with the baseline stated correctly.** The record's `quality_delta` is careful:
"FP8 vs NVFP4, seven benchmarks (**vendor-reported**)" on DeepSeek-R1-0528 — MMLU-Pro 85% vs
84%, GPQA Diamond 81% vs 80%, HLE 15% vs 14%, LiveCodeBench 77% vs 76%, SciCode 40% vs 40%,
Math-500 98% vs 98%, AIME 2024 89% vs 91%. NVIDIA states "1% or less accuracy degradation
versus FP8." And the record's caveat is the important one: "**This is a vendor comparison
against an FP8 baseline, not against fp16, and is vendor-claim-grade.**"

Its "vendor memory/energy claims (~3.5x vs FP16, 25x/50x energy efficiency vs H100) are in
the blog and are **marketing, not measurements**."

### 2.2 MXFP4: native, but a different format with a different risk profile

[[quantization/mxfp4]]: "MXFP4 is the OCP microscaling 4-bit format: E2M1 elements under a
shared power-of-two E8M0 scale for every 32-value block." Same `native_support`
(Blackwell B200/B300/GB200-NVL72), same `emulated_support` set.

The vendor-preference statement is the one to carry: "NVIDIA's own comparison table rates
MXFP4 as 'accelerated hardware scaling: **yes**' but '**risk of noticeable accuracy drop
compared to FP8**', where NVFP4 is 'lower risk of noticeable accuracy drop'... i.e. **the
vendor prefers its own format on accuracy grounds**."

Its `quality_delta` is empty, and the record says why: "No inference fp16 delta recorded
from a controlled benchmark. The closest primary evidence is a PTQ study of SmoothQuant+AWQ+
GPTQ applied to MX formats, which reports 4-bit weights with 8-bit activations in MXINT with
'negligible accuracy loss compared to the uncompressed baseline'." **Do not quote an MXFP4
accuracy number from this record.**

Note also that MXFP4 is **not** MX-INT4: on Blackwell, "blockwise FP8" effectively means
[[quantization/mxfp8]] (E4M3 act/weight with UE8M0 scales on sm100/103/107), whereas the SM90
FP8 recipe is E4M3 with FP32 scales.

### 2.3 The rest of the block-scaled family

[[flops/block-scaled-tensor-core-fp4-fp8]] gives the mechanism and the exact PTX shapes:
warp-level `mma` has block-scaled forms where "the multiplicand type is .e4m3/.e5m2/.e3m2/
.e2m3/.e2m1 times a .ue8m0 scale at .m16n8k32, and .e2m1 times (.ue8m0 or .ue4m3) at
.m16n8k64", and `wgmma` mirrors it at warpgroup level with fp8 at .m64nNk32. The payoff:
"the scale is part of the **MMA OPERAND**, not a separate kernel, so a 32-element scale
block is applied inside the tensor-core pipeline and **the activation quantization no longer
needs a fused dequant pass**."

Three Blackwell-only formats carry it in the data: [[quantization/nvfp4]] (grp 16),
[[quantization/mxfp4]] (grp 32), [[quantization/mxfp8]] (grp 32), plus
[[quantization/fp8-blockwise]] (grp 32). And **sparsity composes with sub-byte**:
[[flops/structured-2-4-sparsity-mma-sp]] notes "For FP4 on Blackwell, CUTLASS lists 2:4
sparsity support for FP4 as a separate feature - sparsity and sub-byte precision compose,
which is the **largest single byte reduction available**."

**The portability tax, which bites exactly when you switch parts:** [[flops/block-scaled-tensor-core-fp4-fp8]]
and [[flops/deepgemm-grouped-masked-moe-gemm]] both record that "SM90 requires scaling
factors in FP32 while SM100 requires them packed in UE8M0 with four packed into a single
int, and the LHS scale must have a TMA-aligned transposed layout for both. So **switching an
FP8 model from H100 to B200 changes the memory layout the engine must produce**, not just
the kernel it calls."

### 2.4 W4A4 with no silicon: SpinQuant and friends

[[quantization/spinquant]] is the cleanest illustration of the confusion and is worth
reading as the cautionary record. It **is** genuinely W4A4 — "both weights and activations
quantized to 4-bit" — and its accuracy claim rests on rotations: W4A4KV4 on LLaMA-2 7B gives
an average zero-shot reasoning accuracy of **64.0**, a 2.9-point gap from full precision,
while LLM-QAT is 22.0 points behind and SmoothQuant 25.0 points behind under identical
precision (benchmarks: [[benchmarks/spinquant-w4a4kv4-zeroshot-accuracy-delta-llama2-7b]],
[[benchmarks/spinquant-w4a8kv8-zeroshot-accuracy-delta-mistral-7b]],
[[benchmarks/spinquant-random-vs-learned-rotation-spread-mistral-7b]]).

**And then the sentence that decides the hardware question:** "**there is NO hardware INT4
tensor-core path.** Its W4A4 numbers come from research Triton kernels plus fake
quantization, not from silicon. On current NVIDIA parts 4-bit x 4-bit is not a tensor-core
op; **compare nvfp4, which is W4A4 *and* native, on Blackwell.**"

The engine record for the harness says the same from the serving side
([[engines/spinquant-research-harness]]): "**NO hardware INT4 tensor-core path**: W4A4KV4
accuracy is a fake-quant / research-kernel result, not a silicon result", plus "SpinQuant_had
needs an online Hadamard kernel, ~8% of network latency overhead per the paper."

---

## 3. Native vs emulated: the per-hardware map

`native_support` / `emulated_support` are populated on the records. The counts below were
computed by tallying those two fields across all format records at the snapshot — they are
a summary of the records, not figures quoted from any single one. Where a cell is a bare
count, cite the accelerator record and re-tally if the data has moved:

| Accelerator | Native formats | Emulated formats |
|---|---|---|
| [[accelerators/nvidia-h100-sxm]] | **13** | 27 |
| [[accelerators/nvidia-l40s]] | **12** | 19 |
| [[accelerators/nvidia-a100-80gb-sxm4]] | **10** | 22 |
| [[accelerators/amd-instinct-mi300x]] | **8** | 13 |
| [[accelerators/nvidia-b200]] / [[accelerators/nvidia-b300]] | **8** (all the block-scaled formats) | see below |
| [[accelerators/nvidia-gb200-nvl72]] | **7** | — |
| [[accelerators/nvidia-l4]] | **8** | 12 |
| [[accelerators/nvidia-rtx-4090]] / [[accelerators/nvidia-a10]] | **6** | 16 / 20 |
| [[accelerators/nvidia-h100-nvl]] | 5 | 8 |
| [[accelerators/intel-gaudi2]] | 3 | — |
| [[accelerators/nvidia-h100-pcie]] / [[accelerators/nvidia-a30]] | 3 | 2 |
| [[accelerators/intel-gaudi3]] | 2 | 2 |
| [[accelerators/intel-arc-b580]] | 2 | 18 |
| [[accelerators/amd-instinct-mi250x]] | 2 | — |
| [[accelerators/amd-radeon-pro-vii]] | 1 | 21 |
| [[accelerators/apple-m3-max]] / [[accelerators/apple-m3-ultra]] | **0** | 1 ([[quantization/mlc-quant-q4f16]]) |

Read this table as the answer to "why did my format not speed anything up?"

### 3.1 What "native" means per format, and the field most likely to mislead you

**GGUF has ZERO native support on every accelerator in the repo — and the record says so
directly.** [[quantization/gguf-iq4-xs]]: "Weight-only by construction... so
`native_support` is deliberately EMPTY and the whole accelerator ecosystem is
`emulated_support`. **No accelerator has an i-quant or k-quant path in silicon. This is the
single most commonly botched field in this repo.**" [[quantization/gguf-tq1-0]] repeats it:
"**No accelerator has a GGUF quant path in silicon.**"

This is not a criticism of GGUF — GGUF's win is *portability and density*, not silicon
support. But if you are choosing a format to buy hardware for, note the consequence:
**no GGUF quant will ever be faster because the GPU supports it.** It is faster because
there are fewer bytes, and llama.cpp's MMQ path reaches them without tensor cores
([[flops/llamacpp-mmq-int8-dp4a-gemm]]).

**AWQ, GPTQ, EXL2 and NF4 all have empty `native_support`.** Every one is
[[quantization/awq]]'s "software dequant-to-fp16 inside a fused GEMM". The speed comes from
kernel engineering (Marlin, Machete, MMQ), never from silicon.

**FP8 is the genuinely native 8-bit regime.** [[quantization/fp8-e4m3]]: "FP8 is genuinely
**w8a8 with fp32 accumulate**, so the low-precision operands actually reach the tensor core.
Hardware-native from Hopper on NVIDIA... and from CDNA3 on AMD — MI300X lists 2614.9 TFLOPs
dense fp8, again 2x fp16." And the boundary: "**NOT native on Ampere** (the A100
accelerator record has no fp8 entry) so FP8 on A100 is emulation by upcast." That matches
the emulated lists exactly: A100 and A10 carry fp8-e4m3 in `emulated_support`; L4, L40S,
H100, H200-class and Blackwell carry it natively.

### 3.2 Per-hardware guidance

**NVIDIA Hopper ([[accelerators/nvidia-h100-sxm]], [[accelerators/nvidia-h100-pcie]],
[[accelerators/nvidia-h100-nvl]])** — the richest native menu in the repo (13 / 3 / 5
formats) but **none of the 4-bit-block-scaled formats**. Concretely: FP8-E4M3 native
(H100 lists 1979 TFLOPS dense fp8, 2x its fp16); INT8-W8A8 native; W4A16 via **Machete**
(sm90, CUDA only). Blackwell-only formats NVFP4, MXFP4, MXFP8, FP8-blockwise and W4A4 are
all `emulated_support` here. Note [[engines/vllm]]'s caveat that FP8 "needs Ada or newer"
and [[quantization/fp8-e4m3]]'s "not native on Ampere".

**NVIDIA Blackwell datacenter ([[accelerators/nvidia-b200]], [[accelerators/nvidia-b300]])** —
the only place W4A4 is real. All eight native formats are the block-scaled family
(nvfp4, mxfp4, mxfp8, fp8-blockwise, fp8-e4m3, fp8-e5m2, modelopt, wq4a4). Critically,
**Machete does not apply** (not sm90), so W4A16 on Blackwell is Marlin or a w4a16 misroute —
which is a gotcha, not a preference (§4).

**Consumer Blackwell ([[accelerators/nvidia-rtx-5090]], [[accelerators/nvidia-rtx-pro-6000-blackwell]])**
— the trap. [[quantization/nvfp4]] says TensorRT-LLM's matrix marks NVFP4 "Y on Blackwell
**sm120**, sm100/103 and Rubin", i.e. SM120 counts as native for the *format*. But
[[flops/sm120-vs-sm100a-kernel-binary-incompatibility]] says sm100a-compiled kernels "are
not compatible with RTX 50 series GPUs", and [[flops/flashinfer-backend-selection]] notes
FlashInfer lists SM100/SM103 and SM120/SM121 separately and that FP4 GEMM (NVFP4/MXFP4) is
"for Blackwell GPUs" **without promising SM120**. There are **no native_support lists for
RTX 5090 or RTX PRO 6000 at all** — they appear only in emulated lists. So the format claim
and the kernel reality do not line up, and the records do not resolve it.

**NVIDIA Ada consumer ([[accelerators/nvidia-rtx-4090]])** — 6 native formats, all
weight-only or int8 paths (autoround-w4a16, bitsandbytes-load-in-4bit, llmcompressor-quant,
ort-matmulnbits-int4, ort-quant-int8, ortgenai-int4-fp8). **No FP8 in `native_support`**
despite the 4090 record listing 330.3 dense FP8 TFLOPS — FP8-E4M3 is not on its native list.
W4A16 goes through Marlin (Ampere or Ada, per [[flops/marlin-weight-only-int4-gemm]]).

**NVIDIA Ampere ([[accelerators/nvidia-a10]], [[accelerators/nvidia-a100-80gb-sxm4]])** —
Marlin's home turf (compute capability ≥ 8.0). But **FP8 is emulated**, and both appear in
`emulated_support` for fp8-e4m3/e5m2. Int8-W8A8 is native on A100, **emulated on A10**.

**NVIDIA L4 ([[accelerators/nvidia-l4]])** — 8 native formats including fp8-e4m3/e5m2 and
int8-w8a8. Its record warns the memory side anyway: "300 GB/s is roughly a tenth of H100
SXM, so it is a **fit-many-small-models card, not a fit-one-big-model card**."

**NVIDIA L40S ([[accelerators/nvidia-l40s]])** — 12 native formats, the second-richest in the
repo, and 48 GB. Its record also carries the clearest dense/sparse warning in the repo
(§2.5 of [01-hardware-selection.md](01-hardware-selection.md)).

**AMD Instinct ([[accelerators/amd-instinct-mi300x]], [[accelerators/amd-instinct-mi250x]])** —
8 native formats on MI300X, all int8 or weight-only paths. The vLLM constraint from
[[engines/vllm]] is the practical limit: "Marlin (GPTQ/AWQ/FP8/FP4) has no AMD, Intel or CPU
path; **AWQ and GPTQ have no AMD GPU path**." So MI300X's native menu exists but the two most
popular W4A16 formats are unavailable there. MI350X/MI355X are **absent from every
native_support and emulated_support list in the repo** — a genuine gap, not an emulated
finding (§5).

**Intel Gaudi ([[accelerators/intel-gaudi2]], [[accelerators/intel-gaudi3]])** —
[[quantization/neural-compressor-stack]] is native on both and nothing else is, which inverts
the pattern: "advanced low-bit LLM/VLM quantization (2-4 bit, MXFP4, NVFP4, FP8 KV cache)
is **not implemented in Neural Compressor — it is integrated FROM AutoRound**". Its note
warns: "Treating the two as one tool would **misattribute the group_size and
asymmetric-quantization restrictions**." Gaudi3 has just 2 native and 2 emulated; its
emulated list is [[quantization/moe-mixed-precision]] and [[quantization/moe-routing-quant-interaction]].

**Intel Arc ([[accelerators/intel-arc-b580]], [[accelerators/intel-arc-pro-b60]])** — 2 native
(autoround-w4a16, ort-matmulnbits-int4) and 18/13 emulated. Combined with
[[engines/llama-cpp]]'s Intel caveats and [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]],
Arc is the thinnest native-quant story in the repo. Its record also notes the capacity wall:
12 GB at 456 GB/s, and PCIe **x8 not x16**.

**Apple Silicon ([[accelerators/apple-m3-max]], [[accelerators/apple-m3-ultra]])** — **no
native quantization support recorded at all.** The only entry anywhere is
[[quantization/mlc-quant-q4f16]] in `emulated_support`. MLX's own formats
([[engines/mlx-lm]]: `mlx-quantized-4bit`, `mlx-quantized-8bit`) are Metal-native by
construction but are **not** in the quantization schema's enum, so the repo has no format
record for them — a schema gap, recorded in §5.

**The parts with no quantization story whatsoever** — [[accelerators/amd-radeon-pro-vii]]
(1 native, 21 emulated), [[accelerators/cerebras-wse-3]], [[accelerators/groq-lpu]],
[[accelerators/google-tpu-v4]] / [[accelerators/google-tpu-v5e]] /
[[accelerators/google-tpu-v5p]] / [[accelerators/google-tpu-v6e]],
[[accelerators/tenstorrent-wormhole-n150]], [[accelerators/tenstorrent-blackhole-p150]],
[[accelerators/aws-inferentia2]], [[accelerators/aws-trainium2]], and every
`asia`-market ASIC record — appear in **no** native or emulated list.

---

## 4. The gotchas, grouped by which regime they poison

Silent-corruption-first ordering; the full list is in
[05-known-traps.md](05-known-traps.md).

**Silently wrong output, weight-only path:**
- [[gotchas/nvfp4-marlin-bf16-garbled-output]] — **blocker**. NVFP4 + `--dtype bfloat16` on
  SM < 100 (4090 SM89, V100 SM70): loads, starts, serves, every response garbled.
- [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]] — **blocker**. sm_121a + WNA16 INT4 MoE +
  `VLLM_MARLIN_INPUT_DTYPE=fp8`: coherent-looking repetition loop, and "the corrupted path is
  also ~2.5% FASTER end-to-end, **so throughput benchmarking selects for the bug**."
- [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]] — **major**. A checkpoint marked
  `W4A16_NVFP4` served with `--moe-backend flashinfer_b12x` runs the **FP4xFP4** micro-kernel.
  Silent wrong output, and the record says explicitly: "**Do not accept a b12x trace showing
  an FP4 kernel for a W4A16 checkpoint.**"
- [[gotchas/nvfp4-fused-moe-w13-global-scale-mismatch]] — **major**. The fused-MoE weight
  processor "keeps only the gate per-tensor global scale. When a checkpoint ships different
  scale_2 values..." The check is cheap and needs no GPU: grep the startup log.
- [[gotchas/nvfp4-gguf-conversion-drops-block-scales]] — **major**. NVFP4 → GGUF conversion
  "produces a GGUF that does not load".
- [[gotchas/nvfp4-marlin-warning-blames-gpu-for-weight-only-checkpoint]] — **minor**, but it is
  a *measurement* gotcha: "'Your GPU does not have native support for FP4 computation'..." is
  emitted for a **weight-only** checkpoint. "Do not read this warning as a hardware capability
  statement." Establish FP4 support with
  `torch.ops._C.cutlass_scaled_mm_supports_fp4(<capability>)`.
- [[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] — **major**, and the one with the clearest
  measured cost. On B200, a compressed-tensors MXFP4 checkpoint with `input_activations: null`
  (i.e. **W4A16**) logs "Using..." and routes to a W4A4 kernel. Two benchmark records capture
  the difference on the identical model: [[benchmarks/mxfp4-w4a16-on-w4a16-marlin-aime-math-gsm8k-avg]]
  **78.16** versus [[benchmarks/mxfp4-w4a16-on-w4a4-kernel-aime-math-gsm8k-avg]] **70.58**,
  against the unquantized [[benchmarks/qwen3-8b-bf16-aime-math-gsm8k-avg-reference]] at
  **82.36**. That is a **7.58-point accuracy drop caused purely by kernel misrouting** — the
  best single illustration in the repo of why the W4A16/W4A4 distinction is not pedantry.
- [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] — **major, silent**. ROCm/gfx1151,
  llama-perplexity PPL jumps two orders of magnitude.
- [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — **major**. On gfx1151, "the HIP
  backend is wrong and Vulkan is right", same machine, same build.
- [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] — **major, silent**. "Any prompt
  longer than n_ubatch produces badly wrong logits on HIP/ROCm on gfx1151 - no error, no
  warning, no crash, and **the numbers look plausible**. Default n_ubatch is 512."

**Silent-corruption consequence of combining quantization with other features:**
- [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] — **blocker**. MTP speculative decoding plus
  prefix caching on a finetuned MoE: ~20% internal classification accuracy drop.
- [[gotchas/spec-decode-greedy-diverges-on-quantized-target]] — **major**. Greedy sampling with
  spec decode produces different text on a quantized target. n-gram speculation is verified
  lossless.
- [[benchmarks/nvfp4-w4a16-marlin-moe-gsm8k]] — the same checkpoint measured through both
  paths: Marlin **0.9136** vs FlashInfer b12x **0.9037** GSM8K. Both `verified`.

**Build/toolchain:**
- [[gotchas/torch-source-build-abi-mismatch]] — **blocker**, affects any custom-quant build.

---

## 5. What the records cannot answer

1. **MI350X / MI355X have no quantization-support entry anywhere** — no native list, no
   emulated list, despite having MLPerf records using WMXFP4 weights with FP8 KV
   ([[benchmarks/mi355x-mlperf-v6-0-llama2-70b-wmxfp4-offline-tokens]]). This is the single
   most consequential gap in this document: the newest AMD datacenter part has no recorded
   quantization compatibility.
2. **No native quantization record for any Apple format.** MLX 4-bit/8-bit are Metal-native
   by construction but have no quantization record (the scheme enum has no MLX member). Any
   Apple quantization advice is currently inferred from the engine record, not from a format
   record.
3. **SM120 vs SM100 for NVFP4 is unresolved.** The format record says SM120 is native; the
   kernel records say sm100a kernels do not run on SM120. No record reconciles them, and
   there are **no** `native_support` entries for RTX 5090 or RTX PRO 6000 at all.
4. **No quality delta for MXFP4 from a controlled benchmark** — the record says so explicitly,
   and the only evidence is a PTQ paper with 8-bit activations, not W4A4.
5. **Format quality deltas are overwhelmingly Llama-3.1-8B or Llama-2-7B.** No format has a
   controlled delta on a modern MoE, which is exactly where quantization behaviour is hardest.
6. **No cost-per-quality curve.** Records give isolated perplexity deltas at isolated bit
   widths with no shared baseline methodology, so there is no defensible "best 4-bit format"
   ranking — the comparisons in §1.3 have to be quoted with their model names attached.
7. **The quality-vs-FLOPs claim for W4A4 on Blackwell is vendor-only.** The 3.5x memory and
   25x/50x energy-efficiency figures in [[quantization/nvfp4]] are labelled "marketing, not
   measurements" by the record itself.
8. **EXL2 has no quality delta at all**, and the format is `draft` at 0.6 with an archived
   upstream.

---

## Related documents

- [01-hardware-selection.md](01-hardware-selection.md) — which part to buy for each precision regime.
- [02-flop-map.md](02-flop-map.md) §3.7 — why quantization speedups are batch-conditional.
- [03-engine-selection.md](03-engine-selection.md) — which engine supports which format on which backend.
- [05-known-traps.md](05-known-traps.md) — every silent-corruption gotcha, severity-ordered.