# 23 — Choosing a quantization format for a given hardware target

<!-- written 2026-10-04 against 3,353 records (103 quantization formats, 106 accelerators, 113 engines, 150 benchmarks). Every wikilink below uses the `[dir/id]` form and resolves to a record in data/; verified with an explicit --type against the committed tree. -->

**The decision this document serves:** you have a hardware target and a workload, and you
must pick a quantization format for it. The answer is *not* "the best format" — the repo
has no defensible ranking, and §9 says why. It is a match between three things that are
recorded separately and rarely read together:

    the format  ×  the kernel that will actually run it  ×  the engine that chooses the kernel

Any two of the three being right is not enough. Most of the expensive failures in this
repo are a format record and a kernel matrix agreeing with each other while the running
server does something else.

---

## 0. The decision rule, short form

| hardware target | pick | and know this | records |
|---|---|---|---|
| Blackwell sm100/103/107 (B200, B300, GB200) | **NVFP4** for W4A4; **MXFP4** only if you need the open standard | NVFP4 is the lower-accuracy-risk of the two; MXFP4 is the portable one | [[quantization/nvfp4]], [[quantization/mxfp4]], [[quantization/qhw-mxfp4-vs-nvfp4]] |
| Consumer Blackwell sm120 (RTX 5090, RTX PRO 6000) | **NVFP4 / MXFP4 / per-tensor FP8** — and **not** AWQ or GPTQ on TensorRT-LLM | TensorRT-LLM ships **no** kernel for any of the four AWQ/GPTQ cells on sm120, the inverse of the Ampere/Ada pattern; Marlin is the other engine's answer | [[quantization/qhw-trtllm-hardware-matrix]], [[accelerators/nvidia-rtx-5090]], [[accelerators/nvidia-rtx-pro-6000-blackwell]] |
| Hopper (H100/H200) | **FP8-E4M3** for W8A8; **AWQ or GPTQ at group 128** for W4A16 via **Machete** | FP8 **rowwise** scaling exists on Hopper and nowhere else in the vendor matrix; Machete is sm90-and-CUDA-only | [[quantization/fp8-e4m3]], [[quantization/qhw-machete-kernel]], [[quantization/qhw-trtllm-hardware-matrix]] |
| Ampere (A100, A10) | **AWQ or GPTQ at group 128** via **Marlin**; INT8-W8A8 on A100 | FP8 is **emulated** on Ampere — it is a per-tensor upcast, not a tensor-core path | [[quantization/awq]], [[quantization/gptq]], [[quantization/qhw-marlin-kernel]], [[flops/marlin-weight-only-int4-gemm]] |
| Ada (RTX 4090, RTX 6000 Ada) | **AWQ or GPTQ via Marlin**; FP8 W8A8 if the engine supports it | Volta is the only generation where GPTQ wins over AWQ — everything else is a tie | [[quantization/qhw-vllm-supported-hardware-matrix]], [[accelerators/nvidia-rtx-4090]], [[accelerators/nvidia-rtx-6000-ada]] |
| AMD Instinct (MI300X) | **INT8-W8A8**, or AWQ **only** outside vLLM | vLLM's AWQ and GPTQ have **no AMD GPU path**; the AMD-native route is Quark, not in these records | [[quantization/qhw-vllm-supported-hardware-matrix]], [[accelerators/amd-instinct-mi300x]], [[quantization/int8-w8a8]] |
| Anything wide-arch — Radeon, CDNA, RDNA, Pascal | **GGUF** via llama.cpp | Zero GGUF accelerator has a silicon path; the win is bytes, not a native kernel | [[quantization/gguf-iq4-xs]], [[flops/llamacpp-mmq-int8-dp4a-gemm]] |
| Apple Silicon | **MLX affine 4-bit**, group 64 | The group size is a **default, not a constant** — the API takes `group_size` per call | [[quantization/mlx-affine-quantized]] |
| x86 or Arm CPU | **GGUF**, or AWQ/GPTQ on x86 | **"Move W4A8 to the CPU" is false on x86** — the one W4A8 row in the matrix is Arm-CPU-only | [[quantization/qhw-vllm-supported-hardware-matrix]] |
| Any part, if you need an int4 KV cache | **KVQuant 3-bit** or **QServe 4-bit + SmoothAttention**, and nothing below 3 | No 8-bit KV record has any controlled accuracy delta at all | [[quantization/kvquant]], [[quantization/qserve-kv4]], [[quantization/qserve-w4a8kv4]] |

---

## 1. Rule 1 — a format record tells you what exists, not what runs

The single most common expensive mistake is reading `native_support` on a format record
as a statement about your GPU. It is a statement about the format. Two records in this
repo exist purely to make that distinction, and they are the most decision-relevant
material here:

- [[quantization/qhw-trtllm-hardware-matrix]] — TensorRT-LLM's own per-architecture
  kernel table, eleven format columns from NVFP4 to W4A16 GPTQ, quoted row by row.
- [[quantization/qhw-vllm-supported-hardware-matrix]] — vLLM's Implementation × Hardware
  table, nine implementation rows across Volta→Hopper plus AMD, Intel, x86 CPU and
  Arm CPU.

Both are filed under a proxy `scheme` value (`nvfp4` and `gptq` respectively) because the
schema's enum has no matrix member, and both say so in their own notes: a query for NVFP4
returns the matrix, not the format. **Read the format record for the format and the
matrix for the kernel; never quote one as the other.** The TRT-LLM matrix says it
explicitly: "Read this table as 'what TensorRT-LLM can do', never as 'what this GPU can
do'."

### 1.1 The sm120 inversion, which is the row that changes a purchase decision

The TRT-LLM matrix row for Blackwell **sm120** — the RTX 50-series and RTX PRO 6000 —
reads: NVFP4 Y, MXFP4 Y, FP8(per tensor) Y, FP8(block scaling) **`.`**, FP8(rowwise)
**`.`**, FP8 KV Y, NVFP4 KV **`.`**, and **all four AWQ/GPTQ cells `.`**.

That last clause is the finding. On Ampere the same four cells are: W4A8 `.`, W4A16
**Y**, W4A8 `.`, W4A16 **Y**. On Ada and Hopper all four are **Y**. So the consumer
Blackwell parts invert the Ampere/Ada pattern at the bottom of the precision range while
gaining formats at the top — **the one generation in the table where the newest 4-bit
format is available and the most popular 4-bit format is not, in the same engine**
([[quantization/qhw-trtllm-hardware-matrix]]).

Three consequences to carry:

1. **"My 5090 runs NVFP4" and "my 5090 runs AWQ" are both true, in different engines.**
   TRT-LLM has the NVFP4 kernel and no AWQ kernel; vLLM and SGLang serve AWQ W4A16 on
   Blackwell through Marlin and Machete, which is a different kernel set entirely. The
   matrix record carries the caveat verbatim.
2. **sm120 has no block-scaled FP8 GEMM and no NVFP4 KV-cache path**, both of which
   sm100/103 and Rubin carry. Two of the three "Blackwell supports it" reflexes are wrong
   on the consumer part.
3. **sm120 is also the row that the kernel records disagree about**
   ([[flops/sm120-vs-sm100a-kernel-binary-incompatibility]]: sm100a-compiled kernels "are
   not compatible with RTX 50 series GPUs"; [[flops/flashinfer-backend-selection]] notes
   FlashInfer lists SM100/SM103 and SM120/SM121 separately). §8 keeps this open.

### 1.2 FP8 rowwise is a Hopper-only format, which is the opposite of intuitive

FP8 **rowwise** scaling needs no per-tensor calibration, so it is the FP8 variant people
reach for by default. In the vendor matrix it is `Y` on **Hopper only** — `.` on sm120,
sm100/103, sm107, Ada and Ampere ([[quantization/qhw-trtllm-hardware-matrix]]). It is the
narrowest-supported FP8 variant in the table despite having the easiest calibration story.

Do not treat "FP8" as one row. The matrix's own recipe line is arch-dependent:
"FP8 block wise scaling GEMM kernels for sm100/103/107 are using MXFP8 recipe (E4M3
act/weight and UE8M0 act/weight scale), which is slightly different from SM90 FP8 recipe
(E4M3 act/weight and FP32 act/weight scale)." **One column covers two numerically
different scale formats**, so a memory or quality calculation for block-scaled FP8 that
assumes the SM90 recipe is wrong on Blackwell in a way the table does not flag.

Independent corroboration from the other direction: [[quantization/qhw-smoothquant-fp8-interaction]]
notes "the FP8 format has limited precision (E4M3 has 3 mantissa bits), so the migration
may not help as much as it does for INT8" — SmoothQuant was designed for INT8, and whether
it still earns its keep against FP8 is recorded as an open research question, not a result.

### 1.3 The W4A8 row is a backend-selection fact, not a hardware fact

vLLM's llm-compressor INT8 **W4A8** row is supported on **Arm CPU and on nothing else in
the matrix** — not on any NVIDIA generation, not AMD, not Intel, not x86
([[quantization/qhw-vllm-supported-hardware-matrix]]). The obvious fallback "the GPU can't
do W4A8, so put it on the CPU" is false on x86, and that is the shape of the trap: a
`.` in an NVIDIA column reads like a hardware limit when the reason is that the
implementation has one backend.

Three more cells in the same table that a flat reading gets wrong:

- **GPTQ is the only row that covers Volta.** AWQ and Marlin both exclude SM7.0. "GPTQ is
  the older format" is backwards as a compatibility statement: on a V100 it is the only
  one of the three that runs.
- **Marlin's Turing cell carries a footnote narrower than the cell.** The star expands to
  "Turing does not support Marlin MXFP4", so the cell means Marlin-minus-MXFP4; a reader
  treating the column as a flat yes picks MXFP4 on a T4 and fails.
- **FP8 W8A8 is Ada-and-Hopper-only** in this table, which is narrower than "FP8 works on
  Ampere and up" — on the Ampere row bitsandbytes and GGUF are YES while llm-compressor
  FP8 is NO, at the same compute capability.

The matrix covers **Volta through Hopper only; Blackwell is absent from it**. That is why
it must be read together with the TRT-LLM matrix rather than instead of it, and it is why
its own scope line warns the chart "is subject to change as vLLM continues to evolve".

---

## 2. Rule 2 — verify which kernel is actually running, every deploy

Given §1, the format you chose is a hypothesis. The running kernel is the fact. Four
records describe the same failure from four directions, and it is the most consequential
class in the repo: **the output is correct and the performance is not what you bought.**

- [[quantization/qhw-silent-precision-degradation]] (confidence 0.9) is the primary record.
  SGLang's `--fp4-gemm-backend` auto-selection order is "1) flashinfer_cutedsl on SM100,
  2) **marlin on SM80-SM90**, 3) flashinfer_cutlass otherwise (including SM120)". Marlin is
  a W4A16 kernel, so an NVFP4 checkpoint on an A100 or H100 is served with FP4 weights
  dequantized to FP16 and FP4 activations upcast to FP16. The record's own phrasing is
  "The user sees the memory saving but not the compute speedup", and the two
  kernel-selection records state the consequence as "The memory saving materializes but
  the compute speedup does not"
  ([[quantization/qhw-silent-precision-degradation]],
  [[quantization/qhw-sglang-kernel-selection]],
  [[quantization/qhw-vllm-kernel-selection]]). The same behaviour is documented for vLLM's
  `modelopt_fp4`: SGLang's platform table, quoted in its record, "shows that modelopt_fp4
  uses 'Marlin W4A16 fallback on Ampere/Hopper and native FP4 backends on Blackwell'"
  ([[quantization/qhw-sglang-kernel-selection]]).
- **The framework names the condition.** vLLM's `--kernel-config` example is
  `--linear-backend cutlass --kernel-config
  {"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}` — `nvfp4_w4a16` is the scheme
  name a W4A16-served NVFP4 checkpoint carries in the config
  ([[quantization/qhw-vllm-kernel-selection]]).
- **And the roofline explains why it costs you.** [[flops/qhw-decode-gemm-memory-bound]]:
  "at decode time with batch size 1, the GEMM is memory-bound, so reducing weight bytes
  directly increases tokens/s ... when NVFP4 is served with Marlin (W4A16) on pre-Blackwell
  hardware, the weight bytes are reduced but the activation bytes are not, so the speedup
  is less than the theoretical maximum."

**Three checks that cost nothing and catch all of it:**

| check | what a bad answer looks like | source |
|---|---|---|
| Read the startup log's kernel/backend line | a W4A16 kernel named for a W4A4 checkpoint | [[quantization/qhw-silent-precision-degradation]], [[quantization/qhw-sglang-kernel-selection]] |
| Read the scheme name in the engine config | `nvfp4_w4a16` | [[quantization/qhw-vllm-kernel-selection]] |
| Ask the hardware directly: `torch.ops._C.cutlass_scaled_mm_supports_fp4(<capability>)` | false on the part you are about to ship FP4 on | [[gotchas/nvfp4-marlin-warning-blames-gpu-for-weight-only-checkpoint]] |

The third check exists because the warning you would naturally trust is a **measurement**
bug, not a capability statement: "Your GPU does not have native support for FP4
computation" is emitted for a **weight-only** checkpoint, and the record's instruction is
"Do not read this warning as a hardware capability statement."

The inverse mistake is also recorded and it is worse, because it is silent and wrong in
the other direction: [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]] (**major**) — a
checkpoint marked `W4A16_NVFP4` served with `--moe-backend flashinfer_b12x` runs the
**FP4×FP4** micro-kernel, and the record's instruction is "**Do not accept a b12x trace
showing an FP4 kernel for a W4A16 checkpoint.**" Two benchmark records price that
misroute on an identical model: [[benchmarks/mxfp4-w4a16-on-w4a16-marlin-aime-math-gsm8k-avg]]
**78.16** versus [[benchmarks/mxfp4-w4a16-on-w4a4-kernel-aime-math-gsm8k-avg]] **70.58**,
against the unquantized [[benchmarks/qwen3-8b-bf16-aime-math-gsm8k-avg-reference]]
**82.36**. **A 7.58-point accuracy drop caused purely by kernel selection.**

---

## 3. Rule 3 — on Blackwell, NVFP4 beats MXFP4 on risk and loses on portability

Both are W4A4 with real FP4 tensor-core paths, and they are **not interchangeable** — a
model quantized to MXFP4 cannot be served with NVFP4 kernels or vice versa
([[quantization/qhw-mxfp4-vs-nvfp4]]).

| | MXFP4 | NVFP4 |
|---|---|---|
| scale group | 32 values | 16 values |
| scale format | E8M0, power-of-two | E4M3, fractional |
| scale levels | one (per block) | two (E4M3 per block + FP32 per tensor) |
| standard | OCP open spec, multi-vendor | NVIDIA-proprietary |
| `native_support` | B200, B300, GB200-NVL72 | B200, B300, GB200-NVL72 |

**Recommendation: NVFP4 by default, MXFP4 when portability is a requirement.** The
accuracy argument is the vendor's own: NVIDIA's comparison table rates MXFP4 as "**risk
of noticeable accuracy drop compared to FP8**" where NVFP4 is "lower risk", and the
quantitative reason is that encoding the scale in E4M3 rather than E8M0 "**drops the
scale-encoding MSE from 0.72 to 0.08**" ([[quantization/nvfp4]],
[[quantization/qhw-mxfp4-vs-nvfp4]]).

**Two honesty constraints on that recommendation.** The MSE ranking is a vendor claim, and
[[quantization/qhw-mxfp4-vs-nvfp4]] records that "no independent controlled benchmark
comparing MXFP4 and NVFP4 on the same model with the same calibration data was found."
And [[quantization/mxfp4]] has **no inference quality delta recorded at all** — the
closest evidence is a PTQ study of SmoothQuant+AWQ+GPTQ on MX formats, which is
W4**A8**, not W4A4. So the format preference rests on a vendor table, and the MXFP4 record
says in its own words that no MXFP4 accuracy number should be quoted from it.

Both formats are Blackwell-only in silicon; on Hopper and Ampere both are
`emulated_support`, which is §2's failure mode rather than a slow path you can accept.

The mechanism underneath is in [[flops/block-scaled-tensor-core-fp4-fp8]]: the scale is
"part of the **MMA OPERAND**, not a separate kernel", which is why "the activation
quantization no longer needs a fused dequant pass". Its portability warning is the one to
remember when you change parts: "SM90 requires scaling factors in FP32 while SM100 requires
them packed in UE8M0 ... So **switching an FP8 model from H100 to B200 changes the memory
layout the engine must produce**, not just the kernel it calls."

**MoE is a separate question from dense.** [[quantization/qhw-moe-mxfp4]] (draft, 0.5)
records that per-tensor rather than per-expert scales are used for MoE, that "the block
scaling in MXFP4 (32 elements) may not align well with MoE's expert structure", and flags
"a potential source of silent accuracy degradation that has not been well-documented."
Treat MoE MXFP4 as unproven on accuracy; do not extrapolate the dense result.

---

## 4. Rule 4 — W4A16: group size is the scheme, and the kernel is chosen for you

### 4.1 Group size 128 is not a tunable

[[quantization/awq]]: "Group size 128 is the scheme's defining choice and the file layout -
**it is not a tunable**." [[quantization/gptq]] adds that 32/64/1024 "are legal and change
both quality and kernel eligibility, which is a frequent source of 'GPTQ is slow' reports."
**If you are handed a GPTQ file at group 64, you are not holding the format the docs
describe**, and the kernel eligibility is a separate question from the quality.

The 128 default is also the recommended default in the toolkit that implements the low-bit
path: AutoRound confirms "bits 2 through 8, every one with group_size=128 and sym=True and
act_bits=16", and its troubleshooting section recommends 32 or mixed bits for better
results — "so 128 is the default, not the optimum"
([[quantization/autoround-w4a16]]).

**The asymmetry limit is the practical line in that record.** `--asym` supports weight bits
**≤ 7** for the auto_round/auto_gptq/auto_awq exports, "because vLLM serves W8 GPTQ-format
weights symmetric-only and Marlin supports zero points at 4 bits only." Two constraints,
one from each side of the stack, that agree.

And a labelling trap in the same record: its README labels MXFP4, MXINT4, MXFP4_RCEIL,
MXFP8, FPW8A16 and FP8_STATIC as "**Research feature, no real kernel**" — do not read them
as deployable.

### 4.2 Which kernel runs, per generation — three records, one precision

[[quantization/qhw-marlin-kernel]] and [[quantization/qhw-machete-kernel]] are the
decision records, and both carry the same load-bearing fact: **Marlin and Machete produce
the same numerical results.** Marlin's is explicit — "despite being called an 'INT4 kernel',
the actual tensor-core operation is **FP16 × FP16** - weights are dequantized to FP16 in
registers before the multiply. This is **NOT** an INT4 tensor-core path (which does not
exist on Ampere/Ada). The speedup comes from reduced memory bandwidth (4-bit weight
storage), not from faster arithmetic." Machete's record says the same for Hopper: "the
kernel is numerically equivalent to Marlin for the same quantized weights, so the Marlin
quality figures apply."

So the kernel choice is **purely a performance decision, and it is invisible in the
output**. Selection logic, from [[quantization/qhw-machete-kernel]]: "vLLM uses Machete
when the GPU is SM90 (Hopper) and the quantization method is AWQ or GPTQ. On Ampere/Ada,
Marlin is used instead."

| part | W4A16 kernel | the number to carry |
|---|---|---|
| Volta SM7.0 | none (GPTQ only, no Marlin) | [[quantization/qhw-vllm-supported-hardware-matrix]] |
| Turing SM7.5 | Marlin, **minus MXFP4** | [[quantization/qhw-marlin-kernel]] |
| Ampere / Ada SM8.0+ | **Marlin** | [[flops/marlin-weight-only-int4-gemm]] — "near-ideal 4x speedups up to batchsizes of 16-32 tokens" on A10, 3.87x theoretical max after 0.125 bits/weight scale overhead |
| Hopper SM9.0 | **Machete** — Marlin is "not yet optimized for Hopper" | [[quantization/qhw-machete-kernel]], [[flops/machete-weight-only-hopper-gemm]] |
| anything (incl. RDNA, CDNA, Pascal) | **llama.cpp MMQ** | [[flops/llamacpp-mmq-int8-dp4a-gemm]] — integer SIMT DP4A, "not tensor-core MMAs", widest arch reach of the three |

The honest comparison is in the MMQ record and it should be quoted whenever anyone ranks
these: llama.cpp "reaches 4-bit throughput through integer SIMT dot products with a **broad
arch reach** (including RDNA and CDNA), while Marlin and Machete reach it through tensor
cores on a **narrower arch list**. Neither is strictly better ... and **any comparison must
state which path was taken**."

Two more kernel facts that cost people deployments. Marlin is a GEMM kernel, not a
format, and both records say so — Marlin "is filed under 'gptq' because Marlin consumes
GPTQ-packed weights and the enum has no kernel-family values", and the proxy will "make
Machete under 'awq' and Marlin under 'gptq' look like two different precisions when they
are the same one". And Marlin's ROCm path has a hard failure worth pre-checking:
[[gotchas/triton-w4a16-gptq-qzeros-assert]] (**major**), a
`qzeros.shape == (K // group_size, N // 8)` assertion when serving GPTQ-Int4 on the ROCm
Triton w4a16 path.

### 4.3 The accuracy budget you can actually cite

[[quantization/qhw-accuracy-cost-summary]] compiles the deltas that papers actually
report, and its pattern line is the most quotable summary in the repo: "**4-bit weight-only
quantization typically costs 0.1-0.3 ppl on WikiText-2. 3-bit costs 0.5-1.0 ppl. 2-bit
costs 2.0+ ppl. KV cache quantization at 4-bit is nearly lossless; at 2-3-bit it costs
0.1-0.5 ppl.**" The same record's own caveat is the load-bearing part: "The accuracy cost
is model-dependent and benchmark-dependent."

The controlled numbers behind it, with models attached:

| method | model / benchmark | delta | records |
|---|---|---|---|
| Marlin INT4 g128 | Llama2-7B WikiText-2 / MMLU | 5.12 → 5.27 (**+0.15 ppl**); MMLU 41.80 → 40.07 (**−1.73**) | [[quantization/qhw-marlin-kernel]] |
| Marlin INT4 g128 | Llama2-13B / 70B | Wikitext +0.10 / +0.09; MMLU −0.97 / −0.62 | [[quantization/qhw-marlin-kernel]] |
| GPTQ 3-bit vs group-128 | Llama-7B WikiText-2 | +2.39 ppl at plain 3-bit, **+0.93 at group 128** | [[quantization/qhw-accuracy-cost-summary]] |
| AWQ INT3 | OPT-6.7B WikiText-2 | 43.2 (RTN) vs 13.0 (**−30.2**) — mechanism demo, not a W4A16 delta | [[quantization/awq]], [[benchmarks/awq-int4g128-wikitext2-ppl-delta-llama2-7b]] |
| GGUF Q4_K_M | Llama-3.1-8B WikiText-2 | 7.56 vs 7.32 F16 (**+0.24**); unweighted mean −0.46%; size −69.41% | [[quantization/gguf-q4-k-m]], [[benchmarks/gguf-q4-k-m-wikitext2-ppl-delta-llama3-1-8b]] |
| SpQR 3–4 bit + sparse outliers | LLaMA, Falcon | **<1% perplexity loss**; 4x+ memory, 15% speedup | [[quantization/qhw-spqr]] |
| SignRound 2-bit | 11-task zero-shot average | **+6.91% to +33.22%** over RTN | [[quantization/qhw-signround]] |

Two rows in that table are the actionable ones. **At 3 bits, group size buys more than the
method does** (+0.93 vs +2.39 ppl for the same format at the same bit width). And
**rounding is a first-class lever at 2 bits**, where [[quantization/qhw-signround]] records
"6.91-33.22% absolute average accuracy improvement" — but it is a rounding *algorithm*, it
"cannot be combined with other quantization methods (like AWQ's scaling or GPTQ's
Hessian-based rounding) without re-running the optimization," and it is implemented by
AutoRound ([[quantization/autoround-w4a16]]).

### 4.4 Bitrate is a target, not a format — EXL2/EXL3

[[quantization/exl2]] mixes quantization levels **within a model**, and can "apply multiple
quantization levels to each individual linear layer - to hit any target average bitrate
between 2 and 8 bpw." Its successor [[quantization/exl3]] is a different format with "no
checkpoint compatibility between them" ([[engines/exllamav3]]) — a lattice-coded,
Hessian-calibrated QTIP derivative rather than a grid quantizer.

The measured throughput curve is the decision-relevant part, from
[[quantization/qhw-exllama-throughput]]: **lower bit width is slower, not faster.**
2.5 bpw 70B runs 33/38 t/s on a 3090 Ti / 4090 while 4.0 bpw 7B runs 185/211 — because
"lower bit widths require more complex dequantization and the memory bandwidth saving is
offset by increased compute overhead." The same mechanism is
[[quantization/qserve-w4a8kv4]]'s central claim from the serving side: "existing INT4
methods suffer **20-90% runtime overhead** from dequantizing weights or partial sums on
GPUs", and INT4 "can be slower than fp16" unless you also fix the dequant path.

**Neither record carries a quality delta** — [[quantization/exl2]]'s `quality_delta` is
empty because the README publishes throughput only, and [[quantization/qhw-exllama-throughput]]
says "No quality delta is published in the README - only throughput numbers." Choosing
EXL2/EXL3 is choosing an unquantified quality risk for a bitrate target. EXL3's group-size
256 is also the **embedding** group specifically — "Each embedding is split into 256-wide
groups that are rotated and trellis-coded without calibration, and the following layers are
calibrated against the quantized table" — which is §5's trap in a second disguise.

---

## 5. Rule 5 — GGUF: match bits-per-weight, and do not trust a group size of 256

### 5.1 The group-size correction, which changes what you can compare

Eight GGUF records previously carried `weight_group_size: 256`. **Seven of those were
wrong, and the error was the K-quant super-block size being read as the scale group.** The
correction was made from the ggml dequantize routines, and it matters because 256 vs 16/32
is the difference between "one fp16 scale per 256 values" and "eight or sixteen sub-block
scales per 256 values".

| record | `weight_group_size` now | the actual structure | why 256 was wrong |
|---|---|---|---|
| [[quantization/gguf-q4-k-m]] | **32** | 8 sub-blocks of 32 per 256-value super-block, each a 6-bit scale + 6-bit min on fp16 d/dmin | "Base type Q4_K has a 32-value scale group" |
| [[quantization/gguf-q4-k-s]] | **32** | same base type, unmixed | same |
| [[quantization/gguf-q5-k-m]] | **32** | same base type, unmixed | same |
| [[quantization/gguf-q6-k]] | **16** | `int is = l/16`, 16 int8 sub-block scales per 256 | "Base type Q6_K has a 16-value scale group" |
| [[quantization/gguf-q3-k-m]] | **16** | `scales[is++]` once per 16 values, 16 sub-block scales per 256 | same |
| [[quantization/gguf-q2-k]] | **16** | one (scale,min) pair per 16 values, 16 sub-blocks per 256 | same |
| [[quantization/gguf-iq4-nl]] | **32** | `QK4_NL = 32`, one fp16 d across all 32, **no 256 super-block at all** | "the 256 was inherited from the K-quant convention" |
| [[quantization/gguf-iq2-xxs]] | 256 | one fp16 `d` for the whole 256-value block, **no sub-block scale plane** | *not* an error — "the super-block size and the scale group DO coincide at 256" |

Four records are **legitimately** at 256 and must not be "corrected" again, because for
them the super-block and the scale group genuinely coincide: [[quantization/gguf-iq1-s]]
("256 IS right here and the distinction matters" — one fp16 d for 256 values, no sub-block
scales), [[quantization/gguf-iq2-xxs]], [[quantization/gguf-iq3-xxs]], and
[[quantization/gguf-tq1-0]] ("TQ1_0 genuinely has a 256-value scale group ... unlike the
K-quants").

**The rule this yields: in GGUF the number that matters for storage is bits-per-weight,
and the number that matters for quality comparison is bits-per-weight on a *matched
budget* — not a group size, and never a quant name.** See §5.2.

### 5.2 Measured bpw, the number to match on

Llama-3.1-8B, measured average bits per weight, straight from the records:

| bpw | records |
|---|---|
| 1.6875 / 2.0625 | [[quantization/gguf-tq1-0]] (ternarized, base-3 packing; 3⁵ = 243 ≤ 256 is the packing constraint) |
| 2.0042 – 2.1460 | [[quantization/gguf-iq1-s]], [[quantization/gguf-iq1-m]] (codebook i-quants) |
| 2.3125 – 2.9294 | [[quantization/gguf-iq2-xxs]], [[quantization/gguf-iq2-xs]], [[quantization/gguf-iq2-s]], [[quantization/gguf-iq2-m]] |
| 3.2548 – 3.6606 | [[quantization/gguf-iq3-xxs]], [[quantization/gguf-iq3-s]] |
| 2.625 / 2.9697 | [[quantization/gguf-q2-k]] |
| 3.9960 | [[quantization/gguf-q3-k-m]] (`contested`) |
| 4.4597 – 4.8944 | [[quantization/gguf-iq4-xs]], [[quantization/gguf-iq4-nl]], [[quantization/gguf-q4-0]], [[quantization/gguf-q4-k-s]], [[quantization/gguf-q4-k-m]] (`contested`), [[quantization/gguf-legacy-q4-1]] |
| 5.7036 | [[quantization/gguf-q5-k-m]] (`contested`) |
| 6.5633 | [[quantization/gguf-q6-k]] (`contested`) |
| 8.5008 | [[quantization/gguf-q8-0]] |

Five K-quant records are `contested` because the measured bpw of the `*_M` mixture types does
not match the pure-block struct comments; both numbers are retained per record.

**Two names are not two configurations.** [[quantization/gguf-q4-k-m]]: "'Q4_K_M **IS A
MIXTURE**, not a type'" — base q4_K with attn_v and ffn_down promoted to Q6_K, and "The
plain `'Q4_K'` CLI name is an **alias for Q4_K_M**". The consequence is that a quant name
does not identify a bit budget across models; see
[[gotchas/same-named-quant-is-not-a-controlled-comparison]] and llama.cpp's
`--target-bpw`, which is the tool that fixes it.

**The GGUF case of the mixture trap is bigger than the K-quant one.**
[[quantization/gguf-mxfp4-moe]] is llama.cpp's `LLAMA_FTYPE_MOSTLY_MXFP4_MOE`: MXFP4 on
expert tensors and **Q8_0 on everything else** — "the embedding, output head, and any
non-expert projection are stored at 8.5 bpw while the experts sit at 4.25. So the
whole-model bpw of an MXFP4_MOE model is **NOT** 4.25**." Its mapping is explicit about
3-D tensors and BailingMoE3 MLA projections. Note also that for MX formats
"the scale group IS the block" (`QK_MXFP4 32`), which is why [[quantization/gguf-mxfp4-moe]]
and [[quantization/mxfp4]] agree at 32 and need no correction.

### 5.3 The lever that costs nothing: imatrix

[[quantization/gguf-imatrix]] is "not a format - a per-tensor activation-energy file" whose
own note calls it "THE MAJOR PRACTICAL LEVER NOBODY RECORDS". It applies across every
format ("every ggml type, every bit width, legacy and k-quant and i-quant alike") and its
effect "is to **REMOVE part of the quantization delta** of the format it is applied to",
with official per-type figures Q4_K_M **+0.1754** and Q5_K_M **+0.056** — which is most of
[[quantization/gguf-q4-k-m]]'s +0.24 total delta.

**If you are quantizing a GGUF yourself, this is a higher-return action than choosing a
different quant type**, and it composes with nothing else in this document.

### 5.4 GGUF has no silicon anywhere

Worth stating because it inverts the usual question. [[quantization/gguf-iq4-xs]]: "so
`native_support` is deliberately EMPTY and the whole accelerator ecosystem is
`emulated_support`. **No accelerator has an i-quant or k-quant path in silicon. This is the
single most commonly botched field in this repo.**" No GGUF quant will ever be faster
*because the GPU supports it*; it is faster because there are fewer bytes, reached by MMQ
([[flops/llamacpp-mmq-int8-dp4a-gemm]]).

---

## 6. Rule 6 — 8-bit KV cache: widely deployed, and completely unmeasured

This is the recommendation with the weakest evidence base in the repo, and the honest
thing is to say so rather than pick a winner.

[[quantization/qhw-fp8-kv-cache]] and [[quantization/qhw-int8-kv-cache]] are both `draft` at
confidence **0.5**, and both `quality_delta` fields say the same thing: **no controlled
benchmark with a named accuracy delta was found.** FP8 KV's field in full: "No controlled
benchmark with named accuracy delta found in the sources accessed ... This is a
significant gap: FP8 KV cache is widely deployed but the accuracy impact is not
well-documented in public sources."

What *is* recorded, and it is enough to act on:

- Both halve KV memory versus FP16.
- The kernel interaction is not free: "the KV cache is quantized after the KV projection,
  so the attention kernel must **dequantize before computing attention scores**. This
  dequantization adds latency to the decode phase."
- Support is not uniform: FP8 KV is native on Hopper and Ada, "On Ampere (SM80), FP8 is
  **emulated** via FP16 with reduced precision", and "The accuracy delta is expected to be
  larger on Ampere where the FP8 path is emulated."
- INT8's record argues a mechanism preference: "INT8 is symmetric around zero while FP8 has
  an exponent. For attention keys that feed into softmax, the **INT8 format may be more
  appropriate** because the key distribution is roughly symmetric." That is a *may* in the
  record's own words.
- **And a silent-corruption combination exists**, which outranks the accuracy question:
  [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] — FP8 KV plus sliding-window
  attention plus speculative decoding, which is the exact configuration the FP8 KV record
  itself warns about.

**Recommendation: use 8-bit KV to fit the cache, not to save accuracy risk** — you are
accepting an unquantified delta in exchange for 2x capacity — and never combine it with
sliding-window attention and speculative decoding until you have measured that
combination. Establish the delta yourself; the repo cannot do it for you.

For **4-bit and below**, there is no such ambiguity, only disagreement. The family
disagrees about the right key bit width and the repo says so: "Note the whole family (KIVI
2/4, KVQuant 3/3, QuaRot 4/4, QServe 4/4, ZipCache mixed) disagrees about the right key bit
width - there is no settled answer" ([[quantization/qserve-kv4]]).

| method | what it does | the number | record |
|---|---|---|---|
| [[quantization/kvquant]] | 3-bit keys and values, per-channel | **<0.1 ppl degradation** on WikiText-2 and C4 across LLaMA, Llama-2, Llama-3, Mistral; LLaMA-7B at 1M context on one A100-80GB | [[quantization/kvquant]] |
| [[quantization/kivi-kv2]] | **asymmetric** 2-bit keys / 4-bit values | "almost the same quality" (unquantified) at **2.6x less peak memory**, 2.35–3.47x throughput | [[quantization/kivi-kv2]] |
| [[quantization/qserve-kv4]] | 4-bit KV + **SmoothAttention** as a correction | 1.2x/1.4x (LLaMA-3-8B, A100/L40S) and 2.4x/3.5x (Qwen1.5-72B) vs TensorRT-LLM | [[quantization/qserve-kv4]], [[quantization/qserve-w4a8kv4]] |
| [[quantization/zipcache]] | mixed precision by token salience | **4.98x** compression for **−0.38%** GSM8k on Mistral-7B | [[quantization/zipcache]] |

**Recommendation: KVQuant 3-bit if you want the best-documented number; KIVI if you want
the most capacity per unit of risk, accepting a qualitative claim; QServe 4-bit only with
SmoothAttention attached**, since its record is explicit that SmoothAttention "is a
**CORRECTION METHOD** ... introduced specifically to counter the accuracy degradation that
4-bit KV quantization causes." And note the asymmetry result is KIVI's central finding, not
an incidental detail — do not carry 4-bit KV and 4-bit values forward to 2-bit keys
assuming the halves are independent.

**One non-quantization answer is sometimes correct.** [[quantization/pensieve]] "solves the
SAME problem as KIVI ... by storing it across tiers instead of compressing it", and the
record's judgement is the useful part: "Since lossless storage is strictly better than
lossy compression when the constraint is **host memory** rather than GPU memory, Pensieve is
the correct answer when an offload path exists and KIVI is the correct answer when it does
not."

---

## 7. Rule 7 — the non-post-training families, which are not choices at all

Two families in the records look like format choices and are not, and picking them as if
they were is a category error.

- **Ternary / BitNet b1.58** — [[quantization/ternary]], [[quantization/bitnet-b158]]:
  "ternary LLM quantization is a **TRAINING-TIME recipe, not a post-training one.** There is
  no such thing as PTQ-ing an existing fp16 LLM to 1.58 bits and keeping its quality." The
  paper's "matches fp16" is a statement about the ternary set's expressiveness under an
  identical token budget. `native_support` is empty on both records. **You cannot select
  this for an existing checkpoint.**
- **W4A4 research without silicon** — [[quantization/spinquant]] is genuinely W4A4 (64.0
  average zero-shot on LLaMA-2 7B, 2.9 points from full precision,
  [[benchmarks/spinquant-w4a4kv4-zeroshot-accuracy-delta-llama2-7b]]) but "there is **NO
  hardware INT4 tensor-core path**. Its W4A4 numbers come from research Triton kernels plus
  fake quantization". [[quantization/atom]] is W4A4 with real integer operators, and its
  record draws the distinction that matters: Atom "is built around 4-bit integer operators -
  but that is an integer-operator path on datacenter GPUs, **not the FP4 tensor-core path
  that Blackwell introduced**". Cite these for accuracy evidence; do not cite them for speed.
  Its own throughput claim (7.73x vs FP16, 2.53x vs INT8) is from the authors' own kernels.

---

## 8. What would change these recommendations

| change | which recommendation flips | record to re-read |
|---|---|---|
| A **vendor** kernel lands for sm120 AWQ/GPTQ | §0 consumer-Blackwell row: AWQ/GPTQ become options on TRT-LLM, and the inversion disappears | [[quantization/qhw-trtllm-hardware-matrix]] |
| vLLM or SGLang adds an **SM120 NVFP4 kernel with activation quantization** | §2 stops being a loss on consumer Blackwell — verify the kernel is W4A4, not Marlin | [[quantization/qhw-silent-precision-degradation]], [[quantization/qhw-sglang-kernel-selection]] |
| An **independent** MXFP4-vs-NVFP4 benchmark on one model and one calibration set appears | §3's NVFP4-over-MXFP4 preference, which currently rests on a vendor table | [[quantization/qhw-mxfp4-vs-nvfp4]] |
| `nvfp4_w4a16` stops being a selectable scheme name in vLLM | §2's third check becomes unnecessary and the failure mode becomes unreachable-by-config | [[quantization/qhw-vllm-kernel-selection]] |
| A controlled **8-bit KV cache** accuracy delta is published | §6 becomes a choice between FP8 and INT8 rather than an unmeasured risk | [[quantization/qhw-fp8-kv-cache]], [[quantization/qhw-int8-kv-cache]] |
| Per-expert MXFP4 scales reach a shipping MoE checkpoint | §3's MoE carve-out lifts | [[quantization/qhw-moe-mxfp4]] |
| Marlin gains an **Marlin-on-Hopper** optimisation | §4.2's Hopper row moves from Machete to Marlin on performance grounds only | [[quantization/qhw-marlin-kernel]], [[quantization/qhw-machete-kernel]] |
| llama.cpp's IQ-quant kernels land in the **llamafile size-optimized CUDA build** | §5's widest-arch GGUF recommendation narrows, since that build "omits the IQ-quant kernels" | [[quantization/gguf-mxfp4-moe]] |
| MI350X/MI355X get quantization-support records | AMD's row in §0 becomes answerable instead of inferred | [[accelerators/amd-instinct-mi355x]] |

---

## 9. What the records cannot answer

1. **There is no cost-per-quality ranking of formats.** The deltas in §4.3 are isolated
   perplexity differences at isolated bit widths with no shared baseline methodology, and
   every one of them has its model name attached. Two of the cleanest comparisons are
   *within* a family (Marlin vs AWQ on the same model; NVFP4 vs MXFP4 on the same card) and
   neither crosses a bit width.
2. **The vLLM matrix stops at Hopper and the TRT-LLM matrix stops at Rubin.** Blackwell's
   vLLM story has to be inferred from the other table plus the sm120 kernel records, which
   **disagree with each other** ([[flops/sm120-vs-sm100a-kernel-binary-incompatibility]] vs
   the format claim that sm120 is native for NVFP4). No record reconciles them, and there
   are no `native_support` entries for the RTX 5090 or RTX PRO 6000 from any format record —
   they appear in matrix records and `emulated_support` lists only.
3. **Zero controlled accuracy data for 8-bit KV cache**, and the least-documented area
   named by the records themselves is "FP8/FP4 KV cache quantization, where no controlled
   benchmark with named accuracy delta was found" ([[quantization/qhw-accuracy-cost-summary]]).
4. **Format quality deltas are overwhelmingly Llama-3.1-8B or Llama-2-7B.** No format has a
   controlled delta on a modern MoE, which is exactly where quantization behaviour is
   hardest — and MXFP4-MoE is the place where the format and the model structure are
   recorded as possibly mismatched.
5. **The W4A4 speed claim on Blackwell is vendor-only.** The 3.5x memory and 25x/50x
   energy figures in [[quantization/nvfp4]] are labelled "marketing, not measurements" by
   the record itself.
6. **EXL2 and EXL3 have no quality delta at all**, and [[quantization/exl2]] is `draft` at
   0.6 against an archived upstream ([[engines/exllamav2]]).
7. **SmoothQuant-on-FP8 is a research question, not a result**
   ([[quantization/qhw-smoothquant-fp8-interaction]], draft, confidence **0.4** — the
   lowest-confidence record cited here).
8. **AMD has no vLLM AWQ/GPTQ path and no Quark record in this set.** The vLLM matrix's
   AMD column is a whole-column NO that "reads like a data gap" and "must not be read as
   'AMD GPUs cannot do quantized inference'" — the AMD-native route the record names is
   covered by quark and rocm-quant records that this document does not cite.
9. **A schema gap that makes Apple advice indirect, and a 04/23 consistency note.**
   [[quantization/mlx-affine-quantized]] is the native format on Apple records and
   lists `apple-m3-max` and `apple-m3-ultra` in `native_support`, at group 64 with
   `emulated_support` empty. [04-quant-selection.md](04-quant-selection.md) §3.2 said
   "no native quantization support recorded at all" for those two parts; **that text
   has been corrected to agree with this document**, so the two no longer contradict
   each other and the base layer and the selection layer say the same thing about
   Apple. What remains open is not the native/emulated question but everything
   downstream of it:
   - **MLX's own `mlx-quantized-4bit` / `mlx-quantized-8bit` engine labels still have
     no record of their own.** The MLX affine record is the closest thing, and it is
     keyed on the affine mode with group 64 rather than on the engine's label.
   - **Three benchmark records still carry `format_id: mlx-quantized-4bit`, which is
     not a record id** — the format record's own notes name these as the dangling
     references its existence was meant to close, and repointing them is a
     `data/benchmarks/` edit this wave does not own.
   - **"Native" is not "comparable".** MLX ships four modes at four group sizes
     (affine 64, mxfp4 32, nvfp4 16, mxfp8 32), so a checkpoint recorded as "4-bit on
     Apple" may be a 64-group affine grid or a 32-group MX grid, and those differ in
     memory arithmetic and quality. The nvfp4 mode at 16 matches the Blackwell
     convention in [[quantization/nvfp4]] and the mxfp4 mode at 32 matches
     [[quantization/mxfp4]]; affine 64 matches neither.

   **For the selection consequence on Apple: MLX affine 4-bit, group 64, and treat the
   group size as a default rather than a constant** — `mlx.nn.quantize` takes
   `group_size` per call, so the number in the record is the API's default and not a
   property of the format.

---

## Related documents

- [04-quant-selection.md](04-quant-selection.md) — the regime taxonomy (W4A16 vs W4A4),
  the per-accelerator native/emulated census, and the quality tables this document draws on.
  Read §3 of that doc as the base layer; read this one for the selection step.
- [01-hardware-selection.md](01-hardware-selection.md) — which part to buy, per precision
  regime. The matrix rows here are the reason its Blackwell and Ampere recommendations
  differ.
- [03-engine-selection.md](03-engine-selection.md) — which engine supports which format on
  which backend, and the build costs of the two engines named in §1.
- [05-known-traps.md](05-known-traps.md) — the full severity-ordered list behind §2.
- [09-model-memory-envelope.md](09-model-memory-envelope.md) — whether the format you
  picked actually fits, and what KV cache it leaves room for.
- [08-cost-per-token.md](08-cost-per-token.md) — what the precision choice costs per
  million output tokens, joined against real prices.
