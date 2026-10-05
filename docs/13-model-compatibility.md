# 13 — Model × engine compatibility: which architecture needs which kernel, and which engine has it

Every cell below cites a record id. **No cell in this document is an inference from an
engine's general reputation.** Where a record is silent the cell reads `unknown`, and
`unknown` is a real answer here: the repo has 71 engine records, and almost none of them
enumerate architecture support. The three that do — [[engines/vllm]], [[engines/sglang]] and
[[engines/tensorrt-llm]] — carry the load for most of the matrix, and one of them
([[engines/tensorrt-llm]], via [[sources/mdl2-trtllm-supported-models]]) is the only source in the repo that
publishes a per-architecture *feature* matrix rather than a per-architecture support list.

Section 7 lists every cell marked `unknown`, with the reason for each.

## 0. What this document adds

Before this document the join did not exist. Model records in `data/models/` carried
`architecture`, `attention_variant`, `kv_cache_bytes_per_token` and a derivation in `notes`;
engine records carried `supported_backends`, `best_for`, `avoid_for`. Nothing connected them.
The result was three specific gaps:

1. **Architecture → required kernel/ISA was unstated.** MLA needs a latent decode kernel, not a
   GQA kernel. A hybrid needs *two* kernel families in one graph. A diffusion LM needs no KV
   cache and no decode loop at all. None of that was written down next to the model.
2. **KV-cost accounting was engine-blind.** `kv_cache_bytes_per_token` is a single number, but
   for a sliding-window or hybrid model it is the *naive* figure — the figure an engine that
   does not implement the architecture's cache structure will actually allocate. The gap
   between the two was nowhere.
3. **Kernel choice silently determines cache format.** Not an abstraction: [[sources/mdl2-trtllm-supported-models]]
   and [[sources/mdl2-llamacpp-arch-class-registry]] both show that the cache layout, page-block size
   and even the cache *class* are set by the attention implementation, not by the model.

Those three are §2, §4–§5 and §6 respectively. §1 sets up the families and §3 is the matrix.

**Method and scope.** 124 model records were classified into 14 architecture families (§1).
Each family's required kernel class was derived from the flop records in `data/flops/` and from
the architecture registries, then each model record's `notes` gained a `SERVICEABILITY JOIN`
block citing the engine records for its family. Nine new sources with the `mdl2-` prefix were
added for the registry reads. Nothing in `data/models/`, `data/engines/` or `schemas/` was
edited outside `data/models/` `notes` and `sources`, and no engine record's capability was
invented: where an engine record is silent, this document says so.

---

## 1. The fourteen families

Classification is per model record, driven by `attention_variant` and `architecture` rather
than by `family`, because the latter is a release-line label. Two corrections fell out of
doing it this way, both recorded in the affected records.

| family | n | what defines it | required kernel / ISA class |
|---|---|---|---|
| `dense-gqa` | 39 | grouped-query attention, `gqa_ratio` > 1 | causal attention + GEMM/GEMV. Baseline: no engine in this repo records it as unsupported. |
| `dense-mha` | 4 | `gqa_ratio` 1.0, no GQA at all | as above; the shape every paged-attention kernel was written for. |
| `mla` | 9 | multi-head latent attention (`kv_lora_rank` present) | **MLA decode kernel.** The cache is a shared low-rank latent + a decoupled RoPE key, not per-head K/V, so a GQA kernel cannot serve it. [[flops/mla-latent-attention]] |
| `sliding-window` | 16 | local/windowed attention on some or all layers | **local-attention decode kernel + per-layer-type cache management.** The two layer classes have opposite scaling — a local layer's decode cost is constant in context, a global layer's grows — so one kernel serves both regimes. [[flops/sliding-window-hybrid-attention]] |
| `hybrid-attention-ssm` | 12 | SSM / linear-attention layers interleaved with attention | **two kernel families plus a fixed-state allocator**: a chunked SSD or delta-rule kernel *and* an attention or MLA kernel. [[flops/ssm-recurrent-state-update]] [[flops/ssm-sequential-scan-tensor-core-mismatch]] |
| `ssm` | 1 | pure Mamba-2, no attention layers | chunked SSD scan only. `llm_arch_is_recurrent` in [[sources/mdl2-llamacpp-arch-class-registry]]. |
| `recurrent` | 1 | RWKV/WKV mixer | RNN-style WKV update; **no chunk-parallel reformulation exists**, so it is the worst case for the tensor-core mismatch. |
| `diffusion` | 1 | masked diffusion LM, `use_cache: false` | **no KV cache and no decode loop.** Needs multi-token parallel decode and a block-diffusion sampler. [[flops/diffusion-lm-denoising-steps]] |
| `encoder-decoder` | 2 | Whisper, cross-attention | **three distinct things in one graph**: a bidirectional encoder with no cache, a decoder self-attention KV that grows, and a decoder cross-attention KV that is *constant per request*. |
| `bi-encoder` | 11 | bidirectional encoder, no causal mask | one full-sequence forward + pooling. No cache, no incremental state. [[flops/embedding-batch-encoding]] |
| `cross-encoder` | 3 | query+passage scored jointly | N independent full forwards, no cache reuse, no early exit. [[flops/cross-encoder-reranking]] |
| `speech-encoder` | 3 | FastConformer / NeMo ASR | conv-augmented attention forward pass. **No engine record in this repo enumerates this class.** |
| `multimodal` | 17 | vision or audio tower + projector | encoder tower on a *separate schedule* from the decoder, plus mrope / DeepStack in the decoder where present. |
| `undisclosed` | 5 | closed API models (Claude, GPT-4.1, o3) | none derivable. The kernel question is **moot**, not unknown: no local engine serves these. |

**Two classification corrections, both recorded in the affected records.**

- [[models/glm-4-5]] and [[models/hunyuan-a13b]] were initially filed under `mla` on the
  strength of their MoE design. Their own `attention_variant` fields say otherwise —
  `GQA (standard) - NOT MLA despite the DeepSeek-derived MoE design` and `no MLA - kv_lora_rank
  explicitly null` — so both are reclassified `dense-gqa`. This matters: it removes the
  FlashMLA and FlashInfer MLA constraints from two checkpoints that do not need them.
- [[models/deepseek-r1-distill-qwen-32b]], [[models/qwen2-5-72b]] and
  [[models/phi-3-5-mini-instruct]] carry a `sliding_window` field that never binds (set to or
  above `max_position_embeddings`). They are served as plain attention and are marked
  window-inert; applying §4 to them would be wrong.

---

## 2. What each family's kernel requirement does to the engine

The point of this section is that the kernel requirement is not "can the engine load the
weights". Four families make an autoregressive serving engine the wrong *shape*.

### 2.1 MLA — the kernel decides the cache format

The strongest finding in this document. [[sources/kern-flashmla-github]] and
[[flops/flashmla-decode-fp8-fp4-kv]]: DeepSeek's own MLA decode kernel, at its 2026.09.30 release,

- requires NVIDIA **SM100/SM103** and CUDA ≥ 13.1,
- **removed Hopper support**, and removed V3, V3.2 and V4.0,
- supports **only FP8 and FP4 KV cache** — unquantized bfloat16 KV is explicitly not supported,
- and changed the cache layout: V4.1 stores **528 bytes/token** (512 of `float8_e4m3` over all
  512 dims — the 64 RoPE dims are quantized too, no bf16 part — plus 16 bytes of `float8_e8m0`
  scales, one per 32 e4m3 values), with an fp4 variant at **288 bytes/token**.

So the kernel choice determines the cache *format*, which determines what the engine must
store, which determines whether the model is servable at all on that part. Three
independent gates on top, all cited:

- **Page-block size.** [[gotchas/sm120-sparse-mla-block-size]] — on 8× RTX PRO 6000 (SM120) at TP8,
  vLLM computes compressed-layer KV page block size as `block_size // compress_ratio` = 32 for
  a ratio-2 layer, while FlashInfer's SM120 dispatch planner hard-codes `_PAGE_BLOCK_SIZE = 64`.
  There is **no global block size that works**: 64 gives pbs=32 and crashes; 128 gives
  `No common block size for 64 (FLASHINFER_MLA_SPARSE_DSV41: [128])`. Workaround: SM100
  datacenter Blackwell, or wait for a FlashInfer planner change.
- **Head size.** [[gotchas/flashinfer-rejects-large-head-dim]] — FlashInfer's template instantiations
  do not cover Gemma-4's `head_dim=256` / `global_head_dim=512`, and vLLM's backend selection
  validates head size at engine init, so the override is accepted on the CLI and only fails at
  startup. A *different* SM120 gap from the page-block one.
- **Feature gating on hardware.** [[sources/mdl2-trtllm-supported-models]] footnotes: chunked prefill
  for MLA only enables on SM90/SM100/SM103/SM107/SM120; KV cache reuse for MLA only on
  SM90/100/103/107/120/**121** and only in BF16/FP8 KV dtype. DeepSeek-V4 is Blackwell SM100+
  only; Kimi K3 is SM100 family only.

The engine-level consequence is stated in [[flops/mla-latent-attention]]: at 68.6 KiB/token MLA moves
the bottleneck off KV and onto weight streaming, so MLA-capable engines find their
paged-attention machinery largely idle and their GQA-era cache-pressure tuning irrelevant.

### 2.2 Hybrid attention-SSM — two kernel families, and the second one fights the hardware

[[flops/ssm-sequential-scan-tensor-core-mismatch]] is the load-bearing record: a selective-SSM step
is `h_t = a_t*h_{t-1} + b_t*x_t`, a multiplicative chain over time, which tensor cores cannot
express — they compute sums of independent products. So the scan must run sequentially, and
the published penalty is **2–8×** between the fused associative scan and the
matrix-multiplication-reformulated SSD for *identical mathematics*. The record's warning is the
one to carry: judge an SSM model's serving speed by measured tok/s on the target part, never by
its flop count, and expect the deficit to be worst on prefill-heavy long-prompt workloads.

**[[sources/mdl2-llamacpp-arch-class-registry]] is the engine agreeing with the arithmetic.** Its
`llm_arch_supports_sm_tensor()` returns **false** for 30 architectures, and that list is not
only the SSM ones — it includes `DEEPSEEK2`, `DEEPSEEK32`, `GEMMA3N`, `OLMO2`, `GROK`, `T5`,
`MINIMAX_M2/M3`, `GLM_DSA`. That is a mainstream engine's own dispatch table saying these
architectures have no matmul-heavy tensor-core path.

Its `llm_arch_is_hybrid()` returns true for exactly 18 entries — `JAMBA`, `FALCON_H1`,
`PLAMO2`, `GRANITE_HYBRID`, `LFM2`, `LFM2MOE`, `NEMOTRON_H`, `NEMOTRON_H_MOE`, `QWEN3NEXT`,
`KIMI_LINEAR`, `BAILINGMOE3`, `KIMI_K3`, `GLM5_NEXT`, `QWEN35`, `QWEN35MOE`, `QWEN4EXP`,
`DEEPSEEK4`, `MINIMAX_01` — which is **one shared hybrid code path** covering exactly the
families vLLM and SGLang each ship as separate rows. That is the structural reason hybrid
support lands in one engine release and not another: it is one code path to write, and engines
write it at different times.

### 2.3 Diffusion — the wrong shape entirely

[[flops/diffusion-lm-denoising-steps]]: each denoising step is a *full* forward pass over the
current sequence, so cost per emitted token is `S · 2N / Lg` for `S` steps and generation
length `Lg` — a function of two user-facing hyperparameters rather than a constant. Attention
is bidirectional, so there is no causal prefix to reuse; the model is dense, so there is no MoE
sparsity to amortise. Arithmetic intensity per step is prefill-like, so **part selection
inverts** (compute and tensor cores, not memory bandwidth), TTFT is excellent and inter-token
latency is terrible, and the weight stream is paid `S` times.

The engine evidence is structural rather than a row in a table. [[engines/vllm]] has exactly **one**
diffusion row in its entire registry — `DiffusionGemmaForBlockDiffusion` — and no LLaDA and no
Dream row ([[sources/mdl2-vllm-supported-models-2026-10]], confirming
[[sources/pap-arch-vllm-supported-models]]). [[engines/sglang]] by contrast gives diffusion its own top-level
docs section, separate from "Large Language Models", with its own launch flags
(`--dllm-algorithm LowConfidence|JointThreshold`, `--no-dllm-fdfo`) and **its own scheduling
primitive**: First-Done-First-Out, "each request leaves the batch as soon as its block is
resolved, instead of advancing in lockstep" ([[sources/mdl2-sglang-diffusion-language-models]]). A
scheduler concept that does not exist on the AR path. llama.cpp carries `DREAM`, `LLADA`,
`LLADA_MOE`, `RND1` in `llm_arch_is_diffusion` and adds two diffusion-only tensors,
`LLM_TENSOR_MASKED_EMBD_CENTROIDS` and `LLM_TENSOR_MASKED_EMBD_ORDERING`
([[sources/mdl2-llamacpp-arch-class-registry]]) — a different *state structure*, not just a sampler.

### 2.4 Encoder-decoder — two structurally different KV stores

[[models/whisper-large-v3]] already records this and the join sharpens it: an engine must
allocate (a) a bidirectional encoder with **no cache at all**, (b) decoder self-attention KV
that grows with generated tokens, and (c) decoder cross-attention K/V over the encoder frames
that is **constant per request** — 234 MiB for large-v3, and 3× larger than a full 448-token
decode's entire self-attention KV. Only (b) is sized by tokens. Concurrency is set by (c).

---

## 3. The engine × architecture matrix

**Legend.** `Y` = a record names this architecture family as supported. `Y*` = supported but the
record attaches a constraint, stated in the cell. `N` = a record explicitly excludes it, or the
model's own registration is absent from a registry that otherwise enumerates its neighbours.
`—` = the engine has no serving surface for this task class at all. `?` = **no record states
this**.

Citations are per cell. The three registries are [[sources/mdl2-vllm-supported-models-2026-10]] (vLLM),
[[sources/mdl2-trtllm-supported-models]] (TensorRT-LLM), and [[sources/mdl2-sglang-generative-models]] /
[[sources/mdl2-sglang-embedding-rerank-models]] / [[sources/mdl2-sglang-multimodal-language-models]] /
[[sources/mdl2-sglang-diffusion-language-models]] (SGLang's five pages).

### 3.1 Generation engines

| family | vllm | sglang | tensorrt-llm | llama.cpp | flashinfer | exllamav3 | mlx-lm | vllm-metal |
|---|---|---|---|---|---|---|---|---|
| `dense-gqa` | Y | Y | Y | Y | ? | Y | Y | ? |
| `dense-mha` | Y | Y | Y | Y | ? | Y | Y | ? |
| `mla` | Y | Y | Y* | Y* | ? | Y | ? | ? |
| `sliding-window` | Y | Y | Y* | Y | ? | Y | Y* | ? |
| `hybrid-attention-ssm` | Y | Y | partial | Y | ? | Y | Y* | ? |
| `ssm` | Y | Y | **N** | Y | ? | ? | ? | ? |
| `recurrent` | **N** | **N** | **N** | **N** | ? | ? | ? | ? |
| `diffusion` | N | partial | ? | Y | ? | Y | ? | ? |
| `encoder-decoder` | Y* | Y | Y* | Y | ? | ? | ? | ? |
| `multimodal` | Y | Y | partial | Y | ? | Y | ? | ? |

**`dense-gqa` / `dense-mha`.** Baseline — no engine record in `data/engines/` states either is
unsupported. vLLM has `LlamaForCausalLM`, `Qwen2`, `Qwen3`, `Mistral`, `Phi3` rows
(vllm); TRT-LLM the same rows (trtllm); SGLang's generative table names the families
(sglang); llama.cpp via `LLM_ARCH_LLAMA` ([[sources/pap-arch-llamacpp-arch-registry]]);
[[engines/exllamav3]] claims the broadest HF architecture coverage of any engine here.
[[engines/mlx-lm]] and [[engines/vllm-metal]] are Apple-Silicon paths whose records state a model list but no
architecture-class support statement, hence `?` rather than `Y`. The constraint that *does*
apply to both is model-shape, not class: [[gotchas/mgpu-tp-size-must-divide-attention-heads]] (vLLM and
SGLang) refuses to start unless TP divides the attention head count. And one hardware gate that
is not about the architecture class at all: [[gotchas/rdna-triton-paged-attn-decode-cliff]] — vLLM's
custom paged-attention kernel on RDNA is gated on `head_size == 128` **and** `block_size == 16`,
so a dense GQA model with `head_dim=256` falls to the Triton 2D fallback and decode falls 12.1
→ 4.2 tok/s between 518 and 32k context, with `kernel_paged_attention_2d` growing 28.3×.

**`mla`.** vLLM rows `DeepseekV2/V3/V32/V4` (vllm); TRT-LLM the same four plus
`GlmMoeDsaForCausalLM`, `Glm4MoeForCausalLM`, `KimiK3ForConditionalGeneration`,
`KimiLinearForCausalLM` (trtllm) — and its `DeepseekV3ForCausalLM` row is labelled
"DeepSeek-V3, Kimi-K2", the clearest evidence in the repo that MLA is one kernel class rather
than one model. SGLang names DeepSeek v1/v2/v3/R1 and Kimi K2 (sglang). llama.cpp reaches it
through `DEEPSEEK2`/`DEEPSEEK32`/`DEEPSEEK4`, all in its `supports_sm_tensor == false` list
(llamacpp). **All four carry the §2.1 constraints**: FlashMLA requires SM100/103 and FP8/FP4 KV
only ([[sources/kern-flashmla-github]]); [[gotchas/sm120-sparse-mla-block-size]] blocks SM120 outright;
[[gotchas/flashinfer-rejects-large-head-dim]] blocks head_size 256/512; TRT-LLM's MLA rows are gated on
SM90+ for chunked prefill and SM90/100/103/107/120/121 in BF16/FP8 for KV reuse, with DeepSeek-V4
Blackwell-only and Kimi K3 SM100-only. [[engines/ik-llama-cpp]] records MLA as a feature that landed
there first and was later upstreamed. `?` for flashinfer (a kernel library with no model
registry), mlx-lm and vllm-metal (no architecture-class statement).

**`sliding-window`.** vLLM has native rows for the specific SWA architectures —
`GptOssForCausalLM`, `Gemma3ForCausalLM`, `Gemma4ForCausalLM`, `Olmo3ForCausalLM`,
`GraniteMoeSWAForCausalLM`, `GraniteSWAForCausalLM` — and also lists sliding attention inside
its fallback envelope ("Attention types: full attention and/or sliding attention") (vllm).
SGLang names GPT-OSS with the mechanism in prose — "hybrid sliding-window-attention MoE model
... with per-layer-type RoPE (YARN on full-attention layers, default on sliding-attention
layers) and a softplus per-head attention gate" (sglang). **TRT-LLM is the informative cell**:
its per-architecture "Sliding Window Attention" column is *not* uniformly `Yes` — `Yes` for
`DeepseekV4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Unified…` and
`Step3p7ForCausalLM`, explicitly **`No` for `Qwen3NextForCausalLM`**, and `N/A` for every
DeepSeek-V3/V3.2, GLM, GPT-OSS, Llama-4, Qwen3.5, Kimi-K3 and Nemotron-H row (trtllm). llama.cpp
exposes SWA as first-class KV *keys* — `%s.attention.sliding_window` and
`%s.attention.sliding_window_pattern` are distinct per-architecture-class keys alongside SWA
width keys, so SWA changes the cache layout rather than adding an attention flag (llamacpp).
[[engines/exllamav3]] vendors Flash Linear Attention kernels under `exllamav3/vendor/fla` and is
broadest-in-coverage. [[engines/mlx-lm]] is `Y*`: its sliding-window layers use `RotatingKVCache`, which
had **no quantized implementation**, so `--kv-bits` passes a `hasattr` presence check and raises
`NotImplementedError: RotatingKVCache Quantization NYI` on the *first request* rather than at
startup; the fix is [[sources/mdl2-mlxlm-pr1584-rotating-quantized-kv]] and the commonly-recommended
monkeypatch silently drops the `max_kv_size` window bound, yielding an unbounded cache
([[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]]). The rest of the constraints are
in §4.

**`hybrid-attention-ssm`.** vLLM has native rows for all of them — `MambaForCausalLM`,
`Mamba2ForCausalLM`, `JambaForCausalLM`, `FalconMambaForCausalLM`, `FalconH1ForCausalLM`,
`NemotronHForCausalLM`, `GraniteMoeHybridForCausalLM`, `Qwen3NextForCausalLM`,
`KimiLinearForCausalLM`, `Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, `OlmoHybridForCausalLM`,
`KimiK3ForConditionalGeneration`, `BailingMoeV3ForCausalLM`, the `Qwen3_5` rows (vllm).
SGLang names them with the mechanism: Granite 4.0 micro as "hybrid Mamba-MoE ... combining
gated short convolutions with a small number of grouped query attention blocks", Kimi Linear as
"hybrid linear attention model ... Kimi Delta Attention (KDA) for up to 6× faster decoding and
75% KV cache reduction vs full attention", Nemotron Nano as "hybrid Mamba-Transformer
architecture combining attention and state-space models", Ling as "hybrid MoE model: per-layer
mix of DeepSeek-style MLA full attention and Qwen3 Gated-Delta-Net (GDN) linear attention",
LFM2 as "hybrid backbone", Mamba-Codestral as "Mamba2 ... holds no KV cache"; its multimodal
page carries the descriptor "hybrid KDA/MLA" (sglang). **TRT-LLM is partial by omission**: rows
exist for `NemotronHForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3_5MoeForCausalLM`,
`Qwen4ExpForCausalLM`, `KimiLinearForCausalLM`, `KimiK3ForConditionalGeneration` — and **no row
at all** for Mamba, Mamba2, Falcon-H1, LFM2 or Granite-MoE-Hybrid (trtllm). That is absence from
a registry, not an explicit `No`, which is why the cell is `partial`. Its key-models matrix is
also the only per-architecture feature record for this class: `Qwen3NextForCausalLM` has SWA
**No** and speculative decoding **No**; Kimi K3 has no MTP or EAGLE-3 head at all (DSpark only)
and needs Blackwell SM100 with a DEP16/TEP16 recipe split; MiniMax-M3's sparse-attention path
supports neither KV cache reuse nor MTP; DeepSeek-V4 and Step-3.7 are `Untested` for
disaggregated serving (trtllm). [[engines/exllamav3]] vendors the FLA chunked linear-attention prefill
kernels, which is the specific reason a hybrid lands early on consumer NVIDIA. llama.cpp's
18-entry hybrid class covers most of the family (llamacpp). [[engines/mlx-lm]] is `Y*` —
[[gotchas/eagle-prefix-cache-last-block-drop]] is not about mlx-lm but the analogous scheduler-alignment
trap is documented there for vLLM. The reproduced class-level bugs are listed in §5.

**`ssm`.** vLLM `MambaForCausalLM` + `Mamba2ForCausalLM` (vllm); SGLang names
`mistralai/Mamba-Codestral-7B-v0.1` with "holds no KV cache" (sglang); llama.cpp via
`LLM_ARCH_MAMBA2` and `llm_arch_is_recurrent` (llamacpp). **TRT-LLM `N` by omission**: no Mamba or
Mamba2 row anywhere in its architecture table (trtllm).

**`recurrent`.** The weakest column in the matrix. [[sources/pap-arch-llamacpp-arch-registry]] records
**no `LLM_ARCH_RWKV5` entry**; llama.cpp's own registry has `RWKV6`, `RWKV6QWEN2`, `RWKV7`,
`ARWKV7` in `llm_arch_is_recurrent`, so this checkpoint (`model_type 'rwkv5'`,
`model_version '5_2'`) is outside the family that engine dispatches (llamacpp). vLLM's
architecture table has no RWKV row of any kind (vllm); neither does SGLang's (sglang) nor
TRT-LLM's (trtllm). Every other engine cell is `?`. **No record in this repo states RWKV
support**, so §7 lists all nine.

**`diffusion`.** See §2.3. vLLM `N` (one DiffusionGemma row, no LLaDA, no Dream). SGLang
`partial`: a dedicated diffusion section exists with its own scheduler, but its four supported
families are LLaDA2.0 (mini, flash), SDAR/JetLM (8B-Chat dense, 30B-A3B-Chat MoE) and
DiffusionGemma — **[[models/llada-8b-instruct]] and Dream 7B are not among them** (sglang-diff).
llama.cpp `Y` (`DREAM`, `LLADA`, `LLADA_MOE`, `RND1` plus two diffusion-only tensors)
(llamacpp). [[engines/exllamav3]] `Y` via `DFlash`, listed among the features that landed there first.
TRT-LLM `?` — no diffusion *text* row, though it hosts a diffusion VisualGen pipeline for
image/video, which is a different task.

**`encoder-decoder`.** vLLM `Y*`: `WhisperForConditionalGeneration` is native (vllm) but
encoder-decoder is explicitly *not* a first-class class there — "For other model architectures
not natively supported, in particular for Encoder-Decoder models, we recommend following a
similar pattern by implementing support through the plugin system", and the only two plugin rows
are `BartForConditionalGeneration` and `Florence2ForConditionalGeneration` (vllm). Plus the
ROCm caveat: the enc-dec *feature* is unsupported on ROCm, and AMD GPUs are on the `avoid_for`
list for enc-dec models alongside multi-step scheduling, async output and Marlin
quantizations ([[engines/vllm]]). SGLang `Y` — `openai/whisper-large-v3` on the audio/transcriptions
route (sglang-mm). TRT-LLM `Y*` with the constraints stated outright: "Use the `TRTLLM` attention
backend for encoder-decoder models; tensor parallelism also requires attention head counts
divisible by the tensor parallel size. **Chunked prefill is not supported for the encoder
phase**, so the complete encoder input must fit in the iteration token budget", and "Whisper's
feature-driven audio encoder runs eagerly" — encoder CUDA graphs cover only BART/mBART/T5
(trtllm). llama.cpp `Y` via whisper.cpp / Whisperfile rather than the main engine
([[engines/llamafile]], [[engines/koboldcpp]], [[engines/lemonade]]).

**`multimodal`.** vLLM `Y` — a long native list including `InternVLChatModel`,
`LlavaOnevisionForConditionalGeneration`, `LlavaNextForConditionalGeneration`,
`MolmoForCausalLM`, `Molmo2ForConditionalGeneration`, `PaliGemmaForConditionalGeneration`,
`SmolVLMForConditionalGeneration`, `Glm4vForConditionalGeneration`, `Qwen2VL`/`Qwen3VL`,
`MiniCPMV`, `Mllama`, `Phi4MM`, `Gemma4`, `NemotronH_Nano_VL_V2`,
`VoxtralForConditionalGeneration`, `Qwen3OmniMoeThinkerForConditionalGeneration` (vllm). SGLang
`Y` — InternLM2-VL, LLaVA-OneVision, LLaVA-NeXT(-Video), Ling-3.0-flash-VL, LFM2.5-VL,
MiniCPM-V-2, GLM-4.5V, GLM-OCR, Qwen3-VL 235B/30B, Qwen3-Omni, Kimi-VL, MiMo-VL, Step3-VL-10B,
dots.ocr, PaddleOCR-VL, Cosmos3, deepseek-vl2, Janus-Pro-7B, NVILA-8B (sglang-mm). TRT-LLM
`partial` with per-row modality strings and per-row `Untested` cells; its encoder work is
optimised by a documented mechanism — a **Multimodal Encoder Side Stream** prefetching on a
separate CUDA stream plus a **Multimodal Embeddings Cache** — available for only seven
architectures (`Gemma4`, `Gemma4Unified`, `Mistral3`, `Qwen3VL`, `Qwen3VLMoe`, `Qwen3_5`,
`Qwen3_5Moe`) (trtllm). Hard tower dependencies, cited: `Gemma3nForConditionalGeneration`
"depends on `timm>=1.0.17` to make use of its MobileNet-v5 vision backbone" and "is only
supported on V1 due to shared KV caching" (vllm); MiniCPM-V 4.6 "requires `transformers>=5.7.0`
... was upstreamed into transformers as a native model type (`minicpmv4_6`) and the checkpoint
ships no remote code (`auto_map`) to fall back on" (trtllm). The class's own memory knob is
vLLM's `--language-model-only`, which sets all supported multimodal modalities to 0 "so that
their multimodal modules will not be loaded to free up more GPU memory for KV cache" for
Llama-4, Step3, Mistral-3 and Qwen-3.5 (vllm). llama.cpp `Y` — but
[[engines/llama-cpp-embedding-server]] states it cannot serve "multimodal encoders requiring a projector
graph" on the embedding route.

### 3.2 Retrieval and pooling engines

`—` below means the engine has no serving surface for that task class. Note that the
generation engines in §3.1 are **all** `—` for cross-encoder and bi-encoder: none of the seven
generation engine records mentions pooling, reranking, embeddings or ColBERT at all, which is
itself the finding — retrieval serving in this repo is a separate set of engines.

| family | vllm-pooling-models | hf-text-embeddings-inference | llama-cpp-embedding-server | sglang (pooling pages) | triton-inference-server | sentence-transformers | colbert-reference |
|---|---|---|---|---|---|---|---|
| `bi-encoder` | Y | Y | Y* | Y | Y | Y | Y |
| `cross-encoder` | Y | Y | Y* | Y* | Y | Y | `—` |
| late-interaction (ColBERT) | **Y** | **N** | **N** | **N** | ? | Y | Y |

**`bi-encoder`.** vLLM pooling has the broadest table in the repo — `BertModel`,
`BgeM3EmbeddingModel`, `GteModel`, `GteNewModel`, `JinaEmbeddingsV5Model`,
`LlamaBidirectionalModel`, `LlamaModel`/`MistralModel`, `ModernBertModel`, `NomicBertModel`,
`Qwen2Model`, `Qwen3Model`, `RobertaModel`, `XLMRobertaModel`, plus `--convert` for any
generative model, with the Transformers modeling backend covering encoder-only architectures
directly ([[engines/vllm-pooling-models]]). TEI is the purpose-built reference and lists BERT,
CamemBERT, XLM-RoBERTa, NomicBERT, JinaBERT, MPNet, ModernBERT plus RoPE decoder backbones
Mistral, Alibaba GTE, Qwen2, Qwen3, Gemma3 ([[engines/hf-text-embeddings-inference]]). SGLang's
embedding page: "Native encoder embedding architectures and `google/embeddinggemma-300m` are
detected automatically. Decoder-style embedding models require `--is-embedding`"; on CUDA it
"automatically uses breakable CUDA graph (BCG) for its full encoder prefill and **disables**
incompatible radix-cache and chunked-prefill behavior" (sglang-embed).
[[engines/llama-cpp-embedding-server]] `Y*`: GGUF embedding in the same binary as LLMs is a real
operational advantage, but its record states the caveats — pooling type must match the
checkpoint's training configuration and is not enforced, and the rerank endpoint is "newer and
less exercised than the embeddings route".

**`cross-encoder`.** vLLM's row list is explicit — `BertForSequenceClassification`,
`GemmaForSequenceClassification` (BAAI/bge-reranker-v2-gemma),
`GteNewForSequenceClassification`, `LlamaBidirectionalForSequenceClassification`,
`ModernBertForSequenceClassification`, `Qwen2ForSequenceClassification`
(mixedbread-ai/mxbai-rerank-base-v2), `Qwen3ForSequenceClassification` (Qwen/Qwen3-Reranker-0.6B),
`RobertaForSequenceClassification`, `XLMRobertaForSequenceClassification`
(BAAI/bge-reranker-v2-m3) — with per-model score templates shipped for the rerankers that need a
specific prompt format ([[engines/vllm-pooling-models]]). TEI has reranking first-class since v0.4.0 for
CamemBERT, RoBERTa, XLM-RoBERTa and GTE sequence-classification cross-encoders at `/rerank`
([[engines/hf-text-embeddings-inference]]). SGLang `Y*`: it splits the class in two and says so —
"**Cross-encoder rerank models**: run with `--is-embedding` (embedding runner)" vs "**Decoder-only
rerank models**: run **without** `--is-embedding` and use next-token logprob scoring (yes/no)" —
and its `BGE-reranker-v2-m3` row carries a hard kernel constraint: "**Currently only support
`attention-backend` `triton` and `torch_native`**", with the example launch passing
`--attention-backend triton`, `--disable-radix-cache` and `--chunked-prefill-size -1`
(sglang-embed). [[engines/triton-inference-server]] `Y` with a batching trap: "Reranker batching via the
sequence batcher requires modelling the query-document pair as state." [[engines/colbert-reference]] is
`—` for cross-encoders: it is a late-interaction engine, a different kernel class.

**Late interaction — the class boundary that decides engine choice.** A cross-encoder is not
ColBERT. [[engines/colbert-reference]]: late interaction encodes each passage into a *matrix* of
token-level embeddings and scores with MaxSim, "so query-time cost scales with corpus size and is
bandwidth-bound on the index rather than compute-bound in the encoder", and ColBERTv2's abstract
states late interaction "inflates the space footprint of these models by an order of magnitude"
relative to single-vector models. Only **vLLM** serves it as a first-class endpoint —
`token_embed` pooling plus MaxSim scoring, recorded as "the only first-class ColBERT-style
scoring in a mainstream serving engine" ([[engines/vllm-pooling-models]]). **TEI `N`**: "the pooling
options are cls, mean, splade and last-token; there is no token-wise output mode, so a ColBERT
model cannot be served correctly here" ([[engines/hf-text-embeddings-inference]]).
**llama-cpp-embedding-server `N`**: "`--pooling none` gives raw per-token output, which is a
building block for ColBERT but not a served ColBERT". **SGLang `N`**: no ColBERT, token-embed or
late-interaction endpoint appears on any of its five pooling pages (sglang-embed).
[[engines/sentence-transformers]] `Y` for *producing* the token-level embeddings — with the explicit
limit that "the library produces the token-level embeddings" and production serving means
wrapping it yourself ([[engines/sentence-transformers]]).

### 3.3 The speech-encoder column

`[[models/canary-1b-flash]]`, `[[models/parakeet-ctc-1-1b]]`,
`[[models/parakeet-tdt-0-6b-v2]]` — FastConformer / NeMo ASR.

**Every cell is `?`.** No engine record in `data/engines/` enumerates a FastConformer or NeMo ASR
architecture. The nearest evidence is indirect and negative: [[engines/onnxruntime-genai]]'s official
architecture list is "language + vision + speech" and names Whisper but not NeMo; the speech
engines recorded here ([[engines/koboldcpp]], [[engines/llamafile]], [[engines/lemonade]], [[engines/openvino-genai]],
[[engines/niche-h2ogpt]]) all reach Whisper specifically. **This is `unknown`, not `unsupported`** — no
record asserts either way, and per the repo's own rule a wrong confident cell is worse than an
open question.

---

## 4. The KV overpay table: sliding-window models

This is the "4× overpay" class, generalised.

**The mechanism.** `kv_cache_bytes_per_token` in the schema is derived as
`num_layers × num_kv_heads × head_dim × 2 × 2`. For a sliding-window model that is the
**naive** figure — what an engine that ignores the local layers allocates. An engine with
local-attention support allocates

> `bytes/token(c) = G·k_layer·min(c, max_ctx) + S·k_layer·min(c, w)`

where `G` = global-attention layers, `S` = sliding layers, `w` = window, `c` = current context,
`k_layer = num_kv_heads · head_dim · 2 · 2`.

The gap is worst at long context and shrinks as `c → 0`, and its asymptote is set entirely by
`G/L`: a 1:1 model asymptotes to exactly 2×, a 5:1 model to `(G·k + S·k·w/c)/(G·k + S·k·w/c)` at
max context. All layer counts, windows and head geometries below are the model records' own
fields; the arithmetic is shown so it can be checked.

| model | layers G+S | `w` | max ctx | naive B/tok | effective B/tok @ max ctx | **overpay @ max ctx** | overpay @ 32k | overpay @ 8k |
|---|---|---|---|---|---|---|---|---|
| [[models/gemma-3-27b]] | 10+52 | 1024 | 131,072 | 507,904 | 85,344 | **5.96×** | 5.33× | 3.76× |
| [[models/gemma-3-12b-it]] | 8+40 | 1024 | 131,072 | 393,216 | 68,096 | **5.77×** | 5.19× | 3.69× |
| [[models/gemma-3-4b-it]] | 6+28 | 1024 | 131,072 | 139,264 | 25,472 | **5.47×** | 4.95× | 3.58× |
| [[models/gemma-3n-e2b]] | 6+24 | 512 | 32,768 | 61,440 | 13,056 | **4.71×** | 4.71× | 4.00× |
| [[models/step-3-5-flash]] | 12+36 | 512 | 262,144 | 196,608 | 49,440 | **3.98×** | 3.82× | 3.37× |
| [[models/olmo-3-7b]] | 8+24 | 4096 | 65,536 | 524,288 | 155,648 | **3.37×** | 2.91× | 1.60× |
| [[models/phi-3-mini-4k-instruct]] | 32 uniform | 2047 | 4,096 | 393,216 | 196,512 | **2.00×** | n/a | n/a |
| [[models/gpt-oss-20b]] | 12+12 | 128 | 131,072 | 49,152 | 24,600 | **2.00×** | 1.99× | 1.97× |
| [[models/gpt-oss-120b]] | 18+18 | 128 | 131,072 | 73,728 | 36,900 | **2.00×** | 1.99× | 1.97× |
| [[models/gemma-2-2b]] | 4+22 | 4096 | 8,192 | 106,496 | 61,440 | **1.73×** | n/a | 1.73× |
| [[models/gemma-2-9b]] | 7+35 | 4096 | 8,192 | 344,064 | 200,704 | **1.71×** | n/a | 1.71× |

**Which models cost most under a naive engine, in absolute bytes** — because the ratio matters
less than the GiB:

| model | naive KV @ max ctx | effective KV @ max ctx | bytes wasted per sequence |
|---|---|---|---|
| [[models/gemma-3-27b]] | 62.00 GiB | 10.41 GiB | **51.6 GiB** |
| [[models/gemma-3-12b-it]] | 48.00 GiB | 8.31 GiB | **39.7 GiB** |
| [[models/step-3-5-flash]] | 48.00 GiB | 12.07 GiB | **35.9 GiB** |
| [[models/olmo-3-7b]] | 32.00 GiB | 9.50 GiB | **22.5 GiB** |
| [[models/gemma-3-4b-it]] | 17.00 GiB | 3.11 GiB | **13.9 GiB** |
| [[models/gemma-3n-e2b]] | 1.88 GiB | 0.40 GiB | **1.5 GiB** |

**Four things this table shows that a per-token number hides.**

1. **Gemma 3 is the worst case in the repo, not Olmo 3.** Olmo 3 has the repo's highest
   KV/token (512 KiB, full MHA) and gets a 3.37× overpay. Gemma 3 27B has a *lower* KV/token
   (496 KiB) and gets **5.96×** — because 52 of its 62 layers are windowed and its `head_dim`
   is 128 with 16 KV heads, so the wasted layers are wide. The overpay is a function of the
   windowed-layer fraction and the per-layer width together, not of KV/token.
2. **Gemma 2 is nearly a non-case.** At `w=4096` against an 8,192-token context the window is
   half the context, so the overpay is 1.71–1.73× and constant. The 5:1 Gemma-2 pattern was
   designed when 4096 *was* long context; it is the weakest SWA design in this repo now.
3. **GPT-OSS's 2.00× is a ceiling, not a saving.** A 1:1 alternating pattern with `w=128` puts
   exactly half the layers at a bounded cost, so the asymptote is 2× and no context length
   improves it. The tiny window is what makes the asymptote *reachable*, not the factor small.
4. **Phi-3 mini is the degenerate case.** A uniform `w=2047` against a 4,096-token context
   caps the saving at 2×, and the window *was* the design rather than an optimisation.

**Window-inert models — do not apply this table.** [[models/deepseek-r1-distill-qwen-32b]]
(`sliding_window` = 131072 = `max_position_embeddings`), [[models/qwen2-5-72b]]
(`sliding_window` 131072 > `max_position_embeddings`), [[models/phi-4-mini-instruct]] and
[[models/phi-4-mini-reasoning]] (262144 vs 131072), [[models/phi-3-5-mini-instruct]] (262144).
Each carries the caveat in its own `notes`.

**Not computable from the schema, and flagged rather than estimated.** [[models/gemma-4-12b]],
[[models/gemma-4-26b-a4b]] and [[models/gemma-4-31b]] all have **separate global head geometry**
(`global_head_dim` 512 with 1 / 2 / 4 global KV heads against local `head_dim` 256 with 8 / 8 /
16), so the local and global layers have different per-token widths and
`kv_cache_bytes_per_token` is **null** in all three. The naive formula has no single answer for
them. The overpay factor is bounded by the local:global layer ratio, but the byte total is not
derivable from the fields the schema carries — which is itself an argument for the
`kv_cache_bytes_per_token_basis` field requested in `docs/09-model-memory-envelope.md` §1.3.

**What an engine actually does when it lacks local-attention support.** It does not fail; it
allocates. That is what makes this table dangerous rather than merely interesting, and it is why
[[models/step-3-5-flash]]'s own `notes` say outright that its 192 KiB figure "is the WORST CASE and
overstates steady-state badly … A reader sizing a server should use that, not the 192 KiB figure."

**Engine-side constraints that make the "supported" cell conditional, not binary:**

- [[sources/mdl2-trtllm-supported-models]]'s per-architecture SWA column: `No` for
  `Qwen3NextForCausalLM`, `Yes` for Gemma 4 and Step-3.7, `N/A` for ten rows. An engine can
  support a model and still not support its window.
- [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] (severity **blocker**): on B200, a
  50-of-60-layer SWA Gemma-4 checkpoint with `--kv-cache-dtype fp8_e4m3` and MTP speculative
  decoding (k=7) produces fluent output containing sporadic wrong tokens and short repetition
  loops once `seq_len` passes the 1024-token window. Two independent defects: flashinfer's
  `trtllm_batch_decode_with_kv_cache` mis-computes sliding-window attention for BF16-Q +
  FP8-KV at certain seq_len/page alignments, and vLLM's `FlashInferImpl.forward` applies
  `bmm1_scale *= q_scale * k_scale` unconditionally even when the flag keeps Q in BF16 so
  nothing was quantized. **This is the cleanest statement in the repo that quantized KV and
  speculative decoding are not independently safe on an SWA model.** Attribution matrix from
  the reporter: fp8-Q + spec clean; bf16-Q + no spec clean; bf16-Q + spec corrupted under both
  `-O3` cudagraphs and `--enforce-eager`.
- [[gotchas/fa4-num-splits-ignored-sm90]]: vLLM 0.30.0's `vllm_flash_attn` silently ignores
  `num_splits` on SM90, costing 48% per-token decode at batch 1 and 36% at batch 4 for
  Gemma-4 26B-A4B-it at ~5.9k context. Worst on global layers (1 query token, 2 KV heads → 2 of
  an H100's 132 SMs with work). `TRITON_ATTN` recovers ~70%.
- [[gotchas/rdna-triton-paged-attn-decode-cliff]]: on RDNA, vLLM's custom paged-attention kernel is
  gated on `head_size == 128` **and** `block_size == 16`; anything else falls to Triton 2D.
- [[gotchas/eagle-prefix-cache-last-block-drop]]: vLLM issue #53786 asks for fine-grained prefix hits
  for sliding-window groups, and hybrid reconciliation already caps the combined hit at the
  shorter boundary.
- [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]]: on Apple Silicon the SWA class
  is a cache-*class* problem, not a kernel problem — `RotatingKVCache` had no quantized
  implementation.

---

## 5. The hybrid overpay table — and the state that replaces the cache

The same arithmetic for models where *most* layers hold no KV at all. Here the naive figure is
`num_layers × per-layer-bytes`, and the overpay is far larger because the discarded fraction is
far larger.

| model | attn layers / total | naive B/tok | recorded B/tok | overpay | resident KV GiB @ max ctx (naive → real) | fixed state per sequence |
|---|---|---|---|---|---|---|
| [[models/nemotron-h-8b-base-8k]] | 4 / 52 | 212,992 | 16,384 | **13.00×** | 1.625 → 0.125 | **96 MiB** |
| [[models/granite-4-0-h-tiny]] | 4 / 40 | 81,920 | 8,192 | **10.00×** | 10.000 → 1.000 | **27 MiB** |
| [[models/minimax-m1]] | 10 / 80 | 327,680 | 40,960 | **8.00×** | 3125.0 → 390.6 | — |
| [[models/qwen3-5-35b-a3b]] | 10 / 40 | 81,920 | 20,480 | 4.00× | 20.000 → 5.000 | — |
| [[models/qwen3-8-flash-next]] | 12 / 48 | 98,304 | 24,576 | 4.00× | 24.000 → 6.000 | — |
| [[models/minicpm-v-4-6]] | 6 / 24 | 49,152 | 12,288 | 4.00× | 12.000 → 3.000 | — |
| [[models/kimi-k3]] | 24 / 93 | 107,136 | 27,648 | 3.88× | 104.6 → 27.0 | — |
| [[models/kimi-linear-48b-a3b]] | 7 / 27 | 31,104 | 8,064 | 3.86× | 30.4 → 7.9 | — |
| [[models/lfm2-1-2b]] | 6 / 16 | 32,768 | 12,288 | 2.67× | 3.9 → 1.5 | 3-timestep conv state |
| [[models/ling-3-0-flash]] | 20 / 42 | 48,384 | 23,040 | 2.10× | 11.8 → 5.6 | — |
| [[models/falcon-h1-7b-base]] | **44 / 44** | 45,056 | 45,056 | **1.00×** | 11.000 → 11.000 | **66 MiB** |

**Falcon-H1 is the row that matters.** Its attention and Mamba-2 mixers run **in parallel on
every one of its 44 layers**, so the KV cache does not shrink at all — the overpay is 1.00× — and
the 66 MiB of recurrent state is still paid on top. It gets the worst of both: full KV cost
*and* a fixed per-sequence tax, for a 2.9× reduction against a comparable dense GQA model where
the interleaved hybrids get 10–13×. This is a design choice, not a serving bug, but it is the
single most counter-intuitive number in this document and the one most likely to be quoted
wrongly.

**The state that replaces the cache is not free, and it scales the other way.**
[[flops/ssm-recurrent-state-update]] states the corollary precisely: "The corollary nobody quotes: the
state is NOT free, it is a fixed per-sequence tax that stops amortising as batch grows, so at
very large batch a pure-attention model with paged KV can hold more total tokens than a hybrid
with 96 MiB/seq of undividable state." Concretely, at batch 256 on Nemotron-H-8B you pay
256 × 96 MiB = 24 GiB of undividable state whether the sequences are 1k or 128k tokens — the
opposite trade from KV cache, which grows with context. And for a pure SSM it is the *entire*
per-sequence state: [[models/mamba-codestral-7b]]'s 128 MiB against zero KV. Its
`notes` already put it well: "not zero memory, but a fixed cost that scales with concurrency and
not with context." No `kv_cache_bytes_per_token`-based planner sees any of it.

**Two more numbers from these records.** [[models/minimax-m1]] carries an arithmetic error that
is corrected in its `notes` as of this document: it previously claimed 654,240 B/token naive
and a 16× reduction; the correct figure is 327,680 B/token and **8.00×**, matching the 10-of-80
full-attention layer count. The verified `kv_cache_bytes_per_token` of 40,960 was never
affected. [[models/kimi-k3]] is the inversion worth stating: **2.78T total parameters, 104B
active, yet KV per token (27 KiB) is *lower* than a dense 8B model's and less than half of
DeepSeek-V3's 68.6 KiB.**

**Reproduced class-level bugs, all cited, all architecture-specific:**

- [[gotchas/eagle-prefix-cache-last-block-drop]] — vLLM's EAGLE `drop_eagle_block` falls back to marking
  *all* KV groups as eagle when the model has no `is_eagle_group` annotation. On dense models the
  alignment unit is a 16-token block; on a hybrid GDN/Mamba layout it is **1,648 tokens**, so the
  same protection costs ~100× more cache hit — 97.5% → 86.9–91.7%, and 30–40% batch throughput
  on prefix-reusing workloads. The reporter explicitly says other hybrid layouts must be
  evaluated by their own alignment; the 100× is layout-specific.
- [[gotchas/gated-deltanet-decode-collapse-long-context]] — llama.cpp decode collapses on hybrid
  gated-deltanet GGUF models at high KV position: 33 t/s at 68K → 1.4 t/s at 91K with the model
  fully resident, surviving `--n-gpu-layers 999`, both q4_0 and q5_0 KV, and FA on or off. Only
  `-c 73728` with q8_0 KV and FA off held 33.1 t/s. Fast prefill (~1300 t/s) with collapsed decode
  rules out memory-capacity and offload. Not root-caused.
- [[gotchas/sycl-hybrid-linear-attention-arch-crash-intel-arc]] — llama.cpp's SYCL path has a regression
  **specific to hybrid linear-attention graphs**: `qwen3next`/`qwen3_35`-architecture models
  crash in `ggml_sycl_op_mul_mat` or return empty output on 2× Arc Pro B60, bisected to builds
  between b8477 (works) and b9479 (fails), while dense-attention models on the same cards are
  unaffected. "Battlemage works" does not generalise across model families.
- [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — adjacent but worth naming: the fp8 KV
  path for a sparse-attention MoE goes through a plain-Python advanced-index scatter that is not
  CUDA-graph-safe, so on 8× A800 (SM80, no native fp8) `--kv-cache-dtype fp8_e5m2` plus any
  CUDA-graph mode produces pure garbage while `--enforce-eager` produces coherent output at
  5.2 tok/s. The bf16 path uses a fused graph-safe op. That is a cache-*write* path bug in a
  model whose attention is sparse-indexed, and it is a different failure mode from §2.1's
  cache-*format* issue.
- [[gotchas/pd-packed-kv-layout-change-silently-breaks-external-caches]] — the external-cache
  counterpart: vLLM PR #44455 changed the KV layout from `[num_blocks, 2, block_size,
  num_heads, head_size]` to a packed `[num_blocks, num_heads, block_size, 2*head_size]`, and
  LMCache's connectors encode that assumption structurally rather than as a validated field. A
  request served from the CPU tier returns **different output than the same request computed
  from scratch, with no exception and no warning**. Nothing in the cache key records which
  layout produced the bytes.

---

## 6. Kernel choice constrains cache format — the findings, collected

Four independent mechanisms, in increasing order of how much they break a deployment:

| # | mechanism | kernel | what it does to the cache | record |
|---|---|---|---|---|
| 1 | KV *precision* is set by the kernel | FlashMLA | FP8/FP4 only; bf16 KV explicitly unsupported. V4.1 layout: 528 B/token (512 e4m3 + 16 e8m0 scales), fp4 variant 288 B/token. The 64 RoPE dims are quantized too — no bf16 part. | [[flops/flashmla-decode-fp8-fp4-kv]] [[sources/kern-flashmla-github]] |
| 2 | KV *page-block size* is set by the kernel | FlashInfer sparse MLA | Planner hard-codes `_PAGE_BLOCK_SIZE = 64`; vLLM computes `block_size // compress_ratio`. No global block size satisfies both → model will not start on SM120. | [[gotchas/sm120-sparse-mla-block-size]] |
| 3 | KV *class* is set by the layer type | mlx-lm | Sliding-window layers use `RotatingKVCache`, a different class from the full-attention `QuantizedKVCache`, and only the latter had a quantized implementation. A feature that exists for one class raises `NotImplementedError` for the other — at first request, not at startup. | [[sources/mdl2-mlxlm-pr1584-rotating-quantized-kv]] [[gotchas/mlx-kv-bits-crashes-on-first-request-for-hybrid-attention]] |
| 4 | KV *layout* is set by the engine version | vLLM + any external KV tier | Packed vs unpacked layout change silently invalidates every block written under the old layout. | [[gotchas/pd-packed-kv-layout-change-silently-breaks-external-caches]] |

Two more that constrain the cache write path rather than its format:
[[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] (a non-graph-safe fp8 KV *insert* combined
with double-applied scales, corrupting SWA output silently) and
[[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] (the same shape on a sparse-indexed MoE).

**The generalisation, and it is the practical point of this document:** the KV cache is not a
function of the model alone. It is a function of (model geometry) × (attention implementation)
× (kernel version) × (engine version), and three of those four move on a schedule you do not
control. Any capacity plan that computes `kv_cache_bytes_per_token × context` and stops is
planning against one point in a four-dimensional space.

---

## 7. Every cell marked `unknown`, with the reason

Counted per cell, in §3.1 and §3.2.

### 7.1 Why a whole column is unknown

- **[[engines/flashinfer]] (all 14 families).** It is a kernel library, not a serving engine: its record
  states "Not an engine: you cannot serve a model with it alone" and `scheduling: none`. It has
  no model registry, so "which architectures does it support" has no meaning — it supports a
  *kernel*, and the model set is whatever the calling engine loads. Its relevant facts are
  architectural (§2.1, §4), not family-level.
- **[[engines/vllm-metal]] (all 14 families).** Its record states "Model coverage is a growing set with
  a separate supported-models matrix, so an architecture not on that list is unsupported" — the
  matrix itself was not read into a record. Honest cell: `?`.
- **[[engines/mlx-lm]] (10 of 14 families).** Same reason: the record names prompt caching, rotating KV
  and `--kv-bits` but enumerates no architecture list.
- **[[engines/exllamav2]] (all families).** No architecture statement of any kind in its record, and the
  project is archived. Its [[engines/exllamav3]] successor claims "very broad HF architecture coverage
  including multimodal (DeepSeek V4, GLM 4.7/5.3 Flash, Qwen 3.5, Gemma 4, MiMo)" and vendors
  FLA kernels, but names no `ssm`, `diffusion`, `encoder-decoder` or `recurrent` row, and records
  two explicit exclusions: Gemma 4 E2B/E4B unsupported, and MiMo-V2.6-Flash without audio.

### 7.2 Unknowns inside columns that are otherwise populated

- **`recurrent` × 9 engines.** The only family with no positive statement anywhere.
  [[sources/pap-arch-llamacpp-arch-registry]] records no `LLM_ARCH_RWKV5` entry while listing RWKV6/7;
  vLLM, SGLang and TRT-LLM have no RWKV row at all. The checkpoint's own `notes` record that
  the transformers port is "for experimentation and demo use, not the production path". **No
  record in this repo states RWKV support.**
- **`speech-encoder` × all engines.** FastConformer/NeMo ASR is enumerated by no engine record.
  [[engines/onnxruntime-genai]]'s architecture list is "language + vision + speech" naming Whisper but
  not NeMo. `unknown`, not `unsupported` — no record asserts either way.
- **`cross-encoder` × [[engines/colbert-reference]]** — not unknown but structurally out of scope: it is
  a late-interaction engine, a different kernel class.
- **`triton-inference-server` × late interaction.** Its record covers bi-encoders, cross-encoder
  rerankers and SPLADE via ONNX, and states "ColBERT indexes are not comparable to single-vector
  indexes" (in [[engines/niche-colbert]]), but the triton record itself makes no ColBERT claim.
- **`diffusion` × TRT-LLM.** The engine hosts a diffusion *VisualGen* pipeline for image/video,
  but its architecture table has no diffusion *text* row. Different task class; `?` rather than
  conflating them.

### 7.3 Unknowns forced by missing model data

- **`kv_cache_bytes_per_token` is null** for [[models/gemma-4-12b]], [[models/gemma-4-26b-a4b]]
  and [[models/gemma-4-31b]] because of separate global head geometry, so their overpay is not
  computable from the schema (§4).
- **`params_b` is null** for [[models/rwkv5-eagle-7b]] — no safetensors metadata exists on the
  Hub, so the "7B" in the name is a name, not a measurement.
- **`undisclosed` family (5 records).** [[models/claude-opus-5-5]], [[models/claude-sonnet-5-5]],
  [[models/claude-haiku-4-5]], [[models/gpt-4-1]], [[models/o3]]. Every cell `?` — and the reason
  is stronger than silence: no open-source engine in `data/engines/` serves a closed API model at
  all. The kernel question is moot rather than unknown, because no KV cache, block manager,
  windowed layer or MLA latent runs on the operator's hardware.

---

## 8. What this document does not close

1. **No measured compatibility data.** Every cell here is a documented capability or a documented
   absence. `docs/10-testing-methodology.md` exists; no test in this document was run. A cell
   that says `Y` means "a record names it", not "it was measured at N tok/s".
2. **Per-cell citations are to registries, not to binaries.** The three registries were read on
   2026-10-03 and both [[engines/vllm]]'s and [[sources/mdl2-trtllm-supported-models]]'s own notes warn the
   tables change as the engines evolve. Treat per-cell support as of that date.
3. **The `?` cells are worth closing in a specific order**, because each blocks a different
   decision: `recurrent` (nothing here can serve the checkpoint on any recorded engine),
   `speech-encoder` (no FastConformer ASR server is recorded at all), `hybrid × TRT-LLM` (the
   partial cell is absence-from-registry, which is the weakest form of evidence in this
   document), and `vllm-metal`/`mlx-lm` (both point at a supported-models matrix that exists and
   was not transcribed).
4. **The schema fields §4 would need** are the ones already requested in
   `docs/09-model-memory-envelope.md` §1.3 and repeated here because this document needs them
   independently: `kv_cache_bytes_per_token_basis` (to distinguish `gqa` / `mha` /
   `mla-latent` / `hybrid-subset` / `sliding-window-bounded` / `none-fixed-state`), and
   `kv_caching_layer_count` + `state_bytes_per_sequence` for hybrids, since §5's fixed-state
   column is invisible to any consumer sizing memory from `kv_cache_bytes_per_token` alone. A
   `sliding_window_bytes` field carrying `G`, `S` and `w` would make the whole of §4 derivable
   rather than hand-computed.