# 03 — Engine selection: which engine for which hardware

<!-- generated against a snapshot of 1,087 records (40 engines, 74 accelerators). data/ grew to 64 engines / 1,281 records while this was written; the 24 later `engines/niche-*` records are parent/child relationships to records already in the table below rather than new engines, and are NOT included — see §4. Counts were a snapshot; data/ is written concurrently. -->

This document is drawn entirely from the `supported_backends`, `hardware_caveats`,
`best_for` and `avoid_for` fields already populated on all 40 engine records. It leads
with the **negative space** — which engines are CUDA-only, which break on ROCm, which
exclude specific architectures — because in this repo the exclusions are better evidenced
than the inclusions and they are what actually bites.

---

## 1. Negative space first: who cannot run what

### 1.1 CUDA-only, no exceptions

These records list `supported_backends` with **no** ROCm, Metal, Vulkan, CPU or NPU entry
at all. There is no path, not a slow path:

| Engine | Backends | Cite |
|---|---|---|
| TensorRT-LLM | cuda | [[engines/tensorrt-llm]] |
| ExLlamaV2 | cuda | [[engines/exllamav2]] |
| ExLlamaV3 | cuda + CPU (AVX512/AVX2) | [[engines/exllamav3]] |
| FasterTransformer | cuda | [[engines/fastertransformer]] |
| FlashInfer | cuda | [[engines/flashinfer]] |
| TinyChat (AWQ) | cuda | [[engines/tinychat-awq]] |
| ColBERT reference | cuda + cpu | [[engines/colbert-reference]] |
| SpinQuant harness | cuda | [[engines/spinquant-research-harness]] |
| Mistral.rs | cuda, metal, cpu | [[engines/mistral-rs]] |

TensorRT-LLM's record states the constraint as "the defining constraint of the engine":
"CUDA only - there is no ROCm, Metal, Vulkan or CPU path." Its `avoid_for` says "Any
non-NVIDIA accelerator."

ExLlamaV2's is equally absolute: "There is no CPU, ROCm, or Metal path." Its
`avoid_for` reads "AMD, Intel, Apple, or CPU-only hardware of any kind."

FlashInfer adds a **compute-capability floor** rather than just a vendor floor: "CUDA-only
and sm75+ only - it is explicitly unsupported below compute capability 7.5, which excludes
**Volta and older**." SGLang inherits this: "FlashInfer is the default attention backend and
only supports sm75 and above."

**Consequence:** FlashInfer and therefore SGLang's default attention path exclude
[[accelerators/nvidia-v100-sxm2-32gb]] and everything older. This is one of the very few
hard architecture exclusions in the repo, and it is why the Volta record exists.

### 1.2 ROCm is not a binary: here is exactly where it degrades

[[engines/vllm]] supports `rocm`, but its `hardware_caveats` are specific enough that
"AMD works" is not a supportable statement. Verbatim, the constraints that matter:

- **Version-pinned wheels.** "prebuilt wheels exist only for specific ROCm versions
  (documented table: rocm700 for vLLM 0.14.0-0.18.0, rocm721 for nightly after a named
  commit), Python 3.12, glibc >= 2.35. MI350 needs ROCm 7.0+, Ryzen AI MAX / AI 300 needs
  ROCm 7.0.2+." And: "ROCM wheels bundle their own PyTorch and ROCm kernels, so mixing with
  another ROCm or PyTorch build **forces a source build**."
- **Feature-level holes.** "On ROCm the encoder-decoder (enc-dec) feature is unsupported;
  multi-step is also unsupported on AMD in the feature x hardware matrix, and async output
  is unsupported on AMD and on CPU."
- **Quantization is not portable.** "Marlin (GPTQ/AWQ/FP8/FP4) has no AMD, Intel or CPU
  path; **AWQ and GPTQ have no AMD GPU path**; bitsandbytes and DeepSpeedFP have no
  AMD/Intel/CPU path; FP8 needs Ada or newer."

That last line is the one that reshapes a hardware decision: on an AMD part you lose AWQ
and GPTQ entirely, and Marlin — the kernel that [[flops/marlin-weight-only-int4-gemm]]
describes as the one that "should extend to batch sizes 4-8x larger than prior kernels
achieved" — is not available at all.

**SGLang's supported-hardware list is narrower than vLLM's** and the difference is
decision-relevant: "NVIDIA A100/H100/H200/H800/H20/B200/B300/GB200/GB300 plus select RTX
30/40/50, and AMD MI300X/MI325X/MI350X/MI355X. **No consumer AMD Radeon entry.**" Its
`avoid_for` says the same: "Consumer AMD Radeon cards, which are not in the documented
support list."

**SGLang also requires CUDA 13.** "the CUDA 12.9 wheel/image lane is retired because
PyTorch 2.14 publishes no CUDA 12.9 builds, and SGLang 0.5.19 is the last release with a
CUDA 12 lane." Its `avoid_for`: "Environments pinned to CUDA 12 - **there is no supported
lane any more**." Plus an operational hazard worth knowing: "Many dependencies publish
pre-release wheels only, so uv installs silently land on an old version (0.5.9) unless
`--prerelease=allow` is passed."

**Llama.cpp's ROCm story is HIP plus a build flag.** [[engines/llama-cpp]] lists `hip`,
not `rocm`: "HIP requires you to name GPU_TARGETS at build time (e.g.
`-DGPU_TARGETS=gfx1100`) or the build compiles for every GPU present; unsupported gfx
versions need `HSA_OVERRIDE_GFX_VERSION`, which **AMD does not support on Windows**."

**Ollama's supported gfx targets are enumerated**, which makes it the most checkable of
the portable engines: "gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100/1101/1102,
gfx1150/1151, gfx1200/1201." And its stated escape hatch: "HSA_OVERRIDE_GFX_VERSION=10.3.0
to run an unsupported card on a nearby LLVM target."

**Mlc-llm's caveat is about honesty of the word "supported":** "the README's matrix marks
AMD/NVIDIA Vulkan and CUDA, Intel Vulkan, Apple and AMD Metal on macOS, WebGPU/WASM in
browsers... but the rows mix targets, so **'supported' means a build exists, not that
performance is competitive**."

### 1.3 Architecture exclusions

| Exclusion | Engines | Cite |
|---|---|---|
| **Below compute capability 7.5** — excludes V100, Titan V, GTX 1000 series | [[engines/hf-text-embeddings-inference]], [[engines/flashinfer]], [[engines/vllm]] | TEI: "CUDA GPUs with compute capability below 7.5 are not supported (V100, Titan V, GTX 1000 series)"; builds are "per-architecture cargo features" and the Docker image "is pinned to a compute capability" |
| **Volta excluded from chunked prefill and automatic prefix caching** (even though CUDA generally works) | [[engines/vllm]] | verbatim caveat |
| **CUDA graphs unavailable** on the CPU backend **and on the Intel GPU platform** | [[engines/vllm]] | verbatim |
| **Machete requires sm90 (Hopper)**, CUDA only — so Ampere, Ada and consumer Blackwell (SM120) all fall through to Marlin | [[flops/machete-weight-only-hopper-gemm]] | hard-coded in vLLM: `get_min_capability()` returns 90 and `can_implement()` returns False with 'Machete only supported on CUDA' |
| **Marlin requires CUDA ≥ 11.8 and compute capability ≥ 8.0**, and "Marlin is not yet optimized for Hopper" | [[flops/marlin-weight-only-int4-gemm]] | so Hopper is the fallback, not the fast path |
| **FlashAttention-3 is Hopper-only** — requires H100/H800 and CUDA ≥ 12.3; "on Ada or Ampere there is no wgmma and no TMA, so the FA-3 techniques are unavailable" | [[flops/flashattention-3-hopper-warp-specialization]] | |
| **FlashMLA 2026.09.30 requires SM100/SM103 with CUDA ≥ 13.1 and REMOVED Hopper support** | [[flops/flashmla-decode-fp8-fp4-kv]] | "it REMOVED Hopper support and earlier models (V3/V3.2/V4.0) while changing the KV cache format, so an older checkout is required" |
| **sm100a kernels do not run on SM120** (consumer Blackwell) | [[flops/sm120-vs-sm100a-kernel-binary-incompatibility]], [[engines/vllm-cpp]] | CUTLASS: "kernels compiled for Blackwell SM100 architecture with arch conditional features (using sm100a) are not compatible with RTX 50 series GPUs" |
| **DeepEP expert parallelism requires Hopper or newer + NVLink + RDMA** | [[flops/deepep-expert-parallel-dispatch-combine]] | "zero-SM RDMA EP is not supported" |
| **Apple Silicon is not an in-tree vLLM backend** — needs the separately maintained plugin | [[engines/vllm]], [[engines/vllm-metal]] | vLLM's own docs point at vLLM-Metal as a separate package |
| **MLX has no CUDA, ROCm, Linux-CPU or Windows path** | [[engines/mlx-lm]] | "This is a hard boundary, not a preference" |
| **OpenVINO GenAI is Intel-only** — "there is no CUDA or ROCm path" | [[engines/openvino-genai]] | |
| **Triton runs on NVIDIA GPUs, x86 and ARM CPU, and AWS Inferentia** — "ROCm, Metal and Apple Silicon are not supported targets" | [[engines/triton-inference-server]] | "the least vendor-neutral engine in this slice by a wide margin" |
| **TEI's ROCm support is documented only for AMD Instinct MI200 and MI300** — "Consumer Radeon is not mentioned" | [[engines/hf-text-embeddings-inference]] | |
| **TEI's ARM64 Docker image has no Metal/MPS** — Apple acceleration comes from the local cargo build or Homebrew, not the container | [[engines/hf-text-embeddings-inference]] | |

The **SM120 vs SM100a** split deserves emphasis because it is invisible from the marketing
name: "two parts both sold as 'Blackwell' can land on entirely different kernels for the
same model and engine — and therefore different decode throughput — even though their
datasheets share a family name." [[accelerators/nvidia-b200]] and
[[accelerators/nvidia-gb200-nvl72]] (SM100a) and [[accelerators/nvidia-rtx-5090]] /
[[accelerators/nvidia-rtx-pro-6000-blackwell]] (SM120) do not share a kernel target.

### 1.4 Engines with no compute backend at all

Three records exist so a reader querying by backend does not mistake them for gaps:

- [[engines/litellm]] — `supported_backends` is an explicit "none": "LiteLLM never loads
  weights and never touches a GPU, CPU or NPU." Its caveat: "Latency is added, not
  removed. Every request pays at least one extra network hop, so it is the wrong layer for
  a single-user local model."
- [[engines/lm-eval-harness]] — "Not an inference engine - it scores, it does not serve.
  **Any tok/s figure attributed to it is from a backend plugin, not from the harness
  itself.**"
- [[engines/smoothquant-reference]] and [[engines/mxptq-research-harness]] — evaluation
  harnesses with `backends: none`. SmoothQuant's is a methodology warning: "the 1.51x/1.56x
  speedups come from the FasterTransformer INT8 integration. **Accuracy and speed in that
  paper are NOT from the same runtime.**"

And four records that are libraries/toolkits rather than servers, listed here so the table
in §2 is not misread: [[engines/hf-optimum]] ("a TOOLKIT, not a serving runtime"),
[[engines/sentence-transformers]] ("no HTTP server, no batching scheduler"),
[[engines/flashinfer]] ("not an engine: you cannot serve a model with it alone"),
[[engines/colbert-reference]] ("Not a server: no HTTP API, no scheduler").

---

## 2. The engine × backend table

Generated from `supported_backends`. `Y` = the record lists that backend (or an obvious
compound of it). `—` = not listed. Note that **absence here means "the record does not
claim it", not "impossible"** — but for the CUDA-only engines in §1.1 it does mean
impossible, because their `hardware_caveats` say so.

| engine | CUDA | ROCm | HIP | Metal | Vulkan | Intel XPU | CPU | OpenCL | OpenVINO | TensorRT | ONNX RT | NPU | TPU | WebGPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vllm | Y | Y | — | Y* | — | Y | Y | — | — | — | — | — | — | — |
| sglang | Y | Y | — | Y | — | Y | Y | — | — | — | — | — | Y | — |
| tensorrt-llm | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| llama-cpp | Y | — | Y | Y | Y | — | Y | Y | Y | — | — | — | — | Y |
| llamafile | Y | Y | Y | Y | Y | — | Y | — | — | — | — | — | — | — |
| ollama | Y | Y | Y | Y | Y | — | Y | — | — | — | — | — | — | — |
| koboldcpp | Y | Y | Y | Y | Y | — | Y | — | — | — | — | — | — | — |
| jan | Y | Y | Y | Y | Y | — | Y | — | — | — | — | — | — | — |
| lemonade | Y | Y | Y | Y | Y | — | Y | — | — | — | — | Y | — | — |
| exllamav3 | Y | — | — | — | — | — | Y | — | — | — | — | — | — | — |
| exllamav2 | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| mlc-llm | Y | Y | — | Y | Y | — | — | Y | — | — | — | — | — | Y |
| mlx-lm | — | — | — | Y | — | — | — | — | — | — | — | — | — | — |
| vllm-metal | — | — | — | Y | — | — | — | — | — | — | — | — | — | — |
| vllm-cpp | Y | — | — | Y | Y | — | Y | — | — | — | — | — | — | — |
| ktransformers | Y | — | — | — | — | — | Y | — | — | — | — | Y | — | — |
| openvino-genai | — | — | — | — | — | — | Y | — | — | — | — | Y | — | — |
| onnxruntime-genai | Y | — | — | — | — | — | Y | — | Y | — | Y | — | — | Y |
| hf-text-embeddings-inference | Y | Y | — | Y | — | — | Y | — | — | — | Y | — | — | — |
| triton-inference-server | Y | — | — | — | — | — | Y | — | Y | Y | Y | — | — | — |
| fastertransformer | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| flashinfer | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| tinychat-awq | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| mistral-rs | Y | — | — | Y | — | — | Y | — | — | — | — | — | — | — |
| text-generation-inference † | Y | Y | — | — | — | Y | Y | — | — | — | — | Y | — | — |
| hf-optimum | Y | Y | — | — | — | — | — | — | Y | Y | Y | Y | — | — |
| sentence-transformers | Y | Y | — | — | — | — | Y | — | Y | — | — | — | — | — |
| infinity-embeddings | Y | Y | — | — | — | — | Y | — | — | — | Y | Y | — | — |
| vllm-pooling-models | Y | Y | — | Y | — | Y | Y | — | — | — | — | — | Y | — |
| xinference | Y | Y | — | — | Y | — | Y | Y | — | — | — | — | — | — |
| ik-llama-cpp | Y | — | — | — | — | — | Y | — | — | — | — | — | — | — |
| clip-as-service | Y | — | — | — | — | — | — | — | — | Y | Y | — | — | — |
| colbert-reference | Y | — | — | — | — | — | Y | — | — | — | — | — | — | — |
| spinquant-research-harness | Y | — | — | — | — | — | — | — | — | — | — | — | — | — |
| llama-cpp-embedding-server | Y | — | Y | Y | Y | — | Y | — | — | — | — | — | — | — |
| qdrant-fastembed | Y | — | — | — | — | — | — | — | — | — | Y | — | — | — |
| litellm | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| lm-eval-harness | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| smoothquant-reference | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| mxptq-research-harness | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

\* [[engines/vllm]] lists `metal` in `supported_backends` **but its `hardware_caveats` say
"Apple Silicon is not an in-tree backend"** and the `notes` field qualifies it: "'metal'
only via the out-of-tree VLLM-Metal plugin, not upstream." The caveat governs.

† [[engines/text-generation-inference]] is **`deprecated`**: "The repository was ARCHIVED
by its owner on 2026-03-21 and is read-only." Its `avoid_for` is "Any new deployment - it
is archived and unmaintained." Recorded because its backend matrix predates the current
generation of engines: "what worked here may not reflect present-day kernels."

---

## 3. Choosing by hardware: the short version

### If you have NVIDIA datacenter silicon (H100/H200/B200/B300/GB200)

[[engines/tensorrt-llm]] and [[engines/vllm]] are the two real choices, and the records
make the tradeoff explicit rather than leaving it to folklore:

**TensorRT-LLM** `best_for`: "Maximum throughput on a fixed NVIDIA fleet, especially MoE
models where expert parallelism and load balancing are the bottleneck"; "Deployments that
can afford a build-time/version-lock step in exchange for hand-tuned kernels, CUDA graphs
and the overlap scheduler." Its cost is stated concretely: "Version pinning is severe: the
pip wheel is built against a specific PyTorch and CUDA Toolkit... and a documented failure
mode is pip replacing your CUDA-13-compatible torch with a CUDA 12.8 build, leaving an
unusable install until you lock torch with a constraints file." Also: "MoE scaling features
(Wide-EP, EPLB, one-sided all-to-all over NVLink, DWDP) are written against NVLink
topologies such as NVL72, so **their published wins do not transfer to PCIe-only boxes**."

**vLLM** `best_for`: "Multi-user servers with high request concurrency, where continuous
batching plus paged KV is what actually buys throughput"; "Broad model and quantization
coverage when you want one engine rather than a per-quantization specialist." Its cost:
"avoid_for: Hardware without a maintained kernel path - every accelerator gain is tied to a
bespoke kernel set that has to be compiled and validated per platform."

The only record that compares them head-to-head on identical hardware and model is the
benchmark pair: [[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-h100]] (vLLM, H100)
versus [[benchmarks/vllm-llama31-405b-fp8-8xh100-tp8-output-tps]] (vLLM) and
[[benchmarks/trtllm-llama31-405b-fp8-h100-tp8-tps]] (TensorRT-LLM, same model, same format,
same 8×H100 TP8) — 2,884.69 tok/s for TensorRT-LLM. That is **one** data point and the
records do not generalise it; treat it as a pointer, not a ranking.

**SGLang** is the third option, distinguished by exactly one mechanism: `radix-cache` — the
`best_for` is "Agentic and RL workloads where many requests share long, branching context -
the radix cache reuses **partial** prefixes, not just exact ones." If your workload shares
partial prefixes, that is the field to shop on.

**Windows is excluded** by [[engines/vllm]]: "Linux only: the project states it does not
support Windows natively; WSL or a community fork is required." Its `avoid_for` repeats
it. TensorRT-LLM does not record a Windows exclusion, so on Windows between those two the
choice is effectively forced.

### If you have NVIDIA consumer silicon (RTX 30/40/50, RTX PRO 6000)

- **Hardware choice:** [[accelerators/nvidia-rtx-pro-6000-blackwell]] (96 GB ECC) or
  [[accelerators/nvidia-rtx-4090]] / [[accelerators/nvidia-rtx-5090]].
- **Engine:** [[engines/exllamav3]] for single-user low-bit quality — "Current-generation
  NVIDIA consumer cards: best-quality low-bit quants plus explicit MoE expert/tensor
  parallelism for machines that are not a datacenter"; architecture coverage "the broadest of
  any engine here." Its costs: "Build-time CUDA/Triton setup is real work compared to
  llama.cpp or ONNX paths" and "On Windows the triton-windows package is a declared
  dependency — attention, cache and recurrent kernels are Triton, and ExLlamaV3 **will not
  import without it**."
- **Portable alternative:** [[engines/llama-cpp]] — its `best_for` includes "Running a
  quantized GGUF on whatever GPU you happen to own, including GPUs with no vendor compute
  SDK." Its `avoid_for` names the consumer-NVIDIA weakness precisely: "Squeezing the last
  FLOP out of a Blackwell or Hopper part - the default **W4A8 fallback for W4A16 4-bit
  layers** is a concrete example."
- **Zero-install:** [[engines/llamafile]] — but "the bundled CUDA library is
  size-optimized and **OMITS the IQ-quant kernels**", so offloading an IQ-quant model to
  NVIDIA "leaves just those layers on the CPU."

### If you have AMD Instinct (MI300X / MI325X / MI350X / MI355X)

[[engines/vllm]] or [[engines/sglang]]. Both support `rocm`; both have Instinct-class parts
in their documented lists. The honest summary of the constraint set is in §1.2: pinned ROCm
wheels, no Marlin, no AWQ/GPTQ on AMD GPU, no enc-dec, no multi-step, no async output on
AMD. And there is a benchmark pair showing the outcome flips by model size:
[[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-output-throughput]] records MI300X at
3171 output tok/s aggregate against [[benchmarks/mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-h100]]
1957 for H100 on Llama 3.1 405B FP8 TP=8 (1.62x), while
[[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]] gives MI300X 15105
against [[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-h100]] 15810 on 70B (0.96x).
That is the single best argument in the repo against choosing hardware by FLOPS — **and** the
reason you must not quote either number alone: the NVIDIA side was measured by AMD with a
different engine (TensorRT-LLM), and the records say the delta "blends silicon and
software" and is "not memory-capacity-neutral" (192 GB against 80 GB).

**The consumer-Radeon trap is now four records deep, and two are silent.** Beyond
[[engines/sglang]] having no Radeon entry and [[gotchas/aiter-gate-skips-rdna3-gfx1100]]
confirming RDNA is off-roadmap, the new records add:
[[gotchas/rocm-backend-silently-serves-weights-from-host-ram-on-gfx1201]] (the ROCm backend
"reports the full VRAM size... and then allocates nothing on the GPU" — 256 GB of system RAM
for a 30 GB model), [[gotchas/rocm-flash-attention-kv-type-graph-split-explosion]] (a q5_0
KV cache fragments the ROCm graph from 2 splits into 34 and costs 68% of prefill),
[[gotchas/vulkan-rebar-on-causes-amd-device-lost]] (**blocker**, device-lost with ReBAR on)
and [[gotchas/vulkan-rebar-off-collapses-amd-decode]] (its slowdown twin). The AMD consumer
path's consistent instruction across all of them is Vulkan — with its own two traps,
[[gotchas/vulkan-suballoc-fragmentation-decode-cliff]] and
[[gotchas/windows-hip-multi-gpu-silent-output-corruption]] (**blocker**, silent word salad
on a multi-GPU Windows HIP pair with healthy-looking throughput).

[[engines/sglang]] has no Radeon entry. [[engines/llama-cpp]]
says of its HIP path: "AMD does not support" `HSA_OVERRIDE_GFX_VERSION` **on Windows**. And
[[engines/vllm-cpp]] is candid: "AMD path is brand new: EXL3 gained a native ROCm path
validated only on gfx1151 with no CPU fallbacks; discrete AMD validation and competitive
performance are **explicitly unmeasured**." If your AMD part is a consumer Radeon, the
portable answer the records support is [[engines/ollama]] or [[engines/llama-cpp]] with the
**Vulkan** path — [[engines/ollama]]'s `best_for` includes "AMD users on a card ROCm does
not list: **Vulkan just works**", and [[engines/llamafile]] covers "Intel Arc and anything
else ROCm does not cover... via the bundled Vulkan path." But read
[[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] before trusting any of it.

### If you have Intel hardware (Arc B580/B60, Gaudi2/Gaudi3, Core Ultra NPU)

- **Gaudi:** [[engines/vllm]] `xpu` and [[engines/hf-optimum]] both list Gaudi. But vLLM's
  caveat: "CUDA graphs are unavailable on the CPU backend **and on the Intel GPU
  platform**." Optimum's caveat is blunter: "Trainium, Gaudi and Furiosa are best-effort
  integrations whose depth varies by model family," and "'Optimum supports this backend'
  does not mean 'this backend is fast for your embedding model'." There are two Gaudi
  benchmark records ([[benchmarks/gaudi2-mlperf-v4-0-llama2-70b-offline-tokens]],
  [[benchmarks/arc-pro-b70-mlperf-v6-1-llama2-70b-offline-tokens]]).
- **Arc: the trap list is now long and all one direction.** Six records converge on "use
  Vulkan instead, or don't": [[gotchas/sycl-device-memory-query-abort-blocks-model-load]]
  (**blocker** — model load aborts with `GGML_ABORT`), [[gotchas/sycl-build-unrecognized-fsycl]]
  (**blocker** — build fails), [[gotchas/sycl-quant-reorder-gap-makes-some-quants-4x-slower]]
  (a **4x** spread between Q4_0 at 23.67 t/s and Q8_0/IQ4_NL on the same B70),
  [[gotchas/sycl-hybrid-linear-attention-arch-crash-intel-arc]] (`qwen3next`/`qwen35` crash
  or return gibberish on builds after b8477), [[gotchas/sycl-multi-gpu-arc-loses-p2p-and-crashes]]
  (no P2P on the OpenCL adapter) and [[gotchas/vllm-arc-b60-no-pcie-p2p-kv-transfer]] (no
  supported P2P path for KV transfer, "the blocker is upstream of vLLM"). **There is no
  combination of Arc + SYCL + llama.cpp in these records that works cleanly.**
- **Arc / Core Ultra:** [[engines/openvino-genai]] is the vendor-supported answer — `best_for`
  "Intel Xeon, Core Ultra and Arc hardware where you want a vendor-supported engine rather
  than a community fork"; "Intel NPU deployment — this is one of the few engines with a
  first-class NPU story." Its `avoid_for`: "NVIDIA or AMD GPUs", "Maximum tok/s per GPU on
  any vendor", "GGUF/EXL2 ecosystems; this is IR-only." And its caveat on precision: "Models
  must be exported to OpenVINO IR first... Quantization is baked in at export (the quickstart
  shows `--weight-format int4`), so precision changes are an **export-time decision**."
- **Arc via llama.cpp:** [[engines/llama-cpp]]'s SYCL note warns "the Intel oneMKL/oneAPI
  CPU build config does not support Intel GPU - use the SYCL path for that," and its
  OpenVINO backend is "explicitly marked In Progress for Intel CPUs, GPUs and NPUs."
  Read [[gotchas/intel-dgpu-vulkan-or-sycl-agentic-stall]] first: "Intel dGPU support in
  llama.cpp should be treated as **experimental for long-context or agentic workloads**;
  validate on your exact SKU before committing to it." And [[gotchas/sycl-build-unrecognized-fsycl]]
  is a **blocker** build failure.

### If you have Apple Silicon

- [[engines/mlx-lm]] — "Local Mac development and personal use"; MLX is Metal-only, full
  stop. Its concurrency cost is specific: "`--kv-bits` (4 or 8 bit KV cache quantization)
  explicitly disables batching and serves one request at a time, so enabling KV
  quantization costs you concurrency."
- [[engines/llama-cpp]] — Apple Silicon "is a first-class target optimized via NEON,
  Accelerate and Metal."
- [[engines/vllm-metal]] — for vLLM serving *semantics* on a Mac. Its `avoid_for`:
  "Multi-tenant datacenter serving - **unified memory bandwidth does not compete with HBM**";
  "Production use where the plugin's release cadence and community-maintained status are
  unacceptable risk."
- Also available: [[engines/koboldcpp]], [[engines/jan]], [[engines/lemonade]],
  [[engines/ollama]], [[engines/mlc-llm]], [[engines/xinference]] (apple-mps),
  [[engines/llamafile]], [[engines/mistral-rs]], [[engines/onnxruntime-genai]],
  [[engines/hf-text-embeddings-inference]].

**Apple's hardware story is capacity, not speed** — see [[accelerators/apple-m3-ultra]] and
[01-hardware-selection.md §1](01-hardware-selection.md#1-single-user-local-chat-laptop-or-desktop-one-stream).

### If you have TPU, Gaudi-NPU, Inferentia, Trainium or an Ascend NPU

- **TPU:** [[engines/sglang]] and [[engines/vllm-pooling-models]] list `tpu`. But
  [[engines/sglang]]'s note: "TPU support is split across sibling projects (SGL-JAX,
  SGL-torchtpu), so TPU parity with the main CUDA path is **not automatic**." And
  [[supply/gcp-cloud-tpu-v4-v5e-v5p]] on the portability cost: "**XLA/JAX only.** Every
  engine, kernel and quantization path elsewhere in this repo is CUDA-shaped."
- **Ascend NPU:** only [[engines/ktransformers]] lists `npu-ascend`, and its caveat is that
  "Ascend NPU support is CPU-expert-offload dependent, so **the NPU is not doing the expert
  math**." For the silicon itself see [[accelerators/huawei-ascend-910b]] (`contested`,
  confidence 0.45, "no primary source at all") and note that
  [[flops/tilelang-tile-dsl-moe-routing-quant]] claims a single TileLang source targeting
  both "NVIDIA SM90/SM100 and Huawei Ascend 950 with identical Python APIs" — a portability
  *claim*, with the record's own caveat that "most kernels achieve performance close to the
  hardware's limits" is DeepSeek's own and "carries no numbers."
- **Inferentia2 / Trainium2:** [[engines/hf-optimum]] and
  [[engines/infinity-embeddings]] list `inferentia2`; [[engines/triton-inference-server]]
  lists `inferentia`. All three describe it as an export-time/best-effort path.
- **Ryzen AI NPU:** [[engines/lemonade]] is the only complete story — "AMD Ryzen AI /
  Strix Halo machines, where it is the most complete story available for NPU plus iGPU
  plus dGPU in one server." Its caveats are strict: "`ryzenai-llm` (NPU) is **Windows-only**";
  "The NPU story is XDNA2-specific — earlier Ryzen AI NPU generations are not covered."

### If you have a GPU with no vendor SDK at all (or a browser)

[[engines/llama-cpp]]'s framing is the cleanest statement of the Vulkan position:
"Vulkan is the **universal escape hatch** but is not the fast path: it is the only route
to a GPU with no vendor SDK, and on Windows it needs the LunarG Vulkan SDK plus
w64devkit/MSYS2 plumbing." [[engines/ollama]] and [[engines/xinference]] also list Vulkan;
[[engines/llamafile]] bundles it. [[engines/mlc-llm]] is the browser/on-device answer
(WebGPU + WASM).

**The Vulkan tax is quantified in the gotcha records**, and it is not small:
[[gotchas/vulkan-flash-attn-slower]] — "on Strix Halo (RADV, warp size 64) llama-bench
shows FA on costing more than it saves: llama-2-7b.Q4_0 pp512 drops **1252.63 → 508.36
tok/s** and tg128 drops **49.91 → 25.80 tok/s**"; its `avoid_for` is "Do not assume FA is a
free win on Vulkan the way it is on CUDA." Plus
[[gotchas/vulkan-decode-cliff-hidden-size-4096]] (gfx1201 hidden-size cliff) and
[[gotchas/vulkan-q4km-speed-regression]] (a bisected Windows regression).

### If your workload is embeddings or rerank, not generation

This is a **different hardware question**, and the records are unanimous about why: "a
BERT-size encoder is **compute-bound**, not memory-bound, at batch 1... a bidirectional
single-pass encoder with no KV cache and no autoregressive streaming runs dense GEMM, so
it wants FLOPS, not bandwidth — **the exact inverse of an LLM decode part** on the same
box." ([[engines/hf-text-embeddings-inference]], and the same paragraph in
[[engines/vllm-pooling-models]], [[engines/sentence-transformers]],
[[engines/triton-inference-server]], [[engines/xinference]],
[[engines/qdrant-fastembed]], [[engines/hf-optimum]], [[engines/colbert-reference]],
[[engines/clip-as-service]].)

The engine choice follows from that:
- [[engines/hf-text-embeddings-inference]] — "the reference implementation most teams reach
  for first, and it is what Hugging Face itself deploys." Its hardware floor is the most
  carefully documented of any engine: images are built per compute capability (75/80/86/89/90/100/120/121),
  "Blackwell images (100, 120, 121) are labelled **experimental**", and "Flash Attention is
  OFF by default in the Turing image because of precision issues... Numerical parity with
  the other images is therefore not automatic."
- [[engines/vllm-pooling-models]] — one server for generation *and* retrieval, and the only
  mainstream engine with MaxSim over `token_embed`. But its own docs undercut the case: "We
  currently support pooling models **primarily for convenience. This is not guaranteed to
  provide any performance improvements** over using Hugging Face Transformers or Sentence
  Transformers directly."
- [[engines/llama-cpp-embedding-server]] — GGUF embeddings in one binary. Its warning
  transfers directly from the LLM side and is worth repeating: "quantizing a 300M-parameter
  reranker to Q4 usually costs more accuracy than it saves memory."
- The new flop records sharpen *why* these engines exist at all:
  [[flops/cross-encoder-reranking]] ("compute-bound even at batch 1... the exact opposite of
  the generative decode regime") and [[flops/embedding-batch-encoding]] (AI = B·t,
  compute-bound at B ≥ 3 for 128-token docs). An encoder-serving box should be provisioned
  like a batch-serving box, not like a chat box.
- [[engines/clip-as-service]] — **effectively dormant**: "Last push was 2024-01-23 - over
  two and a half years stale"; `avoid_for` "Anything new: the project is dormant."
- [[engines/colbert-reference]] — late interaction, and its hardware requirement is the
  inverse of everything above: "Hardware selection therefore needs **BOTH**: a high-FLOPS part
  for encoding the corpus, and high-bandwidth/high-capacity RAM or storage for the index,
  which is the opposite mix from what an LLM decode part wants."

---

## 4. Where the records are thin

1. **No engine × accelerator *performance* matrix.** The table above is a support matrix
   transcribed from docs. The records warn in three separate places that support ≠
   performance ([[engines/hf-optimum]], [[engines/mlc-llm]],
   [[engines/hf-text-embeddings-inference]]), and no record quantifies the gap. This is the
   highest-value gap in engine selection.
2. **No Apple-silicon performance records beyond MLX decode.** The only Apple benchmark
   records are `mlx-lm` on M3 Max and M4 Max, one of them `draft`. There is no Apple
   multi-user serving number anywhere.
3. **No AMD ROCm vs CUDA controlled comparison beyond the two MLPerf-scale vLLM pairs.**
   [[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]] and its 405B
   counterpart are valuable and there are only four such records total.
4. **Consumer-Radeon ROCm maturity is unmeasured by the projects' own admission.**
   [[engines/vllm-cpp]] says so explicitly; [NOTES.md](NOTES.md) carries "How much of the
   ROCm-on-consumer-Radeon story has actually been fixed upstream?" as an open question.
5. **`xpu` is thin.** Only [[engines/vllm]], [[engines/sglang]], [[engines/vllm-pooling-models]]
   and the deprecated [[engines/text-generation-inference]] list it, and vLLM's own caveat is
   that CUDA graphs are unavailable there.
6. **Windows.** [[engines/vllm]] is Linux-only; [[engines/exllamav3]] needs `triton-windows`;
   [[engines/llama-cpp]]'s Vulkan path needs MSYS2 plumbing; [[engines/ollama]] has a
   suspend/resume bug ("Linux suspend/resume can make NVIDIA GPU discovery fail and silently
   fall back to CPU"). No engine record makes Windows a first-class serving target.

---

## Related documents

- [01-hardware-selection.md](01-hardware-selection.md) — choosing the part before choosing the engine.
- [02-flop-map.md](02-flop-map.md) — why the crossovers an engine cannot reach are hardware limits.
- [04-quant-selection.md](04-quant-selection.md) — the format↔engine↔hardware compatibility matrix.
- [05-known-traps.md](05-known-traps.md) — every way the above can fail silently at runtime.