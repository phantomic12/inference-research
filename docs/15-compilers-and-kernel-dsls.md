# 15 — Compilers and kernel DSLs: what actually generates the kernel

<!-- written 2026-10-12 against the 22 records in data/compilers/. Every row is read from the record's own fields. -->

Every engine in [03-engine-selection.md](03-engine-selection.md) is, underneath, a choice
of who writes the kernels: a hand-written CUDA/HIP library, a template library, a DSL, or a
graph compiler. That choice decides three things this repo keeps rediscovering as gotchas —
**which hardware is reachable at all**, **where the compile cost lands** (build time vs.
first request), and **whether an artifact is portable**. The 22 compiler records answer
those three questions; this doc reads them together.

## 0. The decision, short form

| you need | reach for | because |
|---|---|---|
| a custom kernel on NVIDIA or AMD, fast to write | [[compilers/triton]] or [[compilers/w7k-tilelang]] | tile-level Python DSLs, autotuned, JIT; both target CUDA and ROCm |
| peak GEMM/attention on one NVIDIA generation | [[compilers/cutlass-cute]] (or [[compilers/w7k-thunderkittens]]) | the best-known GEMM performance, at a very-high learning curve and long template compile times |
| the same on AMD | [[compilers/w7k-amd-composable-kernel]] | the ROCm template library; a header library, not a compiler |
| a TPU | [[compilers/xla]] and [[compilers/pallas]] | XLA is the only compiler targeting TPUs at scale; Pallas is the kernel escape hatch |
| no JIT at serve time on NVIDIA | [[compilers/w7k-tensorrt]] | AOT engine build with per-layer tactic autotuning — but the engine is arch- and version-bound |
| speed-up for existing PyTorch code | [[compilers/torch-compile-torchinductor]] | one call; watch for graph breaks and recompilation storms |
| a non-GPU accelerator | its vendor stack: [[compilers/w7k-cann-ascend-graphengine]], [[compilers/w7k-tt-forge]], [[compilers/w7k-cerebras-cslc]], [[compilers/w7k-qualcomm-qnn]] | no portable compiler in this corpus reaches them |

## 1. The four families

### Kernel DSLs — write the kernel, the compiler schedules it

| record | category | targets | strategy | autotune | production | learning curve |
|---|---|---|---|---|---|---|
| [[compilers/triton|Triton]] | `dsl` | cuda, rocm | `jit` | yes | True | moderate |
| [[compilers/pallas|Pallas]] | `kernel-dsl` | tpu, cuda | `jit` | no | True | moderate |
| [[compilers/w7k-tilelang|TileLang]] | `kernel-dsl` | cuda, rocm, metal | `mixed` | yes | True | moderate |
| [[compilers/w7k-cerebras-cslc|Cerebras CSL compiler]] | `kernel-dsl` | — (no schema slot) | `aot` | no | True | high |

### Template libraries — hand-tuned building blocks, no compiler

| record | category | targets | strategy | autotune | production | learning curve |
|---|---|---|---|---|---|---|
| [[compilers/cutlass-cute|CUTLASS / CuTe]] | `template-library` | cuda | `template-instantiation` | no | True | very-high |
| [[compilers/w7k-amd-composable-kernel|AMD Composable Kernel]] | `template-library` | rocm | `template-instantiation` | no | True | high |
| [[compilers/w7k-thunderkittens|ThunderKittens]] | `template-library` | cuda, rocm | `template-instantiation` | no | True | high |
| [[compilers/w7k-arm-kleidiai|Arm Kleidi Libraries]] | `template-library` | cpu-avx512 | `template-instantiation` | no | True | low |

### Graph compilers — take a model, emit fused kernels

| record | category | targets | strategy | autotune | production | learning curve |
|---|---|---|---|---|---|---|
| [[compilers/torch-compile-torchinductor|torch.compile / TorchInductor]] | `compiler-framework` | cuda, rocm, cpu-avx2, cpu-avx512 | `jit` | yes | True | moderate |
| [[compilers/xla|XLA]] | `compiler-framework` | cuda, cpu-avx2, tpu | `aot` | no | True | high |
| [[compilers/w7k-tensorrt|TensorRT]] | `compiler-framework` | cuda | `aot` | yes | True | high |
| [[compilers/apache-tvm|Apache TVM]] | `compiler-framework` | cuda, rocm, metal, vulkan, cpu-avx2, opencl | `aot` | yes | False | high |
| [[compilers/iree|IREE]] | `compiler-framework` | cuda, rocm, vulkan, metal, cpu-avx2, spirv, webgpu | `aot` | no | True | high |
| [[compilers/w7k-nvfuser|nvFuser]] | `graph-optimizer` | cuda | `jit` | no | True | high |
| [[compilers/w7k-cann-ascend-graphengine|CANN / Ascend GraphEngine]] | `graph-optimizer` | — (no schema slot) | `graph-rewrite` | no | True | very-high |
| [[compilers/w7k-tt-forge|TT-Forge]] | `compiler-framework` | — (no schema slot) | `aot` | no | True | high |
| [[compilers/w7k-modular-max-mojo|MAX and Mojo]] | `compiler-framework` | cuda, rocm, metal, cpu-avx512 | `mixed` | no | False | high |
| [[compilers/w7k-qualcomm-qnn|Qualcomm AI Engine Direct]] | `compiler-framework` | — (no schema slot) | `aot` | no | None | very-high |

### Infrastructure and runtimes

| record | category | targets | strategy | autotune | production | learning curve |
|---|---|---|---|---|---|---|
| [[compilers/mlir|MLIR]] | `compiler-framework` | cuda, rocm, vulkan, cpu-avx2, spirv | `mixed` | no | True | very-high |
| [[compilers/w7k-cuda-nvrtc|CUDA NVRTC]] | `compiler-framework` | cuda | `jit` | no | True | high |
| [[compilers/onnx-runtime|ONNX Runtime]] | `runtime` | cuda, rocm, cpu-avx2, cpu-avx512, xnnpack, opencl, webgpu | `graph-rewrite` | no | True | low |
| [[compilers/openvino|OpenVINO]] | `runtime` | cpu-avx2, cpu-avx512 | `graph-rewrite` | no | True | low |

## 2. Where the compile cost lands

The strategy column is the cost model. A **JIT** stack pays at first request: Triton
compiles each kernel on first call and autotuning compiles every candidate config
([[compilers/triton]]); NVRTC moves compilation onto process start
([[compilers/w7k-cuda-nvrtc]]); torch.compile recompiles on new shapes
([[compilers/torch-compile-torchinductor]]); Ascend's GraphEngine generates code at serve
time, so cold start is spent there rather than at build
([[compilers/w7k-cann-ascend-graphengine]]). An **AOT** stack pays at build time and ships an
artifact: TensorRT ([[compilers/w7k-tensorrt]]), XLA ([[compilers/xla]]), IREE
([[compilers/iree]]), TVM — whose AOT compilation with autotuning the record puts at minutes
or more ([[compilers/apache-tvm]]). **Template instantiation** pays at the library user's
compile time ([[compilers/cutlass-cute]], [[compilers/w7k-amd-composable-kernel]]).

The cold-start consequences are in [21-deployment-topology.md](21-deployment-topology.md);
the measured warm-up traps are in [05-known-traps.md](05-known-traps.md).

## 3. Portability is the exception, not the default

- TensorRT engines are not portable across GPU architectures without hardware-compatibility
  mode ([[compilers/w7k-tensorrt]]).
- Cerebras CSL binaries must match the WSE generation they run on
  ([[compilers/w7k-cerebras-cslc]]).
- nvFuser wheels are bound to a specific PyTorch build ([[compilers/w7k-nvfuser]]).
- Modular MAX/Mojo's cross-vendor portability claim is vendor-published only; no
  third-party cross-vendor benchmark backs it in this corpus ([[compilers/w7k-modular-max-mojo]]).
- Only four records list both `cuda` and `rocm` as targets for kernel generation:
  [[compilers/triton]], [[compilers/w7k-tilelang]], [[compilers/w7k-thunderkittens]] (via its
  HipKittens port) and [[compilers/torch-compile-torchinductor]] — plus the graph-level
  [[compilers/apache-tvm]], [[compilers/iree]], [[compilers/mlir]] and [[compilers/onnx-runtime]].

## 4. What the records cannot answer

- **No compile-time benchmark records.** `compile_time_seconds` is prose-level in most
  records; there is no measured, methodology-complete cold-start comparison across stacks.
- **Two records are `draft`**: [[compilers/w7k-modular-max-mojo]] and
  [[compilers/w7k-qualcomm-qnn]] (QNN docs sit behind a sign-in wall, so its mechanics are
  unverified from a primary source).
- **Schema gap.** `target_backends` has no slot for Ascend, Tenstorrent, Cerebras or Hexagon,
  so those records carry an empty list rather than a mislabelled one
  ([[compilers/w7k-tt-forge]]).
- **No kernel-level performance join.** Which compiler produced the kernel behind a given
  benchmark row is not recorded on the benchmark.

## Related documents

- [03-engine-selection.md](03-engine-selection.md) — the engines these compilers sit under.
- [02-flop-map.md](02-flop-map.md) — the operation classes the kernels implement.
- [13-model-compatibility.md](13-model-compatibility.md) — which architecture needs which kernel.
- [21-deployment-topology.md](21-deployment-topology.md) — what a cold start costs.
