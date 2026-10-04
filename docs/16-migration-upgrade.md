# 16 — Migration and upgrade: when a new GPU generation is worth it

<!-- written 2026-10-04 against 2,592 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** you are running a serving workload on generation N.
Generation N+1 is available. Should you upgrade? The answer depends on whether your
workload is bound by the thing the new generation improves, and whether the price
increase is offset by the throughput increase. The records give enough data to answer
this for three specific generational transitions.

---

## 1. The one-paragraph version

A new GPU generation is worth it when **your workload is bound by the specific resource
the new generation improves** — and the price/throughput ratio favours the new part.
The records show three transitions where the answer is different:

- **H100 → H200:** worth it only if you are capacity-bound (KV cache or weights don't fit
  on 80 GB). At batch 1 on an 8B model, H200 is 1.14x faster but 2.02x more expensive
  per hour — the upgrade makes $/Mtok **worse** by 1.77x.
- **H100 → B200:** worth it if you need the FP4 tensor-core path or the 8 TB/s bandwidth
  for a workload that is compute-bound at high batch. At batch 1 on a 70B model, B200
  serves 6,725 tok/s on a single GPU where H100 needs TP=2 for 3,708 tok/s — but B200
  costs 3.2x more per hour.
- **A100 → H100:** worth it for almost every workload. H100 is 3.2x the dense bf16 FLOPS
  (989.5 vs 312) and 1.64x the bandwidth (3,350 vs 2,039 GB/s), and the rental price
  spread ($2.47 vs $3.43) is only 1.39x.

---

## 2. H100 → H200: the capacity upgrade that is not a speed upgrade

### 2.1 What changes

| Property | H100 SXM [[accelerators/nvidia-h100-sxm]] | H200 SXM [[accelerators/nvidia-h200-sxm]] | Ratio |
|---|---|---|---|
| Memory | 80 GB HBM3 | 141 GB HBM3e | 1.76x |
| Bandwidth | 3,350 GB/s | 4,800 GB/s | 1.43x |
| Dense bf16 FLOPS | 989.5 TFLOPS | same | 1.0x |
| TDP | 700 W | 700 W | 1.0x |

H200 is **purely a memory-capacity and bandwidth uplift** — same compute, same power.
The record's own summary: "it wins wherever KV cache or weights do not fit, and loses on
nothing."

### 2.2 The measured speedup at batch 1

[[benchmarks/trtllm-llama31-8b-fp8-h100-1k1k-tps]]: 14,991.62 output tok/s (H100, 1 GPU)
[[benchmarks/trtllm-llama31-8b-fp8-h200-1k1k-tps]]: 17,162.49 output tok/s (H200, 1 GPU)

**Speedup: 1.14x.** The 1.43x bandwidth increase yields only 1.14x throughput because
at batch 1 on an 8B model the step is launch-overhead-bound, not bandwidth-bound
([[flops/custom-kernel-launch-overhead]]).

### 2.3 The cost-per-token penalty

| Channel | H100 price | H200 price | H100 $/Mtok | H200 $/Mtok |
|---|---|---|---|---|
| Rental (Vast) | $2.4704 [[supply/rental-h100-sxm-vast-index]] | $5.0005 [[supply/rental-h200-sxm-vast-index]] | $164.8 | $291.3 |
| AWS on-demand | $6.88 [[supply/aws-ec2-p5-h100]] | $7.912 [[supply/aws-ec2-p5en-h200]] | $458.9 | $460.9 |

At rental rates, H200's $/Mtok is **1.77x worse** than H100's. At AWS on-demand the
prices are close enough that the 1.14x speedup makes H200 slightly better. The answer
is channel-dependent.

### 2.4 When H200 is worth it

- **KV cache does not fit on 80 GB.** A 70B model at 128k context with bf16 KV needs
  ~2.9 GB per sequence for the cache alone ([[flops/kv-transfer]] gives the per-token
  formula). At 200 concurrent sequences that is 580 GB — fits on H200's 141 GB with
  room for weights, does not fit on H100's 80 GB.
- **Weights do not fit on 80 GB.** A 405B model in bf16 needs 810 GB; in FP8 it needs
  405 GB. Neither fits on a single H100. H200's 141 GB does not change this — you need
  TP=8 either way. But if you are at TP=2 with an 80 GB card and the model is 130 GB,
  H200 lets you stay at TP=2 where H100 would force TP=4.
- **You are serving long-context and the bandwidth is the binding constraint.** At
  32k+ context, decode attention is bandwidth-bound on KV reads
  ([[flops/decode-attention]]: AI = g, flat in context). H200's 1.43x bandwidth increase
  translates directly to 1.43x decode throughput at long context.

### 2.5 When H200 is not worth it

- **Batch-1 serving of a model that fits on 80 GB.** The 1.14x speedup does not cover
  the 2.02x price increase. Stay on H100.
- **High-concurrency serving where H100 is already compute-bound.** Above batch ~295,
  H100 is compute-bound on the weight GEMM ([[flops/decode-gemm]]: B_ridge = 295). H200
  has the same compute, so there is no throughput gain at high batch.

---

## 3. H100 → B200: the FP4 path and the bandwidth leap

### 3.1 What changes

| Property | H100 SXM [[accelerators/nvidia-h100-sxm]] | B200 [[accelerators/nvidia-b200]] | Ratio |
|---|---|---|---|
| Memory | 80 GB HBM3 | 186 GB HBM3e | 2.33x |
| Bandwidth | 3,350 GB/s | 8,000 GB/s | 2.39x |
| Dense bf16 FLOPS | 989.5 TFLOPS | 2,500 TFLOPS | 2.53x |
| Dense FP8 FLOPS | 1,979 TFLOPS | 5,000 TFLOPS | 2.53x |
| FP4 tensor cores | none | native (block-scaled) | — |
| TDP | 700 W | 1,000 W | 1.43x |

### 3.2 The measured speedup

[[benchmarks/trtllm-llama33-70b-fp8-h100-tp2-1k2k-tps]]: 3,708.93 output tok/s (H100, TP=2)
[[benchmarks/trtllm-llama33-70b-fp4-b200-tp1-tps]]: 6,725.03 output tok/s (B200, TP=1)

**Speedup: 1.81x** — but this compares FP8 on H100 to FP4 on B200, so it blends the
2.53x FLOPS increase with the 2x FP4-vs-FP8 precision gain and the TP=2→TP=1
simplification. The records do not decompose the three.

### 3.3 The cost-per-token comparison

| Channel | H100 price | B200 price | H100 $/Mtok (70B FP8 TP2) | B200 $/Mtok (70B FP4 TP1) |
|---|---|---|---|---|
| Rental (Vast) | $2.4704 [[supply/rental-h100-sxm-vast-index]] | $7.9695 [[supply/rental-b200-vast-index]] | $666.0 | $1,185.1 |

B200's $/Mtok is **1.78x worse** than H100's for 70B serving. The single-GPU
simplification (TP=1 vs TP=2) saves networking and complexity but not cost.

### 3.4 When B200 is worth it

- **You need FP4 precision.** B200 is the only part with a native FP4 tensor-core path
  ([[quantization/nvfp4]]: native_support is Blackwell-only). If your quality target
  met FP4, B200 is the only option.
- **You are serving a model that fits on 186 GB but not on 80 GB at your target
  precision.** A 70B model in FP4 needs ~35 GB of weights — fits on a single B200 with
  room for KV. On H100 you need TP=2 for the same model in FP8.
- **You are compute-bound at high batch and need the 2.53x FLOPS increase.** Above the
  ridge point, B200's compute advantage is real and measurable.

### 3.5 When B200 is not worth it

- **You are serving small models at batch 1.** The launch overhead dominates and the
  extra FLOPS and bandwidth are wasted.
- **Your workload is memory-bandwidth-bound on weights, not compute-bound.** B200's
  bandwidth increase (2.39x) is real but costs 3.2x per hour. If you are already
  bandwidth-bound on H100, the upgrade is worth it; if you are launch-bound, it is not.

---

## 4. A100 → H100: the clear upgrade

### 4.1 What changes

| Property | A100 80GB SXM [[accelerators/nvidia-a100-80gb-sxm4]] | H100 SXM [[accelerators/nvidia-h100-sxm]] | Ratio |
|---|---|---|---|
| Memory | 80 GB HBM2e | 80 GB HBM3 | 1.0x |
| Bandwidth | 2,039 GB/s | 3,350 GB/s | 1.64x |
| Dense bf16 FLOPS | 312 TFLOPS | 989.5 TFLOPS | 3.17x |
| FP8 | emulated | native | — |
| TDP | 400 W | 700 W | 1.75x |

### 4.2 The price comparison

| Channel | A100 price | H100 price | Ratio |
|---|---|---|---|
| AWS on-demand | $3.4309 [[supply/aws-ec2-p4d-p4de-a100]] | $6.88 [[supply/aws-ec2-p5-h100]] | 2.0x |
| Azure on-demand | $4.096 [[supply/azure-nd-a100-v4]] | $12.29 [[supply/azure-nd-h100-v5]] | 3.0x |

H100 is 3.17x the FLOPS for 2.0-3.0x the price. At AWS on-demand the $/FLOP is
**1.58x better** on H100. At Azure it is **1.0x** (no advantage). The upgrade is
clearly worth it at AWS, marginally worth it at Azure.

### 4.3 The FP8 factor

A100 has no native FP8 tensor-core path ([[quantization/fp8-e4m3]]: "NOT native on
Ampere"). H100 has 1,979 dense FP8 TFLOPS. If your workload can use FP8, the effective
FLOPS ratio is 6.3x (3.17x bf16 × 2x fp8), and the upgrade is worth it at any price
ratio below 6.3x.

---

## 5. The migration checklist

Before upgrading, answer these questions with data:

1. **What is my binding constraint?** If it is memory capacity, only a capacity upgrade
   (H200, B200) helps. If it is compute, only a FLOPS upgrade (B200) helps. If it is
   launch overhead at batch 1, nothing helps ([[flops/custom-kernel-launch-overhead]]).
2. **What is my current utilization?** If below 50%, the idle power floor
   ([[benchmarks/us-volume-server-idle-dynamic-range-model-0-44-floor]]: 0.44 of max)
   means you are already paying for capacity you don't use. Upgrade the workload before
   upgrading the hardware.
3. **Does the new part change my $/Mtok?** Compute it: price_per_hour / (tok/s × 3600)
   × 1e6. If the answer is worse, the upgrade is a capacity decision, not a cost
   decision.
4. **Does my engine support the new part?** [[engines/vllm]] supports ROCm but with
   specific caveats. [[engines/tensorrt-llm]] is CUDA-only. Check
   [03-engine-selection.md](03-engine-selection.md) before assuming the new part works.
5. **Does my quantization format have a hardware path on the new part?** FP4 is
   Blackwell-only. FP8 is Hopper+. Check
   [04-quant-selection.md](04-quant-selection.md) before assuming the format carries over.

---

## Related documents

- [15-cost-per-token.md](15-cost-per-token.md) — the $/Mtok arithmetic used throughout.
- [01-hardware-selection.md](01-hardware-selection.md) — the per-workload hardware table.
- [02-flop-map.md](02-flop-map.md) §3.1 — the ridge points that determine what binds.
