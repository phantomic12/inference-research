# 17 — Deployment shape: single GPU, multi-GPU, or multi-node

<!-- written 2026-10-04 against 2,592 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** your model fits on one GPU, but you need more
throughput or more capacity. Should you add a second GPU in the same node, or add a
second node? The answer is determined by the **interconnect fabric**, not by the model.
The repo computes the tensor-parallel crossover batch as a function of the fabric, and
the crossover moves by an order of magnitude depending on whether you are on NVLink,
PCIe, or Ethernet.

---

## 1. The one-paragraph version

Tensor parallelism (TP) splits each layer's weights across N GPUs and inserts an
all-reduce after each row-parallel layer. The all-reduce is pure data movement
(AI = 0) and its cost is **4·B·d bytes per rank per layer**, where B is batch and d is
hidden width. The crossover batch B* — the batch above which the all-reduce dominates
the weight read it competes with — is:

> **B* = P_layer × BW_link / (2 × d × BW_hbm)**

For Llama-2-7B on H100 SXM (BW_hbm = 3,350 GB/s), B* ranges from **184 on a single
25 GB/s NVLink 4 link to 3,318 on a full 450 GB/s per-GPU NVLink domain** — an 18x
spread on identical silicon ([[flops/tensor-parallel-allreduce]],
[[flops/sched-tensor-parallel-crossover]]). The fabric decides whether TP is free or
fatal.

---

## 2. Single GPU: when it is enough

### 2.1 The capacity test

A single GPU is enough when the model weights plus the KV cache for your target
concurrency fit in the GPU's memory. The KV cache per token is:

> 2 × num_kv_heads × head_dim × num_layers × kv_dtype_bytes

([[flops/kv-transfer]]). For a 70B GQA-8 model at bf16, that is ~1.1 MB per token per
sequence. At 100 concurrent sequences with 8k context, the KV cache alone is ~880 GB —
no single GPU holds it. But at 10 concurrent sequences with 2k context, it is ~22 GB —
fits on an 80 GB H100 with room for weights.

### 2.2 The throughput test

At batch 1, a single H100 SXM serves an 8B FP8 model at 14,991 output tok/s
([[benchmarks/trtllm-llama31-8b-fp8-h100-1k1k-tps]]). If your concurrency target is
below the ridge point (B_ridge = 295 for bf16 on H100, [[flops/decode-gemm]]), a
single GPU is memory-bandwidth-bound and adding a second GPU does not help — the
bandwidth is per-GPU, not per-node.

### 2.3 When single GPU is the right answer

- **Batch-1 or low-concurrency serving** of a model that fits on one card.
- **Latency-sensitive workloads** where the all-reduce latency would dominate.
- **Cost-sensitive deployments** where the second GPU's utilization would be too low
  to justify its cost ([[gotchas/idle-and-underbatched-fleet-can-cost-more-than-busy-fleet]]:
  energy per unit of work is U-shaped in the operating point).

---

## 3. Multi-GPU within a node: tensor parallelism

### 3.1 The crossover batch by fabric

[[flops/tensor-parallel-allreduce]] computes B* for Llama-2-7B on H100 SXM across
every fabric in the repo:

| Fabric | BW_link per rank | B* | Verdict |
|---|---|---|---|
| NVLink 4 (full domain, 18 links) | 450 GB/s | 3,318 | TP is free at any real batch |
| NVLink 4 (single link) | 25 GB/s | 184 | TP is free above batch 184 |
| PCIe 4.0 x16 | 31.5 GB/s | 232 | TP is free above batch 232 |
| PCIe 5.0 x16 | 63 GB/s | 465 | TP is free above batch 465 |
| AMD Infinity Fabric (MI300X) | 64 GB/s | 472 | TP is free above batch 472 |
| xGMI (MI250) | 50 GB/s | 369 | TP is free above batch 369 |

The record's conclusion: "the same model on the same HBM sees its tensor-parallel
crossover batch move by a factor of 18 purely on fabric."

### 3.2 When multi-GPU TP is worth it

- **The model does not fit on one GPU.** A 405B model in FP8 needs 405 GB — requires
  TP=8 on 80 GB H100s ([[benchmarks/trtllm-llama31-405b-fp8-h100-tp8-tps]]: 2,884.69
  tok/s on 8x H100).
- **You are above the crossover batch.** At batch 256 on NVLink 4, the all-reduce is
  negligible. At batch 256 on PCIe 4.0, it is 232/256 = 91% of the step time.
- **You have NVLink or better.** The all-reduce is a collective operation that
  benefits from the full domain. [[interconnect/nvswitch-4]] provides 450 GB/s per GPU
  in an 8-GPU domain; [[interconnect/nvswitch-5]] provides 900 GB/s in a 72-GPU domain.

### 3.3 When multi-GPU TP is not worth it

- **You are below the crossover batch.** At batch 1, the all-reduce is pure overhead.
  [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]] records 57.6 tok/s per user at
  batch 1; [[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]] records 12.5 tok/s per
  user at batch 64 — the per-user throughput drops 4.6x because the batch is spread
  across more GPUs.
- **You are on PCIe without NVLink.** The crossover batch is 232-465, which is above
  most serving configurations. TP on PCIe is a capacity play, not a throughput play.
- **Your model is small enough to fit on one GPU.** If the model fits, a single GPU
  is simpler, cheaper, and has no all-reduce overhead.

### 3.4 The NVLink domain boundary

[[flops/bw-nvlink5-nvl72-domain-topology]] records that the GB200 NVL72 connects 72
Blackwell GPUs in a single NVLink domain with 130 TB/s total bandwidth. Crossing the
72-GPU domain boundary falls back to InfiniBand or Ethernet, which has much higher
latency and lower bandwidth. The practical consequence: **TP within a 72-GPU domain is
a different regime from TP across nodes**, and the crossover batch formula changes
completely when you cross the boundary.

---

## 4. Multi-node: when TP is not enough

### 4.1 The scaling decision

When you need more than 8 GPUs (or more than 72 in an NVL72 domain), you must decide
between:

- **TP across nodes:** the all-reduce now crosses the node boundary. On InfiniBand
  NDR 400 ([[interconnect/nic-connectx7-ndr-400gbps]]: 50 Gbps per lane), the
  bandwidth is 4x below a single NVLink 4 link. The crossover batch rises to ~1,472 —
  above any real serving batch. TP across nodes is almost never worth it for decode.
- **Pipeline parallelism (PP):** each node holds a subset of layers. Communication is
  point-to-point (activations) once per stage, not a collective per layer
  ([[flops/sched-pipeline-parallelism]]). The cost is pipeline bubbles — if one stage
  is slower, the entire pipeline stalls.
- **Expert parallelism (EP):** for MoE models, each GPU holds a subset of experts.
  Communication is an all-to-all dispatch/combine, not an all-reduce
  ([[flops/sched-expert-parallelism]]). The cost is load imbalance.
- **Prefill-decode disaggregation:** prefill and decode run on separate nodes. The KV
  cache is transferred between them ([[flops/prefill-decode-disaggregation-kv-handoff]]).
  The cost is the KV transfer, which is AI = 0 (pure data movement).

### 4.2 The measured multi-node throughput

[[benchmarks/sglang-deepseek-v3-96xh100-output-tps-per-node]]: 22,300 output tok/s
per 8-GPU node (96x H100 total, 12 nodes) for DeepSeek-V3 671B MoE FP8. This is
expert-parallel across nodes with prefill-decode disaggregation.

[[benchmarks/sglang-deepseek-v3-gb200-nvl72-decode-tps-per-gpu]]: 7,583 decode tok/s
per GPU on a GB200 NVL72 (72 GPUs in one domain) for the same model. The NVL72
domain acts as a single GPU, so TP is free and EP is within the domain.

### 4.3 When multi-node is worth it

- **The model does not fit in a single node.** A 671B MoE in FP8 needs 671 GB —
  fits in a 72-GPU NVL72 (72 × 186 GB = 13.4 TB) but not in an 8-GPU node.
- **You need more throughput than a single node can provide.** At high concurrency,
  a single 8-GPU node saturates. Adding a second node with PP or EP increases
  throughput.
- **Your workload is MoE and you can use EP.** [[flops/deepep-expert-parallel-dispatch-combine]]
  records that EP requires Hopper+ and NVLink intranode plus RDMA internode. On
  A100 without NVLink, "DeepEP is not the path, and expert parallelism becomes
  bandwidth-bound on the slowest transport."

### 4.4 When multi-node is not worth it

- **Your model fits in a single node.** The networking overhead and operational
  complexity are not justified.
- **You are serving decode at low batch.** The all-to-all or all-reduce across nodes
  dominates at low batch.
- **Your interconnect is Ethernet.** [[interconnect/roce]] and
  [[interconnect/nic-spectrumx-ethernet]] provide scale-out but not scale-up. TP
  across Ethernet is not viable for decode.

---

## 5. The decision procedure

1. **Does the model fit on one GPU?** If yes, start there. Add GPUs only when
   concurrency demands it.
2. **Does the model fit in one node (8 GPUs)?** If yes, use TP within the node.
   Check the crossover batch for your fabric (§3.1).
3. **Does the model fit in a 72-GPU NVL72 domain?** If yes, use TP within the
   domain. The crossover batch is effectively zero.
4. **Do you need more than one node?** Use PP for dense models, EP for MoE models,
   or prefill-decode disaggregation. Do not use TP across nodes for decode.
5. **What is your interconnect?** If it is NVLink, TP is free. If it is PCIe, TP
   is a capacity play. If it is InfiniBand, TP across nodes is not viable for
   decode. If it is Ethernet, TP across nodes is not viable at all.

---

## 6. What the records cannot answer

1. **No measured crossover batch for any real model on any real part.** The crossovers
   in §3.1 are analytic (roofline-derived), and [[flops/decode-gemm]] warns real
   kernels cross *earlier* than the analytic figure.
2. **No attribution of step time to communication vs. compute in a multi-GPU step.**
   The records give the all-reduce cost model but not a measured breakdown.
3. **No multi-node PP bubble measurement for production workloads.**
   [[flops/sched-pipeline-parallelism]] gives the qualitative tradeoff but no
   production-scale measurement.

---

## Related documents

- [02-flop-map.md](02-flop-map.md) §3.7c — the TP crossover derivation.
- [08-cost-per-token.md](08-cost-per-token.md) — how the fabric affects $/Mtok.
- [18-reliability.md](18-reliability.md) — what breaks when you cross the node boundary.
