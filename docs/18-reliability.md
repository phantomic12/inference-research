# 18 — Reliability: what actually goes wrong in production serving

<!-- written 2026-10-04 against 2,592 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** you have a working deployment. What will break,
what will degrade silently, and what should you monitor? The gotcha records in this repo
are the evidence base — 47 records of things that went wrong in real deployments, found
by people who were not looking for them. The ordering below is by **what you can
detect**, not by severity, because the failures you can detect are the ones you can
prevent.

---

## 1. The one-paragraph version

Production serving failures fall into three categories, and they have different
detection profiles:

1. **Silent wrong output** — the server starts, serves every request, emits no error,
   and returns confident wrong text. These are the most dangerous because throughput
   monitoring does not catch them. Several are *faster* when broken, so A/B testing
   selects for the bug.
2. **Silent performance degradation** — the server serves correctly but slower. The
   cause is usually thermal, power-delivery, or clock-state, and it is invisible unless
   you monitor the specific signal.
3. **Measurement traps** — the deployment is fine but the number you are using to make
   decisions is wrong. These are the most common and the least recognized.

---

## 2. Silent wrong output: the failures throughput monitoring cannot catch

### 2.1 The pattern

Every record in this section shares one property: **the server starts, serves every
request, emits no error, no warning and no NaN, and returns confident wrong text or
wrong logits.** The gotcha records are explicit that these were found only because
someone ran a cross-check that a deployment would not run.

### 2.2 The records, grouped by trigger

**Quantization + kernel selection:**
- [[gotchas/nvfp4-marlin-bf16-garbled-output]] — **blocker.** NVFP4 + `--dtype bfloat16`
  on SM < 100 (RTX 4090 SM89, V100 SM70): loads, starts, serves, every response
  garbled. The FP16 path maps the 5-bit exponent directly; the BF16 path does not.
- [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]] — **blocker.** sm_121a + WNA16 INT4
  MoE + `VLLM_MARLIN_INPUT_DTYPE=fp8`: coherent-looking repetition loop, and the
  corrupted path is **~2.5% FASTER end-to-end, so throughput benchmarking selects for
  the bug.**
- [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]] — **major.** A checkpoint marked
  `W4A16_NVFP4` served with `--moe-backend flashinfer_b12x` runs the FP4xFP4
  micro-kernel. Silent wrong output. The record says explicitly: "Do not accept a b12x
  trace showing an FP4 kernel for a W4A16 checkpoint."
- [[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] — **major.** On B200, a
  compressed-tensors MXFP4 checkpoint with `input_activations: null` (i.e. W4A16)
  routes to a W4A4 kernel. Measured cost: 78.16 vs 70.58 on AIME/Math/GSM8K average
  against the unquantized 82.36 — a 7.58-point accuracy drop caused purely by kernel
  misrouting.

**KV cache + CUDA graphs:**
- [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — **blocker, silent
  corruption.** 8x A800-80GB (SM80, no native FP8 tensor cores) with
  `--kv-cache-dtype fp8_e5m2`, TP=8: the server runs and returns wrong output. The
  bf16 KV path writes through a fused CUDA op that is graph-safe; the fp8 path goes
  through plain Python that allocates a fresh tensor and does an advanced-index
  scatter, which under graph capture corrupts the write.

**Speculative decoding + attention:**
- [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] — **blocker.** 1x B200
  (SM100), vLLM 0.23.1rc1, serving a model with sliding-window attention (50 of 60
  layers SWA, window 1024) with `--kv-cache-dtype fp8_e4m3` and MTP num_speculative_tokens=7.
  Two independent defects: the spec-as-decode kernel mis-computes sliding-window
  attention once seq_len > window at certain alignments. Dropping any one of the
  three flags (spec decode, quantized KV, or the attention config) makes output clean.

**Platform-specific:**
- [[gotchas/cc-mode-uva-view-garbage-output]] — **blocker.** On an H100 NVL in CC mode
  inside a TDX guest, vLLM 0.29.0 with the default V2 model runner serves and every
  response is garbage (token id 0 repeated). `VLLM_USE_V2_MODEL_RUNNER=0` fixes it.
- [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — **major.** On gfx1151 (Strix
  Halo APU, ROCm 7.2.4), same machine, same build, byte-identical flags: the HIP
  backend is wrong and Vulkan is right.
- [[gotchas/hip-wrong-logits-when-prompt-exceeds-ubatch]] — **major, silent.** Any
  prompt longer than n_ubatch produces badly wrong logits on HIP/ROCm on gfx1151 —
  no error, no warning, no crash, and the numbers look plausible. Default n_ubatch
  is 512.
- [[gotchas/windows-hip-multi-gpu-silent-output-corruption]] — **blocker.** On Windows
  11 with a gfx1100 + gfx1201 pair, the HIP backend splits a model across both cards
  and returns fluent-but-nonsensical word salad with healthy-looking throughput.
  Building with `-DGGML_CUDA_NO_PEER_COPY=ON` makes the corruption disappear.
- [[gotchas/vulkan-rebar-on-causes-amd-device-lost]] — **blocker.** With SAM/ReBAR
  enabled, every decode on an AMD Radeon AI PRO R9700 fails with
  `vk::Queue::submit: ErrorDeviceLost`. Disabling ReBAR fixes the crash but
  re-introduces the slowdown in [[gotchas/vulkan-rebar-off-collapses-amd-decode]].

### 2.3 What to monitor

The records are unanimous that **none of these are caught by standard health checks**.
The monitoring that would catch them:

1. **Output quality spot-checks.** Run a fixed prompt with a known-good completion at
   a fixed interval and compare. [[gotchas/spec-decode-greedy-diverges-on-quantized-target]]
   records that greedy sampling with spec decode produces different text on a quantized
   target — so a greedy-decode spot-check is the cheapest oracle.
2. **Kernel selection logs.** [[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]] and
   [[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]] both produce log lines that name the
   kernel. Alert on any log line that names a W4A4 kernel for a W4A16 checkpoint.
3. **Perplexity regression.** [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]]
   records that llama-perplexity on ROCm/gfx1151 returns a PPL jump of two orders of
   magnitude. Treat a PPL jump of two orders of magnitude as a hard stop.
4. **Cross-backend comparison.** [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]]
   was found by running the same model on two backends. A periodic cross-backend
   comparison on a fixed prompt is the cheapest detector of backend-specific
   corruption.

---

## 3. Silent performance degradation: the failures health checks cannot catch

### 3.1 Thermal and power-delivery

- [[gotchas/hw-thermal-slowdown-halves-clocks-silently]] — **major.** Sustained-token-rate
  collapses mid-run with no error, no crash and no assertion. A batch that measured X
  tokens/s over a short window measures 0.5x over a long window. The GPU is
  thermally throttling.
- [[gotchas/dvfs-power-cap-moves-tail-latency-under-sustained-load]] — **major.** Mean
  and median latency look fine while the tail degrades. A deployment passes an SLO
  defined on p50 and breaks for users. The failures cluster in the long runs — after
  the node has been at full load for a while, or on the hottest day.
- [[gotchas/dgx-psu-depop-silent-degraded-throughput]] — **major.** After a PSU failure
  takes three of six PSUs offline, the system "continues to function, but at a reduced
  performance level." No crash, no error, no degraded-state alarm in normal monitoring.
  The node serves traffic, passes health checks, and simply delivers less throughput.
- [[gotchas/hw-power-brake-psu-instability-crash]] — **major.** nvidia-smi reports
  'HW Power Brake' active, meaning an external power brake assertion has been triggered
  by the system power supply.

**What to monitor:** `nvidia-smi` throttle-reason codes (SW Power Cap, HW Thermal
Slowdown, HW Power Brake), PSU population per node, and **p99 latency over a window
long enough to contain a real duty cycle** — the MLPerf submissions use
min_duration 2400000 ms, and [[gotchas/dvfs-power-cap-moves-tail-latency-under-sustained-load]]
notes that this node needs roughly 15 minutes to leave its ramp.

### 3.2 Clock state

- [[gotchas/unlocked-clocks-make-cross-part-comparison-meaningless]] — **major.** A
  fresh-process matmul canary reads 25.0 TFLOPS in a quiet window and 119.3 TFLOPS
  minutes later after a heavy workload — on the same machine. Clock parking plus slow
  ramp hysteresis means short-probe TFLOPS is not a state verdict.

**What to monitor:** lock clocks with `nvidia-smi --lock-gpu-clocks=<min>,<max>` for
any cross-part or cross-run comparison. Never use short-probe TFLOPS as a state
verdict.

### 3.3 Engine and framework regressions

- [[gotchas/fa4-num-splits-ignored-sm90]] — **major.** Decode on H100 (SM90) is 48%
  slower per token than on v0.26.0 at batch 1 and 36% slower at batch 4, while TTFT
  is flat. Results stay correct, so nothing fails. The fix is backend selection:
  `attention_backend=TRITON_ATTN` recovers ~70%.
- [[gotchas/vllm-rocm-fp8-cold-start-timeout]] — **major.** vLLM on gfx1100 with an FP8
  checkpoint aborts during startup on the engine-ready timeout (default 600 s);
  measured first init was 1474.26 s. Set `VLLM_ENGINE_READY_TIMEOUT_S=1800` for the
  first cold start; once the Triton cache is warm, init drops to ~49 s.
- [[gotchas/vllm-v1-memory-footprint-growth]] — **major.** Same model and flags: vLLM
  0.6.x ran at 12K context on 4x RTX 3070; 0.7.0 with V1 enabled cannot exceed ~3K.
  Re-tune for the V1 engine rather than carrying V0 values.
- [[gotchas/vllm-kv-cache-block-budget]] — **major.** "No available memory for the cache
  blocks." The KV budget is a first-class constraint and the error message names the
  fix (`--gpu_memory_utilization`, `--max-num-batched-tokens`, `--max-model-len`).
- [[gotchas/eagle-prefix-cache-last-block-drop]] — **major.** Multi-round and
  repeated-prompt workloads lose ~6-10 points of prefix-cache hit rate when
  speculative decoding is on. Measure hit rate with and without `speculative_config`.
- [[gotchas/spec-decode-silently-downgrades-cudagraphs]] — **major.** A single
  `logger.warning` announces a CUDA-graph downgrade; decode throughput 47.5 tok/s on
  FLASHINFER vs 55.2 on FLASH_ATTN (+16%), changing only `--attention-backend`.
- [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] — **blocker.** On a finetuned
  Qwen3.6 35B-A3B with MTP (num_speculative_tokens=2) plus `--enable-prefix-caching`,
  internal classification accuracy drops ~20% versus the identical setup without.

**What to monitor:** engine version and configuration in the deployment manifest.
Any version bump should trigger a re-benchmark and a re-tune of `--max-model-len` and
`--max-num-batched-tokens`. [[gotchas/vllm-v1-memory-footprint-growth]] is the clearest
example of a version bump silently changing the memory envelope.

---

## 4. Measurement traps: the numbers that lie

### 4.1 Power measurement boundary

- [[gotchas/board-power-vs-system-power-measurement-boundary]] — **major.** A
  joules-per-token figure is quoted without stating whether it is measured at the GPU
  package, the accelerator board, the node, or the rack. The number is then compared
  against another figure measured at a different boundary, and the comparison is wrong
  by 40-70%. The 'Where Does the Energy Go?' paper profiled 2x RTX PRO 6000 Blackwell
  and found that GPU-only telemetry misses 41-45% of system energy.
- [[gotchas/vendors-dont-publish-joules-per-token-estimation-error]] — **major.** The
  NVIDIA HGX H100/H200 datasheet publishes NO TDP or rack power figure at all. The
  only available figures are TDP (a thermal design power ceiling, not a measured power
  draw) and peak throughput (a best-case number at maximum batch size). Estimating
  joules-per-token from TDP carries 30-50% error.

**What to monitor:** always state the measurement boundary alongside any
joules-per-token figure. If you must estimate system energy from board power, apply a
1.4-1.7x overhead factor depending on the platform.

### 4.2 Idle power

- [[gotchas/idle-and-underbatched-fleet-can-cost-more-than-busy-fleet]] — **major.**
  Energy per unit of work is U-shaped in the operating point rather than monotonic, so
  the cheapest-to-run configuration is not the busiest one. An idle fleet can cost more
  than a busy fleet.
- [[benchmarks/h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-idle-node-draw]] records
  that an 8x H200 node draws **2,411 W at idle** (zero tokens/s).

**What to monitor:** utilization, not just throughput. A fleet running at 20%
utilization pays 44% of the energy bill for nothing
([[benchmarks/us-volume-server-idle-dynamic-range-model-0-44-floor]]: idle floor is
0.44 of maximum power for volume servers).

### 4.3 Carbon intensity

- [[gotchas/grid-carbon-intensity-by-region-inference-emissions]] — **major.** An
  emissions claim computed as joules-per-token × average grid carbon intensity
  produces a single number that is wrong by an order of magnitude depending on where
  the tokens are actually generated. The claim "inference emits X gCO2 per token" is
  meaningless without the grid region, the carbon intensity value (average or
  marginal), the time of day, the facility PUE, and the joules-per-token figure with
  its measurement boundary.

---

## 5. The reliability checklist

Before deploying, and at every version bump:

1. **Run a greedy-decode spot-check** against a known-good completion. This is the
   cheapest oracle for silent wrong output ([[gotchas/spec-decode-greedy-diverges-on-quantized-target]]).
2. **Log the kernel selection** and alert on any W4A4 kernel for a W4A16 checkpoint
   ([[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]],
   [[gotchas/mxfp4-w4a16-misrouted-to-w4a4-kernel]]).
3. **Monitor throttle-reason codes** (`nvidia-smi` → SW Power Cap, HW Thermal
   Slowdown, HW Power Brake) and PSU population per node
   ([[gotchas/dgx-psu-depop-silent-degraded-throughput]]).
4. **Report p99 latency over a 15-minute window**, not p50 over a 1-minute window
   ([[gotchas/dvfs-power-cap-moves-tail-latency-under-sustained-load]]).
5. **Lock clocks for any cross-part comparison** ([[gotchas/unlocked-clocks-make-cross-part-comparison-meaningless]]).
6. **State the measurement boundary** for every joules-per-token figure
   ([[gotchas/board-power-vs-system-power-measurement-boundary]]).
7. **Re-tune `--max-model-len` and `--max-num-batched-tokens`** after every engine
   version bump ([[gotchas/vllm-v1-memory-footprint-growth]]).
8. **Measure prefix-cache hit rate** with and without speculative decoding before
   enabling spec decode on prefix-heavy workloads
   ([[gotchas/eagle-prefix-cache-last-block-drop]]).
9. **Set `VLLM_ENGINE_READY_TIMEOUT_S=1800`** for the first cold start on ROCm
   ([[gotchas/vllm-rocm-fp8-cold-start-timeout]]).
10. **Do not accept a b12x trace showing an FP4 kernel for a W4A16 checkpoint**
    ([[gotchas/flashinfer-b12x-moe-runs-w4a16-as-w4a4]]).

---

## 6. What the records cannot answer

1. **No production-scale reliability study.** Every gotcha record is a single report
   from a single deployment. There is no base rate: the records cannot tell you how
   likely any of these is in your deployment.
2. **No multi-tenant failure mode.** All records are from single-tenant or
   single-workload deployments. Multi-tenant interference (noisy neighbors, KV cache
   eviction under memory pressure) is not recorded.
3. **No network failure mode.** The records cover GPU, engine, and framework failures
   but not the network failures (RDMA hangs, InfiniBand congestion, Ethernet packet
   loss) that dominate multi-node serving reliability.
4. **No persistent-state corruption.** All records are from stateless serving. A
   model that corrupts its own KV cache across requests, or a prefix cache that
   returns stale entries, would not be caught by any of the monitoring above.

---

## Related documents

- [05-known-traps.md](05-known-traps.md) — every gotcha record, severity-ordered.
- [17-deployment-shape.md](17-deployment-shape.md) — the multi-node failure modes.
- [15-cost-per-token.md](15-cost-per-token.md) — the idle-power cost trap.
