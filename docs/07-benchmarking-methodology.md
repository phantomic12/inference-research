# Benchmarking methodology and its pitfalls

Every number in `data/benchmarks/` is a measurement somebody took, under
conditions they had to write down. This doc is about the conditions: which
omissions make a number meaningless, and which mistakes this repo has actually
recorded as gotchas because they happened in the wild.

Read this before quoting any throughput or latency figure from this repo, and
before adding a new benchmark record.

The governing rule is in [AGENTS.md](../AGENTS.md): a benchmark is only useful
with its methodology. The `benchmark` schema forces `metric`, `value`, `unit`,
`methodology` and `measured_by`; everything below is about filling those fields
in a way that survives contact with another engine.

## 1. Name the metric, and say whose it is

The single most common way a tok/s number misleads is that nobody says whether it
is what one user experiences or what the whole machine produces. Those two
differ by the concurrency, and the repo has a directly measured pair:

- [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]] — MPT-7B on 1x A100 40GB at
  batch size 1: 57.6 decode tokens/s experienced by one user. At batch 1,
  per-user equals aggregate, so the number is unambiguous.
- [[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]] — same model, same GPU, same
  engine, batch size 64: 12.5 tokens/s per user.
- [[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]] — same run: 800 output
  tokens/s aggregate across all 64 in-flight requests.

So one hardware configuration, one model, one engine: aggregate throughput rises
13.9x (57.6 -> 800) while the number the actual user feels **falls 4.6x**
(57.6 -> 12.5). Any comparison that mixes the two conventions is meaningless,
and in this case the error is not even conservative in one direction.

The schema encodes the distinction in the `metric` enum:

| metric | meaning |
|---|---|
| `tok_s_per_user` | decode tokens/s one request experiences |
| `tps_aggregate` | tokens/s summed over all in-flight requests |
| `decode_tok_s` | usually per-request at batch 1; confirm in `methodology` |
| `prefill_tok_s` | prompt-processing rate |
| `ttft_ms` | time to first token |
| `itl_ms` | inter-token latency, or end-to-end time in this repo's records |
| `quality` | accuracy / perplexity delta vs a baseline |
| `memory_gb` | resident footprint |

Two live examples where the same deployment produced both conventions, and
quoting either alone would be wrong:

- [[benchmarks/vllm-qwen38-nvfp4-gb300-nvl72-interactivity]] — 180 generated
  tokens/s per user on a GB300 NVL72 cluster.
- [[benchmarks/vllm-qwen38-nvfp4-gb300-nvl72-tps-per-gpu]] — 5000 tokens/s per
  GPU aggregate on the same cluster.

These are two points on one concurrency curve, not one configuration. The
record notes say so explicitly, and that framing is the point: a "5000 tok/s per
GPU" headline and a "180 tok/s per user" headline describe the same rig under
different load.

Same trap with hardware counts. [[benchmarks/vllm-qwen35-nvfp4-gb200-nvl72-tps-per-gpu]]
is 25000 tokens/s **per GPU across a rack**, and
[[benchmarks/trtllm-llama31-8b-fp8-h100-1k1k-tps]] is 14991.62 tokens/s aggregate
on **one** GPU. Dividing one by the other produces a nonsense ratio.

## 2. Offline/aggregate benchmarks hide the latency you will actually get

MLPerf Inference splits scenarios into Offline (all samples in one batch, no
latency ceiling) and Server (a latency constraint per query). The repo's MLPerf
records are all Offline, and several say so in their `unit` field:

- [[benchmarks/mi355x-mlperf-v6-0-llama2-70b-87gpu-offline-tokens]] — 1,042,110
  output tokens/s at cluster scale, 87 GPUs across 11 nodes. This is a throughput
  ceiling, not a user-facing number.
- [[benchmarks/arc-pro-b70-mlperf-v6-1-llama2-70b-offline-tokens]] — 2464.98
  output tokens/s aggregate on 4x Intel Arc Pro B70.
- [[benchmarks/gaudi2-mlperf-v4-0-llama2-70b-offline-tokens]] — its own notes
  record that comparing this to a newer MI355X result gives an apparent 13x gap,
  then explain why that reading is wrong: three MLPerf versions apart, different
  quantization regimes (Gaudi 2 w8a8 FT vs MI355X WMXFP4), and different memory
  capacity per card. A cross-version MLPerf comparison is not a hardware result.

The same caveat applies to the closed-loop setups. The paired AMD/NVIDIA records
here ([[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]] and
[[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput]]) are explicitly
labelled "closed-loop, infinite-rate client fed requests at an infinite rate: no
delay between messages, no latency constraint". That maximizes throughput and
discards the latency distribution entirely.

Latency at batch 1 is a different regime again, and the same AMD/NVIDIA pair
shows the expected inversion: MI300X is 10x better on end-to-end time-to-last-token
at batch 1 ([[benchmarks/mi300x-vllm-llama31-70b-fp8-tp8-ttlt-batch1]] vs
[[benchmarks/mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-ttlt-batch1-h100]]) while
being behind on aggregate throughput at high batch. That is the signature of a
bandwidth-bound decode path, and it is the whole reason you cannot pick hardware
from one number.

## 3. Pin the thing that changed, or the comparison is not an experiment

An engine version bump, a driver bump, or a different quantization name is a
different experiment. This repo has four separate records that exist only
because someone pinned the commit:

- [[gotchas/vulkan-q4km-speed-regression]] — one llama.cpp commit (`adc5dd92`)
  regressed Vulkan Q4_K_M decode from 42.12 to 36.38 tok/s while prompt
  processing stayed flat. Bisected, so the number is attributable.
- [[gotchas/vllm-rdna4-rowwise-fp8-kernel-auto-selected]] — v0.28.0 opened a
  capability gate that made RDNA4 select a slower FP8 kernel. Same hardware, same
  model, 35.45 tok/s on v0.27.1; the whole difference is one predicate in vLLM.
- [[gotchas/spec-decode-silently-downgrades-cudagraphs]] — one flag
  (`--attention-backend`) moves decode from 55.2 to 47.5 tok/s (+16%) purely by
  silently downgrading CUDA graph capture mode. A single logger warning announces
  it and names neither the cost nor the alternative.
- [[gotchas/fa4-num-splits-ignored-sm90]] — vLLM 0.30.0's FA4 kernel ignores
  `num_splits` on SM90: 48% slower decode than v0.26.0 at batch 1, with TTFT
  completely flat at 61 ms. Results stay correct, so nothing fails.

That last one is the shape to internalise: **the regression was visible only in
decode throughput, invisible in TTFT, and invisible to any correctness check.**

The same applies on the quantization side.
[[gotchas/same-named-quant-is-not-a-controlled-comparison]] is the clearest
statement in the repo that named levels are not budgets: llama.cpp's per-tensor
mixed-precision rules mean two models quantized at "Q4_K_M" land at 4.75 and
4.30 bpw respectively. Comparing across models at a matched *name* is not a
controlled experiment. Normalize to matched bpw.

## 4. A single sweep tells you about one point, not a curve

A benchmark run at one concurrency level cannot tell you whether a configuration
is good or merely impressive. [[gotchas/eagle-prefix-cache-last-block-drop]] is
the clearest example: EAGLE/MTP speculative decoding costs 6-10 points of
prefix-cache hit rate (97.5% -> 86.9-91.7%), and on one hybrid layout the
recompute is large enough to cost 30-40% of batch throughput — but only on
prefix-reusing workloads. The record's own workaround is to measure hit rate with
and without speculation before enabling it, i.e. sweep the axis.

The GB300 pair in section 1 is the same lesson in the other direction: 5000
tok/s/GPU and 180 tok/s/user are both correct, and only the sweep between them
tells you which one your deployment lives at.

For vendor-published curves, the repo records the shape explicitly rather than
the headline. [[benchmarks/silex-llama33-70b-4xh100-tp4-tps-200x200]] is
~7,000 TPS "peak at 500 concurrent users" — and its notes flag both that the
figure is an author's "~" approximation read off a chart image, and that the
same post's H100-vs-A100 gap conflates architecture with form factor because the
A100 used PCIe rather than SXM. Do not quote the ratio.

## 5. Correct output does not mean correct measurement

This is the failure class that motivates the whole doc. Most of it is silent.

### Silent numerical corruption
- [[gotchas/llamacpp-rocm-ppl-explosion-silent-corrupt]] — a ROCm commit moved
  WikiText-2 perplexity from 7.72 to 3024 on a Q4_0 model. No error, no crash.
  The reporter found it by automating the metric, not by reading chat output.
  Note what it does *not* prove: the same value moved on another machine, so it
  is not evidence of a hardware or driver defect.
- [[gotchas/marlin-w4a8-fp8-silent-corrupt-sm121a]] — `VLLM_MARLIN_INPUT_DTYPE=fp8`
  on sm_121a produces a server that starts normally and emits repeated `</think>`
  at temperature 0. No crash, no NaN assert.
- [[gotchas/cc-mode-uva-view-garbage-output]] — on an H100 NVL in Confidential
  Computing mode inside a TDX guest, vLLM's V2 runner serves token-id-0 garbage
  with no warning anywhere; the same model is correct under plain transformers on
  the same GPU.
- [[gotchas/flashinfer-bf16q-fp8kv-spec-decode-corrupts-swa]] — FP8 KV cache plus
  speculative decoding plus a 50-of-60-layer sliding-window model on SM100 gives
  errors up to ~1.8 confined to single tokens.
- [[gotchas/fp8-kv-cache-cudagraph-python-scatter-corruption]] — an FP8 KV write
  path written in plain Python is not graph-safe; the server boots, accepts a
  1M-token window, and generates pure noise, while `--enforce-eager` looks fine.
- [[gotchas/hip-rocm-corrupt-output-vulkan-is-reference]] — same machine, same
  build, byte-identical flags: HIP output is wrong, Vulkan is right, on gfx1151.

The pattern: **CUDA graphs, kernel selection, and quantization paths can all be
broken while every output check passes.** A benchmark that reports throughput
without a quality gate over the same configuration is measuring the speed of a
possibly-wrong model.

### Silent accuracy loss that looks like a win
- [[gotchas/mtp-plus-prefix-caching-accuracy-loss]] — MTP speculative decoding
  combined with prefix caching costs ~20% internal classification accuracy.
  Nothing errors; the server is simply faster and wrong. The four-way ablation
  (plain / +prefix / +MTP / +both) is what localizes it.

### Silent loss of losslessness
- [[gotchas/spec-decode-greedy-diverges-on-quantized-target]] — draft-model
  speculative decoding is *not* lossless at temperature 0 when the target model
  is quantized. If your benchmark compares outputs to a reference, the
  comparison itself may be invalid.

### A warning that is not a capability statement
- [[gotchas/nvfp4-marlin-warning-blames-gpu-for-weight-only-checkpoint]] —
  vLLM's "Your GPU does not have native support for FP4" fires on a *weight-only*
  NVFP4 checkpoint, where a W4A4 cutlass kernel is inapplicable regardless of the
  silicon's capability. The record's workaround is to test the predicate the
  kernel-selection path actually uses
  (`torch.ops._C.cutlass_scaled_mm_supports_fp4`) rather than reading the warning.

If you see this quoted as evidence about hardware, the citation chain is broken.

## 6. Benchmark harnesses lie about what they measured

- **Chart-read numbers.** [[benchmarks/perplexity-llama2-70b-h100-fp8-tp2-bs128-tps-per-gpu]]
  is 767 tok/s/GPU with `reproducible: false` because the engine build and clocks
  are not pinned and the figures come from the publisher's own images rather than
  a released data file. Its notes also record that the baseline is Perplexity's
  own A100 fleet, not a vendor-supplied number.
- **Streaming flags change latency.** [[benchmarks/vllm-qwen35-nvfp4-gb200-nvl72-tps-per-gpu]]
  notes that `--stream-interval 100` buffers output in 100-token chunks, which
  inflates apparent per-token latency. Do not use that record for ITL/TPOT.
- **`llama-bench` does not see allocator state.** [[gotchas/vulkan-suballoc-fragmentation-decode-cliff]]
  records decode falling 40.8 -> 8.9 tok/s between 98K and 131K context purely from
  suballocation fragmentation, with the fix being an env var. The shape of the
  cliff — clean at short context, catastrophic at long — is what gives it away.
- **TTFT scales with tensor parallelism; per-token latency does not.**
  [[benchmarks/mpt7b-a100-bs1-ttft-ms]] records MPT-7B going 46 -> 34 -> 26 ms
  across 1x/2x/4x A100 while Llama-2 70B gains far less going 4x -> 8x, because
  communication overhead eats it. Quoting one without the other misleads.
- **Speculation changes the metric definition.**
  [[benchmarks/llama-cpp-mi300x-deepseek-v3-671b-q4-decode-tok-s]] is explicitly
  the `tg256` test at batch 1 and single stream — that is the one-user case, not a
  serving number, and the paired
  [[benchmarks/llama-cpp-mi300x-deepseek-v3-671b-q4-prefill-tok-s]] is `pp512`.
- **Repeats matter and are not always available.**
  [[benchmarks/llamacpp-radeon-r9700-gfx1201-q1-0-decode-tok-s]] reports the mean of
  two interleaved runs (61.97, 61.34) on the same base commit, against unpatched
  master at 29.69 and 29.11. Single-shot decode numbers from a shared machine are
  worth very little; where the source gives only one run, say so.

## 7. When the harness itself is the variable

Two config flags are load-bearing enough to be benchmark methodology:

- **Resizable BAR.** [[gotchas/vulkan-rebar-off-collapses-amd-decode]] — with
  ReBAR/SAM off in BIOS, Vulkan decode on an RX 7900 XTX collapses 2-6x (13.89 t/s
  for 128 tokens). BIOS settings belong in the methodology field.
- **FP8 kernel auto-selection.** [[gotchas/vllm-rdna4-rowwise-fp8-kernel-auto-selected]]
  again: a version bump, not a hardware difference.

And one that is not a config at all — memory accounting.
[[gotchas/vllm-kv-cache-block-budget]] records that `--gpu-memory-utilization` is
a fraction of *total* VRAM, so allocation fails while memory is visibly free. The
number of blocks you actually got is in the startup log, and that log line is
part of the methodology.

## 8. Checklist for a new benchmark record

1. **Which convention?** `tok_s_per_user` or `tps_aggregate`. If aggregate, state
   the concurrency. If per-user, state the batch size.
2. **What is the baseline, and is it like-for-like?** Same quantization, same
   scenario class (Offline vs Server), same harness version.
3. **Is the engine version pinned** to a commit, not a release name? Same for the
   driver, and for any env var the result depends on.
4. **Does the output pass a quality check under the same configuration?** If the
   result is speed-only, cite the record that establishes the output is
   coherent, or mark it as unverified.
5. **Is it a chart read or a counter read?** Set `reproducible` accordingly and say
   so in `notes`.
6. **`measured_by`** — `vendor`, `third_party`, or `self`. A vendor number on open
   tooling is still `vendor`; see
   [[benchmarks/vllm-qwen35-nvfp4-gb200-nvl72-tps-per-gpu]], which says exactly
   that.
7. **Every number in `notes` cites a record or a source id.** No bare figures.

## Terms with no coverage yet

These concepts show up in real benchmark methodology but this repo has no record
that discusses them. Do not cite this doc as if it did.

- TPOT as a distinct reported metric — the repo uses `itl_ms` and
  `tok_s_per_user`, and "time per output token" appears in no record's own
  definition. [[benchmarks/mpt7b-a100-bs1-ttft-ms]] derives a per-token figure
  inline but does not record it as a metric.
- Goodput as a measured quantity. The term appears in
  [[engines/distserve]] and the Disagg paper, but no benchmark record reports a
  goodput measurement with its SLO.
- A long-context *quality* benchmark (RULER, NIAH) as a record. Long-context
  behavior is covered from the performance side
  ([[gotchas/gated-deltanet-decode-collapse-long-context]],
  [[flops/long-context-rag-prefill]]) but no retrieval-quality-at-length record exists.
- P99 / tail latency as a recorded measurement. The phrase appears in
  interconnect records and in two NVFP4 gotchas, never as a `metric` value.
- SM occupancy and warp scheduling as standalone terms; they appear inside kernel
  gotcha prose but have no record of their own.

## Related

- [AGENTS.md](../AGENTS.md) — the ingest contract, including why
  `vendor_claim` and `*_basis` fields exist.
- [SCHEMA.md](../SCHEMA.md) — the `benchmark` type, field by field.
- `docs/06-glossary.md` — the domain vocabulary these pitfalls are phrased in.