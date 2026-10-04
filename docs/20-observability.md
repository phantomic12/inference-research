# 20 — Observability: which metrics to alert on, and which ones lie

<!-- written 2026-10-04 against 3,290 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** you have a serving deployment and a monitoring
stack. What goes in the SLO, what goes on the alert, and what should you refuse to
build a rule on at all? The records give a clear answer with a clear shape: **alert on
what the user experiences, diagnose with what the engine knows, and treat every
percentage as a trend rather than a threshold.** This document gives the exact metric
names per engine, and then the list of metrics that look load-bearing and are not.

---

## 1. The one-paragraph version

**TTFT and TPOT are the SLOs. Average tokens/sec is not.**

Throughput is a *capacity* number, not an *experience* number, and the two move in
opposite directions because a serving engine trades one for the other: raising
concurrency to hit a throughput target makes TTFT worse without moving the throughput
number at all, so the dashboard stays green while users report the service "feels
stuck" ([[gotchas/obs-serving-slos-ttft-tpot-not-tok-s]]).

- **TTFT** is what a user perceives before any output appears, and it is where
  queueing, prefill and detokenization all land. SGLang's docs are precise that for
  streaming requests TTFT ends when the first output arrives *after detokenization*,
  "which may contain no printable text" — so a detokenizer stall is a TTFT regression.
- **TPOT** is the rate at which a response streams once started.

Serve them from `vllm:time_to_first_token_seconds` and
`vllm:request_time_per_output_token_seconds` (or the `sglang:` / `lmdeploy:` /
`trtllm:` equivalents), keep ITL alongside for stall detection only, and treat queue
depth plus KV usage as the leading indicators that *explain* a breach
([[gotchas/obs-serving-slos-ttft-tpot-not-tok-s]]).

---

## 2. The alert set

### 2.1 Tier 1 — page on these

| Alert | Metric | Why this one |
|---|---|---|
| TTFT p99 | `vllm:time_to_first_token_seconds` | The first thing a user feels. Also absorbs queue time, so it is the SLO that fails first under overload. |
| TPOT p99 | `vllm:request_time_per_output_token_seconds` | Per-request basis, comparable across engines. See §3. |
| **Preemption rate** | `vllm:num_preemptions` (cumulative), `vllm:request_num_preemptions` (per-request histogram) | **Precedes** the latency cliff. Alerting on latency alone means you already lost the requests. |
| Queue depth | `vllm:num_requests_waiting` plus `vllm:num_requests_waiting_by_reason` (reasons sum to the total) | The only signal that exists before degradation reaches users. |
| Stale telemetry | `llm_d_epp_flow_control_stale_endpoints` | Admission control is currently **failing open** on this — see [19-production-operations.md](19-production-operations.md) §7.2. |
| EPP leader changes | control-plane metric | A failOpen window is silent; this is the only way to see it. |

On preemption specifically: alert on the preemption metric, **not on throughput**.
Aggregate output tok/s and request-success rate look normal or even *improve* while
users report stalling ([[gotchas/obs-preemption-hides-latitude-from-throughput]]).

### 2.2 Tier 2 — dashboard, not page

- Output throughput — as a **capacity constraint**, not a target.
- `vllm:num_requests_running` — a concurrency proxy, blind to per-request size.
- Prefix-cache hit rate — **on the token basis**:
  `rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])`, and never
  called a request hit rate ([[gotchas/obs-metric-names-that-look-per-request-are-not]]).
- `vllm:num_requests_waiting{reason="capacity"}` — the signal that means actual
  saturation, as distinct from other deferrals.
- Throttle-reason codes and p99 over a long window — see
  [18-reliability.md](18-reliability.md) §3.

### 2.3 Tier 3 — never build an alert on these

- **Average output tokens/sec** as an SLO ([[gotchas/obs-serving-slos-ttft-tpot-not-tok-s]]).
- **A KV cache usage *percentage* threshold** (§4.2).
- **`sglang:utilization` and `sglang:max_running_requests_under_SLO`** — permanently zero
  on current main (§4.3).
- **Triton's per-request histograms** for cross-engine comparison — they are
  per-model aggregates (§4.1).
- **TTFT on TGI** — the metric does not exist (§5.1).

---

## 3. ITL vs per-request TPOT: pick one and write down which

They are not the same statistic, and the projects say so explicitly
([[gotchas/obs-itl-vs-request-tpot-are-different-metrics]]).

- `vllm:inter_token_latency_seconds` records **one sample per streamed output event** —
  the wall-clock gap between successive outputs.
- `vllm:request_time_per_output_token_seconds` is one sample per request over the
  `(e2e - TTFT)/(n-1)` denominator.

**Recommendation:** report **TPOT** to users (it is the right basis for a "how fast does
this response stream" SLO and it is comparable across engines using that definition),
and alert on **ITL** for stall detection, because ITL does not amortise a stall over a
long response. Swapping them silently changes your SLO: an alert tuned on one does not
fire when the other degrades.

Never compare a TGI number to a vLLM number without checking the denominator, and never
compare a Prometheus histogram quantile against a `vllm bench serve` printed TPOT as if
they were the same measurement ([[gotchas/obs-itl-vs-request-tpot-are-different-metrics]]).

---

## 4. The metric traps

### 4.1 The same metric name shape hides different units of observation

vLLM and SGLang histograms are genuine per-request observations; **Triton reports
per-model aggregates**, and prefix-cache counters are **token counts, not request
counts** ([[gotchas/obs-metric-names-that-look-per-request-are-not]]). A p99 computed
from an engine histogram "looks nothing like the p99 a client measured, and nobody can
say which is right".

**Rule: before comparing or alerting on any metric across engines, write down its unit
of observation — per request, per iteration, per model, or per token.** For cross-engine
latency comparison, use only metrics that are per-request on *both* sides, which in
practice means vLLM, SGLang, LMDeploy and TensorRT-LLM, and **excludes Triton's
counters** ([[gotchas/obs-metric-names-that-look-per-request-are-not]],
[[engines/obs-triton-observability]]).

A second denominator trap in the same family: `vllm:iteration_tokens_total` is a
histogram of tokens per *engine step*, and its source carries a standing TODO that it
may change.

### 4.2 KV cache usage percentage is not a headroom signal

Three distinct problems are stacked under that one metric name
([[gotchas/obs-kv-cache-usage-percentage-is-not-a-headroom-signal]]):

1. **It is a block-occupancy ratio**, not a bytes or contiguity measure. vLLM's
   `block_pool.get_usage()` is literally `1.0 - (free blocks / total blocks)`.
2. **The same number means different things on different engines**, so two replicas of
   the same deployment on different backends need different thresholds.
3. **The rename was not clean.** vLLM renamed `vllm:gpu_cache_usage_perc` to
   `vllm:kv_cache_usage_perc`, and the follow-up was a commit fixing the metric's
   multiprocess handling — so historical dashboards may be reading a metric that no
   longer means what the panel says.

**Recommendation:** alert on the **count**, not the percentage, where the engine
publishes one — `vllm:num_requests_waiting{reason="capacity"}` separates "no capacity"
from other deferrals and is the signal that actually means saturation. Where you only
have a ratio, treat it as a **trend, not a threshold**, and calibrate per engine, per
backend and per prefix-cache setting. The LMDeploy docs themselves say their PyTorch
default "is useful for comparing request load when routing across replicas" — i.e. it is
a comparative signal, not an absolute one.

SGLang's available / evictable / used triple is the best available substitute for a real
headroom measure, because it separates free from merely-evictable.

### 4.3 Two SGLang metrics that look load-bearing are permanently zero on main

`sglang:utilization` and `sglang:max_running_requests_under_SLO` are **documented, present
in the metrics collector, exposed on `/metrics` with a plausible HELP string, and
permanently zero** — the producer is gone but the definition was never removed
([[gotchas/obs-sglang-utilization-metric-is-dead]]).

**Recommendation: do not build an autoscaling or alerting rule on either.** On SGLang,
key on `sglang:num_queue_reqs` plus `sglang:token_usage` (or the per-pool variants) and
on TTFT/TPOT percentiles.

Verify by scraping `/metrics` and confirming the series is *present but constant* —
**presence in the output is not evidence the value is live.** A rule that never fires is
worse than no rule, because it looks like coverage. This is worth recording as a
category of trap: a metric family can be fully present in the collector, documented in
the source docstring, and completely dead.

### 4.4 Generic container metrics are actively misleading

Dashboard green on CPU, memory and request count while the service is degraded:
CPU utilisation is low because the GPU does the compute, memory utilisation is high but
that is *expected* because the KV cache is supposed to fill GPU memory, and request
count does not see a single long-context request
([[gotchas/serve-observability-generic-container-metrics-misleading]]). Export the
engine's own metrics instead.

### 4.5 Prometheus multiprocess state can survive a restart and corrupt the signal

A pre-set `PROMETHEUS_MULTIPROC_DIR` means `vllm:num_requests_running`,
`vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` all carry the previous
process's values ([[gotchas/ops-stale-multiprocess-metrics-dir]]). This corrupts the
autoscaling signal specifically — see [19-production-operations.md](19-production-operations.md)
§7.4.

---

## 5. Per-engine surfaces, with the honest gaps

### 5.1 vLLM — the reference surface

The most complete of any engine: 39 metric names in `vllm/v1/metrics/loggers.py` plus 7
in the spec-decode metrics module, all prefixed `vllm:`
([[engines/obs-vllm-observability]]).

**Do not read metric names from memory.** `vllm:gpu_cache_usage_perc`,
`vllm:num_requests_swapped` and `vllm:cpu_cache_usage_perc` are all gone or renamed, and
`vllm:time_in_queue_requests` was a duplicate. Metrics are collected in the API-server
process, not the engine-core process, which is exactly the mechanism behind §4.5.

**Two settings that will bite you:**

1. **Override the histogram buckets for any sub-second target.** vLLM's default
   `request_latency` family bottoms out at 0.3 s, which cannot resolve a 200 ms SLO —
   use `--custom-histogram-buckets`. Each added boundary costs one extra time series per
   metric per label set ([[gotchas/obs-serving-slos-ttft-tpot-not-tok-s]]).
2. **Metrics are off by default in SGLang, not vLLM** — see §5.2.

### 5.2 SGLang — same latency names, off by default, two dead metrics

All core latency metrics exist with the same names as vLLM
(`sglang:time_to_first_token_seconds`, and so on), but **metrics require an explicit
`--enable-metrics`**; nothing is exposed at `/metrics` without it
([[engines/obs-sglang-observability]]). OpenTelemetry is **not bundled** — the init path
raises `RuntimeError('opentelemetry package is not installed!!!')` unless the SDK and
OTLP exporter are installed.

**Best in the repo for disaggregated serving:** the KV-transfer family —
`sglang:kv_transfer_speed_gb_s`, `sglang:kv_transfer_latency_ms`,
`sglang:kv_transfer_bootstrap_ms`, `sglang:kv_transfer_alloc_ms`.

And do not forget §4.3: two of its most useful-looking gauges are dead.

### 5.3 TGI — no TTFT, no KV cache usage

Measured by grepping the router source: **no occurrence of `time_to_first` anywhere**, and
no `kv_cache` or `cache_usage` metric in the repository at all
([[gotchas/obs-tgi-has-no-ttft-metric]], [[engines/obs-tgi-observability]]).

TGI does expose queue time, inference time and time-per-token, so the *decomposition* it
offers is not the *split* you need. **A TTFT SLO on TGI cannot be served by its
metrics.** Compute TTFT client-side, or approximate it from headers — `x-queue-time`
plus the prefill share of `x-inference-time` — with the explicit caveat that it is an
estimate. Its backpressure signal is `tgi_queue_size` and `tgi_request_queue_duration`,
**not the 429 rate**, which is a concurrency semaphore
([[gotchas/ops-tgi-429-is-semaphore-not-queue]]).

### 5.4 TensorRT-LLM — vLLM's metric names, different buckets

An acknowledged fork of vLLM's collector, with TensorRT-LLM's extra inflight-batching and
multi-tier KV counters ([[engines/obs-tensorrt-llm-observability]]). **All seven latency
histograms take their bucket boundaries as constructor arguments** — you have to pass
buckets yourself, and vLLM's boundaries do not carry over. There is **no preemption
metric at all**, which means §2.1's most important alert cannot be built here.

### 5.5 Triton — request-shaped, not LLM-shaped

Rich, well-documented request-level metrics with per-family type selection via
`--metrics-config` ([[engines/obs-triton-observability]]), but **Triton itself has no
TTFT, TPOT, KV cache usage, preemption or token-throughput metric.**
`nv_inference_first_response_histogram_ms` measures time to the first *response*, which
for a streaming LLM backend is not TTFT. `nv_inference_pending_request_count` is the
honest queue metric.

### 5.6 Tracing: real in four of eight engines

OpenTelemetry is present in vLLM, SGLang, TensorRT-LLM and Triton; absent in the rest,
with no warning ([[gotchas/obs-tracing-status-per-engine-honest-negatives]]). A trace of
one request is **one span in vLLM and three spans in Triton**, and neither is per-token.

**Recommendation:** pick the engine's tracing honestly. For a single-request latency
breakdown on vLLM, read the attributes on the one `llm_request` span — that is
sufficient and cheap. Disaggregated prefill/decode is where traces differ most and where
they are most worth having.

---

## 6. The observability checklist

1. Two SLOs: TTFT p99 and TPOT p99, both as percentiles over **per-request** histograms.
2. Throughput as a capacity constraint, never as an experience target.
3. Alert on `num_preemptions` **before** you alert on latency.
4. Alert on queue depth, not batch occupancy — continuous batching makes the engine look
   permanently busy and the queue is not empty, it has been absorbed into the batch.
5. Override histogram buckets for any sub-second target.
6. Write down the unit of observation for every metric before comparing it to another.
7. Scrape `/metrics` and confirm every series you rely on is *varying*.
8. Check your engine can measure the SLO at all — TGI cannot, llama.cpp has no latency
   metric, Ollama has neither.

---

## 7. What the records cannot answer

1. **No measured detection latency for any alert.** These records establish which
   signal moves first; they do not establish how quickly a paging alert would fire.
2. **No SLO attainment data for any deployment.** Nothing in the repo records a real
   TTFT or TPOT percentile distribution from production traffic.
3. **No guidance on cardinality.** The bucket-override cost is stated
   ("one extra time series per metric per label set") but no budget is given for a
   large multi-model fleet.

---

## Related documents

- [19-production-operations.md](19-production-operations.md) — the autoscaling signal
  depends on this metric set being trustworthy.
- [18-reliability.md](18-reliability.md) — the failures no metric in this set catches.
- [21-deployment-topology.md](21-deployment-topology.md) — where the metrics get
  exported from.
- [17-deployment-shape.md](17-deployment-shape.md) — the ridge-point reasoning behind
  "decode is bandwidth-bound".