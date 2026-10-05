# 24 — Choosing a metrics surface: what each engine can actually measure

<!-- written 2026-10-05 against 3,848 records (295 metric_exposure records across 12
     engines, of which 274 are verified-present and 21 are verified-absent negatives).
     Every citation resolves to a data/ record; verified with an explicit checker
     against the committed tree. -->

**The decision this document serves:** you have chosen an engine (or you have not) and
you are about to write the scrape config. Which of the nine signals that matter can this
engine actually produce, what is its endpoint, and what do you build instead when the
answer is no? This is a different question from
[20-observability.md](20-observability.md)'s, which asks *which metric to alert on*.
20 assumes the metric exists. This document is about the ones that don't, and it is
built on the `metric_exposure` record set — **295 records that no hand-written document
in this repo cited before now.**

---

## 0. The decision rule, short form

**Pick the engine by the metrics you will have to live with, not by the ones you would
like.** Nine signals decide whether a fleet is operable:

    TTFT · TPOT · ITL · end-to-end latency · queue depth · preemption ·
    KV occupancy · speculative-decode acceptance · token counters

| engine | of 9 | endpoint | metrics on by default |
|---|---:|---|---|
| **vLLM** | **9/9** | `:8000/metrics` | yes |
| **SGLang** | **9/9** | `:30000/metrics` | **no — `--enable-metrics`** |
| TensorRT-LLM | 8/9 | **`:8000/prometheus/metrics`** | **no — `--return_perf-metrics`** |
| LMDeploy | 8/9 | `:23333/metrics` | yes |
| TGI | 4/9 | `:9000/metrics` | yes |
| Triton | 3/9 | `:8002/metrics` | yes |
| llama.cpp | 2/9 | `/metrics` + `--metrics` | no |
| Dynamo | 4/9 | `/metrics` on the **frontend** | partial |
| Xinference | 2/9 | `:9090/metrics` | no — exporter host/port required |
| Envoy AI Gateway | 3/9 | `:8080/metrics` | always attached |
| TEI | 2/9 | `:9000/metrics` | yes |
| Ollama | **0/9** | **none — no exporter at all** | n/a |

**The recommendation in one line: if your alert set needs preemption, vLLM and SGLang are
the only two engines in this repo that can give it to you, and TensorRT-LLM's is the
nearest miss.** Everything else in the table is negotiable; that one is not, because
preemption is the signal that precedes the latency cliff
([[gotchas/obs-preemption-hides-latitude-from-throughput]]).

**Four engines have no preemption metric at all**, and the ways they fail differ in a
way that matters — see §3.

---

## 1. Why this document exists, and what a record of an absent metric is for

The `metric_exposure` record set is built on a convention worth stating, because it is
the opposite of the repo's usual one. **A missing metric is recorded, not omitted.**
Each verified-negative record carries the metric name a reader *would go looking for*, so
the absence is **queryable** rather than being implied by a gap in a list. The records
say so themselves, in the same words each time:

> *"ollama_prometheus_any_metric is NOT a metric this engine registers — it is recorded
> here under the conventional name a reader would go looking for so the absence is
> queryable rather than implied by a gap in a list. ... Do not read this record's
> metric_name as something the engine exports; it is the name that is missing."*
> — [[metric-exposures/w3m-ollama-no-ollama-prometheus-any-metric]]

This matters more than it sounds. A gap in a list of 42 metrics is invisible; a record
whose `metric_name` is the missing one can be grepped, joined, and asserted against in
a test. **Twenty-one records in this repo are absences**, and four of them are the
finding in §3.

---

## 2. The endpoint is part of the decision, and two of them will waste your afternoon

**A scrape config is wrong if the path is wrong, and a wrong path returns something
that looks like a working one.** TensorRT-LLM is the case that bites hardest:

| engine | endpoint | the trap |
|---|---|---|
| vLLM | `:8000/metrics` | the default everyone memorises |
| SGLang | `:30000/metrics` | server port, **not** 8000 |
| **TensorRT-LLM** | **`:8000/prometheus/metrics`** | **`/metrics` is a *different endpoint*, and it serves JSON** |
| TGI | `:9000/metrics` | |
| Triton | `:8002/metrics` | the Triton core port, not the model port |
| llama.cpp | `/metrics` + `--metrics` | **off by default;** router mode needs `?model={model_id}` |
| LMDeploy | `:23333/metrics` | |
| TEI | `:9000/metrics` | `--prometheus-port`, default 9000 |
| Xinference | `:9090/metrics` | **only when an exporter host/port is set** |
| Envoy AI Gateway | `:8080/metrics` | reader always attached |
| Dynamo | `/metrics` on the **frontend** (port 8000) | **not on the standalone router** |
| Ollama | **nothing** | no exporter, no endpoint, no metric |

**The TensorRT-LLM trap, in the record's own words:** `/prometheus/metrics` is the
exposition endpoint, "**NOT /metrics**, and the server's own `GET /metrics` serves a
JSON iteration-stats snapshot, so a scraper pointed at `/metrics` gets **JSON rather
than exposition text**"
([[metric-exposures/w3m-tensorrt-llm-trtllm-num-paused-requests]]). That is the worst
shape of failure available: a scraper does not error, it gets a 200 with a body it
cannot parse, and the resulting silence looks like "no traffic".

**Recommendation: put the endpoint in one place, per engine, and assert it in a test.**
A `curl -s <endpoint> | head -1` that does not begin `# HELP` or `# TYPE` is a failed
deployment, not an idle fleet. This is cheap and it is the single highest-value check in
this document.

**Three engines need a flag before anything is exposed at all**, and two of them fail
closed in a way that looks like an outage:

- **SGLang** — "**NOT ON BY DEFAULT**: nothing is exported unless `--enable-metrics`
  (`ServerArgs` default `False`), which is also the only thing that mounts `/metrics`
  on the server port" ([[metric-exposures/w3m-sglang-sglang-num-retracted-reqs]]).
- **TensorRT-LLM** — the collector "is only constructed and `/prometheus/metrics` only
  mounted when `--return_perf-metrics` is set (`llm_args` default `False`)"
  (same record).
- **llama.cpp** — `--metrics`, and in router mode the scrape needs `?model={model_id}`.

---

## 3. Four engines have no preemption metric, and the four failures are different

[20-observability.md](20-observability.md) §2.1 makes preemption the **tier-1 alert**,
on the grounds that it *precedes* the latency cliff. That makes its absence the most
consequential single fact in this document. Four engines have a verified-negative
preemption record, and they fail in three distinct ways:

| engine | what is missing | what exists instead | the failure mode |
|---|---|---|---|
| **TensorRT-LLM** | **any** preemption metric, counter or gauge | `trtllm_num_paused_requests` — a **level** | you can see requests are paused; you cannot count preemptions or the recompute they cause |
| **TGI** | any preemption **or eviction** counter | `tgi_request_failure{err="dropped"}`, `tgi_batch_concat{reason="backpressure"}` | you see the system **refused** work; you never see what that work would have cost |
| **Triton** | any preemption metric | nothing, in core or in the TRT-LLM backend | **nothing schedules sequences**, so nothing can observe preemption |
| **LMDeploy** | any preemption metric | nothing | Turbomind **does** preempt and you cannot see it — a porting gap, not a design one |

The four records are worth reading together, because the reasoning differs:

- **TensorRT-LLM** — "the collector's own class docstring enumerates every metric it
  creates and preemption is not among them; the nearest signal is
  `trtllm_num_paused_requests`, a level. So a TensorRT-LLM deployment **cannot count
  preemptions or the recompute they cause, which is exactly the saturation signal vLLM
  and SGLang both expose**"
  ([[metric-exposures/w3m-tensorrt-llm-no-trtllm-num-preemptions]]). Note also that its
  collector is an acknowledged vLLM fork — "Adapted from
  `vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py`", prefix swapped — so this
  is an omission in a fork that had the capability available to it.
- **TGI** — "the v3 backend can drop a request when its queue overflows and can shrink
  batches under backpressure, and both are observable ... but those are **ADMISSION
  outcomes**. The recompute cost of a preempted sequence is invisible, so the saturation
  story stops at 'the system refused work' **with no measure of what that work would have
  cost**" ([[metric-exposures/w3m-text-generation-inference-no-tgi-preemption-or-eviction-counter]]).
- **Triton** — "Nothing in Triton core schedules sequences, so it cannot observe
  preemption" — this one is structural and would not be fixed by adding a metric
  ([[metric-exposures/w3m-triton-inference-server-no-nv-inference-preemptions]]).
- **LMDeploy** — "**Turbomind DOES preempt under KV pressure and you cannot see it.**
  ... Since it is a vLLM fork whose scheduler is the same shape, the omission is a
  **porting gap rather than a design one**"
  ([[metric-exposures/w3m-niche-lmdeploy-turbomind-no-lmdeploy-preemptions]]).

**Recommendation.** If the tier-1 preemption alert is non-negotiable, that requirement
alone rules out TGI, Triton, LMDeploy and Ollama. TensorRT-LLM is the interesting case
because it is otherwise excellent (§2's endpoint being its only real scrape problem) —
and the substitute is genuinely weaker: a **level** tells you preemption is happening
*now*, which pages on a persistent condition, where a **count** tells you the rate. If
you must run TensorRT-LLM, alert on `trtllm_num_paused_requests` **sustained above zero**
and state plainly in the runbook that it cannot detect a burst.

### 3.1 SGLang is the fourth case, and it is a naming divergence, not a gap

`[[metric-exposures/w3m-sglang-no-sglang-num-preemptions]]` records
`sglang:num_preemptions` as absent — **and SGLang still exposes all nine.** Its word for
preemption is **retraction**, and it measures it more finely than vLLM does:

> "The capability is present; only the vLLM name is absent."
> — [[metric-exposures/w3m-sglang-no-sglang-num-preemptions]]

An instantaneous gauge plus three cumulative counters with an input/output token split:
`sglang:num_retracted_reqs`, `sglang:num_retracted_requests_total`,
`sglang:num_retracted_input_tokens_total`, `sglang:num_retracted_output_tokens_total`
([[metric-exposures/w3m-sglang-sglang-num-retracted-reqs]]).

**This is the failure mode to design against: a portable alert config that assumes vLLM's
metric names silently matches nothing on every other engine.** A single
`num_preemptions` alert copied from a vLLM dashboard is a rule that never fires, and a
rule that never fires is worse than no rule because it looks like coverage. Two of
SGLang's three verified negatives are exactly this shape — see
[[metric-exposures/w3m-sglang-no-sglang-request-queue-time-seconds]], where the metric
exists as `sglang:queue_time_seconds` and not under vLLM's name.

---

## 4. The full matrix, and the four signals with the most coverage damage

Measured against the record set, per engine. `—` means **no verified-positive record
exists**; the "absent" column names the verified negatives.

### 4.1 The nine, by engine

| signal | vLLM | SGLang | TensorRT-LLM | LMDeploy | TGI | Triton | llama.cpp | Dynamo |
|---|---|---|---|---|---|---|---|---|
| **TTFT** | `vllm:time_to_first_token_seconds` | `sglang:time_to_first_token_seconds` | `trtllm_time_to_first_token_seconds` | `lmdeploy:time_to_first_token_seconds` | **—** | **—** | **—** | `dynamo_frontend_time_to_first_token_seconds` |
| **TPOT** | `vllm:request_time_per_output_token_seconds` | `sglang:request_time_per_output_token_seconds` | `trtllm_time_per_output_token_seconds` | `lmdeploy:time_per_output_token_seconds` | `tgi_request_mean_time_per_token_duration` | **—** | **—** | **—** |
| **ITL** | `vllm:inter_token_latency_seconds` | `sglang:inter_token_latency_seconds` | **—** | `lmdeploy:iter_token_latency` | **—** | **—** | **—** | **—** |
| **E2E** | `vllm:e2e_request_latency_seconds` | `sglang:e2e_request_latency_seconds` | `trtllm_e2e_request_latency_seconds` | `lmdeploy:e2e_request_latency_seconds` | `tgi_request_duration` | `nv_inference_request_duration_us` | **—** | `dynamo_frontend_request_duration_seconds` |
| **Queue** | `vllm:num_requests_waiting` (+`_by_reason`) | `sglang:queue_time_seconds`, `sglang:num_queue_reqs` | `trtllm_num_requests_waiting` | `lmdeploy:num_requests_waiting` | `tgi_queue_size` | `nv_inference_pending_request_count` | **—** | `dynamo_frontend_queued_requests` |
| **Preemption** | `vllm:num_preemptions` + per-request histogram | retraction family (§3.1) | `trtllm_num_paused_requests` (level only) | **—** | **—** | **—** | **—** | **—** |
| **KV occupancy** | `vllm:kv_cache_usage_perc` | `sglang:kv_cache_memory_usage_gb` + available/evictable/used | `trtllm_kv_cache_utilization` + 6 block gauges | `lmdeploy:gpu_cache_usage_perc` | **—** | **—** | **—** | **—** |
| **Spec-decode** | 4 `vllm:spec_decode_*` | 6 `sglang:spec_*` | `trtllm_speculative_config_info` (config only) | 7 `lmdeploy:spec_decode_*` | **—** | **—** | 4 `llamacpp:spec_decode_*` | **—** |
| **Token counters** | `vllm:generation_tokens` | `sglang:generation_tokens_total` | `trtllm_generation_tokens_total` | `lmdeploy:generation_tokens_total` | `tgi_request_generated_tokens` | **—** | `llamacpp:tokens_predicted_total` | **—** |

### 4.2 Four signals, ranked by coverage damage

**TTFT is missing from four engines, and two of those are the ones you would choose for
ease.** TGI and Triton have no time-to-first-token metric of any kind:

- **TGI** — "TGI can report queue time, inference time and time-per-token, but **not the
  latency a user perceives before the first token**. A TTFT SLO on TGI has to be
  computed client-side or approximated by joining the `x-queue-time` and
  `x-inference-time` response headers"
  ([[metric-exposures/w3m-text-generation-inference-no-tgi-time-to-first-token]]).
  And the near-miss substitute **cannot** stand in: "**`tgi_request_mean_time_per_token_duration`
  cannot stand in for it either — it is `inference_time / generated_tokens`, so it
  INCLUDES prefill**, which is a different quantity from vLLM's
  `request_time_per_output_token_seconds` as well."
- **Triton** — the trap is the name. `nv_inference_first_response_histogram_ms` "is close
  enough in name to be a trap: for a streaming LLM backend it measures **time to the
  first delivered CHUNK, not the first decoded token**, and for a non-streaming backend
  it is essentially the whole request. **A reader who assumes it is TTFT builds a
  dashboard that silently measures something else**"
  ([[metric-exposures/w3m-triton-inference-server-no-nv-inference-time-to-first-token]]).
  This is the same class as 20's "presence in the output is not evidence the value is
  live" — a plausible name attached to a different quantity.

**KV occupancy is missing from four engines, which makes capacity planning the real
damage.** TGI, Triton, llama.cpp and Ollama have no cache-occupancy metric:

- **llama.cpp** — "Cache reuse IS observable as `llamacpp:prompt_tokens_cached_total`, so
  you can compute a hit rate — but **occupancy is not exposed anywhere**, so a
  multi-tenant deployment **cannot tell how close to the context limit it is**"
  ([[metric-exposures/w3m-llama-cpp-no-llamacpp-kv-cache-usage-perc]]). Reuse rate and
  occupancy are different questions; only one of them is answerable here.
- **TGI** — "**a TGI deployment cannot be capacity-planned from KV occupancy**", and the
  nearest signals "are both batch-shape numbers rather than cache occupancy"
  ([[metric-exposures/w3m-text-generation-inference-no-tgi-kv-cache-usage]]).
- **SGLang is the best of the nine here**, and the reason is worth stealing:
  `sglang:kv_available_tokens`, `sglang:kv_evictable_tokens` and `sglang:kv_used_tokens`
  **separate free from merely-evictable**, which no ratio does — see 20 §4.2 on why a
  percentage is not a headroom signal.

**llama.cpp has no latency metric of any kind**, and its own record is the broadest
negative in the set:

> "the whole metric set is throughput and slot occupancy. There is **no TTFT, no TPOT, no
> ITL, no queue time, no KV-cache utilization, no preemption**. Per-request timing DOES
> exist — the timings object on the chat/completions response carries `prompt_ms`,
> `predicted_per_token_ms`, `predicted_per_second` and `cache_n` — but that is **response
> data returned to the client, not an exported series, so it cannot be aggregated by a
> scraper and TTFT has to be measured client-side**."
> — [[metric-exposures/w3m-llama-cpp-no-llamacpp-time-to-first-token-seconds]]

**Ollama has nothing to scrape.** `server/routes.go` "registers no `/metrics` handler at
all, there is no `prometheus_client` dependency, no histogram, no counter, no gauge —
there is nothing to scrape. Every one of the eight LLM metrics is therefore absent, and
so is any other metric"
([[metric-exposures/w3m-ollama-no-ollama-prometheus-any-metric]]). Its only timing
visibility "arrives only at the end of the stream, so it cannot be aggregated without
**every client logging every response**". The README's five observability integrations
(Opik, OpenLIT, Lunary, Langfuse, HoneyHive) are **third-party** — "that OTel is the
THIRD PARTY's, not Ollama's". And `/api/ps` shows residency, not KV state
([[metric-exposures/w3m-ollama-no-ollama-api-ps-kv-state]]).

**ITL is the narrowest of the nine** — only vLLM, SGLang and LMDeploy have it.
TensorRT-LLM's absence is "real but narrow: TPOT covers per-request decode rate, and
what is missing is the per-step gap that reveals **a single stalled iteration averaged
away across a request**"
([[metric-exposures/w3m-tensorrt-llm-no-trtllm-inter-token-latency-seconds]]). That is
precisely the failure 20 §3 tells you ITL exists to catch — so on TensorRT-LLM you are
choosing TPOT and giving up stall detection.

---

## 5. What would change these recommendations

| change | which recommendation flips | record to re-read |
|---|---|---|
| SGLang adds a `num_preemptions` alias | §3.1 becomes "SGLang and vLLM share a name", and the portability trap in §2 narrows | [[metric-exposures/w3m-sglang-no-sglang-num-preemptions]] |
| TensorRT-LLM adds a preemption **count** | TensorRT-LLM becomes 9/9 and the §0 recommendation can include it | [[metric-exposures/w3m-tensorrt-llm-no-trtllm-num-preemptions]] |
| LMDeploy ports vLLM's preemption counter | Its record calls the omission "a porting gap rather than a design one", so this is a merge, not a redesign | [[metric-exposures/w3m-niche-lmdeploy-turbomind-no-lmdeploy-preemptions]] |
| TGI exposes prefill duration separately | Its TTFT becomes derivable from `x-inference-time` without an estimate | [[metric-exposures/w3m-text-generation-inference-no-tgi-time-to-first-token]] |
| Triton core gains a token-shaped metric family | Its §0 row moves from 3/9 toward the middle of the table | [[metric-exposures/w3m-triton-inference-server-no-nv-inference-time-to-first-token]] |
| Ollama ships a Prometheus exporter | The 0/9 row becomes a normal engine row; today it is a deployment-shape decision | [[metric-exposures/w3m-ollama-no-ollama-prometheus-any-metric]] |
| TensorRT-LLM mounts exposition at `/metrics` | The §2 trap disappears, and the JSON-vs-text failure mode goes with it | [[metric-exposures/w3m-tensorrt-llm-trtllm-num-paused-requests]] |

---

## 6. What the records cannot answer

1. **No scraper configuration is recorded.** Every endpoint here is per-record and
   read from source, but nothing in the repo records a working `prometheus.yml`,
   `scrape_configs` block, or relabeling rule. The paths are known; the working config
   is yours to write and to test.
2. **No cardinality budget.** Label keys are recorded per metric — vLLM and SGLang carry
   `model_name`, `engine`, `tp_rank`, `pp_rank`, `moe_ep_rank` and more, and SGLang adds
   `dp_rank`, `priority` and anything in `extra_metric_labels` — but **no record states
   the resulting series count for a real multi-model fleet.** vLLM's
   "one extra time series per metric per label set" (20 §6.5) is a rate, not a budget.
3. **Nothing is measured.** Every record here is read from source code or docs, not
   scraped from a running engine. **No record in this set confirms that any series
   actually varies** — the "present but permanently zero" trap that 20 §4.3 documents
   for two SGLang gauges was found by a human reading a collector, and the equivalent
   sweep over the other 273 records has not been run. The §2 endpoint assertion is the
   cheapest available substitute and it catches only the grossest version.
4. **Bucket ladders are recorded for some metrics and not others**, and the two records
   that state the mechanism disagree in a way that will bite: vLLM requires
   `--custom-histogram-buckets` with per-boundary series cost; SGLang uses
   `get_histogram_conf_from_env` per histogram. TensorRT-LLM "take[s] their bucket
   boundaries as constructor arguments" and "**vLLM's boundaries do not carry over**".

---

## Related documents

- [20-observability.md](20-observability.md) — which metric to *alert on*, once you know
  it exists. Its tier-1 preemption alert is the requirement §3 defends.
- [19-production-operations.md](19-production-operations.md) — §2 there scales on queue
  depth, which §4 shows exists on 8 of 12 engines under 8 different names.
- [21-deployment-topology.md](21-deployment-topology.md) — where the metrics get
  exported from, once you have an endpoint.
- [18-reliability.md](18-reliability.md) — the failures no metric in this set catches.
- [03-engine-selection.md](03-engine-selection.md) — the choice §0 is arguing you should
  make with the metrics in hand.