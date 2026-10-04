# 19 — Production operations: what to run, what to alert on, what is unsolved

<!-- written 2026-10-04 against 3,290 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** you have a working inference deployment on
Kubernetes and you are about to hand it to users. Which knobs do you set, and which
things do you simply have to accept as unsolved? The records are unusually clear on
the shape of the answer: **most of the defaults in an LLM serving stack are tuned for
web services and are wrong for LLM serving**, and the ones that matter are not
obvious. This document names them, orders them by consequence, and states plainly the
three things the repo records as having no solution.

---

## 1. The one-paragraph version

An LLM serving replica is not a stateless web pod. Three defaults betray that:

1. **Autoscaling.** A CPU-based HPA never fires and a GPU-utilisation HPA is wrong in
   both directions simultaneously ([[gotchas/serve-hpa-cpu-autoscaling-fails-llm]],
   [[gotchas/ops-cpu-hpa-blind-to-llm-load]]). The signal that actually works is
   **queue depth**, which means the queue has to live somewhere you can see.
2. **Draining.** vLLM's SIGTERM handler **aborts** in-flight requests by default, and
   Kubernetes kills the pod 30 seconds after SIGTERM regardless
   ([[gotchas/ops-restart-aborts-in-flight-by-default]],
   [[gotchas/ops-termination-grace-period-kills-the-drain]]). A "zero-downtime
   restart" is not a default behaviour; it is three settings in the right order.
3. **Overload.** What happens when the queue fills is not one behaviour but three, and
   which one you get is a deployment choice you have to make deliberately
   ([[gotchas/ops-queue-fill-modes-are-engine-specific]]).

Two structural gaps have **no configuration that closes them** and are stated here
rather than papered over: there is **no per-tenant compute quota at any layer**, and
**gateway admission control fails open on stale telemetry** (§7).

---

## 2. Autoscaling: scale on queue depth, and put the queue where you can see it

### 2.1 Why CPU and GPU utilisation are both blind

- **CPU.** The serving process is an event loop plus an HTTP frontend; it does no
  model compute. Its utilisation tracks request-arrival bookkeeping, not serving work,
  so an HPA on CPU sits pinned at minimum replicas while requests queue
  ([[gotchas/serve-hpa-cpu-autoscaling-fails-llm]]).
- **GPU utilisation.** Continuous batching holds the SMs busy *by construction*. A
  single running request saturates the device about as effectively as fifty, because
  decode is memory-bandwidth-bound rather than throughput-bound. llm-d's own words:
  "During active token generation, GPU utilization can remain high whether a model
  server is lightly loaded or saturated. A utilization-based autoscaler cannot reliably
  distinguish spare serving capacity from an overloaded batch."
  ([[gotchas/ops-cpu-hpa-blind-to-llm-load]], restated as a plain serving rule in
  [[gotchas/serve-kv-cache-utilization-not-gpu-utilization]]).

So the same GPU-utilisation HPA is **too insensitive when saturated and
over-sensitive when idle**. That is the worst failure mode for an autoscaler: it is
wrong in both directions with one configuration.

### 2.2 What to scale on

The signal hierarchy, in increasing order of care, is recorded in
[[gotchas/ops-cpu-hpa-blind-to-llm-load]]:

| Rank | Signal | Why it is here and not above |
|---|---|---|
| 1 | Request count | Wrong unit. One 8,192-token prompt is 16x the prefill work of a 512-token one, and a counter rates them identically ([[gotchas/ops-ratelimit-token-vs-request-count]]). |
| 2 | Running-request count | A concurrency proxy. Blind to per-request size. |
| 3 | KV cache occupancy | The tightest physical constraint on decode, but it has no rate analogue for prefill. Also, the percentage is not headroom — see
   [20-observability.md](20-observability.md) §4.2. |
| 4 | **Queue depth / pool saturation** | Measures unmet demand directly, and is the only signal that exists **before** degradation reaches users. |
| 5 | Predicted latency vs SLO | Measures the objective rather than a proxy. This is what llm-d's SLO-aware path scales on ([[engines/llm-d]]). |

**Recommendation:** scale on a normalised queue-depth or saturation ratio, evaluated by
KEDA's Prometheus scaler creating an owning `autoscaling/v2` HPA. That needs no custom
controller and no Prometheus Adapter ([[gotchas/ops-cpu-hpa-blind-to-llm-load]]).
Where you must run one engine family only, KServe's Workload Variant Autoscaler reaches
the same place via inference-specific metrics ([[engines/niche-kserve]]).

**This only works if the queue is observable at all**, which is the whole reason §3
exists.

### 2.3 The telemetry-lag trade is real and unfixable by tuning

llm-d names the trade directly: its closed-loop utilization-detector is "highly
accurate but subject to telemetry lag ('thundering herd')", while the open-loop
concurrency-based detector avoids overshoot but is worse at bursty traffic
([[gotchas/ops-autoscaler-telemetry-lag-thundering-herd]]). Choose the detector for the
failure mode you can tolerate:

- **Cost-sensitive steady state, bursts absorb some lag** → closed-loop on utilisation.
- **Latency-sensitive, overshoot is unacceptable** → open-loop on queue depth.

The honest limit: the lag is a property of the *measurement path*, not the controller.
No amount of controller tuning removes it. Every documented mitigation moves latency
somewhere else.

### 2.4 Scale-down will not reach idle on its own

Warmup is asymmetric: "additions take a warmup time T_w to become ready (model load +
torch-compile); removals are immediate"
([[gotchas/ops-scale-down-never-reaches-idle]]). Because the demand signal stays high
for minutes after traffic is gone, the fleet cannot shrink back. Attack warmup
directly — persist the compile cache (see [21-deployment-topology.md](21-deployment-topology.md)
§3), and use a model-aware `startupProbe` on `/v1/models` rather than `/health`
([[gotchas/ops-health-endpoint-lies-about-readiness]]).

Node-level provisioning is outside the autoscaling loop entirely; over-budget replicas
are handled below the autoscaler ([[gotchas/ops-scale-down-never-reaches-idle]]).

---

## 3. Where the queue belongs

**Recommendation: put the central queue at the gateway, and keep only a small healthy
buffer inside the engines.** This is llm-d's stated principle — a buffer "just large
enough to ensure continuous batching engines never starve for work, but small enough
that the vast majority of queuing happens centrally in the EPP where priority and
fairness can be enforced" ([[gotchas/ops-queue-belongs-at-the-gateway]]).

The four reasons an engine-local queue is the wrong place:

1. **Scheduling regret is permanent.** "Once a request is dispatched to a model server
   local queue, the EPP cannot move it."
2. **No governance inside the engine.** Model servers "batch based on arrival or
   sequence length, not business priority" — so an interactive request waits behind a
   large offline batch with no way to express otherwise.
3. **Loss on restart.** Engine-local queues are in-memory and die with the process.
4. **No bound by default.** TGI's backend queue is an unbounded mpsc channel; vLLM's
   `max_num_queued_reqs` / `max_num_queued_tokens` both default to `None`
   ([[gotchas/ops-queue-fill-modes-are-engine-specific]]).

**Be honest about what central queuing buys.** llm-d says Flow Control "cannot remove
wait time when the system is over capacity. By enabling it, you make the explicit choice
to protect TPOT at the expense of queue time." It shifts TTFT; it does not eliminate it.

**And note the default is no central queue at all.** With flow control disabled (the
default), saturation means only negative-priority sheddable requests are rejected with
429 while "all other requests pass directly to the model servers"
([[gotchas/ops-queue-belongs-at-the-gateway]]). Enable the feature gate deliberately or
do not believe you have one.

---

## 4. Capacity: what happens when you hit the wall

### 4.1 Preemption, not rejection, is the default failure mode

When the KV cache fills, vLLM preempts running requests and re-prefills them from token
zero — it frees the KV blocks, sets the status to PREEMPTED, and sets
`request.num_computed_tokens = 0`
([[gotchas/ops-kv-cache-full-preempts-and-recomputes]]). The client sees a **latency
cliff, not an error**: error rate stays at zero and the request is still "served". The
victims are typically long-input / short-output requests.

This is the specific reason a request-count SLO passes while users are unhappy. Alert on
`vllm:num_preemptions` **before** latency, because preemption precedes the cliff
([[gotchas/obs-preemption-hides-latitude-from-throughput]]).

### 4.2 Noisy neighbour is not solved by a request quota

Capacity is consumed **per token**, not per request, and the engines' victim selection
is length-biased against long-prompt requests. One tenant sending massive prompts
degrades every other tenant's p50 and p99
([[gotchas/ops-noisy-neighbor-long-context-evicts-kv]]). A quota on request count
protects nobody. Rate-limit in **tokens** on both phases
([[gotchas/ops-ratelimit-token-vs-request-count]]) — but see §7.1, because that is not
the same as a compute quota.

### 4.3 Three queue-fill behaviours, one choice

| Fill mode | What the client sees | When you get it |
|---|---|---|
| Explicit shed | Immediate 429/503, visible in client error rate | You configured a bound |
| Silent unbounded growth | No error; client times out; TGI's mpsc response senders can hang ([[gotchas/ops-tgi-429-is-semaphore-not-queue]]) | vLLM stock (`None` bounds) or TGI |
| OOM | Node dies | Unbounded growth, eventually |

There is **no cross-engine convention** for what a full queue does
([[gotchas/ops-queue-fill-modes-are-engine-specific]]), so a mixed-engine fleet has no
uniform overload contract. Also note the status-code mismatch: vLLM's own overload
signal is **503, not 429** — `QueueOverflowError` and `MaxQueuedTokensError` both return
503 by its own docstring, so a "retry on 429" client policy sees 100% success while the
server sheds a fifth of its traffic
([[gotchas/ops-vllm-admission-rejects-503-not-429]]). TGI's 429 is worse still: it is
a concurrency **semaphore**, not a queue-depth signal, so `tgi_queue_size` can read zero
while 429s fire ([[gotchas/ops-tgi-429-is-semaphore-not-queue]]).

### 4.4 Losing the control plane silently downgrades the policy

llm-d defaults `failOpen: true`, so an EPP outage means client requests "are not
dropped and continue to be processed by the model server". But the model servers have
no central queue, no priority and no fairness — so a failOpen window is a **silent
policy breach**: per-tenant limits stop applying and priority inversion returns, with
nothing erroring ([[gotchas/ops-failopen-router-failure-opens-a-bypass]]).

Choose `failureMode` explicitly per deployment and alert on EPP leader changes, because
the window is silent. Note also that in the default active-passive EPP, scaling replicas
buys failover latency, not throughput — "only the single active leader replica handles
external processing requests".

Even with failOpen, measured success across four leader-teardown scenarios ranged 98.1%
to 100%, with residual errors attributed to socket teardown at the moment of pod
termination. **"Zero-downtime router failover" is not something the current evidence
supports** ([[gotchas/ops-failopen-router-failure-opens-a-bypass]]).

---

## 5. A zero-downtime restart: the three settings, in order

This is a checklist, not a description. Each item cites the record that makes it
load-bearing.

1. **`--shutdown-timeout N` on every vLLM replica.** vLLM's `shutdown_timeout` defaults
   to 0, and the shutdown handler branches on exactly that value: `mode = 'abort' if
   shutdown_timeout == 0 else 'drain'`. At 0 you get abort, and every rolling update
   fails requests that were seconds from completing. Size N as roughly the p99 request
   duration you want to allow to complete on drain
   ([[gotchas/ops-restart-aborts-in-flight-by-default]]).
2. **`terminationGracePeriodSeconds > --shutdown-timeout + preStop buffer`.** Kubernetes
   defaults the grace period to **30 s** while an LLM drain window needs minutes;
   SIGKILL lands mid-drain ([[gotchas/ops-termination-grace-period-kills-the-drain]]).
   The trap is the ordering — setting `--shutdown-timeout` without raising the grace
   period "looks configured and behaves exactly like no configuration at all".
3. **`/v1/models` for `startupProbe` and `readinessProbe`; `/health` for liveness
   only.** `/health` returns 200 before the model is loaded, so a liveness-style
   readiness probe routes traffic to a replica that will fail every request for minutes
   ([[gotchas/ops-health-endpoint-lies-about-readiness]]). There is no first-class fix;
   vLLM has no model-aware readiness endpoint (tracked upstream as vLLM #6073).
4. **Client-side retry with idempotency keys**, because in-flight requests are lost, not
   retried. An engine-core worker death loses every in-flight request with no retry, no
   replication and no request log ([[gotchas/ops-inflight-requests-are-lost-not-retried]],
   [[gotchas/serve-engine-crash-in-flight-requests-lost]]). The execution state is in
   GPU memory in a single process; when it dies the state is unrecoverable.

**What a drain cannot fix:** queued requests held in the EPP are lost on EPP restart
([[gotchas/ops-queue-belongs-at-the-gateway]]). There is no in-flight request
replication at any layer in these records.

---

## 6. Deploy safely: canary and rollout

Canarying inference is harder than canarying web
([[gotchas/ops-canary-on-inference-is-harder-than-web]]). A request's cost is not
knowable in advance ("driven by long input contexts and the unpredictable
autoregressive decode loop"), and the comparison is non-deterministic. Error-rate
comparison is meaningless because both versions return 200.

Where the routing layer does support version separation, use it — sgl-router's
`--pd-version-group-label` pairs a prefill worker only with decode workers sharing an
EndpointSlice label, "so KV never crosses a version boundary".

**Genuinely unsolved:** there is **no output-quality regression signal in the routing
layer**. Every mechanism addresses safety (don't mix incompatible versions) or fairness
(make the traffic split meaningful), and none of them detects that the new version got
worse ([[gotchas/ops-canary-on-inference-is-harder-than-web]]). A greedy-decode
spot-check against a known-good completion is the only oracle the repo records
(see [18-reliability.md](18-reliability.md) §2.3).

---

## 7. The honest unsolved items

### 7.1 There is no per-tenant compute quota at any layer

This is the single most important thing to know before you promise a tenant isolation
guarantee. The chain, from the serving layer up:

- **The engine has authentication and essentially no authorization.** vLLM has one
  shared API key, no per-tenant identity, no scoping, no policy on which model or
  adapter a caller may reach ([[gotchas/ops-api-key-covers-only-v1-prefixes]]).
- **Router admission observes load without reserving it.** Request size is explicitly
  not part of admission: "buckets already select by input length and context capacity,
  and admission only observes load without reserving it"
  ([[gotchas/ops-gateway-admission-has-no-reservation]]).
- **The gateway's own limiter is per-instance in-memory.** N workers means N times the
  intended limit, and a restart resets every tenant's budget to full
  ([[gotchas/ops-wasm-ratelimit-state-is-per-worker-and-lost-on-restart]]).
- **The fairness metrics that exist are observability, not enforcement.**
  `llm_d_epp_program_aware_jains_fairness_index` and
  `program_aware_avg_wait_time_milliseconds` report; they do not meter a token budget
  ([[gotchas/ops-ratelimit-token-vs-request-count]]).
- **llm-d's per-band `maxBytes` / `maxRequests` are queue-capacity limits between
  priority bands, not per-tenant compute quotas** — the same record says so explicitly.
- **Upstream does not claim to have solved it.** The Gateway API Inference Extension
  roadmap still lists "Fairness and priority between workloads within the same
  criticality band" as **future work**
  ([[gotchas/ops-ratelimit-token-vs-request-count]],
  [[gotchas/ops-noisy-neighbor-long-context-evicts-kv]]).

**What this means for you:** you can enforce a *rate* in tokens, and you can enforce a
*queue bound* per priority band. You cannot enforce a share of the GPU. A single
request's peak footprint is unbounded within any token rate you set.

### 7.2 Gateway admission fails open on stale telemetry

sgl-router's reported metrics are `None` without a fresh, complete report — never zero
— "and such limits fail open; the in-flight count is always known"
([[gotchas/ops-gateway-admission-has-no-reservation]]). So:

- The overload lands **precisely when the control plane is least able to observe it**.
- The only reliably enforced limit is the router-local in-flight counter.
- "Limits are absolute caps; they do not default to capacities reported by the engine",
  so an unconfigured limit is not a derived one.

**Mitigation:** treat router admission as a load-shedding knife-edge, not a quota. Alert
on `llm_d_epp_flow_control_stale_endpoints`, which counts "candidate endpoints whose
metrics are missing or older than staleness threshold" — the observable that admission is
currently failing open on. Note the record's honest status: this is a **contested**
record drawn from a document the project marks as a target design, so verify the
specifics against your build.

The compensating property is that the system is work-conserving — Flow Control "never
artificially throttles traffic if GPUs have spare capacity" — so the failure mode is
burst overshoot rather than persistent refusal. But overshoot on a KV-cache-bound
system is the expensive direction, because recovery runs through preemption and
re-prefill ([[gotchas/ops-kv-cache-full-preempts-and-recomputes]]).

### 7.3 A multi-replica router multiplies your admission limit

Adding a second router replica makes overload *worse*: each replica independently admits,
the effective limit is N times what you configured, and the burst arrives at the worker
all at once ([[gotchas/ops-gateway-admission-has-no-reservation]]).

### 7.4 The metrics that drive autoscaling can themselves be wrong

A pre-set `PROMETHEUS_MULTIPROC_DIR` makes vLLM's gauges carry the previous process's
values across restarts — `vllm:num_requests_running`, `vllm:num_requests_waiting` and
`vllm:kv_cache_usage_perc` all read high when they should read zero. Do not set the
variable and let vLLM manage its own temp directory
([[gotchas/ops-stale-multiprocess-metrics-dir]]). This is the most dangerous class in
this set because it corrupts the **signal** rather than the service: every symptom is
indirect, and an autoscaler built on the corrupted gauge holds replicas up indefinitely.

### 7.5 Routing state has no failure story

Two independent, documented gaps: the router reads a worker's forwarding protocol **once
at registration** and never re-reads it, so a worker that stops serving its registered
protocol is not detected and the breaker opens on live traffic
([[gotchas/ops-router-circuit-breaker-blind-to-h2c-downgrade]]); and disaggregated
deployments have no liveness check before the KV pull, so a dead prefill worker yields
a client-side 500 while the stranded KV is only recovered by a silent recompute
([[gotchas/ops-pd-stranded-kv-and-silent-recompute]]). Set `kv_load_failure_policy`
explicitly and know which one you have.

---

## 8. The operational checklist

Before serving users:

1. Replace any CPU/GPU-utilisation HPA with a queue-depth or saturation rule.
2. Enable central flow control, or accept that you have no central queue.
3. Set both vLLM queue bounds to real numbers.
4. Rate-limit in tokens, not requests, in shared state outside the gateway workers.
5. Set `--shutdown-timeout` **and** raise `terminationGracePeriodSeconds` above it.
6. Probe readiness on `/v1/models`, not `/health`.
7. Put an allowlisting reverse proxy in front of vLLM — see
   [22-safety-and-security.md](22-safety-and-security.md) §1.
8. Alert on `num_preemptions`, queue depth, and TTFT/TPOT p99 — see
   [20-observability.md](20-observability.md).
9. Add client-side retry with idempotency keys.
10. Do not set `PROMETHEUS_MULTIPROC_DIR`.

---

## 9. What the records cannot answer

1. **No base rate for any of these failures.** Every operational gotcha is a single
   report from a single deployment. There is no way to weight §4 by likelihood.
2. **No measured admission-control accuracy.** The contested record's numbers come from
   a design document, not a benchmark.
3. **No cost attribution per tenant.** [[gotchas/serve-cost-attribution-gap]] records
   this as a known unsolved gap: continuous batching and prefix caching make per-request
   cost highly variable, and there is no standard method.
4. **No cross-engine overload contract.** Because each engine's fill mode differs, a
   mixed fleet has no uniform behaviour to test against.

---

## Related documents

- [20-observability.md](20-observability.md) — which metrics to alert on, and the traps.
- [21-deployment-topology.md](21-deployment-topology.md) — where the gateway sits and
  what a cold start costs.
- [22-safety-and-security.md](22-safety-and-security.md) — the auth boundary and tenant
  isolation granularity.
- [18-reliability.md](18-reliability.md) — silent wrong output and thermal degradation.
- [17-deployment-shape.md](17-deployment-shape.md) — the interconnect decision.