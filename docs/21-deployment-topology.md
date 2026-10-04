# 21 — Deployment topology: control plane, data plane, and what a cold start costs

<!-- written 2026-10-04 against 3,290 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** where does the gateway go, which component owns
which decision, and what does it cost when a replica is not warm? The records are
consistent on the structure and unusually specific on the economics. Two findings carry
most of the decision: **a GPU pod's warm state is all node-local and dies together on a
reschedule**, and **vLLM's torch.compile cache defaults inside the container, so
Kubernetes repays the full JIT compile on every cold start forever unless you move one
environment variable.**

---

## 1. The one-paragraph version

Split the topology into two planes with different failure tolerances:

- **Control plane** — the gateway and the endpoint picker. Cheap, stateful, restartable.
  It owns identity, routing, priority, fairness and the queue. Its failure mode is to
  stop enforcing policy, which is why it fails **open** by default and why you have to
  alert on that.
- **Data plane** — the model servers. Expensive, stateful in GPU memory, and *not*
  restartable in milliseconds: they hold weights, KV cache, compiled graphs and a page
  cache that all live on one node.

The gateway's latency cost is **~2 ms** of data-plane overhead — one to two orders of
magnitude below the model call it fronts — so the reasons to avoid a gateway are not
latency. They are a **compatibility ceiling** (it can silently drop request fields and
return 200) and a **throughput ceiling** that can be severe
([[gotchas/dep-gateway-hop-cost-and-value]]).

And the cold-start economics: keep models warm, because keeping capacity warm is the only
way to be interactive, and scale-to-zero is a bill-shaped decision rather than a
latency-shaped one ([[gotchas/dep-serverless-scale-to-zero-only-pays-for-spiky-or-mostly-idle]]).

---

## 2. Where the gateway sits

### 2.1 The Kubernetes-native answer is two hops, not one

llm-d's topology: "inference requests will flow from an Envoy-managed Gateway to your
model servers via the llm-d EPP" — the Gateway does L7 policy, the Endpoint Picker picks
the pod ([[engines/llm-d]], [[gotchas/dep-gateway-hop-cost-and-value]]).

- **Envoy-family gateways** (Envoy AI Gateway) give model-aware routing, token rate
  limiting and observability, all three documented in one hop
  ([[engines/envoy-ai-gateway]]).
- **LiteLLM, Gloo and Apigee sit *above* that** as higher-level AI gateways: one API
  across 100+ providers, virtual-key budgets, fallback chains
  ([[engines/litellm]]).
- **sgl-router is SGLang-specific** and routes to SGLang instances only; its load
  balancing methods are ROUND_ROBIN, FOLLOW_BOOTSTRAP_ROOM, TOTAL_REQUESTS and
  TOTAL_TOKENS ([[engines/sglang-router]]).

**Recommendation: pick one layer per job rather than stacking several.** Choose the layer
by which job it is doing, and place the gateway where its decisions are cheap and its
state is available.

### 2.2 The gateway is a trust boundary that must be verified

A translating gateway rewrites requests schema-to-schema, and when the source schema has a
field the target schema cannot express, the field is dropped — **and the request returns
200** ([[gotchas/dep-gateway-silently-drops-params-with-http-200]]). Reported cases: a
JSON-schema `const` forwarded as `type: object`, and stop sequences that never arrive at
all. Nothing in any log says so.

Treat the gateway as a trust boundary that must be verified, not a transparent pipe: run
`--detailed_debug` and read the exact upstream payload, and assert on the fields that
change output rather than assuming passthrough.

### 2.3 Size the gateway's CPU limit against worker concurrency

An independent validation found "multi-second apparent latency from a Linux CFS
throttling trap when Envoy worker threads exceeded the container CPU quota", fixed by
pinning concurrency to the allocated CPU limit
([[gotchas/dep-gateway-hop-cost-and-value]]). This is a Kubernetes-specific gateway
failure that **looks like a network problem**.

Budget the control plane separately: at 2,000 AIGatewayRoutes the xDS payload exceeded the
default 4 MB gRPC message size and needed raising to 25 MB, and route count has a
practical ceiling from per-route configuration complexity.

### 2.4 The router is a stateful memory consumer

The prefix-cache `HashTrie` that makes prefix-aware routing possible has **unbounded
memory usage**. A reported deployment configured a 2 GB limit and the router pod kept
restarting, each restart dropping in-flight requests, resetting all prefix affinity and
forcing a fresh scrape window ([[gotchas/dep-router-prefix-tree-ram-is-the-router-oom]]).

Size the router's memory limit for **steady-state traffic volume, not concurrent
requests** — long-document workloads hold far more prefix entries per request than
short-request ones.

### 2.5 Two routing failures that look healthy

- **A parse miss looks healthy.** The router's stats parser recognises exactly six
  `vllm:`-prefixed metric names and silently discards everything else; an unrecognised
  scrape returns an **all-zero stats object that is truthy**, so load-aware routing
  degrades to a random pick with no warning anywhere. Every backend reports healthy, the
  scrape logs normally, and traffic flows
  ([[gotchas/dep-router-metric-name-parse-miss-looks-healthy]]). Verify the routing
  signal end to end once per backend at deploy time.
- **The first request of a session latches a bad pick.** When a request's matched prefix
  is shorter than `prefix_min_match_length`, prefix-aware routing falls back to a
  lowest-QPS average; two equal vLLM backends and eight concurrent agent sessions in a
  burst left half the fleet idle while half the sessions queued
  ([[gotchas/dep-prefixaware-subthreshold-latches-a-bad-first-pick]]). Prefer least-in-flight
  computed from engine stats over a windowed completion rate.

Standard load balancers cannot see KV cache state at all — they route blind to cache
locality and cause redundant prefill and higher TTFT
([[gotchas/serve-routing-cache-state-not-visible-to-load-balancer]]).

---

## 3. Cold-start economics

### 3.1 vLLM's torch.compile cache defaults inside the container — this is the headline

**By default vLLM writes the compile cache to a path inside the container
(`/root/.cache/vllm`), which is wiped every time the pod is recreated. So out of the box
you recompile on every single cold start and never see the benefit**
([[gotchas/dep-compile-cache-lives-in-container-and-is-wiped-every-pod]]).

The mechanism is sound — PyTorch Inductor caches generated kernels keyed by a hash over
architecture and layer shapes, dtype and quantization, compile flags, GPU compute
capability, and the vLLM and PyTorch versions. **Invalidation is correct: a stale cache
can never cause a failure, only a one-time slow start.**

**The fix is one environment variable, `VLLM_CACHE_ROOT`, pointed at a persistent volume
or host-local disk.** Measured on Qwen3-6-35B-A3B-NVFP4 on a DGX Spark (EKS Hybrid
Node), local NVMe, vLLM v0.22.1, average of 3 runs, process start to server ready:

| Phase | Default | With persisted cache + weight streamer |
|---|---|---|
| Weight load | 142 s | 59 s |
| torch.compile | **39 s** | **10 s** |
| Profiling / warmup | 136 s | 98 s |
| **Total** | **317 s** | **167 s (1.90x)** |

That is **150 s saved, 47% of total cold start**, from one variable with no change to the
model and no change to output ([[gotchas/dep-compile-cache-lives-in-container-and-is-wiped-every-pod]]).

The **honest residual**: profiling and warmup "is largely fixed engine overhead that the
two optimizations don't directly target — effectively the floor for this approach", at
98-136 s in the measured case. If you are near that floor, the next targets are graph
capture (54 s to 7 s by narrowing `--cuda-graph-sizes`, with the caveat that requests
outside the captured sizes lose throughput) and weight-load parallelism — **not more cache
tuning.**

Treat the cache volume as part of the deployment rather than an optimization: the same
volume is where weights and the flashinfer engine cache belong.

### 3.2 The weight-pull herd is superlinear in burst size

This is the one cold-start failure whose cost is **superlinear in replica count**, which
makes it qualitatively different from every other entry: a 5-minute cold start
replicated 50 times in *parallel* is much worse than 50 times sequentially
([[gotchas/dep-weight-pull-thundering-herd-is-superlinear-in-burst-size]]).

The mechanism: "when your autoscaler spins up 50 or 100 replicas simultaneously — which
is exactly what happens during a traffic burst or a new model rollout — every one of
those pods races to the same upstream source for the same hundreds of gigabytes of data
at the same time. Object storage has bandwidth limits and rate limits. This becomes a
traffic jam."

**A caching layer alone does not fix it**, for two separate reasons: contention persists
if "100 pods on 20 nodes all request the same model simultaneously, and your cache
processes them independently"; and the first pull still has no cache to hit, so you
still depend on upstream uptime, bandwidth and rate limits. That makes the first
deployment of a new model the worst case rather than the steady state — **the opposite
of what a cache is supposed to do.**

**The structure that works has three properties:**

1. **Own the source** — mirror weights to your own infrastructure at push time, so
   deployments have no runtime dependency on an upstream model hub.
2. **Tier the cache by distance** — node-local NVMe first, then a peer node over the
   in-cluster network, then a mirrored origin with parallel byte-range downloads.
3. **Single-flight the downloads** — at node level, pods waiting on the same model share
   one download; at cluster level, consistent hashing assigns each model file to one
   cache node so exactly one node fetches it from origin.

The claimed result is 2-3x faster cold starts — **treat that as a vendor claim, but the
mechanism is the mechanism.**

**These two mechanisms make each other worse.** Reactive autoscaling reacts to current
load, and the new pods it asks for then contend for the same bandwidth
([[gotchas/dep-weight-pull-thundering-herd-is-superlinear-in-burst-size]],
[[gotchas/serve-cold-start-model-loading-blocks-autoscaling]]).

### 3.3 A GPU pod's warm state is all node-local, and dies together

A GPU pod is bound to its device, so **any reschedule onto another node loses the
weights, the KV cache, the compiled graphs and the page cache together** — and re-warming
costs the full measured cold start
([[gotchas/dep-gpu-pod-rescheduling-destroys-all-warm-state]]). An LLM serving replica
is not a stateless app, so the ordinary Kubernetes failure response — delete the pod, let
the scheduler place a new one — is a multi-minute cold start.

Design for the reschedule rather than trying to prevent it: make the node-local artifacts
node-local *and durable* — weight cache, `VLLM_CACHE_ROOT`, engine/flashinfer cache — so
a pod that lands on a node which has served that model before starts warm-ish.

### 3.4 Cache-aware routing cannot rebuild its map after a restart

The KV placement view is built **incrementally from a live event stream**, so any router
or replica restart leaves it permanently wrong. Prefix-cache hit rates drop and never
fully recover, with no error and no unhealthy worker — SGLang's own RFC enumerates the
missing snapshot-and-replay mechanism
([[gotchas/dep-kv-aware-router-view-cannot-be-rebuilt-after-restart]]).

Treat a cache-aware router as having two modes — converged and starting — and **make the
starting mode observable yourself**: gate cache-aware routing on having seen a fresh
BlockStored event from the pool, not on pod readiness.

---

## 4. Scale-to-zero: a bill decision, not a latency decision

**Recommendation: answer one question first — is the traffic spiky enough that you can
absorb minutes of cold start on the burst edges, or is it interactive enough that every
request must land on warm capacity?**

If interactive, scale-to-zero is **not** the cost lever. A warm floor plus predictable
scheduling is, and you are buying dedicated capacity with a better cold-start profile.

The economics, from two independent facts:

1. **Cold start is dominated by weight transfer, not container boot.** The measured
   breakdown for a 7B on one L4 node: 61 s model download, 33 s weight loading, 52 s
   torch.compile, 54 s graph capture, 94 s init — 294 s total, with a 28 GB image taking
   roughly 236 s to pull on a node that did not have it. The "serverless cold start is
   fast" numbers in circulation are **image-restore or container-boot times**: containers
   boot in about a second, but that excludes global-scope imports and model loading.
2. **Keeping capacity warm is per-second billed**, and the knobs are described as a
   straight trade: a larger warm pool "will increase costs but reduce the chance that
   inputs will need to wait for a new container." Default scale-to-zero windows are
   60 s of maximum idle time.

**Practical break-even sits near 40-50% utilization**: below it, idle dominates and
scale-to-zero wins; above it, you pay a serverless premium for a benefit you are not
using ([[gotchas/dep-serverless-scale-to-zero-only-pays-for-spiky-or-mostly-idle]]).

If you do want scale-to-zero, buy it deliberately and **measure the phase, not the
total** — the diagnostic mapping is directly reusable: long *acquiring resources* means
waiting for capacity, long *pulling image* means a large image or replicas landing on
nodes without it, long *pulling weights* means large weights or cache misses. Warm on a
schedule (raise `min_containers` before a known peak, drop after) rather than constantly.

---

## 5. The topology decision procedure

1. **Do you have more than one engine process?** If no, skip the gateway. There is no
   routing decision to make and the hop is pure cost
   ([[gotchas/dep-gateway-hop-cost-and-value]]).
2. **Are tenants mutually untrusted?** If yes, the gateway owns identity and nothing else
   will — vLLM has authentication and essentially no authorization. See
   [22-safety-and-security.md](22-safety-and-security.md) §1.
3. **Which layer?** Kubernetes-native → Envoy-family Gateway + EPP. Multi-provider or
   multi-cloud → put LiteLLM or equivalent *above*, not beside.
4. **Where does the queue live?** At the gateway, with a small healthy buffer inside the
   engines — [19-production-operations.md](19-production-operations.md) §3.
5. **What is in the warm state, and where?** Weights, `VLLM_CACHE_ROOT`, engine cache,
   page cache — all node-local, all lost together on reschedule.
6. **What happens on a burst of N new replicas?** Single-flight the weight pull or accept
   a superlinear cold start.
7. **failOpen or failClose?** A decision, not a default — see §4 of
   [19-production-operations.md](19-production-operations.md).

---

## 6. What the records cannot answer

1. **No multi-tenant failure mode for the data plane.** Noisy-neighbour behaviour is
   recorded from the KV-cache side; there is no record of gateway- or EPP-level
   multi-tenancy in production.
2. **No measured gateway overhead for a *self-hosted* engine.** The ~2 ms figure is
   measured against a mock or vendor backend. The measurement *discipline* transfers
   (hold upstream, account, key, prompt and parameters constant; interleave with rotating
   order; discard warm-up; report medians with confidence intervals plus p95/p99) but the
   number does not.
3. **The 2-3x weight-cache speedup is a single vendor claim.** The mechanism is
   corroborated by two independent practitioner phase breakdowns; the multiplier is not.
4. **No cold-start measurement with disaggregated prefill/decode**, where the KV transfer
   would presumably add its own phase.

---

## Related documents

- [19-production-operations.md](19-production-operations.md) — the queue, admission
  control and restart behaviour this topology has to support.
- [20-observability.md](20-observability.md) — what the gateway and router export, and
  the two routing failures that look healthy.
- [22-safety-and-security.md](22-safety-and-security.md) — what the gateway's auth
  boundary actually covers.
- [17-deployment-shape.md](17-deployment-shape.md) — single GPU vs multi-GPU vs
  multi-node, decided by the interconnect.
- [18-reliability.md](18-reliability.md) — the failure modes this topology inherits.