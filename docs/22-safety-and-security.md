# 22 — Safety and security, from the serving system's side

<!-- written 2026-10-04 against 3,290 records. Every citation resolves to a data/ record. -->

**The decision this document serves:** you are standing up an inference endpoint and
someone asks what safety and security controls it needs. This document stays at the level
of **serving behaviour** — where a filter runs, what it costs, what the isolation
boundary is, what the API leaks — and does not adjudicate what counts as harmful content.
The payload-content question belongs to a policy owner; the questions here are
engineering ones, and the records answer them sharply.

**The four headline findings, stated up front:**

1. **vLLM's `--api-key` is not a security boundary.** It authenticates only the `/v1`,
   `/v2`, `/inference` and `/cohere` path prefixes; `/invocations`, `/pause`,
   `/abort_requests` and `/update_weights` stay open
   ([[gotchas/ops-api-key-covers-only-v1-prefixes]]).
2. **There is no tenant isolation between requests inside one process.** The granularity
   that exists is MIG, a container, or a process — never a request
   ([[gotchas/sec-tenant-isolation-granularity-is-process-or-mig-not-request]]).
3. **Watermarks are a provenance signal, not a safety control, and they are fragile on
   the traffic most deployments actually serve**
   ([[gotchas/sec-watermark-provenance-is-not-a-compliance-control]],
   [[gotchas/sec-watermark-low-entropy-output-is-where-detection-dies]]).
4. **Returning logprobs turns your endpoint into a training-data oracle** — top-k logits
   from under 10,000 queries are enough to reconstruct the output projection and distil a
   clone ([[gotchas/sec-logprobs-on-your-api-are-an-extraction-surface]]).

---

## 1. The auth boundary: `--api-key` covers four path prefixes

**Recommendation: do not rely on `--api-key` as a security boundary.** vLLM's own docs
say so — "Do not rely exclusively on `--api-key` for securing access to vLLM" — and
prescribe an allowlisting reverse proxy (nginx, Envoy, or a Kubernetes Gateway) that
"explicitly allowlists only the endpoints you want to expose to end users, blocks all
other endpoints, including the unauthenticated inference and operational control
endpoints, and implements additional authentication, rate limiting, and logging at the
proxy layer" ([[gotchas/ops-api-key-covers-only-v1-prefixes]]).

Concretely, what stays open even with the key set:

- `POST /invocations` runs inference at full capability, unauthenticated.
- `POST /pause` takes the server out of service.
- `POST /abort_requests` cancels requests.
- `POST /update_weights` replaces the weights.

**Also:** never set `VLLM_SERVER_DEV_MODE=1` or enable profiler endpoints in production;
keep `--grpc-port` off unless required and firewall it; and treat endpoint plugins as
trusted server code with the same weight as the server itself, because a plugin route
"can shadow a core route and silently replace its behavior" with no conflict enforcement.

On the SGLang side, prefer a real admin credential: `--admin-api-key` marks endpoints
admin-only and requires a bearer token.

**The deeper point:** the authz layer barely exists at the serving layer. vLLM has
authentication (one shared API key) and essentially **no authorization** — no per-tenant
identity, no scoping, no policy on which model or adapter a caller may reach. Where a
gateway owns identity, the model server's `--api-key` is really a shared internal secret
and must be treated as such
([[gotchas/ops-api-key-covers-only-v1-prefixes]]). Note the operational consequence: the
router's own probing of `/server_info` and `/model_info` requires the same key to be
passed as `--worker-api-key`, so **that secret has to be distributed to the routing
tier**.

If your control plane can fail open, then during the failOpen window *everything the EPP
enforces — authz, rate limits — is absent*
([[gotchas/ops-failopen-router-failure-opens-a-bypass]]), which is an argument for
keeping the model servers' own controls sufficient on their own rather than relying on the
gateway alone.

---

## 2. Where filtering runs, and what it costs

Three mechanisms, three different bills
([[gotchas/sec-output-filter-three-placement-cost]]). They act at different points with
different side-effect surfaces, and an engineer asking "add content filtering" gets three
mutually exclusive answers depending on who they ask.

| Placement | Where it acts | Cost characteristic |
|---|---|---|
| **Logit masking** (logits processor) | Inside the decode loop, on every forward step, at batch granularity | Scales with **batch size and step count** — per-token-per-batch-row across the whole batch, not per request |
| **Output scanning** | After the text exists | Free on GPU, but on a streaming endpoint the first tokens are already on the wire before the scanner has read anything |
| **Separate guard model** | A second LLM in the request path | Scales with **request count only**, and its floor is seconds |

**Cost ledger, as far as sources establish it** ([[gotchas/sec-output-filter-three-placement-cost]]):

- *Logit masking:* **no published absolute ms or throughput figure exists** for a masking
  processor inside a production server. The best-characterised example claims only
  "negligible" overhead against a self-defined ratio metric, and explicitly applies the
  transform to the first *m* decode steps rather than every step — so **part of its
  cheapness comes from not running in the hot loop.** Treat any quoted figure with that
  caveat attached.
- *Scanning:* free on GPU, but "scan-then-release is only available if you buffer, which
  costs TTFT."
- *Guard model:* seconds at the small end. Llama Guard 3-1B-INT4 is the smallest published
  option (440 MB) and still reports TTFT of 2.5 s or less and at least 30 tok/s **on a
  commodity Android mobile CPU**. That is the floor, not the ceiling.

**Recommendation: budget the guard pass as a latency line item and measure it rather than
assuming it.** Quantise it. If you can only afford one guarded turn, **guard the response**
— where the harmful content actually is — and accept that the prompt is unguarded.

A guard model is a second autoregressive forward pass, and Llama Guard's design applies
one taxonomy-bearing model to **both** the prompt and the response, so a fully guarded
conversation can incur two guard passes
([[gotchas/sec-guard-model-is-a-second-llm-in-the-path]]). The honest summary of the
whole field is that safety training cannot adapt to new attack types and drops
performance, while external safeguards "have been of limited" value — which is why the
architectural answer, not the filter, is the durable one.

### 2.1 Streaming: you cannot un-send a token

Output scanning is a post-hoc decision on text that already exists. On a streaming HTTP
endpoint the token stream is delivered incrementally, so the check necessarily runs after
some prefix has left
([[gotchas/sec-streaming-output-scan-cannot-unretract]]).

**Recommendation: decide explicitly, per endpoint, which contract you are offering:**

1. **Buffered** — scan then release. Costs TTFT. This is the only sound version.
2. **Streaming with a stated exposure prefix** — document the prefix, accept the window.
3. **Streaming unfiltered** — an explicit decision, recorded.

The residual risk for options 2 and 3 is **not zero and scales inversely with your speed**:
the exposure window is a function of generation speed, so **a faster server is a larger
breach**. That interacts directly with speculative decoding.

### 2.2 A greedy-decoding trap in the filter itself

A logits processor that reports `is_argmax_invariant()` **is never called under greedy
decoding** — vLLM's sampler orders the argmax-invariant group (by default `min_p`) inside
the random-sampling path, which the all-greedy branch returns before reaching
([[gotchas/sec-logit-mask-argmax-invariant-skipped-under-greedy]]). Ship a safety logits
processor with `temperature 0` and it passes tests on sampled traffic while returning
unfiltered output to greedy callers.

**Recommendation: never implement a safety transform as a processor reporting
`is_argmax_invariant() = True`.** If the transform can change which token is argmax, it
must report `False`. Treat `all_greedy` as a distinct configuration and test it
separately. (This record is marked **contested**: the sampler ordering is read from source
and is solid; the end-to-end impact is reasoned rather than measured.)

---

## 3. Tenant isolation: process or MIG, never request

**The uncomfortable part, stated plainly: request-scoped KV ownership is a correctness
feature, not a security boundary.** Two tenants sharing a process are separated by
scheduling logic written *for throughput*, and "no request can read another request's
blocks" is a far weaker statement than "tenant A cannot influence tenant B"
([[gotchas/sec-tenant-isolation-granularity-is-process-or-mig-not-request]]).

What actually exists:

1. **MIG is the real hardware primitive.** From Ampere, up to 7 GPU instances, each with
   separate and isolated paths through the *entire* memory system — on-chip crossbar
   ports, L2 cache banks, memory controllers and DRAM address buses all assigned uniquely
   to one instance. That is what lets a CSP give one client QoS and fault isolation from
   others. **But** the NVLink fabric and memory *bandwidth* remain shared across slices,
   so MIG is not a performance boundary either
   ([[gotchas/mgpu-mig-slice-nvlink-and-memory-bandwidth-are-shared-not-isolated]]).
2. **Time-slicing is explicitly not isolation.** NVIDIA's own documentation states that
   unlike MIG there is no memory or fault isolation between replicas, each pod can run
   unlimited processes, and the GPU gives an equal share of time — one tenant's large
   kernels degrade everyone else ([[gotchas/serve-time-slicing-no-throughput-isolation]]).
   It is suitable only for development and testing.
3. **CUDA MPS gives no error isolation.** Multiple processes share one GPU context; if one
   process triggers a CUDA error, it propagates and takes all clients on that GPU down
   ([[gotchas/serve-mps-no-error-isolation]]).
4. **In-engine, the only isolation control vLLM exposes is `cache_salt`,** folded into the
   block hash for the first block only. It prevents cross-tenant *prefix reuse*. It does
   not create separate address spaces, GPU contexts or schedulers.

**Recommendation:** if tenants are mutually untrusted, **isolate at the level that
actually isolates** — separate processes or containers with MIG partitioning, or separate
deployments. Treat one vLLM process serving mutually untrusted tenants as a **single
security principal with internal request separation only.** Within one process, set a
per-tenant `cache_salt` and accept that it addresses the timing side channel and nothing
else.

The cost of MIG is capacity — partitioning reduces sellable GPU. A TEE is cheap in
overhead (below 7% for most typical LLM queries, approaching zero for larger models and
longer sequences) if you need memory isolation without giving up a whole device.

### 3.1 The KV-cache timing side channel, honestly

Prefix caching reuses KV blocks across requests sharing a prompt prefix, which makes the
cache hit rate a function of other tenants' traffic, and a hit is measurably faster than a
miss — a **client-visible TTFT oracle** ([[gotchas/ops-prefix-cache-is-a-cross-tenant-side-channel]],
[[gotchas/sec-kv-cache-timing-side-channel-reconstructs-private-prefixes]]). The
advisory's effect size is strong enough to be usable: distinguishable at AUC 0.99 from
only 8 tokens of guessed prefix.

**But** the channel largely collapses under realistic contention: mean Cohen's *d* falls
from 0.7789 with no synthetic workers to 0.2109 with two workers on live shared serving
([[gotchas/sec-kv-cache-timing-attack-collapses-under-load]]). The honest reading is not
"the side channel is weak" but "**its effect size is a function of your load**" — and
therefore a benchmark number does not transfer to production.

**Recommendation: measure the side channel on your deployment at your load before paying
to remove it.** The experiment is cheap and reproducible: effect size at zero concurrency
versus under realistic concurrency. If you need defensible cross-tenant separation, use
`cache_salt` as the isolation boundary by design — but note it is **opt-in and not passed
by default**, so a deployment is cross-tenant-leaky unless the client SDK sends it.

---

## 4. What the API leaks: the extraction surface

### 4.1 Logprobs are the highest-value leak on your endpoint

**Returning logprobs — especially top-k, especially full-rank — turns an open-weights
endpoint into a training-data oracle.** Clone What You Can't Steal reconstructs the
**output projection matrix** from top-k logits collected in **under 10,000 black-box
queries** via SVD over the logits, then distils the rest of the architecture into a clone
([[gotchas/sec-logprobs-on-your-api-are-an-extraction-surface]]). A rate limit bounds
requests per minute; it does not bound what is leaked.

**Recommendation:** do not return logprobs on an endpoint serving weights worth
protecting, unless the business need is worth a demonstrably extractable clone. If a
product feature needs them, treat that endpoint as public and budget accordingly.

### 4.2 An API key protects capacity, not the weights

A team serves open weights behind an API key and considers the model protected. It is
not: an API key on your endpoint limits who can reach *your capacity*. It does not limit
what an attacker learns from your outputs. The weights are protected by the host-side
licence gate, entirely independent of your API key
([[gotchas/sec-api-key-protects-capacity-not-the-weights]]).

**Recommendation: assume the weights will be obtained by anyone who wants them, and
protect what that actually costs you — serving margin, rate and provenance.** Treat the
API key as a capacity and quota control.

### 4.3 The guard model is a second extraction target

The moderation model is treated as plumbing; it is a trained model with measured
capability, queried over an API, and it can be replicated — JudgeStealer targets exactly
this asset across pointwise scoring, pairwise comparison and listwise ranking
([[gotchas/sec-guard-and-moderation-models-are-themselves-extraction-targets]]). If
moderation capability is part of your product's defensible value, treat the guard
endpoint as an extraction surface with the same controls as the main one.

### 4.4 Per-query anomaly scoring does not catch extraction; window-level testing does

Individual extraction queries resemble benign requests, so per-query scoring raises almost
nothing while an attacker steadily distilling walks away with a clone. The effective
formulation is benign-calibrated **traffic-window distribution testing** over a semantic
embedding ([[gotchas/sec-model-extraction-detectable-at-traffic-level-not-per-query]]).
**Monitor at the traffic-window level, calibrated only against benign traffic, not per
query.**

### 4.5 Internode traffic is unprotected by default

vLLM's own security documentation is unambiguous: "all communications between nodes in a
multi-node vLLM deployment are insecure by default and must be protected by placing the
nodes on an isolated network." PyTorch Distributed has no authorization, sends
unencrypted, and accepts connections from anywhere; `torch.distributed`'s TCPStore listens
on all interfaces. The traffic carries prompts, KV cache contents and logits
([[gotchas/sec-serving-open-weights-internode-network-is-unprotected-by-default]]).

**Recommendation: isolate the node network.** Do not treat any internode port as safe
because the traffic is "internal" — firewall the KV transfer, data-parallel master and
PyTorch Distributed ports explicitly.

The same record flags a second exposure that is easy to miss because it arrives as a
feature: **SSRF via user-supplied media URLs.**

---

## 5. Watermarks: a provenance signal with a fragile detection surface

**Recommendation: decide what problem you are solving before deploying one.** If the
requirement is "users must be able to tell AI text from human text", watermarking is a
reasonable mechanism and marking obligations are what it is for. **If the requirement is
"do not let harmful content out", a watermark does nothing** — it has no ability to
refuse, to filter, or to prevent anything
([[gotchas/sec-watermark-provenance-is-not-a-compliance-control]]).

The compliance framing is the trap: a watermark gives you a defensible answer to "how do
we mark machine-generated content", and it is easy to record that as having addressed
content safety. It has not.

Four documented fragilities, each sufficient on its own to change the recommendation:

1. **Greedy decoding defeats the signal.** KGW's detection rests on a probabilistic
   premise — under the null, a natural writer violates the green-list rule about half the
   time — and under temperature 0 that premise does not apply. Any all-greedy batch
   contributes no watermark signal and also corrupts the detection statistics
   ([[gotchas/sec-watermark-greedy-decoding-defeats-the-signal]]). A watermark cannot be
   made to work by configuration on a deterministic decode path: either require
   temperature > 0 wherever provenance is a contractual claim, or drop the claim.
2. **Detection dies exactly where the output matters most.** KGW works on essays and
   fails on deployment traffic: for low-entropy prompts, humans and machines give similar
   or identical completions and are indistinguishable in principle, and perturbation
   schemes break on repeated tokens. Mark My Words reports detection on only about 25% of
   assistant-style responses ([[gotchas/sec-watermark-low-entropy-output-is-where-detection-dies]]).
3. **The key is not secret after deployment.** A limited number of generations from a
   black-box watermarked model is enough to reverse-engineer the scheme
   ([[gotchas/sec-watermark-stealable-from-black-box-generations]]). Treat the endpoint
   as public and pair the watermark with provenance you control out-of-band.
4. **Paraphrase robustness is genuinely contested.** Kirchenbauer et al. find watermarks
   survive human and machine paraphrasing (after strong human paraphrasing, detection
   needs roughly 800 tokens); another line of work shows a handful of black-box
   generations is enough to estimate the signal. Both primary sources are recorded and
   the record is kept `status: contested` per AGENTS.md — **do not reason about
   paraphrase robustness from a single paper**
   ([[gotchas/sec-watermark-detection-fragile-to-paraphrasing-contested]]).

**Recommendation: before deploying, measure detection rate on *your* traffic
distribution, split by entropy and task type.** If a large share of your output is
low-entropy, a KGW-style watermark will not carry the claim.

**On cost:** the decode-time cost is dominated by the vocabulary permutation, not the
logit bias. If you must watermark, prefer a stateless formulation over a permutation;
if you serve SynthID, the tournament depth is a direct per-token cost knob
([[gotchas/sec-watermark-kgw-green-list-mechanism-and-cost]]).

---

## 6. Logs, compliance, and architectural defence

### 6.1 DEBUG-level logging is a privacy control, not a verbosity choice

vLLM's request logger writes **full prompt bodies and unredacted `prompt_token_ids`** into
the application log stream at DEBUG, and dumps the full request JSON at DEBUG
([[gotchas/sec-engine-request-logging-puts-full-prompts-in-logs-with-no-redaction]]).
Nobody has to enable PII capture: `--enable-log-requests` plus a DEBUG log level does it,
and the flag name is misleading because it sounds like it enables request *logging* rather
than prompt *content* in logs.

**Recommendation: never run a serving node containing regulated or personal data at DEBUG
log level in production.** Treat any log level above INFO as a privacy control.

### 6.2 Auditability and minimisation are in direct conflict

HIPAA requires both, and inference logging collides with both: 164.312(b) wants the
records, and 164.316 keeps them six years, while the privacy principle pushes the other
way ([[gotchas/sec-hipaa-audit-controls-conflict-with-prompt-log-retention]]). Note the
Required-versus-Addressable distinction: unique user identification is **required**, while
automatic logoff and encryption of ePHI at rest are only **addressable** — a materially
weaker obligation.

**Recommendation: separate the audit trail from the content log, and keep them separable
by design.** Emit an audit event carrying request ID, authenticated principal, model id,
timestamp, token counts and finish reason — and never prompt or completion content. The
architectural separation is what makes erasure and retention requests satisfiable at all
([[gotchas/sec-right-to-erasure-has-to-reach-derived-artefacts]],
[[gotchas/sec-gdpr-art30-record-of-processing-is-a-per-deployment-obligation]]).

### 6.3 Prefer capability scoping to prompt filtering

Prompt-injection detectors are **not robust to adaptive attackers** — a multi-layer
universal optimised suffix defeats layer-wise activation-shift detectors simultaneously
([[gotchas/sec-injection-detectors-fail-to-adaptive-attacker]]). **Do not deploy a
representation-drift detector as your primary injection control**; if you run one, assume
the attacker knows about it and measure against an adaptive attack rather than a static
one. Prefer controls that scope what the model can *do* over controls that inspect what
the model was *told* ([[gotchas/sec-architectural-injection-defense-is-capability-scoping]]).

---

## 7. The security checklist

1. Put an allowlisting reverse proxy in front of vLLM. `--api-key` alone leaves
   `/invocations`, `/pause`, `/abort_requests` and `/update_weights` open.
2. Use `--admin-api-key` on SGLang; firewall `--grpc-port`; never set
   `VLLM_SERVER_DEV_MODE=1`.
3. Isolate the node network — internode KV transfer, DP master and TCPStore are
   unencrypted and unauthenticated.
4. Decide the tenant isolation level explicitly. Mutually untrusted ⇒ process, container
   or MIG, never request.
5. Set a per-tenant `cache_salt`; it is opt-in and not passed by default.
6. Do not return logprobs on an endpoint serving weights you care about.
7. Choose a streaming contract per endpoint and document the exposure prefix.
8. Test safety transforms under greedy decoding, separately.
9. Keep production log level at INFO.
10. Separate the audit event from the content log by construction.
11. Treat extraction detection as a traffic-window problem.
12. If you deploy a watermark, measure detection on your own traffic and split by
    entropy.

---

## 8. What the records cannot answer

1. **No security class in the schema.** Several security records are filed under
   `class: framework` because `schemas/gotcha.schema.json` has no `security` value; the
   records themselves note the true class. Until the enum is added, security gotchas are
   not filterable by class.
2. **No requirement text retrieved.** [[gotchas/sec-pci-dss-requirement-text-not-retrieved]]
   records that the PCI DSS obligation was not traced to requirement text — so no
   compliance mapping in these records is a verified mapping.
3. **No logit-masking cost measurement.** The cost ledger for the in-loop filter is
   explicitly a gap: no published absolute ms or throughput figure for a masking processor
   in a production server.
4. **No end-to-end impact measurement for the greedy-decoding filter bypass.** The sampler
   ordering is read from source; the user-visible consequence is reasoned.
5. **No multi-tenant security study.** All records are single-tenant deployments; there
   is no data on cross-tenant interference on a shared endpoint beyond the timing channel.

---

## Related documents

- [19-production-operations.md](19-production-operations.md) — the failOpen window where
  everything the gateway enforces is absent.
- [21-deployment-topology.md](21-deployment-topology.md) — where the auth boundary and
  the shared internal secret sit.
- [20-observability.md](20-observability.md) — which metrics would catch a leak.
- [18-reliability.md](18-reliability.md) — silent wrong output, the failure class a
  filter cannot see.