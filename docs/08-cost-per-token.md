# Cost per output token

This doc closes the gap that [01-hardware-selection.md](01-hardware-selection.md)
listed as its #1 unanswered question: *"Cost per token. No price↔throughput join
exists."* The repo held both halves — ~50 priced `supply` records and 38
throughput `benchmark` records — and nothing joined them.

It now exists, and it is a tool, not a spreadsheet:

    python tools/cost_per_token.py            # the table
    python tools/cost_per_token.py explain    # the arithmetic for every row
    python tools/cost_per_token.py gaps       # what the records cannot answer

Every figure below is the output of that tool against the current records. Each
row carries **two** record ids: one `supply` price and one `benchmark`
throughput. Nothing here is interpolated, extrapolated or averaged; where a
number is not in a record the cell says `no record`.

## The formula, and the two ways to get it wrong

    USD per million output tokens = (USD per GPU-hour) / (output tokens/s per GPU) x 1,000,000

Both denominators are the trap. The price side, because `price_usd` is **not**
per-GPU-hour — a record priced "per 8-GPU node per HOUR" is 8x a per-GPU-hour
quote, and getting that backwards is an 8x error. The throughput side, because
`value` is **not** per-GPU — a TP=8 record is the whole 8-GPU group.

The tool therefore refuses to print anything it cannot show is in a record. Each
price quote stores a literal substring of the `price_basis` it was read from,
and each GPU count stores a literal substring of the benchmark's `unit` or
`methodology`. If a record is edited and no longer contains that substring, the
tool exits non-zero and prints `UNSOURCED INPUT` rather than a stale number.
That check runs in `tools/test_tools.py` against the real `data/`, so an 8x
price-basis error cannot reach this doc quietly.

## ASSUMPTIONS — read before quoting any number here

Every row below requires all of these. They are not hedges; they are the
conditions under which the arithmetic is even meaningful.

**1. The throughput is AGGREGATE, not single-user.** Every joinable record in
the repo is `tps_aggregate`. `--single-user` returns **zero rows**. A row here is
a *fleet-average* cost at whatever concurrency that benchmark used — it is not
what one user pays. The repo documents exactly how wrong this can be:
[[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]] gives 57.6 tok/s for one user
at batch 1, [[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]] gives 800
aggregate at batch 64, and
[[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]] gives 12.5 tok/s for one
user in that same batch-64 run. Aggregate rose 13.9x while the experienced rate
**fell 4.6x**. A single-user cost cannot be derived from these records; see
"What the records cannot answer".

**2. Price tier is not interchangeable, and rows are NOT comparable across
tiers.** The `tier` column is load-bearing:

| tier | meaning | rows carry |
|---|---|---|
| `on-demand` | published hourly list rate, buy and go | the rate you pay |
| `spot` | interruptible capacity | ~5.4x cheaper than on-demand for the same GPU-hour, but preemption risk is not in the price |
| `capacity-block` | AWS reservable, up to 8 weeks ahead | between the two |
| `serverless` | [[supply/runpod-serverless-h100-h200]] bills per *worker*-hour for a scaling endpoint — not a GPU you hold | different product entirely |
| `peer-to-peer-asking` | VAST rental **asking** rates from the rental-h100/h200/b200 records | asking, not realised; excludes storage and broker SLA premium |
| `base-with-multipliers` | [[supply/modal-serverless-gpu]] base rate | non-preemptible execution is **3x** the base and region selection is **1.15–1.75x**; the published figure is the floor |

A spot row and an on-demand row for the same model differ by up to 5.4x
(Azure H100, 8B: $151.50 spot vs $819.79 on-demand). **The cheapest row in this
doc is a spot row and is not available on demand at that price.**

**3. Used-market and auction records are excluded on purpose.** Every
`used-market`, `auction`, `regional-reseller`, `enterprise-distributor`,
`oem-direct` and `integrator` price in the repo is a price **per card or per
configured system**, not per GPU-hour, and the repo holds no amortised runtime,
power draw or utilisation figure that would turn [[supply/used-h100-80gb-sxm-market]]
or [[supply/used-8x-h100-sxm-node-system]] into a cost per token. That is 24
records, listed by `--gaps` and **not** in the table. Any $/Mtok built from them
would require inventing an hours-of-use denominator, which the repo's own rules
forbid. Four further priced records are excluded for other reasons: two
accelerator families with no throughput record
([[supply/aws-ec2-trn1-trn2-trainium]], [[supply/gcp-cloud-tpu-v4-v5e-v5p]]),
one consumer-GPU rental index
([[supply/rental-consumer-4090-3090-5090-vast-index]]), and the electricity
tariff in [[supply/us-electricity-industrial-tariff-2024]], which is a
`$0.0813/kWh` energy price rather than any compute price at all.

**4. The throughput belongs to THAT model, at THAT precision, on THAT engine.**
Do not generalise across models. A B200's 70B FP4 row does not predict its 8B
FP8 cost, and nothing here predicts the cost of a model the repo has no
benchmark for. Likewise the TensorRT-LLM rows are NVIDIA-engine numbers; vLLM
and SGLang rows are different engines on the same silicon.

**5. Throughput is vendor-reported unless stated.** Of the joinable benchmarks,
13 of 15 are `measured_by: vendor`. Two are third-party:
[[benchmarks/perplexity-llama2-70b-h100-fp8-tp2-bs128-tps-per-gpu]] and
[[benchmarks/sglang-deepseek-v3-96xh100-output-tps-per-node]]. Vendor aggregate
figures are published under an offline infinite-rate feed — the best point on a
concurrency sweep the vendor chose, not a sustained production operating point.

**6. Concurrency is usually not stated.** Most joined records say so explicitly
("concurrency NOT stated and NOT fixed"). The aggregate rate depends on
concurrency, so a row's figure is valid only at the concurrency the source used,
which is frequently off the record. This is the single largest source of
optimism in the table.

**7. Price and throughput are observed on different dates and in different
places.** Supply prices were captured 2026-10-03; benchmarks were published
earlier and may predate a price change. AWS's own schedule lowers p4d Capacity
Blocks and raises p5en Capacity Blocks for 2026-10-07 — see the notes in
[[supply/aws-ec2-p5-h100]]. Prices are the most perishable data in this repo.

**8. This is GPU cost only.** No CPU, host RAM, storage, egress, idle-time or
PUE overhead is in the figure. [[supply/modal-serverless-gpu]] charges egress at
$0.04/GiB separately. A real bill is higher.

## The table

`provider × model × hardware`, all rows **AGGREGATE**. Full 203-row output:
`python tools/cost_per_token.py`.

### Llama 3.1 8B FP8 on 1x H100 SXM — the best-served shape in the repo

Throughput from [[benchmarks/trtllm-llama31-8b-fp8-h100-1k1k-tps]]: 14,991.62
output tok/s aggregate on **one** H100 (TP=1), so no GPU-count division is
needed.

| USD/Mtok | tier | provider | price source | notes |
|---:|---|---|---|---|
| **151.50** | spot | [[supply/azure-nd-h100-v5]] | $2.2712/H100-hr spot | cheapest row in the table; not purchasable on demand at this price |
| 164.34 | spot | [[supply/coreweave-h100-h200]] | $19.71/8 = $2.46375 | |
| 164.76 | peer-to-peer-asking | [[supply/rental-h100-sxm-vast-index]] | $2.47/GPU-hr median | asking rate, not realised |
| 232.80 | on-demand | [[supply/runpod-pods-h100-h200]] | $3.49/H100-hr | |
| 263.48 | base-with-multipliers | [[supply/modal-serverless-gpu]] | $3.95/hr base | 3x if non-preemptible |
| 266.15 | on-demand | [[supply/lambda-gpu-cloud-h100]] | $3.99/H100-hr | 8-GPU node rate |
| 319.51 | serverless | [[supply/runpod-serverless-h100-h200]] | $4.79/worker-hr | endpoint, not a GPU |
| 346.26 | capacity-block | [[supply/aws-ec2-p5-h100]] | $41.528/8 = $5.191 | reservable, ~5 weeks ahead |
| 366.20 | on-demand | [[supply/together-gpu-clusters-h100]] | $5.49/GPU-hr | |
| 410.56 | on-demand | [[supply/coreweave-h100-h200]] | $49.24/8 = $6.155 | |
| 433.58 | on-demand | [[supply/baseten-gpu-instances]] | $6.50/hr | |
| 458.92 | on-demand | [[supply/aws-ec2-p5-h100]] | $55.04/8 = $6.88 | p5 carries a $0.00 "no capacity" sentinel |
| 533.63 | on-demand | [[supply/fireworks-on-demand-h100-h200]] | $8.00/hr | inference-shaped runtime included |
| 667.04 | on-demand | [[supply/oracle-bm-gpu-h100-h200]] | $10.00/GPU-hr | quoted per GPU despite the `.8` shape suffix |
| 737.83 | on-demand | [[supply/gcp-a3-h100-h200]] | $88.49/8 = $11.0613 | most expensive on-demand H100 hour |
| 819.79 | on-demand | [[supply/azure-nd-h100-v5]] | $98.32/8 = $12.29 | tied with GCP for dearest |

**5.4x spread between the cheapest spot row and the dearest on-demand row, same
GPU, same model, same benchmark.**

### Llama 3.1 8B FP8 on 1x H200 SXM

Throughput from [[benchmarks/trtllm-llama31-8b-fp8-h200-1k1k-tps]]: 17,162.49
tok/s on one H200.

| USD/Mtok | tier | provider |
|---:|---|---|
| 152.44 | spot | [[supply/coreweave-h100-h200]] |
| 264.53 | base-with-multipliers | [[supply/modal-serverless-gpu]] |
| 267.44 | on-demand | [[supply/runpod-pods-h100-h200]] |
| 291.33 | peer-to-peer-asking | [[supply/rental-h200-sxm-vast-index]] |
| 345.52 | serverless | [[supply/runpod-serverless-h100-h200]] |
| 367.37 | on-demand | [[supply/coreweave-h100-h200]] |
| 400.00 | capacity-block | [[supply/aws-ec2-p5en-h200]] |
| 461.01 | on-demand | [[supply/aws-ec2-p5en-h200]] |
| 466.13 | on-demand | [[supply/fireworks-on-demand-h100-h200]] |
| 582.67 | on-demand | [[supply/oracle-bm-gpu-h100-h200]] |
| 617.63 | on-demand | [[supply/azure-nd-h200-v5]] |

Azure H200 v5 has **no** cheaper spot row because its spot price equals its
on-demand price ($84.80/8) — a 0% discount, which per the record's own notes is
what a sold-out fleet looks like in a price API.

### Llama 3.3 70B FP8 on 2x H100 SXM

Throughput from [[benchmarks/trtllm-llama33-70b-fp8-h100-tp2-1k2k-tps]]: 3,708.93
tok/s across a **TP=2** group = 1,854.46 tok/s per GPU. The division by 2 is
where a naive join would have understated every figure below by 2x.

| USD/Mtok | tier | provider |
|---:|---|---|
| 1,881.94 | on-demand | [[supply/runpod-pods-h100-h200]] |
| 2,151.56 | on-demand | [[supply/lambda-gpu-cloud-h100]] |
| 2,960.42 | on-demand | [[supply/together-gpu-clusters-h100]] |
| 3,319.02 | on-demand | [[supply/coreweave-h100-h200]] |
| 3,505.05 | on-demand | [[supply/baseten-gpu-instances]] |
| 3,709.96 | on-demand | [[supply/aws-ec2-p5-h100]] |
| 4,313.91 | on-demand | [[supply/fireworks-on-demand-h100-h200]] |
| 5,392.39 | on-demand | [[supply/oracle-bm-gpu-h100-h200]] |
| 5,964.69 | on-demand | [[supply/gcp-a3-h100-h200]] |
| 6,627.25 | on-demand | [[supply/azure-nd-h100-v5]] |

### Llama 3.3 70B FP4 on 1x B200

Throughput from [[benchmarks/trtllm-llama33-70b-fp4-b200-tp1-tps]]: 6,725.03
tok/s on a **single** B200 (the FP4 checkpoint fits one GPU, TP=1).

| USD/Mtok | tier | provider |
|---:|---|---|
| 634.01 | spot | [[supply/coreweave-h100-h200]] |
| 929.36 | base-with-multipliers | [[supply/modal-serverless-gpu]] |
| 994.79 | on-demand | [[supply/lambda-gpu-cloud-h100]] |
| 1,009.66 | on-demand | [[supply/runpod-pods-h100-h200]] |
| 1,185.12 | peer-to-peer-asking | [[supply/rental-b200-vast-index]] |
| 1,278.80 | on-demand | [[supply/coreweave-h100-h200]] |
| 1,284.75 | serverless | [[supply/runpod-serverless-h100-h200]] |
| 1,336.80 | on-demand | [[supply/together-gpu-clusters-h100]] |
| 1,933.08 | on-demand | [[supply/fireworks-on-demand-h100-h200]] |

The B200 at FP4 is **1.88x cheaper per token than the H100 running the same 70B at
FP8** on the same provider and the same tier (Runpod on-demand: 1,009.66 vs
1,881.94), on **one** GPU instead of two. Against the H200 on-demand (2,177.76,
CoreWeave) it is 2.16x.

### DeepSeek-V3 671B FP8 per 8xH100 node

Throughput from [[benchmarks/sglang-deepseek-v3-96xh100-output-tps-per-node]]:
22,300 tok/s per 8-GPU node = 2,787.5 tok/s per GPU.

| USD/Mtok | tier | provider |
|---:|---|---|
| 814.78 | spot | [[supply/azure-nd-h100-v5]] |
| 883.86 | spot | [[supply/coreweave-h100-h200]] |
| 886.10 | peer-to-peer-asking | [[supply/rental-h100-sxm-vast-index]] |
| 1,252.02 | on-demand | [[supply/runpod-pods-h100-h200]] |
| 1,417.04 | base-with-multipliers | [[supply/modal-serverless-gpu]] |
| 1,431.39 | on-demand | [[supply/lambda-gpu-cloud-h100]] |
| 1,718.39 | serverless | [[supply/runpod-serverless-h100-h200]] |
| 1,862.24 | capacity-block | [[supply/aws-ec2-p5-h100]] |
| 1,969.51 | on-demand | [[supply/together-gpu-clusters-h100]] |
| 2,208.07 | on-demand | [[supply/coreweave-h100-h200]] |
| 2,331.84 | on-demand | [[supply/baseten-gpu-instances]] |
| 2,468.16 | on-demand | [[supply/aws-ec2-p5-h100]] |
| 2,869.96 | on-demand | [[supply/fireworks-on-demand-h100-h200]] |
| 3,587.44 | on-demand | [[supply/oracle-bm-gpu-h100-h200]] |
| 3,968.18 | on-demand | [[supply/gcp-a3-h100-h200]] |
| 4,408.97 | on-demand | [[supply/azure-nd-h100-v5]] |

### The controlled comparison: two models, one node, one power envelope

[[benchmarks/h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-energy-per-token]] and
[[benchmarks/h200-8x-mlperf-v4-1-maxq-mixtral-8x7b-offline-energy-per-token]] are
the same 8xH200 MaxQ node, same instrument, same locked 1000 MHz clock. Only the
model changes. This is the cleanest cost-per-token contrast the repo can make,
and it needs no price join at all — the records already carry joules per token
(0.2299 vs 0.1195 J/output token).

At one CoreWeave spot rate ($2.61625/GPU-hr), the join reproduces the energy
result exactly, which is a good check that both denominators are right:

| model | tok/s/GPU | USD/Mtok | J/token (from record) |
|---|---:|---:|---:|
| Mixtral 8x7B | 6,123.49 | **427.25** | 0.1195 |
| Llama 2 70B (MaxQ) | 3,157.76 | 828.51 | 0.2299 |

MoE is **1.94x cheaper per token than a dense 70B on identical hardware**. The
ratio of costs (0.5157) equals the reciprocal of the throughput ratio to four
decimal places, as it must when only the model changes.

**But do not read this as "MoE is cheaper."** The records say so themselves: the
two models are not of comparable quality, and ~12.9B of Mixtral's 46.7B
parameters are active per token. Per *useful* token, per byte of weight shipped,
or per unit of output quality, this comparison does not settle anything.

### Llama 2 70B on 8xH200 — the power cap has a price

| variant | tok/s/GPU | USD/Mtok at CoreWeave spot | record |
|---|---:|---:|---|
| MaxP (no power cap) | 3,912.84 | **668.63** | [[benchmarks/h200-8x-mlperf-v4-1-maxp-llama2-70b-offline-vs-maxq]] |
| MaxQ (capped, 5806.9 W) | 3,157.76 | 828.51 | [[benchmarks/h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-energy-per-token]] |
| MaxQ server metric | 2,889.14 | 905.55 | [[benchmarks/h200-8x-mlperf-v4-1-maxq-llama2-70b-server-energy-per-token]] |

**Power-capping a node makes each token 1.24x more expensive.** The cap saves
power but costs throughput, and at a fixed hourly rate throughput loss is
straight-through cost. This is the repo's clearest demonstration that $/Mtok and
J/token are different quantities — J/token *improves* under the cap while $/Mtok
gets worse.

### Llama-2-70B FP8, 128-concurrency, third-party

Throughput from [[benchmarks/perplexity-llama2-70b-h100-fp8-tp2-bs128-tps-per-gpu]]:
767 tok/s **per GPU** (already per-GPU in the record's own unit string), batch 128.
Independent third party on their own AWS p5 fleet.

| USD/Mtok | tier | provider |
|---:|---|---|
| 2,961.15 | spot | [[supply/azure-nd-h100-v5]] |
| 3,212.19 | spot | [[supply/coreweave-h100-h200]] |
| 4,550.20 | on-demand | [[supply/runpod-pods-h100-h200]] |
| 6,767.93 | capacity-block | [[supply/aws-ec2-p5-h100]] |
| 8,970.01 | on-demand | [[supply/aws-ec2-p5-h100]] |
| 13,037.81 | on-demand | [[supply/oracle-bm-gpu-h100-h200]] |
| 14,421.51 | on-demand | [[supply/gcp-a3-h100-h200]] |
| 16,023.47 | on-demand | [[supply/azure-nd-h100-v5]] |

### The dearest rows, and why they are not interesting

[[benchmarks/vllm-llama31-405b-fp8-8xh100-tp8-output-tps]] reports 291.53 output
tok/s across 8xH100 = **36.44 tok/s per GPU**, at 1024-in/128-out. That record's
own methodology says the concurrency was neither stated nor controlled and the
post calls the figures "preliminary". Dividing an hourly rate by it produces
$274,414 to $337,255 per million tokens on on-demand providers.

Those numbers are arithmetically correct and **should not be used as a cost
estimate.** The record is a latency-sensitive short-output workload measured at
an unstated concurrency; a $/Mtok from it describes nothing purchasable. It is
listed here so the range is complete and so nobody later mistakes the top of the
table for a finding.

## Price-basis errors nearly made

Each of these was a live wrong answer until the record text was read.

| error | wrong value | correct | the record that prevents it |
|---|---|---|---|
| **Node-vs-GPU price basis.** Reading `price_usd` as per-GPU-hour | CoreWeave H100 at $6.155 instead of $49.24 | $49.24 is per **8-GPU node** per hour; $6.155 is already the per-GPU figure | [[supply/coreweave-h100-h200]] states "per 8-GPU node per HOUR" |
| **Oracle's `.8` shape suffix.** Reading it as 8 GPUs per hour | $80.00/GPU-hr | $10.00/GPU-hr — Oracle quotes **per GPU per hour**, and `.8` is the GPU count in the shape name | [[supply/oracle-bm-gpu-h100-h200]] |
| **Single number, many GPUs.** Reading one provider's H100 price as applying to every SKU | AWS p5 $6.88 vs Azure $12.29 vs GCP $11.06 collapsed into one "H100 price" | each is a distinct, separately sourced rate | three separate supply records |
| **Ignoring the tier.** Using a record's headline rate | CoreWeave H100 "spot $2.46" cited as its cost | on-demand is $6.155; the record publishes both columns | [[supply/coreweave-h100-h200]] |
| **Modal's multiplier.** Using the base rate as the real price | $3.95/H100-hr | non-preemptible execution is **3x** | [[supply/modal-serverless-gpu]] notes |
| **Aggregate treated as per-GPU AND per-user.** | MPT-7B at batch 64 "800 tok/s → $X/Mtok for one user" | 800 is aggregate across 64 requests; the user gets 12.5 | three paired benchmark records |
| **TP=8 not divided.** | Llama 3.3 70B at $1,881.94 appearing as $940.97 | 3,708.93 tok/s is the whole TP=2 group | [[benchmarks/trtllm-llama33-70b-fp8-h100-tp2-1k2k-tps]] unit says "aggregate, 2 GPUs" |
| **Used-market as rental.** | a used H100 module "at $12,999 ÷ hours" | that is a price **per card**, and the repo has no amortised runtime | [[supply/used-h100-80gb-sxm-market]] price_basis |

## What the records cannot answer yet

`python tools/cost_per_token.py gaps` is the machine version of this section.

**1. Any single-user cost per token. Zero rows.** This is the biggest gap and it
is structural: the repo has per-user records only where `accelerator_ids` is
empty — [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]],
[[benchmarks/mpt7b-a100-bs64-per-user-decode-tps]] — so they cannot be priced.
The one real per-user record on real hardware,
[[benchmarks/vllm-qwen38-nvfp4-gb300-nvl72-interactivity]] (180 tok/s/user), has
a sibling aggregate that is a **total**-tokens figure, so the pair needed for a
per-output-token cost does not exist. Filling this needs a benchmark with both
`tok_s_per_user` **and** an output-token `tps_aggregate` on the same accelerator.

**2. AMD Instinct is now priceable, and the join still cannot use it — because the
tool's price table has not caught up.** This correction is recorded because the
previous version of this document asserted the opposite, and the assertion was
wrong in a way that generalises.

**What is true now.** [[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]] prices
AMD Instinct **per GPU-hour**: MI300X at **$6.00** and MI355X at **$8.60**, read off
Oracle's own accelerated-compute table, in the same column and on the same page as
**NVIDIA BM.GPU.H100.8 at $10.00** ([[supply/oracle-bm-gpu-h100-h200]] prices the H100
and H200 at $10.00). The per-GPU-hour denominator this document called impossible
exists, it is first-party, and it sits beside the NVIDIA figure a like-for-like
comparison needs.

**What the earlier claim got wrong, and why it was not merely out of date.** It
cited four AMD supply records — `amd-instinct-oem-systems`, `dell-poweredge-xe-ai`,
`hpe-proliant-compute-xd`, `qct-gpgpu-servers` — all of which genuinely have
`price_usd: null`. Those four are correct as they stand: they describe **OEM and
reseller channels, which do not quote per-GPU-hour because per-GPU-hour is not a
unit an OEM system sale is priced in.** The error was one of **scope**: the claim
was stated as "AMD hardware is unpriceable" when the only thing established was
"no supply record *in this repo* priced AMD per GPU-hour". The channel that
publishes per-GPU-hour rates for AMD is a cloud provider, not an OEM, so the
negative generalised across a boundary it was never tested at. This is the same
shape of error as `rent-cerebras-api-token-pricing-negative` elsewhere in the
repo: a negative asserted on one channel, generalised to the whole market.

**What is still true, and why `gaps` still reports AMD as unpriceable.** The gap is
closed **on the supply side and still open in the tooling**.
`tools/cost_per_token.py` is driven by a hand-maintained `PRICE_QUOTES` table rather
than by scanning `data/supply`, so the new record does not by itself reach the tool:
`python tools/cost_per_token.py gaps` still lists **11 records on
`amd-instinct-mi355x` and 2 on `amd-instinct-mi300x`** as having no priced
per-GPU-hour supply record. Adding two entries of the form
`dict(accel="amd-instinct-mi300x", usd=6.00, tier="on-demand", gpus=1, basis="BM.GPU.MI300X.8 = $6.00")`
and its MI355X equivalent is an edit in `tools/`, outside this document's scope.
The 19 AMD MI3xx throughput records in the repo — including
[[benchmarks/mi355x-mlperf-v6-0-llama2-70b-wmxfp4-offline-tokens]] at 103,480 tok/s,
[[benchmarks/bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-offline]] at 158,458 tok/s,
[[benchmarks/mi355x-mlperf-v6-0-gpt-oss-120b-offline-tokens]] at 95,004 tok/s and
[[benchmarks/llama-cpp-mi300x-deepseek-v3-671b-q4-decode-tok-s]] at 36.53 tok/s
single-user — would all become priceable on that one edit.

**What it buys, stated at the limit.** $6.00 is an **on-demand asking rate** on a
bare-metal shape with no vCPU/NRAM bundling. For a like-for-like AMD-vs-NVIDIA
comparison use **Oracle's own H100 at $10.00**, not AWS p5's $6.88-equivalent,
which bundles 24 vCPU and 256 GB of RAM into the GPU hour. And Oracle's
availability on GPU shapes is unverified from any public page — its H100 record
says so directly — so **these figures describe price, not purchasability.**

**3. GB200/GB300 NVL72 racks are unpriceable per GPU.**
[[benchmarks/sglang-deepseek-v3-gb200-nvl72-decode-tps-per-gpu]] (7,583 tok/s
per GPU) and [[benchmarks/vllm-qwen35-nvfp4-gb200-nvl72-tps-per-gpu]] (25,000
tok/s per GPU) have no per-GPU-hour supply record. CoreWeave's $42.00/h for a
whole 72-GPU Superchip is in [[supply/coreweave-h100-h200]], but the benchmark's
rack (12 of 14 nodes, 768 GPUs) is not that shape.

**4. A100, L4, L40S and A10 are priced but have no throughput record at all.**
`nvidia-a100-80gb-sxm4` is priced by **9** supply records
([[supply/aws-ec2-p4d-p4de-a100]], [[supply/azure-nd-a100-v4]],
[[supply/gcp-a2-a100]], [[supply/lambda-gpu-cloud-h100]],
[[supply/modal-serverless-gpu]], [[supply/oracle-bm-gpu-a100]],
[[supply/runpod-pods-h100-h200]], [[supply/runpod-serverless-h100-h200]],
[[supply/coreweave-h100-h200]]) and not one throughput record exists for it.
Same for `nvidia-l40s` (6 records), `nvidia-l4` (4), `nvidia-h100-pcie` (2),
`nvidia-a10` (1). **Price is abundant and throughput is absent** — the mirror
image of the AMD problem, and much cheaper to fix.

**5. Consumer and non-NVIDIA accelerators are entirely unpriceable.**
[[benchmarks/llamacpp-rx7900xtx-gfx1100-qwen3-27b-q4-batch1-decode-tok-s]],
[[benchmarks/llamacpp-radeon-r9700-gfx1201-q1-0-decode-tok-s]],
[[benchmarks/mlx-m4-max-qwen3-30b-a3b-4bit-decode-tok-s]],
[[benchmarks/mlx-m3-pro-qwen3-4b-4bit-decode-tok-s]],
[[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]],
[[benchmarks/arc-pro-b70-mlperf-v6-1-llama2-70b-offline-tokens]],
[[benchmarks/gaudi2-mlperf-v4-0-llama2-70b-offline-tokens]] — seven real
single-user measurements with no per-GPU-hour price for their hardware.

**6. Two records measure total tokens, not output tokens, and are excluded.**
[[benchmarks/silex-llama33-70b-4xh100-tp4-tps-200x200]] (7,000 tok/s at 200-in/
200-out) and [[benchmarks/vllm-qwen38-nvfp4-gb300-nvl72-tps-per-gpu]] (5,000
tok/s/GPU at 8192-in/1024-out) report prompt **plus** generated tokens. At 8192
prompt tokens the prompt share dominates, so a $/output-token from them would be
badly wrong. **No record in the repo carries an output-only rate for these.**

**7. No utilisation figure exists anywhere.** The $/Mtok above assumes a GPU-hour
is fully used. Real serving is bounded by KV-cache capacity and by
[[gotchas/dvfs-power-cap-moves-tail-latency-under-sustained-load]]-class
effects. There is no `utilisation` field in the supply schema and no
achieved-throughput-below-peak record, so **cost per token actually delivered to
a user cannot be computed** — only cost per token at vendor peak.

**8. Not answerable at all: input-token cost.** Every figure here is output.
[[benchmarks/silex-llama33-70b-4xh100-tp4-tps-200x200]] shows prefill is a
different rate (52.3k input vs 22.3k output tok/s per node on SGLang), but no
joined record pairs a prefill price with a prefill rate.

## Coverage: what the join does and does not reach

Of **145** benchmark records with a positive `value`:

- **15 joinable — 10.3%.** 203 rows, 17 supply records, 11 models, 13
  hardware×model combinations, 136 provider×model pairs.
- **65 throughput records named in `--gaps`** as unpriceable, which together with
  the 15 joined accounts for all **80** records whose `metric` is one of
  `decode_tok_s`, `tok_s_per_user` or `tps_aggregate`.
- **The other 65 positive-value records are not throughput records at all** and are
  therefore outside this join's denominator by construction: 33 `quality`,
  10 `dimensionless_ratio`, 8 `ttft_ms`, 7 `prefill_tok_s`, 4 `speedup_ratio`,
  3 `itl_ms`.

**This document names 28 of the 145 positive-value records; 117 are never
mentioned here**, and the gap is concentrated in exactly the classes a cost
model needs and this join cannot serve: **33 of the 51 `tps_aggregate` records
and 17 of the 24 `decode_tok_s` records appear nowhere in this doc**, because
they sit on hardware no price record covers (AMD Instinct, GB200/GB300 NVL72,
Apple, consumer Radeon, Gaudi2, Arc) or on a `benchmark`-only channel with no
accelerator id to join on. **65 positive-value records have no accelerator id
at all** and are structurally unpriceable regardless of how many prices exist.
Where another document carries those records, cite it; the point of stating the
count here is that a 10.3% join is not a pricing model, it is a worked example
on the fifth of the corpus the price records happen to reach.

**Every joinable row is aggregate. There are zero single-user rows.** The join
answers "what does a saturated fleet cost per million output tokens", which is a
real and useful question. It does not answer "what will my user pay per token",
which is usually the question being asked.

Three findings the join makes visible that were not obvious from the records
alone:

1. **Price tier dominates hardware choice for a fixed model.** On 8B FP8 on H100,
   spot Azure ($151.50) beats on-demand Runpod ($232.80) beats on-demand
   hyperscaler Oracle ($667.04) — a **2.9x** spread from *when and how you buy*
   alone, before any hardware decision. Across all 16 H100 8B rows the full
   spread is **5.4x** ($151.50 spot to $819.79 on-demand).
2. **A newer GPU at lower precision beats an older one at higher precision.**
   B200 FP4 at $1,009.66 vs H100 FP8 at $1,881.94 for 70B on the same provider
   and tier — 1.86x, on one GPU instead of two.
3. **Power caps trade joules for dollars in the wrong direction for this metric.**
   J/token improves 17% under the MaxQ cap; $/Mtok worsens 24%.

## Reproducing

    python tools/cost_per_token.py                # 203 rows
    python tools/cost_per_token.py explain        # per-row arithmetic
    python tools/cost_per_token.py gaps           # the unpriceable records
    python tools/cost_per_token.py --single-user  # zero rows, by design
    python tools/test_tools.py                    # the evidence check is tested

The tool stores, for every price, the literal substring of the `price_basis` it
was read from, and for every throughput, the literal substring establishing the
GPU count. Editing a record to change a price makes the tool exit non-zero. If a
row in this doc stops matching the tool's output, the tool is right and this doc
is stale.