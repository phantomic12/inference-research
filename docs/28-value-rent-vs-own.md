# 28 — Value: rent vs own, $/Mtok, joules per token, and what the corpus can actually rank

<!-- written 2026-10-12. Every figure below is quoted from the record cited beside it; none is computed here. -->

The `value:` wave left its findings in gotcha records, where each one answers one narrow
question. Read together they say something none says alone: **most "which is the best
value" questions are not answerable from this corpus yet, and the ones that are answerable
are dominated by join rules, not by hardware.** This doc is the map.

## 0. The short form

| question | answer in the corpus | record |
|---|---|---|
| cheapest $/Mtok to rent, by model class | answerable **within** the aggregate (batched) population only | [[gotchas/value-rental-ranking-by-workload-and-the-aggregate-trap]] |
| is spot a budget number? | no — the headline discount is a ceiling, and spot is excluded from committed-use discounts | [[gotchas/rent-spot-discount-is-not-a-budget-number]] |
| what does owning cost over three years? | computable; electricity is a minor line | [[gotchas/value-tco-and-why-rent-vs-own-cannot-be-computed-from-this-corpus]] |
| rent vs own break-even | **not computable** — no accelerator has both a priced rental row and a batch-matched local benchmark | same record |
| cheapest usable VRAM | only five accelerators can be ranked at all, and the ranking is dominated by which model each was measured on | [[gotchas/value-cheapest-usable-vram-and-what-it-actually-runs]] |
| joules per token | 4.11 to 20.53 J/tok on comparable single-device batch-1 rows | [[gotchas/value-joules-per-token-and-why-the-cheapest-energy-is-not-the-cheapest-hardware]] |
| should I buy at all? | a genuine community split, settled by usage pattern and data control, not by $/token | [[gotchas/sent-community-split-buy-local-versus-rent]] |

## 1. Renting: the price is not one number

The cost tool ([08-cost-per-token.md](08-cost-per-token.md), `tools/cost_per_token.py`)
joins `supply` prices to `benchmark` throughput. Before trusting any row, four traps apply:

- **Shape.** The same GPU differs about 2x per-GPU between an 8-GPU and a 1-GPU shape;
  record the per-GPU rate, the shape's GPU count and the node rate it came from
  ([[gotchas/rent-per-gpu-hour-needs-the-shape-too]]).
- **Tier.** Spot ran 34.9% under on-demand on H100 SXM and 43.0% under on H200 SXM, while
  capacity-block cost 148.7% / 149.6% *of* on-demand — capacity-block buys reserved
  capacity, not cheaper tokens
  ([[gotchas/value-rental-ranking-by-workload-and-the-aggregate-trap]]).
- **Token-API tiers.** One provider's own page shows up to 8x between tiers and context bands
  for the same model; store provider, model version, tier, context band, cache state and date
  ([[gotchas/rent-token-rate-without-a-tier-is-meaningless]]).
- **The bill beside the GPU.** An Azure Ultra disk at 28,000 GiB is about $4.59/h, 37% of the
  $12.29 H100-hour it is attached to
  ([[gotchas/rent-persistent-disk-can-cost-more-than-the-gpu-next-to-it]]).

And one population trap: of the cloud rows the tool prices, **only one is a per-user
figure**; the rest are batched aggregates. The two cannot be ranked against each other
([[gotchas/value-rental-ranking-by-workload-and-the-aggregate-trap]]). Where the price
itself comes from matters too: vendor rate tables are often JavaScript-rendered, so the
pricing API is the only reproducible surface
([[gotchas/rent-vendor-pricing-tables-are-js-but-their-apis-are-not]]).

## 2. Owning: capex dominates

The TCO model in [[gotchas/value-tco-and-why-rent-vs-own-cannot-be-computed-from-this-corpus]]
uses the cheapest non-cloud supply record per card, electricity at $0.0891/kWh from
[[supply/us-electricity-industrial-tariff-2024]], card TDP plus a stated 120 W host allowance,
and a 3-year horizon. Its conclusion: electricity is 15-20% of three-year cost for the
~$2,000 cards and under 5% for the $12,000-and-up cards, and a 450 W vs 300 W card differs by
about $76/yr at 30% load. **Power decides thermals, noise and circuit capacity, not value.**

## 3. Energy: comparable rows are rare

The joules-per-token record keeps only single-device, batch-1, sourced rows — nine of them —
and excludes an unsourced figure, multi-GPU runs charged against one card's TDP, and a
batch-2048 row
([[gotchas/value-joules-per-token-and-why-the-cheapest-energy-is-not-the-cheapest-hardware]]).
No vendor publishes J/tok, and TDP-based estimates carry a 30-50% error
([[gotchas/vendors-dont-publish-joules-per-token-estimation-error]]).

## 4. The join rules that make any of this valid

From [[gotchas/value-cheapest-usable-vram-and-what-it-actually-runs]], after two join
artefacts produced 60x and 80x false rankings:

- **Rule A** — a supply price attaches to the item its `price_basis` names, not to every
  accelerator the page mentions.
- **Rule B** — a per-card tok/s is the denominator of every value figure; a cluster, RPC or
  multi-host figure must not be filed under one card's `accelerator_ids`.

The cost tool's own rule is the same idea for GPU counts: a divisor must appear verbatim in
the benchmark's unit or methodology (`check_evidence()`, see
[26-known-defects.md](26-known-defects.md) D2).

## 5. What the records cannot answer

- **Rent-vs-own break-even.** Needs one same-accelerator pair — a priced cloud row and a
  batch-matched local benchmark — or an explicit aggregate-to-per-user conversion. Neither
  exists ([[gotchas/value-tco-and-why-rent-vs-own-cannot-be-computed-from-this-corpus]]).
  This is the single highest-value benchmark to add.
- **Local value ranking across models.** The five rankable accelerators (eight rows) were
  each measured on different models, so cross-card comparison is not a value finding.
  The table itself now lives in
  [[gotchas/value-cheapest-usable-vram-and-what-it-actually-runs]] with the unit stated
  correctly — dollars per 1000 tok/s of decode throughput, lower is better — after the
  "tok/s per $1000" mislabel recorded in
  [26-known-defects.md](26-known-defects.md) was fixed on 2026-10-12, along with the
  over-attributed prices the table still quoted.

## Related documents

- [08-cost-per-token.md](08-cost-per-token.md) — the $/Mtok tool and its arithmetic.
- [01-hardware-selection.md](01-hardware-selection.md) — which part to buy, per workload.
- [16-migration-upgrade.md](16-migration-upgrade.md) — when a new generation is worth it.
- [17-deployment-shape.md](17-deployment-shape.md) — how shape changes the per-GPU price.
