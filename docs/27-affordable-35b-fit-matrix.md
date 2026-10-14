# 27 — Affordable 35B-class fit matrix

Which recorded accelerators can hold a ~35B-class model, and what does the
cheapest one cost per unit of decode throughput.

**Scope, set by the commissioning brief.** The target class is a **35B-A3B
Mixture-of-Experts** model (~3B active parameters at decode), a **~27B dense**
model, and **Llama-3.3-70B at Q4** as the stretch goal. The question is narrow:
*does it fit, and what is the cheapest $/(tok/s) for the 35B-A3B job.* This doc
does not try to rank serving quality, concurrency, or prefill — those need the
measured-benchmark join in `docs/08-cost-per-token.md`, not a roofline.

**Every number in the tables is either copied verbatim from a record or marked
`(derived)`.** Derived means the arithmetic is shown below and a reader can
re-run it. Nothing here is a measurement.

## 1. The selection rule (which records got a row)

An accelerator record gets a row when **all** of these hold:

- `vram_gb >= 8` — below this there is no useful quantised-7B headroom either.
- `memory_bandwidth_gbps >= 300` **OR** it has a linked `supply` record with a
  purchase price `<= $3,000`.

The `<= $3,000` arm exists so that a cheap-but-slow card (the used-Tesla slice)
is not filtered out by the bandwidth floor before its price can speak. That rule
yields **96 qualifying accelerators**, of which **37 have a usable purchase
price** and 59 are bandwidth-qualified but have no bare-card/system price in
`data/supply/`.

**One exception to the `vram_gb` gate.** A `unified_memory` part (an APU/SoC
whose GPU shares the CPU's LPDDR5x pool) leaves `vram_gb` null *on purpose* —
there is no dedicated VRAM to report. The `>= 8` gate would silently drop it.
For those, `tools/fit_matrix_35b.py` substitutes the **configured memory of the
cheapest linked supply record** (a whole system, so its RAM is the usable pool).
Only [[accelerators/amd-radeon-8060s-ryzen-ai-max]] (Strix Halo, 128 GB
configured) needs this override in the current corpus; the Apple Ultra parts
already carry an explicit `vram_gb`.

## 2. The three fit columns and the one ceiling

Model footprints used for the fit flags (these are the brief's numbers, kept as
stated):

| workload | weights | + KV / runtime headroom | needs `vram_gb >=` |
|---|---|---|---|
| **35B-A3B MoE, Q4** | ~20 GB | 4 GB | **24 GB** |
| **27B dense, Q4** | ~16 GB | 4 GB | **20 GB** |
| **70B dense, Q4** | ~40 GB | 8 GB | **48 GB** |

`fit = vram_gb >= (weights + headroom)`, read off each accelerator record's own
`vram_gb`. A `Y` means *resident on a single device* — no multi-GPU span, no
offload. The 35B-A3B threshold of 24 GB is why every 24 GB card in the table is
a `Y` by exactly zero margin.

**The ceiling.** `tok/s ceiling (derived) = memory_bandwidth_gbps /
model_weights_gb`. This is the standard decode roofline: at batch-1 each output
token must read the resident weights once, so a perfectly bandwidth-bound decode
cannot exceed bandwidth / bytes-read. It is a **ceiling, not a forecast** — real
decode lands well under it because of kernel inefficiency, KV reads, and the
fact that a MoE reads only its active experts. It is also *optimistic for MoE*:
a 35B-A3B that activates ~3B/token reads closer to the active-expert bytes than
the full 20 GB resident, so the same ceiling formula on a dense-equivalent basis
overstates the *value* of a big resident pool for MoE. The doc divides by the
full resident weights (20 GB) anyway, which makes the ceiling a **conservative
lower bound on the achievable rate** and keeps the arithmetic identical across
the three columns.

*How* conservative shows up the moment a low-bandwidth part enters the table:
[[accelerators/amd-radeon-8060s-ryzen-ai-max]] (Strix Halo, 256 GB/s shared)
derives to a 12.8 tok/s ceiling on the 20 GB column, but the corpus's own
measurement of the **exact target model** —
[[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]] — records **42.04
tok/s** on it, ~3.3× the derived ceiling, because the MoE reads only ~3B active
experts per token, not the 20 GB resident. The roofline ranks capex against a
*dense-equivalent* number; it understates real MoE throughput most where
bandwidth is scarcest. Treat every ceiling cell as the floor, not the forecast.

**$/tok-s (derived) = cheapest purchase price ÷ the 35B-A3B ceiling.** It ranks
"how little silicon do I buy to move 20 GB of weights." It is a *capex-efficiency*
figure, not a cost-per-token-served figure — it ignores power, the chassis/baseboard
several of these parts need (§5), and that $/tok-s for a 14 tok/s card and a 240
tok/s card are not the same product.

## 3. Ranked: purchase-priced parts by $/tok-s for the 35B-A3B job

Sorted by derived $/tok-s, lowest first. `Fits` columns are 35B/27B/70B.
Only rows with a usable purchase price appear; the `← supply` column is the
record the price was read from.

| rank | accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* | price | ← supply | $/tok-s *(derived)* |
|---|---|---|---|---|---|---|---|---|
| 1 | [[accelerators/nvidia-v100-sxm2-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $500 | [[supply/ebay-used-tesla-v100-sxm2-32gb]] | **11.11** |
| 2 | [[accelerators/nvidia-tesla-m40]] | 24 | 288 | Y/Y/- | 14.4 | $189 | [[supply/used-nvidia-tesla-m40-market]] | 13.12 |
| 3 | [[accelerators/amd-instinct-mi100]] | 32 | 1228 | **Y/Y/-** | 61.4 | $900 | [[supply/ebay-used-amd-mi100-32gb]] | 14.66 |
| 4 | [[accelerators/nvidia-v100-pcie-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $750 | [[supply/ebay-used-tesla-v100-pcie-32gb]] | 16.67 |
| 5 | [[accelerators/nvidia-tesla-p40]] | 24 | 347 | Y/Y/- | 17.4 | $290 | [[supply/ebay-used-tesla-p40-24gb]] | 16.71 |
| 6 | [[accelerators/intel-gaudi2]] | 96 | 2460 | **Y/Y/Y** | 123.0 | $2,100 | [[supply/ebay-used-intel-gaudi2-hl225-oam]] | 17.07 |
| 7 | [[accelerators/amd-radeon-rx-7900-xtx]] | 24 | 960 | **Y/Y/-** | 48.0 | $875 | [[supply/ebay-used-rx-7900-xtx-market-oct2026]] | 18.23 |
| 8 | [[accelerators/nvidia-rtx-3090]] | 24 | 936 | **Y/Y/-** | 46.8 | $950 | [[supply/ebay-used-rtx3090-oct2026]] | 20.30 |
| 9 | [[accelerators/nvidia-rtx-5090]] | 32 | 1792 | **Y/Y/-** | 89.6 | $1,999 | [[supply/nvidia-marketplace-geforce]] | 22.31 |
| 10 | [[accelerators/nvidia-rtx-3090-ti]] | 24 | 1008 | Y/Y/- | 50.4 | $1,240 | [[supply/microcenter-refurb-rtx-3090]] | 24.60 |
| 11 | [[accelerators/huawei-ascend-310p]] | 24 | 205 | Y/Y/- | 10.2 | $265 | [[supply/taobao-used-huawei-ascend-310p]] | 25.88 |
| 12 | [[accelerators/intel-arc-pro-b60]] | 24 | 456 | Y/Y/- | 22.8 | $655 | [[supply/newegg-new-arc-pro-b60-24gb]] | 28.73 |
| 13 | [[accelerators/cambricon-mlu370]] | 48 | 307 | Y/Y/Y | 15.4 | $450 | [[supply/ebay-used-cambricon-mlu370]] | 29.30 |
| 14 | [[accelerators/intel-arc-pro-b70]] | 32 | 608 | Y/Y/- | 30.4 | $949 | [[supply/newegg-new-arc-pro-b70-32gb]] | 31.22 |
| 15 | [[accelerators/nvidia-a100-40gb-sxm4]] | 40 | 1555 | Y/Y/- | 77.8 | $2,499 | [[supply/ebay-used-a100-40gb-sxm4]] | 32.14 |
| 16 | [[accelerators/amd-instinct-mi250]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $5,450 | [[supply/ebay-used-amd-mi250x-oam-128gb]] | 34.06 |
| 17 | [[accelerators/amd-instinct-mi250x]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $5,450 | [[supply/ebay-used-amd-mi250x-oam-128gb]] | 34.06 |
| 18 | [[accelerators/nvidia-rtx-4090]] | 24 | 1008 | Y/Y/- | 50.4 | $2,000 | [[supply/ebay-used-consumer-gpu-market]] | 39.68 |
| 19 | [[accelerators/amd-instinct-mi210]] | 64 | 1600 | **Y/Y/Y** | 80.0 | $4,180 | [[supply/ebay-used-amd-mi210-64gb]] | 52.25 |
| 20 | [[accelerators/tenstorrent-blackhole-p150]] | 32 | 512 | Y/Y/- | 25.6 | $1,399 | [[supply/new-tenstorrent-blackhole-p150]] | 54.65 |
| 21 | [[accelerators/qualcomm-ai100]] | 32 | 137 | Y/Y/- | 6.8 | $400 | [[supply/used-qualcomm-cloud-ai100-pcie]] | 58.39 |
| 22 | [[accelerators/apple-m1-ultra]] | 128 | 800 | **Y/Y/Y** | 40.0 | $2,600 | [[supply/used-mac-studio-m1-ultra-128gb]] | 65.00 |
| 23 | [[accelerators/nvidia-h100-sxm]] | 80 | 3350 | **Y/Y/Y** | 167.5 | $12,999 | [[supply/ebay-used-h100-sxm5-module]] | 77.61 |
| 24 | [[accelerators/apple-m3-ultra]] | 512 | 819 | **Y/Y/Y** | 41.0 | $3,700 | [[supply/used-mac-studio-m3-ultra-96gb]] | 90.35 |
| 25 | [[accelerators/apple-m2-ultra]] | 192 | 800 | **Y/Y/Y** | 40.0 | $3,700 | [[supply/used-mac-studio-m2-ultra-192gb]] | 92.50 |
| 26 | [[accelerators/nvidia-a100-40gb-pcie]] | 40 | 1555 | Y/Y/- | 77.8 | $7,500 | [[supply/ebay-used-a100-40gb-pcie]] | 96.46 |
| 27 | [[accelerators/nvidia-a100-80gb-sxm4]] | 80 | 2039 | **Y/Y/Y** | 102.0 | $10,450 | [[supply/ebay-used-accelerator-market]] | 102.50 |
| 28 | [[accelerators/nvidia-a100-80gb-pcie]] | 80 | 1935 | **Y/Y/Y** | 96.8 | $12,099 | [[supply/ebay-used-a100-80gb-pcie]] | 125.05 |
| 29 | [[accelerators/amd-radeon-8060s-ryzen-ai-max]] | 128* | 256 | **Y/Y/Y** | 12.8 | $1,900 | [[supply/new-minisforum-ms-s1-max-128gb]] | 148.44 |
| 30 | [[accelerators/nvidia-rtx-pro-6000-blackwell]] | 96 | 1792 | **Y/Y/Y** | 89.6 | $16,624 | [[supply/bh-photo-used-workstation-gpus]] | 185.54 |
| 31 | [[accelerators/nvidia-h100-pcie]] | 80 | 2000 | **Y/Y/Y** | 100.0 | $43,000 | [[supply/used-h100-80gb-pcie-market]] | 430.00 |

\* Strix Halo is `unified_memory`: its record leaves `vram_gb` null; the 128 GB
shown is the configured LPDDR5x pool of the cheapest linked supply record
([[supply/new-minisforum-ms-s1-max-128gb]]). See §1 and §6.

### The headline read

- **The cheap-VRAM used-datacenter slice wins on $/tok-s and loses on
  everything else.** [[accelerators/nvidia-v100-sxm2-32gb]] at **$500** is the
  single cheapest 35B-A3B-capable price in the corpus, and
  [[accelerators/amd-instinct-mi100]] at **$900** is the cheapest part with both
  a `Y` on 35B *and* >1 TB/s of bandwidth (61.4 tok/s derived ceiling — faster
  than a 3090 for less money). Both are traps unless you read §5.
- **The honest mainstream answer is the RTX 3090.** $950 used
  ([[supply/ebay-used-rtx3090-oct2026]]), 24 GB exactly, 46.8 tok/s derived
  ceiling, a real PCIe card with a fan and a warranty-adjacent refurb channel.
  It is the first row that is both affordable *and* deployable without a
  baseboard or a ROCm workaround. The [[accelerators/amd-radeon-rx-7900-xtx]] now
  undercuts it on $/tok-s ($875 used, rank 7) but needs the ROCm path.
- **New in this pass — the sub-$1,000 24 GB+ slice thickened.** Three parts now
  fit 35B-A3B at new-retail or used prices that were not in the corpus before:
  [[accelerators/huawei-ascend-310p]] (24 GB, **$265**, Taobao used — cheapest
  fit by sticker, but a Da Vinci NPU with a 10.2 tok/s ceiling),
  [[accelerators/intel-arc-pro-b60]] (24 GB, $655, new retail), and
  [[accelerators/cambricon-mlu370]] (48 GB, $450 — the **cheapest part that fits
  the 70B stretch goal**, though at a 15.4 tok/s ceiling). The
  [[accelerators/intel-arc-pro-b70]] (32 GB, $949) is the cheapest *new* 32 GB
  card in the table.
- **The stretch goal (70B, needs ≥48 GB) has two new cheap answers.** By the
  letter of the fit rule the cheapest is the MLU370 at $450 — but its 307 GB/s
  bus and immature stack make it a capacity fit, not a usable one. The cheapest
  *usable* 70B/MoE fit is [[accelerators/amd-radeon-8060s-ryzen-ai-max]] (Strix
  Halo, 128 GB unified, **$1,900** for a whole mini-PC) — bandwidth-capped but
  the only sub-$2,000 part that both fits 70B *and* has a measured MoE decode on
  the target class (42 tok/s, §2). The cheapest *drop-in discrete* 70B fit is
  [[accelerators/intel-gaudi2]] at $2,100 — but it is an OAM module that cannot
  go in a PCIe slot (§5). After that: [[accelerators/apple-m3-ultra]] /
  [[accelerators/apple-m2-ultra]] at $3,700 (whole Mac Studios), then a big jump
  to [[accelerators/nvidia-a100-80gb-sxm4]] at $10,450.

## 4. Qualified but no purchase price — the "fits, can't price it" list

These records satisfy the bandwidth/fit rule but have **no bare-card or
system purchase price** in `data/supply/` (they may have a *rental* price in
the cloud records — that is a different join, see `tools/cost_per_token.py`).
Listed because a future supply pass can close them; sorted by 35B ceiling.

| accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* | note |
|---|---|---|---|---|---|
| [[accelerators/amd-instinct-mi300x]] | 192 | 5300 | Y/Y/Y | 265.0 | cloud-priced only ([[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]]) |
| [[accelerators/nvidia-h200-sxm]] | 141 | 4800 | Y/Y/Y | 240.0 | cloud/rental only |
| [[accelerators/nvidia-h20]] | 96 | 4000 | Y/Y/Y | 200.0 | export part; the 37-ridge decode specialist of [[09-model-memory-envelope]] |
| [[accelerators/intel-gaudi3]] | 128 | 3700 | Y/Y/Y | 185.0 | no price record |
| [[accelerators/xpu-intel-dc-gpu-max-1550]] | 128 | 3276 | Y/Y/Y | 163.8 | no price |
| [[accelerators/nvidia-rtx-6000-ada]] | 48 | 960 | **Y/Y/Y** | 48.0 | cloud/rental only; cheapest 48 GB *fit* by VRAM but thin BW |
| [[accelerators/nvidia-l40s]] | 48 | 864 | **Y/Y/Y** | 43.2 | cloud/rental only; second-highest ridge in the repo per docs/09 |
| [[accelerators/amd-radeon-pro-w7900]] | 48 | 864 | **Y/Y/Y** | 43.2 | no price record |

The full machine list (all 96 qualifying accelerators, the 59 unpriced included)
is reproducible with `python tools/fit_matrix_35b.py`.

## 5. Why the cheap rows are cheap — read before buying

The $/tok-s ranking is honest arithmetic on **asking prices for the part as
listed**, and four of the top six are cheap for a *reason the price doesn't
show*. These are the corpus's own gotcha records, not new claims:

- **SXM/OAM is not a card.** [[accelerators/nvidia-v100-sxm2-32gb]] (#1),
  [[accelerators/intel-gaudi2]] (#6), [[accelerators/amd-instinct-mi250]] and
  [[accelerators/amd-instinct-mi250x]] (#16/#17) are modules. Per
  [[gotchas/gotcha-sxm-module-needs-carrier-baseboard]], the SXM discount is a
  *form-factor* discount — the module needs a carrier
  baseboard + heatsink and has no PCIe edge. Gaudi2 is worse:
  [[gotchas/gotcha-gaudi2-hl225-oam-only-chassis-required]] records that HL-225
  has *no PCIe path at all* and needs a UBB chassis + fabric. Add that cost to
  the $/tok-s before comparing it to a PCIe row.
- **Pascal/Maxwell has no FP16 tensor path.** [[accelerators/nvidia-tesla-m40]]
  (#2) and [[accelerators/nvidia-tesla-p40]] (#5) look like steals at $189/$290,
  but [[gotchas/gotcha-pascal-p40-m40-no-fp16-tensor-dp4a-only]] is explicit:
  the M40's low sticker is a trap (no DP4A, slower than P40 on the only LLM path
  that matters), both are passive cards needing a shroud+blower, and their
  EPS-12V power connector is *not* a PCIe 8-pin. The P40's DP4A INT8 path is the
  floor buy; the M40 is not a cheaper P40.
- **MI100's $900 needs a ROCm tax.** [[gotchas/gotcha-mi100-multigpu-rocm-segfault]]
  records that MI100 is the best $/(GB·TB/s) in the used slice but multi-GPU
  segfaults on bare-metal ROCm and wants a PCIe-passthrough VM — and gfx908 is
  maintenance-tier on the ROCm matrix.
- **A "fits" flag is not a speed promise — and the new edge-NPU rows prove it.**
  The M40 fits 35B-A3B (24 GB) but its 288 GB/s ceiling is ~14 tok/s derived —
  barely interactive. The three cheapest *new-in-this-pass* fits are slower
  still: [[accelerators/huawei-ascend-310p]] (205 GB/s → 10.2 tok/s ceiling),
  [[accelerators/qualcomm-ai100]] (137 GB/s → 6.8 tok/s) and
  [[accelerators/cambricon-mlu370]] (307 GB/s → 15.4 tok/s) are NPUs/AIDC parts
  whose low sticker reflects an immature or edge-targeted software stack, not a
  serving answer — a `Y` on 70B (MLU370) means "the bytes fit," not "it serves."
  Conversely the unified-memory parts (Apple, Strix Halo) fit large models but
  are bandwidth-capped; [[gotchas/gotcha-strix-halo-moe-not-dense]] measures a
  dense-70B at ~5 tok/s on the ~256 GB/s Strix Halo wall, while the same record
  class measures the 35B-A3B **MoE** target at 42 tok/s
  ([[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]]). The MoE
  target is exactly the workload those parts *are* good for.

## 6. Assumptions and what this table deliberately ignores

- **Price class:** every purchase price is an *asking* price (used-market,
  reseller, or MSRP), not a realised transaction price. The repo stamps each
  with a `PRICE CLASS` and a `RE-VERIFY BY` date; the Oct-2026 used records are
  freshest. `$`/`tok-s` inherits that uncertainty.
- **Unit price ≠ system price.** Prices are per bare card/module where the
  record gives one. Several rows price a *whole system* instead — the part is
  not sold bare: [[supply/used-mac-studio-m1-ultra-128gb]] ($2,600),
  [[supply/used-mac-studio-m2-ultra-192gb]] ($3,700),
  [[supply/used-mac-studio-m3-ultra-96gb]] ($3,700) are whole Mac Studios, and
  all four Strix Halo rows ([[supply/new-minisforum-ms-s1-max-128gb]] $1,900,
  [[supply/new-framework-desktop-strix-halo-128gb]],
  [[supply/new-beelink-gtr9-pro-strix-halo-128gb]],
  [[supply/new-gmktec-evo-x2-strix-halo-128gb]]) are whole mini-PCs/boards, plus
  the DGX-1 (below). Dividing a whole-system price by a per-device ceiling is an
  apples/orange flagged here, not hidden — these parts are listed because no
  bare-silicon price exists, and for the unified-memory rows the *system* is the
  product.
- **Derived, not measured.** `docs/08-cost-per-token.md` joins *measured*
  benchmarks to *rental* prices; this doc joins *roofline ceilings* to
  *purchase* prices. They answer different questions and neither substitutes
  for the other. A `Y` here and a cheap $/tok-s do not mean the part serves the
  model well at concurrency.
- **Deliberately dropped joins** (the raw `accelerator_ids` link would
  over-attribute, per the same trap `tools/cost_per_token.py` documents):
  - `ebay-used-accelerator-market` ($10,449.95) is **one A100 80GB SXM4**
    listing; its H100/H200 ids are "obtainable," not "priced at $10,450." Not
    joined to H100/H200.
  - `ebay-used-a100-40gb-sxm4` ($2,499) prices a **40GB** module but points at
    the 80GB id (no 40GB record when written). Mapped to
    [[accelerators/nvidia-a100-40gb-sxm4]], not the 80GB part.
  - `microcenter-refurb-rtx-3090` is two tiers: 3090 $1,099.99 / 3090 Ti
    $1,239.99 — each joined at its own price.
  - `new-retail-rtx-5090-amazon` headline is the **5090** band; the 5080 sibling
    band lives in `notes` and is not the headline price. 5080 not joined.
  - `new-retail-rtx-4090-3090-consumer` headline $3,299 is the **4090** median;
    the 3090 median is $1,514 in notes. 3090 not joined at $3,299.
- **Whole-system rows kept for context but not mis-joined:**
  `used-dgx1-8x-v100` ($4,000) is an **8-GPU** server → ~$500/GPU-equivalent
  (derived), consistent with the $500 bare-SXM2 row but a *system* price.
  `ebay-used-a100-40gb-rack-liquidator` ($249,000) is a 32-GPU rack;
  `used-8x-h100-sxm-node-system` ($180,000) is an 8-GPU server. None are
  bare-card prices.

## 7. Reproduce

    python tools/fit_matrix_35b.py            # the full 96-row matrix, all three sections
    python tools/validate.py                  # record integrity (no new records added here)

Every figure above is generated by that tool against the committed `data/`
records; where the tool's arithmetic disagrees with a table cell, the tool is
right and the cell is stale. The tool also carries the two judgement calls the
table depends on, documented inline: the curated `PRICE_JOIN` (which corrects
the raw `accelerator_ids` over-attribution the same way
`tools/cost_per_token.py` does) and `UNIFIED_MEM_GB` (the configured-memory
override that keeps `vram_gb`-null unified-memory parts like Strix Halo from
being silently filtered out by the `>= 8 GB` gate).
