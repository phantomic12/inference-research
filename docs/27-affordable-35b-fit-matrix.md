# 27 — Affordable 35B-class fit matrix

Which recorded accelerators can hold a ~35B-class model, and what does the
cheapest one cost per unit of decode throughput.

> **Refresh note (docs/fit-matrix-oct14-r4).** Re-run on the merged Phase-A +
> Phase-C Oct-15 tree (~5,200 records; adds [[accelerators/apple-m2-max]] to
> the corpus since the pass-7 refresh). The ranked $/tok-s table (§3) is
> **unchanged** — no new purchase-price supply records landed, so the 41
> priced rows and their order are identical to the capacity-matched
> `oct16-final` pass. What this pass carries: §4's "fits, can't price it"
> list is now **52** rows (GH200, M4 Ultra and M2 Max all qualify on
> bandwidth but have no bare-part price), and §4b's roofline-vs-measured
> cross-check is kept and extended with the newest A3B-class measured runs —
> Apple M5 Max (two engine paths), M5 Pro, a second M4-Max run, and a
> stronger 72.4 tok/s Strix Halo row that widens its floor-multiplier to
> ~5.7×.
>
> *Prior refresh (docs/fit-matrix-oct14-r2 / oct16 passes):* re-run on the
> merged Phase-A + Phase-C pass-7 tree (4,831 records). Ranked table
> unchanged from pass 5 except capacity-matched Apple joins (§6); §4 grew
> 16→50 and §4b was added against the pass-6/7 measured A3B-MoE benchmarks.

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
yields **101 qualifying accelerators**, of which **41 have a usable purchase
price** and 60 are bandwidth-qualified but have no bare-card/system price in
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
| 1 | [[accelerators/nvidia-tesla-p100]] | 16 | 732 | -/-/- | 36.6 | $115 | [[supply/ebay-used-tesla-p100-16gb]] | **3.14** |
| 2 | [[accelerators/intel-arc-b580]] | 12 | 456 | -/-/- | 22.8 | $200 | [[supply/newegg-refreshed-arc-b580]] | 8.77 |
| 3 | [[accelerators/nvidia-v100-sxm2-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $500 | [[supply/ebay-used-tesla-v100-sxm2-32gb]] | 11.11 |
| 4 | [[accelerators/amd-instinct-mi100]] | 32 | 1228 | **Y/Y/-** | 61.4 | $720 | [[supply/ebay-used-amd-mi100-32gb]] | 11.73 |
| 5 | [[accelerators/nvidia-tesla-m40]] | 24 | 288 | Y/Y/- | 14.4 | $189 | [[supply/used-nvidia-tesla-m40-market]] | 13.12 |
| 6 | [[accelerators/intel-gaudi2]] | 96 | 2460 | **Y/Y/Y** | 123.0 | $2,000 | [[supply/ebay-used-gaudi2-hl225-oam-module]] | 16.26 |
| 7 | [[accelerators/nvidia-v100-pcie-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $750 | [[supply/ebay-used-tesla-v100-pcie-32gb]] | 16.67 |
| 8 | [[accelerators/nvidia-tesla-p40]] | 24 | 347 | Y/Y/- | 17.4 | $290 | [[supply/ebay-used-tesla-p40-24gb]] | 16.71 |
| 9 | [[accelerators/amd-radeon-rx-7900-xtx]] | 24 | 960 | **Y/Y/-** | 48.0 | $875 | [[supply/ebay-used-rx-7900-xtx-market-oct2026]] | 18.23 |
| 10 | [[accelerators/nvidia-rtx-3090-ti]] | 24 | 1008 | Y/Y/- | 50.4 | $950 | [[supply/used-rtx-3090-ti-market-oct2026]] | 18.85 |
| 11 | [[accelerators/nvidia-rtx-3090]] | 24 | 936 | **Y/Y/-** | 46.8 | $950 | [[supply/ebay-used-rtx3090-oct2026]] | 20.30 |
| 12 | [[accelerators/nvidia-rtx-5090]] | 32 | 1792 | **Y/Y/-** | 89.6 | $1,999 | [[supply/nvidia-marketplace-geforce]] | 22.31 |
| 13 | [[accelerators/huawei-ascend-310p]] | 24 | 205 | Y/Y/- | 10.2 | $265 | [[supply/taobao-used-huawei-ascend-310p]] | 25.88 |
| 14 | [[accelerators/amd-radeon-rx-9070-xt]] | 16 | 640 | -/-/- | 32.0 | $829 | [[supply/used-rx-9070-xt-market]] | 25.91 |
| 15 | [[accelerators/intel-arc-pro-b60]] | 24 | 456 | Y/Y/- | 22.8 | $650 | [[supply/new-arc-pro-b60-24gb-retail-oct2026]] | 28.51 |
| 16 | [[accelerators/cambricon-mlu370]] | 48 | 307 | Y/Y/Y | 15.4 | $450 | [[supply/ebay-used-cambricon-mlu370]] | 29.30 |
| 17 | [[accelerators/amd-instinct-mi250]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $4,950 | [[supply/ebay-used-amd-mi250x-128gb-oam]] | 30.94 |
| 18 | [[accelerators/amd-instinct-mi250x]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $4,950 | [[supply/ebay-used-amd-mi250x-128gb-oam]] | 30.94 |
| 19 | [[accelerators/intel-arc-pro-b70]] | 32 | 608 | Y/Y/- | 30.4 | $949 | [[supply/newegg-new-arc-pro-b70-32gb]] | 31.22 |
| 20 | [[accelerators/nvidia-a100-40gb-sxm4]] | 40 | 1555 | Y/Y/- | 77.8 | $2,499 | [[supply/ebay-used-a100-40gb-sxm4]] | 32.14 |
| 21 | [[accelerators/nvidia-rtx-4090]] | 24 | 1008 | Y/Y/- | 50.4 | $2,000 | [[supply/ebay-used-consumer-gpu-market]] | 39.68 |
| 22 | [[accelerators/amd-instinct-mi210]] | 64 | 1600 | **Y/Y/Y** | 80.0 | $4,000 | [[supply/ebay-used-amd-mi210-64gb]] | 50.00 |
| 23 | [[accelerators/tenstorrent-wormhole-n300]] | 24 | 576 | Y/Y/- | 28.8 | $1,449 | [[supply/new-tenstorrent-wormhole-n300]] | 50.31 |
| 24 | [[accelerators/tenstorrent-blackhole-p150]] | 32 | 512 | Y/Y/- | 25.6 | $1,399 | [[supply/new-tenstorrent-blackhole-p150]] | 54.65 |
| 25 | [[accelerators/qualcomm-ai100]] | 32 | 137 | Y/Y/- | 6.8 | $400 | [[supply/used-qualcomm-cloud-ai100-pcie]] | 58.39 |
| 26 | [[accelerators/cambricon-mlu270]] | 16 | 102 | -/-/- | 5.1 | $317 | [[supply/ebay-used-cambricon-mlu270-s4]] | 61.91 |
| 27 | [[accelerators/apple-m1-ultra]] | 128 | 800 | **Y/Y/Y** | 40.0 | $2,500 | [[supply/used-mac-studio-m1-ultra-128gb]] | 62.50 |
| 28 | [[accelerators/tenstorrent-grayskull-e150]] | 8 | 118 | -/-/- | 5.9 | $400 | [[supply/used-tenstorrent-grayskull-e150]] | 67.80 |
| 29 | [[accelerators/tenstorrent-wormhole-n150]] | 12 | 288 | -/-/- | 14.4 | $999 | [[supply/new-tenstorrent-wormhole-n150]] | 69.38 |
| 30 | [[accelerators/nvidia-h100-sxm]] | 80 | 3350 | **Y/Y/Y** | 167.5 | $12,999 | [[supply/ebay-used-h100-sxm5-module]] | 77.61 |
| 31 | [[accelerators/nvidia-a100-40gb-pcie]] | 40 | 1555 | Y/Y/- | 77.8 | $7,000 | [[supply/ebay-used-a100-40gb-pcie-card]] | 90.03 |
| 32 | [[accelerators/apple-m2-ultra]] | 192 | 800 | **Y/Y/Y** | 40.0 | $3,700 | [[supply/used-mac-studio-m2-ultra-192gb]] | 92.50 |
| 33 | [[accelerators/nvidia-rtx-pro-6000-blackwell]] | 96 | 1792 | **Y/Y/Y** | 89.6 | $8,650 | [[supply/new-rtx-pro-6000-blackwell-96gb]] | 96.54 |
| 34 | [[accelerators/nvidia-a100-80gb-sxm4]] | 80 | 2039 | **Y/Y/Y** | 102.0 | $10,450 | [[supply/ebay-used-accelerator-market]] | 102.50 |
| 35 | [[accelerators/nvidia-a100-80gb-pcie]] | 80 | 1935 | **Y/Y/Y** | 96.8 | $12,099 | [[supply/ebay-used-a100-80gb-pcie]] | 125.05 |
| 36 | [[accelerators/amd-radeon-8060s-ryzen-ai-max]] | 128* | 256 | **Y/Y/Y** | 12.8 | $1,900 | [[supply/new-minisforum-ms-s1-max-128gb]] | 148.44 |
| 37 | [[accelerators/amd-ryzen-ai-max-395-framework-desktop]] | 128 | 256 | **Y/Y/Y** | 12.8 | $2,099 | [[supply/new-beelink-gtr9-pro-strix-halo-128gb-2]] | 163.98 |
| 38 | [[accelerators/apple-m3-max]] | 128 | 400 | **Y/Y/Y** | 20.0 | $3,400 | [[supply/used-macbook-pro-m3-max-128gb]] | 170.00 |
| 39 | [[accelerators/apple-m4-max]] | 128 | 546 | **Y/Y/Y** | 27.3 | $4,700 | [[supply/used-macbook-pro-m4-max-128gb]] | 172.16 |
| 40 | [[accelerators/apple-m3-ultra]] | 512 | 819 | **Y/Y/Y** | 41.0 | $9,000 | [[supply/used-mac-studio-m3-ultra-512gb]] | 219.78 |
| 41 | [[accelerators/nvidia-h100-pcie]] | 80 | 2000 | **Y/Y/Y** | 100.0 | $43,000 | [[supply/used-h100-80gb-pcie-market]] | 430.00 |

\* The 8060s Strix Halo row is `unified_memory`: its record leaves `vram_gb`
null; the 128 GB shown is the configured LPDDR5x pool of the cheapest linked
supply record ([[supply/new-minisforum-ms-s1-max-128gb]]). The sibling
`amd-ryzen-ai-max-395-framework-desktop` record carries an explicit
`vram_gb=128` for the same silicon, priced at the Beelink whole-system row.
See §1 and §6.

### The headline read

- **The cheapest *fits* answer is still the used-datacenter V100 SXM2 — but the
  raw $/tok-s top two don't fit any target column at all.**
  [[accelerators/nvidia-tesla-p100]] at **$115** and
  [[accelerators/intel-arc-b580]] at **$200** rank #1/#2 on $/tok-s because they
  are cheap, yet at 16 GB / 12 GB both are `-/-/-`: 16 GB < the 20 GB the 27B
  fit needs once headroom is added, and nowhere near the 24 GB 35B asks for.
  They are in the table because the selection rule lets a sub-$3,000 price
  speak, and they rank high — but for the 35B-A3B job they are a *near-miss*,
  not an answer. The first row that actually fits **35B-A3B** is
  [[accelerators/nvidia-v100-sxm2-32gb]] at **$500** (rank 3), and
  [[accelerators/amd-instinct-mi100]] at **$720** (rank 4) is the cheapest part
  with both a `Y` on 35B *and* >1 TB/s of bandwidth (61.4 tok/s derived
  ceiling — faster than a 3090 for less money). All are traps unless you
  read §5.
- **The honest mainstream answer is the RTX 3090 — now matched by its Ti.** Both
  the 3090 ($950 used, [[supply/ebay-used-rtx3090-oct2026]]) and the 3090 Ti
  ($950 used, [[supply/used-rtx-3090-ti-market-oct2026]]) sit at $950 for 24 GB;
  the Ti's extra 72 GB/s of bandwidth buys a 50.4 vs 46.8 tok/s derived ceiling
  at the same price, so it edges the 3090 to rank 10. Both are real PCIe cards
  with a fan and a refurb channel — the first rows that are affordable *and*
  deployable without a baseboard or a ROCm workaround. The
  [[accelerators/amd-radeon-rx-7900-xtx]] still undercuts both on $/tok-s
  ($875 used, rank 9) but needs the ROCm path.
- **New in this pass — the used-Tesla and Tenstorrent slices filled in, and the
  Oct-14 ingest repriced four rows.** The
  pass-2 ingest added bare-card used prices that were missing before:
  [[accelerators/nvidia-tesla-p100]] (16 GB, **$115** — cheapest sticker in the
  whole table), [[accelerators/tenstorrent-wormhole-n300]] (24 GB, $1,449, a
  *new* retail card that fits 35B), [[accelerators/cambricon-mlu270]] (16 GB,
  $317), and [[accelerators/tenstorrent-grayskull-e150]] (8 GB, $400). The
  merged Oct-14 ingest also re-priced existing records: MI100 to **$720** (was
  $900, now rank 4 and the cheapest >1 TB/s fit), MI210 to **$4,000** (was
  $4,180), the M1 Ultra Studio to **$2,500** (was $2,600), Gaudi2 to **$2,000**
  (was $2,100), MI250/X to **$4,950** (was $5,450), A100-40GB-PCIe to **$7,000**
  (was $7,500), RTX PRO 6000 to **$8,650** (was $16,624 — nearly halved), 3090
  Ti to **$950** (was $1,240). The second Strix Halo record
  ([[accelerators/amd-ryzen-ai-max-395-framework-desktop]]) is the same silicon
  as the 8060s row, priced at the Beelink whole-system $2,099.
- **Capacity-matched joins added three Apple rows and corrected one.** A
  whole-system price is only honest when the configured pool matches the
  `vram_gb` the fit flag claims. Joined this pass: [[accelerators/apple-m3-max]]
  at **$3,400** (128 GB MacBook Pro SKU = record's 128 GB) and
  [[accelerators/apple-m4-max]] at **$4,700** (128 GB SKU) — both fit all three
  targets. Corrected: [[accelerators/apple-m3-ultra]]'s record carries
  `vram_gb=512`; the $3,700 96 GB Studio row priced *less pool than the fit flag
  claims*, so it is replaced by the capacity-matched 512 GB SKU at **$9,000**
  ([[supply/used-mac-studio-m3-ultra-512gb]]). The same rule drops the
  $1,850 64 GB M1-Ultra Studio and the $3,750 192 GB Strix Halo board — both
  price a smaller pool than their record's `vram_gb` (§6).
- **The stretch goal (70B, needs ≥48 GB) got *cheaper*.** The cheapest
  single-accelerator 70B fit by sticker is still the MLU370 at $450 — but its
  307 GB/s bus makes it a capacity fit, not a usable one. The biggest move is
  [[accelerators/nvidia-rtx-pro-6000-blackwell]] (96 GB): a second supply
  channel priced it at **$8,650**, half the prior $16,624 B&H row, dropping it
  to rank 33 — now the cheapest *drop-in* 70B card with >1 TB/s of bandwidth
  that is also a normal PCIe workstation card. Below it: Strix Halo at $1,900
  (whole mini-PC, bandwidth-capped — see §5) and the drop-in-but-OAM Gaudi2 at
  $2,000 (§5). Then [[accelerators/apple-m1-ultra]] /
  [[accelerators/apple-m2-ultra]] at $2,500/$3,700 (whole Mac Studios),
  [[accelerators/apple-m3-max]] / [[accelerators/apple-m4-max]] at
  $3,400/$4,700 (whole MacBooks),
  [[accelerators/amd-instinct-mi210]] at $4,000, the MI250/X OAM pair at
  $4,950, and a jump to [[accelerators/nvidia-a100-80gb-sxm4]] at $10,450.

## 4. Qualified but no purchase price — the "fits, can't price it" list

These records satisfy the bandwidth/fit rule but have **no bare-card or
system purchase price** in `data/supply/` (they may have a *rental* price in
the cloud records — that is a different join, see `tools/cost_per_token.py`).
Listed because a future supply pass can close them; sorted by 35B ceiling.
The merged Oct-14/15 ingest pushed the rest of the accelerator corpus up here —
the whole Blackwell/TPU-v6/v7/MI350X/GH200/Trainium/CloudMatrix/Ascend
generation now qualifies on bandwidth but has no bare-card asking price. The
giant first row is the 384-chip CloudMatrix supernode: it fits trivially and
its ceiling is a rack-scale figure, not a buyable part.

| accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* | note |
|---|---|---|---|---|---|
| [[accelerators/huawei-ascend-910c-cloudmatrix-384]] | 49152 | 1228800 | Y/Y/Y | 61440.0 | 384-chip supernode; rack-scale, not a bare card |
| [[accelerators/amd-instinct-mi350x]] | 288 | 8000 | Y/Y/Y | 400.0 | no price record |
| [[accelerators/amd-instinct-mi355x]] | 288 | 8000 | Y/Y/Y | 400.0 | no price record |
| [[accelerators/nvidia-gb200-nvl72]] | 186 | 8000 | Y/Y/Y | 400.0 | rack unit, not a bare card |
| [[accelerators/nvidia-gb300]] | 279 | 8000 | Y/Y/Y | 400.0 | rack unit, not a bare card |
| [[accelerators/nvidia-b200]] | 180 | 7700 | Y/Y/Y | 385.0 | no price record |
| [[accelerators/nvidia-b300]] | 270 | 7700 | Y/Y/Y | 385.0 | no price record |
| [[accelerators/google-tpu-ironwood]] | 192 | 7380 | Y/Y/Y | 369.0 | cloud only |
| [[accelerators/amd-instinct-mi325x]] | 256 | 6000 | Y/Y/Y | 300.0 | no price record |
| [[accelerators/amd-instinct-mi300a]] | 128 | 5300 | Y/Y/Y | 265.0 | no price record |
| [[accelerators/amd-instinct-mi300x]] | 192 | 5300 | Y/Y/Y | 265.0 | cloud-priced only ([[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]]) |
| [[accelerators/aws-trainium3]] | 144 | 4900 | Y/Y/Y | 245.0 | cloud only |
| [[accelerators/nvidia-h200-nvl]] | 141 | 4800 | Y/Y/Y | 240.0 | cloud/rental only |
| [[accelerators/nvidia-h200-sxm]] | 141 | 4800 | Y/Y/Y | 240.0 | cloud/rental only |
| [[accelerators/rebellions-rebel100]] | 144 | 4800 | Y/Y/Y | 240.0 | no price record |
| [[accelerators/amd-instinct-mi350p]] | 144 | 4000 | Y/Y/Y | 200.0 | no price record |
| [[accelerators/nvidia-gh200]] | 96 | 4000 | Y/Y/Y | 200.0 | Grace Hopper superchip; system, not a bare card |
| [[accelerators/nvidia-h20]] | 96 | 4000 | Y/Y/Y | 200.0 | export part; the 37-ridge decode specialist of [[09-model-memory-envelope]] |
| [[accelerators/nvidia-h100-nvl]] | 94 | 3938 | Y/Y/Y | 196.9 | no price record |
| [[accelerators/intel-gaudi3]] | 128 | 3700 | Y/Y/Y | 185.0 | no price record |
| [[accelerators/xpu-intel-dc-gpu-max-1550]] | 128 | 3276 | Y/Y/Y | 163.8 | no price |
| [[accelerators/huawei-ascend-910c]] | 128 | 3200 | Y/Y/Y | 160.0 | no price record |
| [[accelerators/aws-trainium2]] | 96 | 2875 | Y/Y/Y | 143.8 | cloud only |
| [[accelerators/google-tpu-v5p]] | 95 | 2765 | Y/Y/Y | 138.2 | cloud only |
| [[accelerators/cambricon-mlu590]] | 192 | 2400 | Y/Y/Y | 120.0 | no price record |
| [[accelerators/google-tpu-v6e]] | 32 | 1638 | Y/Y/- | 81.9 | cloud only |
| [[accelerators/biren-br100]] | 64 | 1600 | Y/Y/Y | 80.0 | no price record |
| [[accelerators/huawei-ascend-910b]] | 64 | 1400 | Y/Y/Y | 70.0 | no price record |
| [[accelerators/google-tpu-v4]] | 32 | 1200 | Y/Y/- | 60.0 | cloud only |
| [[accelerators/local-apple-m5-ultra]] | 512 | 1200 | Y/Y/Y | 60.0 | unreleased/rumoured part; no price |
| [[accelerators/amd-radeon-instinct-mi50x]] | 32 | 1024 | Y/Y/- | 51.2 | no price record |
| [[accelerators/amd-radeon-instinct-mi60]] | 32 | 1024 | Y/Y/- | 51.2 | no price record |
| [[accelerators/nvidia-rtx-6000-ada]] | 48 | 960 | **Y/Y/Y** | 48.0 | cloud/rental only; cheapest 48 GB *fit* by VRAM but thin BW |
| [[accelerators/nvidia-a30]] | 24 | 933 | Y/Y/- | 46.6 | no price record |
| [[accelerators/google-tpu-v3]] | 32 | 900 | Y/Y/- | 45.0 | cloud only |
| [[accelerators/nvidia-rtx-pro-4500-blackwell]] | 32 | 896 | Y/Y/- | 44.8 | no price record |
| [[accelerators/nvidia-l40s]] | 48 | 864 | **Y/Y/Y** | 43.2 | cloud/rental only; second-highest ridge in the repo per docs/09 |
| [[accelerators/amd-radeon-pro-w7900]] | 48 | 864 | **Y/Y/Y** | 43.2 | no price record |
| [[accelerators/aws-inferentia2]] | 32 | 820 | Y/Y/- | 41.0 | cloud only |
| [[accelerators/apple-m4-ultra]] | 192 | 819 | Y/Y/Y | 41.0 | whole Mac Studio; no bare-silicon price |
| [[accelerators/aws-trainium1]] | 32 | 800 | Y/Y/- | 40.0 | cloud only |
| [[accelerators/moore-threads-s4000]] | 48 | 768 | Y/Y/Y | 38.4 | no price record |
| [[accelerators/amd-radeon-ai-pro-r9700]] | 32 | 640 | Y/Y/- | 32.0 | no price record |
| [[accelerators/cambricon-mlu370-x8]] | 48 | 614.4 | Y/Y/Y | 30.7 | no price record |
| [[accelerators/local-apple-m5-max]] | 128 | 614 | Y/Y/Y | 30.7 | unreleased/rumoured part; no price |
| [[accelerators/nvidia-a10]] | 24 | 600 | Y/Y/- | 30.0 | no price record |
| [[accelerators/qualcomm-ai200]] | 128 | 548 | Y/Y/Y | 27.4 | no price record |
| [[accelerators/huawei-ascend-950dt]] | 96 | 500 | Y/Y/Y | 25.0 | no price record |
| [[accelerators/huawei-atlas-300i-duo]] | 96 | 408 | Y/Y/Y | 20.4 | no price record |
| [[accelerators/apple-m2-max]] | 96 | 400 | Y/Y/Y | 20.0 | whole MacBook; no capacity-matched price |
| [[accelerators/local-apple-m5-pro]] | 64 | 307 | Y/Y/Y | 15.3 | unreleased/rumoured part; no price |
| [[accelerators/nvidia-l4]] | 24 | 300 | Y/Y/- | 15.0 | no price record |

### Qualified on bandwidth but cannot hold 35B-A3B (all unpriced)

These records clear the `>= 300 GB/s` bandwidth floor but not the 24 GB
resident-weights fit; listed for completeness so the price pass does not
mistake a fast small card for an answer. `-/-/-` means it holds none of the
three targets:

| accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* |
|---|---|---|---|---|
| [[accelerators/amd-radeon-pro-vii]] | 16 | 1024 | -/-/- | 51.2 |
| [[accelerators/nvidia-rtx-5080]] | 16 | 960 | -/-/- | 48.0 |
| [[accelerators/nvidia-rtx-5070-ti]] | 16 | 896 | -/-/- | 44.8 |
| [[accelerators/google-tpu-v5e]] | 16 | 819 | -/-/- | 41.0 |
| [[accelerators/nvidia-rtx-4080]] | 16 | 716.8 | -/-/- | 35.8 |
| [[accelerators/google-tpu-v2]] | 16 | 700 | -/-/- | 35.0 |
| [[accelerators/local-nvidia-rtx-5070]] | 12 | 672 | -/-/- | 33.6 |
| [[accelerators/xpu-intel-dc-gpu-flex-170]] | 16 | 576 | -/-/- | 28.8 |

The full machine list (all 101 qualifying accelerators, the 60 unpriced
included) is reproducible with `python tools/fit_matrix_35b.py`.

## 4b. Roofline vs. measured — the A3B-MoE cross-check

The 35B ceiling column divides by the **full resident** 20 GB, which makes it a
dense-equivalent *floor* for a MoE that reads only ~3B active per token (§2).
The Phase-C benchmark backfill landed **22+ third-party `decode_tok_s` records
on real A3B-class MoE models** (Qwen3-30B-A3B / Qwen3.5/3.6-35B-A3B, Q4-class
quants, batch-1 per-request) — enough to check how far off the floor each part
actually lands. `measured ÷ ceiling` below is derived arithmetic; the measured
numbers themselves come from the cited records.

| accelerator | BW GB/s | ceiling *(derived)* | measured tok/s | × over floor *(derived)* | benchmark record |
|---|---|---|---|---|---|
| [[accelerators/nvidia-rtx-5090]] | 1792 | 89.6 | 234.3 | 2.6× | [[benchmarks/hc-rtx5090-qwen3-30b-a3b-q4kxl-4k-tg128]] |
| [[accelerators/nvidia-rtx-4090]] | 1008 | 50.4 | 109.6 | 2.2× | [[benchmarks/hwcorner-rtx-4090-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/nvidia-rtx-3090-ti]] | 1008 | 50.4 | 98.3 | 1.9× | [[benchmarks/hwcorner-rtx-3090-ti-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/nvidia-rtx-6000-ada]] | 960 | 48.0 | 104.9 | 2.2× | [[benchmarks/hwcorner-rtx-6000-ada-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/nvidia-rtx-5070-ti]] | 896 | 44.8 | 111.1 | 2.5× | [[benchmarks/hwcorner-rtx-5070-ti-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/nvidia-rtx-4080]] | 716.8 | 35.8 | 85.0 | 2.4× | [[benchmarks/hwcorner-rtx-4080-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/local-apple-m5-max]] | 614 | 30.7 | 128.0 | 4.2× | [[benchmarks/ss-m5max-48-qwen35-35b-mlx-4bit-tg]] *(MLX path; llama.cpp Q4_K_M row lands 89.4 → 2.9×, [[benchmarks/ss-m5max-48-qwen35-35b-llamacpp-q4km-tg]])* |
| [[accelerators/apple-m4-max]] | 546 | 27.3 | 113.3 | 4.2× | [[benchmarks/mlx-m4-max-qwen3-30b-a3b-4bit-decode-tok-s]] |
| [[accelerators/apple-m4-max]] | 546 | 27.3 | 92.1 | 3.4× | [[benchmarks/ss-m4max-40core-qwen3-30b-mlx-q4-tg]] *(second, community run on the same part)* |
| [[accelerators/amd-epyc-9454p-genoa]] | 460.8 | 23.0 | 63.1 | 2.7× | [[benchmarks/comm-epyc-9454p-qwen3-moe-30b-decode]] |
| [[accelerators/local-apple-m5-pro]] | 307 | 15.3 | 41.9 | 2.7× | [[benchmarks/ss-m5pro-64-qwen35-35b-llamacpp-q4kl-tg]] |
| [[accelerators/amd-radeon-8060s-ryzen-ai-max]] | 256 | 12.8 | 72.4 | 5.7× | [[benchmarks/lhl-8060s-qwen3-30b-a3b-ud-q4-k-xl-vulkan-tg1]] *(best of 20 Strix Halo runs; the pass-7 row [[benchmarks/kyuz0-strix-8060s-qwen36-35b-a3b-q4kxl-vulkan-radv-tg128]] at 60.4 → 4.7× still stands)* |

Three reads:

- **The ordering is the same.** No measured row overturns the bandwidth
  ordering — a ~2–6× multiplier on a floor does not reorder the ranking, it
  lifts the low-BW rows hardest. The RTX 5090 still leads on both lists.
- **The floor is ~2–6× low, and *where* it is lowest is systematic.** The
  tightest-margin parts (Strix Halo now at 5.7× on its best run, M5 Max and
  M4 Max at 4.2×) are exactly the bandwidth-starved ones — the MoE reads only
  ~3B/token, so a 256 GB/s pool sustains ~72 tok/s real where the
  dense-equivalent floor says 12.8. The discrete GPUs cluster at ~2–2.6×
  because KV reads and kernel overhead bite closer to the roofline. The
  engine matters too: the same M5 Max measures 128.0 tok/s under MLX but
  89.4 under llama.cpp — a 1.4× swing on one part, wider than some
  cross-part gaps. **Do not use the ceiling column to compare a MoE target
  across architectures** — it ranks $/tok-s correctly only because the error
  is monotonic in bandwidth.
- **The cheapest *measured* A3B answer is still the 3090-Ti class.** Of the
  parts with both a price and a measured run, [[accelerators/nvidia-rtx-3090-ti]]
  at $950 with 98.3 measured tok/s (~$9.67/measured-tok/s, derived) is the
  cheapest verified fit — the roofline ranks it #10 on $/tok-s, the
  measurement says its real margin over the floor (1.9×) is the *worst* of
  the measured set, and it still beats everything cheaper that was measured.
  Strix Halo's 5.7× multiplier is why [[gotchas/gotcha-strix-halo-moe-not-dense]]
  calls the MoE target the workload that part is *for*.

## 5. Why the cheap rows are cheap — read before buying

The $/tok-s ranking is honest arithmetic on **asking prices for the part as
listed**, and four of the top six are cheap for a *reason the price doesn't
show*. These are the corpus's own gotcha records, not new claims:

- **SXM/OAM is not a card.** [[accelerators/nvidia-v100-sxm2-32gb]] (#3),
  [[accelerators/intel-gaudi2]] (#6), [[accelerators/amd-instinct-mi250]] and
  [[accelerators/amd-instinct-mi250x]] (#17/#18) are modules. Per
  [[gotchas/gotcha-sxm-module-needs-carrier-baseboard]], the SXM discount is a
  *form-factor* discount — the module needs a carrier
  baseboard + heatsink and has no PCIe edge. Gaudi2 is worse:
  [[gotchas/gotcha-gaudi2-hl225-oam-only-chassis-required]] records that HL-225
  has *no PCIe path at all* and needs a UBB chassis + fabric. Add that cost to
  the $/tok-s before comparing it to a PCIe row.
- **Pascal/Maxwell has no FP16 tensor path.** [[accelerators/nvidia-tesla-m40]]
  (#5) and [[accelerators/nvidia-tesla-p40]] (#8) look like steals at $189/$290,
  but [[gotchas/gotcha-pascal-p40-m40-no-fp16-tensor-dp4a-only]] is explicit:
  the M40's low sticker is a trap (no DP4A, slower than P40 on the only LLM path
  that matters), both are passive cards needing a shroud+blower, and their
  EPS-12V power connector is *not* a PCIe 8-pin. The P40's DP4A INT8 path is the
  floor buy; the M40 is not a cheaper P40.
- **MI100's $720 needs a ROCm tax.** [[gotchas/gotcha-mi100-multigpu-rocm-segfault]]
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
  not sold bare: [[supply/used-mac-studio-m1-ultra-128gb]] ($2,500),
  [[supply/used-mac-studio-m2-ultra-192gb]] ($3,700),
  [[supply/used-mac-studio-m3-ultra-512gb]] ($9,000) are whole Mac Studios,
  [[supply/used-macbook-pro-m3-max-128gb]] ($3,400) and
  [[supply/used-macbook-pro-m4-max-128gb]] ($4,700) are whole MacBook Pros, and
  all six Strix Halo rows ([[supply/new-minisforum-ms-s1-max-128gb]] $1,900,
  [[supply/new-framework-desktop-strix-halo-128gb]],
  [[supply/new-beelink-gtr9-pro-strix-halo-128gb]],
  [[supply/new-gmktec-evo-x2-strix-halo-128gb]],
  [[supply/new-beelink-gtr9-pro-strix-halo-128gb-2]],
  [[supply/new-gmktec-evo-x3-strix-halo-128gb]]) are whole mini-PCs/boards, plus
  the DGX-1 (below). Dividing a whole-system price by a per-device ceiling is an
  apples/orange flagged here, not hidden — these parts are listed because no
  bare-silicon price exists, and for the unified-memory rows the *system* is the
  product.
- **Derived, not measured.** `docs/08-cost-per-token.md` joins *measured*
  benchmarks to *rental* prices; this doc joins *roofline ceilings* to
  *purchase* prices. They answer different questions and neither substitutes
  for the other. A `Y` here and a cheap $/tok-s do not mean the part serves the
  model well at concurrency.
- **Capacity-matched joins only for whole-system rows.** A whole-system price
  is joined only when the configured memory matches the accelerator record's
  `vram_gb` — otherwise the sticker would price a smaller pool than the fit
  flag claims. Dropped on that rule: `used-mac-studio-m1-ultra-64gb` ($1,850,
  64 GB vs the record's 128 GB), `used-mac-studio-m3-ultra-96gb` ($3,700) and
  `used-mac-studio-m3-ultra-256gb` ($6,250) (both below the record's 512 GB —
  the joined row is `used-mac-studio-m3-ultra-512gb` at $9,000), and
  `new-framework-desktop-strix-halo-192gb-pro-495` ($3,750, 192 GB vs the
  record's explicit `vram_gb=128`). `apple-m2-max` (`vram_gb=96`) has no
  capacity-matched whole-system supply record at all, so it stays unpriced.
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
  `used-dgx1-8x-v100` and `used-dgx1-8xv100-32gb-chassis` ($4,000) are **8-GPU**
  servers → ~$500/GPU-equivalent
  (derived), consistent with the $500 bare-SXM2 row but *system* prices.
  `gaudi2-8x-oam-tray-supermicro-820gh` ($11,995) is an **8-module** OAM tray →
  ~$1,500/module-equivalent (derived), a system price not a module price.
  `ebay-used-a100-40gb-rack-liquidator` ($249,000) is a 32-GPU rack;
  `used-8x-h100-sxm-node-system` ($180,000) is an 8-GPU server. None are
  bare-card prices.
- **CPU-host records are the class the `vram_gb` gate is built around.**
  `amd-epyc-9554-genoa`, `amd-epyc-9454p-genoa` and `amd-epyc-9334-genoa-dual`
  are `vram_gb`-null `unified_memory` host CPUs — their resident pool is the
  DDR5 channels, not a VRAM figure the record can honestly report. Their linked
  supply rows price the CPU alone (`used-epyc-9354-genoa-cpu`, $2,000 — no RAM,
  so no resident pool) or a mismatched self-build
  (`used-epyc-9004-genoa-build-384gb`, $7,500 for CPU+board+384 GB), so neither
  is joined. They sit in §4's class even though the tool cannot list them:
  the gotcha their records carry — CPU decode is channel-bound, ~7 tok/s on
  70B Q4 on a 12-channel Genoa — is the answer for that workload class.
- **Edge/SBC whole-board prices are not accelerator prices.**
  `new-raspberry-pi-ai-hat-2` ($200, Hailo-10H + a Pi 5),
  `new-rockchip-rk3588-sbc` ($140, RK3588 board) and
  `used-google-coral-edge-tpu` ($90) price a *whole single-board computer*, not
  the NPU alone — and their accelerator records carry no `vram_gb >= 8` anyway
  (Hailo-10H has 8 GB but no `memory_bandwidth_gbps`; the other two are
  `vram_gb`-null unified-memory SoCs), so they would not qualify even if joined.
  Left unjoined so the price column stays a GPU/part price.

## 7. Reproduce

    python tools/fit_matrix_35b.py            # the full 101-row matrix, all three sections
    python tools/validate.py                  # record integrity (no new records added here)

Every figure above is generated by that tool against the committed `data/`
records; where the tool's arithmetic disagrees with a table cell, the tool is
right and the cell is stale. The tool also carries the two judgement calls the
table depends on, documented inline: the curated `PRICE_JOIN` (which corrects
the raw `accelerator_ids` over-attribution the same way
`tools/cost_per_token.py` does) and `UNIFIED_MEM_GB` (the configured-memory
override that keeps `vram_gb`-null unified-memory parts like Strix Halo from
being silently filtered out by the `>= 8 GB` gate).
