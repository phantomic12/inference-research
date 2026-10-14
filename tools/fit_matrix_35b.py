#!/usr/bin/env python3
"""Build the per-accelerator fit matrix used by docs/27-affordable-35b-fit-matrix.md.

Reads data/accelerators/ + data/supply/, applies the selection rule from the doc
(vram_gb >= 8 AND (memory_bandwidth_gbps >= 300 OR a linked purchase-price
supply record <= $3000)), computes the three fit flags and the rough decode
roofline ceiling (memory_bandwidth_gbps / model_weights_gb), and prints the
ranked rows. Every number it prints is either copied verbatim from a record or
derived by the arithmetic shown in the doc — nothing is measured.
"""
import json, glob, sys, os

REPO = sys.argv[1] if len(sys.argv) > 1 else "."

MODELS = [
    # key, label, weights_gb, kv_headroom_gb
    ("35b", "35B-A3B MoE, Q4", 20, 4),
    ("27b", "27B dense, Q4",   16, 4),
    ("70b", "70B dense, Q4",   40, 8),
]

def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(REPO, d, "*.json"))):
        r = json.load(open(f))
        out[r["id"]] = r
    return out

accs = load("data/accelerators")
supplies = load("data/supply")

def is_purchase(r):
    """Bare-card / module / system purchase price, not a per-hour rental."""
    if r.get("price_usd") is None:
        return False
    if r.get("kind") == "cloud":
        return False
    pb = (r.get("price_basis") or "").lower()
    if "kwh" in pb or "per kwh" in pb:
        return False
    return True

# ---------------------------------------------------------------------------
# PRICE_JOIN: curated (supply_id, accelerator_id) -> per-UNIT USD price.
#
# data/supply records use `accelerator_ids` to list every part a CHANNEL can
# supply, not every part the headline price_usd covers. tools/cost_per_token.py
# documents the same over-attribution trap. The joins below are the ones the
# record's own price_basis/notes actually price; anything a raw join would
# over-attribute is dropped. Each entry notes the correction inline.
# ---------------------------------------------------------------------------
PRICE_JOIN = {
    # bare-card / module, single clear part --------------------------------
    "ebay-used-tesla-p100-16gb":        {"nvidia-tesla-p100": 115.0},
    "newegg-refreshed-arc-b580":        {"intel-arc-b580": 199.99},
    "ebay-used-intel-arc-b580":         {"intel-arc-b580": 290.0},
    "microcenter-in-store-gpus":        {"intel-arc-b580": 309.99},
    "used-intel-arc-b580-market":       {"intel-arc-b580": 329.99},
    "ebay-used-tesla-v100-sxm2-32gb":   {"nvidia-v100-sxm2-32gb": 500.0},
    "ebay-used-tesla-v100-pcie-32gb":   {"nvidia-v100-pcie-32gb": 750.0},
    "used-nvidia-tesla-m40-market":     {"nvidia-tesla-m40": 189.0},
    "ebay-used-tesla-p40-24gb":         {"nvidia-tesla-p40": 290.0},
    "used-nvidia-tesla-p40-market":     {"nvidia-tesla-p40": 397.99},
    "ebay-used-amd-mi100-32gb":         {"amd-instinct-mi100": 900.0},
    "ebay-used-amd-mi210-64gb":         {"amd-instinct-mi210": 4180.0},
    "ebay-used-amd-mi250x-oam-128gb":   {"amd-instinct-mi250": 5450.0,
                                         "amd-instinct-mi250x": 5450.0},
    "ebay-used-intel-gaudi2-hl225-oam": {"intel-gaudi2": 2100.0},
    "ebay-used-cambricon-mlu270":       {"cambricon-mlu270": 318.0},
    "ebay-used-rtx3090-oct2026":        {"nvidia-rtx-3090": 950.0},
    "amazon-rtx3090-renewed-channel":   {"nvidia-rtx-3090": 1849.99},
    "newegg-consumer-gpus":             {"nvidia-rtx-3090": 2399.0},  # price is the 3090 ASUS TUF; B580/4090 are context only
    "used-rtx-3090-market":             {"nvidia-rtx-3090": 2399.0},
    "used-rtx-3090-ti-market":          {"nvidia-rtx-3090-ti": 2489.0},
    "ebay-used-a100-40gb-pcie":         {"nvidia-a100-40gb-pcie": 7500.0},
    "nvidia-marketplace-geforce":       {"nvidia-rtx-5090": 1999.0},
    "used-rtx-4090-market":             {"nvidia-rtx-4090": 4495.0},
    "used-rtx-4090-48gb-modded-market": {"nvidia-rtx-4090": 3510.48},  # 48GB modded variant, not stock 24GB
    "used-rx-7900-xtx-market":          {"amd-radeon-rx-7900-xtx": 1405.53},
    "used-rx-9070-xt-market":           {"amd-radeon-rx-9070-xt": 829.0},
    "bestbuy-rdna4-consumer-discrete":  {"amd-radeon-rx-9070-xt": 869.99},
    "new-framework-desktop-strix-halo-128gb": {"amd-radeon-8060s-ryzen-ai-max": 1999.0},  # whole mini-ITX box
    "new-tenstorrent-wormhole-n150":    {"tenstorrent-wormhole-n150": 999.0},
    "used-mac-studio-m1-ultra-128gb":   {"apple-m1-ultra": 2600.0},   # whole Mac Studio
    "ebay-used-a100-80gb-pcie":         {"nvidia-a100-80gb-pcie": 12099.0},
    "used-a100-80gb-pcie-market":       {"nvidia-a100-80gb-pcie": 21995.0},
    "used-h100-80gb-pcie-market":       {"nvidia-h100-pcie": 43000.0},
    "used-h800-80gb-market":            {"nvidia-h800-80gb": 41221.0},
    "used-h100-80gb-sxm-market":        {"nvidia-h100-sxm": 26880.0},
    "ebay-used-h100-sxm5-module":       {"nvidia-h100-sxm": 12999.0},
    "bh-photo-used-workstation-gpus":   {"nvidia-rtx-pro-6000-blackwell": 16623.95},
    "bh-photo-workstation-gpus":        {"nvidia-rtx-pro-6000-blackwell": 16999.99},  # price is the PRO 6000; 5090 context only
    "cdw-rtx-pro-6000-workstation-cards": {"nvidia-rtx-pro-6000-blackwell": 19977.99},

    # --- corrected / split joins (raw accelerator_ids over-attribute) ------
    # ebay-used-a100-40gb-sxm4 prices a 40GB SXM4 module but points at the
    #   80GB SXM4 id ("no separate 40GB record" at write time). There IS now
    #   nvidia-a100-40gb-sxm4. Map the $2,499 to the 40GB part, not the 80GB.
    "ebay-used-a100-40gb-sxm4":         {"nvidia-a100-40gb-sxm4": 2499.0},
    # ebay-used-accelerator-market is ONE A100 80GB SXM4 listing at $10,449.95;
    #   H100/H200 ids are listed as obtainable-but-unpriced. Do not join them.
    "ebay-used-accelerator-market":     {"nvidia-a100-80gb-sxm4": 10449.95},
    # microcenter refurb: 3090 flat $1,099.99, 3090 Ti flat $1,239.99 (two tiers).
    "microcenter-refurb-rtx-3090":      {"nvidia-rtx-3090": 1099.99,
                                         "nvidia-rtx-3090-ti": 1239.99},
    # new-retail-rtx-5090-amazon price_usd is the 5090 band (n=2); the 5080 band
    #   ($1,549-1,803) is in notes but is NOT the headline price. Drop 5080.
    "new-retail-rtx-5090-amazon":       {"nvidia-rtx-5090": 6549.99},
    # new-retail-rtx-4090-3090-consumer price_usd $3,299 is the 4090 median; the
    #   3090 median is $1,514 in notes, not the headline. Drop 3090 from this row.
    "new-retail-rtx-4090-3090-consumer": {"nvidia-rtx-4090": 3299.0},
    # ebay-used-consumer-gpu-market: $2,000 is the used-4090 low-end ask; 3090 is
    #   named only as a peer. Join 4090 only.
    "ebay-used-consumer-gpu-market":    {"nvidia-rtx-4090": 2000.0},
    # whole-system rows kept for context; per-GPU figure derived in the doc.
    # used-dgx1-8x-v100 is an 8-GPU server at $4,000 -> NOT a bare-card price.
    # ebay-used-a100-40gb-rack-liquidator is a 32-GPU rack at $249,000.
    # used-8x-h100-sxm-node-system is an 8-GPU server at $180,000.
}

# accel_id -> list of (per-unit usd, supply_id). The usd here is the curated
# per-UNIT figure for that part from PRICE_JOIN (which may differ from the
# record's headline price_usd where the headline priced a sibling).
price_map = {}
for sid, join in PRICE_JOIN.items():
    if sid not in supplies:
        continue  # join references a supply record that does not exist - skip
    for aid, usd in join.items():
        price_map.setdefault(aid, []).append((usd, sid))

rows = []
for aid, a in accs.items():
    v = a.get("vram_gb")
    bw = a.get("memory_bandwidth_gbps")
    if v is None or v < 8:
        continue
    sup = price_map.get(aid, [])
    min_price = min((u for u, _ in sup), default=None)
    qualifies = (bw is not None and bw >= 300) or (min_price is not None and min_price <= 3000)
    if not qualifies:
        continue
    fits = {k: (v >= w + kv) for k, _, w, kv in MODELS}
    ceil = {k: (bw / w if bw and w else None) for k, _, w, kv in MODELS}
    cheapest = min(sup, key=lambda x: x[0]) if sup else None  # (usd, supply_id)
    dpt = (cheapest[0] / ceil["35b"]) if cheapest and ceil["35b"] else None
    rows.append(dict(id=aid, a=a, vram=v, bw=bw, fits=fits, ceil=ceil,
                     cheapest=cheapest, supplies=sup, dpt=dpt))

priced = sorted([r for r in rows if r["dpt"] is not None], key=lambda r: r["dpt"])
unpriced_fit35 = sorted([r for r in rows if r["dpt"] is None and r["fits"]["35b"]],
                        key=lambda r: -(r["ceil"]["35b"] or 0))
unpriced_nofit = sorted([r for r in rows if r["dpt"] is None and not r["fits"]["35b"]],
                        key=lambda r: -(r["bw"] or 0))

print(f"=== {len(rows)} qualifying | {len(priced)} priced | {len(unpriced_fit35)+len(unpriced_nofit)} unpriced ===\n")
print("== RANKED (priced, by derived $/(tok/s) for the 35B-A3B workload) ==")
for r in priced:
    a = r["a"]
    usd, sid = r["cheapest"]
    f = "/".join("Y" if r["fits"][k] else "-" for k, *_ in MODELS)
    print(f"{r['id']}\t{a['name']}\t{r['vram']}\t{r['bw']}\t{f}\t"
          f"{r['ceil']['35b']:.1f}\t{r['ceil']['27b'] and round(r['ceil']['27b'],1)}\t{r['ceil']['70b'] and round(r['ceil']['70b'],1)}\t"
          f"{usd}\t{sid}\t{r['dpt']:.2f}")
print("\n== UNPRICED BUT FITS 35B-A3B (by 35B ceiling) ==")
for r in unpriced_fit35:
    a = r["a"]
    f = "/".join("Y" if r["fits"][k] else "-" for k, *_ in MODELS)
    print(f"{r['id']}\t{a['name']}\t{r['vram']}\t{r['bw']}\t{f}\t{r['ceil']['35b']:.1f}")
print("\n== QUALIFYING BUT DOES NOT FIT 35B-A3B (all unpriced) ==")
for r in unpriced_nofit:
    a = r["a"]
    f = "/".join("Y" if r["fits"][k] else "-" for k, *_ in MODELS)
    print(f"{r['id']}\t{a['name']}\t{r['vram']}\t{r['bw']}\t{f}\t{r['ceil']['35b'] and round(r['ceil']['35b'],1)}")
