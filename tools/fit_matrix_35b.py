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

price_map = {}
for r in supplies.values():
    if not is_purchase(r):
        continue
    for aid in (r.get("accelerator_ids") or []):
        price_map.setdefault(aid, []).append(r)

rows = []
for aid, a in accs.items():
    v = a.get("vram_gb")
    bw = a.get("memory_bandwidth_gbps")
    if v is None or v < 8:
        continue
    sup = price_map.get(aid, [])
    min_price = min((s["price_usd"] for s in sup), default=None)
    qualifies = (bw is not None and bw >= 300) or (min_price is not None and min_price <= 3000)
    if not qualifies:
        continue
    fits = {k: (v >= w + kv) for k, _, w, kv in MODELS}
    ceil = {k: (bw / w if bw and w else None) for k, _, w, kv in MODELS}
    cheapest = min(sup, key=lambda s: s["price_usd"]) if sup else None
    dpt = (cheapest["price_usd"] / ceil["35b"]) if cheapest and ceil["35b"] else None
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
    a, c = r["a"], r["cheapest"]
    f = "/".join("Y" if r["fits"][k] else "-" for k, *_ in MODELS)
    print(f"{r['id']}\t{a['name']}\t{r['vram']}\t{r['bw']}\t{f}\t"
          f"{r['ceil']['35b']:.1f}\t{r['ceil']['27b'] and round(r['ceil']['27b'],1)}\t{r['ceil']['70b'] and round(r['ceil']['70b'],1)}\t"
          f"{c['price_usd']}\t{c['id']}\t{r['dpt']:.2f}")
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
