#!/usr/bin/env python3
"""Build joined-view data for the site's decision pages.

Reads data/ directly (like cost_per_token.py) and emits JSON payloads to
site/public/joined/ that the Astro pages fetch at runtime. This script does
NOT modify data/, schemas/, or tools/build_site_data.py.

    python tools/build_joined_views.py            # writes site/public/joined/

Three joins are built:
  1. cost-per-token: supply prices x benchmark throughput -> USD per million tokens
  2. compatibility: model architecture families x engine support matrix
  3. roofline: accelerator specs x flop classes -> ridge points and crossovers
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, TITLES  # noqa: E402

# Also need to import from cost_per_token which is in the same directory

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "public" / "joined"


def load(dirname: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    d = ROOT / "data" / dirname
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out[f.stem] = rec
    return out


# ===========================================================================
# 1. COST PER TOKEN
# ===========================================================================

# Import the price quotes and bench join from cost_per_token.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_per_token import PRICE_QUOTES, BENCH_JOIN, EXCLUDED, THROUGHPUT_METRICS, concurrency_class  # noqa: E402


def build_cost_per_token() -> dict:
    supply = load(DIRS["supply"])
    bench = load(DIRS["benchmark"])
    # DIRS keys are singular: 'model', 'engine', 'accelerator', 'flop'
    rows = []
    unmatched = []

    for bid, spec in sorted(BENCH_JOIN.items()):
        b = bench.get(bid)
        if b is None:
            continue
        if b.get("metric") not in THROUGHPUT_METRICS:
            continue
        gp = spec["gpus"]
        per_gpu_tps = float(b["value"]) / gp if gp else float(b["value"])
        accels = list(b.get("accelerator_ids") or [])
        if not accels:
            unmatched.append({"bench": bid, "reason": "benchmark names no accelerator"})
            continue
        for accel in accels:
            hits = [(sid, q) for sid, qs in PRICE_QUOTES.items()
                    for q in qs if q["accel"] == accel]
            if not hits:
                unmatched.append({
                    "bench": bid, "accel": accel,
                    "reason": f"no supply record prices {accel} per GPU-hour",
                })
                continue
            for sid, q in hits:
                if sid not in supply:
                    continue
                usd_mtok = q["usd"] / per_gpu_tps * 1_000_000
                rows.append({
                    "provider": sid,
                    "accelerator": accel,
                    "model": b.get("model"),
                    "engine": b.get("engine_id"),
                    "metric": b["metric"],
                    "concurrency": concurrency_class(b["metric"], b.get("unit")),
                    "gpus_covered": gp,
                    "tok_s_total": float(b["value"]),
                    "tok_s_per_gpu": round(per_gpu_tps, 2),
                    "usd_per_gpu_hour": q["usd"],
                    "tier": q["tier"],
                    "usd_per_mtok": round(usd_mtok, 4),
                    "supply_ref": f"supply/{sid}",
                    "bench_ref": f"benchmarks/{bid}",
                    "measured_by": b.get("measured_by"),
                    "bench_status": b.get("status"),
                    "supply_status": supply[sid].get("status"),
                    "reproducible": b.get("reproducible"),
                    "unit": b.get("unit"),
                })
    rows.sort(key=lambda r: r["usd_per_mtok"])
    return {"rows": rows, "unmatched": unmatched}


# ===========================================================================
# 2. COMPATIBILITY MATRIX
# ===========================================================================

# The 14 architecture families from docs/13-model-compatibility.md
FAMILIES = [
    "dense-gqa", "dense-mha", "mla", "sliding-window",
    "hybrid-attention-ssm", "ssm", "recurrent", "diffusion",
    "encoder-decoder", "bi-encoder", "cross-encoder",
    "speech-encoder", "multimodal", "undisclosed",
]

# The 8 engines in the matrix
ENGINES = ["vllm", "sglang", "tensorrt-llm", "llama.cpp", "flashinfer", "exllamav3", "mlx-lm", "vllm-metal"]

# Parse the matrix from the doc
def parse_compatibility_matrix() -> dict:
    doc_path = ROOT / "docs" / "13-model-compatibility.md"
    text = doc_path.read_text(encoding="utf-8")

    # Find the generation engines table
    # Format: | family | vllm | sglang | ... |
    # We look for lines starting with | `family` |
    matrix = {}
    in_table = False
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("| `") and "` |" in line:
            # This is a data row
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]  # remove empty from leading/trailing |
            if len(parts) >= 2:
                family = parts[0].strip("`")
                cells = parts[1:]
                matrix[family] = cells
                in_table = True
        elif in_table and line.startswith("|---"):
            continue
        elif in_table and not line.startswith("|"):
            break

    return matrix


def build_compatibility() -> dict:
    matrix = parse_compatibility_matrix()
    models = load(DIRS["model"])
    engines = load(DIRS["engine"])

    # Classify each model into a family
    family_models = defaultdict(list)
    for mid, m in sorted(models.items()):
        arch = m.get("architecture") or ""
        att = m.get("attention_variant") or ""
        kv_lora = m.get("kv_lora_rank")
        max_pos = m.get("max_position_embeddings")
        sliding = m.get("sliding_window")

        # Classification logic from the doc
        if kv_lora is not None:
            family = "mla"
        elif sliding is not None and isinstance(sliding, int) and max_pos and sliding < max_pos:
            family = "sliding-window"
        elif "mamba" in arch.lower() or "ssm" in arch.lower():
            family = "ssm"
        elif "rwkv" in arch.lower() or "wkv" in arch.lower():
            family = "recurrent"
        elif "diffusion" in arch.lower() or m.get("use_cache") is False:
            family = "diffusion"
        elif "whisper" in arch.lower() or "encoder-decoder" in arch.lower():
            family = "encoder-decoder"
        elif "bi-encoder" in arch.lower() or "bge" in mid or "reranker" in mid:
            family = "bi-encoder"
        elif "cross-encoder" in arch.lower() or "colbert" in mid:
            family = "cross-encoder"
        elif "conformer" in arch.lower() or "asr" in arch.lower() or "speech" in arch.lower():
            family = "speech-encoder"
        elif "vision" in arch.lower() or "multimodal" in arch.lower() or "vl" in arch.lower():
            family = "multimodal"
        elif "gqa" in att.lower() or (m.get("gqa_ratio") or 1) > 1:
            family = "dense-gqa"
        elif "mha" in att.lower() or m.get("gqa_ratio") == 1 or m.get("gqa_ratio") is None:
            family = "dense-mha"
        elif "claude" in m.get("name", "").lower() or "gpt" in m.get("name", "").lower() or "o3" in m.get("name", "").lower():
            family = "undisclosed"
        else:
            family = "dense-gqa"  # default

        family_models[family].append({
            "rid": f"models/{mid}",
            "name": m.get("name", mid),
            "status": m.get("status"),
            "architecture": arch,
            "attention_variant": att,
        })

    # Build engine info
    engine_info = {}
    for eid in ENGINES:
        e = engines.get(eid, {})
        engine_info[eid] = {
            "rid": f"engines/{eid}",
            "name": e.get("name", eid),
            "status": e.get("status"),
            "license": e.get("license"),
            "supported_backends": e.get("supported_backends", []),
            "best_for": e.get("best_for", []),
            "avoid_for": e.get("avoid_for", []),
        }

    return {
        "families": FAMILIES,
        "engines": ENGINES,
        "matrix": matrix,
        "family_models": dict(family_models),
        "engine_info": engine_info,
    }


# ===========================================================================
# 3. ROOFLINE EXPLORER
# ===========================================================================

def build_roofline() -> dict:
    accels = load(DIRS["accelerator"])
    flops = load(DIRS["flop"])

    # Compute ridge points for each accelerator
    accel_data = []
    for aid, a in sorted(accels.items()):
        flops_list = a.get("flops") or []
        bw = a.get("memory_bandwidth_gbps")
        if not bw or not flops_list:
            continue

        # Find the highest dense TFLOPS (usually bf16 or fp16)
        dense_tflops = 0
        for f in flops_list:
            if f.get("dense") and f.get("tflops"):
                prec = f.get("precision", "")
                if prec in ("bf16", "fp16", "fp8", "fp4"):
                    dense_tflops = max(dense_tflops, f["tflops"])

        if dense_tflops == 0:
            continue

        # Ridge = 1000 * TFLOPS / (GB/s) = flop/byte
        ridge = 1000 * dense_tflops / bw

        # Find fp8 ridge if available
        fp8_tflops = 0
        for f in flops_list:
            if f.get("dense") and f.get("precision") == "fp8" and f.get("tflops"):
                fp8_tflops = max(fp8_tflops, f["tflops"])
        fp8_ridge = 1000 * fp8_tflops / bw if fp8_tflops else None

        accel_data.append({
            "rid": f"accelerators/{aid}",
            "name": a.get("name", aid),
            "status": a.get("status"),
            "vendor": a.get("vendor"),
            "architecture": a.get("architecture"),
            "vram_gb": a.get("vram_gb"),
            "memory_type": a.get("memory_type"),
            "memory_bandwidth_gbps": bw,
            "dense_tflops": dense_tflops,
            "fp8_tflops": fp8_tflops,
            "ridge": round(ridge, 1),
            "fp8_ridge": round(fp8_ridge, 1) if fp8_ridge else None,
            "tdp_w": a.get("tdp_w"),
        })

    # Classify flop records by bound_by
    flop_classes = []
    for fid, f in sorted(flops.items()):
        flop_classes.append({
            "rid": f"flops/{fid}",
            "name": f.get("name", fid),
            "class": f.get("class"),
            "bound_by": f.get("bound_by"),
            "arithmetic_intensity": f.get("arithmetic_intensity"),
            "scales_with": f.get("scales_with"),
            "status": f.get("status"),
        })

    return {"accelerators": accel_data, "flop_classes": flop_classes}


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Building cost-per-token join...")
    cost = build_cost_per_token()
    (OUT / "cost_per_token.json").write_text(
        json.dumps(cost, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {len(cost['rows'])} rows, {len(cost['unmatched'])} unmatched")

    print("Building compatibility matrix...")
    compat = build_compatibility()
    (OUT / "compatibility.json").write_text(
        json.dumps(compat, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {len(compat['matrix'])} families, {len(compat['engines'])} engines")

    print("Building roofline data...")
    roof = build_roofline()
    (OUT / "roofline.json").write_text(
        json.dumps(roof, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {len(roof['accelerators'])} accelerators, {len(roof['flop_classes'])} flop classes")

    total = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"\n  {total / 1024:.0f} KB total under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
