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

# Capability axes. Averaging individual benchmark entries lets a model that publishes
# AIME 2024 AND AIME 2025 weight "math reasoning" twice, and lets a saturated test
# (where the catalog best is ~100) hand out a cheap 92 that carries the same weight as
# a hard-earned 60 on SWE-bench. Group benchmarks into axes, average within an axis,
# then average across axes: one capability counts once, however many rows it has.
_AXES = (
    ("math", ("AIME", "MATH-500", "MATH", "HMMT", "IMO")),
    ("coding", ("SWE-bench", "SWE Bench", "LiveCodeBench", "HumanEval", "SciCode",
                "ProgramBench", "SWE-Marathon", "DeepSWE")),
    ("agentic", ("Terminal-Bench", "Terminal Bench", "Tau-Bench", "TAU-Bench",
                 "Toolathlon", "MCPMark", "BrowseComp", "DeepSearchQA")),
    ("knowledge", ("MMLU", "GPQA", "HLE", "SuperGPQA", "CMMLU", "CritPt")),
    ("instruction", ("IFEval", "IFBench", "Arena-Hard", "AlignBench")),
)


def _axis_for(name):
    low = name.lower()
    for axis, keys in _AXES:
        for k in keys:
            if k.lower() in low:
                return axis
    return "other"


def load_quant_retention() -> dict:
    """base model record id -> list of retention rows (derivative-sourced).

    Joins on the record's declared base_model when present, else on the HF repo id
    that the derivative names as its base, resolved through each record's sources.
    """
    p = ROOT / "data/quant_retention.jsonl"
    if not p.exists():
        return {}
    import re as _re
    hf_to_rec = {}
    for f in (ROOT / "data/models").glob("*.json"):
        d = json.loads(f.read_text())
        for s in d.get("sources") or []:
            sp = ROOT / "data/sources" / (s + ".json")
            if not sp.exists():
                continue
            txt = sp.read_text()
            for m in _re.finditer(r"huggingface\.co/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", txt):
                hf_to_rec.setdefault(m.group(1).lower(), d["id"])
    # Rows from the widen pass carry no base_repo (the digest strips the header), so
    # recover the derivative's base from the mined derivatives database instead.
    deriv_base = {}
    dp = ROOT / "data/derivatives.jsonl"
    if dp.exists():
        for line in dp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("base"):
                deriv_base[r["id"].lower()] = r["base"].lower()
    out = {}
    for path in (p, ROOT / "data/quant_retention_widen.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            key = d.get("base_model") or hf_to_rec.get((d.get("base_repo") or "").lower())
            if not key:
                b = deriv_base.get((d.get("id") or "").lower())
                key = hf_to_rec.get(b) if b else None
            if key:
                out.setdefault(key, []).append(d)
    return out


def build_worth_running() -> dict:
    _CONTEXT_TOKENS = 8192
    _MULTI_USERS = 16
    models = load(DIRS["model"])
    rows = []
    for stem, r in models.items():
        if r.get("open_weights") is False:
            continue
        # Embeddings, rerankers and speech models are not chat models and cannot be
        # ranked on GPQA/SWE-bench. They used to sit in this table with no score,
        # next to the models someone would actually run.
        if (r.get("task") or "text-generation") in ("embedding", "reranking",
                                                    "speech-to-text", "text-to-speech",
                                                    "image-generation", "decision"):
            # A decision model returns per-option probabilities in one forward pass
            # and generates no text at all, so every benchmark here is meaningless
            # for it. Ranked separately, on the Decision Index: tools/decision_ranking.py.
            continue
        active = r.get("active_params_b")
        total = r.get("params_b")
        if not isinstance(active, (int, float)) or active <= 0:
            continue
        if not isinstance(total, (int, float)) or total <= 0:
            continue
        # The whole checkpoint is resident somewhere even when experts are offloaded
        # to RAM. Pricing only the active slice reported a 600B MoE as 14 GB, which
        # is the per-token read, not the memory it occupies.
        weights = total * 4 / 8  # 4-bit GGUF-class, GiB: the resident file
        traffic = active * 4 / 8  # 4-bit: weights read per decode step
        kv_bpt = r.get("kv_cache_bytes_per_token")
        kv = None
        if isinstance(kv_bpt, (int, float)) and kv_bpt > 0:
            kv = kv_bpt * _CONTEXT_TOKENS / (1024 ** 3)

        scored = []
        for b in r.get("benchmarks") or []:
            if not isinstance(b, dict):
                continue
            score, scale = b.get("score"), b.get("scale")
            if not (isinstance(score, (int, float)) and isinstance(scale, (int, float)) and scale > 0):
                continue
            if b.get("higher_is_better", True) is False:
                continue
            scored.append({
                "name": b.get("name"),
                "score": score,
                "scale": scale,
                "pct": round(score / scale * 100, 1),
                "vendor_claim": bool(b.get("vendor_claim", False)),
                "source": b.get("source"),
            })
        capability = None  # set below, once the catalog best per benchmark is known

        experts = r.get("experts_per_token")
        moe = bool(r.get("num_experts"))
        # A dense model that spills into RAM moves every weight per token, so it
        # only offloads well when there is nothing to spill.
        offload_friendly = moe and active <= 12

        single = round(weights + (kv or 0), 1)
        multi = round(weights + (kv or 0) * _MULTI_USERS, 1) if kv else None

        rows.append({
            "id": stem,
            "name": r.get("name"),
            "family": r.get("family"),
            "release_year": r.get("release_year"),
            "active_params_b": active,
            "total_params_b": total,
            "moe": moe,
            "experts_per_token": experts,
            "weights_q4_gib": round(weights, 1),
            "traffic_q4_gib": round(traffic, 1),
            "kv_8k_gib": round(kv, 2) if kv else None,
            "single_user_gib": single,
            "multi_user_gib": multi,
            "offload_friendly": offload_friendly,
            "capability": capability,
            "benchmarks": scored,
            "vendor_only": bool(scored) and all(s["vendor_claim"] for s in scored),
        })

    # A raw mean of score/scale is NOT comparable across models: a model scored on
    # AIME 2026 and MMLU-Pro averages high, one scored on SWE-bench Verified and
    # Terminal-Bench averages low, and the difference is the test mix, not the model.
    # Score each benchmark relative to the best result recorded for THAT benchmark
    # in the catalog, so 100 means "best known on this test" and models scored on
    # different tests can be ranked against each other. `bench_count` is reported
    # so a model carried by a single benchmark is visible rather than hidden.
    # Only a benchmark reported by >=2 models can rank anything. With a single
    # reporter the catalog max IS that model, so the row would read 100 and hand a
    # model a free perfect score for every test only it ran. Count reporters first.
    _reporters = {}
    for r in rows:
        for b in r["benchmarks"]:
            if b["name"]:
                _reporters.setdefault(b["name"], []).append((b["pct"], r.get("release_year")))

    # A benchmark can only rank a model to the extent it still SEPARATES models.
    # Measured on the catalog's own distribution: the share of reporters that sit
    # within 5 points of the best-known figure. MMLU-CoT, HellaSwag and 8-shot GSM8K
    # are saturated -- dozens of models land inside that band, so being at the top
    # says which era the test was in use, not which model is worth running. A 2024
    # 8B within 2 points of the catalog-best MMLU figure used to average 93.8 and
    # outrank a 397B. Disagreement is the signal: HLE, SWE-bench Pro and
    # Terminal-Bench spread models out, so they carry most of the weight.
    # Benchmarks also decay. A test whose LEADERS are all years old has been
    # superseded: nobody competitive still reports it, so a top score on it is a
    # statement about the era, not the current model. Date each benchmark by the
    # release years of its top quartile, then discount by that age. This is what
    # keeps a 2023 model off the top of a "worth running" list when its best
    # evidence is MMLU-CoT and HellaSwag, both still in the catalog.
    # Steep on purpose. Capability has been moving fast enough that a benchmark
    # whose LEADERS are a year old has been superseded by whatever replaced it, and
    # one whose leaders are two years old is archaeology rather than evidence. A
    # gentle slope let a 2024 8B collect near-100s on tests that were competitive
    # when it shipped; at these weights a 2025-led test keeps 30% and a 2024-led
    # test 6%, so ordering is driven by tests current models actually contest.
    def _recency(years):
        ys = [y for y in years if y]
        if not ys:
            return 0.5          # undated leaders: half weight, never full
        med = sorted(ys)[len(ys) // 2]
        return 1.0 if med >= 2026 else 0.30 if med == 2025 else 0.06 if med == 2024 else 0.0

    best, disc, bench_year = {}, {}, {}
    for nm, vals in _reporters.items():
        if len(vals) < 2:
            continue            # a single reporter would BE the best-known figure
        top = max(p for p, _ in vals)
        best[nm] = top
        share_near = sum(1 for p, _ in vals if top - p <= 5.0) / len(vals)
        w = max(0.0, 1.0 - share_near)
        if len(vals) == 2:
            w *= 0.5            # two reporters do not establish a distribution
        leaders = sorted(vals, key=lambda t: -t[0])[:max(1, len(vals) // 4)]
        rf = _recency([y for _, y in leaders])
        bench_year[nm] = round(rf, 2)
        disc[nm] = round(w * rf, 3)

    for r in rows:
        if not r["benchmarks"]:
            continue
        by_axis = {}
        for b in r["benchmarks"]:
            top = best.get(b["name"])
            if not top:
                continue
            rel = min(b["pct"] / top * 100, 100.0)
            w = disc.get(b["name"], 0.0)
            b["relative"] = round(rel, 1)
            b["discrimination"] = w
            b["saturated"] = w < 0.15
            ax = _axis_for(b["name"])
            if ax == "other":
                # An incomparable grab-bag (OpenCompass, OCRBench, MMBench...). Shown
                # on the row, never averaged into capability.
                continue
            by_axis.setdefault(ax, []).append((rel, w))
        if not by_axis:
            continue
        per_axis, axis_w = {}, {}
        for k, items in by_axis.items():
            tw = sum(w for _, w in items)
            axis_w[k] = round(tw, 2)
            per_axis[k] = (round(sum(rel * w for rel, w in items) / tw, 1) if tw > 0
                           else round(sum(rel for rel, _ in items) / len(items), 1))
        r["axes"] = per_axis
        r["axis_weights"] = axis_w
        r["bench_count"] = len(r["benchmarks"])
        usable = [k for k, w in axis_w.items() if w > 0]
        W = round(sum(axis_w[k] for k in usable), 2)
        r["evidence_weight"] = W
        r["ranked_axes"] = len(usable)
        if len(usable) >= 2 and W >= 1.0:
            r["raw_capability"] = round(sum(per_axis[k] * axis_w[k] for k in usable) / W, 1)
        else:
            r["raw_capability"] = None
        r["thin"] = len(usable) <= 2

    # Thin evidence cannot claim the top of a "worth running" list. A model whose
    # score rests on one axis, or on tests everyone saturates, would otherwise tie
    # the frontier on a technicality: being best-known on a benchmark nobody else
    # still reports is a statement about 2023, not about now. Pull each score toward
    # the catalog median in proportion to how little evidence stands behind it --
    # the standard shrinkage estimator, with K axes of prior weight. Both numbers are
    # published (capability = shrunk, raw_capability = unshrunk) so the adjustment is
    # visible rather than baked in.
    # And the model itself ages, separately from its evidence. A 2024 checkpoint
    # that still looks competitive on a current test is worth running; it is not
    # worth ranking level with a 2026 one, because the 2026 one is what the 2024 one
    # was superseded by. Applied to the published score, not hidden inside a sort.
    def _model_recency(y):
        if not y:
            return 0.85
        return 1.0 if y >= 2026 else 0.90 if y == 2025 else 0.70 if y == 2024 else 0.50 if y == 2023 else 0.35

    _raw = sorted(r["raw_capability"] for r in rows if r.get("raw_capability") is not None)
    PRIOR = round(_raw[len(_raw) // 2], 1) if _raw else 0.0
    K = 2.0
    for r in rows:
        raw, W = r.get("raw_capability"), r.get("evidence_weight") or 0.0
        if raw is None:
            r["capability"] = None
            r["capability_basis"] = ("not ranked: no discriminating evidence. Every benchmark it reports "
                                     "is either reported by it alone or saturated across the catalog, so a "
                                     "top score there measures the era, not the model.")
            continue
        r["capability"] = round((raw * W + PRIOR * K) / (W + K), 1)
        r["recency_factor"] = _model_recency(r.get("release_year"))
        r["worth"] = round(r["capability"] * r["recency_factor"], 1)
        r["capability_prior"] = PRIOR
        r["capability_basis"] = ("evidence-weighted relative score, shrunk toward the catalog median (%.1f) "
                                 "by %.0f axes of prior weight. Weighted: each axis averages "
                                 "score/best-known over its benchmarks, discounted by how far that benchmark "
                                 "still spreads models (1 - share of reporters within 5 points of the catalog "
                                 "best), so saturated tests barely count. Shrunk: %s axes of evidence behind "
                                 "raw %.1f." % (PRIOR, K, W, raw))

    # Attach the best-evidenced low-bit retention for each model. Derivative-sourced,
    # not record-sourced: see gotcha ab-gotcha-derivative-retention-is-hf-card-sourced.
    qr = load_quant_retention()
    for r in rows:
        cands = qr.get(r["id"]) or []
        keep = [c for c in cands if c["bucket"] != "suspect"]
        if keep:
            best = max(keep, key=lambda c: (c.get("n") or 0, -abs(c["median_delta_pct"])))
            r["quant_retention"] = best
            r["quant_retention_basis"] = (
                "delta vs base across n benchmarks, read from the derivative repo's own "
                "comparison table and recomputed with metric direction applied. Not a "
                "record claim; see the derivative-retention gotcha.")

    rows.sort(key=lambda x: (-(x.get("worth") if x.get("worth") is not None else -1), x["single_user_gib"]))
    return {
        "assumptions": {
            "quant_bits": 4,
            "context_tokens": _CONTEXT_TOKENS,
            "multi_users": _MULTI_USERS,
            "weights_basis": "params_b * 4 bits / 8, GiB: the whole checkpoint at 4-bit, which is what has to be resident. MoE expert offload moves most of it to system RAM but does not remove it.",
            "traffic_basis": "active_params_b * 4 bits / 8, GiB: weights touched per token. This is what decode bandwidth scales with, not what must fit.",
            "scope": "Generative text models only. Records whose task is embedding, reranking, speech-to-text, text-to-speech, image-generation or decision are excluded, because chat benchmarks cannot rank an embedder and a decision model emits no text to score. Decision models are ranked on the Decision Index in docs/33-decision-models.md.",
            "capability_basis": "evidence-weighted mean across capability axes (knowledge, math, coding, "
                                "agentic, instruction). Within an axis each benchmark is scored "
                                "relative to the best result recorded for THAT benchmark here, so "
                                "100 = best known on that test, discounted for saturation and for the age of its "
                                "leaders and then shrunk toward the catalog median when the evidence is thin. "
                                "Models tested differently stay "
                                "comparable. Benchmarks are grouped so one capability counts once, "
                                "however many rows it publishes, and a benchmark whose catalog best "
                                "is >=95 is flagged saturated because it cannot separate models.",
        },
        "rows": rows,
    }



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

    print("Building worth-running ranking...")
    worth = build_worth_running()
    (OUT / "worth_running.json").write_text(
        json.dumps(worth, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    scored = sum(1 for r in worth["rows"] if r["capability"] is not None)
    print(f"  {len(worth['rows'])} runnable models, {scored} with a capability score")

    total = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"\n  {total / 1024:.0f} KB total under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


# ===========================================================================
# 4. WORTH RUNNING — capability vs system requirements
# ===========================================================================

# Capability is not comparable across the repo's `quality` benchmarks: they mix
# perplexity deltas, acceptance lengths and accuracy on different harnesses.
# Instead each model record carries `benchmarks`, a list of
# {name, score, scale, higher_is_better, source, vendor_claim}. The capability
# index is the mean of score/scale over the higher-is-better entries, in
# [0, 100]. Requirements come from the structured fields:
#   weights_q4_gib = active_params_b * 4 / 8          (4-bit weights)
#   kv_gib        = kv_cache_bytes_per_token * N / 1024**3
# Single user = batch 1 at 8k context. Multi user = 16 concurrent at 8k, where
# the KV cache is what grows. Offload-friendliness is whether the model is MoE
# with a small active footprint, since only the routed experts move per token.

_CONTEXT_TOKENS = 8192
_MULTI_USERS = 16

