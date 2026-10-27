#!/usr/bin/env python3
"""Build joined-view data for the site's decision pages.

Reads data/ directly (like cost_per_token.py) and emits JSON payloads to
site/public/joined/ that the Astro pages fetch at runtime. This script does
NOT modify data/, schemas/, or tools/build_site_data.py.

    python tools/build_joined_views.py            # writes site/public/joined/

Five joins are built:
  1. cost-per-token: supply prices x benchmark throughput -> USD per million tokens
  2. compatibility: model architecture families x engine support matrix
  3. roofline: accelerator specs x flop classes -> ridge points and crossovers
  4. worth-running: model capability x recency x hardware fit
  5. flops-per-dollar: dense throughput per dollar, new / used / rented
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
        # Steep on purpose: a checkpoint from a previous generation is still runnable,
        # but it is not a contender next to this year's models. 0.85 used to be the
        # undated default, which quietly promoted records with no date at all.
        if not y:
            return 0.50
        return 1.0 if y >= 2026 else 0.60 if y == 2025 else 0.25 if y == 2024 else 0.10 if y == 2023 else 0.05

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
        # Hardware realism. A model that needs a GPU farm is not worth the same as one
        # that fits a single card, even at equal capability: the score is what you can
        # actually deploy. Tiers are 4-bit weight footprint (params_b * 0.5 GiB), which
        # is the way these models are served in practice.
        def _hw_factor(gib, active):
            # gib = resident GiB to serve one user at 4-bit (weights + KV), the same
            # figure the Need column shows. Weights alone understates a long-context
            # giant, which is exactly the case this is meant to price.
            if not gib:
                return None
            if gib <= 24:
                f = 1.00          # single consumer card
            elif gib <= 48:
                f = 0.95          # single 48 GB card
            elif gib <= 80:
                f = 0.92          # single workstation card
            elif gib <= 160:
                f = 0.82          # two cards, or the user's own 128 GiB box
            elif gib <= 400:
                f = 0.68          # small cluster
            else:
                f = 0.55          # GPU farm
            # A sparse model with a small active set runs at its ACTIVE size once the
            # memory exists, so its real penalty is memory, not latency - and experts
            # can sit in system RAM. Floor it rather than scoring it like a dense giant.
            if active and active <= 10 and gib > 24:
                f = max(f, 0.85)
            return f

        r["recency_factor"] = _model_recency(r.get("release_year"))
        _gib = r.get("single_user_gib") or r.get("weights_q4_gib")
        r["footprint_gib"] = round(_gib, 1) if _gib else None
        r["hw_factor"] = _hw_factor(_gib, r.get("active_params_b") or r.get("total_params_b"))
        _fit = r["hw_factor"] if r["hw_factor"] is not None else 1.0
        r["worth"] = round(r["capability"] * r["recency_factor"] * _fit, 1)
        r["worth_pre_fit"] = round(r["capability"] * r["recency_factor"], 1)
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




# ===========================================================================
# 5. FLOPS PER DOLLAR — compute per dollar, bought new, bought used, or rented
# ===========================================================================

# This join is easy to make and easy to make useless, because the raw supply pool
# prices four different things in the same `price_usd` field: one card, eight cards
# in a tray, a whole rack of servers, and an adapter that carries a card. Sorting on
# price alone puts a $130 SXM2-to-PCIe carrier at the top of a GPU table, which is
# not a wrong ranking so much as a wrong quantity. So:
#
#   * quantities. A price whose basis names a count ("8x", "32x", "4 nodes x 8 GPUs")
#     is divided down to one accelerator, and the divisor is kept in the row. Anything
#     still outside 3x that accelerator's median is treated as a different product
#     (chassis, accessory, complete server) and moved to `excluded` with the reason.
#     Nothing is silently discarded.
#   * accessories. A carrier, chassis or fan bundle links to a card's accelerator_id in
#     the supply pool, so the id alone does not mean "one GPU". Only the record NAME is
#     matched for those words: price_basis routinely describes the sibling products
#     ("8x tray asks $11,995") next to the part's own price, and matching on prose drops
#     legitimate single-card listings.
#   * sparse figures. 2:4 sparsity doubles the headline, so only dense figures rank.
#   * precisions. bf16 where the part publishes it, fp16 otherwise — never fp64 or fp32,
#     which say nothing about inference.
#   * what "a dollar" buys. Purchase and rental are different markets. Rental rates come
#     from cost_per_token.PRICE_QUOTES, which already normalises per GPU-hour and records
#     the literal evidence substring; the token-priced cloud records ("$0.6 per 1M
#     TOKENS") are services, not hardware, and cannot share this axis.
#
# Asking price is not a sold price; theoretical FLOPS is not delivered FLOPS. Both are in
# the payload's method block and in every row's caveats.

NEW_KINDS = {"oem-direct", "regional-reseller", "enterprise-distributor", "integrator", "self-build"}
USED_KINDS = {"used-market", "auction", "broker"}
HEADLINE_PRECISION = ("bf16", "fp16")
QUANT_PRECISION = ("fp8", "int8", "fp4", "int4")

_ACCESSORY = re.compile(r"\b(adapter|carrier|chassis|enclosure|cable|riser|dock|"
                        r"power supply|heatsink|fan kit|without gpus|no gpus)\b", re.I)
# A listing that says "bare module, needs carrier + heatsink" is a GPU; one whose product IS
# the carrier is not. The difference is whether a part name appears in it, so the accessory
# words only disqualify a record that names no accelerator at all.
# An accessory names the part it fits ("adapter / carrier board for A100 SXM4 modules"), which
# is how a $900 carrier gets a GPU's accelerator_id and a GPU's headline. Two precise tells:
# the accessory word leads the name, or the name is a thing "for <a part>". A bare module that
# merely mentions needing a carrier later in its name is still a module.
_ACCESSORY_LEAD = re.compile(r"^[^,:(]{0,32}\b(adapter|carrier|chassis|enclosure|cable|riser|"
                             r"dock|converter|bridge)\b", re.I)
_ACCESSORY_FOR = re.compile(r"\bfor\s+(the\s+)?(nvidia|amd|intel|tesla|geforce|rtx|a100|h100|"
                            r"h200|v100|p100|mi[0-9]+|instinct|radeon)\b", re.I)
_GPU_TOKEN = re.compile(r"\b(v100|a100|a800|h100|h200|h800|b100|b200|b300|a10|a16|a30|a40|"
                        r"l4|l40|l40s|p4|p40|p100|k80|m40|t4|rtx|geforce|quadro|tesla|instinct|"
                        r"radeon|mi[0-9]+|mlu[0-9]+|br[0-9]+|ascend|gaudi|grayskull|wormhole|"
                        r"blackhole|tpu|trainium|inferentia|epyc|xeon|strix|m[0-9]\s*(ultra|max))\b",
                        re.I)
_COUNT_PATTERNS = (
    re.compile(r"=\s*(\d+)\s*x\b", re.I),                                     # "= 32x NVIDIA A100"
    re.compile(r"\b(\d+)\s*x\s+(?:NVIDIA|AMD|Intel|Huawei|Gaudi|Ascend|Tesla|GeForce|RTX|"
               r"Instinct|Radeon|Xeon|nvidia|amd)", re.I),                    # "8x Gaudi2 tray"
    re.compile(r"(\d+)\s*nodes?\s*x\s*(\d+)\s*gpu", re.I),                     # "4 nodes x 8 GPUs"
    re.compile(r"\b(\d+)-gpu\s+(?:server|tray|system)", re.I),                 # "8-GPU server"
)


def _bulk_divisor(name: str, basis: str) -> tuple[int, str]:
    """How many accelerators the quoted price covers, and the phrase that said so.

    The count must describe THIS listing, not a sibling product. Prose like "8x Gaudi2 tray
    asks $11,995" sits in the price_basis of a single-module record, and dividing the module's
    own $2,100 by that 8 turns a $2,100 part into a fictional $262 one — the exact arithmetic
    that would put a Gaudi 2 at the top of the table. So the record NAME is authoritative,
    with two explicit basis forms ("N nodes x M GPUs", "= Nx") as fallbacks.
    """
    # the lookbehind matters: "MI250X" and "MI50X" end in a digit-then-x and must not be read
    # as "250 cards" and "50 cards", which priced an Instinct MI250X at $20
    m = re.search(r"(?<![\w.])(\d+)\s*x\b", name or "", re.I)
    if m and int(m.group(1)) > 1:
        return int(m.group(1)), m.group(0)
    m = _COUNT_PATTERNS[2].search(basis)
    if m:
        return int(m.group(1)) * int(m.group(2)), m.group(0)
    m = _COUNT_PATTERNS[0].search(basis)
    if m and int(m.group(1)) > 1:
        return int(m.group(1)), m.group(0)
    return 1, ""


def _best_flops(rec: dict, prefs: tuple[str, ...]):
    """Highest dense figure in the most modern precision the part publishes."""
    for prec in prefs:
        cands = [e for e in (rec.get("flops") or [])
                 if str(e.get("precision")) == prec
                 and isinstance(e.get("tflops"), (int, float)) and e["tflops"] > 0]
        if not cands:
            continue
        dense = [e for e in cands if e.get("dense")]
        if dense:
            return max(dense, key=lambda e: e["tflops"]), prec, "dense"
        return max(cands, key=lambda e: e["tflops"]), prec, "sparse-only"
    return None, None, None


def _buy_candidates(supply: dict[str, dict], accel_id: str,
                    sources: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Per-unit purchase candidates, plus everything rejected and why."""
    keep, drop = [], []
    for sid, s in supply.items():
        price = s.get("price_usd")
        if not isinstance(price, (int, float)) or price <= 0:
            continue
        if accel_id not in (s.get("accelerator_ids") or []):
            continue
        kind = s.get("kind")
        cls = "new" if kind in NEW_KINDS else "used" if kind in USED_KINDS else None
        if cls is None:
            continue
        basis = str(s.get("price_basis") or "")
        name = str(s.get("name") or "")
        if ((_ACCESSORY.search(name) and not _GPU_TOKEN.search(name)
             or _ACCESSORY_LEAD.search(name) or _ACCESSORY_FOR.search(name))
                and len(s.get("accelerator_ids") or []) == 1):
            drop.append({"supply": sid, "price_usd": price,
                         "reason": "the product is the carrier/chassis, not a GPU"})
            continue
        # name only: price_basis often quotes a sibling chassis price ("bare chassis, no GPUs")
        if re.search(r"\b(no|without)\s+gpus?\b|bare chassis", name, re.I):
            drop.append({"supply": sid, "price_usd": price,
                         "reason": "listed without the GPUs installed"})
            continue
        div, phrase = _bulk_divisor(name, basis)
        keep.append({"supply": sid, "class": cls, "price_usd": price / div, "raw_price_usd": price,
                     "divisor": div, "divisor_basis": phrase, "status": s.get("status"),
                     "listing": _listing(sid, s, sources, price / div, price, div)})
    return keep, drop


def _drop_outliers(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove prices that are clearly a different product from the rest for that accelerator."""
    if len(cands) < 2:
        return cands, []
    prices = sorted(c["price_usd"] for c in cands)
    n = len(prices)
    mid = prices[n // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2
    if mid <= 0:
        return cands, []
    keep, drop = [], []
    for c in cands:
        if c["price_usd"] > 3 * mid or c["price_usd"] < mid / 3:
            drop.append({"supply": c["supply"], "price_usd": round(c["raw_price_usd"], 2),
                         "reason": f"outside 3x this accelerator's median (${mid:,.0f}) — looks like "
                                   f"a different product (chassis, accessory or complete server)"})
        else:
            keep.append(c)
    return keep, drop


_OBSERVED = re.compile(r"OBSERVED[:\s]*([0-9]{4}-[0-9]{2}-[0-9]{2})")
_LISTED_TITLE = re.compile(r"listing ['\"\u2018\u201c]([^'\"\u2019\u201d]{6,120})['\"\u2019\u201d]")


def _condition(kind: str, name: str, basis: str) -> str:
    """What the listing says about the part's condition, in the listing's own terms."""
    blob = f"{name} {basis[:400]}".lower()
    if "refurb" in blob or "renewed" in blob:
        return "refurbished"
    if "open box" in blob or "open-box" in blob:
        return "open box"
    if "new old stock" in blob or "nos" in blob or "old stock" in blob:
        return "new old stock"
    if "server pull" in blob or "pull" in blob or "ex-datacenter" in blob or "ex-datacenter" in blob:
        return "used (datacenter pull)"
    if kind == "used-market" or kind == "auction" or kind == "broker":
        return "used"
    if kind in NEW_KINDS:
        return "new"
    return "unspecified"


def _listing(sid: str, s: dict, sources: dict[str, dict], price_per_unit: float,
             raw_price: float, divisor: int) -> dict:
    """Everything a reader needs to go and look at the actual listing.

    A price with no listing behind it is a number from nowhere, so this carries the record id,
    the seller, the condition, the date it was observed, and the source URL(s) — resolved from
    the supply record's `sources` into the source records, which is where the URLs live.
    """
    basis = str(s.get("price_basis") or "")
    urls = []
    for ref in (s.get("sources") or []):
        src = sources.get(ref) or {}
        url = src.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append({"url": url, "publisher": src.get("publisher"),
                         "accessed": src.get("accessed"), "source": ref})
    obs = _OBSERVED.search(basis)
    title = _LISTED_TITLE.search(basis)
    return {
        "supply": sid,
        "title": s.get("name"),
        "listed_title": title.group(1).strip() if title else None,
        "seller": s.get("vendor"),
        "condition": _condition(str(s.get("kind") or ""), str(s.get("name") or ""), basis),
        "observed": obs.group(1) if obs else (s.get("updated") if isinstance(s.get("updated"), str) else None),
        "region": s.get("region"),
        "availability": s.get("availability"),
        "channels": s.get("channels") or [],
        "price_usd": round(price_per_unit, 2),
        "raw_price_usd": round(raw_price, 2) if isinstance(raw_price, (int, float)) else None,
        "divisor": divisor,
        "status": s.get("status"),
        "confidence": s.get("confidence"),
        "urls": urls[:3],
    }


def _pick(cands: list[dict], cls: str) -> dict | None:
    pool = [c for c in cands if c["class"] == cls]
    if not pool:
        return None
    verified = [c for c in pool if c.get("status") == "verified"]
    return min(verified or pool, key=lambda c: c["price_usd"])


def _caveats(rec: dict, entry: dict, prec: str, density: str, cls: str, cand: dict) -> list[str]:
    out = []
    if density == "sparse-only":
        out.append("only a sparse figure is published; the dense number is lower, so this is listed "
                   "but not ranked")
    year = rec.get("release_year")
    if prec == "fp16":
        # only call it pre-Ampere when the date supports it: an RDNA3 card headlines fp16 and
        # does have bf16 support, so a blanket claim would be wrong about the part
        if isinstance(year, int) and year <= 2019:
            out.append("fp16 headline and no bf16 figure: a pre-Ampere part, so modern bf16 kernels "
                       "are not what it runs")
        else:
            out.append("scored on fp16 because no bf16 figure is recorded; the silicon may still run "
                       "bf16, so treat this as a floor")
    if entry.get("vendor_claim"):
        out.append("vendor-published figure")
    if isinstance(year, int) and year <= 2019:
        out.append("pre-Ampere design: no flash-attention kernels in most stacks and a narrow "
                   "software story")
    if cand.get("divisor", 1) > 1:
        out.append(f"price divided by {cand['divisor']} from a multi-unit listing "
                   f"('{cand['divisor_basis']}'), so this is a per-accelerator figure")
    if cand.get("status") != "verified":
        out.append("price is draft, not verified")
    if cls == "rental":
        out.append("per GPU-hour asking rate; excludes host, storage, egress and idle time")
    else:
        out.append("asking price, not a completed sale")
    return out


def build_flops_per_dollar() -> dict:
    accelerators = load(DIRS["accelerator"])
    supply = load(DIRS["supply"])
    sources = load(DIRS["source"])
    rows: list[dict] = []
    quant_rows: list[dict] = []
    excluded: list[dict] = []
    unpriced: list[dict] = []

    for aid, rec in sorted(accelerators.items()):
        entry, prec, density = _best_flops(rec, HEADLINE_PRECISION)
        qentry, qprec, qdensity = _best_flops(rec, QUANT_PRECISION)
        if entry is None and qentry is None:
            continue

        cands, drops = _buy_candidates(supply, aid, sources)
        cands, outliers = _drop_outliers(cands)
        for d in drops + outliers:
            srec = supply.get(d["supply"]) or {}
            ex_listing = _listing(d["supply"], srec, sources, d["price_usd"], d["price_usd"], 1)
            excluded.append(dict(d, accel=aid, accel_name=rec.get("name"),
                                 supply_name=srec.get("name"),
                                 seller=srec.get("vendor"),
                                 observed=ex_listing.get("observed"),
                                 url=(ex_listing["urls"][0]["url"] if ex_listing.get("urls") else None)))

        # rental: curated per-GPU-hour quotes, already divided for multi-GPU listings
        for sid, quotes in PRICE_QUOTES.items():
            for q in quotes:
                if q.get("accel") != aid or not q.get("usd"):
                    continue
                srec = supply.get(sid) or {}
                cands.append({"supply": sid, "class": "rental", "price_usd": float(q["usd"]),
                              "raw_price_usd": float(q["usd"]), "divisor": 1,
                              "divisor_basis": "curated quote", "status": "verified",
                              "tier": q.get("tier"), "quote_basis": q.get("basis"),
                              "listing": _listing(sid, srec, sources, float(q["usd"]),
                                                  float(q["usd"]), 1)})

        if not cands:
            if entry is not None:
                unpriced.append({"id": aid, "name": rec.get("name"), "reason": "no usable price"})
            continue

        for cls in ("new", "used", "rental"):
            cand = _pick(cands, cls)
            if not cand:
                continue
            # every listing behind this class, cheapest first, so the page can show the spread
            # and link out instead of asserting a single number
            same = sorted([c for c in cands if c["class"] == cls], key=lambda c: c["price_usd"])
            listings = [dict(c.get("listing") or {}) for c in same][:12]
            base = {
                "accel": aid, "name": rec.get("name"), "vendor": rec.get("vendor"),
                "listing": (cand.get("listing") or None),
                "listings": listings,
                "listing_count": len(same),
                "price_low": round(same[0]["price_usd"], 2) if same else None,
                "price_high": round(same[-1]["price_usd"], 2) if same else None,
                "year": rec.get("release_year"), "vram_gb": rec.get("vram_gb"),
                "memory_bandwidth_gbps": rec.get("memory_bandwidth_gbps"), "tdp_w": rec.get("tdp_w"),
                "price_class": cls, "price_usd": round(cand["price_usd"], 2),
                "price_unit": "per-gpu-hour" if cls == "rental" else "per-accelerator",
                "price_sources": len([c for c in cands if c["class"] == cls]),
                "price_status": cand.get("status"), "supply": cand["supply"],
                "divisor": cand.get("divisor", 1),
                "tier": cand.get("tier") if cls == "rental" else None,
            }
            if entry is not None:
                tflops = float(entry["tflops"])
                row = dict(base)
                # a 384-accelerator supernode divided down is a real $/TFLOP but not a purchase
                # anyone on this page can make, so it is listed and not ranked
                system_scale = cand.get("divisor", 1) >= 64
                row.update({
                    "system_scale": system_scale,
                    "metric": "flops", "precision": prec, "tflops": tflops, "density": density,
                    "dollars_per_tflop": round(cand["price_usd"] / tflops, 3),
                    "tflops_per_dollar": round(tflops / cand["price_usd"], 4),
                    "tflops_per_watt": (round(tflops / rec["tdp_w"], 3)
                                        if isinstance(rec.get("tdp_w"), (int, float)) and rec["tdp_w"] else None),
                    "ranked": density == "dense" and not system_scale,
                    "caveats": _caveats(rec, entry, prec, density, cls, cand)
                               + (["price is per accelerator inside a large multi-accelerator "
                                   "system, not a single-unit purchase"] if system_scale else []),
                })
                rows.append(row)
            if qentry is not None:
                qt = float(qentry["tflops"])
                qrow = dict(base)
                qrow.update({
                    "metric": "tops", "precision": qprec, "tops": qt, "density": qdensity,
                    "dollars_per_top": round(cand["price_usd"] / qt, 4),
                    "tops_per_dollar": round(qt / cand["price_usd"], 3),
                    "ranked": qdensity == "dense",
                    "caveats": _caveats(rec, qentry, qprec, qdensity, cls, cand),
                })
                quant_rows.append(qrow)

    def order(rs, key):
        rs = [r for r in rs if r["ranked"]]
        rs.sort(key=lambda r: ({"used": 0, "new": 1, "rental": 2}[r["price_class"]], r[key]))
        return rs

    ranked, quant_ranked = order(rows, "dollars_per_tflop"), order(quant_rows, "dollars_per_top")
    return {
        "generated": "2026-10-24",
        "method": {
            "what": "Theoretical dense throughput per dollar of hardware, split by how it is bought: "
                    "new, used, or rented by the GPU-hour. Lower $/TFLOP is better.",
            "flops": "The highest dense figure in the most modern precision the part publishes — bf16 "
                     "where it exists, fp16 otherwise. Sparse-only figures are listed but not ranked, "
                     "because 2:4 sparsity doubles the headline while the dense number is what a "
                     "serving stack sees.",
            "price": "Purchase prices are the cheapest verified asking price per accelerator; a "
                     "multi-unit listing is divided down and the divisor is named in the row. Prices "
                     "outside 3x an accelerator's median are treated as a different product and moved "
                     "to the excluded list with the reason. Rental rates are curated per-GPU-hour "
                     "quotes whose evidence substring is checked by cost_per_token.py.",
            "excluded_units": "Cloud records priced per 1M tokens (Groq, Cerebras, Together serverless, "
                              "the frontier APIs) are services, not hardware rental, and cannot share an "
                              "axis with a GPU-hour.",
            "listings": "Every row carries the listing it came from: the supply record id, the seller, "
                        "the condition the listing states, the date it was observed, and the source URL "
                        "resolved from that record's sources. Expand a row to see all listings for that "
                        "accelerator in that market, cheapest first — the spread is often wider than the "
                        "ranking gap, so a 2% difference between two accelerators may be noise while a "
                        "single accelerator quotes 2x from one seller to the next.",
            "caveat": "Theoretical FLOPS is not delivered FLOPS: a card with no bf16 kernels, no flash "
                      "attention, or a dead software stack will not reach its number. Read the caveats "
                      "column before buying anything on this page.",
        },
        "counts": {"ranked": len(ranked), "listed": len(rows), "quant": len(quant_ranked),
                   "excluded": len(excluded), "unpriced": len(unpriced)},
        "rows": rows, "quant_rows": quant_rows, "excluded": excluded, "unpriced": unpriced,
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

    print("Building FLOPS-per-dollar view...")
    fpd = build_flops_per_dollar()
    (OUT / "flops_per_dollar.json").write_text(
        json.dumps(fpd, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {fpd['counts']['ranked']} ranked rows, {fpd['counts']['quant']} quant rows, "
          f"{fpd['counts']['excluded']} prices excluded")

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

