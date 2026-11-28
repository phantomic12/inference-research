#!/usr/bin/env python3
"""Phase C Oct-15 p8: benchmark backfill — zero/thin coverage, affordable class.

Targets:
  - apple-m2-max        : FIRST coverage. llama.cpp discussion #4167 canonical
                          Llama-2-7B llama-bench table (Metal, commit 8e672ef):
                          M2 Max 30-core + 38-core, F16/Q8_0/Q4_0, pp512+tg128.
                          + siliconscore 70B row + localscore llamafile rows.
  - apple-m3-max        : thin (3). Same #4167 canonical rows for 30c + 40c.
  - nvidia-a100-40gb-pcie : FIRST coverage. llamabench.ai community rows
                          (@wmeddie, llama.cpp b9935, FA on, ctx 0.5k).
  - nvidia-a100-40gb-sxm4 : +2 llamabench ornith rows (b10808, kv q8_0).
  - amd-radeon-pro-w7900  : +2 llamabench rows (@lhl, b9600, shisa 14B).

Skipped: grayskull-e150 (vendor EOL, no LLM bench exists), epyc-9454p (2 exist),
mi250x (2 exist, kuhaku). Unobtainium skipped per brief.
"""
import json, os

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-15"

def w(rec, subdir="benchmarks"):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"]); return
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth,
          measured_by, sources, status="verified", conf=0.8, repro=None,
          notes=""):
    return {
        "id": id, "type": "benchmark", "name": name,
        "engine_id": engine, "accelerator_ids": accs,
        "interconnect_ids": [],
        "model": model, "format_id": fmt, "metric": metric, "value": value,
        "unit": unit, "methodology": meth, "measured_by": measured_by,
        "reproducible": repro, "energy_basis": None, "power_w": None,
        "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
        "thermal_state": None, "notes": notes,
        "status": status, "confidence": conf, "updated": TODAY,
        "sources": sources,
    }

def source(id, name, url, publisher, kind, notes, status="verified", conf=0.5):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url,
            "publisher": publisher, "kind": kind, "published": None,
            "accessed": TODAY, "archived_url": None, "notes": notes}

# ---------------- new sources ----------------

wsrc = lambda rec: w(rec, subdir="sources")

wsrc(source(
    "llamabench-ai-crowdsourced-llama-cpp-results",
    "llamabench.ai — crowd-sourced llama.cpp result database (per-model decode/prefill rows)",
    "https://llamabench.ai/browse",
    "llamabench.ai (community submissions)", "database",
    "Crowd-submitted llama.cpp runs. Each row gives decode tok/s (tg128), prefill tok/s, "
    "backend+build tag (e.g. 'llama.cpp b9935 · FA on'), quant, KV-cache dtype, ctx depth, "
    "speculative-decoding flag, and reporter handle. Per-request batch-1 single-stream "
    "decode; prefill is the site's reported figure. n=1 per row unless the model page shows "
    "more. Lower confidence than a curated scoreboard but a real measurement, not a spec. "
    "Accessed for A100-PCIe-40GB (10 rows), A100-SXM4-40GB (2 rows), W7900 (1 row).",
    conf=0.55))

wsrc(source(
    "localscore-ai-accelerator-benchmark-pages",
    "localscore.ai — measured llamafile/llama.cpp tok/s per accelerator page",
    "https://www.localscore.ai/",
    "LocalScore", "database",
    "Machine-level measured tok/s database; rows cited via siliconscore.com benchmarks.csv "
    "export which links out to localscore.ai/accelerator/<id> pages. llamafile runner, "
    "Q4_K_M models (llama-3.1-8b-instruct, llama-3.2-1b-instruct, qwen-2.5-14b-instruct), "
    "prompt-eval tok/s and avg generation tok/s. Per-request, batch-1.",
    conf=0.6))

# ---------------- apple-m2-max (FIRST COVERAGE) ----------------
# llama.cpp discussion #4167 canonical table, Metal, commit 8e672ef,
# llama-2-7b, pp512 (batch) / tg128 (batch-1 decode), -ngl 99.
S4167 = "llama-cpp-discussion-4167-apple-silicon-llama-bench-table"
METH_4167 = ("llama.cpp discussion #4167 canonical M-series table, maintainer-run "
             "llama-bench on commit 8e672ef (Nov 2023, pre-FA-improvement baseline): "
             "Llama-2-7B GGUF, Metal backend, -ngl 99, pp512 = prompt-processing "
             "batch, tg128 = single-stream decode. Batch-1 per-request decode. "
             "GPU-core variant: {gpu}.")

M2 = "apple-m2-max"
rows_4167_m2 = [
    # (gpu_variant, quant, metric, value)
    ("30-core", "F16", "prefill_tok_s", 600.46), ("30-core", "F16", "decode_tok_s", 24.16),
    ("30-core", "Q8_0", "prefill_tok_s", 540.15), ("30-core", "Q8_0", "decode_tok_s", 39.97),
    ("30-core", "Q4_0", "prefill_tok_s", 537.60), ("30-core", "Q4_0", "decode_tok_s", 60.99),
    ("38-core", "F16", "prefill_tok_s", 755.67), ("38-core", "F16", "decode_tok_s", 24.65),
    ("38-core", "Q8_0", "prefill_tok_s", 677.91), ("38-core", "Q8_0", "decode_tok_s", 41.83),
    ("38-core", "Q4_0", "prefill_tok_s", 671.31), ("38-core", "Q4_0", "decode_tok_s", 65.95),
]
fmt_map = {"F16": None, "Q8_0": "gguf-q8-0", "Q4_0": "gguf-q4-0"}
for gpu, quant, metric, val in rows_4167_m2:
    phase = "pp512" if metric == "prefill_tok_s" else "tg128"
    w(bench(
        f"llamacpp-metal-m2-max-{gpu.replace('-','')}-llama27b-{quant.lower().replace('_','')}-{phase}-8e672ef",
        f"Apple M2 Max ({gpu} GPU), llama.cpp Metal: Llama-2-7B {quant} {phase}",
        "llama-cpp", [M2], "Llama-2-7B (GGUF, TheBloke llama-2-7b.{q}.gguf)".replace("{q}", quant),
        fmt_map[quant], metric, val,
        f"tok/s ({phase}, Metal, -ngl 99)",
        METH_4167.format(gpu=gpu),
        "third_party", [S4167], conf=0.85, repro=True,
        notes="Canonical #4167 table row. 8e672ef predates the Metal FA improvements; "
              "a current build would be higher. Decode on a 7B is memory-bandwidth-bound."))

# siliconscore: M2 Max 38c 64GB Llama-3.3-70B 4bit LM Studio 8.8 tok/s decode
w(bench(
    "comm-apple-m2-max-llama33-70b-4bit-decode",
    "Apple M2 Max (38-core, 64GB), LM Studio/llama.cpp: Llama-3.3-70B 4-bit decode",
    "llama-cpp", [M2], "Llama-3.3-70B 4-bit", None,
    "decode_tok_s", 8.8,
    "tok/s (single-stream, 4-bit, LM Studio)",
    "Silicon Score measured CSV row (M2 Max 38-core GPU, 64 GB, LM Studio, 4bit, "
    "Llama 3.3 70B, avg tok/s 8.8). Per-request batch-1 decode.",
    "third_party",
    ["silicon-score-benchmarks-csv-machine-level-measured-tok-s-export"],
    conf=0.7, repro=None,
    notes="Consistent with the 8.15 t/s M3 Max 64GB Q4_K_M row and the ~8x bandwidth "
          "gap vs M1 Ultra (400 GB/s vs 800 GB/s): ~half the Ultra's 12.55 t/s."))

# localscore llamafile rows (via siliconscore CSV export)
SC = "silicon-score-benchmarks-csv-machine-level-measured-tok-s-export"
LS = "localscore-ai-accelerator-benchmark-pages"
w(bench(
    "localscore-m2-max-38c-96gb-llama31-8b-q4km-decode",
    "Apple M2 Max (38c, 96GB) llamafile: Llama-3.1-8B Q4_K_M decode+prefill",
    "llamafile", [M2], "llama-3-1-8b-instruct Q4_K_M", "gguf-q4-k-m",
    "decode_tok_s", 46.41,
    "tok/s avg generation (batch-1); prefill 484.06 t/s",
    "localscore.ai accelerator/515 measured row, exported via siliconscore "
    "benchmarks.csv: llamafile, Q4_K_M, M2 Max 38-core 96 GB. Per-request batch-1.",
    "third_party", [SC, LS], conf=0.7, repro=None))
w(bench(
    "localscore-m2-max-38c-96gb-llama31-8b-q4km-pp",
    "Apple M2 Max (38c, 96GB) llamafile: Llama-3.1-8B Q4_K_M prompt-eval",
    "llamafile", [M2], "llama-3-1-8b-instruct Q4_K_M", "gguf-q4-k-m",
    "prefill_tok_s", 484.06,
    "tok/s prompt-eval (batch); decode 46.41 t/s",
    "localscore.ai accelerator/515 measured row via siliconscore CSV.",
    "third_party", [SC, LS], conf=0.7, repro=None))
w(bench(
    "localscore-m2-max-38c-96gb-qwen25-14b-q4km-decode",
    "Apple M2 Max (38c, 96GB) llamafile: Qwen-2.5-14B Q4_K_M decode",
    "llamafile", [M2], "qwen-2-5-14b-instruct Q4_K_M", "gguf-q4-k-m",
    "decode_tok_s", 25.23,
    "tok/s avg generation (batch-1); prefill 252.47 t/s",
    "localscore.ai accelerator/515 measured row via siliconscore CSV.",
    "third_party", [SC, LS], conf=0.7, repro=None))
w(bench(
    "localscore-m2-max-30c-32gb-llama31-8b-q4km-decode",
    "Apple M2 Max (30c, 32GB) llamafile: Llama-3.1-8B Q4_K_M decode",
    "llamafile", [M2], "llama-3-1-8b-instruct Q4_K_M", "gguf-q4-k-m",
    "decode_tok_s", 31.19,
    "tok/s avg generation (batch-1); prefill 325.0 t/s",
    "localscore.ai accelerator/1079 measured row via siliconscore CSV.",
    "third_party", [SC, LS], conf=0.7, repro=None,
    notes="30-core/32GB variant; same silicon SKU, lower binned GPU. Useful for the "
          "used-MacBook market where 30c/32GB is the common config."))
# MLX Qwen3.5-27B 8bit reddit row
w(bench(
    "comm-m2-max-38c-96gb-qwen35-27b-8bit-mlx-decode",
    "Apple M2 Max (38c, 96GB) MLX: Qwen3.5-27B 8-bit decode",
    "mlx-lm", [M2], "Qwen3.5-27B 8bit", "mlx-affine-quantized",
    "decode_tok_s", 12.0,
    "tok/s (MLX, 8-bit, per-request)",
    "Silicon Score CSV row sourced to r/LocalLLaMA 1rkkeqh: MLX runtime, 8bit, "
    "~30 GB RAM footprint, M2 Max 38-core 96 GB. Per-request batch-1.",
    "third_party", [SC], conf=0.6, repro=None))

# ---------------- apple-m3-max (thin: 3 -> +14) ----------------
M3 = "apple-m3-max"
rows_4167_m3 = [
    ("30-core", "F16", "prefill_tok_s", 589.41), ("30-core", "F16", "decode_tok_s", 19.54),
    ("30-core", "Q8_0", "prefill_tok_s", 566.40), ("30-core", "Q8_0", "decode_tok_s", 34.30),
    ("30-core", "Q4_0", "prefill_tok_s", 567.59), ("30-core", "Q4_0", "decode_tok_s", 56.58),
    ("40-core", "F16", "prefill_tok_s", 779.17), ("40-core", "F16", "decode_tok_s", 25.09),
    ("40-core", "Q8_0", "prefill_tok_s", 757.64), ("40-core", "Q8_0", "decode_tok_s", 42.75),
    ("40-core", "Q4_0", "prefill_tok_s", 759.70), ("40-core", "Q4_0", "decode_tok_s", 66.31),
]
for gpu, quant, metric, val in rows_4167_m3:
    phase = "pp512" if metric == "prefill_tok_s" else "tg128"
    w(bench(
        f"llamacpp-metal-m3-max-{gpu.replace('-','')}-llama27b-{quant.lower().replace('_','')}-{phase}-8e672ef",
        f"Apple M3 Max ({gpu} GPU), llama.cpp Metal: Llama-2-7B {quant} {phase}",
        "llama-cpp", [M3], f"Llama-2-7B (GGUF, TheBloke llama-2-7b.{quant}.gguf)",
        fmt_map[quant], metric, val,
        f"tok/s ({phase}, Metal, -ngl 99)",
        METH_4167.format(gpu=gpu),
        "third_party", [S4167], conf=0.85, repro=True,
        notes="Canonical #4167 table row, commit 8e672ef baseline build."))

# siliconscore: M3 Max 64GB Llama-3.3-70B Q4_K_M context sweep (llama.cpp)
w(bench(
    "comm-apple-m3-max-llama33-70b-q4km-decode-258ctx",
    "Apple M3 Max (64GB) llama.cpp: Llama-3.3-70B Q4_K_M decode @258 ctx",
    "llama-cpp", [M3], "Llama-3.3-70B Q4_K_M", "gguf-q4-k-m",
    "decode_tok_s", 8.15,
    "tok/s single-stream @ 258 ctx; falls to 6.13 @ 32k ctx",
    "Silicon Score CSV row (source: r/LocalLLaMA 1he2v2n 2x3090-vs-M3Max "
    "comparison): llama.cpp, Q4_K_M, 64 GB M3 Max, pp 67.86 t/s, decode 8.15 t/s "
    "at 258-token context. Per-request batch-1.",
    "third_party", [SC], conf=0.7, repro=None,
    notes="Context-depth sweep on same row: 8.15 @258, 7.48 @8013, 7.0 @16001, "
          "6.13 @32170 — 25% decode decay across 258->32k context, KV-cache "
          "bandwidth cost."))
w(bench(
    "comm-apple-m3-max-llama33-70b-q4km-decode-32kctx",
    "Apple M3 Max (64GB) llama.cpp: Llama-3.3-70B Q4_K_M decode @32k ctx",
    "llama-cpp", [M3], "Llama-3.3-70B Q4_K_M", "gguf-q4-k-m",
    "decode_tok_s", 6.13,
    "tok/s single-stream @ 32,170 ctx",
    "Silicon Score CSV, same source row as the 258-ctx entry: decode 6.13 t/s at "
    "32,170-token context, pp 50.32 t/s.",
    "third_party", [SC], conf=0.7, repro=None))

# ---------------- nvidia-a100-40gb-pcie (FIRST COVERAGE) ----------------
# llamabench.ai community rows, all @wmeddie, llama.cpp b9935, FA on, kv f16,
# ctx 0.5k, per-request batch-1, no speculative decoding.
LB = "llamabench-ai-crowdsourced-llama-cpp-results"
A100P = "nvidia-a100-40gb-pcie"
METH_LB_A100 = ("llamabench.ai community row, reporter @wmeddie: llama.cpp build "
                "b9935, flash-attention on, KV cache f16, 0.5k context depth, "
                "no speculative decoding. Decode = tg128 single-stream "
                "(per-request, batch-1); prefill = site-reported figure. n=1 "
                "per model row — crowd-sourced, not a curated scoreboard.")
a100p_rows = [
    # (model_id_slug, model_name, fmt, decode, prefill)
    ("gpt-oss-20b-f16", "gpt-oss-20B F16", None, 187, 5568),
    ("qwen36-35b-a3b-q4kxl", "Qwen3.6-35B-A3B UD-Q4_K_XL (MoE)", "gguf-q4-k-m", 136, 3413),
    ("ornith10-35b-q4km", "Ornith-1.0-35B Q4_K_M", "gguf-q4-k-m", 148, 3321),
    ("qwen35-9b-q4kxl", "Qwen3.5-9B UD-Q4_K_XL", "gguf-q4-k-m", 125, 4009),
    ("qwen35-4b-q4kxl", "Qwen3.5-4B UD-Q4_K_XL", "gguf-q4-k-m", 166, 5306),
    ("gemma4-26b-a4b-q4kxl", "gemma-4-26B-A4B-it UD-Q4_K_XL (MoE)", "gguf-q4-k-m", 128, 4115),
    ("gemma4-31b-q4kxl", "gemma-4-31B-it UD-Q4_K_XL", "gguf-q4-k-m", 41.2, 1197),
    ("gemma4-12b-q4kxl", "gemma-4-12B-it UD-Q4_K_XL", "gguf-q4-k-m", 86.4, 2930),
    ("qwen36-27b-q4kxl", "Qwen3.6-27B UD-Q4_K_XL", "gguf-q4-k-m", 43.9, 1262),
]
for slug, model, fmt, dec, pp in a100p_rows:
    w(bench(
        f"llamabench-a100pcie40-{slug}-tg128",
        f"A100 40GB PCIe llama.cpp b9935: {model} decode",
        "llama-cpp", [A100P], model, fmt,
        "decode_tok_s", dec,
        f"tok/s decode (tg128, batch-1, ctx 0.5k, FA on); prefill {pp} t/s",
        METH_LB_A100, "third_party", [LB], conf=0.55, repro=None,
        notes="First measured coverage for the 40GB PCIe SKU. Compare the 80GB "
              "PCIe sibling (xiongjiedai RunPod, llama-3-8b Q4_K_M 138.3 t/s): "
              "the 40GB card's identical 1555 GB/s bandwidth puts its decode in "
              "the same class — consistent with the 125-187 t/s seen here on "
              "7-20B models."))
    w(bench(
        f"llamabench-a100pcie40-{slug}-pp",
        f"A100 40GB PCIe llama.cpp b9935: {model} prefill",
        "llama-cpp", [A100P], model, fmt,
        "prefill_tok_s", pp,
        f"tok/s prefill (ctx 0.5k, FA on); decode {dec} t/s tg128",
        METH_LB_A100, "third_party", [LB], conf=0.55, repro=None))

# ---------------- nvidia-a100-40gb-sxm4 (+2 llamabench rows) ----------------
A100S = "nvidia-a100-40gb-sxm4"
METH_LB_A100S = ("llamabench.ai community row, reporter @wmeddie: llama.cpp build "
                 "b10808, flash-attention on, KV cache q8_0, 0.5k context, no "
                 "speculative decoding. tg128 batch-1 decode / site prefill. n=1.")
for slug, model, dec, pp in [
    ("ornith15-9b-q4km", "Ornith-1.5-9B Q4_K_M", 129, 4052),
    ("ornith15-35b-a3b-q4km", "Ornith-1.5-35B-A3B Q4_K_M (MoE, 3B active)", 148, 3291),
]:
    w(bench(
        f"llamabench-a100sxm40-{slug}-tg128",
        f"A100 40GB SXM4 llama.cpp b10808: {model} decode",
        "llama-cpp", [A100S], model, "gguf-q4-k-m",
        "decode_tok_s", dec,
        f"tok/s decode (tg128, batch-1, ctx 0.5k, FA on, kv q8_0); prefill {pp} t/s",
        METH_LB_A100S, "third_party", [LB], conf=0.55, repro=None))

# ---------------- amd-radeon-pro-w7900 (+2 llamabench rows) ----------------
w(bench(
    "llamabench-w7900-shisa14b-q4km-tg128",
    "Radeon Pro W7900 48GB llama.cpp b9600: shisa-v2.1-unphi4-14b Q4_K_M decode",
    "llama-cpp", ["amd-radeon-pro-w7900"], "shisa-v2.1-unphi4-14b Q4_K_M",
    "gguf-q4-k-m", "decode_tok_s", 50.8,
    "tok/s decode (tg128, batch-1, ctx 0.5k, FA on); prefill 1721 t/s",
    "llamabench.ai community row, reporter @lhl: llama.cpp b9600, ROCm, FA on, "
    "kv f16, 0.5k ctx, no speculative decoding. n=1.",
    "third_party", [LB], conf=0.55, repro=None,
    notes="Adds a mid-size (14B) data point to the existing 7B Q4_0 scoreboard "
          "rows (tg128 ~121-127 t/s)."))
w(bench(
    "llamabench-w7900-shisa14b-q4km-pp",
    "Radeon Pro W7900 48GB llama.cpp b9600: shisa-v2.1-unphi4-14b Q4_K_M prefill",
    "llama-cpp", ["amd-radeon-pro-w7900"], "shisa-v2.1-unphi4-14b Q4_K_M",
    "gguf-q4-k-m", "prefill_tok_s", 1721.0,
    "tok/s prefill (ctx 0.5k, FA on); decode 50.8 t/s tg128",
    "llamabench.ai community row, reporter @lhl: llama.cpp b9600, ROCm, FA on, "
    "kv f16, 0.5k ctx. n=1.",
    "third_party", [LB], conf=0.55, repro=None))
