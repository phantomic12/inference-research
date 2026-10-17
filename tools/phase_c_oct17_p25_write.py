#!/usr/bin/env python3
"""Phase C Oct-17 p25: M5 Max 128GB context sweep the 4k siliconscore rows missed.

Source is the original post, not the Hardware Corner rewrite of it:
reddit.com/r/LocalLLaMA/comments/1rqnpvj (u/cryingneko, 2026-03).
mlx_lm.generate, stream_generate, --max-tokens 128, one request.

siliconscore already filed the two 4k rows for Qwen3.5-122B-A10B 4-bit
(881.466 pp, 65.853 tg) and the two 4k rows for Qwen3-Coder-Next 8-bit
(754.927 pp, 79.296 tg). Those four are NOT rewritten. Everything else in
the post — 16k/32k/64k of those two models, plus the whole Qwen3.5-27B
6-bit distill sweep and the whole gpt-oss-120b MXFP4-Q8 sweep — was absent.

Per-request, batch 1. Prompt token counts are the post's own "Prompt: N
tokens" lines, which are a few tokens over the nominal 4k/16k/32k/64k file.
"""
import json
import os

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-17"
SRC = "reddit-localllama-1rqnpvj-m5-max-128gb-mlx-lm-sweep"


def w(rec, subdir):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"])
        return False
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", subdir, rec["id"])
    return True


w({
    "id": SRC, "type": "source",
    "name": "r/LocalLLaMA: M5 Max 128GB mlx_lm.generate sweep (cryingneko, post 1rqnpvj)",
    "status": "verified", "confidence": 0.6, "updated": TODAY,
    "url": "https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/m5_max_just_arrived_benchmarks_incoming/",
    "publisher": "Reddit r/LocalLLaMA", "kind": "forum",
    "published": "2026-03-11", "accessed": TODAY, "archived_url": None,
    "notes": ("u/cryingneko, 14-inch MacBook Pro, M5 Max, 128 GB unified memory, "
              "18-core CPU, 40-core GPU. Numbers pasted from mlx_lm.generate "
              "stdout (Prompt: N tokens, X tokens-per-sec / Generation: M tokens, "
              "Y tokens-per-sec / Peak memory). Author discarded an earlier "
              "BatchGenerator run and re-ran under a fresh venv with "
              "stream_generate; only the stream_generate numbers are in the post. "
              "Hardware Corner's 2026-03-11 writeup rounds these same numbers; "
              "this source is the unrounded original. Forum, single self-report."),
}, "sources")

ACC = "local-apple-m5-max"
ENG = "mlx-lm"
METH = ("mlx_lm.generate on one 14-inch MacBook Pro, M5 Max, 128 GB unified "
        "memory, 40-core GPU (u/cryingneko). Command: mlx_lm.generate --model "
        "<local path> --prompt \"$(cat /tmp/prompt_<N>.txt)\" --max-tokens 128. "
        "ONE request, batch size 1: the prompt-tokens-per-sec line is prefill of "
        "that single prompt and the generation-tokens-per-sec line is decode of "
        "that single completion. NOT aggregate. mlx-lm version not printed. "
        "Author states these replaced an earlier BatchGenerator run.")

# (slug, model name, format_id, ctx label, prompt tokens, gen tokens, pp, tg, peak gb)
rows = [
    ("qwen35-122b-4bit", "Qwen3.5-122B-A10B-4bit", "mlx-affine-quantized",
     "16k", 16394, 128, 1239.734, 60.639, 73.803),
    ("qwen35-122b-4bit", "Qwen3.5-122B-A10B-4bit", "mlx-affine-quantized",
     "32k", 32778, 128, 1067.824, 54.923, 76.397),
    ("qwen3-coder-next-8bit", "Qwen3-Coder-Next-8bit", "mlx-affine-quantized",
     "16k", 16393, 60, 1802.144, 74.293, 88.176),
    ("qwen3-coder-next-8bit", "Qwen3-Coder-Next-8bit", "mlx-affine-quantized",
     "32k", 32777, 58, 1887.158, 68.624, 89.652),
    ("qwen3-coder-next-8bit", "Qwen3-Coder-Next-8bit", "mlx-affine-quantized",
     "64k", 65545, 61, 1432.730, 48.212, 92.605),
    ("qwen35-27b-6bit", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "mlx-affine-quantized", "4k", 4107, 128, 811.134, 23.648, 25.319),
    ("qwen35-27b-6bit", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "mlx-affine-quantized", "16k", 16395, 128, 686.682, 20.311, 27.332),
    ("qwen35-27b-6bit", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "mlx-affine-quantized", "32k", 32779, 128, 591.383, 14.908, 30.016),
    ("qwen35-27b-6bit", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
     "mlx-affine-quantized", "64k", 65547, 128, 475.828, 14.225, 35.425),
    ("gptoss-120b-mxfp4-q8", "gpt-oss-120b-MXFP4-Q8", "gguf-mxfp4-moe",
     "4k", 4164, 128, 1325.062, 87.873, 64.408),
    ("gptoss-120b-mxfp4-q8", "gpt-oss-120b-MXFP4-Q8", "gguf-mxfp4-moe",
     "16k", 16452, 128, 2710.460, 75.963, 64.857),
    ("gptoss-120b-mxfp4-q8", "gpt-oss-120b-MXFP4-Q8", "gguf-mxfp4-moe",
     "32k", 32836, 128, 2537.420, 64.469, 65.461),
]

n = 0
for slug, model, fmt, ctx, ptok, gtok, pp, tg, peak in rows:
    base = "m5max128-mlx-%s-%s" % (slug, ctx)
    common = (METH + " Model path /Volumes/SSD/Models/%s. Prompt: %d tokens. "
              "Generation: %d tokens. Peak memory: %.3f GB." % (model, ptok, gtok, peak))
    note = ("4k rows for this model were already filed from the siliconscore CSV "
            "(ss-m5max-128-*); this is the %s context the CSV pass did not keep." % ctx
            if ctx != "4k" else
            "No 4k row for this model exists; siliconscore only kept the 122B and Coder-Next 4k rows.")
    if w({
        "id": base + "-pp", "type": "benchmark",
        "name": "M5 Max 128GB mlx_lm: %s prefill at %s (%d prompt tokens)" % (model, ctx, ptok),
        "engine_id": ENG, "accelerator_ids": [ACC], "interconnect_ids": [],
        "model": model, "format_id": fmt, "metric": "prefill_tok_s", "value": pp,
        "unit": "tok/s (single-request prefill, %d prompt tokens)" % ptok,
        "methodology": common + " PREFILL of one request, not aggregate.",
        "measured_by": "third_party", "reproducible": True,
        "energy_basis": None, "power_w": None, "power_scope": None, "power_cap": None,
        "clock_lock_mhz": None, "thermal_state": None,
        "notes": "Prompt tokens-per-sec line of the mlx_lm.generate stdout. " + note,
        "status": "verified", "confidence": 0.55, "updated": TODAY, "sources": [SRC],
    }, "benchmarks"):
        n += 1
    if w({
        "id": base + "-tg", "type": "benchmark",
        "name": "M5 Max 128GB mlx_lm: %s decode at %s (%d generated tokens)" % (model, ctx, gtok),
        "engine_id": ENG, "accelerator_ids": [ACC], "interconnect_ids": [],
        "model": model, "format_id": fmt, "metric": "decode_tok_s", "value": tg,
        "unit": "tok/s (single-request decode, %d generated tokens, batch 1)" % gtok,
        "methodology": common + " DECODE of one request, batch size 1, not aggregate.",
        "measured_by": "third_party", "reproducible": True,
        "energy_basis": None, "power_w": None, "power_scope": None, "power_cap": None,
        "clock_lock_mhz": None, "thermal_state": None,
        "notes": "Generation tokens-per-sec line. " + note,
        "status": "verified", "confidence": 0.55, "updated": TODAY, "sources": [SRC],
    }, "benchmarks"):
        n += 1

print("benchmarks written:", n)
