#!/usr/bin/env python3
"""Phase C Oct-17 p8: Database Mart tables the earlier passes did not file.

Three third-party tables, each checked cell-by-cell against data/benchmarks
before writing (accelerator id + rounded value):

  nvidia-a40                 six model rows the A40 vLLM page added after the
                             c50/c100 Llama/Qwen rows already on main. Same
                             harness, so the existing source is extended rather
                             than duplicated.
  nvidia-a100-40gb-pcie      first Database Mart vLLM rows for the 40 GB PCIe
                             card (Gemma 3 4B/12B, Qwen2.5 7B/14B).
  nvidia-rtx-a4000           first Ollama 0.5.11 per-request eval rates.

Dropped on purpose:
  * The V100 Ollama page is a 16 GB HBM2 card. The only V100 records here are
    the 32 GB PCIe and SXM2 SKUs, and a 16 GB number on a 32 GB id would be a
    false match.
  * RTX 2060 has no accelerator record (Turing, 6 GB) and cannot hold the 35B
    class this backfill is about, so no record was created for it.
  * A40 gemma-3-12b-it at 30 requests: the page's own Request Numbers column
    says 50 while the prose says 30, so concurrency is self-contradictory.
  * Every Ollama prompt-eval column: the prompt is one short question.
"""
import json
import os

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-17"


def w(rec, subdir):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"])
        return False
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", subdir, rec["id"])
    return True


def source(i, name, url, notes):
    return {"id": i, "type": "source", "name": name, "status": "verified",
            "confidence": 0.6, "updated": TODAY, "url": url,
            "publisher": "Database Mart", "kind": "benchmark",
            "published": "2026-10-07", "accessed": TODAY, "archived_url": None,
            "notes": notes}


def bench(i, name, engine, acc, model, metric, value, unit, meth, src, note):
    return {"id": i, "type": "benchmark", "name": name, "engine_id": engine,
            "accelerator_ids": [acc], "interconnect_ids": ["pcie-gen4"],
            "model": model, "format_id": None, "metric": metric, "value": value,
            "unit": unit, "methodology": meth, "measured_by": "third_party",
            "reproducible": True, "energy_basis": None, "power_w": None,
            "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
            "thermal_state": None, "notes": note, "status": "verified",
            "confidence": 0.6, "updated": TODAY, "sources": [src]}


VLLM = ("Database Mart hosted-GPU report, page footer dated 10/07/2026. Engine is "
        "vLLM built from github.com/vllm-project/vllm with no release tag or commit "
        "printed, so the engine version is unpinned. Weights are the Hugging Face "
        "checkpoint at the precision the table labels 'Quantization 16' (bf16/fp16 "
        "not distinguished); format_id is null because data/quantization/ has no "
        "fp16/bf16 record. Single GPU, tensor parallel 1. Input length fixed at 100 "
        "tokens, output length fixed at 600 tokens. The 'Output (tokens/s)' column is "
        "aggregate generated tokens across the concurrent requests divided by "
        "benchmark duration, NOT a per-request decode rate. Median TTFT is the median "
        "time to first token under that same concurrency. Third-party measurement by "
        "the GPU host; confidence capped because the harness and the vLLM commit are "
        "unpublished. ")

# --- sources -------------------------------------------------------------
w(source("databasemart-vllm-a100-40gb",
         "Database Mart vLLM benchmark, A100 40GB PCIe, 50 concurrent",
         "https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-40gb",
         VLLM + "Card: the page says 'Nvidia A100' and the spec block (Ampere, 6,912 "
         "CUDA cores, 432 tensor cores, 40 GB HBM2, 19.5 TFLOPS FP32) is the 40 GB "
         "PCIe card, so records attach to nvidia-a100-40gb-pcie and not the 40 GB "
         "SXM4 or either 80 GB SKU. Request count 50 throughout. 14B rows needed "
         "--max-model-len 4096 to load."), "sources")

w(source("databasemart-ollama-rtx-a4000",
         "Database Mart Ollama 0.5.11 benchmark, RTX A4000 16GB",
         "https://www.databasemart.com/blog/ollama-gpu-benchmark-a4000",
         "Database Mart hosted-GPU report, page footer dated 10/07/2026. One NVIDIA "
         "RTX A4000 (the page calls it 'Quadro RTX A4000'; compute capability 8.6, "
         "Ampere, 6,144 CUDA cores, 16 GB GDDR6 is the RTX A4000), run as a Windows "
         "VPS. Engine Ollama 0.5.11 (llama.cpp backend; the llama.cpp commit is not "
         "printed). Quantization is the default Ollama pull, labelled 4-bit, exact "
         "GGUF quant not printed, so format_id is null. Prompt is a single short "
         "question and output length is model-determined, so only the eval-rate "
         "column (per-request decode, batch size 1) is recorded. The gemma2:27b row "
         "is recorded WITH its GPU-utilization figure of 1% and CPU rate of 70-86%, "
         "because the 16 GB weight does not fit in 16 GB VRAM and the number is a "
         "CPU-spill measurement, not a GPU decode rate."), "sources")

# --- A40: six model rows absent from the earlier c50/c100 filing ---------
A40 = ("nvidia-a40", "databasemart-vllm-a40")
A40M = (VLLM + "Card: NVIDIA A40, Ampere, 48 GB GDDR6, 10,752 CUDA cores. ")
a40_rows = [
    ("deepseek-r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 50,
     1155.80, 417.54, 20.61, 5000, 23826),
    ("deepseek-r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 100,
     1333.21, 820.62, 29.21, 10000, 38940),
    ("deepseek-r1-distill-qwen-14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 50,
     446.73, 408.12, 26.72, 5000, 11937),
    ("gemma-3-4b-it", "google/gemma-3-4b-it", 50,
     756.72, 1153.98, 29.67, 5000, 22452),
    ("gemma-3-4b-it", "google/gemma-3-4b-it", 100,
     401.22, 671.18, 48.12, 10000, 19307),
    ("gemma-3-12b-it", "google/gemma-3-12b-it", 50,
     546.93, 1865.75, 40.47, 5000, 22134),
]
for model, hf, n, out, ttft, dur, tin, tout in a40_rows:
    slug = model.replace(".", "")
    base = "dm-a40-vllm-%s-c%d" % (slug, n)
    common = (A40M + "%d requests, input length 100, output length 600, benchmark "
              "duration %.2f s, total input tokens %d, total generated tokens %d. "
              % (n, dur, tin, tout))
    w(bench(base + "-out",
            "A40 vLLM: %s aggregate output throughput, %d concurrent" % (hf, n),
            "vllm", A40[0], model, "tps_aggregate", out,
            "tok/s (aggregate output tokens, %d concurrent)" % n,
            common + "AGGREGATE across the concurrent requests, not a per-request decode rate.",
            A40[1], "Output (tokens/s) column. These six model rows were not in the "
            "earlier A40 filing, which covered Llama 3.1 8B and Qwen2.5 7B/14B only."), "benchmarks")
    w(bench(base + "-ttft",
            "A40 vLLM: %s median TTFT, %d concurrent" % (hf, n),
            "vllm", A40[0], model, "ttft_ms", ttft,
            "ms (median time to first token, %d concurrent)" % n,
            common + "MEDIAN across the concurrent requests, not a single-request TTFT.",
            A40[1], "Median TTFT (ms) column."), "benchmarks")

# --- A100 40GB PCIe ------------------------------------------------------
A100 = "nvidia-a100-40gb-pcie"
A100S = "databasemart-vllm-a100-40gb"
A100M = (VLLM + "Card: A100 40 GB PCIe (6,912 CUDA cores, 40 GB HBM2), not the SXM4 "
         "or 80 GB cards. ")
a100_rows = [
    ("gemma-3-4b-it", "google/gemma-3-4b-it", 3385.72, 401.91, 7.88, 26678),
    ("gemma-3-12b-it", "google/gemma-3-12b-it", 2091.15, 574.88, 12.77, 26704),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", 2225.26, 615.61, 11.43, 25435),
    ("qwen2.5-14b-instruct", "Qwen/Qwen2.5-14B-Instruct", 420.31, 333.83, 27.52, 11567),
]
for model, hf, out, ttft, dur, tout in a100_rows:
    slug = model.replace(".", "")
    base = "dm-a100-40gb-vllm-%s-c50" % slug
    common = (A100M + "50 requests, input length 100, output length 600, benchmark "
              "duration %.2f s, total input tokens 5000, total generated tokens %d. "
              % (dur, tout))
    note14 = (" 14B row required --max-model-len 4096 to load."
              if "14b" in model else "")
    w(bench(base + "-out",
            "A100 40GB PCIe vLLM: %s aggregate output throughput, 50 concurrent" % hf,
            "vllm", A100, model, "tps_aggregate", out,
            "tok/s (aggregate output tokens, 50 concurrent)",
            common + "AGGREGATE across the concurrent requests, not a per-request decode rate.",
            A100S, "Output (tokens/s) column." + note14), "benchmarks")
    w(bench(base + "-ttft",
            "A100 40GB PCIe vLLM: %s median TTFT, 50 concurrent" % hf,
            "vllm", A100, model, "ttft_ms", ttft,
            "ms (median time to first token, 50 concurrent)",
            common + "MEDIAN across the concurrent requests, not a single-request TTFT.",
            A100S, "Median TTFT (ms) column." + note14), "benchmarks")

# --- RTX A4000 Ollama ----------------------------------------------------
A4 = "nvidia-rtx-a4000"
A4S = "databasemart-ollama-rtx-a4000"
A4M = ("Database Mart Ollama 0.5.11 on one RTX A4000 16 GB GDDR6 (Windows VPS, "
      "24 cores, 32 GB RAM). Default Ollama pull, labelled 4-bit, exact GGUF quant "
      "not printed, so format_id is null. Prompt is a single short question and the "
      "output length is model-determined, so this is the eval-rate column only: "
      "per-request decode, batch size 1, NOT aggregate and NOT a prefill rate. "
      "The llama.cpp commit inside Ollama 0.5.11 is not printed. ")
a4000_rows = [
    ("deepseek-r1-7b", "deepseek-r1:7b", 52.61, "77%"),
    ("deepseek-r1-distill-llama-8b", "deepseek-r1:8b", 51.60, "78%"),
    ("deepseek-r1-14b", "deepseek-r1:14b", 30.20, "83%"),
    ("deepseek-coder-v2-16b", "deepseek-coder-v2:16b", 22.89, "40%"),
    ("llama-2-7b", "llama2:7b", 65.06, "82%"),
    ("llama-2-13b", "llama2:13b", 38.46, "89%"),
    ("llama-3.1-8b", "llama3.1:8b", 51.35, "78%"),
    ("mistral-7b", "mistral:7b", 64.16, "81%"),
    ("gemma-2-9b", "gemma2:9b", 39.04, "73%"),
    ("gemma-2-27b", "gemma2:27b", 2.38, "1%"),
    ("qwen2.5-7b", "qwen2.5:7b", 52.68, "12%"),
    ("qwen2.5-14b", "qwen2.5:14b", 30.05, "80%"),
]
for model, tag, rate, gpu in a4000_rows:
    spill = ""
    if tag == "gemma2:27b":
        spill = (" GPU utilization was 1% and CPU rate 70-86% on this row: the 16 GB "
                 "weight does not fit in 16 GB, so this is a CPU-spill number.")
    w(bench("dm-a4000-ollama-%s-eval" % model.replace(".", ""),
            "RTX A4000 Ollama 0.5.11: %s per-request eval rate" % tag,
            "ollama", A4, model, "decode_tok_s", rate,
            "tok/s (per-request decode, batch size 1)",
            A4M + "Ollama tag %s. Reported GPU utilization %s." % (tag, gpu) + spill,
            A4S, "Eval Rate (tokens/s) column. Per-request, not aggregate." + spill),
      "benchmarks")

print("done")
