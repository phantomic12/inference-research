#!/usr/bin/env python3
"""Phase C Oct-17 p7: Database Mart serving tables.

New coverage:
  nvidia-rtx-pro-2000-blackwell  first records ever (vLLM aggregate + Ollama
                                 per-request). Accelerator record created here.
  nvidia-rtx-pro-5000-blackwell  first vLLM aggregate serving rows. Earlier
                                 dm-pro5000-ollama-* records are a different run
                                 (Ollama 0.12.3, 2048-token prompt) and stay.
  nvidia-rtx-pro-6000-blackwell  Ollama 0.13.5 per-request eval rates.
  nvidia-rtx-a6000, nvidia-a40   vLLM aggregate output throughput.

Skipped: the 72 GB RTX PRO 5000 workstation SKU (no published measurement;
Hardware Corner and Database Mart both test the 48 GB card), the A6000
"100 concurrent" table (Request Numbers says 50 while Total Input Tokens says
10000), and every Ollama prompt-eval column (the prompt is one short question).
"""
import json, os
ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-17"

def w(rec, subdir):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"]); return False
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", subdir, rec["id"]); return True

def source(i, name, url, notes):
    return {"id": i, "type": "source", "name": name, "status": "verified",
            "confidence": 0.6, "updated": TODAY, "url": url,
            "publisher": "Database Mart", "kind": "benchmark",
            "published": "2026-10-07", "accessed": TODAY, "archived_url": None,
            "notes": notes}

def bench(i, name, engine, accs, model, metric, value, unit, meth, sources,
          interconnect=None, notes=""):
    return {"id": i, "type": "benchmark", "name": name, "engine_id": engine,
            "accelerator_ids": accs, "interconnect_ids": interconnect or ["pcie-gen4"],
            "model": model, "format_id": None, "metric": metric, "value": value,
            "unit": unit, "methodology": meth, "measured_by": "third_party",
            "reproducible": True, "energy_basis": None, "power_w": None,
            "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
            "thermal_state": None, "notes": notes, "status": "verified",
            "confidence": 0.6, "updated": TODAY, "sources": sources}

VLLM = ("Database Mart hosted-GPU report, page footer dated 10/07/2026. Engine is "
 "vLLM built from github.com/vllm-project/vllm with no release tag or commit "
 "printed, so the engine version is unpinned. Weights are the Hugging Face "
 "checkpoint at the precision the table labels 'Quantization 16' (bf16/fp16 not "
 "distinguished); format_id is null because data/quantization/ has no fp16/bf16 "
 "record. Single GPU, tensor parallel 1. Input length fixed at 100 tokens, output "
 "length fixed at 600 tokens. The 'Output (tokens/s)' column is aggregate generated "
 "tokens across the concurrent requests divided by benchmark duration, NOT a "
 "per-request decode rate. Median TTFT is the median time to first token under "
 "that same concurrency. Third-party measurement by the GPU host; confidence "
 "capped because the harness and the vLLM commit are unpublished. ")

w(source("databasemart-vllm-rtx-pro-5000-blackwell",
 "Database Mart vLLM benchmark, RTX PRO 5000 Blackwell (48 GB)",
 "https://www.databasemart.com/blog/vllm-gpu-benchmark-pro5000",
 VLLM + "Card: NVIDIA RTX Pro 5000 Blackwell Server Edition, 48 GB GDDR7 with ECC, "
 "1,344 GB/s, 300 W, PCIe 5.0 x16. This is the 48 GB SKU, not the 72 GB workstation "
 "SKU. Request count 1000. The page headline quotes about 2,879 output tok/s for "
 "Llama-3.1-8B; the table prints 2,878.77 and that is the recorded value."), "sources")

w(source("databasemart-vllm-rtx-pro-2000-blackwell",
 "Database Mart vLLM benchmark, RTX PRO 2000 Blackwell (16 GB)",
 "https://www.databasemart.com/blog/vllm-gpu-benchmark-pro2000",
 VLLM + "Card: NVIDIA RTX Pro 2000 Blackwell, 16 GB GDDR7 with ECC, 256 GB/s, 70 W, "
 "PCIe 5.0 x8. Request count 1000. The only published LLM measurement found for this SKU."), "sources")

w(source("databasemart-vllm-rtx-a6000",
 "Database Mart vLLM benchmark, RTX A6000, 50 concurrent",
 "https://www.databasemart.com/blog/vllm-gpu-benchmark-a6000",
 VLLM + "Card: the page says 'Nvidia Quadro RTX A6000' but the spec block (Ampere, "
 "compute capability 8.6, 10,752 CUDA cores, 336 tensor cores, 48 GB GDDR6) is the "
 "RTX A6000, not the Turing Quadro RTX 6000, so records attach to nvidia-rtx-a6000. "
 "ONLY the 50-concurrent scenario is recorded: the '100 concurrent' table prints "
 "Request Numbers = 50 while Total Input Tokens = 10000, so its concurrency is "
 "self-contradictory and was dropped."), "sources")

w(source("databasemart-vllm-a40",
 "Database Mart vLLM benchmark, NVIDIA A40, 50 and 100 concurrent",
 "https://www.databasemart.com/blog/vllm-gpu-benchmark-a40",
 VLLM + "Card: NVIDIA A40, Ampere, 48 GB GDDR6, 10,752 CUDA cores. Both the "
 "50-concurrent and the 100-concurrent scenarios are internally consistent "
 "(request count matches total input tokens / 100) and both are recorded. The page "
 "notes 14B models needed --max-model-len 4096 to load, and that the card reached "
 "80 C+ under load."), "sources")

w(source("databasemart-ollama-rtx-pro-6000-blackwell",
 "Database Mart Ollama 0.13.5 benchmark, RTX PRO 6000 Blackwell",
 "https://www.databasemart.com/blog/ollama-gpu-benchmark-pro6000",
 "Database Mart hosted-GPU report, page footer dated 10/07/2026. One NVIDIA RTX PRO "
 "6000 Blackwell, 96 GB GDDR7, compute capability 12.0, 24,064 CUDA cores. Engine "
 "Ollama 0.13.5 (llama.cpp backend; the llama.cpp commit is not printed). "
 "Quantization is the default Ollama pull, labelled 4-bit, exact GGUF quant not "
 "printed, so format_id is null. Prompt is the single question 'What is GPU?' with "
 "prompt token counts of 6 to 70, so the prompt-eval column is NOT recorded as "
 "prefill. Output length is whatever the model produced (356 to 2,729 tokens), not "
 "a fixed generation length. Eval rate is per-request decode, batch size 1, NOT "
 "aggregate. Distinct from the Hardware Corner llama-bench tables already filed."), "sources")

w(source("databasemart-ollama-rtx-pro-2000-blackwell",
 "Database Mart Ollama benchmark, RTX PRO 2000 Blackwell (16 GB)",
 "https://www.databasemart.com/blog/ollama-gpu-benchmark-pro2000",
 "Database Mart hosted-GPU report, page footer dated 10/07/2026. One NVIDIA RTX Pro "
 "2000 Blackwell, 16 GB GDDR7 with ECC. Engine is Ollama; unlike the PRO 6000 page "
 "this one does not print the Ollama version. Quantization is the default Ollama "
 "pull, labelled INT4, exact GGUF quant not printed, so format_id is null. Prompt "
 "is the single question 'What is GPU?' and output length is model-determined, so "
 "only the eval-rate column (per-request decode, batch size 1) is recorded."), "sources")

w({"id": "nvidia-rtx-pro-2000-blackwell", "type": "accelerator",
   "name": "NVIDIA RTX PRO 2000 Blackwell", "status": "verified",
   "confidence": 0.75, "updated": TODAY, "vendor": "nvidia",
   "architecture": "blackwell", "release_year": 2025, "process_nm": None,
   "form_factors": ["pcie"], "vram_gb": 16,
   "memory_type": "gddr7 with ECC", "memory_bus_bit": None,
   "memory_bandwidth_gbps": 256,
   "memory_bandwidth_basis": "Database Mart vLLM benchmark page "
     "(databasemart-vllm-rtx-pro-2000-blackwell, retrieved 2026-10-17) prints "
     "'Memory Bandwidth 256 GB/s' and '16GB GDDR7 with ECC' in its GPU-details "
     "block. Not transcribed from an NVIDIA spec table in this pass, so the bus "
     "width is left null.",
   "tdp_w": 70, "flops": [], "interconnect": ["pcie-gen5"],
   "unified_memory": False, "consumer": False, "onchip_cache": None,
   "sources": ["databasemart-vllm-rtx-pro-2000-blackwell"],
   "notes": "Entry workstation Blackwell card. Spec block taken from the Database "
     "Mart benchmark page (70 W, PCIe 5.0 x8, 256 GB/s, 16 GB GDDR7 ECC) rather "
     "than the NVIDIA product page, which was not re-fetched in this pass. First "
     "LLM measurements are the Database Mart vLLM aggregate table and the Ollama "
     "per-request eval rates added alongside."}, "accelerators")

def vllm_meth(card, model, reqs, dur, in_tok, out_tok):
    return ("Database Mart vLLM bench on one %s. Model %s, quantization column '16' "
            "(bf16/fp16 not distinguished; format_id null, no fp16 record exists). "
            "vLLM commit not printed. Single GPU, tensor parallel 1. %d requests, "
            "input length 100 tokens, output length 600 tokens, benchmark duration "
            "%s s, total input tokens %d, total generated tokens %d. AGGREGATE "
            "across the concurrent requests, not a per-request decode rate."
            % (card, model, reqs, dur, in_tok, out_tok))

P5 = [
 ("meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b", 2878.77, 3605.52, 174.59, 100000, 502596),
 ("Qwen/Qwen3-8B", "qwen3-8b", 2743.63, 3839.02, 178.45, 100000, 489611),
 ("google/gemma-3-12b-it", "gemma-3-12b-it", 1706.59, 5338.65, 291.36, 100000, 497150),
 ("openai/gpt-oss-20b", "gpt-oss-20b", 2113.77, 2747.92, 196.12, 100000, 414552),
]
for hf, mid, out_s, ttft, dur, in_tok, out_tok in P5:
    slug = mid.replace(".", "")
    meth = vllm_meth("RTX PRO 5000 Blackwell 48 GB GDDR7 ECC (Server Edition, not the 72 GB workstation SKU)", hf, 1000, dur, in_tok, out_tok)
    src = ["databasemart-vllm-rtx-pro-5000-blackwell"]
    acc = ["nvidia-rtx-pro-5000-blackwell"]
    w(bench("dm-pro5000-vllm-%s-c1000-out" % slug,
        "RTX PRO 5000 Blackwell vLLM: %s aggregate output throughput, 1000 requests" % hf,
        "vllm", acc, mid, "tps_aggregate", out_s,
        "tok/s (aggregate output tokens, 1000 concurrent)", meth, src, ["pcie-gen5"],
        "Output (tokens/s) column. Aggregate across 1000 requests, not the per-request eval rate."), "benchmarks")
    w(bench("dm-pro5000-vllm-%s-c1000-ttft" % slug,
        "RTX PRO 5000 Blackwell vLLM: %s median TTFT, 1000 requests" % hf,
        "vllm", acc, mid, "ttft_ms", ttft,
        "ms (median time to first token, 1000 concurrent)", meth, src, ["pcie-gen5"],
        "Median TTFT under the same 1000-request run. P99 TTFT is higher and is not recorded."), "benchmarks")

P2 = [
 ("meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b", 943.99, 9976.30, 531.73, 100000, 501950),
 ("Qwen/Qwen3-8B", "qwen3-8b", 941.32, 9595.67, 533.08, 100000, 501798),
 ("google/gemma-3-12b-it", "gemma-3-12b-it", 512.92, 16642.44, 974.96, 100000, 500075),
 ("openai/gpt-oss-20b", "gpt-oss-20b", 1037.61, 10275.87, 480.46, 100000, 498526),
]
for hf, mid, out_s, ttft, dur, in_tok, out_tok in P2:
    slug = mid.replace(".", "")
    meth = vllm_meth("RTX PRO 2000 Blackwell 16 GB GDDR7 ECC, 256 GB/s, 70 W", hf, 1000, dur, in_tok, out_tok)
    src = ["databasemart-vllm-rtx-pro-2000-blackwell"]
    acc = ["nvidia-rtx-pro-2000-blackwell"]
    w(bench("dm-pro2000-vllm-%s-c1000-out" % slug,
        "RTX PRO 2000 Blackwell vLLM: %s aggregate output throughput, 1000 requests" % hf,
        "vllm", acc, mid, "tps_aggregate", out_s,
        "tok/s (aggregate output tokens, 1000 concurrent)", meth, src, ["pcie-gen5"],
        "First published LLM measurement for this SKU. Output (tokens/s) column."), "benchmarks")
    w(bench("dm-pro2000-vllm-%s-c1000-ttft" % slug,
        "RTX PRO 2000 Blackwell vLLM: %s median TTFT, 1000 requests" % hf,
        "vllm", acc, mid, "ttft_ms", ttft,
        "ms (median time to first token, 1000 concurrent)", meth, src, ["pcie-gen5"],
        "Median TTFT under the same 1000-request run. P99 TTFT is several times the median on this 16 GB card and is not recorded."), "benchmarks")

OLL2000 = ("Ollama on one RTX PRO 2000 Blackwell 16 GB. Ollama version not printed "
 "on this page (the sibling PRO 6000 page is 0.13.5; this one is not asserted to "
 "match). Quantization is the default Ollama pull, labelled INT4, exact GGUF quant "
 "not printed, format_id null. Prompt is the single question 'What is GPU?'; output "
 "length is whatever the model produced. Eval rate is per-request decode, batch "
 "size 1, NOT aggregate.")
for slug, mid, name, rate, note in [
 ("llama31-8b", "llama-3-1-8b", "llama3.1:8b", 59.05, "4.9 GB, GPU util 88%."),
 ("qwen3-8b", "qwen3-8b", "qwen3:8b", 48.75, "5.2 GB, GPU util 92%."),
 ("gptoss-20b", "gpt-oss-20b", "gpt-oss:20b", 61.51, "14 GB, GPU util 87%. A 20B model on a 16 GB card landing next to the 8B rows."),
]:
    w(bench("dm-pro2000-ollama-%s-eval" % slug,
        "RTX PRO 2000 Blackwell Ollama: %s eval rate" % name,
        "ollama", ["nvidia-rtx-pro-2000-blackwell"], mid, "decode_tok_s", rate,
        "tok/s (eval rate, per-request, batch 1)", OLL2000,
        ["databasemart-ollama-rtx-pro-2000-blackwell"], ["pcie-gen5"], note), "benchmarks")

OLL6000 = ("Ollama 0.13.5 on one RTX PRO 6000 Blackwell 96 GB GDDR7 (compute "
 "capability 12.0). llama.cpp commit not printed. Quantization is the default "
 "Ollama pull, labelled 4-bit, exact GGUF quant not printed, format_id null. "
 "Prompt is the single question 'What is GPU?'; output length is model-determined. "
 "Eval rate is per-request decode, batch size 1, NOT aggregate. Not comparable to "
 "the Hardware Corner llama-bench rows for this card, which use fixed pp/tg lengths "
 "and a named GGUF quant. ")
for slug, mid, name, rate, out_n, size, note in [
 ("gptoss-20b", "gpt-oss-20b", "gpt-oss:20b", 185.09, 1246, 14, "GPU util 65%, GPU memory 33%."),
 ("gptoss-120b", "gpt-oss-120b", "gpt-oss:120b", 134.28, 2729, 65, "GPU util 60%, GPU memory 77%. A 120B-class model decoding faster than the 70B dense row on the same card."),
 ("llama33-70b", "llama-3.3-70b", "llama3.3:70b", 31.96, 579, 43, "GPU util 94%, GPU memory 41%."),
]:
    w(bench("dm-pro6000-ollama-%s-eval" % slug,
        "RTX PRO 6000 Blackwell Ollama 0.13.5: %s eval rate" % name,
        "ollama", ["nvidia-rtx-pro-6000-blackwell"], mid, "decode_tok_s", rate,
        "tok/s (eval rate, per-request, batch 1)",
        OLL6000 + "This row generated %d tokens; model size %d GB." % (out_n, size),
        ["databasemart-ollama-rtx-pro-6000-blackwell"], ["pcie-gen5"], note), "benchmarks")

A6000 = [
 ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", 1337.24, 583.54, 17.81, 5000, 23817),
 ("Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b-instruct", 695.51, 1095.08, 36.25, 5000, 25213),
 ("meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b", 1218.63, 593.26, 18.26, 5000, 22247),
]
for hf, mid, out_s, ttft, dur, in_tok, out_tok in A6000:
    slug = mid.replace(".", "")
    meth = vllm_meth("RTX A6000 48 GB GDDR6 (page mislabels it 'Quadro RTX A6000'; the Ampere 8.6 / 10752-core spec block is the RTX A6000)", hf, 50, dur, in_tok, out_tok)
    src = ["databasemart-vllm-rtx-a6000"]
    w(bench("dm-a6000-vllm-%s-c50-out" % slug,
        "RTX A6000 vLLM: %s aggregate output throughput, 50 concurrent" % hf,
        "vllm", ["nvidia-rtx-a6000"], mid, "tps_aggregate", out_s,
        "tok/s (aggregate output tokens, 50 concurrent)", meth, src, ["pcie-gen4"],
        "50-concurrent scenario only. The page's 100-concurrent table contradicts itself and is not recorded."), "benchmarks")
    w(bench("dm-a6000-vllm-%s-c50-ttft" % slug,
        "RTX A6000 vLLM: %s median TTFT, 50 concurrent" % hf,
        "vllm", ["nvidia-rtx-a6000"], mid, "ttft_ms", ttft,
        "ms (median time to first token, 50 concurrent)", meth, src, ["pcie-gen4"],
        "Median TTFT for the 50-concurrent run."), "benchmarks")

A40 = [
 (50, "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", 1094.54, 761.11, 23.48, 5000, 25703),
 (50, "meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b", 1016.60, 219.88, 21.81, 5000, 22170),
 (50, "Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b-instruct", 529.88, 1434.29, 47.58, 5000, 25213),
 (100, "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", 1123.99, 1093.55, 42.94, 10000, 48267),
 (100, "meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b", 1705.49, 487.81, 26.61, 10000, 45375),
 (100, "Qwen/Qwen2.5-14B-Instruct", "qwen2.5-14b-instruct", 584.14, 1426.07, 87.61, 10000, 51177),
]
for conc, hf, mid, out_s, ttft, dur, in_tok, out_tok in A40:
    slug = mid.replace(".", "")
    meth = vllm_meth("NVIDIA A40 48 GB GDDR6", hf, conc, dur, in_tok, out_tok)
    src = ["databasemart-vllm-a40"]
    w(bench("dm-a40-vllm-%s-c%d-out" % (slug, conc),
        "A40 vLLM: %s aggregate output throughput, %d concurrent" % (hf, conc),
        "vllm", ["nvidia-a40"], mid, "tps_aggregate", out_s,
        "tok/s (aggregate output tokens, %d concurrent)" % conc, meth, src, ["pcie-gen4"],
        "Both concurrency levels are internally consistent on this page. 14B rows required --max-model-len 4096 to load."), "benchmarks")
    w(bench("dm-a40-vllm-%s-c%d-ttft" % (slug, conc),
        "A40 vLLM: %s median TTFT, %d concurrent" % (hf, conc),
        "vllm", ["nvidia-a40"], mid, "ttft_ms", ttft,
        "ms (median time to first token, %d concurrent)" % conc, meth, src, ["pcie-gen4"],
        "Median TTFT. The page reports the card reaching 80 C+ under these loads; no watt figure is printed."), "benchmarks")

print("done")
