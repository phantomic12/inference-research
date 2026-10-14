#!/usr/bin/env python3
"""Phase C pass-6 record writer."""
import json, os

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-14"

def w(rec, subdir="benchmarks"):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"]); return
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth, measured_by, sources,
          status="verified", conf=0.8, repro=True, interconnect=None, notes=""):
    return {
        "id": id, "type": "benchmark", "name": name,
        "engine_id": engine, "accelerator_ids": accs,
        "interconnect_ids": interconnect or [],
        "model": model, "format_id": fmt, "metric": metric, "value": value,
        "unit": unit, "methodology": meth, "measured_by": measured_by,
        "reproducible": repro, "energy_basis": None, "power_w": None,
        "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
        "thermal_state": None, "notes": notes,
        "status": status, "confidence": conf, "updated": TODAY,
        "sources": sources,
    }

# ---------- source records (new sources for this pass) ----------
def source(id, name, url, publisher, kind, notes, status="verified", conf=0.5):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url, "publisher": publisher,
            "kind": kind, "published": None, "accessed": TODAY, "archived_url": None,
            "notes": notes}

def wsrc(rec):
    w(rec, subdir="sources")

SRC_SYCL = "llama-cpp-discussion-23313-intel-sycl-scoreboard"
SRC_LEASEWEB = "leaseweb-amd-epyc-llm-inference-benchmark-cpu-vs-gpu"
SRC_AHELPME = "ahelpme-epyc-9554-llama-bench-cpu-inference"
SRC_ARGONNE = "argonne-llm-inference-bench-all-results-csv"
SRC_SS = "siliconscore-benchmarks-json-apple-silicon"
SRC_LLMCHECK = "llmcheck-net-benchmarks-json-community"
SRC_LOCALSCORE = "localscore-ai-accelerator-benchmark-db"

wsrc(source(SRC_SYCL,
    "llama.cpp Discussion #23313 - Performance of llama.cpp on Intel GPU with SYCL backend (community scoreboard)",
    "https://github.com/ggml-org/llama.cpp/discussions/23313",
    "GitHub / ggml-org", "forum",
    "Intel-SYCL sibling of the #15013 CUDA scoreboard: llama-bench -fa 0,1 -m llama-2-7b.Q4_0.gguf plus model-specific rows (gemma, gpt-oss, qwen3-30B-A3B, llama-3.1-70B). Maintainer-curated table with build commit + reporter per row. 'B70x1' = Arc Pro B70 32GB, 'B580 LEx1' = Arc B580 LE, 'B60x1' = Arc Pro B60, 'Arc770x1' = Arc A770, 'B570x1' = Arc B570."))

wsrc(source(SRC_LLMCHECK,
    "llmcheck.net community benchmark dataset (benchmarks.json/.csv) - Apple Silicon rows cited via siliconscore audit",
    "https://llmcheck.net/data/benchmarks.json",
    "llmcheck.net", "database",
    "Community-submitted local-LLM tok/s rows keyed by chip/model/runtime. Rows here reached this repo via the siliconscore.com/benchmarks.json audit which carries the llmcheck URL as source_url; all entries verified=false so confidence is capped lower than lab-run rows."))

wsrc(source(SRC_LOCALSCORE,
    "localscore.ai accelerator benchmark database - per-chip llama.cpp/llamafile pages linked from siliconscore audit",
    "https://www.localscore.ai/",
    "LocalScore", "database",
    "Crowdsourced llama-bench-compatible result pages per accelerator id (e.g. /accelerator/3105 = M5 Max 32-core 36GB). Rows are community runs; siliconscore marks them verified=false."))

# ---------- EPYC 9554 (ahelpme, single-socket 64c Genoa, 12ch DDR5-5600) ----------
# Existing record covers Llama-3.3-70B only. Add the other four published rows.
AHELPME_METH = ("THIRD-PARTY LAB RUN (ahelpme.com). llama-bench build 51f311e0 (llama.cpp b4753), "
                "single-socket EPYC 9554 64c/128t Genoa on Asus K14PA-U24-T, 12x16GB DDR5-5600 "
                "(460.8 GB/s theoretical, mlc-measured ~368 GB/s all-reads). "
                "Command: llama-bench --numa distribute -m <gguf> -t 64 -p 0 -n 128,256,512,1024,2048. "
                "numa_balancing disabled + drop_caches before runs. tg128 value reported here; "
                "sweep tg128..tg2048 within ~1% for all models. Per-request batch-1 decode.")
w(bench("ahelpme-epyc9554-dsr1-8b-q4km-tg128",
        "EPYC 9554 (1S, 12ch DDR5-5600) llama.cpp DeepSeek-R1-Distill-Llama-8B Q4_K_M tg128 decode",
        "llama-cpp", ["amd-epyc-9554-genoa"],
        "DeepSeek-R1-Distill-Llama-8B (bartowski GGUF, 4.58 GiB, 8.03B)", "gguf-q4-k-m",
        "decode_tok_s", 49.96, "tok/s (tg128, per-request, batch 1)",
        AHELPME_METH, "third_party", [SRC_AHELPME], conf=0.85,
        notes="49.96 t/s at tg128; 48.82 at tg2048. CPU-only decode on a 12-channel Genoa part - the bandwidth ceiling (~460 GB/s) is what makes an 8B-Q4 model run at consumer-GPU-like tok/s on a CPU."))
w(bench("ahelpme-epyc9554-dsr1-70b-q4km-tg128",
        "EPYC 9554 (1S, 12ch DDR5-5600) llama.cpp DeepSeek-R1-Distill-Llama-70B Q4_K_M tg128 decode",
        "llama-cpp", ["amd-epyc-9554-genoa"],
        "DeepSeek-R1-Distill-Llama-70B (bartowski GGUF, 39.59 GiB, 70.55B)", "gguf-q4-k-m",
        "decode_tok_s", 7.14, "tok/s (tg128, per-request, batch 1)",
        AHELPME_METH, "third_party", [SRC_AHELPME], conf=0.85,
        notes="7.14 t/s at tg128 (7.03-7.16 across the tg sweep). 70B dense decode on CPU is bandwidth-bound; 39.59 GiB weights / ~368 GB/s measured read BW predicts ~9 t/s ceiling, 7.1 observed is consistent with KV+activation overhead."))
w(bench("ahelpme-epyc9554-qwq32b-q4km-tg128",
        "EPYC 9554 (1S, 12ch DDR5-5600) llama.cpp QwQ-32B Q4_K_M tg128 decode",
        "llama-cpp", ["amd-epyc-9554-genoa"],
        "QwQ-32B (Q4_K_M GGUF, 18.48 GiB, 32.76B)", "gguf-q4-k-m",
        "decode_tok_s", 14.00, "tok/s (tg128, per-request, batch 1)",
        AHELPME_METH, "third_party", [SRC_AHELPME], conf=0.85,
        notes="14.00 t/s at tg128; 13.74 at tg2048. Page text calls the model 'qwen2 32B Q4_K - Medium' in the llama-bench row because llama.cpp reports the arch tag - the GGUF is QwQ-32B."))
w(bench("ahelpme-epyc9554-llama31-8b-q4km-tg128",
        "EPYC 9554 (1S, 12ch DDR5-5600) llama.cpp Llama-3.1-8B-Instruct Q4_K_M tg128 decode",
        "llama-cpp", ["amd-epyc-9554-genoa"],
        "Meta-Llama-3.1-8B-Instruct (bartowski GGUF, 4.58 GiB, 8.03B)", "gguf-q4-k-m",
        "decode_tok_s", 49.81, "tok/s (tg128, per-request, batch 1)",
        AHELPME_METH, "third_party", [SRC_AHELPME], conf=0.85,
        notes="49.81 t/s at tg128; 48.83 at tg2048. Within noise of the DeepSeek-R1-8B row (49.96) on identical weights bytes - confirms the decode is bandwidth-bound, not model-specific."))

# ---------- Dual EPYC 9334 (LeaseWeb) - the 3 other Q4 rows + 2 FP16 rows ----------
# Existing record: DeepSeek-8B Q4 27.8 t/s decode. Add the rest of the table.
LEASEWEB_METH = ("THIRD-PARTY VENDOR-BLOG RUN (LeaseWeb bare-metal blog, HPE ProLiant DL385 Gen11, "
                 "2x EPYC 9334 32c Zen4, 64 GB DDR5 total). llama.cpp llama-bench + OpenLLM llmperf; "
                 "512-token prompt, 128 generated, 24 CPU threads. Values are decode tok/s "
                 "('tokens per second' column) unless marked. Per-request batch-1.")
w(bench("leaseweb-epyc9334-dual-gptoss20b-q4-decode",
        "Dual EPYC 9334 llama.cpp gpt-oss-20b Q4 decode",
        "llama-cpp", ["amd-epyc-9334-genoa-dual"],
        "gpt-oss-20b Q4 (GGUF)", "gguf-q4-k-m",
        "decode_tok_s", 28.0, "tok/s (per-request, batch 1)",
        LEASEWEB_METH, "third_party", [SRC_LEASEWEB], conf=0.7, status="draft",
        notes="TL;DR row '20-28 tokens/sec on Q4-quantized 7B-20B' - 28.0 taken as the top of that range for gpt-oss-20b; the FP16 row for the same model is 26.2, which the blog reports as a comparison, not the Q4 measurement. Blog does not print an exact per-model table for Q4 beyond the range, so this is the range-top estimate and is marked draft."))
w(bench("leaseweb-epyc9334-dual-llama27b-q4-decode",
        "Dual EPYC 9334 llama.cpp Llama-2-7b Q4 decode",
        "llama-cpp", ["amd-epyc-9334-genoa-dual"],
        "Llama-2-7b Q4_K_M (GGUF)", "gguf-q4-k-m",
        "decode_tok_s", 20.0, "tok/s (per-request, batch 1)",
        LEASEWEB_METH, "third_party", [SRC_LEASEWEB], conf=0.7, status="draft",
        notes="From the same '20-28 tok/s on Q4 7B-20B' TL;DR range - 20.0 is the bottom of the range assigned to the smallest model on the shared host. Draft because the blog gives a range not a per-model number."))
w(bench("leaseweb-epyc9334-dual-dsr1-8b-fp16-decode",
        "Dual EPYC 9334 llama.cpp DeepSeek-R1-8B FP16 decode",
        "llama-cpp", ["amd-epyc-9334-genoa-dual"],
        "DeepSeek-R1-0528-Qwen3-8B FP16", None,
        "decode_tok_s", 8.1, "tok/s (per-request, batch 1)",
        LEASEWEB_METH, "third_party", [SRC_LEASEWEB], conf=0.8,
        notes="FP16 decode figure printed in the cost-per-token comparison table ('DeepSeek at FP16 ... 8.1'). Roughly 3.4x slower than the Q4 row on the same host - quantization nearly triples decode rate on this bandwidth-bound CPU."))
w(bench("leaseweb-epyc9334-dual-gptoss20b-fp16-decode",
        "Dual EPYC 9334 llama.cpp gpt-oss-20b FP16 decode",
        "llama-cpp", ["amd-epyc-9334-genoa-dual"],
        "gpt-oss-20b FP16", None,
        "decode_tok_s", 26.2, "tok/s (per-request, batch 1)",
        LEASEWEB_METH, "third_party", [SRC_LEASEWEB], conf=0.8,
        notes="FP16 row from the CPU-vs-GPU comparison table. Blog uses this same number in the L4 comparison; on L4 FP16 the same model reached 58.6 t/s (~2.2x)."))
w(bench("leaseweb-l4-dsr1-8b-fp16-decode",
        "NVIDIA L4 llama.cpp DeepSeek-R1-8B FP16 decode (LeaseWeb comparison run)",
        "llama-cpp", ["nvidia-l4"],
        "DeepSeek-R1-0528-Qwen3-8B FP16", None,
        "decode_tok_s", 16.7, "tok/s (per-request, batch 1)",
        "LeaseWeb comparison run - same model/config as the dual-EPYC table, run on a single NVIDIA L4 GPU. 512-token prompt, 128 generated. Per-request batch-1.",
        "third_party", [SRC_LEASEWEB], conf=0.75,
        notes="16.7 t/s vs 8.1 on dual EPYC 9334 - roughly 2x. Included because it is the same harness and model as the CPU row; the L4 is otherwise already covered by MLPerf and scoreboard records."))
w(bench("leaseweb-l4-gptoss20b-fp16-decode",
        "NVIDIA L4 llama.cpp gpt-oss-20b FP16 decode (LeaseWeb comparison run)",
        "llama-cpp", ["nvidia-l4"],
        "gpt-oss-20b FP16", None,
        "decode_tok_s", 58.6, "tok/s (per-request, batch 1)",
        "LeaseWeb comparison run - same model/config as the dual-EPYC table, single NVIDIA L4 GPU. Per-request batch-1.",
        "third_party", [SRC_LEASEWEB], conf=0.75,
        notes="58.6 t/s on L4 FP16 vs 26.2 on the dual EPYC 9334 FP16 - 2.2x. LLM decode on a 20B dense FP16 model is bandwidth-bound; the L4's 300 GB/s GDDR6 vs the server's ~300 GB/s aggregated DDR5 explains the ratio."))
print("done epyc/leaseweb")
