#!/usr/bin/env python3
"""Phase C second-pass record writer: emits benchmark+source JSON records."""
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

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth, measured_by,
          sources, status="verified", conf=0.8, repro=True, interconnect=None, notes=""):
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

def source(id, name, url, publisher, kind, notes, status="verified", conf=0.5):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url, "publisher": publisher,
            "kind": kind, "published": None, "accessed": TODAY, "archived_url": None,
            "notes": notes}

def wsrc(rec):
    w(rec, subdir="sources")

# ---------- new sources ----------
wsrc(source("intel-gaudi3-model-performance-1-24",
    "Intel Gaudi 3 AI Accelerator — Inference Model Performance Data (software 1.24, Optimum-Habana)",
    "https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html",
    "Intel", "benchmark",
    "Vendor-published max-throughput table for Gaudi 3 (Optimum-Habana, Gaudi sw 1.24, FP8). "
    "Verbatim: model/precision/input_len/output_len/#HPU/batch_size/throughput. Numbers are "
    "AGGREGATE tokens/sec across the batch (huge batch sizes), not per-request decode."))

wsrc(source("bench-mlperf-v4-1-dell-r760-h100pcie-gptj",
    "MLPerf Inference v4.1 — Dell R760 2x H100 PCIe 80GB, GPT-J 99%, TensorRT",
    "https://github.com/mlcommons/inference_results_v4.1/tree/main/closed/Dell/results/R760_H100_PCIe_80GBx2_TRT",
    "MLCommons / Dell", "benchmark",
    "Raw submission logs. TensorRT 10.2.0, CUDA 12.4, host 2x Xeon Platinum 8580. "
    "GPT-J 6B FP8, Offline + Server scenarios, tokens/sec (inferred) verbatim from "
    "mlperf_log_summary.txt. Aggregate across 2x H100-PCIe-80GB."))

wsrc(source("bench-mlperf-v4-1-cisco-h100pcie-gptj",
    "MLPerf Inference v4.1 — Cisco C245M8 2x H100 PCIe 80GB, GPT-J 99%, TensorRT",
    "https://github.com/mlcommons/inference_results_v4.1/tree/main/closed/Cisco/results/C245M8_H100_PCIe_80GBx2_TRT",
    "MLCommons / Cisco", "benchmark",
    "Raw submission logs. TensorRT, GPT-J 6B, Offline scenario. Tokens/sec (inferred) "
    "verbatim from mlperf_log_summary.txt."))

wsrc(source("bench-mlperf-v5-0-cisco-h100nvl-llama2-70b",
    "MLPerf Inference v5.0 — Cisco C245M8 2x H100 NVL 94GB, Llama-2-70B, TensorRT",
    "https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/Cisco/results/C245M8_H100NVL_94GBx2_TRT",
    "MLCommons / Cisco", "benchmark",
    "Raw submission logs. TensorRT 10.8, CUDA 12.8, host 2x EPYC 9224. Llama-2-70B FP8, "
    "Offline + Server scenarios, tokens/sec verbatim from mlperf_log_summary.txt. "
    "Aggregate across 2x H100-NVL-94GB."))

wsrc(source("bench-mlperf-v5-0-cisco-x215-h100nvl-llama2-70b",
    "MLPerf Inference v5.0 — Cisco X215M8 2x H100 NVL, Llama-2-70B Offline, TensorRT",
    "https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/Cisco/results/X215M8_H100NVLx2_TRT",
    "MLCommons / Cisco", "benchmark",
    "Raw submission logs. Llama-2-70B FP8 Offline, tokens/sec verbatim from "
    "mlperf_log_summary.txt."))

wsrc(source("bench-mlperf-v4-1-lenovo-8xh100nvl-llama2-70b",
    "MLPerf Inference v4.1 — Lenovo SR675 V3 8x H100 NVL, Llama-2-70B + GPT-J, TensorRT",
    "https://github.com/mlcommons/inference_results_v4.1/tree/main/closed/Lenovo/results/SR675_V3_8xH100_NVL_TRT",
    "MLCommons / Lenovo", "benchmark",
    "Raw submission logs. Llama-2-70B + GPT-J Offline/Server, tokens/sec verbatim. "
    "Aggregate across 8x H100-NVL-94GB."))

wsrc(source("bench-mlperf-v4-1-hpe-4xh100nvl-mixtral",
    "MLPerf Inference v4.1 (open) — HPE DL380a 4x H100 NVL, Mixtral-8x7B, TensorRT",
    "https://github.com/mlcommons/inference_results_v4.1/tree/main/open/HPE/results/HPE_ProLiant_DL380a_H100_NVL_94GBx4_TRT",
    "MLCommons / HPE", "benchmark",
    "Open-division submission logs. Mixtral-8x7B Offline, tokens/sec verbatim. "
    "Aggregate across 4x H100-NVL-94GB."))

wsrc(source("bench-mlperf-v6-0-intel-4x-bmg-b60",
    "MLPerf Inference v6.0 — Intel 1-node-4x Arc Pro B60 (BMG), llama2-70b / llama3.1-8b / gpt-oss-120b, vLLM",
    "https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/Intel/results/1-node-4x-BMG-B60",
    "MLCommons / Intel", "benchmark",
    "Closed-division submission. 4x Arc Pro B60 24GB on Xeon 698X host, PyTorch vLLM, "
    "PCIe Gen5 x8. Offline + Server tokens/sec verbatim from mlperf_log_summary.txt."))

wsrc(source("bench-mlperf-v6-0-intel-4x-bmg-b70",
    "MLPerf Inference v6.0 — Intel 1-node-4x Arc Pro B70 (BMG), llama2-70b / llama3.1-8b / gpt-oss-120b, vLLM",
    "https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/Intel/results/1-node-4x-BMG-B70",
    "MLCommons / Intel", "benchmark",
    "Closed-division submission. 4x Arc Pro B70 32GB on Xeon 698X host, PyTorch vLLM, "
    "PCIe Gen5 x16. Offline + Server tokens/sec verbatim from mlperf_log_summary.txt."))

wsrc(source("bench-mlperf-v5-1-intel-4x-bmg-pro-b60",
    "MLPerf Inference v5.1 — Intel 1-node-4x Arc Pro B60 Dual, llama2-70b / llama3.1-8b(-edge), vLLM",
    "https://github.com/mlcommons/inference_results_v5.1/tree/main/closed/Intel/results/1-node-4x-BMG-Pro-B60-Dual",
    "MLCommons / Intel", "benchmark",
    "Closed-division submission. 4x Arc Pro B60 Dual GPU, vLLM. Also a 1x single-GPU "
    "llama3.1-8b-edge SingleStream run (per-request TTFT/TPOT). tokens/sec verbatim."))

# ---------- Intel Gaudi 3 — vendor Optimum-Habana max-throughput (FP8) ----------
G = "intel-gaudi3-model-performance-1-24"
def gaudi(id, model, hpu, bs, inp, outp, tps, notes=""):
    return bench(id,
        f"Intel Gaudi 3, Optimum-Habana (sw 1.24): {model} FP8 {hpu}HPU bs{bs} {inp}in/{outp}out aggregate tok/s",
        "vllm-gaudi", ["intel-gaudi3"], model, "fp8-e4m3",
        "tps_aggregate", tps,
        f"output tok/s AGGREGATE across {hpu}x Gaudi 3 (batch {bs}; ~{tps/hpu:.0f} tok/s per HPU)",
        f"VENDOR-MEASURED (Intel model-performance table, Gaudi software 1.24, "
        f"Optimum-Habana + Hugging Face model, maxabs FP8). {model} FP8 on {hpu} HPU, "
        f"batch_size={bs}, input_len={inp}, output_len={outp}. This is MAX THROUGHPUT "
        f"under a very large batch — NOT a single-user decode figure. "
        f"~{tps/hpu:.0f} tok/s per HPU.",
        "vendor", [G],
        interconnect=["intel-gaudi3-roce-scaleup"] if hpu > 1 else [],
        notes=notes or ("Optimum-Habana run_generation.py path, use_hpu_graphs + "
                        "flash_attention_recompute. Reproducible via the vendor's "
                        "Docker recipe; numbers are peak-serving, not latency-oriented."))

w(gaudi("gaudi3-1hpu-llama31-8b-fp8-128-128",  "Llama-3.1-8B-Instruct", 1, 1536, 128, 128, 23922))
w(gaudi("gaudi3-1hpu-llama31-8b-fp8-2048-2048","Llama-3.1-8B-Instruct", 1, 256, 2048, 2048, 9040))
w(gaudi("gaudi3-2hpu-llama31-70b-fp8-128-2048","Llama-3.1-70B-Instruct", 2, 450, 128, 2048, 6514,
        notes="TP=2. ~3257 tok/s per HPU at bs450, 128->2048."))
w(gaudi("gaudi3-2hpu-llama31-70b-fp8-2048-2048","Llama-3.1-70B-Instruct", 2, 175, 2048, 2048, 3115))
w(gaudi("gaudi3-8hpu-llama31-70b-fp8-128-2048","Llama-3.1-70B-Instruct", 8, 768, 128, 2048, 21044,
        notes="TP=8 across the OAM baseboard. ~2630 tok/s per HPU."))
w(gaudi("gaudi3-8hpu-llama33-70b-fp8-128-2048","Llama-3.3-70B-Instruct", 8, 768, 128, 2048, 20980))
w(gaudi("gaudi3-8hpu-llama33-70b-fp8-2048-2048","Llama-3.3-70B-Instruct", 8, 600, 2048, 2048, 10747))

# ---------- NVIDIA H100 PCIe 80GB — MLPerf v4.1 GPT-J ----------
def h100pcie(id, sysname, src, scen, tps, per):
    return bench(id,
        f"NVIDIA H100 PCIe 80GB x2 ({sysname}), MLPerf v4.1 {scen}: GPT-J 6B aggregate tok/s",
        "tensorrt-llm", ["nvidia-h100-pcie"], "GPT-J 6B (FP8)", "fp8-e4m3",
        "tps_aggregate", tps,
        f"output tok/s AGGREGATE across 2x H100-PCIe-80GB ({scen}; ~{per} tok/s per card)",
        f"VENDOR-MEASURED (MLPerf v4.1 closed submission, {sysname}). 2x H100-PCIe-80GB, "
        f"TensorRT 10.2.0 / CUDA 12.4, GPT-J 6B FP8, {scen} scenario. 'Tokens per second "
        f"(inferred)' = {tps} verbatim from mlperf_log_summary.txt. AGGREGATE across 2 "
        f"cards and a deep batch — NOT a single-user decode figure (~{per}/card).",
        "vendor", [src],
        interconnect=["pcie-gen4"],
        notes="H100-PCIe is the 350W 2.0 TB/s HBM2e SKU (vs 700W 3.35 TB/s SXM). "
              "GPT-J is a 6B model — small; figure is prefill+decode aggregate.")

w(h100pcie("h100pcie-2x-mlperf-v4-1-gptj-offline-tokens",
           "Dell R760", "bench-mlperf-v4-1-dell-r760-h100pcie-gptj", "Offline",
           3317.08, "1658"))
w(h100pcie("h100pcie-2x-mlperf-v4-1-gptj-server-tokens",
           "Dell R760", "bench-mlperf-v4-1-dell-r760-h100pcie-gptj", "Server",
           2759.33, "1380"))
w(h100pcie("h100pcie-2x-mlperf-v4-1-gptj-offline-tokens-cisco",
           "Cisco C245M8", "bench-mlperf-v4-1-cisco-h100pcie-gptj", "Offline",
           3268.94, "1634"))

# ---------- NVIDIA H100 NVL 94GB — MLPerf ----------
def h100nvl(id, sysname, src, scen, model, tps, cards, note=""):
    return bench(id,
        f"NVIDIA H100 NVL 94GB x{cards} ({sysname}), MLPerf {scen}: {model} aggregate tok/s",
        "tensorrt-llm", ["nvidia-h100-nvl"], model + " (FP8)", "fp8-e4m3",
        "tps_aggregate", tps,
        f"output tok/s AGGREGATE across {cards}x H100-NVL-94GB ({scen}; ~{tps/cards:.0f} tok/s per card)",
        f"VENDOR-MEASURED (MLPerf closed submission, {sysname}). {cards}x H100-NVL-94GB "
        f"(PCIe, 3-card NVLink bridge pair), TensorRT, {model} FP8, {scen}. 'Tokens per "
        f"second' = {tps} verbatim from mlperf_log_summary.txt. AGGREGATE across {cards} "
        f"cards — NOT a single-user decode figure (~{tps/cards:.0f}/card).",
        "vendor", [src],
        interconnect=["pcie-gen5", "nvlink-4"],
        notes=note)

w(h100nvl("h100nvl-2x-mlperf-v5-0-llama2-70b-offline-tokens",
          "Cisco C245M8", "bench-mlperf-v5-0-cisco-h100nvl-llama2-70b", "Offline (v5.0)",
          "Llama-2-70B", 3741.32, 2))
w(h100nvl("h100nvl-2x-mlperf-v5-0-llama2-70b-server-tokens",
          "Cisco C245M8", "bench-mlperf-v5-0-cisco-h100nvl-llama2-70b", "Server (v5.0)",
          "Llama-2-70B", 3337.55, 2))
w(h100nvl("h100nvl-2x-mlperf-v5-0-llama2-70b-offline-tokens-x215",
          "Cisco X215M8", "bench-mlperf-v5-0-cisco-x215-h100nvl-llama2-70b", "Offline (v5.0)",
          "Llama-2-70B", 3879.95, 2))
w(h100nvl("h100nvl-8x-mlperf-v4-1-llama2-70b-offline-tokens",
          "Lenovo SR675 V3", "bench-mlperf-v4-1-lenovo-8xh100nvl-llama2-70b", "Offline (v4.1)",
          "Llama-2-70B", 15875.8, 8))
w(h100nvl("h100nvl-8x-mlperf-v4-1-gptj-offline-tokens",
          "Lenovo SR675 V3", "bench-mlperf-v4-1-lenovo-8xh100nvl-llama2-70b", "Offline (v4.1)",
          "GPT-J 6B", 14073.6, 8))
w(h100nvl("h100nvl-4x-mlperf-v4-1-mixtral-offline-tokens",
          "HPE DL380a", "bench-mlperf-v4-1-hpe-4xh100nvl-mixtral", "Offline (v4.1, open)",
          "Mixtral-8x7B", 18839, 4,
          note="Open division. MoE — high tokens/s is token-count not dense-equivalent."))

# ---------- NVIDIA A100 40GB — Vulkan scoreboard (variant ambiguous, note it) ----------
w(bench("a100-vulkan-llama27b-q40-nofa-tg128",
        "NVIDIA A100, llama.cpp Vulkan (2257758, coopmat2): Llama-2-7B Q4_0 tg128 t/s",
        "llama-cpp", ["nvidia-a100-40gb-sxm4"], "Llama-2-7B Q4_0 GGUF", "gguf-q4-0",
        "decode_tok_s", 160.78,
        "tok/s per-request tg128 (no FA); ±0.16",
        "THIRD-PARTY MEASURED (llama.cpp discussion #10879 Vulkan scoreboard, commit "
        "2257758, coopmat2). Llama-2-7B Q4_0, -ngl 99 -fa 0, tg128 = 160.78 ± 0.16 t/s. "
        "Scoreboard lists the chip only as 'Nvidia A100' — filed under the 40GB SXM4 "
        "record; the 40GB PCIe and 80GB SKUs share the same GA100 die and differ mainly "
        "in HBM capacity, so the decode figure is representative to ~first order.",
        "third_party", ["llama-cpp-discussion-10879-vulkan-scoreboard"],
        conf=0.55, status="draft",
        notes="VARIANT AMBIGUITY: scoreboard does not record which A100 SKU (40GB PCIe / "
              "40GB SXM4 / 80GB). Decode tok/s is bandwidth-bound; all A100 SKUs are "
              "1555-2039 GB/s so the figure is broadly representative but the 80GB "
              "HBM2e SKU would edge higher."))
w(bench("a100-vulkan-llama27b-q40-fa-tg128",
        "NVIDIA A100, llama.cpp Vulkan (2257758, coopmat2): Llama-2-7B Q4_0 FA tg128 t/s",
        "llama-cpp", ["nvidia-a100-40gb-sxm4"], "Llama-2-7B Q4_0 GGUF", "gguf-q4-0",
        "decode_tok_s", 170.56,
        "tok/s per-request tg128 (FA); ±0.02",
        "THIRD-PARTY MEASURED (llama.cpp discussion #10879, commit 2257758, coopmat2). "
        "Llama-2-7B Q4_0, -fa 1, tg128 = 170.56 ± 0.02 t/s. Same variant caveat as the "
        "no-FA row.",
        "third_party", ["llama-cpp-discussion-10879-vulkan-scoreboard"],
        conf=0.55, status="draft",
        notes="Same 'Nvidia A100' unspecified-SKU caveat; pp512 FA = 7064.40 t/s."))

# ---------- Intel Arc Pro B60 — MLPerf v6.0 + v5.1 ----------
def b60(id, scen, model, tps, cards, src, v, note=""):
    return bench(id,
        f"Intel Arc Pro B60 24GB x{cards}, MLPerf {v} {scen}: {model} aggregate tok/s",
        "vllm", ["intel-arc-pro-b60"], model + " (FP8)", None,
        "tps_aggregate", tps,
        f"output tok/s AGGREGATE across {cards}x Arc Pro B60 ({scen}; ~{tps/cards:.0f} tok/s per card)",
        f"VENDOR-MEASURED (MLPerf {v} closed submission, Intel). {cards}x Arc Pro B60 "
        f"24GB GDDR6 (PCIe Gen5 x8) on Xeon 698X host, PyTorch vLLM, {model}, {scen}. "
        f"'Tokens per second' = {tps} verbatim from mlperf_log_summary.txt. AGGREGATE "
        f"across {cards} cards — ~{tps/cards:.0f}/card.",
        "vendor", [src],
        interconnect=["pcie-gen5"],
        notes=note)

w(b60("b60-4x-mlperf-v6-0-llama2-70b-offline-tokens", "Offline", "Llama-2-70B",
      1697.66, 4, "bench-mlperf-v6-0-intel-4x-bmg-b60", "v6.0"))
w(b60("b60-4x-mlperf-v6-0-llama2-70b-server-tokens", "Server", "Llama-2-70B",
      1106.26, 4, "bench-mlperf-v6-0-intel-4x-bmg-b60", "v6.0"))
w(b60("b60-4x-mlperf-v6-0-llama31-8b-offline-tokens", "Offline", "Llama-3.1-8B",
      3347.43, 4, "bench-mlperf-v6-0-intel-4x-bmg-b60", "v6.0"))
w(b60("b60-4x-mlperf-v6-0-gptoss-120b-offline-tokens", "Offline", "gpt-oss-120b",
      841.038, 4, "bench-mlperf-v6-0-intel-4x-bmg-b60", "v6.0",
      note="gpt-oss-120b (MXFP4 MoE) — ~210 tok/s per card aggregate; the affordable "
           "120B-class serving path."))
w(b60("prob60dual-4x-mlperf-v5-1-llama2-70b-offline-tokens", "Offline", "Llama-2-70B",
      3009.38, 4, "bench-mlperf-v5-1-intel-4x-bmg-pro-b60", "v5.1",
      note="4x 'BMG-Pro-B60-Dual' (dual-GPU card = 8 B60 dies) — per-DIE ~376 tok/s."))
w(b60("prob60dual-4x-mlperf-v5-1-llama31-8b-offline-tokens", "Offline", "Llama-3.1-8B",
      6472.37, 4, "bench-mlperf-v5-1-intel-4x-bmg-pro-b60", "v5.1",
      note="4x dual-GPU card = 8 B60 dies, ~809 tok/s per die."))
w(b60("prob60dual-4x-mlperf-v5-1-llama2-70b-server-tokens", "Server", "Llama-2-70B",
      2110.05, 4, "bench-mlperf-v5-1-intel-4x-bmg-pro-b60", "v5.1",
      note="Server scenario, TTFT ~161 ms min. 8 B60 dies, ~264 tok/s per die."))

# ---------- Intel Arc Pro B70 — MLPerf v6.0 ----------
def b70(id, scen, model, tps, note=""):
    return bench(id,
        f"Intel Arc Pro B70 32GB x4, MLPerf v6.0 {scen}: {model} aggregate tok/s",
        "vllm", ["intel-arc-pro-b70"], model + " (FP8)", None,
        "tps_aggregate", tps,
        f"output tok/s AGGREGATE across 4x Arc Pro B70 ({scen}; ~{tps/4:.0f} tok/s per card)",
        f"VENDOR-MEASURED (MLPerf v6.0 closed submission, Intel). 4x Arc Pro B70 32GB "
        f"GDDR6 (PCIe Gen5 x16) on Xeon 698X host, PyTorch vLLM, {model}, {scen}. "
        f"'Tokens per second' = {tps} verbatim from mlperf_log_summary.txt. AGGREGATE "
        f"across 4 cards — ~{tps/4:.0f}/card.",
        "vendor", ["bench-mlperf-v6-0-intel-4x-bmg-b70"],
        interconnect=["pcie-gen5"], notes=note)

w(b70("b70-4x-mlperf-v6-0-llama2-70b-offline-tokens", "Offline", "Llama-2-70B", 2459.18))
w(b70("b70-4x-mlperf-v6-0-llama2-70b-server-tokens", "Server", "Llama-2-70B", 1698.57))
w(b70("b70-4x-mlperf-v6-0-gptoss-120b-offline-tokens", "Offline", "gpt-oss-120b", 1536.9,
      note="gpt-oss-120b MoE on 4x B70 — ~384 tok/s per card aggregate."))
w(b70("b70-4x-mlperf-v6-0-llama31-8b-offline-tokens", "Offline", "Llama-3.1-8B", 4617.4))

# ---------- Intel Arc Pro B70 — llama.cpp Vulkan scoreboard ----------
w(bench("b70-vulkan-llama27b-q40-nofa-tg128",
        "Intel Arc Pro B70, llama.cpp Vulkan (b863507): Llama-2-7B Q4_0 tg128 t/s",
        "llama-cpp", ["intel-arc-pro-b70"], "Llama-2-7B Q4_0 GGUF", "gguf-q4-0",
        "decode_tok_s", 112.02,
        "tok/s per-request tg128 (no FA); ±1.08",
        "THIRD-PARTY MEASURED (llama.cpp discussion #10879 Vulkan scoreboard, commit "
        "b863507). Llama-2-7B Q4_0, -ngl 99 -fa 0, tg128 = 112.02 ± 1.08 t/s, pp512 = "
        "3379.00 ± 47.92. Per-request, batch-1 — the affordable single-user figure.",
        "third_party", ["llama-cpp-discussion-10879-vulkan-scoreboard"],
        notes="Vulkan backend on BMG — the single-user decode path (vs the vLLM "
              "batched-serving MLPerf numbers)."))

print("\nDone. Run: python tools/validate.py")
