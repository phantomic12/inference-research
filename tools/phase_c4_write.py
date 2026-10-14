#!/usr/bin/env python3
import json, os
ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-14"

def w(rec, subdir="benchmarks"):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP", rec["id"]); return
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth, measured_by, sources, status="verified", conf=0.8, repro=True, interconnect=None, notes=""):
    return {"id": id, "type": "benchmark", "name": name, "engine_id": engine,
            "accelerator_ids": accs, "interconnect_ids": interconnect or [],
            "model": model, "format_id": fmt, "metric": metric, "value": value,
            "unit": unit, "methodology": meth, "measured_by": measured_by,
            "reproducible": repro, "energy_basis": None, "power_w": None,
            "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
            "thermal_state": None, "notes": notes, "status": status,
            "confidence": conf, "updated": TODAY, "sources": sources}

def source(id, name, url, publisher, kind, notes, status="verified", conf=0.6):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url, "publisher": publisher,
            "kind": kind, "published": None, "accessed": TODAY, "archived_url": None,
            "notes": notes}

# =====================================================================
# SOURCES
# =====================================================================
def wsrc(rec):
    w(rec, subdir="sources")

wsrc(source("lhl-strix-halo-llm-bench-summary-table",
    "lhl/strix-halo-testing llm-bench summary tables — per-model pp512/tg128 on Framework Desktop (Ryzen AI Max+ 395, 128GB LPDDR5x-8000), best-backend-per-model",
    "https://github.com/lhl/strix-halo-testing/blob/main/llm-bench/README.md",
    "lhl (Framework pre-production unit)", "benchmark",
    "Summary tables 'PP Performance' and 'TG Performance' choose the best backend+flags per model. llama.cpp ~b5863, ROCm 7.0 nightly / Vulkan AMDVLK+RADV, -fa 1 --mmap 0 -ngl 99. pp512=prefill tok/s, tg128=decode tok/s, per-request batch-1. Max MiB resident column included.",
    conf=0.7))

wsrc(source("kyuz0-strix-halo-toolbox-results-json",
    "kyuz0 amd-strix-halo-toolboxes results.json — 554 llama-bench pp512/tg128 runs, 5 backend envs, builds 9187/9193, Ryzen AI Max+ 395 (gfx1151)",
    "https://kyuz0.github.io/amd-strix-halo-toolboxes/results.json",
    "kyuz0 / AMD Strix Halo toolboxes", "benchmark",
    "Machine-run results.json powering the interactive viewer. Each run: model_clean, env (rocm-7_2_3 / rocm6_4_4 / rocm7-nightlies / vulkan_amdvlk / vulkan_radv), fa=1, ngl=99, mmap=0, pp512 or tg128 mean+std t/s. Fedora 43, kernel 6.19.12. Per-request batch-1.",
    conf=0.7))

wsrc(source("xiongjiedai-runpod-llama-cpp-gpu-table",
    "XiongjieDai GPU-Benchmarks-on-LLM-Inference — llama.cpp cuBLAS on RunPod: LLaMA-3 8B/70B Q4_K_M + F16, generate + prompt-eval tok/s across NVIDIA + Apple parts",
    "https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference",
    "Xiongjie Dai (RunPod rental)", "repo",
    "llama.cpp built with cuBLAS (make LLAMA_CUBLAS=1, May-2024 era). ./main -ngl 10000 full offload; llama-bench -p/-n 512/1024/4096/8192. Ubuntu 22.04, CUDA 12.1.1, single-GPU rows are per-request batch-1; multi-GPU rows marked '* N'. Numbers verbatim from README tables.",
    conf=0.65))

wsrc(source("mlperf-v5-0-summary-results-json",
    "MLPerf Inference v5.0 summary_results.json — closed-division datacenter rows (NVIDIA H100-NVL-94GB, H100-PCIe-80GB, H200-NVL-141GB)",
    "https://raw.githubusercontent.com/mlcommons/inference_results_v5.0/main/summary_results.json",
    "MLCommons", "database",
    "Full closed-division summary. Performance_Result + Performance_Units = 'Tokens/s' are AGGREGATE output throughput across total_accelerators. Framework + weight_data_types fields present per row.",
    conf=0.8))

wsrc(source("mlperf-v6-0-summary-results-json",
    "MLPerf Inference v6.0 summary_results.json — closed-division datacenter + edge rows (Arc Pro B60/B70, RTX PRO 4500 Blackwell, MI325X, H200-NVL, GB10)",
    "https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/main/summary_results.json",
    "MLCommons", "database",
    "Closed-division summary. Same aggregate-output-tokens/s convention as v5.0. Includes Intel PyTorch+vLLM Arc Pro B-series submissions and HPE/Cisco TRT RTX PRO 4500 rows.",
    conf=0.8))

# =====================================================================
# STRIX HALO — lhl summary table (best-backend-per-model) + kyuz0 per-env
# Hardware: AMD Ryzen AI Max+ 395 Framework Desktop, Radeon 8060S iGPU,
# 128GB LPDDR5x-8000 UMA. accel ids: amd-ryzen-ai-max-395-framework-desktop
# (host APU) + amd-radeon-8060s-ryzen-ai-max (the gfx1151 iGPU).
# =====================================================================
SH_ACC = ["amd-ryzen-ai-max-395-framework-desktop", "amd-radeon-8060s-ryzen-ai-max"]
SH_SRC = ["lhl-strix-halo-llm-bench-summary-table"]

def sh(id_suffix, model_label, params, active, backend, flags, pp512, tg128, mem_mib, quant):
    meth = (f"THIRD-PARTY MEASURED (lhl/strix-halo-testing llm-bench summary table). "
            f"Hardware: Framework Desktop, AMD Ryzen AI Max+ 395 (Strix Halo) + Radeon 8060S iGPU, "
            f"128GB LPDDR5x-8000 unified memory. llama.cpp ~b5863, {backend} backend, flags {flags}; "
            f"llama-bench -fa 1 --mmap 0 -ngl 99, pp512 + tg128, per-request batch-1. "
            f"Peak resident {mem_mib} MiB. Numbers verbatim from the best-backend summary table.")
    if pp512 is not None:
        w(bench(f"strix-halo-8060s-{id_suffix}-pp512",
            f"Strix Halo 8060S ({backend}) {model_label} {quant}: pp512 prefill tok/s",
            "llama-cpp", SH_ACC, model_label, None, "prefill_tok_s", pp512,
            "tok/s (pp512, per-request batch-1)", meth, "third_party", SH_SRC, conf=0.75,
            interconnect=["pcie-gen4"]))
    if tg128 is not None:
        w(bench(f"strix-halo-8060s-{id_suffix}-tg128",
            f"Strix Halo 8060S ({backend}) {model_label} {quant}: tg128 decode tok/s",
            "llama-cpp", SH_ACC, model_label, None, "decode_tok_s", tg128,
            "tok/s (tg128, per-request batch-1)", meth, "third_party", SH_SRC, conf=0.75,
            interconnect=["pcie-gen4"]))

# PP-best rows (prefill table) — id suffix 'pp'
sh("llama2-7b-q4-0-amdvlk", "Llama 2 7B", 7, 7, "Vulkan AMDVLK", "fa=1", 1294.1, 47.7, 4281, "Q4_0")
sh("gpt-oss-20b-mxfp4-hip", "gpt-oss 20B", 21, 4, "HIP (ROCm)", "hipBLASLt", 1196.2, 46.9, 12958, "MXFP4")
sh("llama2-7b-q4km-hip-rocwmma", "Llama 2 7B", 7, 7, "HIP rocWMMA (ROCm)", "fa=1 hipBLASLt", 1083.0, 41.3, 4720, "Q4_K_M")
sh("shisa-v2-8b-q4km-hip", "Shisa V2 8B", 8, 8, "HIP (ROCm)", "hipBLASLt", 878.2, 37.2, 5308, "i1-Q4_K_M")
sh("qwen3-30b-a3b-udq4-hip", "Qwen 3 30B-A3B", 30, 3, "HIP rocWMMA (ROCm)", "fa=1 hipBLASLt", 669.4, 58.5, 17533, "UD-Q4_K_XL")
sh("gpt-oss-120b-mxfp4-amdvlk", "gpt-oss 120B", 117, 5, "Vulkan AMDVLK", "(default)", 430.6, 33.7, 64187, "MXFP4")
sh("mistral-small-31-udq4-hip", "Mistral Small 3.1 24B", 24, 24, "HIP rocWMMA (ROCm)", "fa=1 hipBLASLt", 368.5, 13.8, 14638, "UD-Q4_K_XL")
sh("gemma3-27b-udq4-hip", "Gemma 3 27B", 27, 27, "HIP (ROCm)", "hipBLASLt", 302.2, 10.7, 17542, "UD-Q4_K_XL")
sh("hunyuan-a13b-udq6-amdvlk", "Hunyuan-A13B", 80, 13, "Vulkan AMDVLK", "fa=1", 296.6, 18.1, 69179, "UD-Q6_K_XL")
sh("llama4-scout-udq4-hip", "Llama 4 Scout 17B-16E", 109, 17, "HIP (ROCm)", "hipBLASLt", 277.4, 17.6, 59720, "UD-Q4_K_XL")
sh("qwen3-32b-q80-hip", "Qwen 3 32B", 32, 32, "HIP (ROCm)", "hipBLASLt", 226.1, 6.4, 33683, "Q8_0")
sh("dots1-udq4-amdvlk", "dots1", 142, 14, "Vulkan AMDVLK", "fa=1", 182.0, 22.1, 84082, "UD-Q4_K_XL")
sh("glm45-air-udq4-amdvlk", "GLM 4.5 Air", 106, 12, "Vulkan AMDVLK", "fa=1", 178.5, 22.6, 68419, "UD-Q4_K_XL")
sh("qwen3-235b-udq3-hip", "Qwen 3 235B-A22B", 235, 22, "HIP (ROCm)", "hipBLASLt", 117.1, 12.9, 99950, "UD-Q3_K_XL")
sh("shisa-v2-70b-q4km-hip", "Shisa V2 70B", 70, 70, "HIP rocWMMA (ROCm)", "hipBLASLt", 94.7, 4.5, 41522, "i1-Q4_K_M")

# =====================================================================
# XIONGJIE DAI RunPod llama.cpp cuBLAS table — A100 PCIe, RTX 6000 Ada,
# L40S, RTX 4080, RTX 3090 Ti, RTX 4090 (Llama-3 8B/70B Q4_K_M + F16).
# Numbers verbatim from README; tg = 1024-token generate avg, pp = 1024-token
# prompt-eval avg. Single-GPU rows are per-request batch-1.
# =====================================================================
XD_SRC = ["xiongjiedai-runpod-llama-cpp-gpu-table"]
def xd(acc_id, acc_name, model_desc, fmt, tg1024, pp1024, extra=""):
    meth = (f"THIRD-PARTY MEASURED (XiongjieDai GPU-Benchmarks-on-LLM-Inference, RunPod rental). "
            f"llama.cpp built with cuBLAS (make LLAMA_CUBLAS=1), May-2024 snapshot. "
            f"{acc_name} full offload (-ngl 10000); llama-bench -p 512/1024/4096/8192 -n same. "
            f"Ubuntu 22.04, CUDA 12.1.1. Value is the 1024-token column. Per-request batch-1, NOT aggregate. {extra}")
    if tg1024 is not None:
        w(bench(f"xd-{acc_id}-{model_desc.split()[0].lower()}-{fmt.lower()}-tg1024",
            f"{acc_name} llama.cpp cuBLAS: LLaMA-3 {model_desc} {fmt} tg1024 decode tok/s",
            "llama-cpp", [acc_id], f"LLaMA-3 {model_desc}", None, "decode_tok_s", tg1024,
            "tok/s (tg1024 avg, per-request batch-1)", meth, "third_party", XD_SRC, conf=0.7))
    if pp1024 is not None:
        w(bench(f"xd-{acc_id}-{model_desc.split()[0].lower()}-{fmt.lower()}-pp1024",
            f"{acc_name} llama.cpp cuBLAS: LLaMA-3 {model_desc} {fmt} pp1024 prefill tok/s",
            "llama-cpp", [acc_id], f"LLaMA-3 {model_desc}", None, "prefill_tok_s", pp1024,
            "tok/s (pp1024 avg, per-request batch-1)", meth, "third_party", XD_SRC, conf=0.7))

# NVIDIA A100 PCIe 80GB (uncovered)
xd("nvidia-a100-80gb-pcie", "NVIDIA A100 PCIe 80GB", "8B", "Q4_K_M", 138.31, 5800.48)
xd("nvidia-a100-80gb-pcie", "NVIDIA A100 PCIe 80GB", "8B", "F16", 54.56, 7504.24)
xd("nvidia-a100-80gb-pcie", "NVIDIA A100 PCIe 80GB", "70B", "Q4_K_M", 22.11, 726.65)
# RTX 6000 Ada 48GB (uncovered)
xd("nvidia-rtx-6000-ada", "NVIDIA RTX 6000 Ada 48GB", "8B", "Q4_K_M", 130.99, 5560.94)
xd("nvidia-rtx-6000-ada", "NVIDIA RTX 6000 Ada 48GB", "8B", "F16", 51.97, 6205.44)
xd("nvidia-rtx-6000-ada", "NVIDIA RTX 6000 Ada 48GB", "70B", "Q4_K_M", 18.36, 547.03)
# L40S 48GB (uncovered)
xd("nvidia-l40s", "NVIDIA L40S 48GB", "8B", "Q4_K_M", 113.60, 5908.52)
xd("nvidia-l40s", "NVIDIA L40S 48GB", "8B", "F16", 43.42, 2491.65)
xd("nvidia-l40s", "NVIDIA L40S 48GB", "70B", "Q4_K_M", 15.31, 649.08)
# RTX 4080 16GB (uncovered)
xd("nvidia-rtx-4080", "NVIDIA RTX 4080 16GB", "8B", "Q4_K_M", 106.22, 5064.99)
xd("nvidia-rtx-4080", "NVIDIA RTX 4080 16GB", "8B", "F16", 40.29, 6758.90)
# RTX 3090 Ti 24GB (uncovered)
xd("nvidia-rtx-3090-ti", "NVIDIA RTX 3090 Ti 24GB", "8B", "Q4_K_M", 106.71, 3556.67)

# =====================================================================
# MLPerf v5.0 / v5.1 / v6.0 — aggregate output Tokens/s, CLOSED division.
# accelerator_ids are the single-card ids; total_accelerators noted in
# methodology. measured_by='vendor' (submitting OEM/vendor ran the harness).
# =====================================================================
M50 = ["mlperf-v5-0-summary-results-json"]
M60 = ["mlperf-v6-0-summary-results-json"]
def mlperf(id_suffix, acc_id, acc_name, model, scenario, value, n_accel, org, platform, fw, src, ver):
    meth = (f"VENDOR-SUBMITTED MLPerf Inference {ver} closed division, Datacenter suite, {scenario} scenario. "
            f"{org} {platform}: {n_accel}x {acc_name}. Framework {fw}. 'Tokens/s' = AGGREGATE output-token "
            f"throughput across the {n_accel} accelerators (NOT per-request). Value verbatim from summary_results.json. "
            f"min_query_count/min_duration per MLPerf rules; binding latency constraint applies to Server/Interactive.")
    w(bench(f"mlperf-{ver}-{id_suffix}-{model.replace('_','-').replace('.','-').lower()}-{scenario.lower()}",
        f"MLPerf {ver} {org} {n_accel}x {acc_name}: {model} {scenario} aggregate tok/s",
        "vllm" if "vLLM" in fw else "tensorrt-llm" if "TensorRT" in fw else "llama-cpp",
        [acc_id], model, None, "tps_aggregate", value,
        f"output tokens/s (aggregate over {n_accel} accelerators, {scenario})", meth,
        "vendor", src, conf=0.85, repro=True))

# H100-NVL-94GB — Cisco v5.0 (1x? platform Cisco UCS C240 M8 4x actually; use Fujitsu 8x)
mlperf("fujitsu-8xh100nvl-llama2-70b", "nvidia-h100-nvl", "NVIDIA H100-NVL 94GB", "llama2-70b-99.9", "Offline", 18084.9, 8, "Fujitsu", "8xH100-NVL-94GB", "TensorRT-LLM", M50, "v5.0")
mlperf("fujitsu-8xh100nvl-mixtral", "nvidia-h100-nvl", "NVIDIA H100-NVL 94GB", "mixtral-8x7b", "Offline", 37713.6, 8, "Fujitsu", "8xH100-NVL-94GB", "TensorRT-LLM", M50, "v5.0")
# H100-PCIe-80GB — Quanta v5.0
mlperf("qct-4xh100pcie-llama2-70b", "nvidia-h100-pcie", "NVIDIA H100-PCIe-80GB", "llama2-70b-99.9", "Offline", 7013.17, 4, "Quanta_Cloud_Technology", "4xH100-PCIe-80GB", "TensorRT-LLM", M50, "v5.0")
mlperf("qct-4xh100pcie-mixtral", "nvidia-h100-pcie", "NVIDIA H100-PCIe-80GB", "mixtral-8x7b", "Offline", 15843.3, 4, "Quanta_Cloud_Technology", "4xH100-PCIe-80GB", "TensorRT-LLM", M50, "v5.0")
# H200-NVL-141GB — Dell v6.0
mlperf("dell-8xh200nvl-llama2-70b", "nvidia-h200-nvl", "NVIDIA H200-NVL-141GB", "llama2-70b-99.9", "Offline", 32004.0, 8, "Dell", "8xH200-NVL-141GB", "TensorRT-LLM", M60, "v6.0")
mlperf("dell-8xh200nvl-llama31-8b", "nvidia-h200-nvl", "NVIDIA H200-NVL-141GB", "llama3.1-8b", "Offline", 53193.6, 8, "Dell", "8xH200-NVL-141GB", "TensorRT-LLM", M60, "v6.0")
mlperf("dell-8xh200nvl-llama2-70b-server", "nvidia-h200-nvl", "NVIDIA H200-NVL-141GB", "llama2-70b-99.9", "Server", 29084.9, 8, "Dell", "8xH200-NVL-141GB", "TensorRT-LLM", M60, "v6.0")
# MI325X — MangoBoost v6.0 (8x, ROCm 6.10.5)
mlperf("mangoboost-8xmi325x-llama2-70b", "amd-instinct-mi325x", "AMD Instinct MI325X 256GB HBM3e", "llama2-70b-99.9", "Offline", 33880.4, 8, "MangoBoost", "8xMI325X_2xEPYC_9655", "ROCm 6.10.5 (vLLM-class)", M60, "v6.0")
mlperf("mangoboost-8xmi325x-llama2-70b-server", "amd-instinct-mi325x", "AMD Instinct MI325X 256GB HBM3e", "llama2-70b-99.9", "Server", 31888.9, 8, "MangoBoost", "8xMI325X_2xEPYC_9655", "ROCm 6.10.5 (vLLM-class)", M60, "v6.0")
# Arc Pro B60 — Intel v6.0 (4x, PyTorch+vLLM)
mlperf("intel-4xarcb60-gptoss-120b", "intel-arc-pro-b60", "Intel Arc Pro B60", "gpt-oss-120b", "Offline", 841.038, 4, "Intel", "1-node-4x-BMG-B60", "PyTorch vLLM", M60, "v6.0")
mlperf("intel-4xarcb60-llama2-70b", "intel-arc-pro-b60", "Intel Arc Pro B60", "llama2-70b-99.9", "Offline", 1697.66, 4, "Intel", "1-node-4x-BMG-B60", "PyTorch vLLM", M60, "v6.0")
mlperf("intel-4xarcb60-llama31-8b", "intel-arc-pro-b60", "Intel Arc Pro B60", "llama3.1-8b", "Offline", 3347.43, 4, "Intel", "1-node-4x-BMG-B60", "PyTorch vLLM", M60, "v6.0")
# Arc Pro B70 — Intel v6.0 (4x)
mlperf("intel-4xarcb70-gptoss-120b", "intel-arc-pro-b70", "Intel Arc Pro B70", "gpt-oss-120b", "Offline", 1536.9, 4, "Intel", "1-node-4x-BMG-B70", "PyTorch vLLM", M60, "v6.0")
mlperf("intel-4xarcb70-llama2-70b", "intel-arc-pro-b70", "Intel Arc Pro B70", "llama2-70b-99.9", "Offline", 2459.18, 4, "Intel", "1-node-4x-BMG-B70", "PyTorch vLLM", M60, "v6.0")
mlperf("intel-4xarcb70-llama31-8b", "intel-arc-pro-b70", "Intel Arc Pro B70", "llama3.1-8b", "Offline", 4617.4, 4, "Intel", "1-node-4x-BMG-B70", "PyTorch vLLM", M60, "v6.0")
# RTX PRO 4500 Blackwell — HPE v6.0 (1x PCIe 32GB and 5x)
mlperf("hpe-1xrtxpro4500-llama31-8b", "nvidia-rtx-pro-4500-blackwell", "NVIDIA RTX PRO 4500 Blackwell 32GB", "llama3.1-8b", "Offline", 2340.75, 1, "HPE", "ProLiant DL145 1xRTX PRO 4500 PCIe-32GB", "TensorRT 10.11.0.33 CUDA 13.0", M60, "v6.0")
mlperf("hpe-5xrtxpro4500-llama31-8b", "nvidia-rtx-pro-4500-blackwell", "NVIDIA RTX PRO 4500 Blackwell 32GB", "llama3.1-8b", "Offline", 11665.0, 5, "HPE", "ProLiant DL380 5xRTX PRO 4500 PCIe-32GB", "TensorRT 10.11.0.33 CUDA 13.0", M60, "v6.0")

# =====================================================================
# KYUZ0 results.json — per-backend pp512/tg128 on Ryzen AI Max+ 395 / 8060S.
# Adds backend-split records for the repo's flagship 35B MoE target model
# and a few large-model rows, keyed to the two strix-halo accel ids.
# =====================================================================
KY_SRC = ["kyuz0-strix-halo-toolbox-results-json"]
def ky(id_suffix, model_label, env, backend_label, pp512, tg128, quant, params_b, gib):
    meth = (f"THIRD-PARTY MEASURED (kyuz0/amd-strix-halo-toolboxes results.json). "
            f"Ryzen AI Max+ 395 + Radeon 8060S (gfx1151), 128GB UMA, Fedora 43 kernel 6.19.12. "
            f"llama.cpp build 9187/9193, env '{env}' ({backend_label}), -fa 1 -ngl 99 --mmap 0, "
            f"pp512 + tg128 mean over repeats, per-request batch-1. "
            f"Model {model_label} ({params_b}B, {gib} GiB).")
    if pp512 is not None:
        w(bench(f"ky0-{id_suffix}-{env}-pp512".replace('_','-'),
            f"Strix Halo 8060S {backend_label} {model_label} {quant}: pp512 tok/s",
            "llama-cpp", SH_ACC, model_label, None, "prefill_tok_s", pp512,
            "tok/s (pp512, per-request batch-1)", meth, "third_party", KY_SRC, conf=0.7,
            interconnect=["pcie-gen4"]))
    if tg128 is not None:
        w(bench(f"ky0-{id_suffix}-{env}-tg128".replace('_','-'),
            f"Strix Halo 8060S {backend_label} {model_label} {quant}: tg128 tok/s",
            "llama-cpp", SH_ACC, model_label, None, "decode_tok_s", tg128,
            "tok/s (tg128, per-request batch-1)", meth, "third_party", KY_SRC, conf=0.7,
            interconnect=["pcie-gen4"]))

# Qwen3.6-35B-A3B — the user's target class. Best env per backend.
ky("qwen36-35b-a3b-udq4", "Qwen3.6-35B-A3B", "vulkan_radv", "Vulkan RADV", 1113.64, 60.43, "UD-Q4_K_XL", 35.51, 21.27)
ky("qwen36-35b-a3b-udq4", "Qwen3.6-35B-A3B", "rocm-7_2_3", "ROCm 7.2.3", 1120.0, 51.53, "UD-Q4_K_XL", 35.51, 21.27)
ky("qwen36-35b-a3b-udq8", "Qwen3.6-35B-A3B", "vulkan_radv", "Vulkan RADV", None, None, "UD-Q8_K_XL", 35.51, 0)  # placeholder guard
# gpt-oss-120b flagship MoE
ky("gpt-oss-120b-mxfp4", "gpt-oss-120b", "vulkan_radv", "Vulkan RADV", 719.91, 56.61, "MXFP4", 117, 0)
