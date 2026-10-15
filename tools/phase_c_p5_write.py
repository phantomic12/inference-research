#!/usr/bin/env python3
"""Phase C pass-5 writer: emit benchmark records assembled from mined sources."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = "2026-10-15"
written = []

def rec(id, name, engine, accel, model, fmt, metric, value, unit, meth, mb, repro, notes, srcs, inter=None, conf=0.7, status="verified"):
    d = dict(id=id, type="benchmark", name=name, engine_id=engine,
             accelerator_ids=accel, interconnect_ids=inter or [], model=model,
             format_id=fmt, metric=metric, value=value, unit=unit,
             methodology=meth, measured_by=mb, reproducible=repro,
             energy_basis=None, power_w=None, power_scope=None, power_cap=None,
             clock_lock_mhz=None, thermal_state=None,
             notes=notes, status=status, confidence=conf, updated=TODAY, sources=srcs)
    p = os.path.join(ROOT, "data/benchmarks", id + ".json")
    if os.path.exists(p):
        print("SKIP existing", id)
        return
    json.dump(d, open(p, "w"), indent=2)
    written.append(id)

XD_METH = ("THIRD-PARTY MEASURED (XiongjieDai GPU-Benchmarks-on-LLM-Inference, RunPod rental). "
           "llama.cpp built with cuBLAS (make LLAMA_CUBLAS=1), May-2024 snapshot. {gpu} full offload (-ngl 10000); "
           "llama-bench pp/tg at 512/1024/4096/8192; value is the 1024-token column. Ubuntu 22.04, CUDA 12.1.1. "
           "Per-request batch-1, NOT aggregate.")
XD_SRC = ["xiongjiedai-runpod-llama-cpp-gpu-table"]

def xd(id, gpu_name, aid, model, fmt, metric, value, extra=""):
    unit = "tok/s (tg1024 avg, per-request batch-1)" if metric == "decode_tok_s" else "tok/s (pp1024 avg, per-request batch-1)"
    nm = f"{gpu_name} llama.cpp cuBLAS: LLaMA-3 {model} {'tg1024 decode' if metric=='decode_tok_s' else 'pp1024 prefill'} tok/s"
    rec(id, nm, "llama-cpp", [aid], f"LLaMA-3 {model}", fmt, metric, value, unit,
        XD_METH.format(gpu=gpu_name), "third_party", True, extra, XD_SRC)

# ---- XiongjieDai single-GPU rows for parts newly added / previously uncovered ----
xd("xd-rtx-a6000-8b-q4km-tg1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 102.22)
xd("xd-rtx-a6000-8b-q4km-pp1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 3621.81)
xd("xd-rtx-a6000-8b-f16-tg1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "8B F16", None, "decode_tok_s", 40.25)
xd("xd-rtx-a6000-8b-f16-pp1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "8B F16", None, "prefill_tok_s", 4315.18)
xd("xd-rtx-a6000-70b-q4km-tg1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "70B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 14.58,
   extra="70B Q4_K_M (~40GB weights) fits in the 48 GB framebuffer - this is the single-GPU 70B decode row.")
xd("xd-rtx-a6000-70b-q4km-pp1024", "NVIDIA RTX A6000 48GB", "nvidia-rtx-a6000", "70B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 466.82)

xd("xd-nvidia-a40-8b-q4km-tg1024", "NVIDIA A40 48GB", "nvidia-a40", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 88.95)
xd("xd-nvidia-a40-8b-q4km-pp1024", "NVIDIA A40 48GB", "nvidia-a40", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 3240.95)
xd("xd-nvidia-a40-8b-f16-tg1024", "NVIDIA A40 48GB", "nvidia-a40", "8B F16", None, "decode_tok_s", 33.95)
xd("xd-nvidia-a40-70b-q4km-tg1024", "NVIDIA A40 48GB", "nvidia-a40", "70B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 12.08)
xd("xd-nvidia-a40-70b-q4km-pp1024", "NVIDIA A40 48GB", "nvidia-a40", "70B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 239.92)

xd("xd-rtx-4000-ada-8b-q4km-tg1024", "NVIDIA RTX 4000 Ada 20GB", "nvidia-rtx-4000-ada", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 58.59)
xd("xd-rtx-4000-ada-8b-q4km-pp1024", "NVIDIA RTX 4000 Ada 20GB", "nvidia-rtx-4000-ada", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 2310.53)
xd("xd-rtx-4000-ada-8b-f16-tg1024", "NVIDIA RTX 4000 Ada 20GB", "nvidia-rtx-4000-ada", "8B F16", None, "decode_tok_s", 20.85)
xd("xd-rtx-4000-ada-8b-f16-pp1024", "NVIDIA RTX 4000 Ada 20GB", "nvidia-rtx-4000-ada", "8B F16", None, "prefill_tok_s", 2951.87)

xd("xd-rtx-5000-ada-8b-q4km-tg1024", "NVIDIA RTX 5000 Ada 32GB", "nvidia-rtx-5000-ada", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 89.87)
xd("xd-rtx-5000-ada-8b-q4km-pp1024", "NVIDIA RTX 5000 Ada 32GB", "nvidia-rtx-5000-ada", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 4467.46)
xd("xd-rtx-5000-ada-8b-f16-tg1024", "NVIDIA RTX 5000 Ada 32GB", "nvidia-rtx-5000-ada", "8B F16", None, "decode_tok_s", 32.67)
xd("xd-rtx-5000-ada-8b-f16-pp1024", "NVIDIA RTX 5000 Ada 32GB", "nvidia-rtx-5000-ada", "8B F16", None, "prefill_tok_s", 5835.41)

xd("xd-rtx-3080-ti-8b-q4km-tg1024", "NVIDIA RTX 3080 Ti 12GB", "nvidia-rtx-3080-ti", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 106.71)
xd("xd-rtx-3080-ti-8b-q4km-pp1024", "NVIDIA RTX 3080 Ti 12GB", "nvidia-rtx-3080-ti", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 3556.67)
xd("xd-rtx-3080-10gb-8b-q4km-tg1024", "NVIDIA RTX 3080 10GB", "nvidia-rtx-3080-10gb", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 106.40)
xd("xd-rtx-3080-10gb-8b-q4km-pp1024", "NVIDIA RTX 3080 10GB", "nvidia-rtx-3080-10gb", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 3557.02)
xd("xd-rtx-3070-8b-q4km-tg1024", "NVIDIA RTX 3070 8GB", "nvidia-rtx-3070", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 70.94)
xd("xd-rtx-3070-8b-q4km-pp1024", "NVIDIA RTX 3070 8GB", "nvidia-rtx-3070", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 2283.62)
xd("xd-rtx-4070-ti-8b-q4km-tg1024", "NVIDIA RTX 4070 Ti 12GB", "nvidia-rtx-4070-ti", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 82.21)
xd("xd-rtx-4070-ti-8b-q4km-pp1024", "NVIDIA RTX 4070 Ti 12GB", "nvidia-rtx-4070-ti", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 3653.07)

xd("xd-apple-m1-max-8b-q4km-tg1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "8B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 34.49)
xd("xd-apple-m1-max-8b-q4km-pp1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "8B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 355.45)
xd("xd-apple-m1-max-8b-f16-tg1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "8B F16", None, "decode_tok_s", 18.43)
xd("xd-apple-m1-max-8b-f16-pp1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "8B F16", None, "prefill_tok_s", 418.77)
xd("xd-apple-m1-max-70b-q4km-tg1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "70B Q4_K_M", "gguf-q4-k-m", "decode_tok_s", 4.09,
   extra="70B Q4_K_M fits in 64 GB unified memory (Metal working-set ~75% cap noted in source).")
xd("xd-apple-m1-max-70b-q4km-pp1024", "Apple M1 Max 32-core GPU 64GB", "apple-m1-max", "70B Q4_K_M", "gguf-q4-k-m", "prefill_tok_s", 33.01)

# ---- Hardware Corner GPU ranking, 16K context decode (Q4_K_XL) ----
HC_METH = ("THIRD-PARTY MEASURED (hardware-corner.net Definitive GPU Ranking for LLMs, accessed 2026-10-15). "
           "llama.cpp llama-bench CUDA 12.8, Ubuntu 24.04, Q4_K_XL GGUF quants, decode tok/s per GPU per model at ~16K context. "
           "Per-request batch-1, NOT aggregate. Value is the 16K-context column.")
HC_SRC = ["hardware-corner-gpu-ranking-local-llm"]
def hc16(id, gpu_name, aid, model, fmt, value, extra=""):
    rec(id, f"Hardware Corner: {gpu_name}, llama.cpp CUDA, {model} Q4_K_XL GGUF, decode {value} t/s @16k ctx",
        "llama-cpp", [aid], f"{model} Q4_K XL GGUF", fmt, "decode_tok_s", value,
        "tok/s per-request decode (batch 1, ~16384-token context)", HC_METH, "third_party", True,
        extra or "Single-card row at 16K context; same table publishes 32K/65K/131K context columns.", HC_SRC,
        inter=["pcie-gen4"])

hc16("hwcorner-rtx-5060-ti-qwen3-8b-q4kxl-decode-16k", "RTX 5060 Ti 16GB", "nvidia-rtx-5060-ti", "Qwen3 8B", "gguf-q4-k-m", 51.41,
     extra="HC names this card 'best price-to-performance' pick in the 16GB tier.")
hc16("hwcorner-rtx-5060-ti-qwen3-14b-q4kxl-decode-16k", "RTX 5060 Ti 16GB", "nvidia-rtx-5060-ti", "Qwen3 14B", "gguf-q4-k-m", 32.91)
hc16("hwcorner-rtx-5060-ti-gptoss-20b-q4kxl-decode-16k", "RTX 5060 Ti 16GB", "nvidia-rtx-5060-ti", "gpt-oss 20B", "gguf-q4-k-m", 82.42)
hc16("hwcorner-rtx-3080-ti-qwen3-8b-q4kxl-decode-16k", "RTX 3080 Ti 12GB", "nvidia-rtx-3080-ti", "Qwen3 8B", "gguf-q4-k-m", 87.94)
hc16("hwcorner-rtx-3080-ti-qwen3-14b-q4kxl-decode-16k", "RTX 3080 Ti 12GB", "nvidia-rtx-3080-ti", "Qwen3 14B", "gguf-q4-k-m", 52.29)
hc16("hwcorner-rtx-3080-10gb-qwen3-8b-q4kxl-decode-16k", "RTX 3080 10GB", "nvidia-rtx-3080-10gb", "Qwen3 8B", "gguf-q4-k-m", 74.24)
hc16("hwcorner-rtx-4070-ti-super-qwen3-8b-q4kxl-decode-16k", "RTX 4070 Ti SUPER 16GB", "nvidia-rtx-4070-ti-super", "Qwen3 8B", "gguf-q4-k-m", 72.20)
hc16("hwcorner-rtx-4070-ti-super-qwen3-14b-q4kxl-decode-16k", "RTX 4070 Ti SUPER 16GB", "nvidia-rtx-4070-ti-super", "Qwen3 14B", "gguf-q4-k-m", 47.18)
hc16("hwcorner-rtx-4070-ti-super-gptoss-20b-q4kxl-decode-16k", "RTX 4070 Ti SUPER 16GB", "nvidia-rtx-4070-ti-super", "gpt-oss 20B", "gguf-q4-k-m", 113.60)
hc16("hwcorner-rtx-4070-ti-qwen3-8b-q4kxl-decode-16k", "RTX 4070 Ti 12GB", "nvidia-rtx-4070-ti", "Qwen3 8B", "gguf-q4-k-m", 57.55)
hc16("hwcorner-rtx-4070-ti-qwen3-14b-q4kxl-decode-16k", "RTX 4070 Ti 12GB", "nvidia-rtx-4070-ti", "Qwen3 14B", "gguf-q4-k-m", 37.75)
hc16("hwcorner-rtx-4070-super-qwen3-8b-q4kxl-decode-16k", "RTX 4070 SUPER 12GB", "nvidia-rtx-4070-super", "Qwen3 8B", "gguf-q4-k-m", 56.21)
hc16("hwcorner-rtx-4070-super-qwen3-14b-q4kxl-decode-16k", "RTX 4070 SUPER 12GB", "nvidia-rtx-4070-super", "Qwen3 14B", "gguf-q4-k-m", 37.15)
hc16("hwcorner-rtx-4070-qwen3-8b-q4kxl-decode-16k", "RTX 4070 12GB", "nvidia-rtx-4070", "Qwen3 8B", "gguf-q4-k-m", 52.07)
hc16("hwcorner-rtx-4070-qwen3-14b-q4kxl-decode-16k", "RTX 4070 12GB", "nvidia-rtx-4070", "Qwen3 14B", "gguf-q4-k-m", 32.66)
hc16("hwcorner-rtx-4080-super-qwen3-8b-q4kxl-decode-16k", "RTX 4080 SUPER 16GB", "nvidia-rtx-4080-super", "Qwen3 8B", "gguf-q4-k-m", 79.36)
hc16("hwcorner-rtx-4080-super-qwen3-14b-q4kxl-decode-16k", "RTX 4080 SUPER 16GB", "nvidia-rtx-4080-super", "Qwen3 14B", "gguf-q4-k-m", 52.76)
hc16("hwcorner-rtx-4080-super-gptoss-20b-q4kxl-decode-16k", "RTX 4080 SUPER 16GB", "nvidia-rtx-4080-super", "gpt-oss 20B", "gguf-q4-k-m", 122.96)
hc16("hwcorner-rtx-4060-ti-qwen3-8b-q4kxl-decode-16k", "RTX 4060 Ti 16GB", "nvidia-rtx-4060-ti", "Qwen3 8B", "gguf-q4-k-m", 34.31,
     extra="128-bit bus limits decode despite 16 GB capacity - capacity-rich, bandwidth-poor.")
hc16("hwcorner-rtx-4060-ti-qwen3-14b-q4kxl-decode-16k", "RTX 4060 Ti 16GB", "nvidia-rtx-4060-ti", "Qwen3 14B", "gguf-q4-k-m", 22.36)
hc16("hwcorner-rtx-4060-ti-gptoss-20b-q4kxl-decode-16k", "RTX 4060 Ti 16GB", "nvidia-rtx-4060-ti", "gpt-oss 20B", "gguf-q4-k-m", 57.76)
hc16("hwcorner-rtx-3060-qwen3-8b-q4kxl-decode-16k", "RTX 3060 12GB", "nvidia-rtx-3060", "Qwen3 8B", "gguf-q4-k-m", 41.97)
hc16("hwcorner-rtx-3060-qwen3-14b-q4kxl-decode-16k", "RTX 3060 12GB", "nvidia-rtx-3060", "Qwen3 14B", "gguf-q4-k-m", 22.66)
hc16("hwcorner-rtx-a6000-qwen3-8b-q4kxl-decode-16k", "RTX A6000 48GB", "nvidia-rtx-a6000", "Qwen3 8B", "gguf-q4-k-m", 64.26)
hc16("hwcorner-rtx-a6000-qwen3-14b-q4kxl-decode-16k", "RTX A6000 48GB", "nvidia-rtx-a6000", "Qwen3 14B", "gguf-q4-k-m", 40.69)
hc16("hwcorner-rtx-a6000-qwen3-30b-a3b-q4kxl-decode-16k", "RTX A6000 48GB", "nvidia-rtx-a6000", "Qwen3 30B-A3B", "gguf-q4-k-m", 75.35)
hc16("hwcorner-rtx-a6000-qwen3-32b-q4kxl-decode-16k", "RTX A6000 48GB", "nvidia-rtx-a6000", "Qwen3 32B", "gguf-q4-k-m", 18.32)
hc16("hwcorner-rtx-a6000-gptoss-20b-q4kxl-decode-16k", "RTX A6000 48GB", "nvidia-rtx-a6000", "gpt-oss 20B", "gguf-q4-k-m", 91.19)

# ---- Hardware Corner Qwen3.8-27B cross-hardware article ----
HC38_METH = ("THIRD-PARTY MEASURED (hardware-corner.net 'We Tested Qwen3.8 27B', accessed 2026-10-15). llama.cpp build "
             "153d324bc (10364), Ubuntu 24.04, NVIDIA driver 595.58.03, CUDA 12.8, Qwen3.8-27B Q4_K Small GGUF (16.68 GiB, "
             "27.32B params; llama.cpp reports arch 'qwen35'), Flash Attention on, MTP off, full GPU offload. {gpu}. "
             "pp = prompt-processing tok/s, tg = token-generation tok/s at stated prefilled context. Per-request batch-1.")
HC38_SRC = ["hardware-corner-qwen38-27b-hardware-tests"]
def hc38(id, gpu_desc, aid, metric, value, ctx, extra=""):
    rec(id, f"Hardware Corner Qwen3.8-27B Q4_K_S: {gpu_desc} {'tg' if metric=='decode_tok_s' else 'pp'} {value} t/s @{ctx} ctx",
        "llama-cpp", [aid], "Qwen3.8 27B Q4_K Small GGUF", "gguf-q4-k-s", metric, value,
        f"tok/s per-request {'decode' if metric=='decode_tok_s' else 'prefill'} (batch 1, {ctx} context)",
        HC38_METH.format(gpu=gpu_desc), "third_party", True, extra, HC38_SRC, inter=["pcie-gen4"])

hc38("hc38-rtx-3090-qwen38-27b-tg-4k", "RTX 3090 24GB", "nvidia-rtx-3090", "decode_tok_s", 40.31, "4k")
hc38("hc38-rtx-3090-qwen38-27b-pp-4k", "RTX 3090 24GB", "nvidia-rtx-3090", "prefill_tok_s", 1308.05, "4k")
hc38("hc38-rtx-3090-qwen38-27b-tg-64k", "RTX 3090 24GB", "nvidia-rtx-3090", "decode_tok_s", 33.95, "64k",
     extra="64k is the practical context ceiling on 24 GB (measured 22 GB VRAM).")
hc38("hc38-rtx-4090-qwen38-27b-tg-4k", "RTX 4090 24GB", "nvidia-rtx-4090", "decode_tok_s", 46.16, "4k")
hc38("hc38-rtx-4090-qwen38-27b-pp-4k", "RTX 4090 24GB", "nvidia-rtx-4090", "prefill_tok_s", 2962.59, "4k")
hc38("hc38-rtx-5090-qwen38-27b-tg-4k", "RTX 5090 32GB", "nvidia-rtx-5090", "decode_tok_s", 74.83, "4k")
hc38("hc38-rtx-5090-qwen38-27b-pp-4k", "RTX 5090 32GB", "nvidia-rtx-5090", "prefill_tok_s", 3749.51, "4k")
hc38("hc38-rtx-5090-qwen38-27b-tg-128k", "RTX 5090 32GB", "nvidia-rtx-5090", "decode_tok_s", 22.79, "128k",
     extra="128k fits in 32 GB; generation rate halves vs 4k.")
hc38("hc38-dual-rtx-5060-ti-qwen38-27b-tg-4k", "dual RTX 5060 Ti 16GB (2x15849 MiB)", "nvidia-rtx-5060-ti", "decode_tok_s", 22.47, "4k",
     extra="TWO-card run, tensor-split across 2x 5060 Ti; NOT single-card. Marked in methodology.")
hc38("hc38-dual-rtx-5060-ti-qwen38-27b-pp-4k", "dual RTX 5060 Ti 16GB (2x15849 MiB)", "nvidia-rtx-5060-ti", "prefill_tok_s", 1167.70, "4k",
     extra="TWO-card run, tensor-split across 2x 5060 Ti; NOT single-card.")
hc38("hc38-dual-rtx-5060-ti-qwen38-27b-tg-128k", "dual RTX 5060 Ti 16GB (2x15849 MiB)", "nvidia-rtx-5060-ti", "decode_tok_s", 15.55, "128k",
     extra="TWO-card run reaching 128k on 32 GB aggregate VRAM.")
hc38("hc38-apple-m5-max-qwen38-27b-tg-4k", "Apple M5 Max 128GB unified", "local-apple-m5-max", "decode_tok_s", 31.4, "4k")
hc38("hc38-apple-m5-max-qwen38-27b-pp-4k", "Apple M5 Max 128GB unified", "local-apple-m5-max", "prefill_tok_s", 834.1, "4k")
hc38("hc38-apple-m5-max-qwen38-27b-tg-64k", "Apple M5 Max 128GB unified", "local-apple-m5-max", "decode_tok_s", 18.2, "64k")
hc38("hc38-nvidia-gb10-qwen38-27b-tg-4k", "NVIDIA GB10 (Dell Pro Max / DGX Spark) 128GB unified", "nvidia-gb10", "decode_tok_s", 12.20, "4k",
     extra="GB10 run used CUDA 13.2 + driver 595.71.05 per source.")
hc38("hc38-nvidia-gb10-qwen38-27b-pp-4k", "NVIDIA GB10 (Dell Pro Max / DGX Spark) 128GB unified", "nvidia-gb10", "prefill_tok_s", 812.60, "4k")
hc38("hc38-nvidia-gb10-qwen38-27b-tg-256k", "NVIDIA GB10 (Dell Pro Max / DGX Spark) 128GB unified", "nvidia-gb10", "decode_tok_s", 5.94, "256k",
     extra="256k context - only the 128 GB unified part in this test could hold it.")

# ---- lhl strix-vs-spark (DGX Spark + Strix Halo, gpt-oss-120b llama.cpp b6792) ----
# NOTE: the four Strix Halo tg32 rows were already ingested in an earlier pass
# (lhl-strix-8060s-*-tg32-*, tracked). This pass adds the DGX Spark column
# (newly-created nvidia-gb10 accelerator) plus the missing pp2048 /
# tuned-rocWMMA rows the earlier pass skipped.
LHL_METH = ("THIRD-PARTY MEASURED (lhl/strix-halo-testing README). llama.cpp build 6792; gpt-oss-120b (ggml-org MXFP4/Q8); "
            "pp2048/tg32 vs depth 0-32768. Spark numbers from ggerganov's llama.cpp discussion #16578 sweep (b6767); "
            "Strix Halo numbers lhl on Framework Desktop, TheRock/ROCm 7.10.0a20251017. Per-request batch-1.")
LHL_SRC = ["lhl-strix-halo-testing-strix-vs-spark-readme"]
def lhl(id, gpu_desc, aid, backend, metric, value, ctx, extra="", acc=None):
    if acc is not None:
        aid = acc
    rec(id, f"lhl/ggerganov: {gpu_desc} {backend} gpt-oss-120b {'tg32' if metric=='decode_tok_s' else 'pp2048'} {value} t/s @{ctx}",
        "llama-cpp", [aid], "gpt-oss-120b MXFP4/Q8 GGUF", "gguf-mxfp4-moe", metric, value,
        f"tok/s per-request {'decode' if metric=='decode_tok_s' else 'prefill'} (batch 1, depth {ctx})",
        LHL_METH + " " + gpu_desc, "third_party", True, extra, LHL_SRC)

# DGX Spark rows (CUDA is the Spark backend ggerganov ran)
lhl("lhl-dgx-spark-gpt-oss-120b-cuda-tg32-d0", "NVIDIA DGX Spark GB10 CUDA", "nvidia-gb10", "CUDA", "decode_tok_s", 52.87, "0")
lhl("lhl-dgx-spark-gpt-oss-120b-cuda-pp2048-d0", "NVIDIA DGX Spark GB10 CUDA", "nvidia-gb10", "CUDA", "prefill_tok_s", 1689.47, "0")
lhl("lhl-dgx-spark-gpt-oss-120b-cuda-tg32-d32k", "NVIDIA DGX Spark GB10 CUDA", "nvidia-gb10", "CUDA", "decode_tok_s", 38.76, "32768",
    extra="Spark tg degrades far less with depth than either STXH backend (lhl's own conclusion).")
lhl("lhl-dgx-spark-gpt-oss-120b-cuda-pp2048-d32k", "NVIDIA DGX Spark GB10 CUDA", "nvidia-gb10", "CUDA", "prefill_tok_s", 1221.23, "32768")
# Strix Halo rows: Vulkan AMDVLK prefill (tg already tracked from earlier pass)
lhl("lhl-strix-8060s-gpt-oss-120b-vulkan-amdvlk-pp2048-d0", "Strix Halo 8060S Vulkan AMDVLK (-ub512)", "amd-radeon-8060s-ryzen-ai-max", "Vulkan-AMDVLK", "prefill_tok_s", 729.10, "0")
# Strix Halo ROCm w/ tuned rocWMMA (best variant; stock rocm tg already tracked)
lhl("lhl-strix-8060s-gpt-oss-120b-rocm-tuned-pp2048-d0", "Strix Halo 8060S ROCm tuned-rocWMMA", "amd-radeon-8060s-ryzen-ai-max", "ROCm-rocWMMA-tuned", "prefill_tok_s", 977.22, "0")
lhl("lhl-strix-8060s-gpt-oss-120b-vulkan-amdvlk-pp2048-d32k", "Strix Halo 8060S Vulkan AMDVLK (-ub512)", "amd-radeon-8060s-ryzen-ai-max", "Vulkan-AMDVLK", "prefill_tok_s", 137.08, "32768",
    extra="pp collapses at depth on Vulkan (-790.9% vs Spark per lhl's table); the long-context prefill wall is the Strix Halo weakness, not tg.")
lhl("lhl-strix-8060s-gpt-oss-120b-rocm-wmma-pp2048-d0", "Strix Halo 8060S ROCm stock rocWMMA (-ub2048)", "amd-radeon-8060s-ryzen-ai-max", "ROCm-rocWMMA", "prefill_tok_s", 1006.65, "0")

# ---- Level1Techs B70 x4 vs 3090 x4 (vLLM, 50 concurrent) ----
rec("l1t-4x-arc-pro-b70-qwen35-27b-vllm-50req-tps",
    "Level1Techs: 4x Intel Arc Pro B70, vLLM, Qwen3.5-27B, 50 concurrent reqs ~1024 tok ctx: ~369 tok/s aggregate output",
    "vllm", ["intel-arc-pro-b70"], "Qwen3.5-27B", None, "tps_aggregate", 369.0,
    "tok/s output, aggregate over 50 concurrent requests",
    "FORUM MEASUREMENT (Level1Techs, relayed by hardware-corner.net B70 launch article). 4x Arc Pro B70 vLLM, Qwen3.5-27B, "
    "50 concurrent requests, ~1024-token context: sustained ~369 tok/s output (peak ~550), avg TTFT ~11.4 s. "
    "AGGREGATE across 50 requests, not per-request.",
    "third_party", False,
    "Companion record l1t-4x-rtx-3090-qwen35-27b-vllm-50req-tps gives the same-test 4x3090 baseline (348 tok/s, TTFT 18.7s).",
    ["level1techs-intel-b70-vs-rtx3090-vllm-50req"], inter=["pcie-gen5"], conf=0.5, status="draft")
rec("l1t-4x-rtx-3090-qwen35-27b-vllm-50req-tps",
    "Level1Techs: 4x RTX 3090, vLLM, Qwen3.5-27B, 50 concurrent reqs ~1024 tok ctx: ~348 tok/s aggregate output",
    "vllm", ["nvidia-rtx-3090"], "Qwen3.5-27B", None, "tps_aggregate", 348.0,
    "tok/s output, aggregate over 50 concurrent requests",
    "FORUM MEASUREMENT (Level1Techs, relayed by hardware-corner.net B70 launch article). 4x RTX 3090 vLLM, Qwen3.5-27B, "
    "50 concurrent requests, ~1024-token context: sustained ~348 tok/s output, avg TTFT ~18.7 s. "
    "AGGREGATE across 50 requests, not per-request.",
    "third_party", False,
    "Same-test baseline for the 4x B70 row: B70 edges throughput and roughly halves TTFT at this concurrency.",
    ["level1techs-intel-b70-vs-rtx3090-vllm-50req"], inter=["pcie-gen4"], conf=0.5, status="draft")
rec("l1t-4x-arc-pro-b70-qwen35-27b-vllm-50req-ttft",
    "Level1Techs: 4x Intel Arc Pro B70, vLLM, Qwen3.5-27B, 50 concurrent: avg TTFT ~11.4 s",
    "vllm", ["intel-arc-pro-b70"], "Qwen3.5-27B", None, "ttft_ms", 11400.0,
    "ms mean time-to-first-token under 50-request load",
    "FORUM MEASUREMENT (Level1Techs via hardware-corner). 4x B70 vLLM Qwen3.5-27B 50 concurrent ~1024 tok: avg TTFT ~11.4 s "
    "(vs ~18.7 s on 4x RTX 3090 in the same test).",
    "third_party", False, "TTFT under saturation, not single-request.", 
    ["level1techs-intel-b70-vs-rtx3090-vllm-50req"], inter=["pcie-gen5"], conf=0.5, status="draft")

# ---- MLPerf v6.1 datacenter closed rows (new submitters/scenarios not yet in repo) ----
MP = "mlperf-inference-v6-1-summary-xlsx"
def mp(id, name, accel, model, metric, value, unit, sysdesc, sw, mb="vendor", conf=0.85, extra=""):
    swl = sw.lower()
    if "vllm" in swl:
        eng = "vllm"
    elif "tensorrt" in swl:
        eng = "tensorrt-llm"
    elif "llama.cpp" in swl:
        eng = "llama-cpp"
    elif "onnx" in swl:
        eng = "local-onnxruntime-mobile"
    else:
        eng = "vllm"  # framework named in methodology; engine id pinned to closest existing record
    rec(id, name, eng, accel, model, None, metric, value, unit,
        f"MLPerf Inference v6.1 {unit.split()[0] if False else ''}published result ({sysdesc}; software: {sw}). "
        "System-AGGREGATE output tokens/s for the named scenario, submitter-measured under MLCommons audit. " + extra,
        mb, True, "ENGINE_ID NOTE: pinned to '" + eng + "' as the closest existing engine record; the submission's actual stack is named in this methodology string. " , [MP], conf=conf)

# Arc Pro B70 4x — three v6.1 submitters, new scenarios
mp("mlperf-v6-1-intel-4xarcb70-llama2-70b-server-tokens",
   "MLPerf v6.1 Intel 1-node-4x-BMG-B70: llama2-70b-99.9 Server 1745.48 tok/s",
   ["intel-arc-pro-b70"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 1745.48,
   "tok/s aggregate Server", "Intel 1-node-4x-BMG-B70, 4x Arc Pro B70 32GB, Xeon host", "PyTorch")
mp("mlperf-v6-1-intel-4xarcb70-llama31-8b-server-tokens",
   "MLPerf v6.1 Intel 1-node-4x-BMG-B70: llama3.1-8b Server 4173.25 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4173.25,
   "tok/s aggregate Server", "Intel 1-node-4x-BMG-B70, 4x Arc Pro B70 32GB", "PyTorch")
mp("mlperf-v6-1-intel-4xarcb70-llama31-8b-offline-tokens",
   "MLPerf v6.1 Intel 1-node-4x-BMG-B70: llama3.1-8b Offline 4613.28 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4613.28,
   "tok/s aggregate Offline", "Intel 1-node-4x-BMG-B70, 4x Arc Pro B70 32GB", "PyTorch")
mp("mlperf-v6-1-intel-4xarcb70-gptoss-120b-offline-tokens",
   "MLPerf v6.1 Intel 1-node-4x-BMG-B70: gpt-oss-120b Offline 1956.4 tok/s",
   ["intel-arc-pro-b70"], "gpt-oss-120b", "tps_aggregate", 1956.4,
   "tok/s aggregate Offline", "Intel 1-node-4x-BMG-B70, 4x Arc Pro B70 32GB", "PyTorch")
mp("mlperf-v6-1-intel-4xarcb70-gptoss-120b-server-tokens",
   "MLPerf v6.1 Intel 1-node-4x-BMG-B70: gpt-oss-120b Server 1296.87 tok/s",
   ["intel-arc-pro-b70"], "gpt-oss-120b", "tps_aggregate", 1296.87,
   "tok/s aggregate Server", "Intel 1-node-4x-BMG-B70, 4x Arc Pro B70 32GB", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-llama2-70b-offline-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: llama2-70b-99.9 Offline 2484.14 tok/s",
   ["intel-arc-pro-b70"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 2484.14,
   "tok/s aggregate Offline", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-llama2-70b-server-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: llama2-70b-99.9 Server 1759.95 tok/s",
   ["intel-arc-pro-b70"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 1759.95,
   "tok/s aggregate Server", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-gptoss-120b-offline-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: gpt-oss-120b Offline 2063.93 tok/s",
   ["intel-arc-pro-b70"], "gpt-oss-120b", "tps_aggregate", 2063.93,
   "tok/s aggregate Offline", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-gptoss-120b-server-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: gpt-oss-120b Server 1351.13 tok/s",
   ["intel-arc-pro-b70"], "gpt-oss-120b", "tps_aggregate", 1351.13,
   "tok/s aggregate Server", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-llama31-8b-offline-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: llama3.1-8b Offline 4596.62 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4596.62,
   "tok/s aggregate Offline", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-supermicro-4xarcb70-llama31-8b-server-tokens",
   "MLPerf v6.1 Supermicro 1-node-4x-BMG-B70: llama3.1-8b Server 4208.73 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4208.73,
   "tok/s aggregate Server", "Supermicro 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")
mp("mlperf-v6-1-quanta-4xarcb70-llama31-8b-offline-tokens",
   "MLPerf v6.1 Quanta 1-node-4x-BMG-B70: llama3.1-8b Offline 4977.67 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4977.67,
   "tok/s aggregate Offline", "Quanta 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch",
   extra="Highest of the three v6.1 4xB70 8B Offline submissions (Intel 4613.28, Supermicro 4596.62).")
mp("mlperf-v6-1-quanta-4xarcb70-llama31-8b-server-tokens",
   "MLPerf v6.1 Quanta 1-node-4x-BMG-B70: llama3.1-8b Server 4122.73 tok/s",
   ["intel-arc-pro-b70"], "Llama-3.1-8B", "tps_aggregate", 4122.73,
   "tok/s aggregate Server", "Quanta 1-node-4x-BMG-B70, 4x Arc Pro B70", "PyTorch")

# RTX PRO 4500 v6.1 new rows (Offline already recorded for cisco-5x and dell-8x)
mp("mlperf-v6-1-cisco-5xrtxpro4500-llama31-8b-server-tokens",
   "MLPerf v6.1 Cisco C240 M8 5x RTX PRO 4500 32GB: llama3.1-8b Server 11426.1 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 11426.1,
   "tok/s aggregate Server", "Cisco UCS C240 M8, 5x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-cisco-5xrtxpro4500-llama31-8b-interactive-tokens",
   "MLPerf v6.1 Cisco C240 M8 5x RTX PRO 4500 32GB: llama3.1-8b Interactive 8494.68 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 8494.68,
   "tok/s aggregate Interactive", "Cisco UCS C240 M8, 5x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-cisco-8xrtxpro4500-llama31-8b-offline-tokens",
   "MLPerf v6.1 Cisco C845A M8 8x RTX PRO 4500 32GB: llama3.1-8b Offline 18800.9 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 18800.9,
   "tok/s aggregate Offline", "Cisco UCS C845A M8, 8x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-cisco-8xrtxpro4500-llama31-8b-server-tokens",
   "MLPerf v6.1 Cisco C845A M8 8x RTX PRO 4500 32GB: llama3.1-8b Server 18186.2 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 18186.2,
   "tok/s aggregate Server", "Cisco UCS C845A M8, 8x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-cisco-4xrtxpro4500-llama31-8b-offline-tokens",
   "MLPerf v6.1 Cisco X210 M8 X580P 4x RTX PRO 4500 32GB: llama3.1-8b Offline 9518.66 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 9518.66,
   "tok/s aggregate Offline", "Cisco UCS X210 M8 X580P PCIe node, 4x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-cisco-4xrtxpro4500-llama31-8b-server-tokens",
   "MLPerf v6.1 Cisco X210 M8 X580P 4x RTX PRO 4500 32GB: llama3.1-8b Server 9277.73 tok/s",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-3.1-8B", "tps_aggregate", 9277.73,
   "tok/s aggregate Server", "Cisco UCS X210 M8 X580P PCIe node, 4x RTX PRO 4500 32GB", "TensorRT 10.14, CUDA 13.1")
mp("mlperf-v6-1-dell-8xrtxpro4500-llama31-8b-offline-tokens",
   "MLPerf v6.1 Dell XE7745 8x RTX PRO 4500: llama2-70b-99.9 Offline 4374.06 tok/s (8B row n/a; 70B only)",
   ["nvidia-rtx-pro-4500-blackwell"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 4374.06,
   "tok/s aggregate Offline", "Dell PowerEdge XE7745, 8x RTX PRO 4500", "TensorRT 10.14, CUDA 13.3")

# MI350P additional scenarios (Offline already recorded for all three models)
mp("mlperf-v6-1-dell-8xmi350p-llama2-70b-server-tokens",
   "MLPerf v6.1 Dell XE7745 8x MI350P: llama2-70b-99.9 Server 41830 tok/s",
   ["amd-instinct-mi350p"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 41830.0,
   "tok/s aggregate Server", "Dell XE7745 8x MI350P, 2x EPYC 9455", "vLLM 0.22.0, PyTorch 2.10.0, ROCm 7.2.2")
mp("mlperf-v6-1-dell-8xmi350p-llama2-70b-interactive-tokens",
   "MLPerf v6.1 Dell XE7745 8x MI350P: llama2-70b-99.9 Interactive 22860.5 tok/s",
   ["amd-instinct-mi350p"], "Llama-2-70B (99.9 acc)", "tps_aggregate", 22860.5,
   "tok/s aggregate Interactive", "Dell XE7745 8x MI350P, 2x EPYC 9455", "vLLM 0.22.0, PyTorch 2.10.0, ROCm 7.2.2")
mp("mlperf-v6-1-dell-8xmi350p-llama31-8b-server-tokens",
   "MLPerf v6.1 Dell XE7745 8x MI350P: llama3.1-8b Server 72305.1 tok/s",
   ["amd-instinct-mi350p"], "Llama-3.1-8B", "tps_aggregate", 72305.1,
   "tok/s aggregate Server", "Dell XE7745 8x MI350P, 2x EPYC 9455", "vLLM 0.22.0, PyTorch 2.10.0, ROCm 7.2.2")
mp("mlperf-v6-1-dell-8xmi350p-llama31-8b-interactive-tokens",
   "MLPerf v6.1 Dell XE7745 8x MI350P: llama3.1-8b Interactive 62723.7 tok/s",
   ["amd-instinct-mi350p"], "Llama-3.1-8B", "tps_aggregate", 62723.7,
   "tok/s aggregate Interactive", "Dell XE7745 8x MI350P, 2x EPYC 9455", "vLLM 0.22.0, PyTorch 2.10.0, ROCm 7.2.2")
mp("mlperf-v6-1-dell-8xmi350p-gptoss-120b-server-tokens",
   "MLPerf v6.1 Dell XE7745 8x MI350P: gpt-oss-120b Server 40867.9 tok/s",
   ["amd-instinct-mi350p"], "gpt-oss-120b", "tps_aggregate", 40867.9,
   "tok/s aggregate Server", "Dell XE7745 8x MI350P, 2x EPYC 9455", "vLLM 0.22.0, PyTorch 2.10.0, ROCm 7.2.2")

# MangoBoost heterogeneous + stock-vllm MI300X rows
mp("mlperf-v6-1-mangoboost-8xmi300x-gptoss-120b-offline-tokens",
   "MLPerf v6.1 MangoBoost 8x MI300X (stock vLLM 0.24.0 ROCm): gpt-oss-120b Offline 30198.4 tok/s",
   ["amd-instinct-mi300x"], "gpt-oss-120b", "tps_aggregate", 30198.4,
   "tok/s aggregate Offline", "8xMI300X_2xEPYC_9534", "vLLM 0.24.0 (stock vllm/vllm-openai-rocm:v0.24.0), ROCm 7.2.1",
   extra="Notable because the image is STOCK vLLM, not a vendor-tuned stack.")
mp("mlperf-v6-1-mangoboost-8xmi300x-gptoss-120b-server-tokens",
   "MLPerf v6.1 MangoBoost 8x MI300X (stock vLLM 0.24.0 ROCm): gpt-oss-120b Server 24863 tok/s",
   ["amd-instinct-mi300x"], "gpt-oss-120b", "tps_aggregate", 24863.0,
   "tok/s aggregate Server", "8xMI300X_2xEPYC_9534", "vLLM 0.24.0 (stock vllm/vllm-openai-rocm:v0.24.0), ROCm 7.2.1")
mp("mlperf-v6-1-mangoboost-het-mi300x-mi355x-gptoss-120b-offline-tokens",
   "MLPerf v6.1 MangoBoost heterogeneous 2x8 MI300X + 2x8 MI355X (LLMBoost P/D): gpt-oss-120b Offline 285454 tok/s",
   ["amd-instinct-mi300x", "amd-instinct-mi355x"], "gpt-oss-120b", "tps_aggregate", 285454.0,
   "tok/s aggregate Offline", "Heterogeneous 4-cluster: 2x8 MI300X 192GB + 2x8 MI355X 288GB, LLMBoost", "ROCm / LLMBoost",
   extra="32-GPU prefill/decode-disaggregated cluster; largest single gpt-oss-120b Offline figure in v6.1.")
mp("mlperf-v6-1-mangoboost-het-mi300x-mi355x-gptoss-120b-server-tokens",
   "MLPerf v6.1 MangoBoost heterogeneous 2x8 MI300X + 2x8 MI355X (LLMBoost P/D): gpt-oss-120b Server 253501 tok/s",
   ["amd-instinct-mi300x", "amd-instinct-mi355x"], "gpt-oss-120b", "tps_aggregate", 253501.0,
   "tok/s aggregate Server", "Heterogeneous 4-cluster: 2x8 MI300X + 2x8 MI355X, LLMBoost", "ROCm / LLMBoost")

# Single-H200 RDI submission
mp("mlperf-v6-1-1xh200-llama31-8b-offline-tokens",
   "MLPerf v6.1 Naeem Khoshnevis 1x H200-SXM-141GB (RDI): llama3.1-8b Offline 7796.59 tok/s",
   ["nvidia-h200-sxm"], "Llama-3.1-8B FP8", "tps_aggregate", 7796.59,
   "tok/s aggregate Offline", "single H200-SXM-141GB, RDI availability category", "TensorRT-LLM 1.0.0 (FP8 W8A8 trtllm-build/trtllm-serve), CUDA 12.9",
   extra="RDI (research/dev/internal) category - not an Available-systems submission. Rare single-GPU H200 row.")
mp("mlperf-v6-1-1xh200-llama31-8b-server-tokens",
   "MLPerf v6.1 Naeem Khoshnevis 1x H200-SXM-141GB (RDI): llama3.1-8b Server 7376.6 tok/s",
   ["nvidia-h200-sxm"], "Llama-3.1-8B FP8", "tps_aggregate", 7376.6,
   "tok/s aggregate Server", "single H200-SXM-141GB, RDI category", "TensorRT-LLM 1.0.0 (FP8 W8A8), CUDA 12.9")

# ---- MLPerf v6.1 EDGE rows: agentic-edge SingleStream latency (ms, lower = better) ----
def mpe(id, name, accel, value, sysdesc, sw, extra="", eng="tensorrt-llm"):
    swl = sw.lower()
    if "llama.cpp" in swl:
        eng = "llama-cpp"
    rec(id, name, eng, accel, "MLPerf 'agentic edge' workload", None, "itl_ms", value,
        "ms SingleStream latency for the agentic-edge benchmark (lower is better; scenario-completion latency)",
        f"MLPerf Inference v6.1 EDGE closed submission ({sysdesc}; software: {sw}). SingleStream latency in ms for the "
        "'agentic edge' benchmark - this is a LATENCY result, not throughput; recorded under itl_ms as the only ms-latency "
        "member of the metric enum that fits a per-request latency measure. " + extra,
        "vendor", True, "", [MP], conf=0.7)

mpe("mlperf-v6-1-edge-dgx-spark-gb10-agentic-singlestream-lat",
    "MLPerf v6.1 edge: NVIDIA DGX Spark (GB10) agentic-edge SingleStream 3807.66 ms",
    ["nvidia-gb10"], 3807.66351136147, "Atlas_Inference, NVIDIA DGX Spark GB10, NVFP4", "Atlas (spark), CUDA NVFP4",
    extra="NVFP4-quantized run. ~1.85x faster than the same-submitter Strix Halo row (7058.99 ms).")
mpe("mlperf-v6-1-edge-strix-halo-8060s-agentic-singlestream-lat",
    "MLPerf v6.1 edge: AMD Strix Halo (Ryzen AI Max+ 395 / 8060S) agentic-edge SingleStream 7058.99 ms",
    ["amd-ryzen-ai-max-395-framework-desktop", "amd-radeon-8060s-ryzen-ai-max"], 7058.989199685204,
    "Atlas_Inference, AMD Strix Halo desktop", "Atlas (spark), native-HIP NVFP4")
mpe("mlperf-v6-1-edge-rtxpro4500-1x-agentic-singlestream-lat",
    "MLPerf v6.1 edge: HPE DL145 1x RTX PRO 4500 SE 32GB agentic-edge SingleStream 2757.95 ms",
    ["nvidia-rtx-pro-4500-blackwell"], 2757.954404037736,
    "HPE ProLiant DL145 Gen11, 1x RTX PRO 4500 SE 32GB", "llama.cpp",
    extra="Notable: an llama.cpp submission inside MLPerf closed division - rare.")
mpe("mlperf-v6-1-edge-rtxpro4500-4x-agentic-singlestream-lat",
    "MLPerf v6.1 edge: HPE DL345 4x RTX PRO 4500 SE 32GB agentic-edge SingleStream 2616.16 ms",
    ["nvidia-rtx-pro-4500-blackwell"], 2616.15900548858,
    "HPE ProLiant DL345 Gen12, 4x RTX PRO 4500 SE 32GB", "llama.cpp")
mpe("mlperf-v6-1-edge-jetson-thor-agentic-singlestream-lat",
    "MLPerf v6.1 edge: NVIDIA Jetson AGX Thor 128GB agentic-edge SingleStream 1466.01 ms",
    ["nvidia-jetson-agx-thor"], 1466.007675716981,
    "NVIDIA Jetson AGX Thor Developer Kit 128G", "TensorRT 10.16, CUDA 13.2",
    extra="Fastest agentic-edge SingleStream in the v6.1 edge closed division.")
mpe("mlperf-v6-1-edgeopen-rtxpro4500-4x-llama31-8b-singlestream-lat",
    "MLPerf v6.1 edge OPEN: HPE DL345 4x RTX PRO 4500 PCIe 32GB llama3.1-8b-edge SingleStream 1203.79 ms",
    ["nvidia-rtx-pro-4500-blackwell"], 1203.785699,
    "HPE ProLiant DL345 Gen12, 4x RTX PRO 4500-PCIe-32GB (open division)", "TensorRT 10.15.1.29, CUDA 13.2",
    extra="Open division; model = llama3.1-8b-edge.")

# ---- siliconscore M3 Ultra community rows (frontier MoE models) ----
SS_METH = ("COMMUNITY-UNVERIFIED row aggregated by siliconscore.com M3 Ultra chip page (accessed 2026-10-15); underlying "
           "evidence is the linked r/LocalLLaMA/HF measurement. MLX runtime, 4-bit-ish quant as named. avg decode tok/s "
           "single-request. Low-confidence forum-class evidence.")
SS_SRC = ["siliconscore-m3-ultra-benchmark-table"]
def ss(id, model, quant, value, ram, extra=""):
    rec(id, f"SiliconScore community row: Apple M3 Ultra, MLX, {model} {quant}, decode {value} t/s",
        "apple-mlx", ["apple-m3-ultra"], f"{model} ({quant})", None, "decode_tok_s", value,
        "tok/s avg decode (single-request)",
        SS_METH + f" RAM tier: {ram}. " + extra, "third_party", False, "", SS_SRC, conf=0.4, status="draft")

ss("ss-m3-ultra-qwen35-397b-a17b-41bit-decode", "Qwen3.5-397B-A17B", "q4.1bit MLX", 40.2, "512 GB",
   extra="A ~400B-class MoE at 40 tok/s is the M3 Ultra's headline MoE row; source is the inferencerlabs HF model card.")
ss("ss-m3-ultra-glm5-4bit-decode", "GLM-5", "4bit MLX", 16.7, "512 GB",
   extra="Best of five GLM-5 rows on the same machine (prompt tok/s 187.0 on this row).")
ss("ss-m3-ultra-qwen35-122b-a10b-mxfp4-decode", "Qwen3.5-122B-A10B", "MXFP4 MLX", 57.0, "256 GB")
ss("ss-m3-ultra-qwen35-122b-a10b-8bit-decode", "Qwen3.5-122B-A10B", "8bit MLX", 43.0, "256 GB")
ss("ss-m3-ultra-glm45-air-4bit-decode", "GLM-4.5-Air", "4bit MLX", 54.0, "256 GB")
ss("ss-m3-ultra-qwen3-coder-next-4bit-decode", "Qwen3-Coder-Next", "4bit MLX", 74.0, "256 GB")

print(f"\nwrote {len(written)} benchmark records")
for w in written: print(" ", w)
