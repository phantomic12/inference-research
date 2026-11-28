#!/usr/bin/env python3
"""Phase C pass-8 record writer: Argonne unmined rows + ROCm scoreboard gaps."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = "2026-10-14"

def w(rec, subdir="benchmarks"):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP:", rec["id"]); return
    open(out, "w").write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth,
          measured_by, sources, status="verified", conf=0.8, repro=True,
          interconnect=None, notes=""):
    return {"id": id, "type": "benchmark", "name": name, "engine_id": engine,
            "accelerator_ids": accs, "interconnect_ids": interconnect or [],
            "model": model, "format_id": fmt, "metric": metric, "value": value,
            "unit": unit, "methodology": meth, "measured_by": measured_by,
            "reproducible": repro, "energy_basis": None, "power_w": None,
            "power_scope": None, "power_cap": None, "clock_lock_mhz": None,
            "thermal_state": None, "notes": notes, "status": status,
            "confidence": conf, "updated": TODAY, "sources": sources}

A = "argonne-llm-inference-bench-all-results-csv"
R = "llama-cpp-discussion-15021-amd-rocm-scoreboard"
M7 = "llama-2-7b.Q4_0.gguf (TheBloke, 3.56 GiB, 6.74B params)"

def arg(id, name, engine, accs, model, val, detail, eng_desc, ic=None, notes="", conf=0.8):
    m = ("THIRD-PARTY MEASURED (Argonne LLM-Inference-Bench, arXiv:2411.00136, "
         "All_results.csv). " + eng_desc + ", FP16/BF16, 128 in / 128 out. "
         "Throughput = batch*(in+out)/end-to-end latency (paper Eq.2) = "
         "AGGREGATE over the batch, NOT per-request decode. " + detail)
    return bench(id, name, engine, accs, model, None, "tps_aggregate", val,
                 "tok/s aggregate (batch*(in+out)/latency)", m, "third_party",
                 [A], conf=conf, interconnect=ic, notes=notes)

A100 = "nvidia-a100-40gb-sxm4"
A100N = ("CSV column 'Nvidia A100 GPU' = ALCF ThetaGPU node (4x A100 40GB "
         "SXM4 + NVSwitch, paper Table II). Filed under nvidia-a100-40gb-sxm4 "
         "ONLY; the PCIe 40GB SKU shares the 1555 GB/s HBM2 but was not the "
         "measured part, so nvidia-a100-40gb-pcie stays uncovered.")
ENG = {"vllm": "vLLM 0.6.1 benchmark_throughput.py",
       "trtllm": "TensorRT-LLM 0.11.0",
       "llamacpp": "llama.cpp (ggml CUDA backend)",
       "dsmii": "DeepSpeed-MII",
       "sambaflow": "SambaFlow",
       "deepspeed": "DeepSpeed-MII (Gaudi fork)"}

# ---- A100 40GB SXM4 (ThetaGPU) ----
for (rid, name, eng, model, val, det, note) in [
 ("argonne-a100sxm40-vllm-llama2-7b-fp16-b1-128io-tps",
  "Argonne: A100 40GB SXM x1, vLLM 0.6.1, Llama-2-7B FP16, 128io, b1",
  "vllm", "meta-llama/Llama-2-7b-hf", 154.9,
  "Batch 1, 1x A100-40GB-SXM4. Latency 1.653s.",
  "bs1 = de-facto per-request decode for a 7B model. " + A100N),
 ("argonne-a100sxm40-vllm-llama2-7b-fp16-b64-128io-tps",
  "Argonne: A100 40GB SXM x1, vLLM 0.6.1, Llama-2-7B FP16, 128io, b64",
  "vllm", "meta-llama/Llama-2-7b-hf", 4919.5,
  "Batch 64, 1x A100-40GB-SXM4. Latency 3.330s.", A100N),
 ("argonne-a100sxm40-vllm-llama3-8b-fp16-b1-128io-tps",
  "Argonne: A100 40GB SXM x1, vLLM 0.6.1, Llama-3-8B FP16, 128io, b1",
  "vllm", "meta-llama/Meta-Llama-3-8B", 140.2,
  "Batch 1, 1x A100-40GB-SXM4. Latency 1.826s.",
  "GQA model slightly below Llama-2-7B, matches paper. " + A100N),
 ("argonne-a100sxm40-vllm-llama3-8b-fp16-b64-128io-tps",
  "Argonne: A100 40GB SXM x1, vLLM 0.6.1, Llama-3-8B FP16, 128io, b64",
  "vllm", "meta-llama/Meta-Llama-3-8B", 4981.9,
  "Batch 64, 1x A100-40GB-SXM4. Latency 3.289s.", A100N),
 ("argonne-a100sxm40-vllm-mistral-7b-fp16-b1-128io-tps",
  "Argonne: A100 40GB SXM x1, vLLM 0.6.1, Mistral-7B FP16, 128io, b1",
  "vllm", "mistralai/Mistral-7B-v0.1", 148.5,
  "Batch 1, 1x A100-40GB-SXM4. Latency 1.724s.", A100N),
 ("argonne-a100sxm40-trtllm-llama2-7b-fp16-b1-128io-tps",
  "Argonne: A100 40GB SXM x1, TensorRT-LLM 0.11.0, Llama-2-7B FP16, 128io, b1",
  "trtllm", "meta-llama/Llama-2-7b-hf", 169.0,
  "Batch 1, 1x A100-40GB-SXM4. Latency 1.515s.",
  "TRT-LLM fastest single-card engine in the sweep. " + A100N),
 ("argonne-a100sxm40-llamacpp-llama2-7b-fp16-b1-128io-tps",
  "Argonne: A100 40GB SXM x1, llama.cpp CUDA, Llama-2-7B FP16, 128io, b1",
  "llamacpp", "meta-llama/Llama-2-7b-hf", 58.9,
  "Batch 1, 1x A100-40GB-SXM4. Latency 4.345s.",
  "llama.cpp FP16 ~2.9x slower than vLLM at identical bs1. " + A100N),
 ("argonne-a100sxm40-vllm-llama2-70b-fp16-4x-b1-128io-tps",
  "Argonne: A100 40GB SXM x4 TP, vLLM 0.6.1, Llama-2-70B FP16, 128io, b1",
  "vllm", "meta-llama/Llama-2-70b-hf", 59.7,
  "Batch 1, 4x A100-40GB-SXM4 TP (NVSwitch node). Latency 4.285s.",
  "70B FP16 needs 4x40GB. " + A100N),
 ("argonne-a100sxm40-trtllm-llama2-70b-fp16-4x-b64-128io-tps",
  "Argonne: A100 40GB SXM x4 TP, TensorRT-LLM 0.11.0, Llama-2-70B FP16, 128io, b64",
  "trtllm", "meta-llama/Llama-2-70b-hf", 2526.0,
  "Batch 64, 4x A100-40GB-SXM4 TP. Latency 6.486s.",
  "Best 70B aggregate on 4xA100 in the CSV. " + A100N),
 ("argonne-a100sxm40-dsmii-llama2-70b-fp16-4x-b64-128io-tps",
  "Argonne: A100 40GB SXM x4 TP, DeepSpeed-MII, Llama-2-70B FP16, 128io, b64",
  "dsmii", "meta-llama/Llama-2-70b-hf", 1804.2,
  "Batch 64, 4x A100-40GB-SXM4 TP. Latency 9.081s.",
  "DS-MII ~1.4x behind TRT-LLM at bs64. " + A100N),
]:
    w(arg(rid, name, eng, [A100], model, val, det, ENG[eng],
          ic=["nvlink-3", "nvswitch-3"] if "4x" in rid else None, notes=note))

# ---- MI300X (argonne, unmined combos) ----
M3 = "amd-instinct-mi300x"
M3N = ("CSV 'AMD MI300X GPU' = 8x MI300X node (xGMI Infinity Fabric). "
       "Num-of-Hardware = cards used.")
for (rid, name, eng, model, val, det, note, ic) in [
 ("argonne-mi300x-vllm-llama3-8b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Llama-3-8B FP16, 128io, b1",
  "vllm", "meta-llama/Meta-Llama-3-8B", 118.0,
  "Batch 1, 1x MI300X. Latency 2.170s.", M3N, None),
 ("argonne-mi300x-vllm-llama3-8b-fp16-b64-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Llama-3-8B FP16, 128io, b64",
  "vllm", "meta-llama/Meta-Llama-3-8B", 4185.4,
  "Batch 64, 1x MI300X. Latency 3.915s.", M3N, None),
 ("argonne-mi300x-vllm-llama2-70b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Llama-2-70B FP16, 128io, b1",
  "vllm", "meta-llama/Llama-2-70b-hf", 32.8,
  "Batch 1, 1x MI300X. Latency 7.796s.",
  "Single-card 70B FP16 fits in 192GB HBM3 - no TP needed. " + M3N, None),
 ("argonne-mi300x-vllm-llama2-70b-fp16-b64-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Llama-2-70B FP16, 128io, b64",
  "vllm", "meta-llama/Llama-2-70b-hf", 458.9,
  "Batch 64, 1x MI300X. Latency 35.701s.",
  "Single-GPU bs64 70B: KV-cache pressure visible vs 4x A100. " + M3N, None),
 ("argonne-mi300x-vllm-llama3-70b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Llama-3-70B FP16, 128io, b1",
  "vllm", "meta-llama/Meta-Llama-3-70B", 35.9,
  "Batch 1, 1x MI300X. Latency 7.127s.", M3N, None),
 ("argonne-mi300x-vllm-mixtral-8x7b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Mixtral-8x7B FP16, 128io, b1",
  "vllm", "mistralai/Mixtral-8x7B-v0.1", 161.0,
  "Batch 1, 1x MI300X. Latency 1.590s.",
  "MoE decode is bandwidth-light; 5x the dense-70B figure. " + M3N, None),
 ("argonne-mi300x-vllm-mixtral-8x7b-fp16-b64-128io-tps",
  "Argonne: MI300X x1, vLLM 0.6.1, Mixtral-8x7B FP16, 128io, b64",
  "vllm", "mistralai/Mixtral-8x7B-v0.1", 2718.9,
  "Batch 64, 1x MI300X. Latency 6.026s.", M3N, None),
 ("argonne-mi300x-llamacpp-llama3-8b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, llama.cpp HIP, Llama-3-8B FP16, 128io, b1",
  "llamacpp", "meta-llama/Meta-Llama-3-8B", 63.8,
  "Batch 1, 1x MI300X. Latency 4.013s.",
  "llama.cpp HIP ~1.85x behind vLLM at bs1 on MI300X. " + M3N, None),
 ("argonne-mi300x-llamacpp-mixtral-8x7b-fp16-b1-128io-tps",
  "Argonne: MI300X x1, llama.cpp HIP, Mixtral-8x7B FP16, 128io, b1",
  "llamacpp", "mistralai/Mixtral-8x7B-v0.1", 33.0,
  "Batch 1, 1x MI300X. Latency 7.764s.", M3N, None),
]:
    w(arg(rid, name, eng, [M3], model, val, det, ENG[eng], ic=ic, notes=note))

# ---- MI250 extra combos ----
M2 = "amd-instinct-mi250"
M2N = ("CSV 'AMD MI250 GPU' = ALCF Polaris node (4x MI250, Infinity Fabric). "
       "The single-GCD MI250 SKU (not MI250X); prior records establish this.")
for (rid, name, eng, model, val, det, note, ic) in [
 ("argonne-mi250-vllm-mistral-7b-fp16-b1-128io-tps",
  "Argonne: MI250 x1, vLLM 0.6.1, Mistral-7B FP16, 128io, b1",
  "vllm", "mistralai/Mistral-7B-v0.1", 179.1,
  "Batch 1, 1x MI250. Latency 1.430s.", M2N, None),
 ("argonne-mi250-vllm-mistral-7b-fp16-b64-128io-tps",
  "Argonne: MI250 x1, vLLM 0.6.1, Mistral-7B FP16, 128io, b64",
  "vllm", "mistralai/Mistral-7B-v0.1", 6245.5,
  "Batch 64, 1x MI250. Latency 2.623s.", M2N, None),
 ("argonne-mi250-vllm-llama3-70b-fp16-4x-b1-128io-tps",
  "Argonne: MI250 x4 TP, vLLM 0.6.1, Llama-3-70B FP16, 128io, b1",
  "vllm", "meta-llama/Meta-Llama-3-70B", 47.1,
  "Batch 1, 4x MI250 TP. Latency 5.432s.",
  "GQA 70B on 4xMI250 ~= Llama-2-70B rate. " + M2N,
  ["amd-infinity-fabric-link"]),
 ("argonne-mi250-vllm-llama3-70b-fp16-4x-b64-128io-tps",
  "Argonne: MI250 x4 TP, vLLM 0.6.1, Llama-3-70B FP16, 128io, b64",
  "vllm", "meta-llama/Meta-Llama-3-70B", 1744.2,
  "Batch 64, 4x MI250 TP. Latency 9.394s.", M2N,
  ["amd-infinity-fabric-link"]),
 ("argonne-mi250-llamacpp-mistral-7b-fp16-b1-128io-tps",
  "Argonne: MI250 x1, llama.cpp HIP, Mistral-7B FP16, 128io, b1",
  "llamacpp", "mistralai/Mistral-7B-v0.1", 41.4,
  "Batch 1, 1x MI250. Latency 6.190s.",
  "llama.cpp ~4.3x behind vLLM at bs1 on CDNA2. " + M2N, None),
]:
    w(arg(rid, name, eng, [M2], model, val, det, ENG[eng], ic=ic, notes=note))

# ---- Gaudi2 + SN40L extras ----
for (rid, name, eng, accs, model, val, det, note) in [
 ("argonne-gaudi2-deepspeed-llama3-8b-bf16-8x-b64-128io-tps",
  "Argonne: Gaudi2 x8, DeepSpeed-MII, Llama-3-8B BF16, 128io, b64",
  "deepspeed", ["intel-gaudi2"], "meta-llama/Meta-Llama-3-8B", 18187.1,
  "Batch 64, 8x Gaudi2 HL-225B (Habana SynapseAI). Latency 0.901s.",
  "8-chip aggregate; 18187 t/s is the CSV's biggest Gaudi2 number."),
 ("argonne-gaudi2-deepspeed-mistral-7b-bf16-8x-b64-128io-tps",
  "Argonne: Gaudi2 x8, DeepSpeed-MII, Mistral-7B BF16, 128io, b64",
  "deepspeed", ["intel-gaudi2"], "mistralai/Mistral-7B-v0.1", 14502.3,
  "Batch 64, 8x Gaudi2. Latency 1.130s.", ""),
 ("argonne-sn40l-sambaflow-llama3-8b-8x-b16-128io-tps",
  "Argonne: SN40L x8, SambaFlow, Llama-3-8B, 128io, b16",
  "sambaflow", ["sambanova-sn40l"], "meta-llama/Meta-Llama-3-8B", 7381.3,
  "Batch 16, 8x SN40L RDU sockets. Latency 0.555s.",
  "SambaFlow compiles a dataflow graph per model - numbers are not "
  "comparable 1:1 with GPU per-request decode."),
]:
    w(arg(rid, name, eng, accs, model, val, det, ENG[eng], notes=note))

# ---- ROCm #15021 missing rows on existing accel ids ----
def rocm(rid, name, chip_acc, chip_desc, mem, pp, tg, std, commit, who, fa, extra=""):
    metric = "decode_tok_s" if tg else "prefill_tok_s"
    val = tg if tg else pp
    test = "tg128" if tg else "pp512"
    m = ("COMMUNITY-SCOREBOARD (llama.cpp discussion #15021, maintainer-"
         "curated). llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa " +
         str(fa) + ". " + chip_desc + " (" + mem + "). Build commit " +
         commit + ", reporter @" + who + ". Mean +/- std (±" + std +
         "). Single-GPU run per thread rules. Per-request batch-1, "
         "NOT aggregate.")
    n = ("Maintainer-verified scoreboard row from #15021. Identical "
         "model/test/flags across all GPUs. tg128 = decode, pp512 = "
         "prompt processing. " + extra)
    unit = ("tok/s (" + test + ", per-request, batch 1)")
    return bench(rid, name, "llama-cpp", [chip_acc], M7, "gguf-q4-0",
                 metric, val, unit, m, "third_party", [R], conf=0.8, notes=n)

# MI60 FA
w(rocm("llamacpp-rocm-mi60-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on Radeon Instinct MI60: Llama-2-7B Q4_0 FA-on pp512",
   "amd-radeon-instinct-mi60", "AMD Radeon Instinct MI60", "32 GB / HBM2 / 4096 bit",
   1327.0, None, "0.62", "504af20", "Said-Akbar", 1,
   "FA-on variant; paired with the noFA row already in the repo."))
w(rocm("llamacpp-rocm-mi60-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on Radeon Instinct MI60: Llama-2-7B Q4_0 FA-on tg128",
   "amd-radeon-instinct-mi60", "AMD Radeon Instinct MI60", "32 GB / HBM2 / 4096 bit",
   None, 96.04, "0.13", "504af20", "Said-Akbar", 1,
   "FA-on decode. Vega20's 1TB/s HBM2 keeps tg respectable for a 2018 part."))
# Pro VII FA
w(rocm("llamacpp-rocm-pro-vii-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on Radeon Pro VII: Llama-2-7B Q4_0 FA-on pp512",
   "amd-radeon-pro-vii", "AMD Radeon Pro VII", "16 GB / HBM2 / 4096 bit",
   1116.0, None, "1.18", "2739a71", "8XXD8", 1))
w(rocm("llamacpp-rocm-pro-vii-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on Radeon Pro VII: Llama-2-7B Q4_0 FA-on tg128",
   "amd-radeon-pro-vii", "AMD Radeon Pro VII", "16 GB / HBM2 / 4096 bit",
   None, 89.95, "0.04", "2739a71", "8XXD8", 1))
# AI PRO R9700 noFA + FA
w(rocm("llamacpp-rocm-r9700-llama27b-q40-nofa-pp512",
   "llama.cpp ROCm on Radeon AI PRO R9700: Llama-2-7B Q4_0 FA-off pp512",
   "amd-radeon-ai-pro-r9700", "AMD Radeon AI PRO R9700", "32 GB / GDDR6 / 256 bit",
   4443.54, None, "339.25", "bd4ef13", "gogich77", 0,
   "RDNA4 workstation; pp std is large (±339) - first-run variance."))
w(rocm("llamacpp-rocm-r9700-llama27b-q40-nofa-tg128",
   "llama.cpp ROCm on Radeon AI PRO R9700: Llama-2-7B Q4_0 FA-off tg128",
   "amd-radeon-ai-pro-r9700", "AMD Radeon AI PRO R9700", "32 GB / GDDR6 / 256 bit",
   None, 93.84, "0.26", "bd4ef13", "gogich77", 0,
   "640 GB/s GDDR6 caps decode around ~94 tok/s for Q4_0 7B."))
w(rocm("llamacpp-rocm-r9700-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on Radeon AI PRO R9700: Llama-2-7B Q4_0 FA-on pp512",
   "amd-radeon-ai-pro-r9700", "AMD Radeon AI PRO R9700", "32 GB / GDDR6 / 256 bit",
   4773.07, None, "49.30", "bd4ef13", "gogich77", 1))
w(rocm("llamacpp-rocm-r9700-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on Radeon AI PRO R9700: Llama-2-7B Q4_0 FA-on tg128",
   "amd-radeon-ai-pro-r9700", "AMD Radeon AI PRO R9700", "32 GB / GDDR6 / 256 bit",
   None, 97.98, "0.13", "bd4ef13", "gogich77", 1))
# RX 9070 XT noFA + FA
w(rocm("llamacpp-rocm-9070xt-llama27b-q40-nofa-pp512",
   "llama.cpp ROCm on RX 9070 XT: Llama-2-7B Q4_0 FA-off pp512",
   "amd-radeon-rx-9070-xt", "AMD Radeon RX 9070 XT", "16 GB / GDDR6 / 256 bit",
   5055.19, None, "109.58", "583cb83", "Hadrianneue", 0,
   "RDNA4 prefill is strong - ahead of MI100 on compute-bound pp."))
w(rocm("llamacpp-rocm-9070xt-llama27b-q40-nofa-tg128",
   "llama.cpp ROCm on RX 9070 XT: Llama-2-7B Q4_0 FA-off tg128",
   "amd-radeon-rx-9070-xt", "AMD Radeon RX 9070 XT", "16 GB / GDDR6 / 256 bit",
   None, 101.27, "0.27", "583cb83", "Hadrianneue", 0,
   "640 GB/s GDDR6; sits between MI100 (110) and W7900 (121) on decode."))
w(rocm("llamacpp-rocm-9070xt-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on RX 9070 XT: Llama-2-7B Q4_0 FA-on pp512",
   "amd-radeon-rx-9070-xt", "AMD Radeon RX 9070 XT", "16 GB / GDDR6 / 256 bit",
   4903.51, None, "96.36", "583cb83", "Hadrianneue", 1))
w(rocm("llamacpp-rocm-9070xt-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on RX 9070 XT: Llama-2-7B Q4_0 FA-on tg128",
   "amd-radeon-rx-9070-xt", "AMD Radeon RX 9070 XT", "16 GB / GDDR6 / 256 bit",
   None, 97.28, "0.13", "583cb83", "Hadrianneue", 1))
# MI300X scoreboard rows (ROCm, different from Argonne vLLM rows)
w(rocm("llamacpp-rocm-mi300x-llama27b-q40-nofa-pp512",
   "llama.cpp ROCm on Instinct MI300X: Llama-2-7B Q4_0 FA-off pp512 (#15021)",
   "amd-instinct-mi300x", "AMD Instinct MI300X", "192 GB / HBM3 / 8192 bit",
   11476.40, None, "72.79", "ee3a9fc", "yeahdongcn", 0,
   "Top of the ROCm scoreboard by a wide margin."))
w(rocm("llamacpp-rocm-mi300x-llama27b-q40-nofa-tg128",
   "llama.cpp ROCm on Instinct MI300X: Llama-2-7B Q4_0 FA-off tg128 (#15021)",
   "amd-instinct-mi300x", "AMD Instinct MI300X", "192 GB / HBM3 / 8192 bit",
   None, 232.92, "0.53", "ee3a9fc", "yeahdongcn", 0,
   "5.3 TB/s HBM3: fastest tg128 on the ROCm scoreboard."))
w(rocm("llamacpp-rocm-mi300x-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on Instinct MI300X: Llama-2-7B Q4_0 FA-on pp512 (#15021)",
   "amd-instinct-mi300x", "AMD Instinct MI300X", "192 GB / HBM3 / 8192 bit",
   11945.97, None, "54.29", "ee3a9fc", "yeahdongcn", 1))
w(rocm("llamacpp-rocm-mi300x-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on Instinct MI300X: Llama-2-7B Q4_0 FA-on tg128 (#15021)",
   "amd-instinct-mi300x", "AMD Instinct MI300X", "192 GB / HBM3 / 8192 bit",
   None, 218.53, "0.09", "ee3a9fc", "yeahdongcn", 1))
# 7900 XTX scoreboard rows
w(rocm("llamacpp-rocm-7900xtx-llama27b-q40-nofa-pp512",
   "llama.cpp ROCm on RX 7900 XTX: Llama-2-7B Q4_0 FA-off pp512 (#15021)",
   "amd-radeon-rx-7900-xtx", "AMD Radeon RX 7900 XTX", "24 GB / GDDR6 / 384 bit",
   3552.27, None, "101.96", "2f0c2db", "Diablo-D3", 0))
w(rocm("llamacpp-rocm-7900xtx-llama27b-q40-nofa-tg128",
   "llama.cpp ROCm on RX 7900 XTX: Llama-2-7B Q4_0 FA-off tg128 (#15021)",
   "amd-radeon-rx-7900-xtx", "AMD Radeon RX 7900 XTX", "24 GB / GDDR6 / 384 bit",
   None, 167.11, "0.50", "2f0c2db", "Diablo-D3", 0,
   "960 GB/s; the consumer ROCm decode leader on this board."))
w(rocm("llamacpp-rocm-7900xtx-llama27b-q40-fa-pp512",
   "llama.cpp ROCm on RX 7900 XTX: Llama-2-7B Q4_0 FA-on pp512 (#15021)",
   "amd-radeon-rx-7900-xtx", "AMD Radeon RX 7900 XTX", "24 GB / GDDR6 / 384 bit",
   3874.25, None, "11.92", "2f0c2db", "Diablo-D3", 1))
w(rocm("llamacpp-rocm-7900xtx-llama27b-q40-fa-tg128",
   "llama.cpp ROCm on RX 7900 XTX: Llama-2-7B Q4_0 FA-on tg128 (#15021)",
   "amd-radeon-rx-7900-xtx", "AMD Radeon RX 7900 XTX", "24 GB / GDDR6 / 384 bit",
   None, 170.12, "0.56", "2f0c2db", "Diablo-D3", 1))
