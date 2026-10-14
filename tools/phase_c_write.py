#!/usr/bin/env python3
"""Phase C record writer: emits benchmark+source JSON records into the worktree."""
import json, os

ROOT = "/opt/data/projects/inference-research/wt-bench"
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

def source(id, name, url, publisher, kind, notes, status="verified", conf=0.5):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url, "publisher": publisher,
            "kind": kind, "published": None, "accessed": TODAY, "archived_url": None,
            "notes": notes}

def wsrc(rec):
    w(rec, subdir="sources")

SRC_CUDA = "llama-cpp-discussion-15013-nvidia-cuda-scoreboard"
SRC_ROCM = "llama-cpp-discussion-15021-amd-rocm-scoreboard"
SRC_APPLE = "llama-cpp-discussion-4167-apple-silicon-llama-bench-table"
SRC_KYUZ0 = "kyuz0-strix-halo-toolboxes-llamabench-results-json"
SRC_ML61 = "bench-mlperf-v6-1-summary-csv-llm-rows"
SRC_ML50_L40S = "bench-mlperf-v5-0-dell-xe7745-l40sx8-llama2-70b"
SRC_ML51_L40S = "bench-mlperf-v5-1-redhat-ibm-l40s-vllm-llama3-8b"
SRC_HC5090 = "hardware-corner-rtx-5090-llm-benchmarks"
SRC_VULKAN = "llama-cpp-discussion-10879-vulkan-scoreboard"
SRC_LHL = "lhl-strix-halo-testing-llm-bench"
SRC_AHELPME = "ahelpme-epyc-9554-llama-bench-cpu-inference"

MODEL7B = "llama-2-7b.Q4_0.gguf (TheBloke, 3.56 GiB, 6.74B params)"

# ---------- new sources ----------
wsrc(source(SRC_HC5090,
    "Hardware Corner — RTX 5090 32GB local LLM llama-bench run (Qwen3 Q4_K_XL context sweep)",
    "https://www.hardware-corner.net/rtx-5090-llm-benchmarks/",
    "Hardware Corner (Allan Witt)", "benchmark",
    "llama-bench build 466c1911 on RTX 5090 32GB GDDR7, EPYC 7642 host, Ubuntu 24.04, -fa 1 -ngl 99 -p 1024/2048 -n 128, -d context-depth sweep 4k..131k on Qwen3 8B/14B/30B-A3B/32B Q4_K_XL + gpt-oss 20B. Per-request batch-1."))

wsrc(source(SRC_VULKAN,
    "llama.cpp Discussion #10879 — Performance of llama.cpp with Vulkan (community scoreboard)",
    "https://github.com/ggml-org/llama.cpp/discussions/10879",
    "GitHub / ggml-org", "forum",
    "Vulkan-backend sibling of the #15013 CUDA and #15021 ROCm scoreboards: llama-2-7b.Q4_0 pp512/tg128, maintainer-curated rows with build commit + reporter."))

wsrc(source(SRC_LHL,
    "lhl/strix-halo-testing llm-bench — context sweeps + DGX Spark vs Strix Halo comparison (ROCm rocWMMA / Vulkan AMDVLK-RADV)",
    "https://github.com/lhl/strix-halo-testing",
    "Lee Henderson (lhl) / Framework pre-production cluster", "repo",
    "gpt-oss-120b MXFP4 tg32/pp2048 at context depths 0..32768 on Ryzen AI Max+ 395 (Radeon 8060S gfx1151), llama.cpp build 6792, TheRock/ROCm nightly 7.10.0a20251017 + RADV 25.2.4/AMDVLK 2025.Q2.1 vs DGX Spark CUDA build 6767. DGX figures transcribed from ggerganov's #16578 sweep."))

# ---------- CUDA scoreboard rows for uncovered accelerators ----------
def cuda_meth(commit, reporter, fa, mem):
    return (f"COMMUNITY-SCOREBOARD (llama.cpp discussion #15013, maintainer-curated). "
            f"llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa {fa}. "
            f"Build commit {commit}, reporter {reporter}. llama-bench default -r 5 repeats, "
            f"mean +/- std. Single-GPU per thread rules. Per-request batch-1, NOT aggregate. "
            f"Hardware row: {mem}.")

NOTE_SB = ("Maintainer-verified scoreboard row from ggml-org/llama.cpp#15013. Identical "
           "model/test/flags across every GPU makes this the cleanest cross-GPU llama.cpp "
           "comparison in the repo. tg128 is bandwidth-bound; pp512 tracks compute.")

cuda_rows = [
    ("nvidia-tesla-p40", "Tesla P40", "24 GB / GDDR5 / 384 bit",
     1007.42, 54.74, "c76b420", "@m18coppola", 1079.66, 53.73, "c76b420", "@m18coppola"),
    ("nvidia-tesla-m40", "Tesla M40", "24 GB / GDDR5 / 384 bit",
     282.65, 38.04, "97d5117", "@Hedede", 290.17, 39.98, "97d5117", "@Hedede"),
    ("nvidia-v100-pcie-32gb", "Tesla V100 (PCIe 32GB)", "32 GB / HBM2 / 4096 bit",
     3042.64, 129.08, "51f5a45", "@Hedede", 2973.78, 134.76, "51f5a45", "@Hedede"),
    ("nvidia-rtx-pro-4500-blackwell", "RTX PRO 4500 Blackwell", "32 GB / GDDR7 / 256 bit",
     None, None, None, None, 7251.66, 168.90, "becc481", "@Hedede"),
    ("nvidia-l40s", "L40", "48 GB / GDDR6 / 384 bit",
     8870.49, 152.01, "ee09828", "@Hedede", 10097.64, 153.76, "ee09828", "@Hedede"),
]
for acc, chip, mem, pp0, tg0, c0, r0, pp1, tg1, c1, r1 in cuda_rows:
    short = acc.replace("nvidia-","")
    if tg0 is not None:
        w(bench(f"llamacpp-cuda-{short}-llama27b-q40-nofa-tg128",
                f"llama.cpp CUDA on {chip}: Llama-2-7B Q4_0 noFA tg128 decode tok/s (scoreboard #15013)",
                "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "decode_tok_s", tg0,
                "tok/s (tg128, per-request, batch 1)", cuda_meth(c0,r0,0,mem), "third_party",
                [SRC_CUDA], notes=NOTE_SB))
        w(bench(f"llamacpp-cuda-{short}-llama27b-q40-nofa-pp512",
                f"llama.cpp CUDA on {chip}: Llama-2-7B Q4_0 noFA pp512 prefill tok/s (scoreboard #15013)",
                "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "prefill_tok_s", pp0,
                "tok/s (pp512, batch 512)", cuda_meth(c0,r0,0,mem), "third_party",
                [SRC_CUDA], notes=NOTE_SB))
    if tg1 is not None:
        w(bench(f"llamacpp-cuda-{short}-llama27b-q40-fa-tg128",
                f"llama.cpp CUDA on {chip}: Llama-2-7B Q4_0 FA-on tg128 decode tok/s (scoreboard #15013)",
                "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "decode_tok_s", tg1,
                "tok/s (tg128, per-request, batch 1)", cuda_meth(c1,r1,1,mem), "third_party",
                [SRC_CUDA], notes=NOTE_SB))
        w(bench(f"llamacpp-cuda-{short}-llama27b-q40-fa-pp512",
                f"llama.cpp CUDA on {chip}: Llama-2-7B Q4_0 FA-on pp512 prefill tok/s (scoreboard #15013)",
                "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "prefill_tok_s", pp1,
                "tok/s (pp512, batch 512)", cuda_meth(c1,r1,1,mem), "third_party",
                [SRC_CUDA], notes=NOTE_SB))

# ---------- ROCm scoreboard additions (uncovered AMD parts) ----------
def rocm_meth(commit, reporter, fa, mem):
    return (f"COMMUNITY-SCOREBOARD (llama.cpp discussion #15021, maintainer-curated). "
            f"llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa {fa}, ROCm/HIP backend. "
            f"Build commit {commit}, reporter {reporter}. Default -r 5 repeats, mean +/- std. "
            f"Single GPU per run. Per-request batch-1, NOT aggregate. Hardware row: {mem}.")

NOTE_RS = ("Maintainer-verified scoreboard row from ggml-org/llama.cpp#15021 (ROCm). "
           "Identical model/test/flags across every AMD GPU.")

# MI250X is NOT on the ROCm scoreboard; skip. AMD 8060S Strix Halo IS on it (DDR5 rows).
# Note: "AI Max+ 395 | 128 GB LPDDR5" row = Strix Halo ROCm, and "Radeon 8060S | DDR5" = iGPU.
rocm_rows = [
    ("amd-radeon-8060s-ryzen-ai-max", "AI Max+ 395 (Strix Halo)", "128 GB / LPDDR5",
     911.36, 50.01, "e60f241", "@firefox42", None, None, None, None),
    ("amd-radeon-8060s-ryzen-ai-max", "Radeon 8060S (System Shared)", "System Shared / DDR5",
     351.36, 47.97, "1d0125b", "@hspak", None, None, None, None),
]
i = 0
for acc, chip, mem, pp0, tg0, c0, r0, pp1, tg1, c1, r1 in rocm_rows:
    i += 1
    w(bench(f"llamacpp-rocm-8060s-run{i}-llama27b-q40-nofa-tg128",
            f"llama.cpp ROCm on {chip}: Llama-2-7B Q4_0 noFA tg128 decode tok/s (scoreboard #15021)",
            "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "decode_tok_s", tg0,
            "tok/s (tg128, per-request, batch 1)", rocm_meth(c0,r0,0,mem), "third_party",
            [SRC_ROCM], notes=NOTE_RS))
    w(bench(f"llamacpp-rocm-8060s-run{i}-llama27b-q40-nofa-pp512",
            f"llama.cpp ROCm on {chip}: Llama-2-7B Q4_0 noFA pp512 prefill tok/s (scoreboard #15021)",
            "llama-cpp", [acc], MODEL7B, "gguf-q4-0", "prefill_tok_s", pp0,
            "tok/s (pp512, batch 512)", rocm_meth(c0,r0,0,mem), "third_party",
            [SRC_ROCM], notes=NOTE_RS))

# ---------- Apple Silicon scoreboard (#4167) ----------
# Old rows: commit 8e672ef (2023-11). Newer rows (M5 gen): c1d0e7a / 60081bb.
# llama-bench -hf ggml-org/Llama-2-7B-GGUF:{F16,Q8_0,Q4_0} -p 512 -n 128 -ngl 99
def apple_meth(commit, cores, ram_bw):
    return (f"COMMUNITY-SCOREBOARD (llama.cpp discussion #4167, ggerganov-maintained). "
            f"llama-bench -hf ggml-org/Llama-2-7B-GGUF -p 512 -n 128 -ngl 99 --delay 30, "
            f"Metal backend, full GPU offload. Build commit {commit}. "
            f"pp512 = prefill at batch 512; tg128 = batch-1 decode. mean +/- std over default -r 5. "
            f"Per-request, NOT aggregate. Chip: {cores} GPU cores, {ram_bw} GB/s memory BW.")

NOTE_AP = ("Maintainer-curated Apple Silicon llama-bench table (discussion #4167). "
           "Same Llama-2-7B weights, same pp512/tg128 protocol as the CUDA/ROCm scoreboards, "
           "so cross-vendor comparisons against #15013/#15021 rows are apples-to-apples.")

apple_rows = [
    # acc_id, chip label, gpu_cores, bw, commit, [(metric, quant, value)]
    ("apple-m1-ultra", "M1 Ultra (64 GPU)", 64, 800, "8e672ef",
     [("prefill_tok_s","F16",1168.89),("decode_tok_s","F16",37.01),
      ("decode_tok_s","gguf-q4-0",83.73),("prefill_tok_s","gguf-q4-0",1030.04),
      ("decode_tok_s","gguf-q8-0",59.87)]),
    ("apple-m2-ultra", "M2 Ultra (76 GPU)", 76, 800, "8e672ef",
     [("prefill_tok_s","F16",1401.85),("decode_tok_s","F16",41.02),
      ("decode_tok_s","gguf-q4-0",94.27),("prefill_tok_s","gguf-q4-0",1238.48),
      ("decode_tok_s","gguf-q8-0",66.64)]),
    ("apple-m2-ultra", "M2 Ultra (76 GPU)", 76, 800, "c1d0e7a",  # 2026-08 refresh, +FA
     [("prefill_tok_s","F16",1627.82),("decode_tok_s","F16",49.55),
      ("decode_tok_s","gguf-q4-0",125.21),("prefill_tok_s","gguf-q4-0",1488.81),
      ("decode_tok_s","gguf-q8-0",82.83)]),
    ("apple-m3-ultra", "M3 Ultra (80 GPU)", 80, 800, "8e672ef",
     [("prefill_tok_s","F16",1538.34),("decode_tok_s","F16",39.78),
      ("decode_tok_s","gguf-q4-0",92.14),("prefill_tok_s","gguf-q4-0",1471.24),
      ("decode_tok_s","gguf-q8-0",63.93)]),
    ("local-apple-m3-pro", "M3 Pro (18 GPU)", 18, 150, "8e672ef",
     [("prefill_tok_s","F16",357.45),("decode_tok_s","F16",9.89),
      ("decode_tok_s","gguf-q4-0",30.74),("prefill_tok_s","gguf-q4-0",341.67),
      ("decode_tok_s","gguf-q8-0",17.53)]),
    ("local-apple-m4-pro", "M4 Pro (20 GPU)", 20, 273, "8e672ef",
     [("prefill_tok_s","F16",464.48),("decode_tok_s","F16",17.18),
      ("decode_tok_s","gguf-q4-0",50.74),("prefill_tok_s","gguf-q4-0",439.78),
      ("decode_tok_s","gguf-q8-0",30.69)]),
    ("local-apple-m5", "M5 (10 GPU)", 10, 154, "c1d0e7a",
     [("decode_tok_s","gguf-q8-0",18.42),("decode_tok_s","gguf-q4-0",31.88),
      ("prefill_tok_s","gguf-q8-0",715.42),("prefill_tok_s","gguf-q4-0",722.79)]),
    ("local-apple-m5-pro", "M5 Pro (20 GPU)", 20, 307, "c1d0e7a",
     [("prefill_tok_s","F16",1588.78),("decode_tok_s","F16",21.55),
      ("decode_tok_s","gguf-q4-0",66.33),("prefill_tok_s","gguf-q4-0",1620.64),
      ("decode_tok_s","gguf-q8-0",38.92)]),
]
for acc, chip, cores, bw, commit, rows in apple_rows:
    short = acc.replace("apple-","").replace("local-","")
    for metric, fmt, val in rows:
        test = "pp512" if metric=="prefill_tok_s" else "tg128"
        q = fmt.lower().replace("gguf-","")
        bid = f"llamacpp-metal-{short}-llama27b-{q}-{test}-{commit}"
        w(bench(bid,
                f"llama.cpp Metal on {chip}: Llama-2-7B {fmt} {test} tok/s (scoreboard #4167, {commit})",
                "llama-cpp", [acc], "Llama-2-7B " + (fmt if fmt=="F16" else fmt+".gguf"),
                None if fmt=="F16" else fmt, metric, val,
                f"tok/s ({test}, " + ("batch 512" if metric=="prefill_tok_s" else "batch 1, per-request") + ")",
                apple_meth(commit, cores, bw), "third_party", [SRC_APPLE], notes=NOTE_AP))

# ---------- MLPerf records ----------
def mlperf_meth(desc):
    return ("MLPerf Inference closed-division submission. " + desc +
            " Values quoted verbatim from the submission's mlperf_log_summary.txt "
            "(performance/run_1). AGGREGATE throughput across all accelerators in the "
            "system, NOT per-request.")

# L40S: 8x Dell XE7745 v5.0 (llama2-70b-99.9, TRT-LLM)
w(bench("mlperf-v5-0-dell-xe7745-8xl40s-llama2-70b-offline-tokens",
        "MLPerf v5.0 Dell XE7745 8x L40S: llama2-70b-99.9 Offline aggregate tokens/s",
        "tensorrt-llm", ["nvidia-l40s"], "llama2-70b-99.9", None,
        "tps_aggregate", 3481.53, "tok/s AGGREGATE over 8x L40S 48GB",
        mlperf_meth("v5.0, Dell PowerEdge XE7745, 8x NVIDIA L40S 48GB PCIe, TensorRT-LLM. "
                    "Offline scenario: 'Tokens per second: 3481.53', VALID."),
        "vendor", [SRC_ML50_L40S], conf=0.85,
        interconnect=["pcie-gen4"],
        notes="Per-GPU ~435 tok/s aggregate. See also Server scenario 3201.13 tok/s."))
w(bench("mlperf-v5-0-dell-xe7745-8xl40s-llama2-70b-server-tokens",
        "MLPerf v5.0 Dell XE7745 8x L40S: llama2-70b-99.9 Server aggregate tokens/s",
        "tensorrt-llm", ["nvidia-l40s"], "llama2-70b-99.9", None,
        "tps_aggregate", 3201.13, "tok/s AGGREGATE over 8x L40S 48GB",
        mlperf_meth("v5.0, Dell XE7745, 8x L40S, TensorRT-LLM. Server scenario: "
                    "'Completed tokens per second: 3201.13', 11.05 completed samples/s, VALID."),
        "vendor", [SRC_ML50_L40S], conf=0.85, interconnect=["pcie-gen4"]))
# L40S: 1x IBM Cloud vLLM v5.1 (llama3.1-8b) — single-GPU datapoint
w(bench("mlperf-v5-1-ibm-1xl40s-vllm-llama31-8b-offline-tokens",
        "MLPerf v5.1 IBM Cloud gx3 1x L40S vLLM: llama3.1-8b Offline aggregate tokens/s",
        "vllm", ["nvidia-l40s"], "llama3.1-8b", None,
        "tps_aggregate", 1642.22, "tok/s (single L40S 48GB, Offline)",
        mlperf_meth("v5.1, IBM Cloud gx3-48x240x2l40s (only 1 of 2 L40S in test), "
                    "vLLM 0.10.0, RHEL 9.6, CUDA 12.9. Offline: 'Tokens per second: 1642.22'."),
        "vendor", [SRC_ML51_L40S], conf=0.85, interconnect=["pcie-gen4"]))
w(bench("mlperf-v5-1-ibm-1xl40s-vllm-llama31-8b-server-tokens",
        "MLPerf v5.1 IBM Cloud 1x L40S vLLM: llama3.1-8b Server completed tokens/s",
        "vllm", ["nvidia-l40s"], "llama3.1-8b", None,
        "tps_aggregate", 1207.14, "tok/s (single L40S, Server scenario)",
        mlperf_meth("v5.1, IBM Cloud 1x L40S, vLLM 0.10.0. Server: "
                    "'Completed tokens per second: 1207.14', 9.37 scheduled samples/s."),
        "vendor", [SRC_ML51_L40S], conf=0.85, interconnect=["pcie-gen4"]))
# RTX PRO 4500 Blackwell: MLPerf v6.1
w(bench("mlperf-v6-1-cisco-5xrtxpro4500-llama31-8b-offline-tokens",
        "MLPerf v6.1 Cisco C240M8 5x RTX PRO 4500 32GB: llama3.1-8b Offline aggregate tok/s",
        "tensorrt-llm", ["nvidia-rtx-pro-4500-blackwell"], "llama3.1-8b", None,
        "tps_aggregate", 12017.7, "tok/s AGGREGATE over 5x RTX PRO 4500",
        mlperf_meth("v6.1, Cisco UCS C240 M8, 5x RTX PRO 4500 Blackwell 32GB PCIe, TensorRT-LLM. "
                    "Offline: 'Tokens per second: 12017.7', VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"],
        notes="~2403 tok/s per 4500. Same system also submitted Interactive+Server."))
w(bench("mlperf-v6-1-dell-8xrtxpro4500-llama2-70b-offline-tokens",
        "MLPerf v6.1 Dell XE7745 8x RTX PRO 4500 32GB: llama2-70b-99.9 Offline aggregate tok/s",
        "tensorrt-llm", ["nvidia-rtx-pro-4500-blackwell"], "llama2-70b-99.9", None,
        "tps_aggregate", 4374.06, "tok/s AGGREGATE over 8x RTX PRO 4500",
        mlperf_meth("v6.1, Dell XE7745, 8x RTX PRO 4500 Blackwell 32GB, TensorRT-LLM. "
                    "Offline: 'Tokens per second: 4374.06', VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"],
        notes="~547 tok/s/GPU on 70B. Server: 3012.33 tok/s."))
w(bench("mlperf-v6-1-dell-8xrtxpro4500-llama2-70b-server-tokens",
        "MLPerf v6.1 Dell XE7745 8x RTX PRO 4500: llama2-70b-99.9 Server aggregate tok/s",
        "tensorrt-llm", ["nvidia-rtx-pro-4500-blackwell"], "llama2-70b-99.9", None,
        "tps_aggregate", 3012.33, "tok/s AGGREGATE over 8x RTX PRO 4500",
        mlperf_meth("v6.1, Dell XE7745 8x RTX PRO 4500. Server: "
                    "'Completed tokens per second: 3012.33'."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"]))
# H200-NVL v5.0 8x (covers nvidia-h200-nvl)
w(bench("mlperf-v5-0-dell-xe7745-8xh200nvl-llama2-70b-offline-tokens",
        "MLPerf v5.0 Dell XE7745 8x H200 NVL 141GB: llama2-70b-99.9 Offline aggregate tok/s",
        "tensorrt-llm", ["nvidia-h200-nvl"], "llama2-70b-99.9", None,
        "tps_aggregate", 31149.9, "tok/s AGGREGATE over 8x H200 NVL",
        mlperf_meth("v5.0, Dell XE7745, 8x NVIDIA H200 NVL 141GB PCIe (NVLink bridge pairs), "
                    "TensorRT-LLM. Offline: 'Tokens per second: 31149.9', VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["nvlink-4"],
        notes="~3894 tok/s/GPU. H200 NVL = PCIe form-factor with 4-way NVLink bridge."))

# ---------- kyuz0 Strix Halo toolboxes sweep (results.json, llama.cpp b9187) ----------
def kyuz0_meth(env):
    return ("kyuz0 amd-strix-halo-toolboxes automated llama-bench sweep (results.json, "
            "generated 2026-05-16, Fedora 43 kernel 6.19.12, llama.cpp build 9187 0253fb21f). "
            f"Environment: {env} on AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), "
            "-fa 1 -ngl 99 --mmap 0, default pp512/tg128 context. "
            "Per-request batch-1, NOT aggregate. mean +/- std over -r repeats.")

kyu = [
    # id-slug, model label, quant fmt, env, metric, value
    ("strix-8060s-qwen36-35b-a3b-q4kxl-rocm723-tg128", "Qwen3.6-35B-A3B-UD-Q4_K_XL", "gguf-q4-k-m",
     "rocm-7_2_3", "decode_tok_s", 51.53, "MoE 35B-A3B, ~3B active"),
    ("strix-8060s-qwen36-35b-a3b-q4kxl-vulkan-radv-tg128", "Qwen3.6-35B-A3B-UD-Q4_K_XL", "gguf-q4-k-m",
     "vulkan_radv", "decode_tok_s", 60.43, "MoE 35B-A3B, ~3B active. RADV beats ROCm on tg here."),
    ("strix-8060s-qwen36-35b-a3b-q8kxl-rocm723-tg128", "Qwen3.6-35B-A3B-UD-Q8_K_XL", "gguf-q8-0",
     "rocm-7_2_3", "decode_tok_s", 46.18, "MoE 35B-A3B"),
    ("strix-8060s-gpt-oss-120b-mxfp4-rocm723-tg128", "gpt-oss-120b-mxfp4", "gguf-mxfp4-moe",
     "rocm-7_2_3", "decode_tok_s", 51.81, "120B MoE, MXFP4"),
    ("strix-8060s-gpt-oss-120b-mxfp4-vulkan-radv-tg128", "gpt-oss-120b-mxfp4", "gguf-mxfp4-moe",
     "vulkan_radv", "decode_tok_s", 56.61, "120B MoE, MXFP4. RADV > ROCm on tg128."),
    ("strix-8060s-gpt-oss-120b-mxfp4-rocm723-pp512", "gpt-oss-120b-mxfp4", "gguf-mxfp4-moe",
     "rocm-7_2_3", "prefill_tok_s", 625.58, "120B MoE prefill"),
    ("strix-8060s-glm47-flash-q8kxl-vulkan-radv-tg128", "GLM-4.7-Flash-UD-Q8_K_XL", "gguf-q8-0",
     "vulkan_radv", "decode_tok_s", 41.17, "GLM-4.7-Flash ~30B MoE"),
    ("strix-8060s-gemma4-26b-a4b-q4kxl-rocm723-tg128", "gemma-4-26B-A4B-it-UD-Q4_K_XL", "gguf-q4-k-m",
     "rocm-7_2_3", "decode_tok_s", 48.46, "Gemma-4 26B-A4B MoE"),
    ("strix-8060s-qwen35-122b-a10b-q5kxl-rocm723-tg128", "Qwen3.5-122B-A10B-UD-Q5_K_XL", "gguf-q5-k-m",
     "rocm-7_2_3", "decode_tok_s", 20.12, "122B MoE ~10B active"),
]
for bid, model, fmt, env, metric, val, note in kyu:
    w(bench("kyuz0-" + bid,
            f"kyuz0 Strix Halo sweep: {model} {env} {metric} on Radeon 8060S (gfx1151)",
            "llama-cpp", ["amd-radeon-8060s-ryzen-ai-max"], model, fmt,
            metric, val, "tok/s " + ("(tg128, batch 1)" if metric=="decode_tok_s" else "(pp512, batch 512)"),
            kyuz0_meth(env), "third_party", [SRC_KYUZ0], conf=0.75, interconnect=["pcie-gen4"],
            notes=note + ". 554-run machine-readable sweep; values are per-request batch-1."))

# ---------- lhl strix-vs-spark: gpt-oss-120b MXFP4 tg32 at ctx depth ----------
def lhl_meth(backend, ctx):
    return (f"lhl/strix-halo-testing DGX Spark vs Strix Halo comparison (Framework Desktop, "
            f"Arch 6.17, all amd_iommu/tuned opts). llama.cpp build 6792, gpt-oss-120b MXFP4, "
            f"backend {backend}, tg32 at context depth {ctx}. Per-request batch-1. "
            f"DGX Spark column from ggerganov's #16578 sweep (build 6767, CUDA).")

for bid, env_lbl, val in [
    ("strix-8060s-gpt-oss-120b-vulkan-amdvlk-tg32-d0", "Vulkan AMDVLK 2025.Q2.1 (-ub 512)", 50.05),
    ("strix-8060s-gpt-oss-120b-vulkan-amdvlk-tg32-d32k", "Vulkan AMDVLK @ d32768", 31.54),
    ("strix-8060s-gpt-oss-120b-rocm-wmma-tg32-d0", "ROCm rocWMMA (TheRock nightly 7.10.0a20251017)", 46.56),
    ("strix-8060s-gpt-oss-120b-rocm-wmma-tg32-d32k", "ROCm rocWMMA @ d32768", 17.82),
    ("strix-8060s-gpt-oss-120b-rocm-tuned-tg32-d32k", "lhl tuned rocWMMA @ d32768", 36.43),
]:
    w(bench("lhl-" + bid,
            f"lhl Strix Halo: gpt-oss-120b MXFP4 tg32 on Radeon 8060S ({env_lbl})",
            "llama-cpp", ["amd-radeon-8060s-ryzen-ai-max"], "gpt-oss-120b-mxfp4",
            "gguf-mxfp4-moe", "decode_tok_s", val, "tok/s (tg32, batch 1)",
            lhl_meth(env_lbl.split(" @")[0], env_lbl.split("@ ")[-1] if "@ " in env_lbl else "0"),
            "third_party", [SRC_LHL], conf=0.75,
            notes="Tuned rocWMMA branch (lhl/llama.cpp rocm-wmma-tune) reaches 36.43 tg32 at d32768 vs 17.82 stock — the best known long-context Strix Halo config."))

# ---------- hardware-corner RTX 5090 (canonical, ctx-swept) ----------
def hc_meth(model, ctx):
    return (f"Hardware Corner (Allan Witt) llama-bench build 466c1911 on RTX 5090 32GB GDDR7 "
            f"(EPYC 7642 host, Ubuntu 24.04). {model} Q4_K_XL, -fa 1 -ngl 99, -p 1024/2048 -n 128 "
            f"at context depth {ctx}. Per-request batch-1, NOT aggregate.")

for bid, model, ctx, tg, pp in [
    ("hc-rtx5090-qwen3-8b-q4kxl-4k", "Qwen3 8B", "4k", 185.91, 10406.47),
    ("hc-rtx5090-qwen3-8b-q4kxl-32k", "Qwen3 8B", "32k", 111.91, 3687.76),
    ("hc-rtx5090-qwen3-14b-q4kxl-4k", "Qwen3 14B", "4k", 123.79, 6497.55),
    ("hc-rtx5090-qwen3-30b-a3b-q4kxl-4k", "Qwen3 30B-A3B (MoE)", "4k", 234.30, 6630.42),
    ("hc-rtx5090-qwen3-32b-q4kxl-4k", "Qwen3 32B dense", "4k", 61.38, 2931.32),
    ("hc-rtx5090-gpt-oss-20b-q4kxl-4k", "gpt-oss 20B", "4k", 298.2, 9443.81),
]:
    w(bench(bid + "-tg128",
            f"RTX 5090 llama.cpp: {model} Q4_K_XL decode at {ctx} ctx (Hardware Corner)",
            "llama-cpp", ["nvidia-rtx-5090"], model, "gguf-q4-k-m",
            "decode_tok_s", tg, "tok/s (tg128, batch 1, per-request)",
            hc_meth(model, ctx), "third_party", [SRC_HC5090], conf=0.7,
            interconnect=["pcie-gen5"],
            notes=f"PP at same ctx: {pp} t/s. Full ctx sweep 4k..131k in source."))
    w(bench(bid + "-pp1024",
            f"RTX 5090 llama.cpp: {model} Q4_K_XL prefill at {ctx} ctx (Hardware Corner)",
            "llama-cpp", ["nvidia-rtx-5090"], model, "gguf-q4-k-m",
            "prefill_tok_s", pp, "tok/s (pp~1024-2048 at ctx depth)",
            hc_meth(model, ctx), "third_party", [SRC_HC5090], conf=0.7,
            interconnect=["pcie-gen5"]))

# ---------- MLPerf MI350X / MI350P / MI325X / MI300X-1x ----------
w(bench("mlperf-v6-1-mitac-8xmi350x-llama2-70b-offline-tokens",
        "MLPerf v6.1 MiTAC 8x MI350X: llama2-70b-99.9 Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi350x"], "llama2-70b-99.9", None,
        "tps_aggregate", 84016.4, "tok/s AGGREGATE over 8x MI350X 288GB",
        mlperf_meth("v6.1, MiTAC, 8x AMD Instinct MI350X 288GB HBM3e + 2x EPYC 9755. "
                    "Offline: 'Tokens per second: 84016.4', VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["infinity-fabric"],
        notes="~10502 tok/s/GPU. MI350X = CDNA4, 288GB HBM3e."))
w(bench("mlperf-v6-1-mitac-8xmi350x-gpt-oss-120b-offline-tokens",
        "MLPerf v6.1 MiTAC 8x MI350X: gpt-oss-120b Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi350x"], "gpt-oss-120b", None,
        "tps_aggregate", 97019.7, "tok/s AGGREGATE over 8x MI350X",
        mlperf_meth("v6.1, MiTAC 8x MI350X. Offline gpt-oss-120b: 97019.7 tok/s, VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["infinity-fabric"]))
w(bench("mlperf-v6-1-dell-8xmi350p-llama2-70b-offline-tokens",
        "MLPerf v6.1 Dell XE7745 8x MI350P: llama2-70b-99.9 Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi350p"], "llama2-70b-99.9", None,
        "tps_aggregate", 42286.2, "tok/s AGGREGATE over 8x MI350P",
        mlperf_meth("v6.1, Dell XE7745, 8x AMD Instinct MI350P (PCIe, 144GB) + 2x EPYC 9455. "
                    "Offline: 42286.2 tok/s, VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"],
        notes="~5286 tok/s/GPU. MI350P is the PCIe air-cooled CDNA4 sibling."))
w(bench("mlperf-v6-1-dell-8xmi350p-llama31-8b-offline-tokens",
        "MLPerf v6.1 Dell XE7745 8x MI350P: llama3.1-8b Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi350p"], "llama3.1-8b", None,
        "tps_aggregate", 73643, "tok/s AGGREGATE over 8x MI350P",
        mlperf_meth("v6.1, Dell XE7745 8x MI350P. llama3.1-8b Offline: 73643 tok/s."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"]))
w(bench("mlperf-v6-1-dell-8xmi350p-gpt-oss-120b-offline-tokens",
        "MLPerf v6.1 Dell XE7745 8x MI350P: gpt-oss-120b Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi350p"], "gpt-oss-120b", None,
        "tps_aggregate", 48971.1, "tok/s AGGREGATE over 8x MI350P",
        mlperf_meth("v6.1, Dell XE7745 8x MI350P. gpt-oss-120b Offline: 48971.1 tok/s."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["pcie-gen5"]))
w(bench("mlperf-v5-1-mitac-8xmi325x-llama2-70b-offline-tokens",
        "MLPerf v5.1 MiTAC 8x MI325X: llama2-70b-99.9 Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi325x"], "llama2-70b-99.9", None,
        "tps_aggregate", 33703, "tok/s AGGREGATE over 8x MI325X 256GB",
        mlperf_meth("v5.1, MiTAC, 8x AMD Instinct MI325X 256GB HBM3e + 2x EPYC 9755. "
                    "Offline: 'Tokens per second: 33703', VALID."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["infinity-fabric"]))
w(bench("mlperf-v5-1-mitac-8xmi325x-mixtral-8x7b-offline-tokens",
        "MLPerf v5.1 MiTAC 8x MI325X: mixtral-8x7b Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi325x"], "mixtral-8x7b", None,
        "tps_aggregate", 67964.4, "tok/s AGGREGATE over 8x MI325X",
        mlperf_meth("v5.1, MiTAC 8x MI325X. mixtral-8x7b Offline: 67964.4 tok/s."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["infinity-fabric"]))
# MI300X single-GPU v4.1 row
w(bench("mlperf-v4-1-amd-1xmi300x-llama2-70b-offline-tokens",
        "MLPerf v4.1 AMD 1x MI300X: llama2-70b-99.9 Offline aggregate tok/s",
        "vllm", ["amd-instinct-mi300x"], "llama2-70b-99.9", None,
        "tps_aggregate", 3062.72, "tok/s (single MI300X 192GB)",
        mlperf_meth("v4.1, AMD reference system 1x MI300X 192GB + 2x EPYC 9374F, ROCm+vLLM. Offline: 3062.72 tok/s, VALID. SINGLE GPU."),
        "vendor", [SRC_ML61], conf=0.8, interconnect=["infinity-fabric"]))

# ---------- Atlas_Inference MLPerf v6.1 agentic SingleStream ----------
wsrc(source("mlperf-v6-1-atlas-inference-strix-halo-dgx-spark-qwen36-27b",
    "MLPerf v6.1 Atlas_Inference agentic SingleStream: Strix Halo + DGX Spark GB10 on Qwen3.6-27B-NVFP4",
    "https://github.com/mlcommons/inference_results_v6.1/tree/main/closed/Atlas_Inference/results",
    "MLCommons / Atlas_Inference", "benchmark",
    "Agentic-edge submission (BFCL function-calling harness): Strix Halo (gfx1151, ROCm core-7.13) wall 7108.6s, TTFT med 2713ms, TPOT med 47.3ms, 10.2 TPS; DGX Spark GB10 (sm_121a, CUDA 13.0) wall 3834.4s, TTFT med 1153.7ms, TPOT med 32.9ms, 20.08 TPS. Qwen3.6-27B NVFP4, 1007 turns, temp 0."))

w(bench("mlperf-v6-1-atlas-strix-halo-qwen36-27b-tpot",
        "MLPerf v6.1 Atlas_Inference Strix Halo (8060S): Qwen3.6-27B NVFP4 SingleStream median TPOT",
        "llama-cpp", ["amd-radeon-8060s-ryzen-ai-max"], "Qwen3.6-27B-NVFP4", "nvfp4",
        "itl_ms", 47.28, "ms median TPOT (agentic SingleStream)",
        "MLPerf v6.1 closed Atlas_Inference agentic submission: 1007 turns BFCL, single-stream, "
        "temp 0, wall 7108.6s. Strix Halo Ryzen AI Max+ (Radeon 8060S gfx1151), ROCm core-7.13, "
        "atlas engine stack @eabfa8f. TPOT median 47.28ms (~21.1 tok/s), TTFT median 2713ms, "
        "TPS 10.22. Per-request.",
        "third_party", ["mlperf-v6-1-atlas-inference-strix-halo-dgx-spark-qwen36-27b"],
        conf=0.75, interconnect=["pcie-gen4"],
        notes="Agentic single-stream: TPOT med 47.3ms ≈ 21.1 decode tok/s on a 27B dense NVFP4. BFCL accuracy 86.23."))
w(bench("mlperf-v6-1-atlas-strix-halo-qwen36-27b-ttft",
        "MLPerf v6.1 Atlas_Inference Strix Halo: Qwen3.6-27B NVFP4 SingleStream median TTFT",
        "llama-cpp", ["amd-radeon-8060s-ryzen-ai-max"], "Qwen3.6-27B-NVFP4", "nvfp4",
        "ttft_ms", 2713.5, "ms median TTFT",
        "Same run: TTFT median 2713.5ms (avg 3774ms, p90 6073ms).",
        "third_party", ["mlperf-v6-1-atlas-inference-strix-halo-dgx-spark-qwen36-27b"],
        conf=0.75, interconnect=["pcie-gen4"]))

# ---------- MLPerf v6.1 GB300 (8x Oracle) — nvidia-gb300 uncovered ----------
w(bench("mlperf-v6-1-oracle-8xgb300-llama2-70b-offline-tokens",
        "MLPerf v6.1 Oracle 8x GB300 (Blackwell Ultra): llama2-70b-99.9 Offline aggregate tok/s",
        "tensorrt-llm", ["nvidia-gb300"], "llama2-70b-99.9", None,
        "tps_aggregate", 132253, "tok/s AGGREGATE over 8x GB300 288GB (NVL72 partition)",
        mlperf_meth("v6.1, Oracle GB300-NVL72 aarch64x8 partition (8x GB300 288GB + Grace), "
                    "TensorRT-LLM. Offline: 'Tokens per second: 132253', VALID."),
        "vendor", [SRC_ML61], conf=0.85, interconnect=["nvlink-5","nvswitch-4"],
        notes="~16532 tok/s/GPU. Per-GPU record for the same part exists under nvidia-b300 submissions."))
w(bench("mlperf-v6-1-oracle-8xgb300-llama2-70b-server-tokens",
        "MLPerf v6.1 Oracle 8x GB300: llama2-70b-99.9 Server aggregate tok/s",
        "tensorrt-llm", ["nvidia-gb300"], "llama2-70b-99.9", None,
        "tps_aggregate", 127983.87, "tok/s AGGREGATE over 8x GB300",
        mlperf_meth("v6.1, Oracle 8x GB300. Server: 'Completed tokens per second: 127983.87'."),
        "vendor", [SRC_ML61], conf=0.85, interconnect=["nvlink-5","nvswitch-4"]))

# ---------- Vulkan scoreboard rows for uncovered accel ids ----------
def vk_meth(commit, comment, fa):
    return (f"COMMUNITY-SCOREBOARD (llama.cpp discussion #10879, netrunnereve-maintained). "
            f"llama-bench -m llama-2-7b.Q4_0.gguf -ngl 100 -fa {fa}, Vulkan backend. "
            f"Build commit {commit}. {comment} mean +/- std over default -r 5. "
            f"Per-request batch-1, NOT aggregate.")
# nvidia-a10: Vulkan board has no plain A10; comment-section A10G run (sm_86, 24GB, 498 GB/s) applies to the A10 family — record under nvidia-a10 with caveat.
w(bench("llamacpp-cuda-a10g-llama27b-q40-fa-tg128",
        "llama.cpp CUDA on NVIDIA A10G (A10 AWS variant): Llama-2-7B Q4_0 FA-on tg128 tok/s",
        "llama-cpp", ["nvidia-a10"], MODEL7B, "gguf-q4-0", "decode_tok_s", 111.51,
        "tok/s (tg128, per-request, batch 1)",
        "llama.cpp discussion #15013 comment by @Hastwell (2026-07-19): A10G custom SKU "
        "(sm_86, 24GB GDDR6, 498 GB/s), build 571d0d5, -ngl 99 -fa 1. tg128 111.51, "
        "pp512 2190.43. A10G is the AWS-sourced 300W 2-slot A10 derivative — same silicon "
        "and memory system as the A10, so the number applies to the A10 family with the "
        "caveat that thermals differ.",
        "third_party", [SRC_CUDA], conf=0.6, status="draft",
        notes="NOT the scoreboard-maintained row (comment data). A10G SKU is a custom variant — kept draft pending a plain-A10 run."))
w(bench("llamacpp-cuda-a10g-llama27b-q40-fa-pp512",
        "llama.cpp CUDA on NVIDIA A10G: Llama-2-7B Q4_0 FA-on pp512 tok/s",
        "llama-cpp", ["nvidia-a10"], MODEL7B, "gguf-q4-0", "prefill_tok_s", 2190.43,
        "tok/s (pp512, batch 512)", "Same @Hastwell A10G run, pp512.", "third_party",
        [SRC_CUDA], conf=0.6, status="draft"))

# ---------- Apple M3 Ultra via Vulkan (MoltenVK) — distinct path ----------
w(bench("llamacpp-vulkan-m3-ultra-llama27b-q40-nofa-tg128",
        "llama.cpp Vulkan/MoltenVK on Apple M3 Ultra: Llama-2-7B Q4_0 noFA tg128 decode tok/s",
        "llama-cpp", ["apple-m3-ultra"], MODEL7B, "gguf-q4-0", "decode_tok_s", 115.54,
        "tok/s (tg128, per-request, batch 1)",
        "COMMUNITY-SCOREBOARD (llama.cpp discussion #10879). llama-bench -ngl 100 -fa 0, "
        "Vulkan backend via MoltenVK on macOS. Build 2d451c8. Per-request batch-1.",
        "third_party", [SRC_VULKAN], conf=0.7,
        notes="Vulkan path hits 115.5 tg128 vs Metal 92.1 (same chip, #4167) — Vulkan/MoltenVK outperforms Metal here on Q4_0 decode."))
w(bench("llamacpp-vulkan-m3-ultra-llama27b-q40-nofa-pp512",
        "llama.cpp Vulkan/MoltenVK on Apple M3 Ultra: Llama-2-7B Q4_0 noFA pp512 tok/s",
        "llama-cpp", ["apple-m3-ultra"], MODEL7B, "gguf-q4-0", "prefill_tok_s", 1116.83,
        "tok/s (pp512)", "Same MoltenVK run: pp512 1116.83.",
        "third_party", [SRC_VULKAN], conf=0.7))
# Intel Arc Pro B70 Vulkan row (b70 covered but adds Vulkan datapoint)
w(bench("llamacpp-vulkan-arc-pro-b70-llama27b-q40-nofa-tg128",
        "llama.cpp Vulkan on Intel Arc Pro B70: Llama-2-7B Q4_0 noFA tg128 decode tok/s",
        "llama-cpp", ["intel-arc-pro-b70"], MODEL7B, "gguf-q4-0", "decode_tok_s", 112.02,
        "tok/s (tg128, per-request, batch 1)",
        "COMMUNITY-SCOREBOARD (#10879). llama-bench -ngl 100 -fa 0, Vulkan backend. "
        "Build b863507. Per-request batch-1.",
        "third_party", [SRC_VULKAN], conf=0.7,
        notes="Vulkan decode on B70 (112) is competitive with ROCm/CUDA mid-range cards on the same workload."))
