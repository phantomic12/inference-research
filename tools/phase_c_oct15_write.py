#!/usr/bin/env python3
"""Phase C Oct-15 tick: benchmark backfill for still-uncovered / thinly-covered
affordable-class accelerators. Writes JSON records directly (same shape as
tools/new_record.py output, fields filled).

Run from repo root: python3 tools/phase_c_oct15_write.py
Idempotent: skips ids that already exist.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = "2026-10-15"

def path(*p): return os.path.join(REPO, *p)

def write(rec, subdir):
    p = path("data", subdir, rec["id"] + ".json")
    if os.path.exists(p):
        print("SKIP (exists)", rec["id"]); return False
    with open(p, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("WROTE", p)
    return True

def source(rid, name, url, publisher, kind, notes=""):
    return {"id": rid, "type": "source", "name": name, "status": "verified",
            "confidence": 0.85, "updated": TODAY, "url": url,
            "publisher": publisher, "kind": kind, "published": None,
            "accessed": TODAY, "archived_url": None, "notes": notes}

def bench(rid, name, engine_id, acc, metric, value, unit, methodology,
          measured_by, sources, model=None, format_id=None, interconnect=None,
          reproducible=None, notes="", status="verified", confidence=0.8):
    return {"id": rid, "type": "benchmark", "name": name, "status": status,
            "confidence": confidence, "updated": TODAY, "sources": sources,
            "engine_id": engine_id, "accelerator_ids": acc,
            "interconnect_ids": interconnect or [], "model": model,
            "format_id": format_id, "metric": metric, "value": value,
            "unit": unit, "methodology": methodology,
            "measured_by": measured_by, "reproducible": reproducible,
            "energy_basis": None, "power_w": None, "power_scope": None,
            "power_cap": None, "clock_lock_mhz": None, "thermal_state": None,
            "notes": notes}

SRCS = [
    source("llmcheck-net-benchmarks-csv",
           "LLMCheck Apple Silicon LLM benchmark dataset (CSV export)",
           "https://llmcheck.net/data/benchmarks.csv",
           "llmcheck.net", "benchmark",
           "Open dataset of measured tok/s + TTFT on Apple Silicon Macs (M1..M6, "
           "Pro/Max/Ultra), 259 rows. Each row: model, params, quant, chip, RAM GB, "
           "engine (Ollama/MLX/LM Studio/llama.cpp), tps, ttft s, test date. "
           "Site labels each figure sourced/community/estimated; the CSV rows are "
           "the machine-readable form. Decode = avg tps for a chat-style generation."),
    source("aws-neuron-inf1-inference-performance",
           "AWS Neuron documentation: Inf1 Inference Performance tables",
           "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/benchmarks/inf1/index.html",
           "AWS", "benchmark",
           "Vendor reference performance for Inf1 (Inferentia gen-1) instances: "
           "encoder (BERT/DistilBERT/DistilRoBERTa), CNN (ResNet-50, YOLOv4, "
           "OpenPose) tables. NeuronPerf-measured, Neuron SDK 2.12-2.20, fp32/fp16, "
           "batch + real-time rows, seq len 128. Last update 2024-09-16."),
    source("aws-neuron-trn1-inference-performance",
           "AWS Neuron documentation: Trn1/Trn1n Inference Performance tables",
           "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/trn1/trn1-inference-performance.html",
           "AWS", "benchmark",
           "Vendor reference performance for Trn1 (Trainium gen-1) instances: "
           "encoder models (BERT/RoBERTa/ALBERT/ELECTRA/XLM-R) + encoder-decoder "
           "text generation (flan-t5-xl, t5-3b) tables. Neuron SDK 2.17-2.28, "
           "NeuronX Distributed, tensor-parallel degree 8 on trn1.32xlarge, "
           "FP32 model dtype autocast to Matmult-BF16. Last update 2026-02-26."),
    source("markaicode-vllm-vs-ollama-guidellm-a100-pcie",
           "Markaicode: vLLM vs Ollama benchmark — Red Hat GuideLLM study on A100-PCIE-40GB",
           "https://markaicode.com/benchmarks/cuda-qwen-benchmark/",
           "Markaicode (citing Red Hat Developer GuideLLM study, Aug 2025)",
           "benchmark",
           "Republishes Red Hat's GuideLLM v0.2.1 study with full test environment: "
           "single NVIDIA A100-PCIE-40GB, driver 550.144.03, CUDA 12.4, OpenShift "
           "4.17.15, vLLM 0.9.1 vs Ollama 0.9.2, Llama-3.1-8B-Instruct fp16, "
           "concurrency 1..256, 300 s per level. Peak: vLLM 793 output tok/s vs "
           "Ollama 41 tok/s (OLLAMA_NUM_PARALLEL=32 tuned); P99 TTFT 80 vs 673 ms."),
    source("aws-trn3-ultraservers-whats-new-dec2025",
           "AWS What's New: EC2 Trn3 UltraServers GA (Trainium3)",
           "https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/",
           "AWS", "vendor-blog",
           "GA announcement for Trn3 UltraServers (Dec 2 2025). Per-chip: 2.52 PF "
           "FP8, 144 GB HBM3e, 4.9 TB/s. Vendor-relative claims: Trn3 delivers up "
           "to 4.4x higher performance / 4x better perf-per-watt vs Trn2 "
           "UltraServer; on Bedrock Trainium3 is 'up to 3x faster than Trainium2 "
           "with over 5x higher output tokens per megawatt at similar latency "
           "per user'. Ratios, not absolute tok/s."),
]

RECS = []

# ---------------- AWS Inferentia gen-1 (aws-inferentia1): encoder + CNN ------
# These are inference/sec and tokens/sec on non-decode models; the metric enum
# has tps_aggregate (aggregate throughput) which is the right home.
RECS += [
    bench("aws-inf1-bert-base-batch128-tput",
          "AWS Neuron Inf1: BERT-base-uncased fp32, inf1.xlarge, batch 6, throughput-optimized",
          "alt-neuron-compiler", ["aws-inferentia1"], "tps_aggregate", 1181.0,
          "inferences/s (batch-throughput-optimized, seq len 128)",
          "VENDOR MEASURED (AWS NeuronPerf, Neuron SDK 2.20.0). inf1.xlarge = 1 "
          "Inferentia chip (4 NeuronCores). bert-base-uncased FP32, sequence "
          "length 128, batch size 6, data-parallel run mode, 'Batch' application "
          "type (max throughput). Latency P50 41 ms / P99 45 ms. AGGREGATE across "
          "the 4 NeuronCores; not a decode tok/s — BERT is an encoder, so the "
          "figure is complete inferences/s.",
          "vendor", ["aws-neuron-inf1-inference-performance"],
          model="bert-base-uncased", format_id=None,
          reproducible=True,
          notes="Throughput-optimized row from AWS's Inf1 table. The companion "
                "latency-optimized run of the same model gives 285 inf/s at "
                "batch 3 (P50 10 ms) — recorded as a separate record. This is "
                "the strongest fp32 encoder figure AWS publishes for a single "
                "Inferentia gen-1 chip and the only measured throughput row "
                "the repo has for this part."),
    bench("aws-inf1-bert-base-lat-opt-tput",
          "AWS Neuron Inf1: BERT-base-uncased fp32, inf1.xlarge, batch 3, latency-optimized",
          "alt-neuron-compiler", ["aws-inferentia1"], "tps_aggregate", 285.0,
          "inferences/s (latency-optimized, seq len 128)",
          "VENDOR MEASURED (AWS NeuronPerf, Neuron SDK 2.20.0). inf1.xlarge, "
          "bert-base-uncased FP32, seq len 128, batch 3, data-parallel, 'Real "
          "Time' application type (min-latency). Latency P50 10 ms / P99 11 ms. "
          "Aggregate across 4 NeuronCores.",
          "vendor", ["aws-neuron-inf1-inference-performance"],
          model="bert-base-uncased", format_id=None,
          reproducible=True,
          notes="Latency-optimized row; the batch-optimized counterpart "
                "achieves 1181 inf/s at batch 6 — the ~4x throughput-vs-latency "
                "tradeoff is characteristic of the part. Records the same "
                "model/quant/config axis as the throughput row."),
    bench("aws-inf1-resnet50-fp16-batch10-tput",
          "AWS Neuron Inf1: ResNet-50 fp16, inf1.xlarge, batch 10, throughput-optimized",
          "alt-neuron-compiler", ["aws-inferentia1"], "tps_aggregate", 2207.0,
          "inferences/s (batch-throughput-optimized)",
          "VENDOR MEASURED (AWS Neuron SDK tutorial numbers, Neuron 2.12.0). "
          "inf1.xlarge, ResNet-50 FP16, batch 10, data-parallel, 'Batch' "
          "application. Latency P50 18 ms / P99 23 ms.",
          "vendor", ["aws-neuron-inf1-inference-performance"],
          model="ResNet-50", format_id=None,
          reproducible=True,
          notes="CV-model row: the Inf1 chip's headline fp16 number. Not an "
                "LLM figure — kept because the accelerator record existed with "
                "zero measurements of any kind and the Neuron table is the "
                "primary public source for the part."),
]

# ---------------- AWS Trainium gen-1 (aws-trainium1) -------------------------
RECS += [
    bench("aws-trn1-flan-t5-xl-tp8-b1-128io-tput",
          "AWS Neuron Trn1: flan-t5-xl, trn1.32xlarge TP8, batch 1, 128in/84out, throughput-optimized",
          "alt-neuron-compiler", ["aws-trainium1"], "tps_aggregate", 122.52,
          "tokens/s (input+output counted, TP8 across 16 Trainium chips)",
          "VENDOR MEASURED (AWS Neuron, Neuron SDK 2.17.0, NeuronX Distributed). "
          "trn1.32xlarge = 16 Trainium chips; run as tensor-parallel degree 8, "
          "data-parallel degree 1, batch 1, sequence 128, input 128 / output 84 "
          "tokens, flan-t5-xl FP32 model dtype autocast to Matmult-BF16. "
          "Latency per token P50 8.16 ms / P99 8.19 ms. The table notes "
          "throughput counts BOTH input and output tokens.",
          "vendor", ["aws-neuron-trn1-inference-performance"],
          model="google/flan-t5-xl", format_id=None,
          reproducible=True,
          notes="First measured generative-model token figure for "
                "aws-trainium1. Throughput counts input+output tokens (T5 "
                "encoder-decoder), so it is not directly comparable to a "
                "decode-only tok/s on a decoder-only model. ~123 tok/s on a "
                "3B encoder-decoder across 16 chips illustrates that gen-1 "
                "Trainium's cost story is throughput/$ at batch, not single-"
                "request decode speed."),
    bench("aws-trn1-t5-3b-tp8-b1-128io-tput",
          "AWS Neuron Trn1: t5-3b, trn1.32xlarge TP8, batch 1, 128in/84out, throughput-optimized",
          "alt-neuron-compiler", ["aws-trainium1"], "tps_aggregate", 116.29,
          "tokens/s (input+output counted, TP8 across 16 Trainium chips)",
          "VENDOR MEASURED (AWS Neuron, Neuron SDK 2.17.0, NeuronX Distributed). "
          "trn1.32xlarge TP8/DP1, batch 1, seq 128, in 128 / out 84, t5-3b FP32 "
          "autocast Matmult-BF16. Latency per token P50 8.58 ms / P99 8.66 ms.",
          "vendor", ["aws-neuron-trn1-inference-performance"],
          model="t5-3b", format_id=None,
          reproducible=True,
          notes="Companion T5 row. The latency-optimized variant of the same "
                "config measures 110.23 tok/s (P50 9.07 ms) — the throughput/"
                "latency spread is small because batch is already 1."),
    bench("aws-trn1-bert-large-batch4-tput",
          "AWS Neuron Trn1: bert-large-uncased fp32, trn1.2xlarge, batch 4, throughput-optimized",
          "alt-neuron-compiler", ["aws-trainium1"], "tps_aggregate", 1104.44,
          "inferences/s (batch-throughput-optimized, seq len 128)",
          "VENDOR MEASURED (AWS NeuronPerf, Neuron SDK 2.27.0). trn1.2xlarge = "
          "1 Trainium chip. bert-large-uncased FP32 autocast Matmult-BF16, seq "
          "128, batch 4, data-parallel, 'Batch' application. Latency P50 7.24 "
          "ms / P99 7.3 ms.",
          "vendor", ["aws-neuron-trn1-inference-performance"],
          model="bert-large-uncased", format_id=None,
          reproducible=True,
          notes="Encoder throughput row — the strongest single-chip number "
                "AWS publishes for gen-1 Trainium and the direct comparison "
                "point to the Inf1 record for the same model class."),
]

# ---------------- AWS Inferentia2 (aws-inferentia2): add an Inf2 row ---------
# (record exists; add the bert-large + flan-t5-xl measured rows for depth)
RECS += [
    bench("aws-inf2-flan-t5-xl-tp8-b1-128io-tput",
          "AWS Neuron Inf2: flan-t5-xl, inf2.24xlarge TP8, batch 1, 128in/84out, throughput-optimized",
          "alt-neuron-compiler", ["aws-inferentia2"], "tps_aggregate", 117.61,
          "tokens/s (input+output counted, TP8)",
          "VENDOR MEASURED (AWS Neuron, Neuron SDK 2.17.0, NeuronX Distributed). "
          "inf2.24xlarge (Inferentia2), TP8/DP1, batch 1, seq 128, in 128/out 84, "
          "flan-t5-xl FP32 autocast Matmult-BF16. Latency per token P50 8.51 "
          "ms / P99 8.53 ms.",
          "vendor", ["aws-neuron-trn1-inference-performance"],
          model="google/flan-t5-xl", format_id=None,
          reproducible=True,
          notes="Gives aws-inferentia2 a second benchmark record alongside the "
                "existing fp16 alt-aws-inf2 row, and enables a same-model "
                "Trainium1-vs-Inferentia2 comparison (122.5 vs 117.6 tok/s)."),
    bench("aws-inf2-bert-large-batch4-tput",
          "AWS Neuron Inf2: bert-large-uncased fp32, inf2.xlarge, batch 4, throughput-optimized",
          "alt-neuron-compiler", ["aws-inferentia2"], "tps_aggregate", 950.05,
          "inferences/s (batch-throughput-optimized, seq len 128)",
          "VENDOR MEASURED (AWS NeuronPerf, Neuron SDK 2.21.0). inf2.xlarge, "
          "bert-large-uncased FP32 autocast Matmult-BF16, seq 128, batch 4, "
          "data-parallel, 'Batch' application. Latency P50 8.41 ms / P99 8.85 ms.",
          "vendor", ["aws-neuron-trn1-inference-performance"],
          model="bert-large-uncased", format_id=None,
          reproducible=True,
          notes="Direct gen-over-gen comparison to aws-trainium1's "
                "bert-large row: Trn1 hits 1104 inf/s on the same model/batch "
                "config where Inf2 manages 950 — consistent with Trainium "
                "being the higher-compute sibling."),
]

# ---------------- NVIDIA A100 40GB PCIe (nvidia-a100-40gb-pcie) --------------
RECS += [
    bench("guidellm-a100-pcie-40gb-llama31-8b-fp16-vllm-peak-tps",
          "GuideLLM (Red Hat): A100-PCIE-40GB, vLLM 0.9.1, Llama-3.1-8B fp16, peak aggregate output tok/s",
          "vllm", ["nvidia-a100-40gb-pcie"], "tps_aggregate", 793.0,
          "output tok/s (aggregate at saturation, concurrency up to 256)",
          "THIRD-PARTY LAB MEASURED (Red Hat GuideLLM v0.2.1 study, Aug 2025; "
          "republished by Markaicode). Single NVIDIA A100-PCIE-40GB, driver "
          "550.144.03, CUDA 12.4, OpenShift 4.17.15. vLLM 0.9.1 serving "
          "meta-llama/Llama-3.1-8B-Instruct FP16. Concurrency ramped 1..256, "
          "300 s per level, fixed prompt-response dataset. Reported peak "
          "output TPS at saturation; P99 TTFT at peak = 80 ms. AGGREGATE "
          "across concurrent requests, NOT per-request decode.",
          "third_party", ["markaicode-vllm-vs-ollama-guidellm-a100-pcie"],
          model="meta-llama/Llama-3.1-8B-Instruct", format_id=None,
          reproducible=True,
          notes="The aggregate number for the part. vLLM's continuous batching "
                "is what produces the 793 figure — per-request decode on the "
                "same card is in the ~30-45 tok/s band. Companion Ollama "
                "record shows the contrast. This is the only PCIe-40GB "
                "measurement the repo has; the two A100-80GB-SXM4 records are "
                "different silicon clocks (400 W vs 250 W cap)."),
    bench("guidellm-a100-pcie-40gb-llama31-8b-fp16-ollama-peak-tps",
          "GuideLLM (Red Hat): A100-PCIE-40GB, Ollama 0.9.2 (NUM_PARALLEL=32), Llama-3.1-8B fp16, peak aggregate output tok/s",
          "ollama", ["nvidia-a100-40gb-pcie"], "tps_aggregate", 41.0,
          "output tok/s (aggregate at saturation, concurrency up to 256)",
          "THIRD-PARTY LAB MEASURED (Red Hat GuideLLM v0.2.1 study, Aug 2025; "
          "republished by Markaicode). Same host as the vLLM row: single "
          "A100-PCIE-40GB, CUDA 12.4, OpenShift 4.17.15. Ollama 0.9.2 with "
          "OLLAMA_NUM_PARALLEL=32 (the highest stable value on this card). "
          "llama3.1:8b-instruct-fp16. Peak output TPS at saturation; P99 TTFT "
          "at peak = 673 ms. AGGREGATE.",
          "third_party", ["markaicode-vllm-vs-ollama-guidellm-a100-pcie"],
          model="meta-llama/Llama-3.1-8B-Instruct", format_id=None,
          reproducible=True,
          notes="Same-card, same-model, same-harness counterpart to the vLLM "
                "793 tok/s record — the ~19x gap is a scheduler (continuous "
                "batching vs bounded parallel slots) story, not a hardware "
                "capability difference. Do NOT read 41 tok/s as the card's "
                "decode ceiling."),
]

# ---------------- AWS Trainium3 (aws-trainium3): vendor ratios only ----------
RECS += [
    bench("aws-trn3-vs-trn2-perf-ratio-vendor",
          "Trainium3 vs Trainium2 vendor-claimed performance ratio (Trn3 vs Trn2 UltraServer)",
          "alt-neuron-compiler", ["aws-trainium3"], "speedup_ratio", 4.4,
          "x faster than Trainium2 UltraServer (vendor claim, workload unstated)",
          "VENDOR CLAIM (AWS What's New, Trn3 UltraServer GA, Dec 2 2025). "
          "'Trn3 delivers up to 4.4x higher performance... compared to our "
          "Trn2 UltraServers.' Baseline named in unit field. No absolute tok/s "
          "published; workload, model, and quant unstated.",
          "vendor", ["aws-trn3-ultraservers-whats-new-dec2025"],
          model=None, format_id=None,
          reproducible=False, status="contested", confidence=0.55,
          notes="Vendor marketing ratio, not a measurement. Recorded as "
                "speedup_ratio per the schema's rule that a bare relative "
                "number MUST name its baseline in unit. 'Up to' makes the "
                "real factor a range, not a point."),
    bench("aws-trn3-vs-trn2-bedrock-tokens-per-mw-ratio",
          "Trainium3 vs Trainium2 on Amazon Bedrock: output tokens per megawatt ratio",
          "alt-neuron-compiler", ["aws-trainium3"], "dimensionless_ratio", 5.0,
          "x higher output tokens per megawatt at similar latency per user (vendor claim)",
          "VENDOR CLAIM (AWS What's New, Trn3 UltraServer GA, Dec 2 2025). "
          "'On Amazon Bedrock, Trainium3 is our fastest accelerator, "
          "delivering... over 5x higher output tokens per megawatt at similar "
          "latency per user' vs Trainium2. Energy-normalized throughput ratio, "
          "not a speedup.",
          "vendor", ["aws-trn3-ultraservers-whats-new-dec2025"],
          model=None, format_id=None,
          reproducible=False, status="contested", confidence=0.5,
          notes="Filed as dimensionless_ratio (not speedup_ratio) because it "
                "is a throughput-per-watt figure, not a faster/slower time "
                "ratio. The only energy-normalized number for the part."),
]

# ---------------- Apple M5 Max (local-apple-m5-max): llmcheck measured -------
M5MAX = "local-apple-m5-max"
RECS += [
    bench("llmcheck-m5-max-llama33-70b-q4km-ollama-decode",
          "LLMCheck: Apple M5 Max (128GB), Ollama, Llama 3.3 70B Q4_K_M decode",
          "ollama", [M5MAX], "decode_tok_s", 12.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset, 2026-03..09 rows). "
          "Apple M5 Max 128GB unified memory, Ollama engine, Llama 3.3 70B "
          "Q4_K_M. avg tok/s for a chat-style generation; TTFT 2.8 s. "
          "Per-request decode, batch 1.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Llama 3.3 70B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Replaces the previous estimate-band figure (~15-25 tok/s) "
                "with a measured one. Matches the M5 Max bandwidth story "
                "(614 GB/s on a ~35 GB Q4 weight read -> low-teens decode)."),
    bench("llmcheck-m5-max-llama33-70b-q4km-mlx-decode",
          "LLMCheck: Apple M5 Max (128GB), MLX, Llama 3.3 70B Q4_K_M decode",
          "mlx-lm", [M5MAX], "decode_tok_s", 12.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max 128GB, MLX "
          "engine, Llama 3.3 70B Q4_K_M, TTFT 2.4 s, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Llama 3.3 70B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="MLX vs Ollama lands at the same 12 tok/s on this model — "
                "decode on a dense 70B at Q4 is pure bandwidth, so the engine "
                "difference is noise-level."),
    bench("llmcheck-m5-max-qwen35-35b-a3b-mlx-decode",
          "LLMCheck: Apple M5 Max (128GB), MLX, Qwen3.5-35B-A3B Q4_K_M decode",
          "mlx-lm", [M5MAX], "decode_tok_s", 52.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max, MLX, "
          "Qwen3.5-35B-A3B (MoE, ~3B active) Q4_K_M, TTFT 0.9 s, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3.5-35B-A3B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="The MoE point the user cares about: ~4.3x the dense-70B "
                "decode on the same chip. Qwen3.6-35B-A3B lands at 55 tok/s "
                "(separate record)."),
    bench("llmcheck-m5-max-qwen36-35b-a3b-mlx-decode",
          "LLMCheck: Apple M5 Max (128GB), MLX, Qwen3.6-35B-A3B Q4_K_M decode",
          "mlx-lm", [M5MAX], "decode_tok_s", 55.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max, MLX, "
          "Qwen3.6-35B-A3B Q4_K_M, TTFT 0.6 s, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3.6-35B-A3B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="The repo's target-class model on the target-class chip. "
                "Ollama run of the same model/quant gives 48 tok/s — the "
                "MLX-vs-Ollama spread on an MoE is ~13%."),
    bench("llmcheck-m5-max-qwen36-35b-a3b-ollama-decode",
          "LLMCheck: Apple M5 Max (64GB), Ollama, Qwen3.6-35B-A3B Q4_K_M decode",
          "ollama", [M5MAX], "decode_tok_s", 48.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max 64GB, Ollama, "
          "Qwen3.6-35B-A3B Q4_K_M, TTFT 0.8 s, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3.6-35B-A3B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Lower-RAM (64GB) M5 Max config; same ballpark as the 128GB "
                "MLX run, confirming the figure is bandwidth-bound not "
                "capacity-bound at this model size."),
    bench("llmcheck-m5-max-glm45-air-mlx-decode",
          "LLMCheck: Apple M5 Max, MLX, GLM-4.5-Air 106B-A?B Q4_K_M decode",
          "mlx-lm", [M5MAX], "decode_tok_s", 34.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max, MLX, "
          "GLM-4.5-Air (106B MoE) Q4_K_M, TTFT 0.9 s, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="GLM-4.5-Air", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Second large-MoE data point; the repo's GLM-4.6-Air target "
                "model is in the same class."),
    bench("llmcheck-m5-max-qwen3-235b-mlx-decode",
          "LLMCheck: Apple M5 Max (128GB), MLX, Qwen3-235B-A22B Q4_K_M decode",
          "mlx-lm", [M5MAX], "decode_tok_s", 18.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Max 128GB, MLX, "
          "Qwen3-235B-A22B (22B active) Q4_K_M, TTFT 1.8 s, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3-235B-A22B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="A 235B MoE fits in 128GB and still decodes at reading speed "
                "— the capacity-bandwidth trade the Max form factor sells."),
]

# ---------------- Apple M5 Ultra (local-apple-m5-ultra) ----------------------
M5U = "local-apple-m5-ultra"
RECS += [
    bench("llmcheck-m5-ultra-llama33-70b-q4km-mlx-decode",
          "LLMCheck: Apple M5 Ultra (96GB), MLX, Llama 3.3 70B Q4_K_M decode",
          "mlx-lm", [M5U], "decode_tok_s", 24.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset, dated 2026-08). M5 "
          "Ultra 96GB, MLX, Llama 3.3 70B Q4_K_M, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Llama 3.3 70B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Roughly 2x the M5 Max 70B figure (24 vs 12 tok/s) — tracks "
                "the 1.2 TB/s vs 614 GB/s bandwidth ratio almost exactly, "
                "textbook bandwidth-bound decode."),
    bench("llmcheck-m5-ultra-qwen3-235b-mlx-decode",
          "LLMCheck: Apple M5 Ultra (256GB), MLX, Qwen3-235B-A22B Q4_K_M decode",
          "mlx-lm", [M5U], "decode_tok_s", 37.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset, dated 2026-08). M5 "
          "Ultra 256GB, MLX, Qwen3-235B-A22B Q4_K_M, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3-235B-A22B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="37 tok/s on a 235B MoE is the Ultra's reason to exist: "
                "double the Max's 18 tok/s on the same model, comfortably "
                "interactive."),
    bench("llmcheck-m5-ultra-llama31-405b-q4km-mlx-decode",
          "LLMCheck: Apple M5 Ultra (512GB), MLX, Llama 3.1 405B Q4_K_M decode",
          "mlx-lm", [M5U], "decode_tok_s", 4.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset, dated 2026-08). M5 "
          "Ultra 512GB, MLX, Llama 3.1 405B Q4_K_M, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Llama 3.1 405B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="The capacity headline: a 405B dense model runs at all on a "
                "desktop chip, at ~4 tok/s (batch-offline speed). Only the "
                "512GB config has the headroom."),
    bench("llmcheck-m5-ultra-qwen3-coder-next-mlx-decode",
          "LLMCheck: Apple M5 Ultra (96GB), MLX, Qwen3-Coder-Next 80B Q4_K_M decode",
          "mlx-lm", [M5U], "decode_tok_s", 72.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset, dated 2026-08). M5 "
          "Ultra 96GB, MLX, Qwen3-Coder-Next (80B MoE) Q4_K_M, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3-Coder-Next", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="An 80B MoE decodes at 72 tok/s — well into interactive "
                "territory on unified memory."),
]

# ---------------- Apple M5 Pro (local-apple-m5-pro) --------------------------
RECS += [
    bench("llmcheck-m5-pro-gemma4-26b-q4km-ollama-decode",
          "LLMCheck: Apple M5 Pro (24GB), Ollama, Gemma 4 26B-A4B Q4_K_M decode",
          "ollama", ["local-apple-m5-pro"], "decode_tok_s", 35.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M5 Pro 24GB, Ollama, "
          "Gemma 4 26B-A4B Q4_K_M, TTFT 0.8 s, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Gemma 4 26B-A4B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Mid-tier Apple chip on a 26B MoE. Complements the existing "
                "llama.cpp M5 Pro rows (which cover Llama-2-7B only)."),
]

# ---------------- Apple M3 Ultra (apple-m3-ultra): llmcheck GLM row -----------
RECS += [
    bench("llmcheck-m3-ultra-glm5-753b-iq1s-lmstudio-decode",
          "LLMCheck: Apple M3 Ultra (512GB), LM Studio, GLM 5.2 753B-A40B IQ1_S decode",
          "llama-cpp", ["apple-m3-ultra"], "decode_tok_s", 22.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M3 Ultra 512GB, LM "
          "Studio, GLM 5.2 753B-A40B at IQ1_S (extreme ~1.7-bit quant), TTFT "
          "1.5 s, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="GLM 5.2 753B-A40B", format_id=None,
          reproducible=True,
          notes="Extreme-quant data point: a 753B MoE only fits at IQ1_S on "
                "512GB, and still decodes at reading speed (22 tok/s). The "
                "Q4_K_M MLX row for the same model is 15 tok/s — recorded "
                "separately. Note quant IQ1_S has no dedicated quantization "
                "record in the repo."),
    bench("llmcheck-m3-ultra-glm5-753b-q4km-mlx-decode",
          "LLMCheck: Apple M3 Ultra (512GB), MLX, GLM 5.2 753B-A40B Q4_K_M decode",
          "mlx-lm", ["apple-m3-ultra"], "decode_tok_s", 15.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M3 Ultra 512GB, MLX, "
          "GLM 5.2 753B-A40B Q4_K_M, TTFT 2.0 s, per-request.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="GLM 5.2 753B-A40B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Q4_K_M row: 753B at 4-bit needs ~400GB for weights alone — "
                "only the 512GB SKU has the headroom, and decode is ~15 "
                "tok/s. The IQ1_S row (22 tok/s) shows the quant-vs-speed "
                "lever."),
]

# ---------------- Apple M4 Max (apple-m4-max): llmcheck rows ------------------
RECS += [
    bench("llmcheck-m4-max-qwen36-35b-a3b-mlx-decode",
          "LLMCheck: Apple M4 Max, MLX, Qwen3.6-35B-A3B Q4_K_M decode",
          "mlx-lm", ["apple-m4-max"], "decode_tok_s", 48.0,
          "tok/s (decode, per-request)",
          "THIRD-PARTY MEASURED (LLMCheck open dataset). M4 Max, MLX, "
          "Qwen3.6-35B-A3B Q4_K_M, per-request decode.",
          "third_party", ["llmcheck-net-benchmarks-csv"],
          model="Qwen3.6-35B-A3B", format_id="gguf-q4-k-m",
          reproducible=True,
          notes="Gives apple-m4-max a MoE data point on the target model "
                "class; ~13% slower than M5 Max on the same model/quant, "
                "consistent with 546 vs 614 GB/s bandwidth."),
]

# ---------------- AMD EPYC 9554 (amd-epyc-9554-genoa): llmcheck EPYC row ------
# llmcheck is Apple-only; use the existing ahelpme record's companion numbers.
# Already covered (1 record); skip new EPYC rows — the measured EPYC numbers
# in the repo are adequate and additional web hits were estimates.

for s in SRCS:
    write(s, "sources")
for r in RECS:
    write(r, "benchmarks")
print(f"\nDone: {len(SRCS)} sources, {len(RECS)} benchmarks.")
