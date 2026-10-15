#!/usr/bin/env python3
"""Phase C Oct-15 p3: backfill uncovered/near-uncovered accelerators on main."""
import json, os, sys

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-15"

def w(rec, subdir="benchmarks"):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP (exists):", rec["id"]); return
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

def bench(id, name, engine, accs, model, fmt, metric, value, unit, meth, measured_by, sources,
          status="verified", conf=0.8, repro=True, interconnect=None, notes="",
          energy_basis=None, power_w=None, power_scope=None, thermal=None):
    return {
        "id": id, "type": "benchmark", "name": name,
        "engine_id": engine, "accelerator_ids": accs,
        "interconnect_ids": interconnect or [],
        "model": model, "format_id": fmt, "metric": metric, "value": value,
        "unit": unit, "methodology": meth, "measured_by": measured_by,
        "reproducible": repro, "energy_basis": energy_basis, "power_w": power_w,
        "power_scope": power_scope, "power_cap": None, "clock_lock_mhz": None,
        "thermal_state": thermal, "notes": notes,
        "status": status, "confidence": conf, "updated": TODAY,
        "sources": sources,
    }

def source(id, name, url, publisher, kind, notes, status="verified", conf=0.5, published=None):
    return {"id": id, "type": "source", "name": name, "status": status,
            "confidence": conf, "updated": TODAY, "url": url, "publisher": publisher,
            "kind": kind, "published": published, "accessed": TODAY, "archived_url": None,
            "notes": notes}

def wsrc(rec):
    w(rec, subdir="sources")

# ---------------- new sources ----------------
wsrc(source(
    "qwen-docs-h20-speed-benchmark",
    "Qwen docs — Speed Benchmark: Qwen3 family on NVIDIA H20 96GB (SGLang + HF Transformers)",
    "https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html",
    "Qwen Team / Alibaba", "benchmark",
    "Vendor benchmark page. Hardware: NVIDIA H20 96GB. Two stacks: (a) SGLang 0.4.6.post1 + sgl-kernel 0.1.0 + PyTorch 2.6.0+cu124 + vLLM 0.7.2 (AWQ-marlin), context_length=140000, enable_mixed_chunk=True, mem_fraction_static=0.85, skip_tokenizer_init with input_ids; (b) HF Transformers 4.51.3 + FlashAttention 2.7.4 + PyTorch 2.6.0, GPTQModel 2.2.0 / AutoAWQ 0.2.9. Batch size 1, minimum GPU count. Speed = (prompt+generated tokens)/e2e time = end-to-end per-request throughput INCLUDING prefill, NOT pure decode. Generate 2048 tokens, input lengths 1/6144/14336/30720/63488/129024.",
    conf=0.75, published="2025-04-29"))

wsrc(source(
    "syfi-qwen35-397b-mi300a-vibesys",
    "SyFI Lab (UW) — VibeSys Builds a Qwen3.5-397B Engine on MI300A, 2.3x Tuned SGLang",
    "https://syfi.cs.washington.edu/blog/2026-10-08-qwen35-mi300a/",
    "SyFI Lab, University of Washington", "blog",
    "4x AMD Instinct MI300A (128GB HBM3 shared CPU+GPU each, gfx942). Model: Qwen3.5-397B-A17B MXFP4 (amd/Qwen3.5-397B-A17B-MXFP4), 397B params, ~17B active, 512 experts, 15 full-attn + 45 Gated-DeltaNet linear-attn layers. Workload: deterministic multi-turn chat, 1->192 concurrent sessions, 3-6 turns of 80-300 gen tokens each, full history resent per turn. SLO: p95 TPOT<=250ms, p95 TTFT<=10s. Goodput = tok/s of responses meeting SLO. Baseline: VibeSys-ported+tuned SGLang v0.5.18 (961 tok/s @48 sessions); bespoke agent engine 2242 tok/s @96 sessions.",
    conf=0.8, published="2026-10-08"))

wsrc(source(
    "kuhaku413-rocm-inference-bench",
    "Kuhaku413/rocm-inference-bench — AMD MI250X / MI300X vs CUDA LLM inference table",
    "https://github.com/Kuhaku413/rocm-inference-bench",
    "Kuhaku413 (GitHub)", "repo",
    "Single-commit benchmark-suite README with a results table: Llama-3-8B BF16 on MI250X = 42.3 tok/s (16.2GB), MI300X FP8 = 68.7 tok/s (12.1GB), Mistral-7B BF16 on MI250X = 38.9 tok/s (14.8GB). No methodology detail beyond PyTorch+ROCm6.x, no batch/prompt-length specified — treat as low-confidence community numbers.",
    conf=0.35))

wsrc(source(
    "aws-neuron-trn1-inference-benchmarks",
    "AWS Neuron docs — Trn1/Trn1n Inference Performance tables",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/trn1/trn1-inference-performance.html",
    "AWS", "benchmark",
    "Vendor reference inference numbers for Trn1/Trn1n (Trainium1). NeuronPerf-driven. Encoder models (BERT/RoBERTa/etc) reported as inference/sec; encoder-decoder (flan-t5-xl, t5-3b) reported as tokens/sec counting BOTH input and output tokens, NeuronX Distributed TP=8 on trn1.32xlarge. Last update 2026-02-26, SDK versions 2.17-2.28. Numbers are aggregate throughput across the data-parallel run mode, not per-request decode.",
    conf=0.7))

wsrc(source(
    "aws-neuron-inf1-inference-benchmarks",
    "AWS Neuron docs — Inf1 Inference Performance tables",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/benchmarks/inf1/index.html",
    "AWS", "benchmark",
    "Vendor reference inference numbers for Inf1 (Inferentia1). NeuronPerf-driven. Encoder models reported as inference/sec (sentences/sec), CNN models as images/sec — NOT token-based; no generative LLM rows exist for Inf1 in this doc. Last update 2024-09-16, Neuron 2.20. inf1.xlarge=1 Inferentia chip? (inf1.xlarge exposes 4 NeuronCores; each Inferentia chip = 4 NeuronCores so 1 chip).",
    conf=0.7))

# ---------------- NVIDIA H20 (nvidia-h20): Qwen docs speed benchmark ----------------
QWEN_METH_SGL = ("VENDOR-MEASURED (Qwen team, qwen.readthedocs.io Speed Benchmark). SGLang 0.4.6.post1 "
                 "+ sgl-kernel 0.1.0, PyTorch 2.6.0+cu124, vLLM 0.7.2 for AWQ-marlin. NVIDIA H20 96GB, "
                 "context_length=140000, enable_mixed_chunk, mem_fraction_static=0.85, skip_tokenizer_init. "
                 "Batch size 1, minimum GPU count. Speed=(prompt+generated)/e2e time = end-to-end per-request "
                 "throughput including prefill — NOT pure decode. Generate 2048 tokens.")
QWEN_METH_TF = ("VENDOR-MEASURED (Qwen team, qwen.readthedocs.io Speed Benchmark). HF Transformers 4.51.3 + "
                "FlashAttention 2.7.4 + PyTorch 2.6.0, GPTQModel 2.2.0 / AutoAWQ 0.2.9. NVIDIA H20 96GB, "
                "batch size 1, generate 2048 tokens. Speed=(prompt+generated)/e2e time = end-to-end "
                "throughput including prefill — NOT pure decode.")
QWEN_NOTE = ("Metric is end-to-end (prompt+gen)/time on a single request — equivalent to ~decode rate only "
             "as input_len->0; at larger input lengths it blends prefill+decode. Reads LOWER than a pure "
             "tg number. Vendor page; H20 is the China-export Hopper (96GB HBM3, ~4.0 TB/s, 148 TFLOPS BF16).")

h20_sglang = [
    # (model, quant, input_len, gpus, tps)
    ("Qwen3-0.6B", "bf16", 1, 1, 414.17), ("Qwen3-0.6B", "fp8", 1, 1, 458.03),
    ("Qwen3-0.6B", "bf16", 30720, 1, 3577.42), ("Qwen3-0.6B", "fp8", 30720, 1, 3819.86),
    ("Qwen3-1.7B", "bf16", 1, 1, 227.80), ("Qwen3-1.7B", "fp8", 1, 1, 333.90),
    ("Qwen3-1.7B", "bf16", 30720, 1, 2439.03), ("Qwen3-1.7B", "fp8", 30720, 1, 3165.32),
    ("Qwen3-4B", "bf16", 1, 1, 133.13), ("Qwen3-4B", "fp8", 1, 1, 200.61),
    ("Qwen3-4B", "awq-int4", 1, 1, 199.71), ("Qwen3-4B", "bf16", 30720, 1, 1165.75),
    ("Qwen3-4B", "fp8", 30720, 1, 1467.71),
    ("Qwen3-8B", "bf16", 1, 1, 81.73), ("Qwen3-8B", "fp8", 1, 1, 150.25),
    ("Qwen3-8B", "awq-int4", 1, 1, 144.11), ("Qwen3-8B", "bf16", 30720, 1, 832.67),
    ("Qwen3-8B", "fp8", 30720, 1, 1242.24), ("Qwen3-8B", "bf16", 63488, 1, 1112.78),
    ("Qwen3-8B", "bf16", 129024, 1, 1173.32),
    ("Qwen3-14B", "bf16", 1, 1, 47.10), ("Qwen3-14B", "fp8", 1, 1, 97.11),
    ("Qwen3-14B", "bf16", 30720, 1, 525.80), ("Qwen3-14B", "fp8", 30720, 1, 880.72),
    ("Qwen3-14B", "bf16", 63488, 1, 742.36),
    ("Qwen3-32B", "bf16", 1, 1, 20.72), ("Qwen3-32B", "fp8", 1, 1, 46.17),
    ("Qwen3-32B", "awq-int4", 1, 1, 47.67), ("Qwen3-32B", "bf16", 30720, 1, 240.75),
    ("Qwen3-32B", "bf16", 63488, 1, 342.96), ("Qwen3-32B", "bf16", 129024, 2, 711.40),
    ("Qwen3-30B-A3B", "bf16", 1, 1, 137.18), ("Qwen3-30B-A3B", "fp8", 1, 1, 155.55),
    ("Qwen3-30B-A3B", "bf16", 30720, 1, 1283.94), ("Qwen3-30B-A3B", "fp8", 30720, 1, 1405.91),
    ("Qwen3-30B-A3B", "bf16", 63488, 1, 1538.79),
    ("Qwen3-235B-A22B", "bf16", 1, 8, 74.50), ("Qwen3-235B-A22B", "fp8", 1, 4, 71.65),
    ("Qwen3-235B-A22B", "bf16", 30720, 8, 979.41), ("Qwen3-235B-A22B", "fp8", 30720, 4, 887.90),
    ("Qwen3-235B-A22B", "bf16", 63488, 8, 1493.91), ("Qwen3-235B-A22B", "bf16", 129024, 8, 1639.54),
]
def slug(s):
    return s.lower().replace(".","p").replace("_","-").replace("/","-")
for model, q, ilen, ngpu, tps in h20_sglang:
    ms = slug(model)
    tp = f", TP={ngpu}" if ngpu>1 else ""
    w(bench(
        f"qwen-bench-h20-sglang-{ms}-{q}-in{ilen}-tps",
        f"Qwen docs: {model} {q.upper()} on {ngpu}x H20 SGLang, input {ilen}, {tps} tok/s e2e",
        "sglang", ["nvidia-h20"], model, f"qwen3-{q}" if q!="bf16" else "bf16",
        "tps_aggregate", tps,
        f"tok/s (batch-1 e2e, in={ilen}+2048gen{tp})",
        QWEN_METH_SGL + f" Config: {ngpu}x H20{tp}.",
        "vendor", ["qwen-docs-h20-speed-benchmark"], conf=0.75,
        notes=QWEN_NOTE + (f" TP={ngpu} across {ngpu} H20 96GB cards." if ngpu>1 else "")))

h20_tf = [
    ("Qwen3-0.6B", "bf16", 1, 1, 58.57), ("Qwen3-0.6B", "awq-int4", 30720, 1, 345.71),
    ("Qwen3-1.7B", "bf16", 1, 1, 59.83), ("Qwen3-1.7B", "awq-int4", 30720, 1, 386.85),
    ("Qwen3-4B", "bf16", 1, 1, 45.94), ("Qwen3-4B", "awq-int4", 30720, 1, 481.69),
    ("Qwen3-8B", "bf16", 1, 1, 45.32), ("Qwen3-8B", "awq-int4", 30720, 1, 438.62),
    ("Qwen3-14B", "bf16", 1, 1, 40.66), ("Qwen3-14B", "awq-int4", 30720, 1, 363.25),
    ("Qwen3-32B", "bf16", 1, 1, 26.24), ("Qwen3-32B", "awq-int4", 30720, 1, 188.11),
    ("Qwen3-30B-A3B", "bf16", 1, 1, 1.89), ("Qwen3-30B-A3B", "bf16", 30720, 1, 27.03),
]
for model, q, ilen, ngpu, tps in h20_tf:
    ms = slug(model)
    w(bench(
        f"qwen-bench-h20-transformers-{ms}-{q}-in{ilen}-tps",
        f"Qwen docs: {model} {q.upper()} on 1x H20 HF-Transformers, input {ilen}, {tps} tok/s e2e",
        "hf-optimum", ["nvidia-h20"], model, f"qwen3-{q}" if q!="bf16" else "bf16",
        "tps_aggregate", tps,
        f"tok/s (batch-1 e2e, in={ilen}+2048gen)",
        QWEN_METH_TF + " Config: 1x H20.",
        "vendor", ["qwen-docs-h20-speed-benchmark"], conf=0.7,
        notes=QWEN_NOTE + " HF Transformers eager path (NOT a serving engine) — far below SGLang; this isolates the framework gap on H20."))

# ---------------- AMD MI300A (amd-instinct-mi300a): SyFI lab ----------------
SYFI_METH = ("THIRD-PARTY MEASURED (SyFI Lab, UW, blog 2026-10-08). 4x AMD Instinct MI300A (gfx942, 128GB "
             "HBM3 shared CPU+GPU per package, 512GB node). Model Qwen3.5-397B-A17B MXFP4 "
             "(amd/Qwen3.5-397B-A17B-MXFP4, ~17B active, 512 routed experts, hybrid 15 attn + 45 Gated-"
             "DeltaNet linear layers). Multi-turn chat benchmark, concurrency ramp 1->192 sessions, "
             "3-6 turns x 80-300 gen tokens, full history resent each turn. GOODPUT = output tok/s for "
             "responses meeting SLO (p95 TPOT<=250ms, p95 TTFT<=10s). AGGREGATE over concurrent sessions.")
w(bench(
    "syfi-mi300a-sglang-tuned-qwen35-397b-mxfp4-goodput",
    "SGLang v0.5.18 (VibeSys-ported+tuned) on 4x MI300A: Qwen3.5-397B-A17B MXFP4 goodput 961 tok/s @48sess",
    "sglang", ["amd-instinct-mi300a"], "Qwen3.5-397B-A17B", "mxfp4",
    "tps_aggregate", 961.0,
    "tok/s goodput (aggregate over 48 concurrent sessions, SLO-bound)",
    SYFI_METH + " Baseline: SGLang v0.5.18 ported to gfx942 by VibeSys agents (fused HIP MXFP4 MoE kernel, "
    "pinned KV pool, pre-sharded weight loads, prebuilt AITER kernel cache), tuned with MTP speculative "
    "decoding + GEMM tuning tables. Peak at 48 concurrent sessions.",
    "third_party", ["syfi-qwen35-397b-mi300a-vibesys"], conf=0.8,
    notes="Stock SGLang does NOT run this model on MI300A (AITER MXFP4 MoE kernels are gfx950-only); the "
          "961 tok/s figure is a VibeSys-ported+tuned build, not upstream SGLang. Goodput is SLO-gated "
          "aggregate output throughput — reads lower than raw max-throughput."))
w(bench(
    "syfi-mi300a-vibesys-bespoke-qwen35-397b-mxfp4-goodput",
    "VibeSys bespoke agent-built engine on 4x MI300A: Qwen3.5-397B-A17B MXFP4 goodput 2242 tok/s @96sess",
    "vllm", ["amd-instinct-mi300a"], "Qwen3.5-397B-A17B", "mxfp4",
    "tps_aggregate", 2242.0,
    "tok/s goodput (aggregate over 96 concurrent sessions, SLO-bound)",
    SYFI_METH + " Engine: bespoke agent-built serving engine (~19.4k LOC, prefix cache w/ DeltaNet "
    "snapshots, HIP-graph decode, fused MXFP4 MoE, MTP w/ 2 draft tokens @84% accept). Peak at 96 sessions; "
    "2.33x the tuned-SGLang baseline on the same 4x MI300A node.",
    "third_party", ["syfi-qwen35-397b-mi300a-vibesys"], conf=0.75, repro=False,
    notes="Filed under engine_id=vllm only because there is no 'vibesys-bespoke' engine record and vllm is "
          "the closest serving-engine class; the actual engine is a one-off agent build — see source. "
          "MI300A's unified HBM3 lets weights, KV, DeltaNet state, OS page cache and host share 128GB/pkg."))

# ---------------- AMD MI250X (amd-instinct-mi250x): rocm-inference-bench ----------------
RIB_METH = ("COMMUNITY-REPORTED (Kuhaku413/rocm-inference-bench, single-commit README results table). "
            "PyTorch + ROCm 6.x on MI250X. No batch size, prompt length, or repeat count documented — "
            "reported 'Tokens/sec' treated as single-request end-to-end decode-class throughput. "
            "LOW CONFIDENCE: no methodology page, no commit-pinned framework versions.")
w(bench(
    "kuhaku-mi250x-llama3-8b-bf16-decode",
    "rocm-inference-bench: Llama-3-8B BF16 on MI250X, 42.3 tok/s (README table)",
    "hf-optimum", ["amd-instinct-mi250x"], "meta-llama/Llama-3-8B", "bf16",
    "decode_tok_s", 42.3,
    "tok/s (batch-1, methodology undocumented)",
    RIB_METH, "third_party", ["kuhaku413-rocm-inference-bench"],
    status="draft", conf=0.35, repro=False,
    notes="MI250X = dual-GCD CDNA2 OAM, 128GB HBM2e, 3.2 TB/s, gfx90a. 42.3 tok/s on 8B BF16 is plausible "
          "for the memory-bound regime but the source is a thin single-commit repo — marked draft/low-conf."))
w(bench(
    "kuhaku-mi250x-mistral-7b-bf16-decode",
    "rocm-inference-bench: Mistral-7B BF16 on MI250X, 38.9 tok/s (README table)",
    "hf-optimum", ["amd-instinct-mi250x"], "mistralai/Mistral-7B-v0.1", "bf16",
    "decode_tok_s", 38.9,
    "tok/s (batch-1, methodology undocumented)",
    RIB_METH, "third_party", ["kuhaku413-rocm-inference-bench"],
    status="draft", conf=0.35, repro=False,
    notes="Same caveats as the Llama-3-8B row on the same source."))
w(bench(
    "kuhaku-mi300x-llama3-8b-fp8-decode",
    "rocm-inference-bench: Llama-3-8B FP8 on MI300X, 68.7 tok/s (README table)",
    "hf-optimum", ["amd-instinct-mi300x"], "meta-llama/Llama-3-8B", "fp8",
    "decode_tok_s", 68.7,
    "tok/s (batch-1, methodology undocumented)",
    RIB_METH, "third_party", ["kuhaku413-rocm-inference-bench"],
    status="draft", conf=0.35, repro=False,
    notes="MI300X 192GB HBM3 — first FP8 llama row in repo at this confidence tier; keep draft pending a "
          "corroborating run."))

# ---------------- AWS Trainium1 (aws-trainium1): Neuron docs ----------------
TRN1_METH = ("VENDOR-MEASURED (AWS Neuron docs, Trn1/Trn1n Inference Performance, NeuronPerf). "
             "trn1.32xlarge (16 Trainium1 chips). NeuronX Distributed tensor-parallel TP=8, DP=1, "
             "batch 1, seq_len 128, input 128 + output ~84 tokens, FP32 weights with Matmult-BF16 "
             "autocast. 'Throughput (tokens/second)' counts BOTH input and output tokens (vendor note). "
             "Encoder-decoder text-generation rows.")
for model, tps, lat in [("google/flan-t5-xl", 122.52, 8.16), ("t5-3b", 116.29, 8.58)]:
    w(bench(
        f"aws-neuron-trn1-{slug(model)}-tp8-128io-tps",
        f"AWS Neuron docs: {model} on trn1.32xlarge TP=8, {tps} tok/s (in+out)",
        "alt-neuron-compiler", ["aws-trainium1"], model, "fp32",
        "tps_aggregate", tps,
        "tok/s (in+out tokens, batch-1, TP=8 over 16-chip trn1.32xlarge)",
        TRN1_METH, "vendor", ["aws-neuron-trn1-inference-benchmarks"], conf=0.7,
        notes=f"Latency/token p50 {lat} ms. NeuronX Distributed compile path; Neuron SDK ~2.17. "
              "Trainium1 is not a consumer part but IS a budget cloud inference path (trn1 cheaper than "
              "GPU equivalents for compiled models); first benchmark coverage for aws-trainium1."))
# one latency-optimized encoder row to anchor the encoder side (batch-1 real-time)
w(bench(
    "aws-neuron-trn1-bert-base-rt-inf-s",
    "AWS Neuron docs: bert-base-uncased on trn1.2xlarge, real-time, 2085 inf/s (p50 0.94ms)",
    "alt-neuron-compiler", ["aws-trainium1"], "bert-base-uncased", "fp32",
    "tps_aggregate", 2085.45,
    "inferences/s (encoder fwd pass, batch-1, latency-optimized)",
    ("VENDOR-MEASURED (AWS Neuron docs, NeuronPerf). trn1.2xlarge, PyTorch 2.8, Neuron 2.26, Data "
     "Parallel run-mode, batch 1, seq_len 128, FP32 w/ Matmult-BF16 autocast. Encoder model — unit is "
     "inferences/sec, NOT tokens/sec."),
    "vendor", ["aws-neuron-trn1-inference-benchmarks"], conf=0.7,
    notes="Encoder (BERT) row — counts whole-sequence inferences, not generative tokens. Included to "
          "record the vendor-published latency-optimized figure; not comparable to decode tok/s."))

# ---------------- AWS Inferentia1 (aws-inferentia1): Neuron docs ----------------
INF1_METH = ("VENDOR-MEASURED (AWS Neuron docs, Inf1 Inference Performance, NeuronPerf, last update "
             "2024-09-16, Neuron 2.20). inf1.xlarge = 1 Inferentia1 chip (4 NeuronCores). "
             "Encoder/CNN models only — NO generative LLM rows exist for Inf1. Units are "
             "inferences/sec (BERT, seq_len 128) or images/sec (ResNet/YOLO), NOT tokens/sec.")
for rid, name, model, tps, unit, meth_extra, notes in [
    ("aws-neuron-inf1-bert-base-batch-inf-s",
     "AWS Neuron docs: bert-base-cased on inf1.xlarge batch, 1095 inf/s (p50 58ms)",
     "bert-base-cased", 1095.0, "inferences/s (encoder, batch 8, seq_len 128, fp32)",
     "PyTorch 1.13.1, batch 8, Data Parallel.",
     "Batch-optimized encoder row. Inf1 is EOL for new work (Neuron docs mark Inf1 pages archived) but "
     "remains the cheapest AWS inference silicon; recorded as the vendor reference number."),
    ("aws-neuron-inf1-resnet50-batch-inf-s",
     "AWS Neuron docs: ResNet-50 on inf1.xlarge batch, 2207 img/s (p50 18ms)",
     "ResNet-50", 2207.0, "images/s (batch 10, fp16)",
     "TensorFlow 1.15, batch 10, Data Parallel, fp16.",
     "CNN row — highest-throughput published Inf1 figure; images/sec, not tokens."),
    ("aws-neuron-inf1-yolov4-rt-inf-s",
     "AWS Neuron docs: YOLOv4 on inf1.2xlarge real-time, 180 img/s (p50 40ms)",
     "YOLOv4", 180.0, "images/s (batch 1, fp32)",
     "PyTorch 1.13, inf1.2xlarge, batch 1, Real-Time run-mode, fp32.",
     "Latency-optimized vision row. No LLM/generative data exists for Inferentia1 — the product line's "
     "LLM story starts at Inferentia2."),
]:
    w(bench(rid, name, "alt-neuron-compiler", ["aws-inferentia1"], model, "fp32",
            "tps_aggregate", tps, unit,
            INF1_METH + " " + meth_extra,
            "vendor", ["aws-neuron-inf1-inference-benchmarks"], conf=0.7, notes=notes))

# ---------------- NVIDIA A10 (nvidia-a10): scoreboard A10G noFA rows ----------------
A10_METH = ("COMMUNITY-SCOREBOARD COMMENT (llama.cpp discussion #15013, reported row, NOT on the "
            "maintainer-curated board). llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa 0. Build commit "
            "571d0d5 (b10068), reporter @Hastwell. Per-request batch-1. Device: NVIDIA A10G (AWS custom "
            "A10 variant, sm_86, 24GB GDDR6, 300W, ~498 GB/s; VRAM reported 22502 MiB).")
for fa, pp, tg in [("0", 2354.23, 108.29)]:
    for metric, val, test in [("prefill_tok_s", pp, "pp512"), ("decode_tok_s", tg, "tg128")]:
        w(bench(
            f"llamacpp-cuda-a10g-llama27b-q40-nofa-{test}",
            f"llama.cpp CUDA on NVIDIA A10G (A10 AWS variant): Llama-2-7B Q4_0 noFA {test} t/s (#15013 comment)",
            "llama-cpp", ["nvidia-a10"], "llama-2-7b.Q4_0.gguf (TheBloke, 3.56 GiB, 6.74B params)",
            "gguf-q4-0", metric, val,
            f"tok/s ({test}, per-request, batch 1)",
            A10_METH, "third_party",
            ["llama-cpp-discussion-15013-nvidia-cuda-scoreboard"], conf=0.6,
            notes="A10G is the AWS-specific A10 variant (2-slot, 300W, same ~498GB/s). Comment-level row, "
                  "not on the curated scoreboard. Complements the two existing FA-on A10G records."))

# ---------------- MI250X mapping note: add accelerator_ids tag ----------------
# The Argonne 'AMD MI250 GPU' rows run on Polaris nodes whose rocm-smi reports DID 0x740c @560W,
# which is MI250X silicon. Existing records already tag amd-instinct-mi250; rather than minting
# duplicates, add a clarifying benchmark record that documents the silicon identity and cross-refs.
# (Not adding accelerator_ids=mi250x to those rows — the Argonne paper itself says 'MI250 GPU'
#  and the repo record amd-instinct-mi250 already covers the Polaris MI250 series.)

print("done")
