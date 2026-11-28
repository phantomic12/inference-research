import json, os
ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-15"

def w(rec, subdir):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP exists:", rec["id"]); return
    with open(out, "w") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", rec["id"])

SRC = "cug2025-mi300a-kd-vllm-inference"

# --- source record ---
w({
 "id": SRC, "type": "source",
 "name": "Dickmann et al. — Evaluating AMD Instinct MI300A APU: Performance Insights on LLM Training via Knowledge Distillation (CUG 2025)",
 "status": "verified", "confidence": 0.75, "updated": TODAY,
 "url": "https://cug.org/proceedings/cug2025_proceedings/includes/files/pap144s2-file1.pdf",
 "publisher": "Cray User Group 2025 proceedings (Seedbox / HPE / HLRS / AMD)",
 "kind": "paper",
 "published": "2025-05-08",
 "accessed": TODAY, "archived_url": None,
 "notes": ("Section 5.2 reports vLLM batch-mode inference on ONE MI300A node (4 APUs, gfx942, "
           "128GB unified HBM per APU, ~5.3 TB/s, ROCm 6.2.2) using 100 diverse prompts. "
           "Two rows: KafkaLM-15B-Base (2:4-sparse student distilled from Mistral-Small-24B) and "
           "Mistral-Small-24B-Base-2501 teacher. Reports TTFT and aggregate tokens/sec. "
           "Paper explicitly notes this is the first published MI300A LLM inference data "
           "(MI250X/MI300X exist, MI300A did not).")
}, "sources")

METH = ("THIRD-PARTY MEASURED, peer-reviewed (CUG 2025 proceedings, HLRS 'Hunter' HPE Cray EX4000). "
        "vLLM batch-inference mode on 1 MI300A node = 4 APUs (gfx942/CDNA3, 128GB unified HBM each, "
        "~5.3 TB/s peak, Infinity Fabric/XGMI + PCIe Gen5 x16), ROCm 6.2.2, PyTorch 2.6.0. "
        "Workload = 100 diverse prompts; metrics are AGGREGATE across the batched run "
        "(prompts/sec and total tokens/sec), NOT per-request decode tok/s. Figure 5. "
        "No speculative decoding, no quantization beyond model-native precision.")

def bench(bid, name, model, metric, value, unit, notes):
    w({
     "id": bid, "type": "benchmark", "name": name,
     "engine_id": "vllm",
     "accelerator_ids": ["amd-instinct-mi300a"],
     "interconnect_ids": ["amd-infinity-fabric-cdna3", "amd-xgmi", "amd-pcie-gen5-x16"],
     "model": model,
     "format_id": None,
     "metric": metric, "value": value, "unit": unit,
     "methodology": METH,
     "measured_by": "third_party", "reproducible": True,
     "energy_basis": None, "power_w": None, "power_scope": None,
     "power_cap": None, "clock_lock_mhz": None, "thermal_state": "unknown",
     "notes": notes,
     "status": "verified", "confidence": 0.72, "updated": TODAY,
     "sources": [SRC],
    }, "benchmarks")

# Kafka 15B (distilled student) — 812 tok/s aggregate, TTFT 2.46s, 6.55 prompts/s
bench("cug2025-mi300a-vllm-kafka15b-tok-s",
      "MI300A x1 node (4 APU): KafkaLM-15B, vLLM batch inference, 812 tok/s aggregate (CUG 2025)",
      "Seedbox/KafkaLM-15B-Base (15B, 2:4-sparse distilled from Mistral-Small-24B)",
      "tps_aggregate", 812, "tok/s aggregate across 100-prompt batch (whole node)",
      "Aggregate decode+prefill throughput over the 100-prompt vLLM batch on a single 4-APU node. "
      "First published MI300A LLM inference figure. 6.55 prompts/s end-to-end. Confidence 0.72: "
      "peer-reviewed but single-run, throughput is node-aggregate not per-request.")

bench("cug2025-mi300a-vllm-kafka15b-ttft-ms",
      "MI300A x1 node (4 APU): KafkaLM-15B, vLLM batch inference, TTFT 2.46 s (CUG 2025)",
      "Seedbox/KafkaLM-15B-Base (15B, 2:4-sparse distilled from Mistral-Small-24B)",
      "ttft_ms", 2460, "ms time-to-first-token (mean over 100 prompts, whole node)",
      "TTFT 2.46s (p<0.01 vs teacher). Node-aggregate serving latency, not a single-stream number.")

# Mistral-Small-24B teacher — 579 tok/s aggregate, TTFT 4.91s, 4.7 prompts/s
bench("cug2025-mi300a-vllm-mistral-small-24b-tok-s",
      "MI300A x1 node (4 APU): Mistral-Small-24B-Base-2501, vLLM batch inference, 579 tok/s aggregate (CUG 2025)",
      "mistralai/Mistral-Small-24B-Base-2501 (24B dense teacher)",
      "tps_aggregate", 579, "tok/s aggregate across 100-prompt batch (whole node)",
      "Aggregate decode+prefill throughput over the 100-prompt vLLM batch on a single 4-APU node. "
      "4.7 prompts/s end-to-end. Confidence 0.72: peer-reviewed but single-run, node-aggregate "
      "not per-request; the 24B dense teacher is the baseline against which the 15B student's "
      "40% throughput gain is measured.")

bench("cug2025-mi300a-vllm-mistral-small-24b-ttft-ms",
      "MI300A x1 node (4 APU): Mistral-Small-24B-Base-2501, vLLM batch inference, TTFT 4.91 s (CUG 2025)",
      "mistralai/Mistral-Small-24B-Base-2501 (24B dense teacher)",
      "ttft_ms", 4910, "ms time-to-first-token (mean over 100 prompts, whole node)",
      "TTFT 4.91s on the dense 24B teacher. The 15B student's 50% TTFT reduction is the paper's "
      "headline inference result.")

print("done")
