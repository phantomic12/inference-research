#!/usr/bin/env python3
"""Join supply prices to benchmark throughput: USD per million output tokens.

    python tools/cost_per_token.py                      # the table
    python tools/cost_per_token.py --explain            # show the derivation
    python tools/cost_per_token.py --format json
    python tools/cost_per_token.py --single-user        # only per-user metrics
    python tools/cost_per_token.py --gaps               # what the records cannot do

WHY A TOOL AND NOT A SPREADSHEET. The repo holds ~50 priced supply records and
~38 throughput benchmark records, and NOTHING joins them. The join is easy to
get wrong in two specific ways, so both are handled structurally here rather
than by care:

1. PRICE BASIS. `price_usd` is not per-GPU-hour. A supply record priced "per
   8-GPU node per HOUR" is 8x a per-GPU-hour quote; Oracle's ".8" shapes are
   quoted per GPU per hour despite the suffix; Runpod's serverless figures are
   per WORKER-hour. PRICE_QUOTES below therefore stores an explicit
   per-GPU-HOUR number for each (supply record, accelerator) together with the
   literal substring of the record it was read out of, and `check_evidence()`
   fails loudly if that substring is not present in the record on disk. A quote
   that cannot be shown to exist in the record is not a quote.

2. AGGREGATE IS NOT PER-USER. `tps_aggregate` is output tokens/s summed across
   every in-flight request. The repo's own paired records show batch-64
   aggregate at 800 tok/s while the per-user experience is 12.5 tok/s - a 64x
   gap that runs the WRONG WAY, because bigger batches raise aggregate and
   LOWER per-user latency-adjusted throughput. An aggregate tok/s divided by a
   GPU-hour yields a fleet-average cost that is NOT what a single user pays.
   Rows are therefore labelled `aggregate` or `single-user`, and `--single-user`
   prints only the latter. Never compare the two columns.

Every input is an existing record. Nothing here interpolates, extrapolates or
carries a default: a combination with no price or no throughput is reported as
`no record`, never as a guess. Run `python tools/cost_per_token.py --gaps`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

THROUGHPUT_METRICS = ("decode_tok_s", "tok_s_per_user", "tps_aggregate")

# Does the record itself state how many devices its aggregate covers? This is
# only used to choose the WORDS of a `--gaps` message, never to pick a divisor:
# the divisor still has to be quoted verbatim in BENCH_JOIN and proven by
# check_evidence(). The pattern deliberately accepts only a count immediately
# followed by a device noun, so "87 GPUs across 11 nodes" reports "87 GPUs".
GPU_COUNT_HINT = re.compile(r"\b\d+x\s[^\s]+|\b\d+[\s-]GPUs?\b")

# --------------------------------------------------------------------------
# PRICE_QUOTES: (supply record, accelerator) -> USD per GPU-hour.
#
# usd_per_gpu_hour  the normalised rate actually used in the arithmetic
# basis           where that rate came from, as a LITERAL substring that must
#                 appear in the record's price_basis/notes (see check_evidence)
# tier            on-demand | spot | capacity-block | low-priority | serverless
#                 | peer-to-peer-asking | base-with-multipliers
# gpus_in_quote    how many GPUs the raw quoted figure covers, for the record
# --------------------------------------------------------------------------
PRICE_QUOTES: dict[str, list[dict]] = {
    # ---- AMD Instinct, Oracle OCI bare metal ---------------------------
    # Added 2026-10-05. These close the repo's longest-standing cost gap:
    # docs/08-cost-per-token.md asserted for months that every AMD benchmark
    # was unpriceable, and that was true only of the records that existed. The
    # OEM/reseller channels cited as evidence do not quote per-GPU-hour because
    # that is not their unit - the negative generalised across a boundary it was
    # never tested at. Oracle publishes the rate directly.
    #
    # DO NOT DIVIDE BY 8. Oracle's footnote defines the column as a per-GPU
    # rate; the '.8' in the shape name is the GPU count. This is the same trap
    # this table already caught on Oracle's H100 record and on CoreWeave.
    "w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour": [
        dict(accel="amd-instinct-mi300x", usd=6.00, tier="on-demand", gpus=8,
             basis="BM.GPU.MI300X.8 = 8x AMD MI300X 192GB Matrix Core, CDNA 3, 8x1x400 Gb/sec RDMA, $6.00/GPU-hr"),
        dict(accel="amd-instinct-mi355x", usd=8.60, tier="on-demand", gpus=8,
             basis="MI355X is $8.60 per MI355X-hour"),
    ],
    # ---- H100 SXM -------------------------------------------------------
    "aws-ec2-p5-h100": [
        dict(accel="nvidia-h100-sxm", usd=6.88, tier="on-demand", gpus=8,
             basis="On-demand us-east-1 $55.04/instance-hr / 8 = $6.88"),
        dict(accel="nvidia-h100-sxm", usd=5.191, tier="capacity-block", gpus=8,
             basis="Capacity Block $41.528/8 = $5.191 per H100 in us-east"),
    ],
    "azure-nd-h100-v5": [
        dict(accel="nvidia-h100-sxm", usd=12.29, tier="on-demand", gpus=8,
             basis="ND96isr H100 v5 = 8x H100, Linux on-demand $98.32/instance-hr / 8 = $12.29 in eastus"),
        dict(accel="nvidia-h100-sxm", usd=2.2712, tier="spot", gpus=8,
             basis="Spot $18.169536 = $2.2712"),
    ],
    "gcp-a3-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=11.0613, tier="on-demand", gpus=8,
             basis="a3-highgpu-8g = 8x H100 at $88.490000119/instance-hr / 8 = $11.0613"),
    ],
    "baseten-gpu-instances": [
        dict(accel="nvidia-h100-sxm", usd=6.50, tier="on-demand", gpus=1,
             basis="H100 80GB $0.10833/min = $6.50/hr"),
    ],
    "coreweave-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=6.155, tier="on-demand", gpus=8,
             basis="HGX H100 $49.24 on-demand"),
        dict(accel="nvidia-h100-sxm", usd=2.46375, tier="spot", gpus=8,
             basis="HGX H100 $49.24 on-demand, $19.71 spot (North America)"),
        dict(accel="nvidia-h200-sxm", usd=6.305, tier="on-demand", gpus=8,
             basis="HGX H200 $50.44 / $20.93"),
        dict(accel="nvidia-h200-sxm", usd=2.61625, tier="spot", gpus=8,
             basis="HGX H200 $50.44 / $20.93"),
        dict(accel="nvidia-b200", usd=8.60, tier="on-demand", gpus=8,
             basis="HGX B200 $68.80 / $34.11"),
        dict(accel="nvidia-b200", usd=4.26375, tier="spot", gpus=8,
             basis="HGX B200 $68.80 / $34.11"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=2.70, tier="on-demand", gpus=8,
             basis="HGX A100 80GB $21.60 / $9.65"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=1.20625, tier="spot", gpus=8,
             basis="HGX A100 80GB $21.60 / $9.65"),
        dict(accel="nvidia-l40s", usd=2.25, tier="on-demand", gpus=8,
             basis="L40S $18.00 / $7.88"),
        # CoreWeave's price_basis also lists an L40 rate ($10.00/8 = $1.25), but
        # nvidia-l40 is absent from the record's accelerator_ids, so there is no
        # accelerator id to join it to. Left out deliberately; see --gaps.
    ],
    "fireworks-on-demand-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=8.00, tier="on-demand", gpus=1,
             basis="H100 80GB $0.134/min = $8.00/h"),
        dict(accel="nvidia-h200-sxm", usd=8.00, tier="on-demand", gpus=1,
             basis="H200 141GB $0.134/min = $8.00/h"),
        dict(accel="nvidia-b200", usd=13.00, tier="on-demand", gpus=1,
             basis="B200 180GB $0.217/min = $13.00/h"),
        dict(accel="nvidia-b300", usd=15.00, tier="on-demand", gpus=1,
             basis="B300 288GB $0.250/min = $15.00/h"),
    ],
    "lambda-gpu-cloud-h100": [
        dict(accel="nvidia-h100-sxm", usd=3.99, tier="on-demand", gpus=8,
             basis="per GPU-hour, 8-GPU instance (208 vCPU / 1800 GiB / 22 TiB SSD): H100 SXM $3.99"),
        dict(accel="nvidia-h100-pcie", usd=3.29, tier="on-demand", gpus=1,
             basis="1x H100 PCIe $3.29"),
        dict(accel="nvidia-b200", usd=6.69, tier="on-demand", gpus=8,
             basis="B200 SXM6 $6.69"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=2.79, tier="on-demand", gpus=8,
             basis="A100 SXM 80GB $2.79"),
    ],
    "modal-serverless-gpu": [
        dict(accel="nvidia-h100-sxm", usd=3.95, tier="base-with-multipliers", gpus=1,
             basis="H100 SXM5 $0.001097/sec = $3.95/hr"),
        dict(accel="nvidia-h200-sxm", usd=4.54, tier="base-with-multipliers", gpus=1,
             basis="H200 SXM $0.001261 = $4.54"),
        dict(accel="nvidia-b200", usd=6.25, tier="base-with-multipliers", gpus=1,
             basis="B200 $0.001736 = $6.25"),
        dict(accel="nvidia-b300", usd=7.10, tier="base-with-multipliers", gpus=1,
             basis="B300 $0.001972 = $7.10"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=2.50, tier="base-with-multipliers", gpus=1,
             basis="A100 80GB $0.000694 = $2.50"),
        dict(accel="nvidia-l40s", usd=1.95, tier="base-with-multipliers", gpus=1,
             basis="L40S $0.000542 = $1.95"),
        dict(accel="nvidia-l4", usd=0.80, tier="base-with-multipliers", gpus=1,
             basis="L4 $0.000222 = $0.80"),
    ],
    "oracle-bm-gpu-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=10.00, tier="on-demand", gpus=1,
             basis="BM.GPU.H100.8 = $10.00"),
        dict(accel="nvidia-h200-sxm", usd=10.00, tier="on-demand", gpus=1,
             basis="BM.GPU.H200.8 = $10.00"),
    ],
    "runpod-pods-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=3.49, tier="on-demand", gpus=1,
             basis="H100 SXM $3.49"),
        dict(accel="nvidia-h100-pcie", usd=2.89, tier="on-demand", gpus=1,
             basis="H100 PCIe $2.89"),
        dict(accel="nvidia-h200-sxm", usd=4.59, tier="on-demand", gpus=1,
             basis="H200 $4.59"),
        dict(accel="nvidia-b200", usd=6.79, tier="on-demand", gpus=1,
             basis="B200 $6.79"),
        dict(accel="nvidia-b300", usd=7.89, tier="on-demand", gpus=1,
             basis="B300 $7.89"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=1.59, tier="on-demand", gpus=1,
             basis="A100 80GB $1.59"),
        dict(accel="nvidia-l40s", usd=1.09, tier="on-demand", gpus=1,
             basis="L40S $1.09"),
        dict(accel="nvidia-l4", usd=0.49, tier="on-demand", gpus=1,
             basis="L4 $0.49"),
    ],
    "runpod-serverless-h100-h200": [
        dict(accel="nvidia-h100-sxm", usd=4.79, tier="serverless", gpus=1,
             basis="H100 $4.79"),
        dict(accel="nvidia-h200-sxm", usd=5.93, tier="serverless", gpus=1,
             basis="H200 $5.93"),
        dict(accel="nvidia-b200", usd=8.64, tier="serverless", gpus=1,
             basis="B200 $8.64"),
        # Runpod serverless also quotes a B300 at $9.98 in price_basis, but nvidia-b300
        # is absent from that record's accelerator_ids, so it is not joinable.
        dict(accel="nvidia-a100-80gb-sxm4", usd=2.72, tier="serverless", gpus=1,
             basis="A100 $2.72"),
    ],
    "together-gpu-clusters-h100": [
        dict(accel="nvidia-h100-sxm", usd=5.49, tier="on-demand", gpus=1,
             basis="NVIDIA HGX H100 $5.49"),
        dict(accel="nvidia-b200", usd=8.99, tier="on-demand", gpus=1,
             basis="HGX B200 $8.99"),
    ],
    "rental-h100-sxm-vast-index": [
        dict(accel="nvidia-h100-sxm", usd=2.47, tier="peer-to-peer-asking", gpus=1,
             basis="USD per GPU-HOUR for the GPU component"),
    ],
    "rental-h200-sxm-vast-index": [
        dict(accel="nvidia-h200-sxm", usd=5.00, tier="peer-to-peer-asking", gpus=8,
             basis="USD per GPU-HOUR for the GPU component"),
    ],
    "rental-b200-vast-index": [
        dict(accel="nvidia-b200", usd=7.97, tier="peer-to-peer-asking", gpus=8,
             basis="USD per GPU-HOUR for the GPU component"),
    ],
    # ---- prices with no throughput record on the same accelerator ----------
    "aws-ec2-p5en-h200": [
        dict(accel="nvidia-h200-sxm", usd=7.912, tier="on-demand", gpus=8,
             basis="On-demand us-east-1 $63.296/instance-hr / 8 = $7.912"),
        dict(accel="nvidia-h200-sxm", usd=6.865, tier="capacity-block", gpus=8,
             basis="Capacity Block $54.92/8 = $6.865 in us-east"),
    ],
    "azure-nd-h200-v5": [
        dict(accel="nvidia-h200-sxm", usd=10.60, tier="on-demand", gpus=8,
             basis="ND96isr H200 v5 = 8x H200, on-demand $84.80/instance-hr / 8 = $10.60 in eastus2 and westus2"),
        # Azure publishes Spot for H200 v5 at EXACTLY the on-demand rate
        # ($84.80), a 0% discount, which is what a sold-out fleet looks like in
        # a price API. There is no cheaper spot row to add.
    ],
    "aws-ec2-p4d-p4de-a100": [
        dict(accel="nvidia-a100-80gb-sxm4", usd=3.4309, tier="on-demand", gpus=8,
             basis="p4de.24xlarge = 8x A100 80GB, on-demand $27.44705/instance-hr / 8 = $3.4309"),
    ],
    "azure-nd-a100-v4": [
        dict(accel="nvidia-a100-80gb-sxm4", usd=4.096, tier="on-demand", gpus=8,
             basis="ND96amsr A100 v4 = 8x A100 80GB, Linux on-demand $32.77/instance-hr / 8 = $4.096"),
        dict(accel="nvidia-a100-80gb-sxm4", usd=1.0552, tier="spot", gpus=8,
             basis="Spot $8.441552 = $1.0552"),
    ],
    "gcp-a2-a100": [
        dict(accel="nvidia-a100-80gb-sxm4", usd=5.0688, tier="on-demand", gpus=8,
             basis="a2-ultragpu-8g = 8x A100 80GB at $40.550383123/instance-hr / 8 = $5.0688"),
    ],
    "gcp-g2-l4": [
        dict(accel="nvidia-l4", usd=0.8536, tier="on-demand", gpus=1,
             basis="g2-standard-8 = 1x L4 at $0.853624312/instance-hr"),
    ],
    "aws-ec2-g6-g6e-l4-l40s": [
        dict(accel="nvidia-l4", usd=1.6688, tier="on-demand", gpus=8,
             basis="g6.48xlarge = 8x L4 at $13.3504/instance-hr / 8 = $1.6688"),
        dict(accel="nvidia-l40s", usd=3.7664, tier="on-demand", gpus=8,
             basis="g6e.48xlarge = 8x L40S at $30.13118/8 = $3.7664"),
        dict(accel="nvidia-a10", usd=2.036, tier="on-demand", gpus=8,
             basis="g5.48xlarge = 8x A10G at $16.288/8 = $2.036"),
    ],
    "oracle-bm-gpu-a100": [
        dict(accel="nvidia-a100-80gb-sxm4", usd=4.00, tier="on-demand", gpus=8,
             basis="BM.GPU.A100-v2.8 (8x A100 80GB) = $4.00"),
        dict(accel="nvidia-l40s", usd=3.50, tier="on-demand", gpus=4,
             basis="BM.GPU.L40S-NC.4 (4x L40S) = $3.50"),
    ],
    "rental-datacenter-l40s-6000ada-vast-index": [
        dict(accel="nvidia-l40s", usd=0.7085, tier="peer-to-peer-asking", gpus=1,
             basis="USD per GPU-HOUR for the GPU component"),
    ],
}

# --------------------------------------------------------------------------
# BENCH_JOIN: benchmark id -> how many GPUs the throughput figure covers.
#
# gpus=None means the record's `unit` already says "per GPU", so no division
# is needed. `evidence` is a literal substring of the record's unit or
# methodology that establishes the GPU count; check_evidence() enforces it.
# Excluded benchmarks are listed in EXCLUDED with a reason instead.
# --------------------------------------------------------------------------
BENCH_JOIN: dict[str, dict] = {
    # --- 1-GPU runs: the unit is already per-GPU, no normalisation needed ---
    "trtllm-llama31-8b-fp8-h100-1k1k-tps": dict(gpus=1, evidence="1x NVIDIA H100 SXM 80GB"),
    "trtllm-llama31-8b-fp8-h200-1k1k-tps": dict(gpus=1, evidence="1x NVIDIA H200 SXM 141GB"),
    "trtllm-llama33-70b-fp4-b200-tp1-tps": dict(gpus=1, evidence="1x NVIDIA B200 180GB"),
    # --- multi-GPU aggregate: divide by the GPU count -----------------------
    "trtllm-llama33-70b-fp8-h100-tp2-1k2k-tps": dict(gpus=2, evidence="2x NVIDIA H100 SXM 80GB"),
    "trtllm-llama33-70b-fp8-h200-tp2-1k2k-tps": dict(gpus=2, evidence="2x NVIDIA H200 SXM 141GB"),
    "trtllm-llama31-405b-fp8-h100-tp8-tps": dict(gpus=8, evidence="8x NVIDIA H100 SXM 80GB"),
    "mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-h100": dict(gpus=8, evidence="8x H100 SXM on Xeon"),
    "mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-h100": dict(gpus=8, evidence="8x H100 SXM on a Xeon host"),
    "vllm-llama31-405b-fp8-8xh100-tp8-output-tps": dict(gpus=8, evidence="8x H100 on a single Lambda 1-Click node"),
    "sglang-deepseek-v3-96xh100-output-tps-per-node": dict(gpus=8, evidence="output (decode) tokens/s per 8-GPU node (aggregate)"),
    "h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-energy-per-token": dict(gpus=8, evidence="8x NVIDIA H200-SXM-141GB on ONE node"),
    "h200-8x-mlperf-v4-1-maxq-mixtral-8x7b-offline-energy-per-token": dict(gpus=8, evidence="SAME closed-division system (H200-SXM-141GBx8_TRT_MaxQ)"),
    "h200-8x-mlperf-v4-1-maxp-llama2-70b-offline-vs-maxq": dict(gpus=8, evidence="NVIDIA NON-POWER submission system H200-SXM-141GBx8_TRT"),
    "h200-8x-mlperf-v4-1-maxq-llama2-70b-server-energy-per-token": dict(gpus=8, evidence="H200-SXM-141GBx8_TRT_MaxQ"),
    # --- already per-GPU by its own unit string -----------------------------
    "perplexity-llama2-70b-h100-fp8-tp2-bs128-tps-per-gpu": dict(gpus=None, evidence="output tokens/s per GPU"),
    # --- AMD Instinct, joined 2026-10-05 -----------------------------------
    # The price existed (Oracle, per-GPU-hour) from 2026-10-05, but these rows
    # still could not join until each declared its GPU count, because the tool
    # refuses to guess a divisor. A price is not enough: an aggregate rate over
    # an unknown number of devices is not a per-device rate.
    "bench-mlperf-v6-1-amd-mi355x-llama2-70b-offline": dict(
        gpus=8, evidence="8x AMD Instinct MI355X 288GB HBM3e"),
    "llama-cpp-mi300x-deepseek-v3-671b-q4-decode-tok-s": dict(
        gpus=8, evidence="8x AMD Instinct MI300X"),
    # ---- AMD Instinct, joined 2026-10-06 (defect D2) --------------------
    # Every entry below names the GPU count as a LITERAL substring of that
    # record's own unit/methodology, which is what check_evidence() enforces.
    # The two clusters that still do NOT price and must not be forced:
    #   - mi300x-vs-h100-vllm-llama31-{405b,70b}-fp8-tp8-output-throughput
    #     name TWO platforms at once and are correctly unpriceable as single rows.
    #   - mi325x-mlperf-v5-0-mangoboost-llama2-70b-offline says "4 nodes" and
    #     never states GPUs per node, so the total device count is unknown.
    # The last one is the row that proves the scheme works: 87 GPUs, not 8.
    "mi355x-mlperf-v6-0-llama2-70b-87gpu-offline-tokens": dict(
        gpus=87, evidence="87 GPUs across 11 nodes"),
    "mi355x-mlperf-v6-0-llama2-70b-wmxfp4-offline-tokens": dict(
        gpus=8, evidence="Hardware: 8x AMD Instinct MI355X, 288GB HBM3E each"),
    "mi355x-mlperf-v6-0-gpt-oss-120b-offline-tokens": dict(
        gpus=8,
        evidence="Same 8xMI355X / 8xEPYC 9575F system and same software stack "
                 "as the Llama 2 70B Offline record"),
    "bench-mlperf-v6-0-dell-mangoboost-mi355x-powercap-1000w-offline": dict(
        gpus=8,
        evidence="Hardware: 1 node, 8x AMD Instinct MI355X 288GB HBM3e, "
                 "Dell PowerEdge XE9785."),
    "bench-mlperf-v6-1-oracle-mi355x-llama3-1-8b-server": dict(
        gpus=8, evidence="SERVER scenario on Oracle's 8xMI355X_2xEPYC_9575F node"),
    # ---- AMD Instinct, joined 2026-10-06 after the methodology edit -------
    # These four records state their GPU count only in the record NAME
    # ("AMD 8xMI355X", "Dell XE9785L 8xMI355X"), which check_evidence() cannot
    # see. The count is now also in methodology, quoting the MLPerf system id
    # the row was run on, so each entry is provable from the record itself.
    "bench-mlperf-v6-1-amd-mi355x-llama2-70b-server": dict(
        gpus=8, evidence="8xAMD Instinct MI355X 288GB HBM3e"),
    "bench-mlperf-v6-1-amd-mi355x-llama2-70b-interactive": dict(
        gpus=8, evidence="8xAMD Instinct MI355X 288GB HBM3e"),
    "bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-server": dict(
        gpus=8, evidence="8xAMD Instinct MI355X 288GB HBM3e"),
    "bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-interactive": dict(
        gpus=8, evidence="8xAMD Instinct MI355X 288GB HBM3e"),
    # Its OFFLINE sibling, bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-offline,
    # already carried its count in methodology ("Hardware: 1 node, 8x AMD
    # Instinct MI355X 288GB HBM3e, Dell PowerEdge XE9785L."), so it needed no
    # edit at all - it is a BENCH_JOIN-only row, like the four below.
    "bench-mlperf-v6-1-dell-mi355x-llama3-1-8b-offline": dict(
        gpus=8,
        evidence="Hardware: 1 node, 8x AMD Instinct MI355X 288GB HBM3e, "
                 "Dell PowerEdge XE9785L."),
    # ---- NVIDIA MLPerf, joined 2026-10-10 --------------------------------
    # Each entry names the GPU count as a LITERAL substring of that record's
    # own unit/methodology, which is what check_evidence() enforces.
    "bench-mlperf-v5-1-cisco-h200-tgp700w-llama2-70b-offline": dict(
        gpus=8,
        evidence="Hardware: 1 node, 8x NVIDIA H200-SXM-141GB, Cisco UCS C880A M8."),
    "bench-mlperf-v6-1-coreweave-gb300-nvl72-llama2-70b-offline": dict(
        gpus=72,
        evidence="Hardware: 72x NVIDIA GB300-288GB aarch64, 18 nodes, 4 per node."),
    "bench-mlperf-v6-1-nvidia-b300-llama2-70b-offline": dict(
        gpus=8,
        evidence="Hardware: 8x NVIDIA B300-SXM-270GB, 1 node"),
    # The GB300 8-GPU slice rows ran on 8 of the NVL72's 72 GPUs. The divisor
    # is the slice size (8), not the whole rack (72) - the record's own
    # methodology says "8-GPU SLICE" and "taken across 2 nodes with 4
    # accelerators per node".
    "bench-mlperf-v6-1-nvidia-gb300-8gpu-llama2-70b-offline": dict(
        gpus=8,
        evidence="on an 8-GPU SLICE of a GB300 NVL72 (72 GPUs total on the NVL72)"),
}

EXCLUDED: dict[str, str] = {
    # --- aggregate value is zero on purpose; it is a power record ----------
    "h200-8x-mlperf-v4-1-maxq-llama2-70b-offline-idle-node-draw":
        "value is 0 by design (a whole-node draw record at idle) - dividing by it would give an infinite cost",
    # --- throughput is TOTAL tokens (prompt + generated), not output -------
    "silex-llama33-70b-4xh100-tp4-tps-200x200":
        "unit is 'total tokens/s (prompt + generated)' at 200-in/200-out, so the prompt share is not separable from the record - cost per OUTPUT token is not derivable",
    "vllm-qwen38-nvfp4-gb300-nvl72-tps-per-gpu":
        "unit is 'total tokens/s per GPU' (prompt + generated) - not an output-token rate",
    # --- per-user figure with no matching aggregate, and vice versa ---------
    "vllm-qwen38-nvfp4-gb300-nvl72-interactivity":
        "single-user 180 tok/s exists but its sibling aggregate is a TOTAL-tokens figure, so the pair needed for $/Mtok output does not exist",
    # --- the record's accelerator_ids name TWO platforms at once -----------
    "mi300x-vs-h100-vllm-llama31-405b-fp8-tp8-output-throughput":
        "accelerator_ids lists MI300X and H100 together (it is the head-to-head row); the H100-only and MI300X-only twins carry the split values",
    "mi300x-vs-h100-vllm-llama31-70b-fp8-tp8-output-throughput":
        "accelerator_ids lists MI300X and H100 together (it is the head-to-head row); the H100-only twin carries the H100 value",
    # --- no accelerator ids at all, so no price can join -------------------
    "mpt7b-a100-bs1-per-user-decode-tps":
        "accelerator_ids is empty - the A100 is named in the id but not as a record id, and no $/A100-hour can be joined",
    "mpt7b-a100-bs64-per-user-decode-tps":
        "accelerator_ids is empty - no price join; also the single-user counterpart of an aggregate record",
    "mpt7b-a100-bs64-aggregate-output-tps":
        "accelerator_ids is empty - no price join",
    # --- GPU count is genuinely unknown, not merely unstated ---------------
    "mi325x-mlperf-v5-0-mangoboost-llama2-70b-offline":
        "the methodology states 4 nodes but never states GPUs per node, and the record itself says the per-node GPU count, TP size and engine are all absent from the source - so the divisor is unknown and no divisor may be inferred from the sibling MI300X submissions",
}


def load(dirname: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    d = ROOT / "data" / dirname
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out[f.stem] = rec
    return out


def check_evidence() -> list[str]:
    """Every quoted price basis and GPU-count claim must exist verbatim in its
    record. Returns a list of problems; empty means the whole join is sourced.

    SCOPE, deliberately narrow. This fails only when a record EXISTS and
    contradicts the join: a quoted figure that is not in the record, a
    benchmark whose metric is not throughput, an accelerator the supply record
    does not claim to carry, or a price_usd that is null.

    A join spec naming a record that is ABSENT is not a contradiction - it is a
    gap, and a gap is handled by reporting it (`--gaps`), not by crashing. That
    distinction is what lets the same table run against a partial fixture repo
    in the tests while still catching a silent 8x price-basis error in the real
    data.
    """
    supply = load(DIRS["supply"])
    bench = load(DIRS["benchmark"])
    problems: list[str] = []
    for sid, quotes in PRICE_QUOTES.items():
        rec = supply.get(sid)
        if rec is None:
            continue  # absent record = gap, reported by cmd_gaps
        blob = f"{rec.get('price_basis') or ''}\n{rec.get('notes') or ''}"
        for q in quotes:
            if q["basis"] not in blob:
                problems.append(
                    f"{sid}: quoted basis not found in record: {q['basis']!r}")
            if rec.get("price_usd") is None:
                problems.append(f"{sid}: has a price quote but price_usd is null")
            if q["accel"] not in (rec.get("accelerator_ids") or []):
                problems.append(
                    f"{sid}: accelerator {q['accel']} is not in the record's accelerator_ids")
    for bid, spec in BENCH_JOIN.items():
        rec = bench.get(bid)
        if rec is None:
            continue  # absent record = gap, reported by cmd_gaps
        if rec.get("metric") not in THROUGHPUT_METRICS:
            problems.append(f"{bid}: metric {rec.get('metric')!r} is not a throughput metric")
        if not float(rec.get("value") or 0) > 0:
            problems.append(f"{bid}: throughput value is {rec.get('value')!r}; "
                            "a non-positive rate cannot be divided")
        blob = f"{rec.get('unit') or ''}\n{rec.get('methodology') or ''}"
        if spec["evidence"] not in blob:
            problems.append(
                f"{bid}: GPU-count evidence not found in record: {spec['evidence']!r}")
    for bid in EXCLUDED:
        if bid not in bench:
            continue  # excluded-and-absent is harmless
    return problems


def concurrency_class(metric: str, unit: str = "") -> str:
    """aggregate = summed over in-flight requests. single-user = what one
    request sees. These must never share a column: the repo's own batch-1 vs
    batch-64 pair has aggregate 13.9x HIGHER and per-user 4.6x LOWER.

    The record's own `unit` string WINS over the metric enum. `metric` is a
    coarse enum and the repo shows at least one record where the enum and the
    prose disagree, so the prose is the authoritative statement of what was
    actually measured.
    """
    text = (unit or "").lower()
    if "aggregate" in text:
        return "aggregate"
    if "per user" in text or "per-request" in text or "per-request" in text:
        return "single-user"
    return "single-user" if metric in ("decode_tok_s", "tok_s_per_user") else "aggregate"


def build_rows() -> tuple[list[dict], list[dict]]:
    supply = load(DIRS["supply"])
    bench = load(DIRS["benchmark"])
    rows, unmatched = [], []
    for bid, spec in sorted(BENCH_JOIN.items()):
        b = bench.get(bid)
        if b is None:
            continue  # gap, not a number: reported by cmd_gaps
        if b.get("metric") not in THROUGHPUT_METRICS:
            continue  # check_evidence already rejects this loudly
        gp = spec["gpus"]
        per_gpu_tps = float(b["value"]) / gp if gp else float(b["value"])
        # An accelerator list that mixes platforms cannot be priced per-GPU:
        # a rate divided by "GPUs" is meaningless when the GPUs differ.
        accels = list(b.get("accelerator_ids") or [])
        if not accels:
            unmatched.append(dict(bench=bid, reason="benchmark names no accelerator"))
            continue
        for accel in accels:
            quotes = [q for q in PRICE_QUOTES.get("", []) if False]
            hits = [(sid, q) for sid, qs in PRICE_QUOTES.items()
                    for q in qs if q["accel"] == accel]
            if not hits:
                unmatched.append(dict(
                    bench=bid, accel=accel,
                    reason=f"no supply record prices {accel} per GPU-hour"))
                continue
            for sid, q in hits:
                if sid not in supply:
                    continue  # gap: the price record itself is absent
                usd_mtok = q["usd"] / per_gpu_tps * 1_000_000
                rows.append(dict(
                    provider=sid, accelerator=accel, model=b.get("model"),
                    engine=b.get("engine_id"), metric=b["metric"],
                    concurrency=concurrency_class(b["metric"], b.get("unit")),
                    gpus_covered=gp,
                    tok_s_total=float(b["value"]), tok_s_per_gpu=per_gpu_tps,
                    usd_per_gpu_hour=q["usd"], tier=q["tier"],
                    usd_per_mtok=usd_mtok,
                    supply_ref=f"supply/{sid}", bench_ref=f"benchmarks/{bid}",
                    measured_by=b.get("measured_by"),
                    bench_status=b.get("status"), supply_status=supply[sid].get("status"),
                    reproducible=b.get("reproducible"),
                    unit=b.get("unit"),
                ))
    rows.sort(key=lambda r: r["usd_per_mtok"])
    return rows, unmatched


def cmd_table(args) -> None:
    problems = check_evidence()
    if problems:
        if not args.force:
            print("UNSOURCED INPUT - refusing to print numbers. Each line names a "
                  "quote or a GPU count that is not present verbatim in its record:")
            for p in problems:
                print(f"  ! {p}")
            print("\nRe-run with --force to print anyway (do not do that in a doc).")
            raise SystemExit(1)
        # --force still says so loudly. A forced print is a debugging aid, not
        # a licence to be quiet about which inputs could not be verified.
        print(f"UNSOURCED INPUT - {len(problems)} problem(s), printing anyway "
              f"because --force was passed. DO NOT QUOTE THESE NUMBERS:")
        for p in problems:
            print(f"  ! {p}")
        print()
    rows, _ = build_rows()
    if args.single_user:
        rows = [r for r in rows if r["concurrency"] == "single-user"]
    if args.tier:
        rows = [r for r in rows if r["tier"] in args.tier]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return
    print(f"{len(rows)} cost-per-token rows. "
          "AGGREGATE and single-user are separate populations: an aggregate "
          "tok/s is summed over all in-flight requests and is a fleet-average "
          "cost, not what one user pays.\n")
    hdr = (f"{'USD/Mtok':>9} {'concurrency':<12} {'tier':<22} {'provider':<28} "
           f"{'accel':<22} {'model':<34} {'tok/s/GPU':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        model = (r["model"] or "")[:33]
        print(f"{r['usd_per_mtok']:9.4f} {r['concurrency']:<12} {r['tier']:<22} "
              f"{r['provider']:<28} {r['accelerator']:<22} {model:<34} "
              f"{r['tok_s_per_gpu']:11.2f}")


def cmd_explain(args) -> None:
    rows, _ = build_rows()
    seen: set[tuple] = set()
    for r in rows:
        key = (r["bench_ref"], r["supply_ref"], r["usd_per_gpu_hour"])
        if key in seen:
            continue
        seen.add(key)
        g = r["gpus_covered"]
        print(f"\n{r['provider']}  x  {r['bench_ref'].split('/')[1]}")
        print(f"  model         {r['model']}  ({r['engine']}, "
              f"{r['metric']}, {r['concurrency']}, measured_by={r['measured_by']})")
        if g:
            print(f"  step 1  price     ${r['usd_per_gpu_hour']:.4f} per GPU-hour "
                  f"[{r['tier']}]   <- {r['supply_ref']} price_basis")
            print(f"  step 2  throughput {r['tok_s_total']:.2f} tok/s aggregate "
                  f"across {g} GPU(s) = {r['tok_s_per_gpu']:.2f} tok/s per GPU "
                  f"<- {r['bench_ref']}")
        else:
            print(f"  step 1  price     ${r['usd_per_gpu_hour']:.4f} per GPU-hour "
                  f"[{r['tier']}]   <- {r['supply_ref']} price_basis")
            print(f"  step 2  throughput {r['tok_s_per_gpu']:.2f} tok/s per GPU, "
                  f"already per-GPU in the record's own unit string "
                  f"<- {r['bench_ref']}")
        print(f"  step 3  ${r['usd_per_gpu_hour']:.4f} / {r['tok_s_per_gpu']:.2f} "
              f"tok/s/GPU x 1,000,000 = ${r['usd_per_mtok']:.4f} per million output tokens")


def cmd_gaps(args) -> None:
    supply = load(DIRS["supply"])
    bench = load(DIRS["benchmark"])
    priced_accels = {q["accel"] for qs in PRICE_QUOTES.values() for q in qs}
    print("THROUGHPUT BENCHMARKS THAT CANNOT BE PRICED\n")
    for bid in sorted(bench):
        rec = bench[bid]
        if rec.get("metric") not in THROUGHPUT_METRICS or rec.get("value") in (None, 0):
            continue
        if bid in BENCH_JOIN:
            continue
        accels = list(rec.get("accelerator_ids") or [])
        if not accels:
            reason = EXCLUDED.get(bid, "benchmark names no accelerator")
        elif all(a not in priced_accels for a in accels):
            reason = EXCLUDED.get(
                bid, f"no priced per-GPU-hour supply record for {', '.join(accels)}")
        elif bid in EXCLUDED:
            # A row can be withheld deliberately even when its accelerator is
            # priced: the mixed-platform pair below, and mi325x-mlperf-v5-0,
            # whose methodology quotes "8xMI325X" only as a CORROBORATING
            # CONTEXT sentence about a DIFFERENT submission and whose own device
            # count is genuinely unknown. The explicit reason outranks any
            # pattern-match, because the explicit reason is the one a human
            # checked the record for.
            reason = EXCLUDED[bid]
        else:
            # The accelerator IS priced, so the only thing standing between this
            # row and a price is the device count the aggregate covers. The fall
            # used to read "see EXCLUDED", which sent readers to a dict that did
            # not contain the row and did not describe its blocker. The message
            # must name what this specific row is actually missing.
            if len(accels) > 1:
                # Several DISTINCT platforms in one row: a tok/s divided by
                # "GPUs" is meaningless when the GPUs differ, so no count would
                # make this row priceable. The split twins carry the value.
                reason = (f"names several platforms at once ({', '.join(accels)}) - "
                          f"a rate divided by a device count means nothing when "
                          f"the devices differ; the single-platform twins carry "
                          f"the split values")
            else:
                stated = GPU_COUNT_HINT.search(
                    f"{rec.get('unit') or ''}\n{rec.get('methodology') or ''}")
                if stated:
                    # A count is ON DISPLAY but the tool still will not use it
                    # unless it is quoted verbatim in BENCH_JOIN, so this is a
                    # one-line join fix, not a data gap.
                    reason = (f"GPU count is stated in the record "
                              f"({stated.group(0)}) but is not in BENCH_JOIN - "
                              f"add the entry there")
                else:
                    reason = ("no declared GPU count - see BENCH_JOIN "
                              "(the record's unit/methodology does not state how "
                              "many devices the aggregate covers)")
        print(f"  benchmarks/{bid}")
        print(f"      {rec['model']}  {rec['value']} {rec.get('unit','')}")
        print(f"      -> {reason}\n")
    print("PRICED ACCELERATORS WITH NO THROUGHPUT RECORD AT ALL")
    with_bench = {a for b in bench.values() if b.get("metric") in THROUGHPUT_METRICS
                  for a in (b.get("accelerator_ids") or [])}
    for accel in sorted(priced_accels - with_bench):
        holders = sorted(sid for sid, qs in PRICE_QUOTES.items()
                         if any(q["accel"] == accel for q in qs))
        print(f"  {accel}: priced by {len(holders)} record(s) "
              f"({', '.join(holders)}) but no throughput record -> "
              f"cost per token = no record")
    print("\nPRICES EXCLUDED FROM THE JOIN ON PURPOSE")
    for sid in sorted(set(load(DIRS["supply"])) - set(PRICE_QUOTES)):
        rec = supply[sid]
        if rec.get("price_usd") is None:
            continue
        print(f"  supply/{sid}: {rec['price_usd']} "
              f"[{rec.get('kind')}] {str(rec.get('price_basis'))[:70]}...")
    print("  (retail/resale prices are per CARD, not per GPU-hour, and have no "
          "amortised runtime in the repo, so they cannot yield a cost per token)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The bare `python tools/cost_per_token.py` form is the table, so the
    # table's options are declared on the TOP-LEVEL parser: argparse does not
    # populate a subparser's defaults when no subcommand is named.
    ap.add_argument("--format", choices=["table", "json"], default="table")
    ap.add_argument("--single-user", action="store_true",
                    help="only rows whose throughput is a single user's rate")
    ap.add_argument("--tier", action="append",
                    help="filter by price tier, e.g. --tier spot")
    ap.add_argument("--force", action="store_true",
                    help="print even if a quote fails its evidence check")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("explain", help="show the arithmetic for every distinct row"
                   ).set_defaults(fn=cmd_explain)
    sub.add_parser("gaps", help="what these records cannot answer"
                   ).set_defaults(fn=cmd_gaps)
    args = ap.parse_args()
    getattr(args, "fn", cmd_table)(args)