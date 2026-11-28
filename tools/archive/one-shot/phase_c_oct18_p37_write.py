#!/usr/bin/env python3
"""Phase C Oct-18 p37: Apple Silicon base/Pro chips missing from the accelerator set.

llama.cpp discussion #4167 maintainer scoreboard (build 8e672ef, Llama-2-7B,
pp512/tg128, Metal, -ngl 99). Six chips in that table had measured rows and no
accelerator record, so the numbers were never filed:

  M1 7-core and 8-core GPU (F16 column blank — only Q8_0 and Q4_0)
  M1 Pro 14-core and 16-core GPU
  M2 8-core and 10-core GPU (F16 column blank)
  M2 Pro 16-core and 19-core GPU
  M3 8-core and 10-core GPU (F16 column blank)
  M3 Pro 14-core GPU (the 18-core M3 Pro is already local-apple-m3-pro)

Bandwidth is the figure the scoreboard prints, which is Apple's published
unified-memory bandwidth for that die. Two GPU bins of one die share one
bandwidth, so they share one accelerator record; the GPU-core count lives in
each benchmark's methodology. Blank F16 cells are not invented.
"""
import json
import os

ROOT = "/opt/data/projects/inference-research/repo"
TODAY = "2026-10-18"
SRC = "llama-cpp-discussion-4167-apple-silicon-llama-bench-table"

CHIPS = {
    "local-apple-m1": {
        "name": "Apple M1 (base, 7- or 8-core GPU)",
        "year": 2020, "nm": 5, "vram": 16, "bw": 68,
        "mem": "lpddr4x unified", "arch": "m-series (M1)",
        "bw_basis": (
            "68 GB/s is the figure the llama.cpp #4167 scoreboard prints for both "
            "the 7-core and 8-core GPU M1, and it is Apple's published unified-memory "
            "bandwidth for the base M1 (LPDDR4X-4266). One die, two GPU bins; they "
            "share this bandwidth, so they share this record."
        ),
        "notes": (
            "Base M1 as binned in the #4167 scoreboard: 7-core GPU and 8-core GPU, "
            "both at 68 GB/s. Up to 16 GB unified memory. The 7-core bin is the "
            "entry MacBook Air; the 8-core bin is the Mac mini / 13-inch MacBook Pro "
            "/ iMac. M1 Pro and M1 Max already have their own records. memory_bus_bit "
            "is null: Apple publishes bandwidth, not a bus width, and none is "
            "back-computed here."
        ),
    },
    "local-apple-m1-pro": {
        "name": "Apple M1 Pro (14- or 16-core GPU)",
        "year": 2021, "nm": 5, "vram": 32, "bw": 200,
        "mem": "lpddr5 unified", "arch": "m-series (M1 Pro)",
        "bw_basis": (
            "200 GB/s is the figure the llama.cpp #4167 scoreboard prints for both "
            "the 14-core and 16-core GPU M1 Pro, and it is Apple's published "
            "unified-memory bandwidth for M1 Pro. One die, two GPU bins."
        ),
        "notes": (
            "M1 Pro as binned in the #4167 scoreboard: 14-core and 16-core GPU, both "
            "at 200 GB/s, up to 32 GB unified memory (MacBook Pro 14/16-inch, 2021). "
            "Distinct from apple-m1-max (400 GB/s). memory_bus_bit is null; not "
            "back-computed from bandwidth."
        ),
    },
    "local-apple-m2": {
        "name": "Apple M2 (base, 8- or 10-core GPU)",
        "year": 2022, "nm": 5, "vram": 24, "bw": 100,
        "mem": "lpddr5 unified", "arch": "m-series (M2)",
        "bw_basis": (
            "100 GB/s is the figure the llama.cpp #4167 scoreboard prints for both "
            "the 8-core and 10-core GPU M2, and it is Apple's published unified-memory "
            "bandwidth for the base M2. One die, two GPU bins."
        ),
        "notes": (
            "Base M2 as binned in the #4167 scoreboard: 8-core and 10-core GPU, both "
            "at 100 GB/s, up to 24 GB unified memory (MacBook Air / 13-inch MacBook "
            "Pro / Mac mini, 2022-2023). M2 Pro and M2 Max already have their own "
            "records. memory_bus_bit is null; not back-computed from bandwidth."
        ),
    },
    "local-apple-m2-pro": {
        "name": "Apple M2 Pro (16- or 19-core GPU)",
        "year": 2023, "nm": 5, "vram": 32, "bw": 200,
        "mem": "lpddr5 unified", "arch": "m-series (M2 Pro)",
        "bw_basis": (
            "200 GB/s is the figure the llama.cpp #4167 scoreboard prints for both "
            "the 16-core and 19-core GPU M2 Pro, and it is Apple's published "
            "unified-memory bandwidth for M2 Pro. One die, two GPU bins."
        ),
        "notes": (
            "M2 Pro as binned in the #4167 scoreboard: 16-core and 19-core GPU, both "
            "at 200 GB/s, up to 32 GB unified memory (MacBook Pro 14/16-inch and Mac "
            "mini, 2023). Distinct from apple-m2-max. memory_bus_bit is null; not "
            "back-computed from bandwidth."
        ),
    },
    "local-apple-m3": {
        "name": "Apple M3 (base, 8- or 10-core GPU)",
        "year": 2023, "nm": 3, "vram": 24, "bw": 100,
        "mem": "lpddr5 unified", "arch": "m-series (M3)",
        "bw_basis": (
            "100 GB/s is the figure the llama.cpp #4167 scoreboard prints for both "
            "the 8-core and 10-core GPU M3, and it is Apple's published unified-memory "
            "bandwidth for the base M3. One die, two GPU bins. A secondary summary "
            "table rounds this to 102 GB/s; the scoreboard and Apple's own figure "
            "used here are 100."
        ),
        "notes": (
            "Base M3 as binned in the #4167 scoreboard: 8-core and 10-core GPU, both "
            "at 100 GB/s, up to 24 GB unified memory (iMac / MacBook Pro 14-inch / "
            "MacBook Air, late 2023). Distinct from local-apple-m3-pro (150 GB/s) and "
            "apple-m3-max. memory_bus_bit is null; not back-computed from bandwidth."
        ),
    },
    "local-apple-m3-pro-14core": {
        "name": "Apple M3 Pro (14-core GPU bin)",
        "year": 2023, "nm": 3, "vram": 36, "bw": 150,
        "mem": "lpddr5 unified", "arch": "m-series (M3 Pro)",
        "bw_basis": (
            "150 GB/s is the figure the llama.cpp #4167 scoreboard prints for the "
            "14-core GPU M3 Pro, and it is Apple's published unified-memory bandwidth "
            "for M3 Pro (same bandwidth as the 18-core bin). Filed separately from "
            "local-apple-m3-pro because that record describes the 18-core GPU "
            "configuration, and the 14-core bin measures a different GPU."
        ),
        "notes": (
            "The binned M3 Pro: 11-core CPU (5P+6E) and 14-core GPU, 150 GB/s, up to "
            "36 GB unified memory (14-inch MacBook Pro, late 2023). local-apple-m3-pro "
            "is the 18-core GPU / 12-core CPU bin of the same die and is NOT this "
            "record. Same published bandwidth, fewer GPU cores; the F16 column of the "
            "scoreboard is blank for this bin and is not filed. memory_bus_bit is "
            "null; not back-computed from bandwidth."
        ),
    },
}


def dump(rec, subdir):
    out = os.path.join(ROOT, "data", subdir, rec["id"] + ".json")
    if os.path.exists(out):
        print("SKIP", rec["id"])
        return False
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print("wrote", subdir, rec["id"])
    return True


def accelerator(aid):
    c = CHIPS[aid]
    return {
        "id": aid, "type": "accelerator", "name": c["name"],
        "status": "verified", "confidence": 0.8, "updated": TODAY,
        "vendor": "apple", "architecture": c["arch"],
        "release_year": c["year"], "process_nm": c["nm"],
        "form_factors": ["soc"], "vram_gb": c["vram"],
        "memory_type": c["mem"], "memory_bus_bit": None,
        "memory_bandwidth_gbps": c["bw"],
        "memory_bandwidth_basis": c["bw_basis"],
        "onchip_cache": None, "flops": [], "tdp_w": None,
        "interconnect": [], "unified_memory": True, "consumer": True,
        "notes": c["notes"], "sources": [SRC],
    }


# (accelerator, gpu_cores, bw_gbps, quant, pp512, tg128)
# F16 cells the scoreboard leaves blank are absent from this table.
TABLE = [
    ("local-apple-m1", 7, 68, "q8-0", 108.21, 7.92),
    ("local-apple-m1", 7, 68, "q4-0", 107.81, 14.19),
    ("local-apple-m1", 8, 68, "q8-0", 117.25, 7.91),
    ("local-apple-m1", 8, 68, "q4-0", 117.96, 14.15),
    ("local-apple-m1-pro", 14, 200, "f16", 262.65, 12.75),
    ("local-apple-m1-pro", 14, 200, "q8-0", 235.16, 21.95),
    ("local-apple-m1-pro", 14, 200, "q4-0", 232.55, 35.52),
    ("local-apple-m1-pro", 16, 200, "f16", 302.14, 12.75),
    ("local-apple-m1-pro", 16, 200, "q8-0", 270.37, 22.34),
    ("local-apple-m1-pro", 16, 200, "q4-0", 266.25, 36.41),
    ("local-apple-m2", 8, 100, "q8-0", 147.27, 12.18),
    ("local-apple-m2", 8, 100, "q4-0", 145.91, 21.70),
    ("local-apple-m2", 10, 100, "q8-0", 181.40, 12.21),
    ("local-apple-m2", 10, 100, "q4-0", 179.57, 21.91),
    ("local-apple-m2-pro", 16, 200, "f16", 312.65, 12.47),
    ("local-apple-m2-pro", 16, 200, "q8-0", 288.46, 22.70),
    ("local-apple-m2-pro", 16, 200, "q4-0", 294.24, 37.87),
    ("local-apple-m2-pro", 19, 200, "f16", 384.38, 13.06),
    ("local-apple-m2-pro", 19, 200, "q8-0", 344.50, 23.01),
    ("local-apple-m2-pro", 19, 200, "q4-0", 341.19, 38.86),
    ("local-apple-m3", 8, 100, "q8-0", 158.43, 12.12),
    ("local-apple-m3", 8, 100, "q4-0", 157.89, 21.62),
    ("local-apple-m3", 10, 100, "q8-0", 187.52, 12.27),
    ("local-apple-m3", 10, 100, "q4-0", 186.75, 21.34),
    ("local-apple-m3-pro-14core", 14, 150, "q8-0", 272.11, 17.44),
    ("local-apple-m3-pro-14core", 14, 150, "q4-0", 269.49, 30.65),
]

QUANT = {
    "f16": ("F16", None, "Llama-2-7B F16"),
    "q8-0": ("Q8_0", "gguf-q8-0", "Llama-2-7B gguf-q8-0.gguf"),
    "q4-0": ("Q4_0", "gguf-q4-0", "Llama-2-7B gguf-q4-0.gguf"),
}

SHORT = {
    "local-apple-m1": "m1",
    "local-apple-m1-pro": "m1-pro",
    "local-apple-m2": "m2",
    "local-apple-m2-pro": "m2-pro",
    "local-apple-m3": "m3",
    "local-apple-m3-pro-14core": "m3pro14",
}


def bench(aid, cores, bw, quant, pp, tg):
    label, fmt, model = QUANT[quant]
    short = SHORT[aid]
    chip = CHIPS[aid]["name"]
    method = (
        "COMMUNITY-SCOREBOARD (llama.cpp discussion #4167, ggerganov-maintained). "
        "llama-bench -hf ggml-org/Llama-2-7B-GGUF -p 512 -n 128 -ngl 99 --delay 30, "
        "Metal backend, full GPU offload. Build commit 8e672ef. pp512 = prefill of one "
        "prompt at batch 512; tg128 = batch-1 decode of 128 tokens. mean over the "
        "default -r 5. Per-request, NOT aggregate. Chip: %s, %d GPU cores, %d GB/s "
        "unified-memory bandwidth as printed in the scoreboard."
    ) % (chip, cores, bw)
    out = []
    for metric, value, test in (
        ("prefill_tok_s", pp, "pp512"),
        ("decode_tok_s", tg, "tg128"),
    ):
        rid = "llamacpp-metal-%s-%dcore-llama27b-%s-%s-8e672ef" % (short, cores, quant, test)
        unit = "tok/s (%s, %s)" % (
            test, "batch 512, per-request" if test == "pp512" else "batch 1, per-request")
        out.append({
            "id": rid, "type": "benchmark",
            "name": "llama.cpp Metal on %s (%d GPU): Llama-2-7B %s %s tok/s (scoreboard #4167, 8e672ef)" % (
                chip, cores, label, test),
            "engine_id": "llama-cpp",
            "accelerator_ids": [aid],
            "interconnect_ids": [],
            "model": model,
            "format_id": fmt,
            "metric": metric,
            "value": value,
            "unit": unit,
            "methodology": method,
            "measured_by": "third_party",
            "reproducible": True,
            "energy_basis": None, "power_w": None, "power_scope": None,
            "power_cap": None, "clock_lock_mhz": None, "thermal_state": None,
            "notes": (
                "Maintainer-curated Apple Silicon llama-bench table (discussion #4167), "
                "the same Llama-2-7B weights and pp512/tg128 protocol as the CUDA/ROCm "
                "scoreboards. This row is the %d-core GPU bin; the other bin of the same "
                "die is a separate record. F16 cells the scoreboard leaves blank are not "
                "filed."
            ) % cores,
            "status": "verified", "confidence": 0.8, "updated": TODAY,
            "sources": [SRC],
        })
    return out


def main():
    n_acc = n_bench = 0
    for aid in CHIPS:
        if dump(accelerator(aid), "accelerators"):
            n_acc += 1
    for row in TABLE:
        for rec in bench(*row):
            if dump(rec, "benchmarks"):
                n_bench += 1
    print("accelerators", n_acc, "benchmarks", n_bench)


if __name__ == "__main__":
    main()
