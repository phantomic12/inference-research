#!/usr/bin/env python3
"""Write a skeleton record with the right fields and defaults.

    python tools/new_record.py accelerator --name "NVIDIA H100 SXM" \
        --source nvidia-h100-sxm-datasheet
    python tools/new_record.py source --name "H100 datasheet" \
        --url https://... --kind spec-sheet --publisher NVIDIA

Writes data/<dir>/<slug>.json. Refuses to clobber unless --force.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TODAY = date.today().isoformat()

TEMPLATES = {
    "accelerator": {
        "vendor": None, "architecture": None, "release_year": None, "process_nm": None,
        "form_factors": [], "vram_gb": None, "memory_type": None, "memory_bus_bit": None,
        "memory_bandwidth_gbps": None, "memory_bandwidth_basis": None,
        "flops": [], "tdp_w": None, "interconnect": [], "unified_memory": None,
        "consumer": None, "notes": "",
    },
    "flop": {
        "class": None, "arithmetic_intensity": None, "bound_by": None,
        "scales_with": [], "affected_by_hardware": [], "workarounds": [], "notes": "",
    },
    "engine": {
        "repo": None, "languages": [], "license": None, "first_release_year": None,
        "design_goal": [], "scheduling": [], "supported_backends": [],
        "supported_formats": [], "notable_features": [], "hardware_caveats": [],
        "best_for": [], "avoid_for": [], "docs_url": None, "notes": "",
    },
    "quantization": {
        "scheme": None, "bits": None, "weight_group_size": None,
        "activation_scheme": None, "native_support": [], "emulated_support": [],
        "kernels": [], "quality_delta": None, "notes": "",
    },
    "interconnect": {
        "kind": None, "version": None, "bandwidth_gbps": None, "bandwidth_basis": None,
        "link_count": None, "topology": None, "scale_up": None, "scale_out": None,
        "switching": None, "notes": "",
    },
    "benchmark": {
        "engine_id": None, "accelerator_ids": [], "interconnect_ids": [],
        "model": None, "format_id": None, "metric": None, "value": None,
        "unit": None, "methodology": None, "measured_by": None,
        "reproducible": None, "notes": "",
    },
    "gotcha": {
        "class": None, "affects": [], "symptom": None, "root_cause": None,
        "workaround": None, "severity": None, "notes": "",
    },
    "source": {
        "url": None, "publisher": None, "kind": None, "published": None,
        "accessed": TODAY, "archived_url": None, "notes": "",
    },
}


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:64] or "unnamed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("type", choices=TYPES)
    ap.add_argument("--name", required=True)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--source", action="append", default=[], help="source id; repeatable")
    ap.add_argument("--url", default=None)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--publisher", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    slug = args.slug or slugify(args.name)
    out = ROOT / "data" / DIRS[args.type] / f"{slug}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to clobber {out} (use --force)")

    rec = {"id": slug, "type": args.type, "name": args.name,
           "status": "draft", "confidence": 0.5, "updated": TODAY}
    # `source` records have no `sources` field — they ARE the sources.
    if args.type != "source":
        rec["sources"] = args.source
    rec.update(TEMPLATES[args.type])
    if args.type == "source":
        rec["url"] = args.url
        rec["kind"] = args.kind or "blog"
        rec["publisher"] = args.publisher

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()