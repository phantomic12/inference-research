#!/usr/bin/env python3
"""Build the agentic ranking view the /agentic/ page is rendered from.

The worth-running join already did the expensive work: it scored every model on
capability axes, discounted saturated and stale evidence, shrunk thin evidence,
and priced the hardware needed to serve it. Re-deriving any of that here would
give the ranking page a second, subtly different opinion of the same catalog.

What is specific to agentic work is *which* benchmarks count as agentic tasks —
terminal, SWE, browser, tool-use or computer-use — and that a model needs an
agentic score at all before it belongs in the ranking. So this view is a narrow
projection of worth_running.json: keep the models with an agentic axis, tag
their task families from the shared table in agentic_ranking.py (the one the
markdown ranking uses), and carry the existing scores and memory figures into
an Astro-importable payload. No scores are recomputed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "public" / "joined"

# Same directory as the FAMILIES table; importing it here keeps one definition
# of what "terminal", "swe", "web", "tool" and "computer" mean on the site.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic_ranking import FAMILIES  # noqa: E402

_FIELDS = (
    "id", "name", "agentic", "coding", "instruction", "knowledge", "axis_count",
    "thin", "vendor_only", "families", "release_year", "active_params_b",
    "total_params_b", "moe", "weights_q4_gib", "traffic_q4_gib", "single_user_gib", "multi_user_gib",
    "offload_friendly", "quant_retention", "worth",
)


def build_agentic_rows() -> list[dict]:
    """Project the worth-running rows onto the agentic-ranking columns."""
    source = OUT / "worth_running.json"
    rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
    eligible = [r for r in rows if (r.get("axes") or {}).get("agentic") is not None]

    out: list[dict] = []
    for r in eligible:
        axes = r.get("axes") or {}
        bench_names = [b.get("name") or "" for b in r.get("benchmarks") or []]
        families = [name for name, pattern in FAMILIES
                    if any(re.search(pattern, bench, re.IGNORECASE) for bench in bench_names)]
        out.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "agentic": axes["agentic"],
            "coding": axes.get("coding"),
            "instruction": axes.get("instruction"),
            "knowledge": axes.get("knowledge"),
            "axis_count": r.get("ranked_axes"),
            "thin": bool(r.get("thin")),
            "vendor_only": bool(r.get("vendor_only")),
            "families": families,
            "release_year": r.get("release_year"),
            "active_params_b": r.get("active_params_b"),
            "total_params_b": r.get("total_params_b"),
            "moe": bool(r.get("moe")),
            "weights_q4_gib": r.get("weights_q4_gib"),
            "traffic_q4_gib": r.get("traffic_q4_gib"),
            "single_user_gib": r.get("single_user_gib"),
            "multi_user_gib": r.get("multi_user_gib"),
            "offload_friendly": bool(r.get("offload_friendly")),
            "quant_retention": r.get("quant_retention"),
            "worth": r.get("worth"),
        })

    assert all(r["agentic"] is not None for r in out), "an unscored row slipped into the agentic view"
    assert len(out) == len(eligible), "agentic view lost or gained rows during projection"
    assert len(out) >= 40, f"only {len(out)} agentic rows; expected at least 40"
    assert all(tuple(r) == _FIELDS for r in out), "agentic payload field order changed"
    return out


def main() -> None:
    rows = build_agentic_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "agentic.json"
    out.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  {len(rows)} models with an agentic axis -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
