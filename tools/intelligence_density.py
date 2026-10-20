#!/usr/bin/env python3
"""Intelligence density: capability relative to model size.

Three densities, because "size" means three different things:
  total    capability per total parameter (what you must store)
  active   capability per active parameter (what you pay per token)
  resident capability per GiB at 4-bit (what you must stream)

Capability is the axis-averaged relative score built by build_joined_views, so it
already normalises per benchmark against the catalog best. That makes the density
a ratio of two relative quantities, and it inherits a bias worth stating loudly:
models that report few, easy benchmarks are compared against other models that
report the same few easy benchmarks, so they can top an axis without being
strong in absolute terms. Every row therefore carries its benchmark names and
reporter counts.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
JOINED = ROOT / "site" / "public" / "joined" / "worth_running.json"
OUT = ROOT / "docs" / "30-intelligence-density.md"

MIN_AXES = 3        # a density built on two axes is a coincidence, not a finding
MIN_REPORTERS = 4   # denominators thinner than this swing the ratio


def load():
    rows = json.loads(JOINED.read_text())["rows"]
    models = {}
    for p in (ROOT / "data" / "models").glob("*.json"):
        m = json.loads(p.read_text())
        models[m["id"]] = m
    return rows, models


def reporters(rows):
    """How many models report each benchmark name. A ratio out of one model is 100."""
    seen = {}
    for r in rows:
        for b in r.get("benchmarks") or []:
            seen[b["name"]] = seen.get(b["name"], 0) + 1
    return seen


def build():
    rows, models = load()
    rep = reporters(rows)
    out = []
    for r in rows:
        cap = r.get("capability")
        m = models.get(r["id"])
        if not cap or not m or not m.get("params_b"):
            continue
        axes = r.get("axes") or {}
        if len(axes) < MIN_AXES:
            continue
        total = float(m["params_b"])
        active = float(m.get("active_params_b") or total)
        bms = [b for b in (r.get("benchmarks") or []) if rep.get(b["name"], 0) >= MIN_REPORTERS]
        if len(bms) < 3:
            continue
        out.append({
            "id": r["id"], "cap": float(cap), "total": total, "active": active,
            "axes": axes, "bench": [b["name"] for b in bms],
        })
    return out


def pareto(rows):
    """Smallest model that reaches each capability level."""
    out = []
    for lvl in (50, 60, 70, 80, 85, 90, 95):
        c = [r for r in rows if r["cap"] >= lvl]
        if c:
            out.append((lvl, min(c, key=lambda x: x["total"])))
    return out


def main():
    rows = build()
    rows.sort(key=lambda r: -r["cap"] / r["total"])
    lines = ["# Intelligence density", "",
             "Capability per parameter, from the joined worth-running view. "
             "`cap` is 0-100 relative to the best known score on each benchmark the "
             "model reports, averaged within capability axes. Density is that number "
             "divided by size, so it rewards a model that is both good and small.", ""]
    lines += ["## Density per total parameter", "",
              "| model | cap | total | active | cap/param | axes | evidence |",
              "|---|---|---|---|---|---|---|"]
    for r in rows[:20]:
        lines.append("| `%s` | %.1f | %.1fB | %.1fB | **%.2f** | %d | %s |" % (
            r["id"], r["cap"], r["total"], r["active"], r["cap"] / r["total"],
            len(r["axes"]), ", ".join(r["bench"][:4])))

    moe = [r for r in rows if r["active"] < r["total"] * 0.9]
    moe.sort(key=lambda r: -r["cap"] / r["active"])
    lines += ["", "## Density per active parameter (MoE)", "",
              "What you pay per token, not per byte stored.", "",
              "| model | cap | active | total | cap/active | axes |",
              "|---|---|---|---|---|---|"]
    for r in moe[:15]:
        lines.append("| `%s` | %.1f | %.1fB | %.1fB | **%.1f** | %d |" % (
            r["id"], r["cap"], r["active"], r["total"], r["cap"] / r["active"], len(r["axes"])))

    lines += ["", "## Frontier: smallest model at each capability level", "",
              "| capability | smallest model | total | active | density |",
              "|---|---|---|---|---|"]
    for lvl, r in pareto(rows):
        lines.append("| >= %d | `%s` | %.1fB | %.1fB | %.2f |" % (
            lvl, r["id"], r["total"], r["active"], r["cap"] / r["total"]))

    lines += ["", "## Why this ranking is not absolute intelligence", "",
              "Capability is normalised per benchmark against whoever else reports it. "
              "Frontier models report AIME, GPQA and SWE-bench; small models report "
              "GSM8K, MMLU and HumanEval. Each is therefore scored against its own "
              "peer group, and a 2B model can lead the knowledge axis by winning "
              "MMLU among 2B-class reporters while a 400B model scores lower for "
              "having attempted harder tests. Read the evidence column before "
              "believing a density number.", "",
              "Rows with fewer than %d axes are excluded, and benchmarks with fewer "
              "than %d reporters are dropped, because a ratio whose denominator is "
              "one or two models is noise." % (MIN_AXES, MIN_REPORTERS), ""]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT.relative_to(ROOT), "|", len(rows), "models")


if __name__ == "__main__":
    main()
