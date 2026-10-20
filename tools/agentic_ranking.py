#!/usr/bin/env python3
"""Rank models for real exploratory agentic work.

Averaged capability ranks a narrow specialist above an agent that can actually
finish a long task. This ranks agentic evidence first, then gates on whether the
model can run on one workstation.
"""
import json, re, pathlib

R = pathlib.Path(__file__).resolve().parent.parent

FAMILIES = [
    ("terminal", r"(?i)terminal|\btb2?\b"),
    ("swe", r"(?i)swe|deepswe|repo"),
    ("web", r"(?i)browsecomp|webarena|webvoy|browser|agentic search"),
    ("tool", r"(?i)tau2|tau-?bench|toolathlon|\bbfcl\b|mcp"),
    ("computer", r"(?i)osworld|androidworld|agentworld"),
]
MAX_RAM_GIB = 128
MAX_ACTIVE_B = 40


def families(model):
    names = [b["name"] for b in model.get("benchmarks") or []]
    return [f for f, pat in FAMILIES if any(re.search(pat, n) for n in names)]


def k128(model):
    kv = model.get("kv_cache_bytes_per_token")
    return round(kv * 131072 / 2**30, 2) if kv else None


def load():
    rows = json.loads((R / "site/public/joined/worth_running.json").read_text())["rows"]
    models = {}
    for f in (R / "data/models").glob("*.json"):
        m = json.loads(f.read_text())
        models[m["id"]] = m
    return rows, models


def build():
    rows, models = load()
    out = []
    for r in rows:
        ax = r.get("axes") or {}
        if not ax.get("agentic"):
            continue
        m = models.get(r["id"], {})
        total = m.get("params_b") or 0
        active = m.get("active_params_b") or total
        out.append(dict(
            id=r["id"], ag=ax["agentic"], ins=ax.get("instruction"),
            know=ax.get("knowledge"), cod=ax.get("coding"),
            total=total, active=active, ctx=m.get("max_position_embeddings") or 0,
            k128=k128(m), fams=families(m),
            ram=round(total * 0.5, 1),
        ))
    return out


def tier(r):
    """A: runnable + broad. B: runnable, narrow. C: not runnable. D: weak agentic."""
    if r["ag"] < 70:
        return "D"
    runnable = r["active"] <= MAX_ACTIVE_B and r["ram"] <= MAX_RAM_GIB
    if not runnable:
        return "C"
    broad = len(r["fams"]) >= 2
    longctx = r["ctx"] >= 131072
    if broad and longctx:
        return "A"
    return "B"


def fmt(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v))


def render(rows):
    for r in rows:
        r["tier"] = tier(r)
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows.sort(key=lambda r: (order[r["tier"]], -r["ag"]))
    return rows


def markdown(rows):
    L = ["# Agentic ranking", "",
         "Ranked by agentic evidence, then gated on whether one workstation can run it "
         "(active params <= %dB, 4-bit weights <= %d GiB)." % (MAX_ACTIVE_B, MAX_RAM_GIB), ""]
    titles = {"A": "Tier A - runnable, broad agentic evidence",
              "B": "Tier B - runnable, narrow agentic evidence",
              "C": "Tier C - strong agentic, not runnable locally",
              "D": "Tier D - weak agentic evidence"}
    for t in "ABCD":
        sub = [r for r in rows if r["tier"] == t]
        if not sub:
            continue
        L += ["## " + titles[t], "",
              "| model | agentic | instr | know | total | active | ctx | KV@128k GiB | families | 4-bit GiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub[:20]:
            L.append("| [[models/%s]] | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], fmt(r["ag"]), fmt(r["ins"]), fmt(r["know"]), fmt(r["total"], 0),
                fmt(r["active"]), r["ctx"] or "-", fmt(r["k128"]), ", ".join(r["fams"]) or "-", fmt(r["ram"])))
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rows = render(build())
    out = R / "docs/31-agentic-ranking.md"
    out.write_text(markdown(rows))
    counts = {}
    for r in rows:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1
    print("wrote", out.relative_to(R), "|", counts)
