#!/usr/bin/env python3
"""Aggregate per-repo derivative verdicts into one report."""
import collections, json, pathlib

R = pathlib.Path(__file__).resolve().parent.parent
ANA = pathlib.Path("/opt/data/analysis")
DB = R / "data/derivatives.jsonl"
OUT = R / "docs/33-derivative-analysis.md"


def load():
    db = {}
    for line in DB.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            db[r["id"]] = r
    rows = []
    for f in sorted(ANA.glob("batch_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["dl"] = db.get(r["id"], {}).get("downloads", 0)
                r["base"] = db.get(r["id"], {}).get("base", "")
                rows.append(r)
    return rows, db


def main():
    rows, db = load()
    n = len(rows)
    vc = collections.Counter(r["verdict"] for r in rows)
    dt = sum(1 for r in rows if r["publishes_delta"])
    L = ["# Derivative analysis: quants and fine-tunes", "",
         "%d of %d repos in `data/derivatives.jsonl` analysed." % (n, len(db)), "",
         "| verdict | repos | publishes a delta |", "|---|---|---|"]
    for v, c in vc.most_common():
        d = sum(1 for r in rows if r["verdict"] == v and r["publishes_delta"])
        L.append("| %s | %d | %d (%.1f%%) |" % (v, c, d, 100.0 * d / c))
    L.append("| **all** | **%d** | **%d (%.1f%%)** |" % (n, dt, 100.0 * dt / n))
    return L, rows


if __name__ == "__main__":
    L, rows = main()
    lb = [r for r in rows if r["publishes_delta"] and r["verdict"] == "quant-lowbit"]
    L += ["", "## Low-bit builds that publish measured degradation", "",
          "The only low-bit quants that give a number to argue with.", "",
          "| repo | downloads | evidence |", "|---|---|---|"]
    for r in sorted(lb, key=lambda x: -x["dl"])[:25]:
        ev = (r["delta_evidence"] or r["note"]).replace("|", "/")[:110]
        L.append("| `%s` | %s | %s |" % (r["id"], "{:,}".format(r["dl"]), ev))
    ab = [r for r in rows if r["verdict"] == "abliterated"]
    abd = sum(1 for r in ab if r["publishes_delta"])
    L += ["", "## Abliteration", "",
          "%d repos, %d publish any measured comparison (%.1f%%)." % (len(ab), abd, 100.0 * abd / max(len(ab), 1)), "",
          "Refusal removal is the single largest category of derivative work and almost",
          "none of it reports whether the model still works.", ""]
    OUT.write_text("\n".join(L))
    print("wrote", OUT.name, n_ := len(rows), "analysed")
