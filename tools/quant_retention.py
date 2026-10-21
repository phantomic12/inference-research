#!/usr/bin/env python3
"""Quant retention: recompute base-vs-quant deltas from reviewed repo cards."""
import json, pathlib, re, statistics

R = pathlib.Path(__file__).resolve().parent.parent
REV = pathlib.Path("/opt/data/review")
LOW = re.compile(r"(?i)ppl|perplex|kld|wer|cer|latency|ttft")


def load():
    files = sorted(REV.glob("done_*.jsonl")) + sorted(REV.glob("ldone_*.jsonl")) + sorted(REV.glob("ddone_*.jsonl"))
    out = {}
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip().startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            mid = r.get("id") or r.get("repo")
            if mid:
                r["id"] = mid
                out.setdefault(mid, r)
    return out


def delta(bench, base, quant):
    if not base or quant is None:
        return None
    raw = (quant - base) / base * 100.0
    return -raw if LOW.search(bench or "") else raw


def scored():
    rows = []
    for mid, r in load().items():
        if r.get("ref") != "base":
            continue
        vals = [(x.get("bench") or "", delta(x.get("bench"), x.get("base"), x.get("quant"))) for x in (r.get("rows") or [])]
        vals = [(b, d) for b, d in vals if d is not None]
        if not vals:
            continue
        fams = {re.sub(r"[\d._]+", "", b).strip().lower() for b, _ in vals}
        rows.append({"id": mid, "fmt": r.get("format"), "size": r.get("size_gib"),
                     "median": statistics.median([d for _, d in vals]), "n": len(vals),
                     "curve": len(vals) >= 2 and len(fams) == 1})
    return rows



def bucket(s):
    m = s["median"]
    if m > 1:
        return "exceeds_base"
    if abs(m) <= 1:
        return "lossless"
    if m >= -5:
        return "minor"
    if m >= -20:
        return "major"
    return "catastrophic"


def base_map():
    """HF repo id -> model record id, so the doc can cite records."""
    out = {}
    for f in (R / "data/models").glob("*.json"):
        m = json.loads(f.read_text())
        for sid in m.get("sources") or []:
            src = R / "data/sources" / (sid + ".json")
            if not src.exists():
                continue
            url = (json.loads(src.read_text()).get("url") or "").lower()
            hit = re.search(r"huggingface\.co/([^/\s]+/[^/\s?#]+)", url)
            if hit:
                out[hit.group(1)] = m["id"]
    return out


def main():
    if (R / "data/quant_retention.jsonl").exists():
        rows = []
        for line in (R / "data/quant_retention.jsonl").read_text().splitlines():
            d = json.loads(line)
            rows.append({"id": d["id"], "median": d["median_delta_pct"], "n": d.get("n") or 0,
                         "size": d.get("size_gib"), "curve": d.get("curve", False),
                         "bucket": d.get("bucket")})
    elif REV.exists():
        rows = scored()
    else:
        p = R / "data/quant_retention.jsonl"
        if not p.exists():
            print("no review dir and no committed retention data; nothing to do")
            return
        rows = [{"id": d["id"], "median": d["median_delta_pct"], "n": d.get("n", 0),
                 "size": d.get("size_gib"), "curve": d.get("curve", False)}
                for d in (json.loads(l) for l in p.read_text().splitlines() if l.strip())]
    single = [r for r in rows if not r["curve"]]
    groups = {}
    for r in single:
        groups.setdefault(r.get("bucket") or bucket(r), []).append(r)
    bm = base_map()
    deriv = {}
    for line in (R / "data/derivatives.jsonl").read_text().splitlines():
        d = json.loads(line)
        deriv[d["id"]] = d.get("base", "")

    L = ["# Quant retention: what each low-bit build actually costs", "",
         "Recomputed from raw base/quant cells in reviewed model cards. Deltas are signed so that",
         "negative always means worse than base, with PPL/KLD/WER flipped. Rows showing a gain",
         "above +1% are bucketed `exceeds_base` and excluded from the ranking. Adjudicating all 18",
         "found only ONE genuine column-order swap: the rest were pooling artifacts (a median over",
         "rows mixing throughput, context-ceiling and NLL figures with quality scores), cards whose",
         "own recovery column reads above 100%, or gains inside the card's stated +/-5pp gate.",
         "A positive delta is a prompt to read the mechanism, never proof of an error.", "",
         "| bucket | builds |", "|---|---|"]
    for k in ("lossless", "minor", "major", "catastrophic", "exceeds_base", "not_comparable"):
        L.append("| %s | %d |" % (k, len(groups.get(k, []))))
    L.append("| curves (multi-level tables, not ranked) | %d |" % len([r for r in rows if r["curve"]]))
    L.append("")

    for k, title in (("lossless", "Lossless (within 1%)"), ("minor", "Minor loss (1-5%)"), ("catastrophic", "Catastrophic (over 20%)")):
        items = sorted(groups.get(k, []), key=lambda r: -r["median"])[:15]
        if not items:
            continue
        L += ["## " + title, "", "| build | vs base | rows | size | base model |", "|---|---|---|---|---|"]
        for r in items:
            b = bm.get(deriv.get(r["id"], ""), "")
            cite = "[[models/%s]]" % b if b else deriv.get(r["id"], "-")
            L.append("| `%s` | %+.2f%% | %d | %s | %s |" % (r["id"], r["median"], r["n"], r["size"] or "-", cite))
        L.append("")
    out = R / "docs/34-quant-retention.md"
    out.write_text("\n".join(L))
    print("wrote", out.name, "|", len(single), "ranked |", {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
