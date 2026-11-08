#!/usr/bin/env python3
"""Cross-check figure-derived benchmark rows against the card's own HTML tables.

Why this exists
---------------
Many cards publish the same numbers twice: once as a chart image (which the
ingest pipeline reads by eye) and once as an HTML table in the README (exact
text). The two are independent carriers, so a figure row that does not appear
in this model's column of the text table is either (a) a legitimate
figure-only row, or (b) a row read off the WRONG COLUMN -- the failure mode
that put the 24GB IQ2 quant's numbers into the BF16 `nemotron-3-5-bolt-74b-a9b`
record, and that the `minicpm-o-2-6` radar reads hit four times by picking up
another series' printed label.

A name-based mixup detector cannot see this: the benchmark name is right, only
the column is wrong.

Usage::

    python3 tools/crosscheck_figures.py                 # whole catalog
    python3 tools/crosscheck_figures.py --ids a,b,c     # selected records
    python3 tools/crosscheck_figures.py --offline DIR   # use saved READMEs

Design notes
------------
* HTML tables are parsed into a real grid with rowspan/colspan expanded. Naive
  `<td>` scraping shifts every column after a spanning cell, which is how an
  earlier hand-rolled pass produced ~100 false mismatches on `qwen3-5-*`.
* A cell is only compared when the row's benchmark name matches after
  normalisation AND the value is a bare number. `HumanEval` vs `HumanEval+`
  must not normalise together, so a trailing `+`/`-` is preserved.
* Aggregate rows are skipped: they double count the rows beside them.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"

FIGURE = re.compile(r"figure|read off|by eye|bar label|chart|panel", re.I)
AGGREGATE = re.compile(r"^(average|avg|overall|mean|total|overall score)$", re.I)


def _headers() -> dict[str, str]:
    h = {"User-Agent": "Mozilla/5.0"}
    token = None
    for p in (Path("/opt/data/home/.cache/huggingface/token"), Path.home() / ".cache/huggingface/token"):
        try:
            if p.exists() and p.read_text().strip():
                token = p.read_text().strip()
                break
        except OSError:
            pass
    import os
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def hf_repo(record: dict, sources: dict[str, dict]) -> str | None:
    """The record's own HF repo, preferring the slug-matching one.

    Records often list a base or sibling first; 'first URL wins' would audit a
    different model's card and report every row as absent.
    """
    cands = []
    nonmodel = {"api", "datasets", "spaces", "docs", "blog", "collections", "papers",
                "tasks", "models", "organizations", "settings", "join", "login", "new"}
    for sid in record.get("sources") or []:
        url = (sources.get(sid) or {}).get("url") or ""
        for m in re.finditer(r"huggingface\.co/([^/\"#?\s]+/[^/\"#?\s]+)", url):
            r = m.group(1).split("/tree")[0].split("/blob")[0].split("/resolve")[0]
            if r.split("/")[0].lower() in nonmodel:
                continue
            cands.append(r)

    def slug(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-")

    want = slug(record["id"])
    for r in cands:
        if slug(r.split("/")[-1]) == want:
            return r
    return cands[0] if cands else None


def parse_tables(text: str) -> list[list[list[str]]]:
    """HTML tables -> grids with rowspan/colspan expanded."""
    out = []
    for tab in re.findall(r"<table.*?</table>", text, re.S):
        rows = re.findall(r"<tr.*?</tr>", tab, re.S)
        if not rows:
            continue
        cells = []
        for ri, row in enumerate(rows):
            ci = 0
            for m in re.finditer(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S):
                attrs, inner = m.group(1), m.group(2)
                rs = int((re.search(r'rowspan=["\']?(\d+)', attrs) or [0, 1])[1])
                cs = int((re.search(r'colspan=["\']?(\d+)', attrs) or [0, 1])[1])
                txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", inner))).strip()
                cells.append((ri, ci, txt, rs, cs))
                ci += cs
        if not cells:
            continue
        maxr = max(c[0] + c[3] for c in cells)
        maxc = max(c[1] + c[4] for c in cells)
        grid = [["" for _ in range(maxc)] for _ in range(maxr)]
        for ri, ci, txt, rs, cs in cells:
            for dr in range(rs):
                for dc in range(cs):
                    if ri + dr < maxr and ci + dc < maxc:
                        grid[ri + dr][ci + dc] = txt
        out.append(grid)
    return out


def norm(s: str) -> str:
    """Comparison key. Keeps +/- so HumanEval and HumanEval+ stay distinct."""
    s = s.lower().replace("&", "and")
    plus = s.endswith("+")
    minus = s.endswith("-") or s.endswith("↓")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s + ("+" if plus else "-" if minus else "")


def bare_number(v: str) -> float | None:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)$", v.strip())
    return float(m.group(1)) if m else None


def bench_key(name: str) -> str:
    """Last '·'-separated segment, so 'Coding · HumanEval' matches table 'HumanEval'."""
    return norm(re.split(r"[·]", name)[-1])


def audit(record: dict, grids: list[list[list[str]]]) -> dict:
    key, name = norm(record["id"]), norm(record.get("name") or "")
    rows = [b for b in record.get("benchmarks") or [] if FIGURE.search(b.get("config") or "")]
    if not rows:
        return {"fig_rows": 0, "matched": 0, "absent": [], "mismatched": []}
    matched, absent, mismatched = 0, [], []
    table_values = set()
    for grid in grids:
        for r in grid:
            for c in r[1:]:
                v = bare_number(c)
                if v is not None:
                    table_values.add(v)
    for grid in grids:
        for hi, hrow in enumerate(grid[:15]):
            for ci, h in enumerate(hrow):
                hc = norm(h)
                if not hc or len(hc) < 5:
                    continue
                if not (hc == key or hc == name or (abs(len(hc) - len(key)) <= 2 and (hc in key or key in hc))):
                    continue
                for r in grid[hi + 1:]:
                    if len(r) <= ci:
                        continue
                    val = bare_number(r[ci])
                    bname = r[0]
                    if val is None or not bname or AGGREGATE.match(bname.strip()):
                        continue
                    bk = bench_key(bname)
                    for b in rows:
                        if norm(b["name"]) != bk:
                            continue
                        try:
                            score = float(b["score"])
                        except (TypeError, ValueError):
                            continue
                        if abs(score - val) <= 1e-9:
                            matched += 1
                            b["_seen"] = True
                        else:
                            mismatched.append({"benchmark": b["name"], "record": score,
                                               "card": val, "card_row": bname})
                        break
    for b in rows:
        if b.pop("_seen", False):
            continue
        try:
            score = float(b["score"])
        except (TypeError, ValueError):
            continue
        absent.append({"benchmark": b["name"], "score": score,
                       "in_card_anywhere": score in table_values})
    return {"fig_rows": len(rows), "matched": matched, "absent": absent, "mismatched": mismatched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated record ids")
    ap.add_argument("--offline", help="directory of <id>.md README dumps")
    ap.add_argument("--only-suspicious", action="store_true", default=True)
    args = ap.parse_args()

    sources = {}
    for p in (ROOT / "data" / "sources").glob("*.json"):
        try:
            sources[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    want = set(args.ids.split(",")) if args.ids else None
    findings, checked = [], 0
    for p in sorted(MODELS.glob("*.json")):
        if want and p.stem not in want:
            continue
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        repo = hf_repo(record, sources)
        text = None
        if args.offline:
            f = Path(args.offline) / f"{p.stem}.md"
            text = f.read_text(encoding="utf-8") if f.exists() else None
        elif repo:
            try:
                req = urllib.request.Request(
                    f"https://huggingface.co/{repo}/raw/main/README.md", headers=_headers())
                text = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
            except Exception:
                text = None
        if not text or "<table" not in text:
            continue
        grids = parse_tables(text)
        if not grids:
            continue
        res = audit(record, grids)
        if not res["fig_rows"]:
            continue
        checked += 1
        if res["mismatched"] or any(not a["in_card_anywhere"] for a in res["absent"]):
            findings.append((p.stem, repo, res))

    print(f"audited {checked} records with both figure rows and an HTML table")
    if not findings:
        print("no figure row contradicted the card's text table")
        return 0
    for mid, repo, res in findings:
        print(f"\n== {mid}  ({repo})  fig_rows={res['fig_rows']} matched={res['matched']}")
        for m in res["mismatched"]:
            print(f"   MISMATCH {m['benchmark']}: record {m['record']} vs card {m['card']} (row '{m['card_row']}')")
        for a in res["absent"]:
            flag = "NOT IN TABLE" if not a["in_card_anywhere"] else "value present under another row"
            print(f"   ABSENT   {a['benchmark']}: {a['score']} -> {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
