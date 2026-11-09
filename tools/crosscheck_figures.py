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
* Two carriers count as text tables: HTML `<table>` and markdown pipe tables.
  Only HTML was understood at first, so the ~60 records whose card prints its
  results as a pipe table looked figure-only and were skipped in silence.
* HTML tables are parsed into a real grid with rowspan/colspan expanded. Naive
  `<td>` scraping shifts every column after a spanning cell, which is how an
  earlier hand-rolled pass produced ~100 false mismatches on `qwen3-5-*`.
* A pipe table is usually transposed (models down the rows), so the grid is
  flipped when the model name sits in the first column.
* A card names its model with a suffix the record id lacks ('MiniCPM-o 4.5-Instruct'
  for `minicpm-o-4-5`); a single build/pass token is accepted so the column is
  found, but sibling variants stay distinct measurements -- a value matching one
  and not another is reported as VARIANT rather than silently accepted.
* A card may publish more than one column for the same model (`twil-lm3` has
  `TwIL-LM3` and `TwIL-LM3*`, two scoring passes). A row agrees if it matches
  ANY of that model's columns; only a value in none of them is a mixup.
* Scale (0.693 vs 69.3) and precision (52.21 vs a record's 52.2) are not
  disagreements -- the record's decimal places set the tolerance.
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


def _md_cell(cell: str) -> str:
    """Strip the markdown that carries no meaning for a numeric compare."""
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", cell)   # [text](url) -> text
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"(?<![\w.])_|_(?![\w.])", "", s)            # _emphasis_ only
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(.*?)%$", r"\1", s)                        # 58.2% -> 58.2
    return s.strip("*").strip()


def parse_pipe_tables(text: str) -> list[list[list[str]]]:
    """Markdown pipe tables -> grids, same shape as parse_tables().

    Plenty of cards publish their results table as a markdown pipe table rather
    than an HTML <table>. Without this, those records look like they have no
    second carrier at all and the cross-check silently skips them -- which is
    how 62 records were left looking figure-only when the text was right there.

    A pipe table is a run of consecutive lines that both start and end with '|'.
    The `|---|---|` delimiter row is dropped, and a row without a trailing pipe is
    still accepted (the closing pipe is optional in most renderers) as long as it
    carries at least two cells.
    """
    out = []
    block: list[str] = []

    def flush() -> None:
        if len(block) < 2:
            block.clear()
            return
        grid = []
        for line in block:
            s = line.strip()
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|"):
                s = s[:-1]
            cells = [_md_cell(c) for c in s.split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
                continue  # delimiter row
            grid.append(cells)
        if len(grid) >= 2:
            width = max(len(r) for r in grid)
            out.append([r + [""] * (width - len(r)) for r in grid])
        block.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 2:
            block.append(line)
        else:
            flush()
    flush()
    return out


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


def same_value(record_score, card_value: float) -> bool:
    """Do these two numbers agree, allowing for how the card writes them?

    Three things are not disagreements and must not be reported as one:

    * **Scale.** Cards print either 69.3 or 0.693 for the same result. A card
      value <= 1.0 beside a record value > 1.0 is compared as a percentage.
    * **Precision.** The figure prints 52.21 where the record keeps one decimal.
      The record's own decimal places set the tolerance, because that is the
      precision it claims.
    * **Exactness.** Otherwise the two must be identical.

    A genuine wrong-column read still fails all three: 64.7 vs 64.8 differs at
    the record's precision, and a neighbouring model's 52.1 vs 61.5 is nowhere
    near.
    """
    try:
        score = float(record_score)
        card_value = float(card_value)
    except (TypeError, ValueError):
        return False
    text = str(record_score)
    decimals = len(text.split(".")[1]) if "." in text and "e" not in text.lower() else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9 if decimals else 0.5 + 1e-9
    if abs(score - card_value) <= tol:
        return True
    if 0 < card_value <= 1.0 < score and abs(score - card_value * 100) <= tol:
        return True
    return False


# A record names its rows with the metric it scored: 'APPS pass@1', 'GSM8K
# (accuracy)', 'Rule induction (derivation score)'. A card's results table heads
# the row with the benchmark alone ('apps', 'GSM8K', 'Rule induction'), because
# the metric is stated once in the table's caption. norm() strips punctuation but
# not words, so 'appspass1' never equalled 'apps': 28 figure rows on 2 records
# were announced NOT IN TABLE against a table that carried their exact value,
# and 4 rows (`twil-lm3`) reported a value under 'another row' that was in fact
# their own row under a shorter name.
#
# Only a trailing parenthesised metric or a pass@N token is stripped, and only
# to build a SECOND lookup key. 'HumanEval' vs 'HumanEval+' is untouched, and
# 'Minerva pass@1' still matches the card's 'minerva'.
# `pass@N` is a harness name the card may or may not repeat in its header, so it
# is stripped whether or not it is bracketed ('APPS pass@1', 'APPS (pass@1)');
# the softer words are only stripped inside brackets or after a separator, so a
# benchmark whose own name ends in 'score' ('Derivation score') keeps it.
METRIC = re.compile(
    r"(?:\s*[\(\[]\s*(?:avg|average|accuracy|acc|score|rate)\s*[\)\]]"
    r"|\s+(?:pass@\d+|accuracy|acc|score|rate)"
    r"|\s*[\(\[]\s*pass@\d+\s*[\)\]])\s*$", re.I)


def _strip_metric(name: str) -> str:
    prev = None
    s = name
    while prev != s:
        prev = s
        s = METRIC.sub("", s).strip()
    return s


def bench_keys(name: str) -> list[str]:
    """Every key this row name may be filed under, in matching precedence."""
    seg = re.split(r"[·]", name)[-1]
    out = [norm(seg)]
    alt = norm(_strip_metric(seg))
    if alt not in out:
        out.append(alt)
    return out


def bench_key(name: str) -> str:
    """Last '·'-separated segment, so 'Coding · HumanEval' matches table 'HumanEval'."""
    return norm(re.split(r"[·]", name)[-1])


# A card names its own model with a build or deployment suffix the record id does
# not carry: `minicpm-o-4-5` is published as 'MiniCPM-o 4.5-Instruct' and
# 'MiniCPM-o 4.5-Thinking', `deepseek-v4-pro` as 'DeepSeek-V4-Pro-Base'. The
# matcher used to allow only a two-character length difference, so those cards
# never identified their own column: 13 records / 157 figure rows had a results
# table in the README covering their own benchmarks and went uncompared, while
# the summary still counted them as audited.
#
# Exactly one suffix token, and only a token naming a build or inference pass of
# the SAME model. A hardware-backend column such as 'MiniCPM-o-4.5-hygon-FlagOS'
# is a different measurement, not an alias.
VARIANT = re.compile(
    r"(instruct|thinking|nonthink|chat|base|preview|exp|sft|dpo|rl|it|hf|awq|gptq|fp8|"
    r"gguf|mlx|bf16|f16|fp16)$")


def _variant_of(cell: str, key: str, name: str) -> str | None:
    """The suffix that turns this model id into the card's column label, or ''."""
    hc = norm(cell)
    if not hc.startswith(key) or hc == key:
        return "" if hc == key else None
    rest = hc[len(key):]
    return rest if VARIANT.fullmatch(rest) else None


# A size token the card truncated, or the tail of a long id the card did not
# print, is not a different model. These are the only two kinds of remainder
# accepted when the header is a prefix of the record id.
_SIZE_TAIL = re.compile(r"\d+(?:\.\d+)?b")
_BUILD_TAIL = re.compile(r"(?:instruct|thinking|nonthink|chat|preview|base|sft|dpo|rl|it|hf"
                         r"|fp8|bf16|f16)")


def _matches_model(cell: str, key: str, name: str) -> bool:
    """Is this header cell a column OF THIS MODEL?

    The old test accepted any header within two characters of the id, which is
    not a spelling difference but a different model: `...-1.5B-v2` was read as
    `...-1.5B` (so the v2 column's 46.39 APPS could stand in for v1's 42.0) and
    `Darwin-180B-RSI-R3` as `Darwin-180B-RSI` -- R3 is a separate Hugging Face
    repo, the next self-improvement round, not an alias. It also MISSED legit
    columns it should have taken: 'Euston' for `euston-8b`, 'Mica v0.5' for
    `mica-v0-5-4b`, 'Qwen2.5-VL-7B' for `qwen2-vl-7b-instruct`.

    Accepted forms, each requiring the shared part to be the whole id:
      * exact, or the record's display name;
      * one build/pass suffix (`-Instruct`, `-Thinking`, ...);
      * a superscript footnote the card glued on (`Nanbeige4.3-33B<sup>1</sup>`);
      * a prefix of the id whose remainder is just a size or build token.
    """
    hc = norm(cell)
    if not hc or len(hc) < 5:
        return False
    if hc == key or hc == name:
        return True
    if _variant_of(cell, key, name) is not None:
        return True
    if re.fullmatch(re.escape(key) + r"\d", hc):
        return True
    if key.startswith(hc):
        rest = key[len(hc):]
        if _SIZE_TAIL.fullmatch(rest) or _BUILD_TAIL.fullmatch(rest):
            return True
    return False


def orient(grid: list[list[str]], key: str, name: str) -> list[list[str]]:
    """Return the grid with models as columns.

    audit() looks for the model name in a header row and reads the benchmark
    rows beneath it. That is the HTML-table layout. A markdown pipe table is
    usually transposed: benchmarks across the top, one row per model. Reading
    such a grid untransposed finds no header match and the record looks like it
    has no second carrier, so transpose when the model sits in the first column.
    """
    for r in grid[1:]:
        if r and _matches_model(r[0], key, name):
            return [list(col) for col in zip(*grid)]
    return grid


def audit(record: dict, grids: list[list[list[str]]]) -> dict:
    key, name = norm(record["id"]), norm(record.get("name") or "")
    rows = [b for b in record.get("benchmarks") or [] if FIGURE.search(b.get("config") or "")]
    if not rows:
        return {"fig_rows": 0, "matched": 0, "absent": [], "mismatched": [],
                "variant_mixed": [], "no_column": False}
    matched, absent, mismatched, variant_mixed = 0, [], [], []
    table_values = set()
    for grid in grids:
        for r in grid:
            for c in r[1:]:
                v = bare_number(c)
                if v is not None:
                    table_values.add(v)

    # Gather EVERY cell the table offers for this model and each benchmark name
    # before judging. A card can carry several columns for one model (twil-lm3
    # publishes 'TwIL-LM3' and 'TwIL-LM3*', different scoring passes), so taking
    # the first matching column flags a mismatch on a value the record is
    # entitled to use. The failure this hunts -- a figure read off the WRONG
    # model's column -- has no agreeing column at all.
    seen: dict[str, list[tuple[float, str]]] = {}
    # Value per published column label, kept separately from `seen`. A card can
    # carry several columns for one model that are genuinely different
    # measurements -- `minicpm-o-4-5` publishes 'MiniCPM-o 4.5-Instruct'
    # (OpenCompass 77.6) beside 'MiniCPM-o 4.5-Thinking' (78.2). Treating those
    # as interchangeable would silently accept a row read off the sibling
    # variant, which is the same wrong-column failure this tool exists to catch.
    by_variant: dict[str, dict[str, float]] = {}
    for grid in grids:
        grid = orient(grid, key, name)
        for hi, hrow in enumerate(grid[:15]):
            for ci, h in enumerate(hrow):
                if not _matches_model(h, key, name):
                    continue
                # Key on the normalised label, not the display string: a card
                # writes the same model as 'Qwen3.6-35BA3B' and 'Qwen3.6-35B-A3B',
                # and those two spellings are one column, not two variants.
                label = norm(h)
                for r in grid[hi + 1:]:
                    if len(r) <= ci:
                        continue
                    val = bare_number(r[ci])
                    bname = (r[0] or "").strip()
                    if val is None or not bname or AGGREGATE.match(bname):
                        continue
                    seen.setdefault(bench_key(bname), []).append((val, bname))
                    by_variant.setdefault(label, {})[bench_key(bname)] = val

    for b in rows:
        # A row may be filed under its full name or its benchmark-only name; the
        # card picks one and the record the other. Try the exact key first so a
        # card that really does carry two rows ('GSM8K' and 'GSM8K (accuracy)')
        # is never collapsed onto whichever came first.
        cands, used_key = None, None
        for k in bench_keys(b["name"]):
            if seen.get(k):
                cands, used_key = seen[k], k
                break
        if not cands:
            try:
                score = float(b["score"])
            except (TypeError, ValueError):
                continue
            absent.append({"benchmark": b["name"], "score": score,
                           "in_card_anywhere": score in table_values})
            continue
        if any(same_value(b["score"], val) for val, _ in cands):
            matched += 1
            k = used_key
            agreeing = [lab for lab, col in by_variant.items()
                        if k in col and same_value(b["score"], col[k])]
            if len(by_variant) > 1 and len(agreeing) < len(by_variant):
                variant_mixed.append({"benchmark": b["name"], "score": b["score"],
                                      "agrees_with": agreeing,
                                      "differs_from": sorted(set(by_variant) - set(agreeing))})
            continue
        try:
            score = float(b["score"])
        except (TypeError, ValueError):
            continue
        val, bname = cands[0]
        mismatched.append({"benchmark": b["name"], "record": score,
                           "card": val, "card_row": bname,
                           "also_in_row": [v for v, _ in cands[1:]]})
    # A table that offers this model no numbers is not a carrier for these rows,
    # so none of them is 'absent' from it -- they are simply unverified. Reporting
    # them as absent is what made 589 figure rows look like a defect.
    if not seen:
        absent = []
    return {"fig_rows": len(rows), "matched": matched, "absent": absent,
            "mismatched": mismatched, "variant_mixed": variant_mixed,
            "no_column": not seen}


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
    findings, checked, no_column = [], 0, []
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
        if not text or ("<table" not in text and "|" not in text):
            continue
        grids = parse_tables(text) + parse_pipe_tables(text)
        if not grids:
            continue
        res = audit(record, grids)
        if not res["fig_rows"]:
            continue
        if res["no_column"]:
            # The README has a table, but not a results table for THIS model:
            # `internvl3-14b` publishes only a family/architecture list, so its
            # 84 figure rows were all reported 'NOT IN TABLE' against nothing
            # and the record still counted as audited. 53 of 102 audited
            # records were in this state, covering 589 figure rows.
            no_column.append(p.stem)
            continue
        checked += 1
        if res["mismatched"] or res["variant_mixed"] or any(
                not a["in_card_anywhere"] for a in res["absent"]):
            findings.append((p.stem, repo, res))

    print(f"audited {checked} records with both figure rows and a results table "
          f"for the model")
    if no_column:
        print(f"skipped {len(no_column)} records whose text table has no column "
              f"of numbers for the model (family/architecture tables)")
    if not findings:
        print("no figure row contradicted the card's text table")
        return 0
    for mid, repo, res in findings:
        print(f"\n== {mid}  ({repo})  fig_rows={res['fig_rows']} matched={res['matched']}")
        for m in res["mismatched"]:
            print(f"   MISMATCH {m['benchmark']}: record {m['record']} vs card {m['card']} (row '{m['card_row']}')")
        for v in res["variant_mixed"]:
            print(f"   VARIANT  {v['benchmark']}: {v['score']} agrees with "
                  f"{', '.join(v['agrees_with']) or 'no column'} but not "
                  f"{', '.join(v['differs_from'])}")
        for a in res["absent"]:
            flag = "NOT IN TABLE" if not a["in_card_anywhere"] else "value present under another row"
            print(f"   ABSENT   {a['benchmark']}: {a['score']} -> {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
