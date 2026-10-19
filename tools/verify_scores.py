#!/usr/bin/env python3
"""Cross-check recorded benchmark scores against the vendor's own HTML table.

Why: a score can be wrong in ways `tools/validate.py` cannot see — a value read
from the wrong column of a multi-model leaderboard, or from the wrong one of two
tables on the same card. Both happened here (see
gotchas/gotcha-ornith15-card-column-misread and
gotchas/gotcha-self-inflicted-score-regressions-oct19).

How: fetch the model's HuggingFace card README, parse each <table> with its own
header row, resolve the column for this model, and compare every recorded
benchmark against that column. Column choice is validated by which candidate
actually reproduces the recorded numbers, so a card carrying two tables or
competitor columns cannot silently mislead it.

It reports; it never edits. A mismatch is a prompt to go read the source, not
proof the record is wrong.

Usage:  python3 tools/verify_scores.py [model-id ...]
Exit:   0 always (reporting tool); read STATUS/MISMATCHES in stdout.
"""

import json, re, sys, pathlib, urllib.request

R = pathlib.Path(__file__).resolve().parent.parent
UA = {'User-Agent': 'Mozilla/5.0'}

def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', 'replace')

def norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def rowkey(label):
    # bind on the benchmark name only: parenthetical harness suffixes
    # ("Terminal-Bench 2.1 (Terminus-2)") collapse, but distinct benchmarks
    # ("MMLU" vs "MMLU-Pro") stay distinct.
    return norm(str(label).split('(')[0])

def cells(row):
    return [re.sub(r'<[^>]+>', '', c).replace('&nbsp;', ' ').strip()
            for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.S)]

def parse_tables(html):
    out = []
    for tb in re.findall(r'<table.*?</table>', html, re.S):
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tb, re.S)
        grid = [cells(r) for r in rows]
        grid = [g for g in grid if g]
        if len(grid) >= 2:
            out.append(grid)
    return out

def model_keys(mid, rec):
    keys = set()
    n = norm(mid)
    keys.add(n)
    for tok in n.split():
        if len(tok) > 4:
            keys.add(tok)
    p = rec.get('params_b')
    a = rec.get('active_params_b')
    if p:
        keys.add(norm('%sb' % int(p)))
        if a:
            keys.add(norm('%sba%sb' % (int(p), int(a))))
            keys.add(norm(('%sba%sb' % (int(p), round(a, 1))).replace('.', '')))
    return {k for k in keys if len(k) >= 5}

def candidate_cols(grids, keys):
    cands = []
    for gi, g in enumerate(grids):
        hdr = g[0]
        for ci, h in enumerate(hdr):
            if ci == 0:
                continue
            hn = norm(h)
            if not hn:
                continue
            score = 0
            for k in keys:
                if k and (k in hn or hn in k):
                    score = max(score, len(k))
            if score:
                cands.append((score, gi, ci, h))
    cands.sort(reverse=True)
    if not cands:
        return []
    top = cands[0][0]
    return [c for c in cands if c[0] == top]

def score_col(grid, ci, benchmarks):
    """Return (value_matches, resolved_rows): the column that *reproduces* the
    recorded numbers wins, so a card with two tables cannot mislead the pick."""
    ok = resolved = 0
    for b in benchmarks:
        bn = norm(b.get('name'))
        want = num(b.get('score'))
        for row in grid:
            if len(row) <= ci:
                continue
            if rowkey(row[0]) and rowkey(row[0]) == bn:
                got = num(row[ci])
                if got is not None:
                    resolved += 1
                    if want is not None and abs(got - want) <= 0.051:
                        ok += 1
                break
    return ok, resolved

def num(s):
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r'-?\d+(?:\.\d+)?', (str(s) if s is not None else '').replace(',', ''))
    return float(m.group(0)) if m else None

def check(mid):
    p = R / 'data/models' / (mid + '.json')
    rec = json.loads(p.read_text())
    urls = []
    for ref in (rec.get('sources') or []):
        if isinstance(ref, dict):
            urls.append(ref.get('url') or '')
        else:
            sp = R / 'data/sources' / (str(ref) + '.json')
            if sp.exists():
                urls.append(json.loads(sp.read_text()).get('url') or '')
    card = None
    for u in urls:
        m = re.match(r'https://huggingface\.co/([^/]+/[^/?#]+)', u)
        if m and '/docs/' not in u:
            card = m.group(1)
            break
    if not card:
        return mid, 'no-hf-card', []
    try:
        html = fetch('https://huggingface.co/%s/raw/main/README.md' % card)
    except Exception as e:
        return mid, 'fetch-fail: %s' % type(e).__name__, []
    grids = parse_tables(html)
    if not grids:
        return mid, 'no-tables', []
    keys = model_keys(mid, rec)
    bms = rec.get('benchmarks') or []
    cands = candidate_cols(grids, keys)
    if not cands:
        return mid, 'no-column', []
    best = None
    for score, gi, ci, h in cands:
        hdr_w = len(grids[gi][0])
        widths = [len(r) for r in grids[gi][1:] if len(r) > 1]
        if not widths:
            continue
        modal = max(set(widths), key=widths.count)
        off = hdr_w - modal          # rowspan/colspan on the left shifts cells left
        eff = ci - off
        if off < 0 or eff < 0 or eff >= modal:
            continue
        ok, resolved = score_col(grids[gi], eff, bms)
        key = (ok, resolved)
        if best is None or key > best[0]:
            best = (key, gi, eff, h)
    if best is None:
        return mid, 'ragged-table', []
    if best[0][1] == 0:
        return mid, 'ambiguous-column', []
    gi, ci = best[1], best[2]
    results = []
    for b in (rec.get('benchmarks') or []):
        want = num(b.get('score'))
        bn = norm(b.get('name'))
        hit = None
        for row in grids[gi]:
            if len(row) <= ci:
                continue
            if rowkey(row[0]) and rowkey(row[0]) == bn:
                hit = row
                break
        if hit is None:
            results.append((b.get('name'), want, None, 'no-row'))
            continue
        got = num(hit[ci])
        if got is None:
            results.append((b.get('name'), want, None, 'no-num'))
        elif want is None:
            results.append((b.get('name'), want, got, 'record-empty'))
        elif abs(got - want) <= 0.051:
            results.append((b.get('name'), want, got, 'ok'))
        else:
            results.append((b.get('name'), want, got, 'MISMATCH'))
    return mid, 'checked', results

if __name__ == '__main__':
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(p.stem for p in (R / 'data/models').glob('*.json'))
    stats = {}
    bad = {}
    for mid in ids:
        try:
            mid, st, res = check(mid)
        except Exception as e:
            mid, st, res = mid, 'error: %s' % type(e).__name__, []
        stats[st] = stats.get(st, 0) + 1
        for name, want, got, verdict in res:
            if verdict == 'MISMATCH':
                bad.setdefault(mid, []).append((name, want, got))
    print('STATUS:', json.dumps(stats, indent=1))
    print('MISMATCHES:', json.dumps(bad, indent=1))
