
import json, os, re, sys, pathlib, urllib.request

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
    t = str(label).split('(')[0].replace('+', 'plus').replace('＋', 'plus')
    return norm(t)

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

def parse_md_tables(doc):
    """Markdown pipe tables (most HF cards use these, not HTML)."""
    out, cur = [], []
    sep = re.compile(r'^:?-{2,}:?$')
    for line in doc.splitlines():
        t = line.strip()
        if t.startswith('|') and t.count('|') >= 2:
            row = [c.strip() for c in t.strip('|').split('|')]
            if row and all(sep.match(c.replace(' ', '')) for c in row if c):
                continue
            cur.append(row)
        else:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
    if len(cur) >= 2:
        out.append(cur)
    return [g for g in out if len(g) >= 2]

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

def candidate_cols(grids, keys, hints=()):
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
            if any(x in hn for x in hints):
                score += 8
            if score:
                cands.append((score, gi, ci, h))
    cands.sort(reverse=True)
    # Fallback: single-model cards often use a header like "Benchmark | Score"
    # with no model name in it. Offer every numeric column as a candidate; the
    # reproduce-the-values guard decides whether it is believable.
    numish = 0
    for gi, g in enumerate(grids):
        hdr = g[0]
        for ci in range(1, len(hdr)):
            vals = [r[ci] for r in g[1:] if len(r) > ci]
            hits = sum(1 for v in vals if num(v) is not None)
            if vals and hits >= 0.8 * len(vals):
                cands.append((0, gi, ci, hdr[ci] if ci < len(hdr) else ''))
                numish += 1

    if not cands:
        return []
    named = [c for c in cands if c[0] > 0]
    fb = [c for c in cands if c[0] == 0]
    if named:
        top = max(c[0] for c in named)
        named = [c for c in named if c[0] == top]
    return named + fb

def num(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r'-?\d+(?:\.\d+)?', str(s).replace(',', ''))
    return float(m.group(0)) if m else None

def hint_words(bms):
 w=set()
 for b in bms:
  c=(b.get("config") or "").lower()
  for k in ("max","nothink","non-think","instruct","it column","base","thinking"):
   if k in c: w.add(k)
 return w

def find_row(grid, name, ci):
    raw = norm(str(name).replace('+', 'plus'))
    for row in grid:
        if len(row) > ci and norm(str(row[0]).replace('+', 'plus')) == raw:
            return row            # exact label, parentheses included
    if '(' in str(name):
        # the parenthetical carried meaning ("RuBQ (ru audio QA)" vs the card's
        # text RuBQ row); never fall back to a paren-stripped key here
        return None
    bn = rowkey(name)
    for row in grid:
        if len(row) > ci and rowkey(row[0]) == bn:
            return row            # harness suffix dropped (Terminus-2 etc.)
    return None

def score_col(grid, ci, benchmarks):
    """Return (value_matches, resolved_rows): the column that *reproduces* the
    recorded numbers wins, so a card with two tables cannot mislead the pick."""
    ok = resolved = 0
    for b in benchmarks:
        want = num(b.get('score'))
        row = find_row(grid, b.get('name'), ci)
        if row is None:
            continue
        got = num(row[ci])
        if got is None:
            continue
        resolved += 1
        if want is not None and abs(got - want) <= 0.051:
            ok += 1
    return ok, resolved

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
    grids = parse_tables(html) + parse_md_tables(html)
    if not grids:
        return mid, 'no-tables', []
    keys = model_keys(mid, rec)
    bms = rec.get('benchmarks') or []
    cands = candidate_cols(grids, keys, hint_words(bms))
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
            best = (key, gi, eff, h, score)
    if best is None:
        return mid, 'ragged-table', []
    if best[0][1] == 0:
        return mid, 'ambiguous-column', []
    if best[4] == 0 and best[0][0] < 2:
        # column had no model name in its header; only trust it if it
        # independently reproduces at least two recorded values
        return mid, 'no-column', []
    gi, ci = best[1], best[2]
    results = []
    for b in (rec.get('benchmarks') or []):
        want = num(b.get('score'))
        bn = rowkey(b.get('name'))
        hit = find_row(grids[gi], b.get('name'), ci)
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
            # Before accusing, check whether ANOTHER column on the same card
            # reproduces the recorded value. Cards routinely publish a base-model
            # table next to the headline table; reading the wrong one is our bug,
            # not a data error.
            explained = False
            for _sc, gi2, ci2, _h in cands:
                g2 = grids[gi2]
                if not g2:
                    continue
                w2 = [len(r) for r in g2[1:] if len(r) > 1]
                if not w2:
                    continue
                md = max(set(w2), key=w2.count)
                off2 = len(g2[0]) - md
                eff2 = ci2 - off2
                if off2 < 0 or eff2 < 0 or eff2 >= md:
                    continue
                row2 = find_row(g2, b.get('name'), eff2)
                if row2 is not None:
                    v2 = num(row2[eff2])
                    if v2 is not None and abs(v2 - want) <= 0.051:
                        explained = True
                if explained:
                    break
            results.append((b.get('name'), want, got, 'ok-elsewhere' if explained else 'MISMATCH'))
    return mid, 'checked', results

def presence(mid):
    """Does the source each entry cites actually contain that benchmark name?

    For every benchmark it resolves the entry's OWN source id (falling back to
    the model's first HF card) and looks for the name in that document. A flag
    therefore means: the document we cite does not contain the number. That is a
    provenance gap, not automatically a fabricated value -- Qwen3-8B's card never
    says "MMLU", so those rows need the tech report added as a source.

    Names are matched on a normalised alphanumeric form, with aliases for
    "SWE Verified"/"SWE-bench Verified" and "GPQA"/"GPQA Diamond".

    KNOWN BLIND SPOTS: arXiv/whitepaper PDFs and JS-rendered pages are skipped,
    because raw bytes carry no readable table. Those sources yield no verdict
    rather than a flag. Do not treat this as a CI gate -- it is a worklist of
    records whose provenance needs a better citation.
    """
    p = R / 'data/models' / (mid + '.json')
    if not p.exists():
        return mid, 'missing-record', []
    rec = json.loads(p.read_text())
    texts, card = {}, None
    for ref in (rec.get('sources') or []):
        sp = R / 'data/sources' / (str(ref) + '.json')
        if not sp.exists():
            continue
        u = json.loads(sp.read_text()).get('url') or ''
        m = re.search(r'huggingface\.co/([^/]+/[^/?#]+)', u)
        try:
            if m:
                d = fetch('https://huggingface.co/%s/raw/main/README.md' % m.group(1))
            elif u.startswith('http'):
                d = fetch(u)
            else:
                continue
        except Exception:
            continue
        if d.startswith('%PDF') or '\x00' in d[:400]:
            continue
        if card is None:
            card = d
        texts[str(ref)] = d

    def nt(d):
        return ' ' + re.sub(r'[^a-z0-9]+', ' ', d.lower()).strip() + ' '

    if not texts:
        return mid, 'no-fetchable-source', []
    def compact(x):
        x = re.sub(r'v(?=\d)', '', x.lower())
        return re.sub(r'[^a-z0-9]+', '', x)

    def ckeys(nm):
        base = compact(nm)
        out = {base, compact(re.sub(r'\(.*?\)', '', nm))}
        out.add(re.sub(r'(bench|diamond)', '', base))
        out.add(re.sub(r'(bench|diamond|v\d+)', '', base))
        return {c for c in out if len(c) >= 3}

    ccard = compact(card) if card and len(card) >= 800 else None
    ctexts = {k: compact(v) for k, v in texts.items() if len(v) >= 400}
    miss = []
    for b in rec.get('benchmarks') or []:
        nm = b.get('name') or ''
        ct = ctexts.get(str(b.get('source'))) or ccard
        if not ct:
            continue
        if not any(c and c in ct for c in ckeys(nm)):
            miss.append(nm)
    return mid, ('ok' if not miss else 'unsupported'), miss


if __name__ == '__main__':
    argv = sys.argv[1:]
    flags = [a for a in argv if a.startswith('--')]
    ids = [a for a in argv if not a.startswith('--')]
    ci = '--ci' in flags
    if '--ids-file' in argv:
        ids = [l.strip() for l in open(argv[argv.index('--ids-file') + 1]) if l.strip()]
    if not ids:
        if ci:
            print('no model records changed; nothing to verify')
            sys.exit(0)
        ids = sorted(q.stem for q in (R / 'data/models').glob('*.json'))
    if '--present' in flags:
        bad2 = {}
        for mid in ids:
            try:
                mid, st, miss = presence(mid)
            except Exception:
                continue
            if miss:
                bad2[mid] = miss
        print('UNSUPPORTED:', json.dumps(bad2, indent=1))
        sys.exit(1 if bad2 else 0)
    stats, bad, checked = {}, {}, []
    for mid in ids:
        try:
            mid, st, res = check(mid)
        except Exception as e:
            mid, st, res = mid, 'error: %s' % type(e).__name__, []
        stats[st] = stats.get(st, 0) + 1
        if st == 'checked':
            ok_n = sum(1 for x in res if x[3] in ('ok', 'ok-elsewhere'))
            no_n = sum(1 for x in res if x[3] == 'no-row')
            checked.append((mid, ok_n, ok_n + no_n))
        for name, want, got, verdict in res:
            if verdict == 'MISMATCH':
                bad.setdefault(mid, []).append((name, want, got))
    print('STATUS:', json.dumps(stats, indent=1))
    print('MISMATCHES:', json.dumps(bad, indent=1))
    if not ci:
        sys.exit(0)
    lines = ['## Score verification', '',
             '`%d` model(s) cross-checked against vendor cards.' % len(checked), '',
             '| model | rows aligned |', '|---|---|']
    lines += ['| `%s` | %d/%d |' % c for c in checked]
    if bad:
        lines += ['### :x: %d mismatch(es)' % len(bad), '',
                  '| model | benchmark | recorded | vendor card |', '|---|---|---|---|']
        for m, rows in sorted(bad.items()):
            for n, w, g in rows:
                lines.append('| `%s` | %s | %s | %s |' % (m, n, w, g))
        lines += ['', 'Re-read the vendor table before editing. Bind cells to that',
                  'table\'s own header row, not to a header from another table.']
    else:
        lines += [':white_check_mark: No mismatches.', '']
    skipped = [k for k in sorted(stats) if k != 'checked']
    if skipped:
        lines += ['', 'Not cross-checked (could not align a column): ' +
                  ', '.join('%s=%d' % (k, stats[k]) for k in skipped)]
    out = '\n'.join(lines)
    print(out)
    sp = os.environ.get('GITHUB_STEP_SUMMARY')
    if sp:
        with open(sp, 'a') as fh:
            fh.write(out + '\n')
    sys.exit(1 if bad else 0)
