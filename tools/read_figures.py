#!/usr/bin/env python3
"""Ask a vision endpoint for one model's column in a benchmark figure.

  read_figures.py <image> --model-name InternVL3-14B [--chain m1,m2,...] [--attempts 2]

Providers on the gateway are flaky in different ways, so this walks a chain: each model gets
N attempts, and the first one that returns a usable answer wins. The reply records which
model answered, because "no rows" from a model that timed out means something very different
from "no rows" from a model that looked and found the model absent.

Chain note (measured 2026-10-23, all on the same InternVL3-14B figure):
  nous-portal-main/ornith/ornith-1.5-122b:free        34 s, 12/12 values, intermittent 504
  nous-portal-main/meituan/longcat-2.5-preview:free   72 s, values correct, intermittent 504
  cap/vision (alias -> gemini-3.8-flash -> mimo)      6-50 s, 12/12, steadiest
  nous-portal-main/stepfun/step-3.7-flash:free        400 "missing tags" — Kilo resale, unusable
"""
import argparse, base64, json, mimetypes, os, pathlib, re, sys, time, urllib.request, urllib.error

GW = os.environ.get('AI_GATEWAY_URL', 'http://host.docker.internal:8090/v1')
KEY = os.environ.get('AI_GATEWAY_KEY', '')
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'
DEFAULT_CHAIN = ('nous-portal-main/ornith/ornith-1.5-122b:free,'
                 'cap/vision,'
                 'nous-portal-main/meituan/longcat-2.5-preview:free')


def prompt_for(name):
    return ("This image is a benchmark figure from the Hugging Face model card for %s.\n"
            "Find the row or column for exactly this model and transcribe its numbers.\n\n"
            "Rules:\n"
            "- Output ONLY a JSON array: [{\"name\":\"<benchmark as printed>\",\"score\":<number>,"
            "\"scale\":<100 unless the table header states another maximum, e.g. a summed MME out of 3000,>"
            ",\"higher_is_better\":<true|false>}]\n"
            "- Copy the number exactly as printed. Never estimate, interpolate, average or reformat.\n"
            "- If this model is NOT in the figure, output exactly: not_in_chart\n"
            "- If the figure contains no benchmark numbers at all, output exactly: not_a_table\n"
            "- If a cell is blank or illegible, omit that row rather than guessing.\n"
            "- Skip aggregate rows such as Overall, Average or Mean.\n" % name)


def _split_combined(name, score):
    """'MMB (EN / CN)' with '85.6 / 84.1' -> two rows. Returns None if not a combined cell.

    Charts routinely stack two numbers in one cell, and filing it as one row silently loses a
    benchmark or, worse, files "85.6 / 84.1" as a single value.
    """
    if not isinstance(score, str) or '/' not in score:
        return None
    parts = [x.strip() for x in score.split('/')]
    labels = None
    m = re.search(r'\(([^)]*)\)', name or '')
    if m and '/' in m.group(1):
        labels = [x.strip() for x in m.group(1).split('/')]
    if not labels or len(labels) != len(parts):
        return None
    out = []
    for lab, val in zip(labels, parts):
        try:
            v = float(val.replace(',', ''))
        except Exception:
            return None
        # swap the stacked label group for this one label, keeping the rest of the name
        named = name[:m.start()] + '(' + lab + ')' + name[m.end():]
        out.append({'name': re.sub(r'\s+', ' ', named).strip(), 'score': v})
    return out


# only a name that IS an aggregate, not one that merely contains the word: 'HallBench (avg)'
# and 'POPE (avg)' are real benchmarks and dropping them loses data
AGG = re.compile(r'^\s*(overall|average|mean|total|avg)\s*(\(.*\))?\s*$', re.I)


def normalize(rows):
    """Clean up what vision models reliably get wrong: aggregates, stacked cells, absurd scales.

    Aggregates double count everything beside them. A 0-100 cell holding 3.49 is nearly always
    a score out of 10, so it is flagged rather than quietly trusted at the wrong scale.
    """
    clean, notes = [], []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get('name') or '').strip()
        if not name:
            continue
        if AGG.search(name):
            notes.append('dropped aggregate: %s' % name)
            continue
        split = _split_combined(name, r.get('score'))
        if split:
            for s in split:
                s['scale'] = r.get('scale') or 100
                s['higher_is_better'] = bool(r.get('higher_is_better', True))
                clean.append(s)
            notes.append('split stacked cell: %s' % name)
            continue
        try:
            score = float(str(r.get('score')).split('/')[0].strip())
        except Exception:
            notes.append('dropped unparseable score: %s' % name)
            continue
        row = {'name': name, 'score': score, 'scale': r.get('scale') or 100,
               'higher_is_better': bool(r.get('higher_is_better', True))}
        if row['scale'] == 100 and 0 < score <= 10:
            row['needs_scale_check'] = True
            notes.append('suspicious scale: %s = %s on a 0-100 cell' % (name, score))
        clean.append(row)
    return clean, notes


def _parse(txt):
    """First JSON array in the reply, plus a verdict on what the reply was."""
    if 'not_in_chart' in txt:
        return [], 'not_in_chart'
    if 'not_a_table' in txt:
        return [], 'not_a_table'
    m = txt.find('[')
    if m < 0:
        return [], 'text'
    depth = 0
    for i, ch in enumerate(txt[m:], start=m):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[m:i + 1]), 'rows'
                except Exception:
                    return [], 'unparsed'
    return [], 'unparsed'


def ask(model, p, mime_b64, name, timeout):
    payload = {'model': model,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'text', 'text': prompt_for(name)},
                   {'type': 'image_url', 'image_url': {'url': 'data:%s;base64,%s' % mime_b64}}]}],
               'temperature': 0, 'max_tokens': 8000}
    req = urllib.request.Request(GW.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json',
                                          'User-Agent': UA})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {'error': 'HTTP %s %s' % (e.code, e.read().decode()[:160])[:220],
                'seconds': round(time.time() - t0, 1), 'rows': [], 'kind': 'error'}
    except Exception as e:
        return {'error': str(e)[:150], 'seconds': round(time.time() - t0, 1), 'rows': [], 'kind': 'error'}
    msg = (d.get('choices') or [{}])[0].get('message', {}) or {}
    txt = msg.get('content') or msg.get('reasoning_content') or ''
    rows, kind = _parse(txt)
    return {'rows': rows, 'kind': kind, 'seconds': round(time.time() - t0, 1),
            'served_by': d.get('model'), 'error': None if txt else 'empty reply'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--gw-model', default=None, help='single model; overrides --chain')
    ap.add_argument('--chain', default=DEFAULT_CHAIN)
    ap.add_argument('--attempts', type=int, default=2)
    ap.add_argument('--normalize', action='store_true',
                    help='drop aggregates, split stacked cells, flag implausible scales')
    ap.add_argument('--timeout', type=int, default=240)
    a = ap.parse_args()
    p = pathlib.Path(a.image)
    mime = mimetypes.guess_type(p.name)[0] or 'image/png'
    if mime == 'image/svg+xml':
        print(json.dumps({'image': str(p), 'error': 'svg'})); return 2
    b64 = base64.b64encode(p.read_bytes()).decode()
    chain = [a.gw_model] if a.gw_model else [m.strip() for m in a.chain.split(',') if m.strip()]
    tried = []
    for model in chain:
        for attempt in range(1, a.attempts + 1):
            r = ask(model, p, (mime, b64), a.model_name, a.timeout)
            tried.append({'model': model, 'attempt': attempt, 'kind': r['kind'],
                          'seconds': r['seconds'], 'error': r['error']})
            # a 400/401/402 means this route will never work; don't retry it
            if r['error'] and ('HTTP 4' in r['error'] or 'HTTP 402' in r['error']):
                break
            if r['kind'] in ('rows', 'not_in_chart', 'not_a_table'):
                rows, notes = normalize(r['rows']) if a.normalize and r['kind'] == 'rows' else (r['rows'], [])
                print(json.dumps({'image': str(p), 'model': a.model_name, 'gw_model': model,
                                  'served_by': r.get('served_by'), 'kind': r['kind'],
                                  'seconds': r['seconds'], 'rows': rows, 'notes': notes,
                                  'attempts': tried}))
                return 0
            time.sleep(3 * attempt)
    print(json.dumps({'image': str(p), 'model': a.model_name, 'kind': 'failed',
                      'rows': [], 'attempts': tried}))
    return 2


if __name__ == '__main__':
    sys.exit(main())

# Note: values can come back as strings with combined cells ("85.6 / 84.1"); the caller
# splits those, never the reader. A reply of `not_in_chart` from a model that answered is a
# real verdict; `failed` (every route timed out) is not — treat those two differently.

