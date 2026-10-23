#!/usr/bin/env python3
"""Ask the vision endpoint for one model's column in a benchmark figure.

  read_column.py <image> --model-name InternVL3-14B [--gw-model cap/vision] [--json]

Returns the rows it can read, plus the wall time. Nothing is applied to the catalog here:
the caller decides, after comparing against a second reader.
"""
import argparse, base64, json, mimetypes, os, pathlib, sys, time, urllib.request, urllib.error

GW = os.environ.get('AI_GATEWAY_URL', 'http://host.docker.internal:8090/v1')
KEY = os.environ.get('AI_GATEWAY_KEY', '')
UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--gw-model', default='cap/vision')
    ap.add_argument('--timeout', type=int, default=240)
    a = ap.parse_args()
    p = pathlib.Path(a.image)
    mime = mimetypes.guess_type(p.name)[0] or 'image/png'
    if mime == 'image/svg+xml':
        print(json.dumps({'image': str(p), 'error': 'svg'})); return 2
    b64 = base64.b64encode(p.read_bytes()).decode()
    payload = {'model': a.gw_model,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'text', 'text': prompt_for(a.model_name)},
                   {'type': 'image_url', 'image_url': {'url': 'data:%s;base64,%s' % (mime, b64)}}]}],
               'temperature': 0, 'max_tokens': 8000}
    req = urllib.request.Request(GW.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json',
                                          'User-Agent': UA})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(json.dumps({'image': str(p), 'error': 'HTTP %s %s' % (e.code, e.read().decode()[:200])})); return 2
    except Exception as e:
        print(json.dumps({'image': str(p), 'error': str(e)[:150]})); return 2
    dt = time.time() - t0
    msg = (d.get('choices') or [{}])[0].get('message', {}) or {}
    txt = msg.get('content') or ''
    if not txt and msg.get('reasoning_content'):
        # this endpoint is a reasoning model: it can spend the whole budget thinking and
        # still hand back the table inside reasoning_content, with finish_reason 'length'
        txt = msg['reasoning_content']
    if not txt:
        sys.stderr.write('EMPTY RESPONSE: %s\n' % json.dumps(d)[:400])
    rows, kind = [], 'text'
    if 'not_in_chart' in txt:
        kind = 'not_in_chart'
    elif 'not_a_table' in txt:
        kind = 'not_a_table'
    else:
        m = txt.find('[')
        if m >= 0:
            depth = 0
            for i, ch in enumerate(txt[m:], start=m):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            rows = json.loads(txt[m:i + 1])
                            kind = 'rows'
                        except Exception:
                            kind = 'unparsed'
                        break
    print(json.dumps({'image': str(p), 'model': a.model_name, 'gw_model': a.gw_model,
                      'kind': kind, 'seconds': round(dt, 1), 'rows': rows,
                      'raw_head': txt[:200] if kind in ('unparsed', 'text') else None}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

# Note: read_column.py parses the first JSON array in the reply and reports the
# wall time. Values may come back as strings with combined cells ("85.6 / 84.1");
# the caller must split those, never the reader.
