#!/usr/bin/env python3
"""Pick the next ingestion batch and remember what has been handed out.

Cron jobs call this each tick so the next run continues where the last stopped
instead of re-trawling the same rows. State lives outside the repo on purpose:
it is operational bookkeeping, not catalog data.

  ingest_queue.py stats
  ingest_queue.py deriv --n 120 --chunks 6     -> writes chunk files, prints them
  ingest_queue.py img   --n 8                  -> downloads images, prints model/files
  ingest_queue.py mark  --type deriv --chunk <file>      (marks that chunk's ids seen)
"""
import argparse
import glob
import json
import os
import pathlib
import re
import sys
import urllib.request

REPO = pathlib.Path('/opt/data/projects/inference-research/repo')
STATE = pathlib.Path('/opt/data/ingest_state')
STATE.mkdir(parents=True, exist_ok=True)
SEEN = STATE / 'seen.json'
IMG = pathlib.Path('/opt/data/img_evals')

QUANT = re.compile(r'(?i)(-gguf|-awq|-gptq|-fp8|-nvfp4|-bnb|-mlx|-exl[23]|-apex|-\d+bpw'
                   r'|-int[48]-|-w4a16|-w8a8|-onnx|-litert|-prismaquant|-q[2-8]_k_m'
                   r'|-dflash|-static|-dynamic-quant)')

# The same normalization mk_finetune_record.slug_of() uses for filenames. Record
# slugs have dots folded to hyphens ("Qwen3.8-4B" -> "qwen3-8-4b"), so comparing
# data/models/*.json stems against the raw repo name misses them and the queue
# re-serves repos that are already catalogued. See REPACK there too.
SLUGREPACK = re.compile(r'(-GGUF|-AWQ|-GPTQ|-FP8|-NVFP4|-bnb|-MLX|-EXL[23]|-\d+bpw|I?Q\d_|APEX)', re.I)


def record_slug(repo_id):
    name = SLUGREPACK.sub('', repo_id.split('/')[-1])
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def load_seen():
    if SEEN.exists():
        return json.loads(SEEN.read_text())
    return {'deriv': [], 'img': []}


def save_seen(s):
    SEEN.write_text(json.dumps({k: sorted(set(v)) for k, v in s.items()}, indent=0))


def model_ids():
    return {pathlib.Path(f).stem for f in glob.glob(str(REPO / 'data/models/*.json'))}


def deriv_candidates():
    rows = [json.loads(l) for l in (REPO / 'data/derivatives.jsonl').read_text().splitlines() if l.strip()]
    have = model_ids()
    # Ids keep their original case in the ledger, so compare case-insensitively;
    # matching rid.lower() against the raw list only ever caught the few entries
    # that happened to be lowercase already, and re-served everything else.
    seen = {x.lower() for x in load_seen()['deriv']}
    out = []
    for r in rows:
        rid = r.get('id') or ''
        if not rid or '/' not in rid:
            continue
        if QUANT.search(rid):
            continue
        if record_slug(rid) in have or rid.lower() in seen:
            continue
        if not (r.get('base') or ''):
            continue
        out.append({'id': rid, 'base': r['base'],
                    'downloads': r.get('downloads') or 0, 'likes': r.get('likes') or 0,
                    'created': r.get('created')})
    out.sort(key=lambda r: (-(r['downloads'] or 0), r['id']))
    return out


def card_scan():
    p = IMG / 'card_scan.json'
    return json.loads(p.read_text()) if p.exists() else []


def fetch(url, dest):
    """Download one card image. HF serves LFS-backed images under /raw/main/ as a
    131-byte git-LFS pointer stub rather than the image, so /resolve/main/ is the path
    that actually returns pixels; retry through it when the first fetch looks like a stub."""
    candidates = [url]
    if 'huggingface.co/' in url and '/raw/main/' in url:
        candidates.append(url.replace('/raw/main/', '/resolve/main/'))
    for u in candidates:
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            continue
        if len(data) < 2000 or data[:40].startswith(b'version https://git-lfs'):
            continue
        dest.write_bytes(data)
        return len(data)
    return 0


def img_pools():
    """Two triage classes, because `n_bench_ctx` is a keyword heuristic and not a verdict.

    `strong` = images the scanner matched benchmark wording near. `other` = images it could not
    classify: captioned only 'Performance', or with the numbers baked into the PNG and no
    benchmark word in the surrounding markdown. The strong class drains long before the figures
    do -- it sat at remaining: 0 while 78 cards still carried an unread image, and the real
    charts in that residue (Nanbeige4.1-3B, Nemotron IQ2) were invisible to `img` forever. So
    serve `other` as a second, explicitly weaker class once strong is empty.
    """
    s = load_seen()
    strong, other = [], []
    for r in card_scan():
        if r.get('n_bench_ctx') and r['id'] not in set(s['img']):
            strong.append(r)
        elif r.get('n_other') and not r.get('n_bench_ctx') and r['id'] not in set(s.get('img_other') or []):
            other.append(r)
    strong.sort(key=lambda x: (x.get('benchmarks', 0), -x.get('n_bench_ctx', 0)))
    other.sort(key=lambda x: (x.get('benchmarks', 0), -x.get('n_other', 0)))
    return strong, other


def img_batch(n):
    strong, other = img_pools()
    # Triaged-out cards are legitimate zeros (logos, architecture diagrams), so the weak class is
    # served to exhaustion too -- but only after the strong class, and marked separately so a
    # card can never be re-served as if it were still unread.
    cls, picks = ('img', strong[:n]) if strong else ('img_other', other[:n])
    out = []
    for r in picks:
        d = IMG / r['id']
        d.mkdir(parents=True, exist_ok=True)
        files = sorted(str(p) for p in d.glob('*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
        if not files:
            imgs = r.get('images') if r.get('n_bench_ctx') else r.get('other')
            # No [:8] cap: the stored list is the ONLY route to a figure, so capping it here
            # made every image past the 8th permanently unreachable and let `img` report
            # remaining: 0 with unread tables still on the card.
            for i, im in enumerate(imgs or []):
                ext = os.path.splitext(im['url'].split('?')[0])[1].lower() or '.png'
                if ext not in ('.png', '.jpg', '.jpeg', '.webp'):
                    ext = '.png'
                dest = d / ('%02d%s' % (i, ext))
                fetch(im['url'], dest)
            files = sorted(str(p) for p in d.glob('*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
        out.append({'model': r['id'], 'repo': r['repo'], 'class': cls,
                    'existing_rows': r.get('benchmarks', 0), 'files': files})
    return out


def cmd_stats(_):
    s = load_seen()
    d = deriv_candidates()
    strong, other = img_pools()
    print(json.dumps({'deriv_pending': len(d), 'deriv_seen': len(s['deriv']),
                      'img_pending': len(strong) + len(other),
                      'img_pending_strong': len(strong), 'img_pending_other': len(other),
                      'img_seen': len(s['img']), 'img_other_seen': len(s.get('img_other') or []),
                      'models_in_catalog': len(model_ids())}, indent=1))


def cmd_deriv(a):
    c = deriv_candidates()[:a.n]
    chunks = min(a.chunks, max(1, len(c)))
    paths = []
    for k in range(chunks):
        part = c[k::chunks]
        p = STATE / ('deriv_%02d.jsonl' % k)
        p.write_text('\n'.join(json.dumps(x) for x in part) + '\n')
        paths.append({'chunk': str(p), 'n': len(part)})
    print(json.dumps({'total': len(c), 'chunks': paths, 'remaining_after': len(deriv_candidates()) - len(c)}, indent=1))


def cmd_img(a):
    before_strong, before_other = (len(x) for x in img_pools())
    b = img_batch(a.n)
    if b and b[0]['class'] == 'img':
        rem_s, rem_o = before_strong - len(b), before_other
    else:
        rem_s, rem_o = before_strong, max(0, before_other - len(b))
    print(json.dumps({'models': b, 'remaining': rem_s + rem_o,
                      'remaining_strong': rem_s, 'remaining_other': rem_o}, indent=1))


def cmd_mark(a):
    s = load_seen()
    ids = []
    if a.chunk:
        ids = [json.loads(l)['id'] for l in pathlib.Path(a.chunk).read_text().splitlines() if l.strip()]
    if a.ids:
        ids += [x.strip() for x in a.ids.split(',') if x.strip()]
    s.setdefault(a.type, []).extend(ids)
    save_seen(s)
    print('marked %d %s ids seen (total %d)' % (len(ids), a.type, len(set(s[a.type]))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('stats').set_defaults(fn=cmd_stats)
    d = sub.add_parser('deriv'); d.add_argument('--n', type=int, default=120); d.add_argument('--chunks', type=int, default=6); d.set_defaults(fn=cmd_deriv)
    i = sub.add_parser('img'); i.add_argument('--n', type=int, default=8); i.set_defaults(fn=cmd_img)
    m = sub.add_parser('mark'); m.add_argument('--type', required=True, choices=['deriv', 'img', 'img_other'])
    m.add_argument('--chunk'); m.add_argument('--ids'); m.set_defaults(fn=cmd_mark)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
