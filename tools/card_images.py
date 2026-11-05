#!/usr/bin/env python3
"""Find the figures a Hugging Face model card actually embeds.

The queue decides a card is "image-free" from this module's output, and that verdict is
silent: a card whose figures are missed looks exactly like a card with no figures, gets
marked seen, and is never looked at again. Two spelling variants have already cost real
rows -- GitHub's `.../nah.png?raw=true` query string (which hid all of DeepSeek-V2's
benchmark charts) and an unquoted `src=` (google/gemma-4-12B-it). A third, an `<img>` whose
`src="` is split across a line break, is handled here too (omniomni/omni-0-mini-preview).

Two lessons are baked into the matching:

* Match over the WHOLE card, then recover the line number from the match offset. A
  line-by-line scan cannot see a tag that wraps, and markdown image syntax is allowed to
  wrap as well.
* Markdown/code that merely NAMES an image is not an embed. `Image.open("screenshot.png")`
  and `base64 < paragon.jpg` appear in usage snippets on dozens of cards; treating those
  as figures downloads sample JPEGs and wastes a queue slot on every pass.
"""
from __future__ import annotations

import re

# Extensions a vision model can be shown. Animated .gif is deliberately absent: these are
# always demo reels (2048/Snake/Tetris, handwriting animations), never result charts.
EXTS = r"(?:png|jpg|jpeg|webp|svg)"

# '![alt](url)' where url may carry a query string or fragment, may be wrapped in <>, and
# may itself be split across lines.
IMG = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]*?\.%s(?:[?#][^)\s>]*)?)" % EXTS, re.I | re.S)

# '<img ... src="url">', quoted or unquoted, with src= possibly separated from its value by
# whitespace/newlines. `[^>]*?` cannot swallow the closing '>' so the match stays one tag.
HTML_IMG = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?:\"([^\"]*?\.%s(?:[?#][^\"]*)?)\""
    r"|([^\s>'\"]*?\.%s(?:[?#][^\s>'\"]*)?))" % (EXTS, EXTS),
    re.I | re.S,
)

# Surrounding wording that suggests benchmark results rather than a logo or a photo.
BENCH = re.compile(r"(?i)(bench|eval|score|mmlu|gpqa|aime|swe-?bench|humaneval|hellaswag"
                   r"|arc|coding|math|leaderboard|result)")

# Badge/label artwork is not a figure. shields.io/colab/arxiv SVGs sit inside lines full of
# words like 'eval' or 'docs', so BENCH scores them as benchmark images and they eat the
# image cap -- on Qwen3-Omni-30B 5 of 6 'strong' images were colab badges.
BADGE = re.compile(r"(?i)(shields\.io|badge|colab\.research\.google\.com/assets|/badges?/"
                   r"|arxiv\.org/static|img\.shields|travis-ci|codecov|/logo[-_.]|[-_.]logo\.)")


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos)


def extract(card: str, repo: str | None = None, window: int = 3) -> list[dict]:
    """Return every embedded figure in `card`, in document order.

    Each entry is {url, line, bench_context, badge, ctx}. `url` is absolute when `repo` is
    given, since a relative figure path is meaningless to the downloader.
    """
    lines = card.splitlines()
    found: list[tuple[int, str]] = []

    # One pass per syntax, then merge in document order. Using finditer over the whole text
    # (rather than search() per line) is what recovers a wrapped tag.
    for rx in (IMG, HTML_IMG):
        for m in rx.finditer(card):
            url = m.group(1) or (m.group(2) if (m.lastindex or 0) >= 2 else None)
            if url:
                found.append((m.start(), url))
    found.sort()

    out: list[dict] = []
    seen: set[str] = set()
    for pos, url in found:
        raw = url.strip()
        if not raw:
            continue
        if raw.startswith("//"):
            url = "https:" + raw
        elif raw.startswith("./"):
            url = "https://huggingface.co/%s/resolve/main/%s" % (repo, raw[2:])
        elif not raw.startswith("http"):
            url = ("https://huggingface.co/%s/resolve/main/%s" % (repo, raw.lstrip("/"))
                   if repo else raw)
        else:
            url = raw
        if url in seen:          # a figure referenced twice is still one download
            continue
        seen.add(url)
        i = _line_of(card, pos)
        ctx = " ".join(lines[max(0, i - window):i + window])
        out.append({
            "url": url,
            "line": i,
            "bench_context": bool(BENCH.search(ctx)) or bool(BENCH.search(url)),
            "ctx": re.sub(r"\s+", " ", ctx)[:200],
        })
    for x in out:
        if BADGE.search(x["url"]):
            x["bench_context"] = False
            x["badge"] = True
    return out


def classify(figures: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (strong, other) for the queue's two-pass serving order.

    'other' is not proven irrelevant -- a figure captioned only "Performance", or one with
    the numbers baked into the PNG and no benchmark word nearby, lands there. Keeping it
    lets the queue run a weaker second pass instead of declaring itself empty.
    """
    strong = [x for x in figures if x["bench_context"]]
    other = [x for x in figures if not x["bench_context"] and not x.get("badge")]
    return strong, other
