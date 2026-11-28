#!/usr/bin/env python3
"""Audit every agentic benchmark row against the source record it cites.

This is the permanent, in-repo version of the throwaway row-by-row audit that
caught two fabricated-looking scores and four corrupted source URLs. The old
scripts lived in a scratch directory that is pruned after 24h idle; the method
belongs here, where it can be re-run in review and covered by tests.

Method
------
1. Collect every model benchmark row whose name canonicalizes to an agentic
   family (tools/benchmark_names.py is the ONLY name table).
2. Resolve the row's ``source`` id through ``data/sources/<id>.json``. A URL
   embedded in the row is never trusted -- the source record is the audit trail.
3. Fetch that URL with a browser User-Agent, retrying transient failures 5x.
   PDFs are read with pypdf (an arXiv PDF has no scrapable text layer).
4. Normalise before comparing: 70.2, 70.20 and 0.702 are the same reported
   score under different precision / percent-vs-fraction conventions. The row's
   own ``scale`` decides which alternate form we offer, and the output says
   which form actually matched.
5. Report per-row PASS / MISMATCH / UNVERIFIABLE and the three counts.

Honesty limit
-------------
PASS means "this numeric value appears somewhere in the cited source text".
That is NECESSARY but NOT SUFFICIENT: the number may belong to another model in
the same table, another harness, or another benchmark. The tool prints a
snippet so a human can read the surrounding context; it does not and cannot
prove a row is correct. MISMATCH means "fetched the source, score not found" --
also not proof of error (a number can be present only in a figure image).
UNVERIFIABLE means the source could not be read at all; it says nothing about
the row's correctness.

    python3 tools/audit_agentic_rows.py --model deepseek-v3-2
    python3 tools/audit_agentic_rows.py --all
    python3 tools/audit_agentic_rows.py --all --json

Always dry: this tool never writes to data/ or anywhere else.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"
SOURCES = ROOT / "data" / "sources"

# Direct execution puts tools/ on sys.path already; an import by path (as the
# tests do) may not. Fall back rather than duplicating the name table.
try:
    from benchmark_names import canonical as _canonical  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised via test import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_names import canonical as _canonical  # noqa: E402

# A real browser UA: several vendor cards and CDNs 403 the default urllib
# agent. The retry pattern was the one the original audit used successfully.
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"
ATTEMPTS = 5
TIMEOUT = 30

# Number tokens, not substrings: "170.2" must not be read as containing 70.2.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TOL_ABS = 0.0051   # half a point when the source prints the score as a fraction
_TOL_REL = 0.002    # 0.2% when the source prints it on the row's own scale

PASS, MISMATCH, UNVERIFIABLE = "PASS", "MISMATCH", "UNVERIFIABLE"


class FetchError(Exception):
    """The source could not be read; the row is UNVERIFIABLE, not wrong."""


def _hf_token() -> str | None:
    """Optional HF token: many cited cards are gated and 401 anonymously."""
    import os
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".cache/huggingface/token",
              Path("/opt/data/home/.cache/huggingface/token")):
        try:
            if p.exists() and p.read_text(encoding="utf-8").strip():
                return p.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return None


def fetch_source(url: str, *, attempts: int = ATTEMPTS, timeout: int = TIMEOUT,
                 sleep=None) -> tuple[str, str]:
    """Fetch one source URL. Return (text, 'pdf'|'text'); raise FetchError.

    Browser UA, 5 attempts on transient failures (network error, timeout, 429,
    5xx). A clear 4xx is permanent: retrying it wastes 4 timeouts and changes
    nothing. PDF bytes go through pypdf; HTML is decoded with replacement so a
    stray non-UTF8 byte cannot kill an otherwise readable card.
    """
    sleep = sleep or time.sleep
    headers = {"User-Agent": UA, "Accept": ACCEPT}
    if "huggingface.co" in url:
        tok = _hf_token()
        if tok:
            headers["Authorization"] = "Bearer " + tok
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
                ctype = response.headers.get("Content-Type", "")
            if data[:4] == b"%PDF" or "pdf" in ctype.lower():
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                kind = "pdf"
            else:
                text = data.decode("utf-8", "replace")
                kind = "text"
            if not text.strip():
                raise FetchError("source fetched but no text could be extracted")
            return text, kind
        except FetchError as exc:
            # empty PDF text is a permanent property of the document
            raise FetchError(f"{url}: {exc}") from exc
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break  # permanent HTTP errors will not fix themselves
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            sleep(min(2 ** (attempt - 1), 8))
    raise FetchError(f"failed after {attempts} attempt(s): {last}")


def resolve_source(row: dict, sources_dir: Path = SOURCES) -> tuple[str | None, str | None, str | None]:
    """Return (url, source_id, error). The row's own URL field is ignored."""
    sid = row.get("source")
    if not isinstance(sid, str) or not sid.strip():
        return None, None, "row has no source id"
    sid = sid.strip()
    path = sources_dir / f"{sid}.json"
    if not path.is_file():
        return None, sid, f"no {path.relative_to(ROOT) if ROOT in path.parents else path}"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, sid, f"unreadable source record: {exc}"
    url = rec.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None, sid, f"source record has no usable url ({url!r})"
    if "[" in url or "]" in url:
        # Source URLs are schema-validated against exactly this corruption.
        # Reconstructing it by hand was a one-off repair, not an audit step.
        return None, sid, f"source url contains a redaction placeholder: {url!r}"
    return url, sid, None


def score_candidates(score: float, scale: float | None = 100) -> list[dict]:
    """Numeric forms the same score may take in a source, with an interpretation.

    ``scale=100`` (the catalog default): the card may print 70.2 or 0.702.
    ``scale=1``: the card may print 0.702 or 70.2. Any other/missing scale offers
    both forms and reports which one matched, instead of guessing.
    """
    value = float(score)
    out = [{"value": value, "interpretation": "as stored (row scale)"}]
    try:
        sc = float(scale) if scale is not None else 100.0
    except (TypeError, ValueError):
        sc = 100.0
    if sc == 100:
        out.append({"value": value / 100.0,
                    "interpretation": "fraction of 1 (row score / 100)"})
    elif sc == 1:
        out.append({"value": value * 100.0,
                    "interpretation": "percent (row score * 100)"})
    else:
        out.append({"value": value / 100.0, "interpretation": "fraction of 1 (/100)"})
        out.append({"value": value * 100.0, "interpretation": "percent (*100)"})
    # Drop duplicate forms (e.g. a zero score) keeping the first interpretation.
    seen, unique = set(), []
    for cand in out:
        key = round(cand["value"], 9)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def _close(a: float, b: float) -> bool:
    diff = abs(a - b)
    if diff <= _TOL_ABS:
        return True
    magnitude = max(abs(a), abs(b))
    return magnitude > 0 and diff / magnitude <= _TOL_REL


def compare(text: str, score: float, scale: float | None = 100) -> dict:
    """Pure, network-free core: does the fetched text contain this score?

    Returns PASS with the interpretation and a context snippet, or MISMATCH.
    Presence is necessary, not sufficient -- the snippet is for a human to read.
    """
    found = [(float(m.group(0)), m.start(), m.end()) for m in _NUMBER.finditer(text)]
    result = {
        "score": float(score),
        "candidates": score_candidates(score, scale),
        "matched_interpretation": None,
        "snippet": "",
    }
    for cand in result["candidates"]:
        for value, start, end in found:
            if _close(value, cand["value"]):
                snippet = re.sub(r"\s+", " ", text[max(0, start - 70):end + 70]).strip()
                return {**result, "status": PASS,
                        "matched_interpretation": cand["interpretation"],
                        "snippet": snippet}
    return {**result, "status": MISMATCH}


def iter_rows(models_dir: Path = MODELS, model: str | None = None):
    """Yield (slug, model_name, row) for every canonical agentic row."""
    paths = ([models_dir / f"{model}.json"] if model else sorted(models_dir.glob("*.json")))
    for path in paths:
        if not path.is_file():
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in rec.get("benchmarks") or []:
            if not isinstance(row, dict) or not _canonical(row.get("name") or ""):
                continue
            yield path.stem, rec.get("name") or path.stem, row


def audit_rows(rows, *, sources_dir: Path = SOURCES, fetcher=None) -> list[dict]:
    """Audit rows, caching one fetch per source URL. ``fetcher`` is injectable
    for offline tests; it must have fetch_source's (url) -> (text, kind) shape."""
    fetch = fetcher or fetch_source
    cache: dict[str, tuple[str, str] | str] = {}
    results = []
    for slug, model_name, row in rows:
        url, sid, err = resolve_source(row, sources_dir)
        base = {
            "model": slug, "model_name": model_name,
            "benchmark": row.get("name"),
            "canonical": _canonical(row.get("name") or ""),
            "score": row.get("score"), "source": sid, "url": url or "",
        }
        if err or url is None:
            result = {**base, "status": UNVERIFIABLE, "detail": err or "source returned no url",
                      "matched_interpretation": None, "snippet": ""}
        else:
            if url not in cache:
                try:
                    cache[url] = fetch(url)
                except Exception as exc:  # noqa: BLE001 - any fetch fault is unverifiable
                    cache[url] = f"{type(exc).__name__}: {exc}"
            fetched = cache[url]
            if isinstance(fetched, str):
                result = {**base, "status": UNVERIFIABLE, "detail": fetched,
                          "matched_interpretation": None, "snippet": ""}
            else:
                text, kind = fetched
                cmp = compare(text, row.get("score"), row.get("scale", 100))
                result = {**base, "status": cmp["status"], "detail": kind,
                          "matched_interpretation": cmp["matched_interpretation"],
                          "snippet": cmp["snippet"]}
        results.append(result)
    return results


def counts(results: list[dict]) -> dict[str, int]:
    return {status: sum(r["status"] == status for r in results)
            for status in (PASS, MISMATCH, UNVERIFIABLE)}


def print_report(results: list[dict]) -> None:
    for r in results:
        detail = r.get("matched_interpretation") or r.get("detail", "")
        print(f"{r['status']:<13} {r['model']} | {r['benchmark']} = {r['score']} "
              f"| source={r['source']} | {detail}")
        if r.get("snippet"):
            print(f"              snippet: {r['snippet'][:240]}")
    c = counts(results)
    print(f"\nPASS={c[PASS]} MISMATCH={c[MISMATCH]} UNVERIFIABLE={c[UNVERIFIABLE]} "
          f"TOTAL={len(results)}")
    print("NOTE: PASS means the score's numeric value was found somewhere in the cited "
          "source text.\nThat is necessary but NOT sufficient: the number may belong "
          "to a different model, harness,\nor benchmark in the same document. A human "
          "must read the surrounding context (snippets above).\nMISMATCH = source read "
          "but score not found (it may exist only in a figure image).\nUNVERIFIABLE = "
          "source could not be read; this says nothing about the row's correctness.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--model", help="audit one model by slug (review loop)")
    scope.add_argument("--all", action="store_true", help="audit every agentic row")
    ap.add_argument("--json", action="store_true", help="emit results as JSON")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="per-request seconds")
    args = ap.parse_args(argv)
    if not args.all and not args.model:
        args.all = True  # default scope is all; the run is always dry/read-only

    fetcher = fetch_source if args.timeout == TIMEOUT else (
        lambda url: fetch_source(url, timeout=args.timeout))
    results = audit_rows(iter_rows(model=args.model), fetcher=fetcher)
    if args.json:
        print(json.dumps({"counts": counts(results), "results": results}, indent=2))
    else:
        print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
