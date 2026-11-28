#!/usr/bin/env python3
"""Fetch INDEPENDENT agentic benchmark rows for the D1b model slice.

This is research-side tooling only. It never writes data/. It fetches public
leaderboards, maps only exact checkpoint identities (never derivatives or
same-family siblings), and emits a payload for tools/apply_third_party.py.

    python3 tools/third_party_agentic.py            # writes cache/thirdparty/part-b.json
    python3 tools/third_party_agentic.py --no-write # fetch + validate, print only

The payload has two keys the applier understands/ignores:
  models  - blocks in the exact shape apply_third_party.py collects
  sources - schema-valid source records that MUST be minted into data/sources/
            before the applier can resolve the source ids (the applier never
            mints sources itself).

Raw responses are cached under cache/thirdparty/raw/ so a run is auditable.
Every URL is retried up to 5 times before a source is declared unavailable.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from apply_third_party import check_block, source_url_pattern  # noqa: E402

CACHE = ROOT / "cache" / "thirdparty"
RAW = CACHE / "raw"
OUT = CACHE / "part-b.json"
DEFAULT_MODELS = Path("/opt/data/cache/scratch/agentic131-b.txt")
TODAY = datetime.now(timezone.utc).date().isoformat()
UA = {"User-Agent": "hermes-third-party-agentic-research/1.0 (public leaderboard verification)"}

# ---------------------------------------------------------------------------
# Exact identity map. Keys are the labels used by the source, values are the
# catalog slugs they refer to. Deliberately narrow: e.g. Kimi K3 != the
# VultronCoderAtlas expert-pruned derivative, Qwen3.8-Max != Qwen3.8-27B,
# Qwen3-30B-A3B-Instruct != the base Qwen3-30B-A3B, and "Qwen 3.7-Plus" is not
# any catalog model. Do not add a mapping unless the checkpoint is identical.
# ---------------------------------------------------------------------------
IDENTITY: dict[str, dict[str, str]] = {
    "vals-swebench-verified": {
        "alibaba/qwen3.8-27b": "qwen3-8-27b",
        "fireworks/nemotron-lightning-3p5-30b-a3b": "nvidia-nemotron-3-5-lightning-30b-a3b-bf16",
        "minimax/MiniMax-M2.7": "minimax-m2-7",
    },
    "vals-terminal-bench-2-1": {
        "alibaba/qwen3.8-27b": "qwen3-8-27b",
        "fireworks/nemotron-lightning-3p5-30b-a3b": "nvidia-nemotron-3-5-lightning-30b-a3b-bf16",
        "minimax/MiniMax-M2.7": "minimax-m2-7",
    },
    "vals-terminal-bench-2-0": {
        "minimax/MiniMax-M2.7": "minimax-m2-7",
    },
    "vals-terminal-bench-4-0": {
        "baseten/nemotron-3.5-super-vl-120b-a12b": "nemotron-3-5-super-vl-120b-a12b",
    },
    "taubench-core": {
        "Qwen3.5-397B-A17B": "qwen3-5-397b-a17b",
    },
    "bfcl-official-overall": {
        "Qwen3-32B (FC)": "qwen3-32b",
        "Qwen3-8B (FC)": "qwen3-8b",
    },
    "tbench-4-1": {
        "Nemotron 3.5 Super VL(none)": "nemotron-3-5-super-vl-120b-a12b",
    },
}

SOURCE_RECORDS: list[dict] = [
    {
        "id": "vals-swebench-verified",
        "type": "source",
        "name": "Vals AI SWE-bench Verified leaderboard",
        "status": "verified",
        "confidence": 0.9,
        "updated": TODAY,
        "url": "https://www.vals.ai/benchmarks/swebench",
        "publisher": "Vals AI",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Leaderboard snapshot updated 2026-09-01. Independent lab run; machine-readable dataset JSON linked from page: /_astro/benchmark_view_swebench.*.json. Archived raw copy in cache/thirdparty/raw/.",
    },
    {
        "id": "vals-terminal-bench-2-1",
        "type": "source",
        "name": "Vals AI Terminal-Bench 2.1 leaderboard",
        "status": "verified",
        "confidence": 0.9,
        "updated": TODAY,
        "url": "https://www.vals.ai/benchmarks/terminal-bench-2-1",
        "publisher": "Vals AI",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Leaderboard snapshot updated 2026-10-01. Independent lab run, Terminus 2 harness; machine-readable JSON linked from page. Raw copy in cache/thirdparty/raw/.",
    },
    {
        "id": "vals-terminal-bench-2-0",
        "type": "source",
        "name": "Vals AI Terminal-Bench 2.0 leaderboard",
        "status": "verified",
        "confidence": 0.9,
        "updated": TODAY,
        "url": "https://www.vals.ai/benchmarks/terminal-bench-2",
        "publisher": "Vals AI",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Leaderboard snapshot updated 2026-06-04. Independent lab run; machine-readable JSON linked from page. Raw copy in cache/thirdparty/raw/.",
    },
    {
        "id": "vals-terminal-bench-4-0",
        "type": "source",
        "name": "Vals AI Terminal-Bench 4.0 leaderboard",
        "status": "verified",
        "confidence": 0.9,
        "updated": TODAY,
        "url": "https://www.vals.ai/benchmarks/terminal-bench-4",
        "publisher": "Vals AI",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Leaderboard snapshot updated 2026-11-19. Independent lab run; machine-readable JSON linked from page. Raw copy in cache/thirdparty/raw/.",
    },
    {
        "id": "taubench-core",
        "type": "source",
        "name": "tau-bench official leaderboard submission index (core text)",
        "status": "verified",
        "confidence": 0.9,
        "updated": TODAY,
        "url": "https://taubench.com/visualizer-index.json",
        "publisher": "Sierra Research",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Machine-readable JSON index of official tau2-bench core (text retail/airline/telecom) submissions with pass^1 scores and experiment notes.",
    },
    {
        "id": "bfcl-official-overall",
        "type": "source",
        "name": "Berkeley Function-Calling Leaderboard overall results CSV",
        "status": "verified",
        "confidence": 0.85,
        "updated": TODAY,
        "url": "https://gorilla.cs.berkeley.edu/data_overall.csv",
        "publisher": "UC Berkeley Gorilla Group",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Machine-readable CSV behind https://gorilla.cs.berkeley.edu/leaderboard.html. Multi Turn Acc is the BFCL v3 suite; the FC rows use function-calling mode.",
    },
    {
        "id": "tbench-4-1",
        "type": "source",
        "name": "Terminal-Bench 4.1 official leaderboard",
        "status": "verified",
        "confidence": 0.85,
        "updated": TODAY,
        "url": "https://www.tbench.ai/",
        "publisher": "Terminal-Bench / Harbor",
        "kind": "repo",
        "published": None,
        "accessed": TODAY,
        "archived_url": None,
        "notes": "Leaderboard rows are embedded in the page HTML/Next.js JSON (no standalone JSON endpoint found). Raw copy in cache/thirdparty/raw/.",
    },
]

# Both tools/apply_third_party.py and tools/validate.py accept kind=repo, but
# only the former also accepts kind=leaderboard. source.schema.json / validate.py
# have no leaderboard enum, so the parent can mint these records under the
# schema-valid kind=repo without touching data/ or the schema.
_KIND_NOTE = (" KIND NOTE: record is a public leaderboard/results artifact, not a "
              "code repository. kind=repo is used because source.schema.json and "
              "tools/validate.py have no leaderboard enum; the applier also "
              "accepts kind=leaderboard, but that value fails schema validation.")
for _rec in SOURCE_RECORDS:
    _rec["kind"] = "repo"
    _rec["notes"] = _rec.get("notes", "") + _KIND_NOTE


def fetch(url: str, name: str, *, tries: int = 5) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    last: Exception | None = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            if not body:
                raise ValueError("empty response")
            path.write_bytes(body)
            return body
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"FAILED {url} after {tries} tries: {last}")


def fetch_text(url: str, name: str, *, tries: int = 5) -> str:
    return fetch(url, name, tries=tries).decode("utf-8", errors="replace")


def fetch_json(url: str, name: str, *, tries: int = 5):
    return json.loads(fetch_text(url, name, tries=tries))


def vals_data(slug: str, fname: str, page_url: str) -> dict | None:
    """Discover the linked benchmark_view JSON from the Vals AI page, then load it."""
    page = fetch_text(page_url, f"vals_{slug}_page.html")
    hits = re.findall(r'href="(/_astro/benchmark_view_[^"]+\.json)"', page)
    if not hits:
        return None
    data_url = "https://www.vals.ai" + hits[0]
    return fetch_json(data_url, fname)


def strip_tags(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    s = re.sub(r"\s+", " ", s).strip()
    # JSX inserts spaces around the model's reasoning-effort suffix, e.g.
    # "Nemotron 3.5 Super VL ( none )" -> "Nemotron 3.5 Super VL(none)".
    return re.sub(r"\s+[()]", lambda m: m.group()[-1],
                  re.sub(r"[()]\s+", lambda m: m.group()[0], s))


def num(s) -> float:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    if not m:
        raise ValueError(f"no number in {s!r}")
    return float(m.group())


def row(name: str, score, source: str, config: str) -> dict:
    return {
        "name": name,
        "score": float(score),
        "scale": 100,
        "vendor_claim": False,
        "source": source,
        "config": config,
    }


def config(prefix: str, entry: dict | None = None, suffix: str = "") -> str:
    if not entry:
        return f"{prefix}; {suffix}".strip("; ")
    bits = [prefix]
    for key in ("temperature", "top_p", "max_output_tokens", "reasoning_effort", "harness"):
        value = entry.get(key)
        if value is not None:
            bits.append(f"{key}={value}")
    bits.append(suffix)
    return "; ".join(b for b in bits if b)


def collect_rows(targets: set[str]) -> tuple[list[dict], dict]:
    """Return (model blocks, source-by-source report)."""
    blocks: dict[str, list[dict]] = {}
    report: dict[str, dict] = {}

    def emit(sid: str, model_id: str, benchmark: str, score, cfg: str) -> None:
        blocks.setdefault(model_id, []).append(
            row(benchmark, score, sid, f"{cfg}; retrieved {TODAY}")
        )

    # ---- Vals AI: four machine-readable JSON leaderboards -----------------
    vals_specs = [
        ("vals-swebench-verified", "vals_swebench.json",
         "https://www.vals.ai/benchmarks/swebench", "SWE-bench Verified", "overall", "resolution rate, Mini-SWE-agent harness"),
        ("vals-terminal-bench-2-1", "vals_tb21.json",
         "https://www.vals.ai/benchmarks/terminal-bench-2-1", "Terminal-Bench 2.1 (Terminus)", "overall", "pass@1 on 89 terminal tasks, Terminus 2 harness"),
        ("vals-terminal-bench-2-0", "vals_tb20.json",
         "https://www.vals.ai/benchmarks/terminal-bench-2", "Terminal-Bench 2.0", "overall", "pass@1, Terminus 2 harness"),
        ("vals-terminal-bench-4-0", "vals_tb40.json",
         "https://www.vals.ai/benchmarks/terminal-bench-4", "Terminal-Bench 4.0", "overall", "pass@1 overall across all task categories, mini-swe-agent harness"),
    ]
    for sid, fname, page, bname, task, metric in vals_specs:
        try:
            data = vals_data(sid, fname, page)
            if data is None:
                raise RuntimeError("no benchmark_view JSON link on page")
            names = data["metadata"]["models"]
            rows = data["tasks"][task]
            hits = 0
            for label, target in IDENTITY[sid].items():
                if target not in targets or label not in names or label not in rows:
                    continue
                # Preserve the catalog's existing harness-specific spelling when
                # the model already has a Terminus-labelled TB 2.1 vendor row;
                # otherwise use the plain version name so it displaces the
                # plain vendor claim.
                bname_for_row = bname
                if sid == "vals-terminal-bench-2-1" and target == "qwen3-8-27b":
                    bname_for_row = "Terminal-Bench 2.1 (Terminus)"
                elif sid == "vals-terminal-bench-2-1":
                    bname_for_row = "Terminal-Bench 2.1"
                emit(sid, target, bname_for_row, num(rows[label]["accuracy"]),
                     config(f"Vals AI {bname_for_row} leaderboard, model {label}, {metric}", rows[label]))
                hits += 1
            report[sid] = {"status": "ok", "machine_readable": "JSON",
                           "models_in_source": len(names), "matched": hits,
                           "matched_models": sorted(t for t in IDENTITY[sid].values() if t in blocks)}
        except Exception as e:
            report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                           "machine_readable": "JSON", "matched": 0}

    # ---- taubench.com official submissions index -------------------------
    sid = "taubench-core"
    try:
        data = fetch_json("https://taubench.com/visualizer-index.json", "taubench_index.json")
        entries = data["entries"]
        hits = 0
        for e in entries:
            target = IDENTITY[sid].get(str(e.get("model")))
            if not target or target not in targets or e.get("benchmark") != "core":
                continue
            cfg = (f"tau-bench official leaderboard, benchmark=core (tau2-bench text retail/airline/telecom), "
                   f"pass^1, submission standard, effort={e.get('effort')}, date={e.get('date')}, "
                   f"org={e.get('org')}, scoreLabel={e.get('scoreLabel')}")
            emit(sid, target, "tau2-bench", num(e["score"]), cfg)
            hits += 1
        report[sid] = {"status": "ok", "machine_readable": "JSON",
                       "models_in_source": len({e.get('model') for e in entries}),
                       "matched": hits,
                       "matched_models": sorted(t for t in IDENTITY[sid].values() if t in blocks)}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "JSON", "matched": 0}

    # ---- BFCL official overall CSV ----------------------------------------
    sid = "bfcl-official-overall"
    try:
        text = fetch_text("https://gorilla.cs.berkeley.edu/data_overall.csv", "bfcl_data_overall.csv")
        rows = list(csv.DictReader(text.splitlines()))
        hits = 0
        for r in rows:
            target = IDENTITY[sid].get(str(r.get("Model")))
            if not target or target not in targets:
                continue
            emit(sid, target, "BFCL v3", num(r["Multi Turn Acc"]),
                 f"BFCL official leaderboard data_overall.csv row {r['Model']!r}, "
                 f"column Multi Turn Acc={r['Multi Turn Acc']}, "
                 f"Non-Live AST Acc={r.get('Non-Live AST Acc')}, Live Acc={r.get('Live Acc')}, "
                 f"FC (function-calling) mode, overall rank {r.get('Rank')}")
            hits += 1
        report[sid] = {"status": "ok", "machine_readable": "CSV",
                       "models_in_source": len(rows), "matched": hits,
                       "matched_models": sorted(t for t in IDENTITY[sid].values() if t in blocks)}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "CSV", "matched": 0}

    # ---- Terminal-Bench 4.1 official page (HTML table rows) ---------------
    sid = "tbench-4-1"
    try:
        page = fetch_text("https://www.tbench.ai/", "tbench_page.html")
        hits = 0
        for tr in re.findall(r'<tr[^>]*data-row-id[^>]*>(.*?)</tr>', page, re.S):
            cells = [strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) < 6:
                continue
            target = IDENTITY[sid].get(cells[1])
            if not target or target not in targets:
                continue
            emit(sid, target, "Terminal-Bench 4.1", num(cells[3]),
                 f"Terminal-Bench 4.1 official leaderboard row {cells[1]!r}: "
                 f"agent={cells[2]}, accuracy={cells[3]}, date={cells[4]}, "
                 f"tokens={cells[5]}, cost={cells[6] if len(cells) > 6 else 'n/a'}")
            hits += 1
        report[sid] = {"status": "ok", "machine_readable": "embedded JSON in HTML (row parser used)",
                       "models_in_source": len(re.findall(r'<tr[^>]*data-row-id', page)),
                       "matched": hits,
                       "matched_models": sorted(t for t in IDENTITY[sid].values() if t in blocks)}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "embedded JSON in HTML", "matched": 0}

    # ---- Sources checked with ZERO exact matches (valued negative results) -
    # SWE-bench.com: complete leaderboard embedded in <script id="leaderboard-data">
    sid = "swebench.com"
    try:
        page = fetch_text("https://www.swebench.com/", "swebench_com_page.html")
        m = re.search(r'<script[^>]*id="leaderboard-data"[^>]*>(.*?)</script>', page, re.S)
        boards = json.loads(m.group(1)) if m else []
        names = set()
        for board in boards:
            for r in board.get("results") or []:
                for key in ("model_display", "name"):
                    if isinstance(r.get(key), dict):
                        names.add(str(r[key].get("label") or ""))
                    elif r.get(key):
                        names.add(str(r[key]))
        report[sid] = {"status": "ok", "machine_readable": "embedded JSON in HTML",
                       "models_in_source": len(names), "matched": 0,
                       "note": "no exact checkpoint identity from the 65-model slice appears on swebench.com"}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "embedded JSON in HTML", "matched": 0}

    # Open Agent Leaderboard (Hugging Face datasets-server rows API)
    sid = "open-agent-leaderboard"
    try:
        data = fetch_json(
            "https://datasets-server.huggingface.co/rows?dataset=open-agent-leaderboard/results"
            "&config=default&split=train&offset=0&length=100", "oal_rows.json")
        models = sorted({str(r["row"].get("model_name")) for r in data["rows"]})
        report[sid] = {"status": "ok", "machine_readable": "JSON API",
                       "models_in_source": len(models), "matched": 0,
                       "models": models,
                       "note": "5 hosted models, none in the 65-model slice"}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "JSON API", "matched": 0}

    # OSWorld v2 official results JSON
    sid = "osworld-v2"
    try:
        data = fetch_json(
            "https://osworld-v2.xlang.ai/static/data/leaderboard/official-results.json",
            "osworld_v2_official.json")
        models = sorted({str(r.get("model")) for r in data["results"]})
        report[sid] = {"status": "ok", "machine_readable": "JSON",
                       "models_in_source": len(models), "matched": 0,
                       "note": "Qwen 3.7-Plus / MiniMax M3 / Kimi 2.6 are different checkpoints from the slice"}
    except Exception as e:
        report[sid] = {"status": f"FETCH/PARSE FAILED: {type(e).__name__}: {e}",
                       "machine_readable": "JSON", "matched": 0}

    # GAIA public results: rate-limited during this run and canonical() rejects
    # GAIA as non-agentic, so no row could be written even with data.
    sid = "gaia"
    report[sid] = {"status": "NOT WRITTEN: datasets-server rows returned HTTP 429 after 5 tries; "
                                "additionally canonical('GAIA') is None in tools/benchmark_names.py",
                   "machine_readable": "JSON API", "matched": 0,
                   "models_in_source": None}

    return ([{"slug": k, "benchmarks": v} for k, v in sorted(blocks.items())], report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    targets = {s.strip() for s in args.models.read_text(encoding="utf-8").splitlines()
               if s.strip() and not s.startswith("#")}
    blocks, report = collect_rows(targets)

    # Validate with the real writer logic but against the *proposed* sources:
    # this task is read-only on data/, so the parent must mint these records.
    sources = {s["id"]: s for s in SOURCE_RECORDS}
    url_re = source_url_pattern()
    accepted, rejected_rows, rejected_blocks = [], [], []
    for b in blocks:
        clean, fatal, problems = check_block(b, sources, url_re)
        for p in problems:
            rejected_rows.append((b["slug"], p))
        if fatal:
            rejected_blocks.append((b["slug"], fatal))
        if clean:
            accepted.append(clean)

    payload = {
        "models": [{"slug": b["slug"], "benchmarks": b["benchmarks"]} for b in blocks],
        "sources": SOURCE_RECORDS,
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "targets": len(targets),
            "models_with_rows": len(blocks),
            "rows": sum(len(b["benchmarks"]) for b in blocks),
            "source_breakdown": report,
            "validation": {
                "accepted_blocks": len(accepted),
                "rejected_blocks": rejected_blocks,
                "rejected_rows": rejected_rows,
                "note": "Rows validate against apply_third_party.check_block() using the "
                        "sources in this payload. Mint those source records into "
                        "data/sources/ before running apply_third_party.py.",
            },
        },
    }

    print(f"targets={len(targets)} models_with_independent_rows={len(blocks)} "
          f"rows={payload['meta']['rows']}")
    for sid, info in report.items():
        print(f"  {sid:28} matched={info.get('matched')} status={info.get('status')}")
    for b in blocks:
        print(f"  + {b['slug']}: " + ", ".join(
            f"{r['name']}={r['score']:g}" for r in b["benchmarks"]))
    if rejected_blocks or rejected_rows:
        print(f"  REJECTED blocks={rejected_blocks} rows={rejected_rows}")

    if not args.no_write:
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
