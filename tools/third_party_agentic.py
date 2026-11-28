#!/usr/bin/env python3
"""Look up INDEPENDENT agentic-benchmark rows for catalog model slugs.

This is the read-only research half of the third-party evidence pipeline. It
fetches public leaderboards, matches model slugs to the exact leaderboard row,
and emits payloads in the shape ``tools/apply_third_party.py`` accepts. It never
writes to ``data/``: the companion writer is the only sanctioned writer.

    python3 tools/third_party_agentic.py --models /path/to/models.txt \
        --out cache/thirdparty/part-a.json
    python3 tools/third_party_agentic.py --check cache/thirdparty/part-a.json
    python3 tools/third_party_agentic.py --self-test

Sources (all fetched live, five attempts each):

  * swebench.com  -- official leaderboards, machine-readable JSON
      https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json
    Only rows the SWE-bench team performed or checked (``checked: true``) are
    treated as independent. Vendor submissions with ``checked: false`` are
    recorded as rejected candidates, never emitted.
  * Open Agent Leaderboard (Exgentic) -- HF dataset, machine-readable rows API
      https://datasets-server.huggingface.co/rows?dataset=open-agent-leaderboard/results
    The maintainers run every agent x model x benchmark cell. A fixed neutral
    scaffold is used (tool_calling, then smolagents_code) so the choice is not
    "best score across scaffolds".
  * Terminal-Bench -- official leaderboard, server-rendered HTML table
      https://www.tbench.ai/
    Rows are maintainer-reviewed submissions (see the TB submission policy).
    The 2.1/2.0 tabs are client-rendered and expose no row data in the server
    HTML; they are recorded as unavailable rather than scraped from a browser.
  * GAIA -- HF dataset, machine-readable rows API
      https://datasets-server.huggingface.co/rows?dataset=gaia-benchmark/results_public
    Rows are matched and reported, but ``tools/benchmark_names.py`` explicitly
    classifies GAIA as non-agentic, so no GAIA row can be written. This is a
    project rule, not an oversight.

Model identity is deliberately strict: compact alphanumeric equality, plus a
small hand-verified alias table for the three cases where the leaderboard label
differs from the catalog slug (Kimi K2 Instruct, DeepSeek V3.2, Beam 1.0).
Substring matching is never used, because "Kimi K3.1 Pro" is not Kimi K3 and
"DeepSeek V4 Pro 0813" is not DeepSeek V4 Pro.

A row is emitted only when all three are true: the number was fetched, the
harness/settings are quoted from the source, and the source id resolves to a
real record in ``data/sources/``. Rows whose proposed source record does not
exist yet are written to ``blocked`` with the full evidence, so D2 can add the
record and re-run this tool to promote them without re-researching.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_names import canonical as canonical_benchmark, family_of  # noqa: E402

UA = "Mozilla/5.0 (compatible; inference-research/1.0; third-party agentic lookup)"
RETRIES = 5

# Proposed source records. These are NOT written by this tool; they are the
# contract D2 must satisfy in data/sources/ before any row below can be applied.
# Schema-valid source records D2 can create today. schemas/source.schema.json
# does not allow kind="leaderboard" (its enum stops at "database"), even though
# apply_third_party.INDEPENDENT_KINDS accepts it. These templates use kind="repo"
# so they pass the schema gate without a schema change; the exact fetched URL is
# recorded in notes. If the parent prefers a semantic "leaderboard" kind, the
# schema enum must be extended first.
SOURCE_RECORD_TEMPLATES = {
    "leaderboard-swebench-com": {
        "id": "leaderboard-swebench-com", "type": "source",
        "name": "SWE-bench official leaderboards (swebench.com)",
        "status": "verified", "confidence": 0.95, "updated": "2026-11-28",
        "url": "https://github.com/SWE-bench/swe-bench.github.io",
        "publisher": "SWE-bench Team", "kind": "repo", "published": None,
        "accessed": "2026-11-28", "archived_url": None, "vendor_silence": None,
        "notes": "Official leaderboard. Data fetched from https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json; rendered at https://www.swebench.com/. Only checked=true rows are independent.",
    },
    "leaderboard-open-agent-leaderboard": {
        "id": "leaderboard-open-agent-leaderboard", "type": "source",
        "name": "Open Agent Leaderboard (Exgentic) results dataset",
        "status": "verified", "confidence": 0.9, "updated": "2026-11-28",
        "url": "https://huggingface.co/datasets/open-agent-leaderboard/results",
        "publisher": "Exgentic / Open Agent Leaderboard", "kind": "repo", "published": None,
        "accessed": "2026-11-28", "archived_url": None, "vendor_silence": None,
        "notes": "Maintainer-run agent x model x benchmark matrix. Rows fetched from the HF datasets-server rows API; fixed neutral scaffold tool_calling.",
    },
    "leaderboard-terminal-bench": {
        "id": "leaderboard-terminal-bench", "type": "source",
        "name": "Terminal-Bench official leaderboard",
        "status": "verified", "confidence": 0.9, "updated": "2026-11-28",
        "url": "https://github.com/harbor-framework/terminal-bench",
        "publisher": "Harbor Framework / Laude Institute", "kind": "repo", "published": None,
        "accessed": "2026-11-28", "archived_url": None, "vendor_silence": None,
        "notes": "Official leaderboard rendered at https://www.tbench.ai/ from maintainer-reviewed submissions. Rows fetched from the server-rendered 4.1 table.",
    },
    "leaderboard-gaia": {
        "id": "leaderboard-gaia", "type": "source",
        "name": "GAIA leaderboard results (results_public)",
        "status": "verified", "confidence": 0.9, "updated": "2026-11-28",
        "url": "https://huggingface.co/datasets/gaia-benchmark/results_public",
        "publisher": "GAIA benchmark authors", "kind": "repo", "published": None,
        "accessed": "2026-11-28", "archived_url": None, "vendor_silence": None,
        "notes": "Matched for completeness only: GAIA is non-agentic per tools/benchmark_names.py, so no row is emittable.",
    },
}

PROPOSED_SOURCES = {
    "leaderboard-swebench-com": {
        "kind": "leaderboard",
        "url": "https://www.swebench.com/",
        "name": "SWE-bench official leaderboards (swebench.com)",
        "machine_readable": True,
        "endpoint": "https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json",
        "note": "Official leaderboard. Only checked=true rows (run performed or directly checked by the SWE-bench team) are emitted.",
    },
    "leaderboard-open-agent-leaderboard": {
        "kind": "leaderboard",
        "url": "https://huggingface.co/datasets/open-agent-leaderboard/results",
        "name": "Open Agent Leaderboard (Exgentic) results dataset",
        "machine_readable": True,
        "endpoint": "https://datasets-server.huggingface.co/rows?dataset=open-agent-leaderboard/results",
        "note": "Maintainer-run agent x model x benchmark matrix; fixed neutral scaffold (tool_calling).",
    },
    "leaderboard-terminal-bench": {
        "kind": "leaderboard",
        "url": "https://www.tbench.ai/",
        "name": "Terminal-Bench official leaderboard",
        "machine_readable": False,
        "endpoint": "https://www.tbench.ai/ (server-rendered HTML table; 2.1/2.0 tabs are client-rendered)",
        "note": "Official leaderboard of maintainer-reviewed submissions; harness/effort quoted per row.",
    },
    "leaderboard-gaia": {
        "kind": "leaderboard",
        "url": "https://huggingface.co/spaces/gaia-benchmark/leaderboard",
        "name": "GAIA leaderboard results (results_public)",
        "machine_readable": True,
        "endpoint": "https://datasets-server.huggingface.co/rows?dataset=gaia-benchmark/results_public",
        "note": "Matched for completeness only: tools/benchmark_names.py classifies GAIA as non-agentic, so no row is emittable.",
    },
}

# Hand-verified aliases: catalog slug -> {source key: exact leaderboard labels}.
# Everything not listed here must match by compact alphanumeric equality.
ALIASES = {
    "deepseek-v3-2": {
        "swebench": ["DeepSeek V3.2"],
        "oal": ["openai/azure/DeepSeek-V3.2"],
    },
    "kimi-k2": {
        "swebench": ["Kimi K2 Instruct"],
    },
    "beam-1-0-501b-a23b": {
        "tbench": ["Beam 1.0"],
    },
}

# Fixed OAL scaffold preference. Never "max across agents": that would be
# cherry-picking. First present scaffold wins.
OAL_AGENT_PREFERENCE = ["tool_calling", "smolagents_code"]

OAL_BENCHMARK_NAMES = {
    "tau2_retail": "tau2-bench Retail",
    "tau2_airline": "tau2-bench Airline",
    "tau2_telecom": "tau2-bench Telecom",
    "browsecompplus": "BrowseComp Plus",
    "swebench": "SWE-bench",
}

SWEBENCH_BOARD_NAMES = {
    "Verified": "SWE-bench Verified",
    "Multilingual": "SWE-bench Multilingual",
    "Lite": "SWE-bench Lite",
    "Test": "SWE-bench",
    "Multimodal": "SWE-bench Multimodal",
}


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def fetch(url: str, tries: int = RETRIES, timeout: int = 90) -> bytes | None:
    """GET with bounded exponential retry. Returns None only after all tries."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network/HTTP/timeout are all retryable here
            last = exc
            if attempt + 1 < tries:
                # 429s from the HF rows API are per-minute; use a long enough
                # backoff that a fresh run after a rate limit still succeeds.
                time.sleep(min(40, 2 ** (attempt + 1)))
    print(f"  ! fetch failed after {tries} attempts: {url} ({type(last).__name__}: {last})",
          file=sys.stderr)
    return None


def fetch_json(url: str):
    raw = fetch(url)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        print(f"  ! not JSON: {url} ({exc})", file=sys.stderr)
        return None


def matches(label: str, slug: str, source_key: str) -> bool:
    """Exact identity only: compact equality, or a hand-verified alias."""
    if compact(label) == compact(slug):
        return True
    return label in ALIASES.get(slug, {}).get(source_key, [])


# --------------------------------------------------------------------------
# swebench.com
# --------------------------------------------------------------------------
def swebench_rows(models: list[str]) -> tuple[list[dict], list[dict], dict]:
    data = fetch_json(
        "https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json"
    )
    stats = {"source": "swebench.com", "machine_readable": True, "boards": {}, "rejected": []}
    if not data:
        stats["error"] = "fetch failed"
        return [], [], stats
    emitted, blocked = [], []
    for board in data.get("leaderboards", []):
        board_name = board.get("name")
        results = board.get("results", [])
        stats["boards"][board_name] = len(results)
        bench = SWEBENCH_BOARD_NAMES.get(board_name)
        if not bench:
            continue
        for row in results:
            label = row.get("name") or ""
            for slug in models:
                if not matches(label, slug, "swebench"):
                    continue
                checked = bool(row.get("checked"))
                if not checked:
                    stats["rejected"].append(
                        {"slug": slug, "board": board_name, "name": label,
                         "score": row.get("resolved"),
                         "reason": "checked=false: vendor/agent submission, not independent"}
                    )
                    continue
                version = row.get("mini-swe-agent_version") or "unknown"
                config = (
                    f"mini-SWE-agent v{version}, bash-only (no tools, ReAct loop), "
                    f"{bench} board; run performed or directly checked by the SWE-bench team "
                    f"(checked=true); date {row.get('date')}"
                )
                item = {
                    "name": bench,
                    "score": float(row["resolved"]),
                    "scale": 100,
                    "vendor_claim": False,
                    "source": "leaderboard-swebench-com",
                    "config": config,
                }
                (emitted if _source_exists("leaderboard-swebench-com") else blocked).append(
                    {"slug": slug, "benchmark": item, "evidence": {
                        "board": board_name, "label": label, "checked": checked,
                        "agent": row.get("agent"), "date": row.get("date"),
                        "url": "https://www.swebench.com/",
                    }}
                )
    return emitted, blocked, stats


# --------------------------------------------------------------------------
# Open Agent Leaderboard
# --------------------------------------------------------------------------
def _oal_pages() -> list[dict]:
    rows, offset = [], 0
    while True:
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=open-agent-leaderboard%2Fresults&config=default&split=train"
               f"&offset={offset}&length=100")
        data = fetch_json(url)
        if not data:
            break
        page = data.get("rows", [])
        rows.extend(r.get("row", {}) for r in page)
        total = data.get("num_rows_total", 0)
        offset += len(page)
        if not page or offset >= total:
            break
        time.sleep(0.4)  # stay below the rows-API per-minute ceiling
    return rows


def oal_rows(models: list[str]) -> tuple[list[dict], list[dict], dict]:
    rows = _oal_pages()
    stats = {"source": "open-agent-leaderboard", "machine_readable": True,
             "rows": len(rows), "models": sorted({r.get("model_name") for r in rows if r.get("model_name")}),
             "agents": sorted({r.get("agent") for r in rows if r.get("agent")})}
    emitted, blocked = [], []
    if not rows:
        stats["error"] = "no rows fetched"
        return emitted, blocked, stats
    for slug in models:
        candidates = [r for r in rows
                      if matches(r.get("model_name") or "", slug, "oal")
                      or matches(r.get("model") or "", slug, "oal")]
        if not candidates:
            continue
        agents_present = {r.get("agent") for r in candidates if r.get("agent")}
        agent = next((a for a in OAL_AGENT_PREFERENCE if a in agents_present), None)
        if agent is None:
            stats.setdefault("skipped", []).append(
                {"slug": slug, "reason": f"no preferred scaffold; present={sorted(a for a in agents_present if a)}"})
            continue
        for row in candidates:
            if row.get("agent") != agent:
                continue
            bench_key = row.get("benchmark")
            bench = OAL_BENCHMARK_NAMES.get(bench_key)
            if not bench:
                continue
            score = row.get("benchmark_score")
            if not isinstance(score, (int, float)):
                continue
            config = (
                f"Open Agent Leaderboard (Exgentic), agent={agent}, "
                f"benchmark={bench_key}, {row.get('successful_sessions')}/"
                f"{row.get('total_sessions')} sessions successful, "
                f"benchmark_score={score}"
            )
            item = {"name": bench, "score": float(score) * 100, "scale": 100,
                    "vendor_claim": False, "source": "leaderboard-open-agent-leaderboard",
                    "config": config}
            (emitted if _source_exists("leaderboard-open-agent-leaderboard") else blocked).append(
                {"slug": slug, "benchmark": item, "evidence": {
                    "agent": agent, "model": row.get("model_name"),
                    "benchmark": bench_key, "sessions": row.get("total_sessions"),
                    "url": "https://huggingface.co/datasets/open-agent-leaderboard/results",
                }}
            )
    return emitted, blocked, stats


# --------------------------------------------------------------------------
# Terminal-Bench
# --------------------------------------------------------------------------
def _clean_cell(cell: str) -> str:
    text = re.sub(r"<[^>]+>", " ", cell)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def terminal_bench_rows(models: list[str]) -> tuple[list[dict], list[dict], dict]:
    raw = fetch("https://www.tbench.ai/")
    stats = {"source": "terminal-bench", "machine_readable": False,
             "url": "https://www.tbench.ai/", "note":
             "4.1 table is server-rendered HTML; 2.1/2.0 tabs are client-rendered "
             "and expose no rows in the server payload"}
    emitted, blocked = [], []
    if raw is None:
        stats["error"] = "fetch failed"
        return emitted, blocked, stats
    doc = raw.decode("utf-8", "replace")
    trs = re.findall(r'<tr[^>]*data-row-id="[^"]+"[^>]*>(.*?)</tr>', doc, re.S)
    stats["rows"] = len(trs)
    stats["version"] = "4.1" if "Terminal-Bench 4.1" in doc else "unknown"
    for tr in trs:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(cells) < 7:
            continue
        label = _clean_cell(cells[1])
        model_label = re.sub(r"\s*\(\s*\w+\s*\)\s*$", "", label).strip()
        agent = _clean_cell(cells[2])
        rate = _clean_cell(cells[3])
        date = _clean_cell(cells[4])
        model_url = ""
        agent_url = ""
        mh = re.search(r'<a href="([^"]+)"', cells[1])
        ah = re.search(r'<a href="([^"]+)"', cells[2])
        if mh:
            model_url = html.unescape(mh.group(1))
        if ah:
            agent_url = html.unescape(ah.group(1))
        eff = re.search(r"\(\s*(\w+)\s*\)\s*$", label)
        effort = eff.group(1) if eff else "none"
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)%", rate)
        if not m:
            continue
        score = float(m.group(1))
        ci = re.search(r"±\s*([0-9.]+)%", rate)
        for slug in models:
            if not (matches(model_label, slug, "tbench") or matches(model_url, slug, "tbench")):
                continue
            config = (
                f"Terminal-Bench 4.1 official leaderboard, agent={agent}, "
                f"reasoning_effort={effort}, {score:.1f}% ± {ci.group(1) if ci else '?'}% "
                f"(95% CI), model release {date}"
            )
            item = {"name": "Terminal-Bench 4.1", "score": score, "scale": 100,
                    "vendor_claim": False, "source": "leaderboard-terminal-bench",
                    "config": config}
            (emitted if _source_exists("leaderboard-terminal-bench") else blocked).append(
                {"slug": slug, "benchmark": item, "evidence": {
                    "label": label, "model_url": model_url, "agent": agent,
                    "agent_url": agent_url, "rate": rate, "date": date,
                    "url": "https://www.tbench.ai/",
                }}
            )
    return emitted, blocked, stats


# --------------------------------------------------------------------------
# GAIA (matched, but not emittable: canonical() rejects GAIA as non-agentic)
# --------------------------------------------------------------------------
def gaia_report(models: list[str]) -> dict:
    rows, offset = [], 0
    while True:
        url = ("https://datasets-server.huggingface.co/rows?"
               "dataset=gaia-benchmark%2Fresults_public&config=2023&split=test"
               f"&offset={offset}&length=100")
        data = fetch_json(url)
        if not data:
            break
        page = data.get("rows", [])
        rows.extend(r.get("row", {}) for r in page)
        total = data.get("num_rows_total", 0)
        offset += len(page)
        if not page or offset >= total:
            break
    hits = {}
    for row in rows:
        blob = compact((row.get("model") or "") + " " + (row.get("model_family") or ""))
        for slug in models:
            key = compact(slug)
            if len(key) >= 7 and key in blob:
                hits.setdefault(slug, []).append(
                    {"agent": row.get("model"), "family": row.get("model_family"),
                     "score": row.get("score"), "org": row.get("organisation")})
    return {
        "source": "gaia",
        "machine_readable": True,
        "rows": len(rows),
        "matched_models": {k: v[:5] for k, v in hits.items()},
        "emitted": 0,
        "reason": "tools/benchmark_names.py classifies GAIA as non-agentic "
                  "(_NON_AGENTIC); canonical() returns None, so the writer rejects GAIA rows",
    }


def _source_exists(source_id: str) -> bool:
    return (ROOT / "data" / "sources" / f"{source_id}.json").exists()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build(models: list[str]) -> dict:
    emitted, blocked = [], []
    stats = {}
    for fn in (swebench_rows, oal_rows, terminal_bench_rows):
        e, b, s = fn(models)
        emitted.extend(e)
        blocked.extend(b)
        stats[s["source"]] = s
    stats["gaia"] = gaia_report(models)

    by_slug: dict[str, list[dict]] = {}
    for row in emitted:
        by_slug.setdefault(row["slug"], []).append(row["benchmark"])
    blocked_by_slug: dict[str, list[dict]] = {}
    for row in blocked:
        blocked_by_slug.setdefault(row["slug"], []).append(row)

    # Round the published rate to the precision the leaderboard prints; never
    # let a float artefact (56.00000000000001) reach the record diff.
    for rows in by_slug.values():
        for row in rows:
            row["score"] = round(float(row["score"]), 2)
    for entry in blocked:
        row = entry["benchmark"]
        row["score"] = round(float(row["score"]), 2)
        entry["proposed_source"] = PROPOSED_SOURCES[row["source"]]

    agentic_families = ("terminal", "web", "tool", "computer")

    def _agentic(rows):
        return any(family_of(canonical_benchmark(r["name"])) in agentic_families
                   for r in rows)

    emitted_agentic = {s for s, rows in by_slug.items() if _agentic(rows)}
    fetched_agentic = {s for s, rows in
                       list(by_slug.items()) + [(k, [r["benchmark"] for r in v])
                                                for k, v in blocked_by_slug.items()]
                       if _agentic(rows)}
    fetched_slugs = set(by_slug) | set(blocked_by_slug)

    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_searched": len(models),
        "models_with_emitted_rows": len(by_slug),
        "models_with_emitted_agentic_rows": len(emitted_agentic),
        "models_with_fetched_independent_data": len(fetched_slugs),
        "models_with_fetched_independent_agentic_data": len(fetched_agentic),
        "models_with_blocked_rows": len(blocked_by_slug),
        "rows_emitted": sum(len(v) for v in by_slug.values()),
        "rows_blocked_pending_source": sum(len(v) for v in blocked_by_slug.values()),
        "proposed_sources": PROPOSED_SOURCES,
        "source_record_templates": SOURCE_RECORD_TEMPLATES,
        "schema_note": "schemas/source.schema.json has no 'leaderboard' kind; templates use kind='repo' so they validate today. Extending the enum is a parent/D2 decision.",
        "models": [{"slug": slug, "benchmarks": rows} for slug, rows in sorted(by_slug.items())],
        "blocked": [{"slug": slug, "rows": rows} for slug, rows in sorted(blocked_by_slug.items())],
        "source_stats": stats,
        "notes": [
            "Rows are emitted only when the number, harness/config and fetched URL are all present.",
            "swebench.com rows require checked=true; vendor submissions (checked=false) are rejected.",
            "Open Agent Leaderboard uses a fixed neutral scaffold (tool_calling), never max-across-agents.",
            "Terminal-Bench 4.1 rows are official-leaderboard submissions reviewed by maintainers; "
            "the 2.1/2.0 tabs are client-rendered and were not machine-readable.",
            "GAIA rows are matched but cannot be written: the project's canonicalizer excludes GAIA.",
            "HF SWE-bench Verified leaderboard rows are model-card self-reports (verified=false) and are excluded.",
            "Rows in 'blocked' name proposed source ids that do not yet exist in data/sources/; "
            "D2 must add those records, then re-run this tool to promote them.",
        ],
    }


def check_payload(path: Path) -> int:
    """Validate a payload with the real writer's checker; print accepted/blocked."""
    from apply_third_party import check_block, load_sources, source_url_pattern
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = load_sources()
    url_re = source_url_pattern()
    accepted, blocked = 0, 0
    for block in data.get("models", []):
        clean, fatal, problems = check_block(block, sources, url_re)
        if clean and clean["benchmarks"]:
            accepted += 1
            print(f"  ACCEPT {block['slug']}: {len(clean['benchmarks'])} row(s)")
        else:
            blocked += 1
            print(f"  BLOCK  {block.get('slug')}: {fatal or problems}")
    # Prove the blocked rows are well-formed: validate them against the
    # schema-valid source templates in memory. Nothing is written to data/.
    simulated = dict(sources)
    for sid, rec in SOURCE_RECORD_TEMPLATES.items():
        simulated.setdefault(sid, rec)
    ok = bad = 0
    for entry in data.get("blocked", []):
        for row in entry["rows"]:
            clean, fatal, problems = check_block(
                {"slug": entry["slug"], "benchmarks": [row["benchmark"]]},
                simulated, url_re)
            if clean and clean["benchmarks"]:
                ok += 1
            else:
                bad += 1
                print(f"  MALFORMED {entry['slug']} {row['benchmark']['name']}: {fatal or problems}")
    print(f"check: {accepted} accepted block(s), {blocked} blocked block(s); "
          f"{ok} blocked row(s) valid once source records exist, {bad} malformed")
    return 0


def self_test() -> int:
    """Parser/identity checks that need no network."""
    assert compact("Kimi-K2 Instruct") == compact("Kimi K2 Instruct")
    assert matches("Kimi K2 Instruct", "kimi-k2", "swebench")
    assert matches("DeepSeek V3.2", "deepseek-v3-2", "swebench")
    assert matches("Beam 1.0", "beam-1-0-501b-a23b", "tbench")
    assert not matches("Kimi K3.1 Pro", "kimi-k3", "tbench")
    assert not matches("DeepSeek V4 Pro 0813", "deepseek-v4-pro", "tbench")
    assert canonical_benchmark("Terminal-Bench 4.1") == "Terminal-Bench 4.1"
    assert canonical_benchmark("tau2-bench Retail") == "tau2-bench Retail"
    assert canonical_benchmark("BrowseComp Plus") == "BrowseComp-Plus"
    assert canonical_benchmark("GAIA") is None
    assert canonical_benchmark("SWE-bench Verified") == "SWE-bench Verified"
    print("self-test: ok")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, help="file with one model slug per line")
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "thirdparty" / "part-a.json")
    ap.add_argument("--check", type=Path, help="validate an existing payload and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if args.check:
        raise SystemExit(check_payload(args.check))
    if not args.models:
        ap.error("--models is required unless --check or --self-test")

    models = [line.strip() for line in args.models.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    payload = build(models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"models searched: {payload['models_searched']}")
    print(f"models with fetched independent data: {payload['models_with_fetched_independent_data']}")
    print(f"models with fetched independent agentic data: {payload['models_with_fetched_independent_agentic_data']}")
    print(f"models with emitted rows (sources resolved): {payload['models_with_emitted_rows']}")
    print(f"rows emitted: {payload['rows_emitted']}; blocked pending source: {payload['rows_blocked_pending_source']}")
    for key, st in payload["source_stats"].items():
        print(f"  {key}: {json.dumps(st)[:400]}")


if __name__ == "__main__":
    main()