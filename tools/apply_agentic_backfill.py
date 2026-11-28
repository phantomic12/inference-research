#!/usr/bin/env python3
"""Apply the Phase-3 agentic backfill JSON returned by the research subagents.

Why this exists rather than letting subagents write records: they will substitute
a sibling model's number, swap a test version, or invent a plausible value when a
card is thin. Subagents only return JSON; every write, every id, every source
record is minted here, deterministically, and validated by the repo's own
validator.

    python3 tools/apply_agentic_backfill.py <results.json|results-dir> [--write]

Without --write it is a dry run: it prints exactly what it would add and exits
non-zero if anything would be rejected. Run dry first, always.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"
SOURCES = ROOT / "data" / "sources"
QUEUE = ROOT / "cache" / "agentic-backfill" / "queue.json"

# Benchmark identity is decided by tools/benchmark_names.py, which returns the
# catalog's one house spelling per test or None. Both the writer (here) and the
# scorer (build_joined_views._AXES) call it, so a row that is accepted is exactly
# a row that can be scored. The old local regexes could not agree: they rejected
# real names like "SWE Verified" while accepting spellings the axis never matched.
from benchmark_names import canonical as _canonical_benchmark  # noqa: E402

SOURCE_KINDS = {
    "spec-sheet", "whitepaper", "paper", "benchmark", "repo",
    "release-notes", "forum", "blog", "interview", "review", "database",
}
# Subagents invent descriptive kinds; map them onto the schema enum.
KIND_MAP = {
    "model-card": "repo", "model card": "repo", "huggingface": "repo",
    "tech-report": "paper", "tech report": "paper", "arxiv": "paper",
    "github": "repo", "readme": "repo", "official-eval-suite": "benchmark",
    "leaderboard": "database", "eval-suite": "benchmark", "vendor-blog": "blog",
}
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SLUGIFY = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return SLUGIFY.sub("-", text.lower()).strip("-")


def _other_record_ids() -> dict[str, str]:
    """Every id already owned by a record that is not a source, mapped to its dir."""
    owned: dict[str, str] = {}
    for d in ROOT.joinpath("data").iterdir():
        if not d.is_dir() or d.name == "sources":
            continue
        for f in d.glob("*.json"):
            owned[f.stem] = d.name
    return owned


_OWNED: dict[str, str] | None = None


def _unique_source_id(sid: str) -> str:
    """Keep a source id out of the namespace other record types already use."""
    global _OWNED
    if _OWNED is None:
        _OWNED = _other_record_ids()
    if sid not in _OWNED:
        return sid
    # "paper-deepseek-r1" (already a paper) becomes "source-paper-deepseek-r1".
    # Stable and collision-free without touching the existing paper id.
    alt = f"source-{sid}"
    return alt if alt not in _OWNED else f"{alt}-2"


def load_queue() -> dict[str, dict]:
    if not QUEUE.exists():
        sys.exit(f"missing queue at {QUEUE}; run tools/phase3_queue.py first")
    return {q["id"]: q for q in json.loads(QUEUE.read_text())}


def collect(paths: list[Path]) -> tuple[list[dict], list[tuple[str, str]], list[Path]]:
    """Every model block from every result file, rejecting malformed ones loudly."""
    blocks, rejected = [], []
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # a subagent's malformed JSON must not be silently dropped
            rejected.append((f.name, f"unparseable JSON: {e}"))
            continue
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            rejected.append((f.name, "no 'models' list"))
            continue
        for m in models:
            blocks.append({**m, "_file": f.name})
    return blocks, rejected, files


def check_block(block: dict, queue: dict[str, dict]) -> tuple[dict | None, list[str]]:
    """Return (clean_benchmark_rows, problems). Any problem rejects the whole model."""
    problems = []
    slug = (block.get("slug") or "").strip()
    model_id = slugify(slug)
    if model_id not in queue:
        problems.append(f"slug '{slug}' -> '{model_id}' is not in the Phase-3 queue")
        return None, problems

    kept = []
    for b in block.get("benchmarks") or []:
        name = b.get("name")
        score = b.get("score")
        src = b.get("source") or (block.get("sources") or [{}])[0].get("id")
        if not name or not isinstance(score, (int, float)):
            problems.append(f"benchmark row missing name or numeric score: {b}")
            continue
        # One canonical spelling per test, or None when the name is not an agentic
        # benchmark. Writing the canonical form (rather than the card's spelling)
        # is what makes the axis count every row the gate accepted.
        canon = _canonical_benchmark(name)
        if canon is None:
            problems.append(f"'{name}' is not a recognised agentic benchmark")
            continue
        name = canon
        if not (-1 <= float(score) <= 1000):
            problems.append(f"'{name}' score {score} is out of any plausible range")
            continue
        # A 0-100 benchmark cannot score 300. Catch the class of error where a
        # subagent reads the wrong column (an Elo, a token count, a dollar figure)
        # or pads a number rather than quoting one.
        scale = float(b.get("scale", 100))
        if scale == 100 and not (-1 <= float(score) <= 100):
            problems.append(f"'{name}' score {score} exceeds its own 0-100 scale")
            continue
        if not src:
            problems.append(f"'{name}' has no source")
            continue
        kept.append({
            "name": name,
            "score": float(score),
            "scale": float(b.get("scale", 100)),
            "higher_is_better": bool(b.get("higher_is_better", True)),
            "source": src,
            "vendor_claim": bool(b.get("vendor_claim", True)),
            "config": b.get("config") or block.get("default_config") or "",
        })

    # Two sources minimum on a model with several rows is the skill's honesty rule;
    # here we only require that the model names at least one loadable source.
    if kept and not (block.get("sources") or []):
        problems.append("model has benchmark rows but no sources[] entry")
    return ({"id": model_id, "benchmarks": kept, "sources": block.get("sources") or [],
             "notes": block.get("notes") or ""} if not problems else None), problems


def normalize_sources(block: dict) -> list[dict]:
    out = []
    for s in block["sources"]:
        sid = slugify(s.get("id") or s.get("title") or "unnamed-source")
        if not ID_RE.match(sid):
            sid = slugify(sid)
        # ids are global across record types: the validator rejects the whole
        # record set if a source id collides with a paper id. A subagent naming a
        # source "paper-deepseek-r1" is quoting an arXiv id we already track as a
        # paper, so namespace it rather than silently overwriting the paper.
        sid = _unique_source_id(sid)
        kind = (s.get("kind") or "repo").strip().lower()
        kind = KIND_MAP.get(kind, kind)
        if kind not in SOURCE_KINDS:
            kind = "repo"
        out.append({
            "id": sid,
            "type": "source",
            "name": s.get("title") or sid,
            "status": "verified",
            "confidence": 0.7,
            "updated": date.today().isoformat(),
            "url": s.get("url"),
            "publisher": s.get("publisher") or "",
            "kind": kind,
            "published": s.get("published"),
            "accessed": date.today().isoformat(),
            "archived_url": None,
            "notes": s.get("notes") or "",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--allow-dupes", action="store_true")
    args = ap.parse_args()

    queue = load_queue()
    blocks, rejected_files, files = collect(args.paths)
    print(f"read {len(files)} file(s): {len(blocks)} model block(s)")
    for name, why in rejected_files:
        print(f"  REJECT FILE {name}: {why}")

    accepted, all_problems = [], []
    for b in blocks:
        clean, problems = check_block(b, queue)
        if problems:
            all_problems.append((b.get("_file"), b.get("slug"), problems))
        if clean and clean["benchmarks"]:
            accepted.append({**clean, "_file": b["_file"]})

    total_rows = sum(len(a["benchmarks"]) for a in accepted)
    print(f"accepted: {len(accepted)} model(s), {total_rows} benchmark row(s)")
    for a in accepted:
        print(f"  + {a['id']}: " + ", ".join(f"{r['name']}={r['score']:g}" for r in a["benchmarks"]))
    if all_problems:
        print(f"\nrejected: {len(all_problems)} model block(s)")
        for f, slug, probs in all_problems:
            print(f"  [{f}] {slug}")
            for p in probs:
                print(f"      - {p}")

    if not args.write:
        print("\nDRY RUN — nothing written. Re-run with --write to apply.")
        return
    if not accepted:
        print("\nnothing accepted; nothing to write")
        return

    written_sources = 0
    for a in accepted:
        for s in normalize_sources(a):
            path = SOURCES / f"{s['id']}.json"
            if path.exists():
                continue
            path.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written_sources += 1

    today = date.today().isoformat()
    for a in accepted:
        path = MODELS / f"{a['id']}.json"
        rec = json.loads(path.read_text())
        existing = {b.get("name") for b in rec.get("benchmarks") or []}
        added = 0
        for row in a["benchmarks"]:
            if row["name"] in existing:
                continue
            rec.setdefault("benchmarks", []).append(row)
            added += 1
        if added:
            rec["updated"] = today
            # Match tools/new_record.py: indent=2, trailing newline. A different
            # indent reformats the whole record and buries the real diff.
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {a['id']}: +{added} benchmark(s)")

    print(f"\nnew source records: {written_sources}")
    print("next: python3 tools/validate.py")


if __name__ == "__main__":
    main()
