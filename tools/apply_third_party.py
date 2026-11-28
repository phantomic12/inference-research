#!/usr/bin/env python3
"""Apply INDEPENDENT (third-party) agentic benchmark rows to existing models.

Mirror image of tools/apply_agentic_backfill.py. That writer is the deterministic
path for vendor claims and defaults every row to ``vendor_claim: true``. This one
will only write evidence that is explicitly marked as not a vendor claim, and it
rejects a row unless its benchmark name, source and scale all survive machine
checks. Research agents return JSON; every write happens here, deterministically.

    python3 tools/apply_third_party.py <results.json|results-dir> [--write]

Dry run is the default: it prints what would be added AND which vendor rows would
be superseded, and writes nothing. --write is the only path that touches
data/models/. It never mints source records: a third-party number has to name a
source that is already in the catalog.

    python3 tools/apply_third_party.py results.json          # safe preview
    python3 tools/apply_third_party.py results.json --write  # apply
    python3 tools/validate.py                                # then validate
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
SOURCE_SCHEMA = ROOT / "schemas" / "source.schema.json"

# One canonicalizer for the whole project. This file must NOT grow its own
# benchmark table: a name that canonical() resolves here is exactly a name that
# build_joined_views.py scores, so "accepted by the writer" == "counted on the
# site". A second table is what previously let rows pass the gate and score zero.
from benchmark_names import canonical as _canonical_benchmark  # noqa: E402

# Independent evidence may only cite a primary artifact or an independent result
# list. A vendor blog, a forum post or an aggregator database is not independent
# evidence merely because someone set vendor_claim: false on the row.
INDEPENDENT_KINDS = {"paper", "repo", "spec-sheet", "leaderboard"}

# Every name canonical() resolves is an agentic *rate*: resolved/accuracy/pass@1.
# Those are published either as a fraction (scale 1) or a percentage (scale 100).
# Nothing else is plausible, and letting a payload declare scale=1000 would let
# an Elo, a token count or a padded number through the "within scale" check.
# This is a scale rule, not a second benchmark name table.
ALLOWED_SCALES = (1.0, 100.0)

SLUGIFY = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return SLUGIFY.sub("-", (text or "").lower()).strip("-")


def collect(paths: list[Path]) -> tuple[list[dict], list[tuple[str, str]], list[Path]]:
    """Every model block from every result file, rejecting malformed ones loudly."""
    blocks, rejected = [], []
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # malformed JSON must not be silently dropped
            rejected.append((f.name, f"unparseable JSON: {e}"))
            continue
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            rejected.append((f.name, "no 'models' list"))
            continue
        for m in models:
            if isinstance(m, dict):
                blocks.append({**m, "_file": f.name})
            else:
                rejected.append((f.name, f"model block is not an object: {m!r}"))
    return blocks, rejected, files


def load_sources(directory: Path = SOURCES) -> dict[str, dict]:
    """Source records by file stem; a missing file is a missing source id."""
    out: dict[str, dict] = {}
    for f in directory.glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue  # a broken source fails the schema gate anyway
        if isinstance(rec, dict):
            out[f.stem] = rec
    return out


def source_url_pattern() -> re.Pattern:
    """Read the URL rule from the source schema instead of copying its regex."""
    schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    return re.compile(schema["properties"]["url"]["pattern"])


def source_problem(src: dict, url_re: re.Pattern) -> str | None:
    """Why this source cannot back an independent row, or None if it can."""
    kind = src.get("kind")
    if kind not in INDEPENDENT_KINDS:
        return f"source kind {kind!r} is not independent evidence " \
               f"(need one of {', '.join(sorted(INDEPENDENT_KINDS))})"
    url = src.get("url")
    if not isinstance(url, str) or url_re.match(url) is None:
        return f"source URL {url!r} fails schemas/source.schema.json"
    return None


def check_block(
    block: dict,
    sources: dict[str, dict],
    url_re: re.Pattern | None = None,
    models_dir: Path = MODELS,
) -> tuple[dict | None, list[str], list[str]]:
    """Return (clean_block, fatal_problems, row_problems).

    A fatal problem (provenance, unknown model, no surviving rows) drops the
    whole block. A bad benchmark row drops only that row: good evidence from the
    same research payload must not be lost because one row was mis-transcribed.
    """
    url_re = url_re or source_url_pattern()
    fatal: list[str] = []
    rows: list[str] = []

    # A vendor_claim: true payload is the other writer's job, not ours. Refuse it
    # even if every individual row says false: a contradictory block is exactly
    # the provenance ambiguity this guard exists to remove.
    if block.get("vendor_claim") is True:
        fatal.append("block declares vendor_claim: true; use apply_agentic_backfill.py")

    slug = (block.get("slug") or "").strip()
    model_id = slugify(slug)
    if not model_id or not (models_dir / f"{model_id}.json").exists():
        fatal.append(f"slug {slug!r} -> {model_id!r} has no model record in the catalog")

    default_config = block.get("default_config")
    seen: set[str] = set()
    kept = []
    for i, b in enumerate(block.get("benchmarks") or []):
        if not isinstance(b, dict):
            rows.append(f"row {i}: not an object")
            continue
        raw_name = b.get("name")
        name = _canonical_benchmark(raw_name) if isinstance(raw_name, str) else None
        if name is None:
            rows.append(f"row {i}: {raw_name!r} is not a recognised agentic benchmark")
            continue

        # Never default to a vendor claim. Both an explicit row flag and the
        # model-wide flag count; silence counts as a rejection.
        declared = b.get("vendor_claim", block.get("vendor_claim"))
        if declared is not False:
            rows.append(f"row {i} ({name}): vendor_claim must be explicitly false, got {declared!r}")
            continue

        config = b.get("config") if b.get("config") is not None else default_config
        if not isinstance(config, str) or not config.strip():
            rows.append(f"row {i} ({name}): missing non-empty config string")
            continue

        sid = b.get("source")
        if not isinstance(sid, str) or sid not in sources:
            rows.append(f"row {i} ({name}): source {sid!r} does not exist in data/sources/")
            continue
        bad_source = source_problem(sources[sid], url_re)
        if bad_source:
            rows.append(f"row {i} ({name}): {bad_source}")
            continue

        score, scale = b.get("score"), b.get("scale", 100)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            rows.append(f"row {i} ({name}): score {score!r} is not numeric")
            continue
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) \
                or float(scale) not in ALLOWED_SCALES:
            rows.append(f"row {i} ({name}): scale {scale!r} is not a rate scale (1 or 100)")
            continue
        if not (0 <= float(score) <= float(scale)):
            rows.append(f"row {i} ({name}): score {score:g} is outside 0..{float(scale):g}")
            continue

        if name in seen:
            rows.append(f"row {i} ({name}): duplicate canonical benchmark in the payload")
            continue
        seen.add(name)
        kept.append({
            "name": name,
            "score": float(score),
            "scale": float(scale),
            "higher_is_better": bool(b.get("higher_is_better", True)),
            "source": sid,
            "vendor_claim": False,
            "config": config,
        })

    if not fatal and not kept:
        fatal.append("no valid benchmark rows survived validation")
    clean = {"id": model_id, "benchmarks": kept} if not fatal else None
    return clean, fatal, rows


def _canon_key(row: dict) -> str:
    name = row.get("name") or ""
    return _canonical_benchmark(name) or name


def _num(value) -> str:
    """Format a recorded score for a provenance note without crashing on junk."""
    return f"{float(value):g}" if isinstance(value, (int, float)) \
        and not isinstance(value, bool) else str(value)


def _superseded_note(old_rows: list[dict]) -> str:
    """Keep the displaced numbers in the new row, not in a second scored row."""
    parts = []
    for old in old_rows:
        label = "vendor claim" if old.get("vendor_claim", True) else "older independent row"
        parts.append(
            f"{label} score={_num(old.get('score'))}/{_num(old.get('scale', 100))} "
            f"source={old.get('source')} config={old.get('config') or 'none'}"
        )
    return "[third-party evidence supersedes " + "; ".join(parts) + "]"


def merge_benchmarks(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], list[str]]:
    """Dedupe by canonical name; the independent row replaces every collision.

    WHY REPLACE AND NOT KEEP BOTH: build_joined_views.build_worth_running() loops
    over every benchmark row on the model and averages them into the axis
    (``by_axis.setdefault(ax, []).append((rel, w))``). Keeping the vendor row
    beside the independent one would average the vendor number straight back in,
    so the independent measurement would not win ranking. Replacing it with one
    row carrying ``vendor_claim: false`` makes the independent number the only
    scored row for that benchmark. Nothing is lost: the displaced scores, scales,
    source ids and configs are preserved verbatim in the surviving row's
    ``config`` string, which is a schema-valid free-text field and shows up in
    the record diff for a human to audit.
    """
    incoming_by_key = {_canon_key(row): row for row in incoming}
    displaced: dict[str, list[dict]] = {}
    for row in existing:
        if isinstance(row, dict):
            key = _canon_key(row)
            if key in incoming_by_key:
                displaced.setdefault(key, []).append(row)

    merged, emitted, notes = [], set(), []
    for row in existing:
        if not isinstance(row, dict):
            merged.append(row)
            continue
        key = _canon_key(row)
        if key in incoming_by_key:
            if key in emitted:
                continue  # a second spelling of the same test: folded into the note
            emitted.add(key)
            new = dict(incoming_by_key[key])
            new["config"] = f"{new['config']} {_superseded_note(displaced[key])}"
            notes.append(
                f"replaced {key}: " + ", ".join(
                    _num(old.get("score")) for old in displaced[key]
                ) + f" -> {_num(new['score'])}"
            )
            merged.append(new)
        else:
            merged.append(row)
    for key, row in incoming_by_key.items():
        if key not in emitted:
            merged.append(dict(row))
            notes.append(f"added {key}={_num(row['score'])}")
    return merged, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    sources = load_sources()
    url_re = source_url_pattern()
    blocks, rejected_files, files = collect(args.paths)
    print(f"read {len(files)} file(s): {len(blocks)} model block(s)")
    for name, why in rejected_files:
        print(f"  REJECT FILE {name}: {why}")

    accepted: list[dict] = []
    fatal_blocks: list[tuple[str, str, list[str]]] = []
    row_rejects: list[tuple[str, str, str]] = []
    for b in blocks:
        clean, fatal, row_problems = check_block(b, sources, url_re)
        if fatal:
            fatal_blocks.append((b.get("_file", "?"), str(b.get("slug")), fatal))
        for p in row_problems:
            row_rejects.append((b.get("_file", "?"), str(b.get("slug")), p))
        if clean and clean["benchmarks"]:
            accepted.append({**clean, "_file": b["_file"]})

    added = sum(len(a["benchmarks"]) for a in accepted)
    print(f"accepted: {len(accepted)} model(s), {added} benchmark row(s)")
    for a in accepted:
        print(f"  + {a['id']}: " + ", ".join(
            f"{r['name']}={r['score']:g}" for r in a["benchmarks"]))

    if fatal_blocks:
        print(f"\nrejected blocks: {len(fatal_blocks)}")
        for f, slug, probs in fatal_blocks:
            print(f"  [{f}] {slug}")
            for p in probs:
                print(f"      - {p}")
    if row_rejects:
        print(f"\nrejected rows: {len(row_rejects)}")
        for f, slug, p in row_rejects:
            print(f"  [{f}] {slug}: {p}")
    print(f"summary: +{len(accepted)} model(s)/{added} row(s); "
          f"-{len(fatal_blocks)} block(s)/{len(row_rejects)} row(s)")

    if not args.write:
        # Preview the dedupe too: a dry run that only says "accepted" hides the
        # most consequential effect of the write -- which vendor row it displaces.
        for a in accepted:
            path = MODELS / f"{a['id']}.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            _, notes = merge_benchmarks(rec.get("benchmarks") or [], a["benchmarks"])
            for note in notes:
                print(f"  would {a['id']}: {note}")
        print("\nDRY RUN — nothing written. Re-run with --write to apply.")
        return
    if not accepted:
        print("\nnothing accepted; nothing to write")
        return

    today = date.today().isoformat()
    for a in accepted:
        path = MODELS / f"{a['id']}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        before = rec.get("benchmarks") or []
        merged, notes = merge_benchmarks(before, a["benchmarks"])
        if merged != before:
            rec["benchmarks"] = merged
            rec["updated"] = today
            # Match tools/new_record.py and apply_agentic_backfill.py: indent=2
            # and a trailing newline, or the real diff is buried in reformatting.
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        for note in notes:
            print(f"  {a['id']}: {note}")
    print(f"\nnext: python3 tools/validate.py")


if __name__ == "__main__":
    main()
