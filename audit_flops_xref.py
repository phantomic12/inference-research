#!/usr/bin/env python3
"""Wave 10 cross-reference audit over data/flops/ (and optionally data/gotchas/).

Three checks, from the wave-8 reverify findings (commit 5848379):

  1. dangling `affected_by_hardware` values -- resolved against the accelerator
     record set, exactly as tools/validate.py does via registry.REF_FIELDS.
     Architecture/RLT target names (gfx942, gfx950, gfx1151) are budgeted debt:
     they name real silicon but no accelerator record exists to point at.
  2. undeclared `source_id`s -- a source id cited anywhere inside a record that
     is not declared in that record's own `sources[]`. validate.py checks that
     sources[] entries RESOLVE, never that a cited id is DECLARED, which is why
     the two accelerator cases of wave-8 defect D-B passed 121/121 tests.
  3. literal `%s` printf placeholders -- the wave-3 corruption (commit 9968769)
     that the wave-8 pass repaired on 34 accelerator records.

Reports only; the caller decides what to edit.

    python audit_flops_xref.py            # flops only
    python audit_flops_xref.py --gotchas  # flops + gotchas
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT / "tools"))
from registry import DIRS  # noqa: E402

# Architecture / RTL target names that name real silicon but have no accelerator
# record in this repo. Budgeted debt, not typos: no record can be invented to
# point at, and the value still tells a reader which hardware family the row
# applies to. Mirrors the wave-7 ledger's D6 enumeration of the 30.
BUDGETED_TARGETS = {"gfx942", "gfx950", "gfx1151"}


def load(directory: str) -> dict[str, dict]:
    out = {}
    for path in sorted((DATA / directory).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out[rec["id"]] = rec
    return out


def strings(obj, path=""):
    """Yield (path, string) for every string anywhere in the record."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from strings(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from strings(value, f"{path}[{index}]")
    elif isinstance(obj, str):
        yield path, obj


def check_affected_by_hardware(records, accelerators, label):
    """Resolve every affected_by_hardware value against the accelerator set."""
    dangling_rows, budgeted_rows = [], []
    total = 0
    for rid in sorted(records):
        for target in records[rid].get("affected_by_hardware") or []:
            total += 1
            if target in accelerators:
                continue
            row = (rid, target)
            (budgeted_rows if target in BUDGETED_TARGETS else dangling_rows).append(row)
    print(f"CHECK 1 [{label}] - affected_by_hardware: "
          f"{len(dangling_rows) + len(budgeted_rows)} unresolved of {total} values")
    print(f"  budgeted architecture targets (no record to point at): {len(budgeted_rows)}")
    for rid, target in budgeted_rows:
        print(f"    {rid} -> {target}")
    print(f"  DANGLING (typo or renamed record - fix): {len(dangling_rows)}")
    for rid, target in dangling_rows:
        print(f"    {rid} -> {target}")
    print()
    return dangling_rows


def check_undeclared_sources(records, sources, label):
    """A source id cited anywhere in the record but absent from sources[].

    Matched as a BOUNDED token, not a raw substring: `arch-deepseek-v3` is
    contained in the declared id `arch-deepseek-v32-dsa`, and a naive substring
    match reports it undeclared. The alternation is sorted longest-first so the
    longest id at a position wins, and the lookarounds reject any match abutting
    a slug character.
    """
    pattern = re.compile(
        r"(?<![A-Za-z0-9_-])("
        + "|".join(sorted((re.escape(sid) for sid in sources), key=len, reverse=True))
        + r")(?![A-Za-z0-9_-])"
    )
    undeclared = []
    for rid in sorted(records):
        declared = set(records[rid].get("sources") or [])
        for path, text in strings(records[rid]):
            for sid in sorted(set(pattern.findall(text))):
                if sid not in declared:
                    undeclared.append((rid, path, sid))
    print(f"CHECK 2 [{label}] - undeclared source ids: {len(undeclared)}")
    for rid, path, token in undeclared:
        print(f"    {rid} {path} -> {token} (missing from sources[])")
    print()
    return undeclared


def check_placeholders(records, label):
    placeholders = []
    for rid in sorted(records):
        for path, text in strings(records[rid]):
            if "%s" in text:
                placeholders.append((rid, path, text))
    print(f"CHECK 3 [{label}] - literal %s placeholders: {len(placeholders)}")
    for rid, path, text in placeholders:
        print(f"    {rid} {path}: {text[:200]}")
    print()
    return placeholders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gotchas", action="store_true",
                    help="also audit data/gotchas/, which this wave owns")
    args = ap.parse_args()

    flops = load("flops")
    accelerators = load(DIRS["accelerator"])
    sources = load(DIRS["source"])

    print(f"== flops records audited: {len(flops)} "
          f"(accelerator ids: {len(accelerators)}, source ids: {len(sources)})\n")

    check_affected_by_hardware(flops, accelerators, "flops")
    undeclared_flops = check_undeclared_sources(flops, sources, "flops")
    ph_flops = check_placeholders(flops, "flops")

    print("TOTALS [flops]: "
          f"{len(undeclared_flops)} undeclared source id(s), "
          f"{len(ph_flops)} literal %s placeholder(s)")

    if args.gotchas:
        print("\n" + "=" * 62 + "\n")
        gotchas = load("gotchas")
        print(f"== gotcha records audited: {len(gotchas)}\n")
        check_undeclared_sources(gotchas, sources, "gotchas")
        check_placeholders(gotchas, "gotchas")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
