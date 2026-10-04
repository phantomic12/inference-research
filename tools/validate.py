#!/usr/bin/env python3
"""Validate every record in data/ against its schema and cross-check references.

    python tools/validate.py                # everything
    python tools/validate.py --quiet        # errors only
    python tools/validate.py --type engine  # one type

Cross-references are resolved against the set of record ids, using the one
field table in registry.REF_FIELDS. A reference that does not resolve is a
dangling reference: reported individually, counted, and fatal only once the
count exceeds the budget in tools/ref_budget.json. See DANGLING_BUDGET_NOTE.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, HARD_REF_FIELDS, NULLABLE_REF_FIELDS, REF_FIELDS, TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
BUDGET_FILE = ROOT / "tools" / "ref_budget.json"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# ---------------------------------------------------------------------------
# DANGLING REFERENCE BASELINE
#
# data/ carried 110 dangling cross-references when this check was added
# (measured by tools/build_site_data.py --check). Most are
# `affected_by_hardware` / `hardware_relevance` slugs like `nvidia-a100-sxm`
# and `mi300x` that name real parts under ids this repo uses differently
# (`nvidia-a100-80gb-sxm4`, `amd-instinct-mi300x`) — legitimate debt, but debt.
#
# Fixing them means editing data/, which this validator must never do, and
# quietly deleting them would be worse than reporting them. So the budget is
# recorded in tools/ref_budget.json and the check FAILS once the count exceeds
# it: every dangling reference stays visible in the output, any NEW one fails
# CI immediately, and the debt can only shrink.
#
# The budget is a ratchet and MUST ratchet down. Keeping it at the CURRENT
# count is the whole point of a ceiling: the moment the debt starts shrinking,
# a stale ceiling silently stops catching anything, so every change that reduces
# the count must lower this number in the same commit. The figure printed by
# `python tools/validate.py` is the one to copy. RAISING it to make CI green
# defeats the check entirely.
# ---------------------------------------------------------------------------


def load_budget() -> int:
    """The recorded ceiling on dangling references.

    A missing or malformed budget file is fatal: defaulting to 0 would fail
    CI on the 110 known-good-as-documented, and defaulting to infinity would
    silently disable the check. Neither is safe.
    """
    try:
        budget = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        value = budget["max_dangling_refs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"cannot read dangling-reference budget from {BUDGET_FILE}: {exc}. "
            f"See DANGLING_BUDGET_NOTE in tools/validate.py."
        ) from exc
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"{BUDGET_FILE}: max_dangling_refs must be a non-negative integer")
    return value


def ref_values(value) -> list[str]:
    """The string references held by one field value.

    A reference field is either a list of bare slugs or a single bare slug
    (possibly null). Non-strings are ignored here; the schema layer already
    reports type errors, and a reference count should not depend on them.
    """
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str) and x]
    if isinstance(value, str) and value:
        return [value]
    return []


def _budget_error(count: int, budget: int) -> str:
    return (
        f"dangling references: {count}, budget of {budget}. "
        f"Fix a reference, or lower max_dangling_refs in tools/ref_budget.json "
        f"once the debt is genuinely repaid; never raise it to go green."
    )


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        # Unresolved cross-references, one entry each. Kept as data, not just as
        # warnings, because the budget check needs the COUNT.
        self.dangling: list[str] = []

    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            self.errors.append(msg)

    def warn(self, cond: bool, msg: str) -> None:
        if not cond:
            self.warnings.append(msg)


def check_budget(rep: Report, budget: int) -> None:
    """Fail the run when dangling references exceed `budget`.

    Every individual dangling reference is already in rep.warnings; this adds a
    single fatal line so a regression is loud rather than another line item in
    a long list nobody diffs.
    """
    if len(rep.dangling) > budget:
        rep.errors.append(_budget_error(len(rep.dangling), budget))


def validate_value(value, schema: dict, path: str, rep: Report) -> None:
    """JSON Schema subset: const, enum, type, pattern, required,
    additionalProperties:false, items, properties."""
    if "const" in schema and value != schema["const"]:
        rep.errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
        return
    if "enum" in schema and value not in schema["enum"]:
        rep.errors.append(f"{path}: {value!r} not in allowed enum {schema['enum']}")

    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(
            (tt == "object" and isinstance(value, dict))
            or (tt == "array" and isinstance(value, list))
            or (tt == "string" and isinstance(value, str))
            or (tt == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (tt == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (tt == "boolean" and isinstance(value, bool))
            or (tt == "null" and value is None)
            for tt in types
        )
        if not ok:
            rep.errors.append(f"{path}: expected type {t}, got {type(value).__name__}")
            return

    if isinstance(value, str) and "pattern" in schema and not re.match(schema["pattern"], value):
        rep.errors.append(f"{path}: {value!r} does not match {schema['pattern']}")

    if isinstance(value, dict):
        for req in schema.get("required", []):
            rep.check(req in value, f"{path}: missing required field {req!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in value:
                rep.check(k in props, f"{path}: unknown field {k!r}")
        for k, sub in props.items():
            if k in value:
                validate_value(value[k], sub, f"{path}.{k}", rep)

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate_value(item, schema["items"], f"{path}[{i}]", rep)


def load_schemas(types: list[str]) -> dict[str, dict]:
    out = {}
    for t in types:
        p = SCHEMAS / f"{t}.schema.json"
        if not p.exists():
            raise SystemExit(f"missing schema {p}")
        out[t] = json.loads(p.read_text(encoding="utf-8"))
    return out


def run(types: list[str], quiet: bool, budget: int | None = None) -> tuple[Report, int]:
    """Validate every record of `types`.

    `budget` defaults to the recorded baseline in tools/ref_budget.json. It is a
    parameter only so a caller can raise the ceiling for a deliberate,
    separately-reviewed check; CI passes nothing and gets the recorded value.
    """
    rep = Report()
    schemas = load_schemas(types)
    records: dict[str, tuple[str, dict]] = {}

    for t in types:
        d = ROOT / "data" / DIRS[t]
        if not d.exists():
            rep.check(False, f"data/{DIRS[t]}: directory missing")
            continue
        for f in sorted(d.glob("*.json")):
            rid = f"{DIRS[t]}/{f.stem}"
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                rep.errors.append(f"{rid}: invalid JSON: {e}")
                continue
            if not isinstance(rec, dict):
                rep.errors.append(f"{rid}: top level must be an object")
                continue
            validate_value(rec, schemas[t], rid, rep)
            rep.check(rec.get("id") == f.stem,
                      f"{rid}: id {rec.get('id')!r} != filename {f.stem!r}")
            rep.check(rec.get("type") == t, f"{rid}: type field must be {t!r}")
            rep.check(bool(SLUG_RE.match(f.stem)), f"{rid}: filename is not a valid slug")
            for dfield in ("updated", "accessed"):
                v = rec.get(dfield)
                if isinstance(v, str):
                    rep.check(bool(DATE_RE.match(v)),
                              f"{rid}.{dfield}: must be YYYY-MM-DD, got {v!r}")
            c = rec.get("confidence")
            if c is not None:
                rep.warn(isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0,
                         f"{rid}: confidence {c} outside 0.0-1.0")
            records[rid] = (t, rec)

    # Records are addressed two ways: fully qualified "dir/slug" and bare "slug".
    # The `sources` field and id-reference fields hold BARE slugs, so resolution
    # must compare against the bare id set, not the qualified set.
    #
    # An AMBIGUOUS bare slug (same stem in two directories) is removed from the
    # resolution set rather than resolved by guess: a reference to it is already
    # an error in its own right, and guessing would let the two rules disagree.
    qualified = set(records)
    bare_all = {rid.split("/", 1)[1] for rid in records}
    bare = set(bare_all)
    ambiguous = {
        b for b in bare_all
        if sum(1 for rid in records if rid.split("/", 1)[1] == b) > 1
    }
    bare -= ambiguous
    for d in sorted(ambiguous):
        rep.errors.append(f"ambiguous id {d!r}: appears in more than one record type")
        rep.errors.append(
            f"ambiguous id {d!r} cannot resolve any reference; make the two ids unique")

    # Dangling references: every reference in an id-bearing field that resolves
    # to no record. Collected on the Report so the budget check can count them.
    for rid, (t, rec) in records.items():
        for sid in rec.get("sources") or []:
            # A dangling `sources` entry has always been a hard error: a record
            # with no readable provenance cannot be audited, and there is none
            # outstanding, so it is not part of the budget.
            rep.check(sid in bare,
                      f"{rid}: cites unknown id {sid!r} (need data/sources/{sid}.json)")
        for field in REF_FIELDS.get(t, {}):
            values = ref_values(rec.get(field))
            if not values and field in NULLABLE_REF_FIELDS:
                continue
            for x in values:
                if x in bare:
                    continue
                msg = f"{rid}: {field} {x!r} is not a record id"
                if (t, field) in HARD_REF_FIELDS:
                    # Never budgeted: an unresolvable engine makes the
                    # benchmark's number uninterpretable.
                    rep.errors.append(msg)
                    continue
                rep.warnings.append(f"{msg} (unresolved; budgeted debt)")
                rep.dangling.append(msg)

    # Scan for records in non-canonical directories
    all_canonical_dirs = set(DIRS.values())
    data_dir = ROOT / "data"
    if data_dir.exists():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and d.name not in all_canonical_dirs:
                for f in sorted(d.glob("*.json")):
                    rep.errors.append(
                        f"{d.name}/{f.stem}: record in non-canonical directory "
                        f"'{d.name}' (expected one of: {', '.join(sorted(all_canonical_dirs))})"
                    )

    del qualified
    # The budget applies to the whole repo, so it is checked here and overridable
    # by callers that want a different ceiling (intake.py validates a subset).
    check_budget(rep, budget if budget is not None else load_budget())
    return rep, len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print only errors")
    ap.add_argument("--type", action="append", choices=TYPES)
    ap.add_argument("--max-dangling-refs", type=int, default=None,
                    help="override the budget from tools/ref_budget.json "
                         "(for a one-off check or a test; CI uses the file)")
    args = ap.parse_args()
    budget = load_budget() if args.max_dangling_refs is None else args.max_dangling_refs
    rep, n = run(args.type or TYPES, args.quiet, budget)
    if not args.quiet:
        for w in rep.warnings:
            print(f"WARN   {w}")
    for e in rep.errors:
        print(f"ERROR  {e}")
    print(f"\n{n} records validated, {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    print(f"{len(rep.dangling)} dangling reference(s) against a budget of {budget}")
    if rep.dangling and len(rep.dangling) <= budget:
        print(f"  known debt, within budget: fix them and lower max_dangling_refs "
              f"in tools/ref_budget.json. Never raise it to go green.")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())