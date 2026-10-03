#!/usr/bin/env python3
"""Validate every record in data/ against its schema and cross-check references.

    python tools/validate.py                # everything
    python tools/validate.py --quiet        # errors only
    python tools/validate.py --type engine  # one type
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            self.errors.append(msg)

    def warn(self, cond: bool, msg: str) -> None:
        if not cond:
            self.warnings.append(msg)


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


def run(types: list[str], quiet: bool) -> tuple[Report, int]:
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
    qualified = set(records)
    bare = {rid.split("/", 1)[1] for rid in records}
    duplicate_bare = sorted(
        b for b in bare
        if sum(1 for rid in records if rid.split("/", 1)[1] == b) > 1
    )
    for d in duplicate_bare:
        rep.errors.append(f"ambiguous id {d!r}: appears in more than one record type")

    for rid, (t, rec) in records.items():
        for sid in rec.get("sources") or []:
            rep.check(sid in bare,
                      f"{rid}: cites unknown id {sid!r} (need data/sources/{sid}.json)")
        if t == "benchmark":
            rep.check(rec.get("engine_id") in bare,
                      f"{rid}: engine_id {rec.get('engine_id')!r} is not a record id")
            for a in rec.get("accelerator_ids") or []:
                rep.warn(a in bare, f"{rid}: accelerator {a!r} not a record id (ok if planned)")
        if t == "gotcha":
            for a in rec.get("affects") or []:
                rep.warn(a in bare, f"{rid}: affects {a!r} not a record id (ok if planned)")
        if t == "flop":
            for a in rec.get("affected_by_hardware") or []:
                rep.warn(a in bare, f"{rid}: affected_by_hardware {a!r} not a record id (ok if planned)")
        if t == "quantization":
            for a in (rec.get("native_support") or []) + (rec.get("emulated_support") or []):
                rep.warn(a in bare, f"{rid}: hardware {a!r} not a record id (ok if planned)")
        if t == "accelerator":
            for a in rec.get("interconnect") or []:
                rep.warn(a in bare, f"{rid}: interconnect {a!r} not a record id (ok if planned)")
        if t == "supply":
            for a in rec.get("accelerator_ids") or []:
                rep.warn(a in bare, f"{rid}: accelerator {a!r} not a record id (ok if planned)")
        if t == "benchmark" and isinstance(rec.get("model"), str):
            # benchmark.model is a free-text model NAME, not a record id; the
            # schema allows it because benchmarks predate the model type.
            pass

    del qualified
    return rep, len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print only errors")
    ap.add_argument("--type", action="append", choices=TYPES)
    args = ap.parse_args()
    rep, n = run(args.type or TYPES, args.quiet)
    if not args.quiet:
        for w in rep.warnings:
            print(f"WARN   {w}")
    for e in rep.errors:
        print(f"ERROR  {e}")
    print(f"\n{n} records validated, {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())