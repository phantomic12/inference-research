#!/usr/bin/env python3
"""Query the knowledge base.

    python tools/query.py list engines
    python tools/query.py get engines/vllm
    python tools/query.py search "fp8"              # every field, ranked
    python tools/query.py where vendor=nvidia       # exact match
    python tools/query.py where bound_by=memory --format json
    python tools/query.py refs engines/llama-cpp    # outgoing + incoming links
    python tools/query.py stats
    python tools/query.py gaps                      # record types with no data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, REF_FIELDS, TITLES, TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SHOW_FIELDS = ("notes", "best_for", "hardware_caveats", "symptom", "root_cause",
               "workaround", "arithmetic_intensity", "quality_delta",
               "memory_bandwidth_basis", "notable_features")


def load_all() -> dict[str, tuple[str, dict]]:
    out: dict[str, tuple[str, dict]] = {}
    for t in TYPES:
        d = ROOT / "data" / DIRS[t]
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out[f"{DIRS[t]}/{f.stem}"] = (t, rec)
    return out


def flatten(obj, prefix="") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj
    return out


def cmd_list(args) -> None:
    recs = load_all()
    ids = sorted(i for i in recs if i.startswith(DIRS[args.type] + "/"))
    if not ids:
        print(f"no records in data/{DIRS[args.type]}/")
        return
    if args.format == "json":
        print(json.dumps([{"id": i, "name": recs[i][1].get("name")} for i in ids], indent=2))
        return
    w = max(len(i) for i in ids) + 2
    print(f"{'ID'.ljust(w)}{'STATUS'.ljust(11)}{'CONF':<6}NAME")
    for i in ids:
        r = recs[i][1]
        print(f"{i.ljust(w)}{str(r.get('status','')).ljust(11)}"
              f"{str(r.get('confidence','')):<6}{r.get('name','')}")


def cmd_get(args) -> None:
    recs = load_all()
    key = args.id if "/" in args.id else f"{DIRS[args.type]}/{args.id}"
    if key not in recs:
        near = [k for k in sorted(recs) if key.split("/")[-1].split("-")[0] in k]
        print(f"no such record: {key}")
        if near:
            print("did you mean:\n  " + "\n  ".join(near[:12]))
        raise SystemExit(1)
    print(json.dumps(recs[key][1], indent=2, ensure_ascii=False))


def cmd_search(args) -> None:
    recs = load_all()
    q = args.query.lower()
    hits = []
    for rid, (_, r) in recs.items():
        score = 0
        for field, val in flatten(r).items():
            if isinstance(val, str) and q in val.lower():
                score += 3 if field in ("name", "id") else 1
        if score:
            hits.append((score, rid, r))
    hits.sort(key=lambda x: (-x[0], x[1]))
    if not hits:
        print(f"no matches for {args.query!r}")
        return
    print(f"{len(hits)} matches for {args.query!r}\n")
    for score, rid, r in hits[: args.limit]:
        print(f"[{score:>3}] {rid}")
        for f in SHOW_FIELDS:
            if r.get(f):
                print(f"      {f}: {str(r[f])[:240]}")
        print()


def cmd_where(args) -> None:
    recs = load_all()
    key, _, want = args.filter.partition("=")
    key, want = key.strip(), want.strip()
    rows = []
    for rid, (_, r) in recs.items():
        flat = flatten(r)
        if key in flat and str(flat[key]).lower() == want.lower():
            rows.append((rid, r))
    if not rows:
        print(f"no records where {key} == {want!r}")
        return
    if args.format == "json":
        print(json.dumps([r for _, r in rows], indent=2, ensure_ascii=False))
        return
    for rid, r in rows:
        print(f"{rid}")
        print(f"  name: {r.get('name')}   [{r.get('status')}, conf {r.get('confidence')}]")
        for f in ("vendor", "architecture", "kind", "class", "bound_by", "severity",
                  "metric", "value", "unit", "model", "scheme", "bits",
                  "best_for", "hardware_caveats", "notable_features"):
            if r.get(f) not in (None, [], ""):
                print(f"  {f}: {r[f]}")


def cmd_refs(args) -> None:
    recs = load_all()
    key = args.id if "/" in args.id else f"{DIRS[args.type]}/{args.id}"
    if key not in recs:
        raise SystemExit(f"no such record: {key}")
    t, r = recs[key]
    bare = key.split("/", 1)[1]
    bare_to_qualified = {i.split("/", 1)[1]: i for i in recs}
    print(f"{key}  ({t})")
    print("  cites:")
    for s in r.get("sources") or []:
        ok = s in bare_to_qualified
        print(f"    [{'ok' if ok else 'MISSING'}] {s}  "
              f"{recs[bare_to_qualified[s]][1].get('name','') if ok else ''}")
    print("  referenced by:")
    # Back-references are INCOMING, so the field list must come from the OTHER
    # record's type, not this one's. Reading it off `t` only found records that
    # cite a record of their own type, which is why benchmark -> engine and
    # supply -> accelerator back-references were invisible. REF_FIELDS in
    # registry.py is the single source of truth and covers every type, so a new
    # record type gets back-references for free rather than needing a second
    # edit to a hardcoded table here.
    found = False
    for rid, (tt, rr) in sorted(recs.items()):
        if rid == key:
            continue
        for f in list(REF_FIELDS.get(tt, {})) + ["sources"]:
            v = rr.get(f)
            vals = [str(x) for x in v] if isinstance(v, list) else [str(v)] if v else []
            # reference fields hold bare slugs; compare on the bare slug
            if bare in vals:
                print(f"    {rid}  (via {f})")
                found = True
                break
    if not found:
        print("    (nothing)")


def cmd_stats(args) -> None:
    recs = load_all()
    print(f"{len(recs)} records\n")
    for t in TYPES:
        n = sum(1 for _, (tt, _) in recs.items() if tt == t)
        print(f"  {TITLES[t]:<24}{n}")
    st, conf = defaultdict(int), []
    for _, (_, r) in recs.items():
        st[r.get("status", "?")] += 1
        if isinstance(r.get("confidence"), (int, float)):
            conf.append(float(r["confidence"]))
    print(f"\nstatus: {dict(st)}")
    if conf:
        print(f"mean confidence: {sum(conf)/len(conf):.2f}")


def cmd_gaps(args) -> None:
    recs = load_all()
    empty = [t for t in TYPES if not any(tt == t for _, (tt, _) in recs.items())]
    print(f"{len(recs)} records")
    if empty:
        print("\nempty record types:")
        for t in empty:
            print(f"  [ ] {TITLES[t]:<24}data/{DIRS[t]}/")
    else:
        print("\nno empty record types")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list records of a type")
    p.add_argument("type", choices=TYPES)
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("get", help="print one record")
    p.add_argument("id"); p.add_argument("type", choices=TYPES, nargs="?")
    p.set_defaults(fn=cmd_get)

    p = sub.add_parser("search", help="full-text across every field")
    p.add_argument("query"); p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("where", help="exact-match field filter")
    p.add_argument("filter")
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.set_defaults(fn=cmd_where)

    p = sub.add_parser("refs", help="what a record cites and what cites it")
    p.add_argument("id"); p.add_argument("type", choices=TYPES, nargs="?")
    p.set_defaults(fn=cmd_refs)

    sub.add_parser("stats", help="counts by type and status").set_defaults(fn=cmd_stats)
    sub.add_parser("gaps", help="record types with no data").set_defaults(fn=cmd_gaps)
    args = ap.parse_args()
    args.fn(args)