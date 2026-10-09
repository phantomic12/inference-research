#!/usr/bin/env python3
"""Validate every record in data/ against its schema and cross-check references.

    python tools/validate.py                # everything
    python tools/validate.py --quiet        # errors only
    python tools/validate.py --type engine  # one type

Cross-references are resolved against the set of record ids, using the one
field table in registry.REF_FIELDS. A reference that does not resolve is a
dangling reference: reported individually, counted, and fatal only once the
count exceeds the budget in tools/ref_budget.json. See DANGLING_BUDGET_NOTE.

docs/ is checked too: every `[[dir/id]]` citation in every docs/*.md file is
resolved against the record set, the same resolution build_bare and
rewrite_docs already perform when the site is built. A doc citation that
resolves to nothing degrades to inline code rather than a broken link, so it
used to be invisible: the docs rendered, the site built, and nothing failed.
See DOCS_BUDGET_NOTE.
"""  # noqa: D401
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
DOCS_BUDGET_FILE = ROOT / "tools" / "docs_ref_budget.json"

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


# ---------------------------------------------------------------------------
# DOCS CITATION BASELINE
#
# docs/*.md cites records with the same [[dir/id]] grammar the record fields
# use, and tools/build_site_data.py resolves each citation when it rewrites
# the docs for the site. An unresolvable citation does not raise there: it
# degrades to inline code, so a citation that rots (a renamed slug, a record
# deleted as duplicate, a typo in a doc) is silently rendered as code and
# nothing in the repo notices. This check closes that gap by resolving every
# citation in every docs/*.md file against the record set -- the SAME
# resolution build_bare/rewrite_docs performs -- and reporting each one that
# resolves to nothing.
#
# It does NOT change how index.py or build_site_data.py renders an
# unresolvable citation. Degrading to inline code is the correct display
# behaviour: the reader sees the label rather than a dead link. The defect
# was that nothing made the failure detectable, and this is the detection.
#
# The measured count at the time this check landed is 8 (measured 2026-10-08
# over 27 doc files and 3,979 records):
#   1. 06-glossary.md:300                  [SCHEMA.md](../SCHEMA.md)
#   2. 06-glossary.md:301                  [AGENTS.md](../AGENTS.md)
#   3. 07-benchmarking-methodology.md:11   [AGENTS.md](../AGENTS.md)
#   4. 07-benchmarking-methodology.md:289  [AGENTS.md](../AGENTS.md)
#   5. 07-benchmarking-methodology.md:291  [SCHEMA.md](../SCHEMA.md)
#   6. 16-migration-upgrade.md:201     [15-cost-per-token.md](15-cost-per-token.md)
#   7. 17-deployment-shape.md:205      [15-cost-per-token.md](15-cost-per-token.md)
#   8. 18-reliability.md:284           [15-cost-per-token.md](15-cost-per-token.md)
# All 8 are markdown links, not `[[dir/id]]` wikilinks: 5 point outside docs/
# (../SCHEMA.md, ../AGENTS.md) and 3 at a doc number that was renumbered
# (15-cost-per-token is now 08). The citation grammar build_site_data accepts
# resolves all 8 against the record set and finds nothing, so they degrade
# instead of linking -- which is the point: this check makes that whole class
# of invisible failure countable. Confirmed by a second method, feeding each
# citation to rewrite_docs() and counting the ones that return inline code.
#
# As with max_dangling_refs, these 8 are recorded as the budget in
# tools/docs_ref_budget.json so CI passes on the committed tree and fails on
# the 9th. Lower the budget in the same change that fixes one; never raise it
# to go green.
#
# What the 8 actually are, because the fix depends on the class:
#   5 are `[label](../SCHEMA.md)` / `[label](../AGENTS.md)` links in 06-glossary
#     (2) and 07-benchmarking-methodology (3). The targets are REAL files --
#     SCHEMA.md and AGENTS.md exist at the repo root -- but they sit outside
#     docs/, and the DOC_MD_LINK grammar in build_site_data resolves a .md
#     target against the docs/ stems only, so it degrades them to plain text.
#     They render as text on the live site, so the defect is real and visible.
#   3 are `[15-cost-per-token.md](15-cost-per-token.md)` in 16, 17 and 18.
#     That doc was renumbered to 08-cost-per-token.md and no 15- file exists,
#     so these are genuine rot: a renamed doc pointer that degrades silently.
#     This is exactly the failure mode the check was written to catch, and it
#     is the first thing to fix -- fixing it takes the count to 5 and the
#     budget in the same commit.
# ---------------------------------------------------------------------------


def load_docs_budget() -> int:
    """The recorded ceiling on unresolvable citations in docs/.

    A missing or malformed budget file is fatal, for the same reason as the
    dangling-reference budget: defaulting to 0 would fail CI on the committed
    tree, and defaulting to infinity would silently disable the check. See
    DOCS_BUDGET_NOTE.
    """
    try:
        budget = json.loads(DOCS_BUDGET_FILE.read_text(encoding="utf-8"))
        value = budget["max_docs_dangling_refs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"cannot read docs-citation budget from {DOCS_BUDGET_FILE}: {exc}. "
            f"See DOCS_BUDGET_NOTE in tools/validate.py."
        ) from exc
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(
            f"{DOCS_BUDGET_FILE}: max_docs_dangling_refs must be a non-negative integer")
    return value


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


def _docs_budget_error(count: int, budget: int) -> str:
    return (
        f"unresolvable docs citations: {count}, budget of {budget}. "
        f"Fix a citation in docs/, or lower max_docs_dangling_refs in "
        f"tools/docs_ref_budget.json once the debt is genuinely repaid; never "
        f"raise it to go green."
    )


# The two citation grammars docs/*.md uses. Both are mirrored verbatim from
# tools/build_site_data.py, because a citation this checker accepts but the
# site renderer rejects (or the reverse) is exactly the divergence this check
# exists to catch, and a second, hand-written pattern is how that happens.
#   [[target]] or [[target|label]]  -> WIKILINK
#   [label](file.md)                 -> DOC_MD_LINK
DOC_WIKILINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\]]*))?\]\]")
DOC_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:\.{1,2}/)*(?:docs/)?([A-Za-z0-9._-]+\.md)(#[^)]*)?\)")


def resolve_citation(target: str, records: dict[str, tuple[str, dict]],
                     bare: set[str], doc_slugs: set[str]) -> str | None:
    """Resolve one docs/ citation target, or None if nothing would link it.

    This is the resolution `rewrite_docs()` performs in build_site_data.py,
    restated so the checker can tell "resolves" from "degrades to inline code"
    without string-comparing rendered output. The two MUST agree; when they
    do not, the site is rendering a citation this checker reports, or vice
    versa, and that divergence is itself a defect.

    A doc stem is a doc-to-doc link ([[05-known-traps]]), not a record
    citation, and always resolves.
    """
    if target in doc_slugs:
        return target
    if "/" in target:
        # Qualified: dir/slug, or a data/ path, either with an optional
        # .json suffix. rewrite_docs resolves the trailing slug against the
        # bare-id set first, and falls back to the literal dir/slug.
        dir_, stem = target.split("/", 1)
        if dir_ == "data":
            if stem.startswith("sources/"):
                stem = stem[len("sources/"):]
            elif "/" in stem:
                stem = stem.rsplit("/", 1)[1]
        if stem.endswith(".json"):
            stem = stem[:-5]
        if stem in bare:
            return stem
        cand = target[:-5] if target.endswith(".json") else target
        return cand if cand in records else None
    return target if target in bare else None


def check_docs_citations(records: dict[str, tuple[str, dict]],
                         bare: set[str]) -> tuple[list[str], str | None]:
    """Every citation in every docs/*.md file that resolves to nothing.

    Returns (rows, skip_reason). `skip_reason` is non-None when the check
    could not run, which is reported rather than swallowed: a check that
    silently stops running is worse than one that fails.

    Two scoping rules, both about telling ROT from an ABSENT corpus:

    1. docs/00-index.md is generated by tools/index.py from data/, and
       build_site_data.load_docs() excludes it from the rendered docs for the
       same reason. Its citations are transcribed from record prose by the
       generator, so they cannot rot independently of data/: rename a record
       and the regenerated index transcribes the new slug, while the CI
       staleness step refuses a stale one. A generated file cannot be the
       subject of a citation-integrity check on its own authors.

    2. A citation into an EMPTY directory is not rot. If docs/ cites
       `accelerators/x` and data/accelerators/ holds no record at all, the tree
       is a subset -- that is what tools/test_tools.py builds (full docs/,
       near-empty data/) -- and every citation into that directory would be
       reported as broken. Such a tree cannot distinguish "this citation rotted"
       from "this corpus was never here", so the check is skipped and the
       reason is printed. The signal is directory-level, not a magic ratio,
       and it holds for the real repo: docs/ cites 12 directories, all of them
       populated (116 to 2,013 records each).

    Counted per citation, not per file: a rotted citation repeated four times
    is four failures, and a doc should not pass by repeating it. Each row is
    `file:line [[target]]` so the failure is directly editable.
    """
    docs_dir = ROOT / "docs"
    if not docs_dir.is_dir():
        return [], None
    doc_files = sorted(f for f in docs_dir.glob("*.md") if f.name != "00-index.md")
    doc_slugs = {f.stem for f in docs_dir.glob("*.md")}

    # Which directories does the authored docs actually cite, and which of
    # those hold any record here? A cited-but-empty directory means this tree
    # is a subset, so citations into it are not evidence of rot.
    cited_dirs = set()
    for f in doc_files:
        for m in DOC_WIKILINK.finditer(f.read_text(encoding="utf-8")):
            target = m.group(1)
            if target in doc_slugs or "/" not in target:
                continue
            cited_dirs.add(target.split("/", 1)[0])
    populated = {d for d in cited_dirs
                 if any(rid.split("/", 1)[0] == d for rid in records)}
    missing = sorted(cited_dirs - populated)
    if missing:
        return [], (f"docs/ cites {len(cited_dirs)} record directories, but "
                    f"{len(missing)} hold no records in this tree "
                    f"({', '.join(missing)}); a citation into an empty "
                    f"directory cannot be told apart from an absent corpus, "
                    f"so the check is skipped rather than reported as rot")

    unresolved: list[str] = []
    for f in doc_files:
        text = f.read_text(encoding="utf-8")
        for m in DOC_WIKILINK.finditer(text):
            if resolve_citation(m.group(1), records, bare, doc_slugs) is None:
                unresolved.append(f"{f.name}:{text[:m.start()].count(chr(10)) + 1} "
                                  f"[[{m.group(1)}]]")
        for m in DOC_MD_LINK.finditer(text):
            # A .md link's target is a doc stem. rewrite_docs strips the .md
            # before looking the stem up (stem = target[:-3]), and a link with
            # a label degrades to plain text when the stem is not a doc, so
            # "not a doc stem" is the failure either way. The label and any
            # #fragment are not part of the citation.
            if resolve_citation(m.group(2)[:-3], records, bare, doc_slugs) is None:
                unresolved.append(f"{f.name}:{text[:m.start()].count(chr(10)) + 1} "
                                  f"[{m.group(1)}]({m.group(2)})")
    return unresolved, None


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        # Unresolved cross-references, one entry each. Kept as data, not just as
        # warnings, because the budget check needs the COUNT.
        self.dangling: list[str] = []
        # Unresolved citations in docs/, one entry each, same reasoning.
        self.docs_dangling: list[str] = []

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


def check_docs_budget(rep: Report, budget: int) -> None:
    """Fail the run when unresolvable docs citations exceed `budget`.

    Same shape as check_budget, against a separate budget: the two counts are
    unrelated debt, and merging them would let one go green by being small.
    """
    if len(rep.docs_dangling) > budget:
        rep.errors.append(_docs_budget_error(len(rep.docs_dangling), budget))


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


def load_record_ids() -> tuple[dict[str, tuple[str, dict]], set[str]]:
    """({qualified id -> (type, {})}, bare slugs) for data/, filename only.

    The docs citation graph spans every record type, so it is resolved against
    the WHOLE record set: `--type engine` must not report an accelerator
    citation as unresolvable. Resolution needs only ids -- the slug is the
    filename stem, and validate_value already enforces `id == stem` -- so this
    reads no JSON and costs nothing next to the schema pass.
    """
    records: dict[str, tuple[str, dict]] = {}
    bare_all: set[str] = set()
    for t, dname in DIRS.items():
        d = ROOT / "data" / dname
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            records[f"{dname}/{f.stem}"] = (t, {})
            bare_all.add(f.stem)
    ambiguous = {b for b in bare_all
                 if sum(1 for rid in records if rid.split("/", 1)[1] == b) > 1}
    return records, bare_all - ambiguous


def run(types: list[str], quiet: bool, budget: int | None = None,
        docs_budget: int | None = None) -> tuple[Report, int]:
    """Validate every record of `types`.

    `budget` / `docs_budget` default to the recorded baselines in
    tools/ref_budget.json and tools/docs_ref_budget.json. They are parameters
    only so a caller can lower a ceiling for a deliberate, separately-reviewed
    check; CI passes nothing and gets the recorded values.
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

    # docs/ citations. Resolved against the whole record set rather than the
    # `types` subset, because a citation may name any type -- including one
    # this particular run is not validating.
    #
    # This needs no extra file reads: the record ids come from the filenames,
    # which validate_value has already proved equal each record's `id`. The
    # count is therefore identical whether run over the whole repo (CI, the
    # normal path) or a `--type` subset.
    doc_records, doc_bare = load_record_ids()
    rows, skip_reason = check_docs_citations(doc_records, doc_bare)
    if skip_reason is not None:
        # Not run, not skipped silently: the reason is printed so a tree that
        # cannot support the check says so instead of looking clean.
        rep.warnings.append(f"docs citation check skipped: {skip_reason}")
    rep.docs_dangling = rows
    for row in rows:
        rep.warnings.append(f"{row} (unresolved; budgeted debt)")
    if skip_reason is None:
        check_docs_budget(rep, docs_budget if docs_budget is not None
                          else load_docs_budget())
    return rep, len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="print only errors")
    ap.add_argument("--type", action="append", choices=TYPES)
    ap.add_argument("--max-dangling-refs", type=int, default=None,
                    help="override the budget from tools/ref_budget.json "
                         "(for a one-off check or a test; CI uses the file)")
    ap.add_argument("--max-docs-dangling-refs", type=int, default=None,
                    help="override the budget from tools/docs_ref_budget.json "
                         "(for a one-off check or a test; CI uses the file)")
    args = ap.parse_args()
    budget = load_budget() if args.max_dangling_refs is None else args.max_dangling_refs
    docs_budget = (load_docs_budget() if args.max_docs_dangling_refs is None
                   else args.max_docs_dangling_refs)
    rep, n = run(args.type or TYPES, args.quiet, budget, docs_budget)
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
    print(f"{len(rep.docs_dangling)} unresolvable docs citation(s) against a "
          f"budget of {docs_budget}")
    if rep.docs_dangling and len(rep.docs_dangling) <= docs_budget:
        print(f"  known debt, within budget: fix them and lower "
              f"max_docs_dangling_refs in tools/docs_ref_budget.json. Never "
              f"raise it to go green.")
    elif len(rep.docs_dangling) == 0:
        print(f"  (see the skip notice above if no docs/ files were resolvable)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())