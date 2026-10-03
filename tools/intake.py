#!/usr/bin/env python3
"""Intake: gate uncommitted crawler output, then commit it on a branch.

Designed for a repo where many subagents write records concurrently. A record is
only eligible if it is COMPLETE (no required-but-null fields), which stops a
half-written skeleton from being committed. Eligibility is decided per file, so
one agent's in-flight work never blocks another agent's finished records.

    python tools/intake.py status          # eligible / blocked / per-type counts
    python tools/intake.py check           # validate only the eligible set
    python tools/intake.py commit -m "..." # commit eligible records to a branch
    python tools/intake.py stage           # dry run: list exactly what would commit

Writes a report to $TMPDIR/intake-last.json for the PR step to consume.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, TYPES  # noqa: E402
from validate import run as validate_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Fields that must be non-null for a record to count as finished. Chosen from
# each schema's required list: a required field that is present but null means a
# generator ran and the agent never filled it in.
REQUIRED_BY_TYPE = {
    "accelerator": ["vendor", "architecture", "vram_gb", "memory_bandwidth_basis", "flops"],
    "flop": ["class", "bound_by", "arithmetic_intensity", "scales_with", "workarounds"],
    "engine": ["repo", "supported_backends", "scheduling", "supported_formats", "best_for"],
    "quantization": ["scheme", "bits", "weight_group_size", "kernels"],
    "interconnect": ["kind", "bandwidth_gbps", "topology"],
    "benchmark": ["engine_id", "metric", "value", "methodology", "measured_by"],
    "gotcha": ["class", "symptom", "root_cause", "workaround", "severity"],
    "supply": ["kind", "accelerator_ids", "availability"],
    "model": ["architecture", "params_b", "num_layers", "num_kv_heads", "head_dim"],
    "source": ["url", "kind", "accessed"],
}

EMPTY = (None, [], {}, "")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=check)


def changed_records() -> tuple[list[Path], list[Path]]:
    """Uncommitted record files, split into new/modified vs deleted."""
    out = git("status", "--porcelain", check=True).stdout
    new, deleted = [], []
    for line in out.splitlines():
        code, _, name = line[:2].strip(), line[2:3], line[3:].strip()
        if not name.endswith(".json"):
            continue
        p = ROOT / name
        if code == "D":
            deleted.append(p)
        else:
            new.append(p)
    return new, deleted


def is_complete(path: Path) -> tuple[bool, str]:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    if not isinstance(rec, dict):
        return False, "top level is not an object"
    t = rec.get("type")
    if t not in REQUIRED_BY_TYPE:
        return False, f"unknown or missing type {t!r}"
    missing = [f for f in REQUIRED_BY_TYPE[t] if rec.get(f) in EMPTY]
    if missing:
        return False, f"unfilled required fields: {', '.join(missing)}"
    return True, ""


def collect() -> dict:
    new, deleted = changed_records()
    eligible, blocked = [], []
    for p in new:
        ok, why = is_complete(p)
        (eligible if ok else blocked).append({"path": p, "why": why})
    counts: dict[str, int] = {}
    for item in eligible:
        t = json.loads(item["path"].read_text(encoding="utf-8")).get("type")
        counts[t] = counts.get(t, 0) + 1
    return {"eligible": eligible, "blocked": blocked, "deleted": deleted,
            "counts": counts}


def cmd_status(args) -> int:
    state = collect()
    print(f"eligible records : {len(state['eligible'])}")
    print(f"blocked (partial): {len(state['blocked'])}")
    print(f"deleted          : {len(state['deleted'])}")
    if state["counts"]:
        print("\nby type:")
        for t, n in sorted(state["counts"].items(), key=lambda x: -x[1]):
            print(f"  {t:<14}{n}")
    if state["blocked"]:
        print("\nblocked (first 15) - these agents are still working or stalled:")
        for item in state["blocked"][:15]:
            print(f"  {item['path'].relative_to(ROOT)}: {item['why'][:70]}")
    return 0


def build_overlay(dest: Path, eligible: list[dict]) -> Path:
    """Materialise exactly HEAD + the eligible set into `dest`, and return it.

    This is the ONLY correct tree to validate or index against in a live repo:
    running the tools over the real working tree would also pick up other
    agents' in-flight skeletons, which are neither committed nor complete.
    """
    for sub in ("tools", "schemas"):
        shutil.copytree(ROOT / sub, dest / sub)
    for t in TYPES:
        (dest / "data" / DIRS[t]).mkdir(parents=True, exist_ok=True)
    # committed tree first. Read the blobs from git, not from the working tree:
    # a crawler agent editing an ALREADY-committed record would otherwise leak
    # its half-written version into the overlay.
    committed = git("ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
    names = [n for n in committed if n.endswith(".json") and n.startswith("data/")]
    if names:
        # Read each blob at HEAD: with `git cat-file --batch`. Reading from the
        # working tree instead would leak another agent's in-progress edit of
        # an already-committed record into the overlay.
        specs = "\n".join(f"HEAD:{n}" for n in names) + "\n"
        raw = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=ROOT, input=specs.encode(),
            capture_output=True, check=True).stdout
        off = 0
        for n in names:
            nl = raw.index(b"\n", off)
            header = raw[off:nl].split(b" ")
            off = nl + 1
            if len(header) != 3:
                raise RuntimeError(f"cat-file could not read HEAD:{n}")
            size = int(header[2])
            dst = dest / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw[off:off + size])
            off += size + 1
    # then the eligible overlay on top
    for item in eligible:
        dst = dest / item["path"].relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(item["path"].read_bytes())
    return dest


def cmd_check(args) -> int:
    """Validate ONLY the eligible set, in a temp overlay, so other agents'
    in-flight skeletons elsewhere in the tree do not produce false errors."""
    state = collect()
    if not state["eligible"]:
        print("nothing eligible")
        return 0
    tmp = build_overlay(Path(tempfile.mkdtemp(prefix="intake-")), state["eligible"])
    try:
        sys.path.insert(0, str(tmp / "tools"))
        import importlib
        for mod in ("registry", "validate"):
            if mod in sys.modules:
                del sys.modules[mod]
        sys.path.insert(0, str(tmp / "tools"))
        vr = importlib.import_module("validate")
        rep, n = vr.run(TYPES, quiet=True)
        mine = {str(p.relative_to(ROOT)).replace("\\", "/") for p in
                (i["path"] for i in state["eligible"])}
        relevant = [e for e in rep.errors if e.split(":")[0].strip() in mine
                    or any(m in e for m in mine)]
        for e in relevant:
            print(f"ERROR  {e}")
        print(f"\n{n} records in overlay, {len(relevant)} errors in the eligible set")
        return 1 if relevant else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def generate_index(tree: Path) -> str:
    """Run tools/index.py against `tree` (must be an overlay root) and return
    the generated markdown.

    The tool runs in a subprocess because index.py derives its ROOT from
    __file__; executing the copy inside `tree` keeps the output bound to the
    overlay, never to the live working tree.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, str(tree / "tools" / "index.py")],
                       cwd=tree, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"index.py failed in overlay:\n{r.stdout}\n{r.stderr}")
    return (tree / "docs" / "00-index.md").read_text(encoding="utf-8")


def cmd_commit(args) -> int:
    state = collect()
    if not state["eligible"]:
        print("nothing eligible to commit")
        return 0
    rc = cmd_check(args)
    if rc != 0 and not args.force:
        print("\nrefusing to commit: eligible set has validation errors (use --force to override)")
        return rc

    # Build the exact tree we are about to commit (HEAD + eligible), and index
    # THAT. Indexing the working tree would fold in other agents' in-flight
    # records into the index without folding them into the commit.
    index_text = None
    tmp = Path(tempfile.mkdtemp(prefix="intake-commit-"))
    try:
        build_overlay(tmp, state["eligible"])
        index_text = generate_index(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    git("checkout", "-b", args.branch, check=False)
    for item in state["eligible"]:
        rel = item["path"].relative_to(ROOT)
        git("add", "--", str(rel), check=False)
    for p in state["deleted"]:
        git("add", "--", str(p.relative_to(ROOT)), check=False)
    if index_text is not None:
        idx = ROOT / "docs" / "00-index.md"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(index_text, encoding="utf-8")
        git("add", "--", "docs/00-index.md", check=False)

    body = args.message
    if state["counts"]:
        summary = ", ".join(f"{n} {t}" for t, n in sorted(state["counts"].items()))
        body += f"\n\nRecords: {summary}"
    if state["blocked"]:
        body += f"\n\nSkipped {len(state['blocked'])} incomplete records still being written."
    git("commit", "-m", body, check=False)

    report = {
        "branch": args.branch,
        "committed": len(state["eligible"]),
        "counts": state["counts"],
        "blocked": len(state["blocked"]),
        "files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in
                  (i["path"] for i in state["eligible"])],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path.home() / "AppData" / "Local" / "Temp" / "intake-last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ncommitted {len(state['eligible'])} records to {args.branch}")
    print(f"report: {out}")
    return 0


def cmd_stage(args) -> int:
    state = collect()
    for item in state["eligible"]:
        print(f"+ {item['path'].relative_to(ROOT)}")
    print(f"\n{len(state['eligible'])} files would be committed, "
          f"{len(state['blocked'])} blocked")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("status", cmd_status), ("check", cmd_check),
                     ("stage", cmd_stage), ("commit", cmd_commit)):
        p = sub.add_parser(name)
        p.add_argument("-m", "--message", default="Intake: crawler output")
        p.add_argument("-b", "--branch", default="intake")
        p.add_argument("--force", action="store_true")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)