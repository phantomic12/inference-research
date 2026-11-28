#!/usr/bin/env python3
"""One canonical spelling per agentic benchmark, shared by the writer and the scorer.

Why this module exists: published cards spell the same benchmark 83 different ways.
"Terminal Bench 2.0", "TerminalBench", "Terminal-Bench v2.1"; "TAU-Retail",
"Tau-Bench Retail", "τ²-Bench"; "BFCLv3", "BFCL V3", "BFCL_v3_MultiTurn (Acc)".
Two independent regex tables used to decide agentic-ness:

  * tools/apply_agentic_backfill.py decided whether a row may be written at all;
  * tools/build_joined_views.py._AXES decided whether a written row counts.

The first was strict enough to reject real scores ("SWE Verified"), and the second
knew only six spellings, so a row could pass the gate and then silently score
nothing. Both now call canonical() here: "accepted" and "counted" are one question.

canonical() returns the catalog's house spelling (version and harness preserved,
because "Terminal-Bench 2.0" and "Terminal-Bench 2.1" are different difficulty),
or None when the name is not an agentic benchmark at all.

    python3 tools/benchmark_names.py "τ²-Bench Retail" "SWE Verified"
"""
from __future__ import annotations

import re
import sys

# Unicode spellings that show up in cards and PDFs. τ is the Greek tau that vendors
# use for the tau-bench family; ² / ³ are superscript version markers.
_FOLD = str.maketrans({
    "τ": "tau", "Τ": "tau", "²": "2", "³": "3", "–": "-", "—": "-", "\u00a0": " ",
})

# Glued version suffixes: "BFCLv3", "tau2 bench", "OSWorld2.0". Split the acronym
# from the digits so one version regex handles both spellings.
_GLUED = re.compile(
    r"\b(bfcl|tau|terminal ?bench|swe ?bench|deepswe|osworld|androidworld"
    r"|webarena|browsecomp|webvoyager)\s*(v?\d+(?:\.\d+)?)\b"
)

# Parentheticals that state an estimator, not a harness. "(Acc)" and "(Pass@1)"
# describe how the same test was scored; "(OpenHands)" describes a different test.
_ESTIMATOR = re.compile(
    r"^(acc|accuracy|pass@1|pass at 1|resolved|avg|average|mean|score|overall"
    r"|average over \d+)$"
)


def _strip_estimators(name: str) -> str:
    """Remove estimator tails that carry no benchmark identity.

    Cards write "SWE-bench Verified, OpenHands pass@1" and "Tau-Bench (Pass@1)
    Airline". The `pass@1` / `acc` tail says how the number was scored, not which
    test it is, so it must not stop the row from matching. A harness named in the
    tail ("OpenHands") is identity and is preserved as a parenthetical.
    """
    s = (name or "").strip()
    # "..., OpenHands pass@1" -> " (OpenHands)"
    s = re.sub(r",\s*([A-Za-z0-9\-]+)\s+pass@1\s*$", r" (\1)", s, flags=re.I)
    # "(Pass@1)" anywhere, and bare trailing "pass@1" / "acc pass@1" / "acc"
    s = re.sub(r"\(\s*(pass@1|acc|accuracy|resolved)\s*\)", " ", s, flags=re.I)
    s = re.sub(r"[,;\s]+(acc\s+)?pass@1\s*$", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _norm(name: str) -> str:
    """Lowercase, fold unicode, split glued versions, flatten separators."""
    s = _strip_estimators(name).translate(_FOLD).lower()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9.()*+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _GLUED.sub(r"\1 \2", s)
    # "τ²-Bench" folds to "tau2 bench"; the version belongs between "tau" and
    # "bench" so one tau regex handles every spelling.
    s = re.sub(r"\btau(\d+)\s+bench\b", r"tau \1 bench", s)
    # Glued acronyms with no version to split on.
    s = re.sub(r"\bmcp ?(atlas|mark)\b", r"mcp \1", s)
    # "BrowseComp+" glues the plus onto the name; every other spelling ("BrowseComp
    # Plus", "browsecomp plus") already resolves, so folding it here keeps the one
    # suite in one bucket instead of silently dropping it as unrecognized.
    s = re.sub(r"\b(browsecomp)\+", r"\1 plus", s)
    return s


def _split_parens(s: str) -> tuple[str, str | None]:
    """Split a trailing parenthetical off a name; drop it if it is an estimator."""
    m = re.search(r"\(([^)]*)\)\s*$", s)
    if not m:
        return s.strip(), None
    inner = m.group(1).strip()
    stem = s[: m.start()].strip()
    if _ESTIMATOR.match(inner):
        return stem, None
    return stem, inner


# Explicitly NOT agentic. Hard reasoning/knowledge/coding tests whose names contain
# agentic-looking substrings ("Arena-Hard" is not WebArena; "MMLU" is not MCP-Atlas).
# Checked first, so no later family rule can undo the rejection. Deliberately
# specific: a bare "arena" would kill WebArena.
_NON_AGENTIC = re.compile(
    r"aime|math|gpqa|mmlu|hle\b|humanity|arc-agi|hellaswag|truthfulqa|bbh"
    r"|big.?bench|livebench|ifeval|ifbench|mt.?bench|arena.?hard|alignbench"
    r"|livecodebench|humaneval|scicode|mmmu|critpt|c-eval"
    r"|gaia|erqa|omnidoc|longbench|imo|hmmt|supergpqa|cmmlu"
)

_SWE_VARIANTS = {
    "verified": "SWE-bench Verified",
    "pro": "SWE-bench Pro",
    "multilingual": "SWE-bench Multilingual",
    "lite": "SWE-bench Lite",
}
_SWE_PREFIX = {
    "multiswe bench": "Multi-SWE-Bench",
    "multi swe bench": "Multi-SWE-Bench",
    "deepswe": "DeepSWE",
}
_TOOL_NAMES = {"bfcl": "BFCL", "mcp atlas": "MCP-Atlas", "mcp mark": "MCPMark",
               "toolathlon": "Toolathlon"}
# BrowseComp has named sub-suites that are genuinely different tests (a Chinese
# set, a "Plus" hard set, a top-100 subset). They keep their names because the
# axis scores each benchmark name against its own best-known figure.
_BROWSECOMP_VARIANTS = {"plus", "zh", "top100", "live"}
_TAU_SUBS = {"retail", "airline", "telecom", "banking", "verified"}

_VER = r"v?\d+(?:\.\d+)?"


def canonical(name: str) -> str | None:
    """House spelling for an agentic benchmark name, or None if it is not one.

    Returns None for non-agentic tests, unparseable names, and empties. A caller
    must treat None as "do not write this row" rather than guessing a name.

    Two rows can legitimately carry the same test name and different scores when
    they were measured with different harnesses (Kimi K2 reports Terminal-Bench 30
    on its in-house framework and 25 on Terminus). The harness belongs to the
    identity when the name states one, so "Terminal-Bench (OpenCode)" and
    "Terminal-Bench" stay distinct; a maker-internal harness that is not named in
    the benchmark title cannot be recovered here and is the caller's problem, to be
    recorded in `config`.
    """
    if not (name or "").strip():
        return None
    low = _norm(name)
    if not low or _NON_AGENTIC.search(low):
        return None
    stem, harness = _split_parens(low)
    words = stem.split()
    if not words:
        return None

    # ---- Terminal-Bench -------------------------------------------------
    m = re.match(rf"^terminal bench(?: {_VER})?(?: (hard|lite))?$", stem)
    if m:
        ver = re.search(_VER, stem)
        tier = m.group(1)
        out = "Terminal-Bench"
        if ver:
            out += f" {ver.group(0).lstrip('v')}"
        if tier:
            out += f" {tier.capitalize()}"
        return f"{out} ({harness})" if harness else out
    if re.match(rf"^terminalbench(?: {_VER})?$", stem):
        ver = re.search(_VER, stem)
        return "Terminal-Bench" + (f" {ver.group(0).lstrip('v')}" if ver else "")

    # ---- SWE-bench family ------------------------------------------------
    # "Multi-SWE-Bench" is its own multilingual suite, not a spelling of
    # SWE-bench, so its prefix is tested before the bare `swe bench` form.
    m = re.match(rf"^(multiswe bench|multi swe bench|swe bench|swe|deepswe)"
                 rf"(?: (verified|pro|multilingual|lite))?(?: {_VER})?$", stem)
    if m:
        prefix, variant = m.group(1), m.group(2)
        ver = re.search(_VER, stem)
        if prefix in _SWE_PREFIX:
            out = _SWE_PREFIX[prefix]
            if ver and prefix == "deepswe":
                out += f" {ver.group(0).lstrip('v')}"
        elif variant:
            out = _SWE_VARIANTS.get(variant, f"SWE-bench {variant.capitalize()}")
            # A version on the variant is the test's identity: SWE-bench Pro V2 and
            # SWE-bench Pro are different evaluations, and merging them would let a
            # score from one normalise against the best of the other.
            if ver:
                out += f" v{ver.group(0).lstrip('v')}"
        else:
            out = "SWE-bench"
        # The harness is part of the number: Verified with OpenHands and with
        # Agentless are not comparable, so it must survive canonicalization.
        return f"{out} ({harness})" if harness else out
    # ---- Web / browsing --------------------------------------------------
    # BrowseComp sub-suites ("Plus", "-ZH", "Top100") are distinct tests and keep
    # their suffix, so each is scored against its own best-known figure. A
    # leading "Agentic coding: " / "Search: " label is a table caption, not part
    # of the name.
    stem = re.sub(r"^(agentic( coding)?|search|agentic browsing):\s*", "", stem)
    m = re.match(r"^(browsecomp|webarena|webvoyager)(?: (verified))?"
                 rf"(?: ([a-z0-9.]+(\+)?))?(?: {_VER})?$", stem)
    if m:
        out = {"browsecomp": "BrowseComp", "webarena": "WebArena",
               "webvoyager": "WebVoyager"}[m.group(1)]
        if m.group(2):
            out += "-Verified"
        tok = (m.group(3) or "").rstrip("+")
        ver = re.search(_VER, stem)
        if m.group(3):
            # "BrowseComp+" and "BrowseComp-Plus" are the same suite.
            if tok in _BROWSECOMP_VARIANTS or tok in ("plus",):
                label = "Plus" if tok in ("plus",) else tok.upper() if tok == "zh" else tok.capitalize()
                out += f"-{label}" if tok not in ("top100",) else " Top100"
            else:
                return None
        elif ver and m.group(1) != "webarena":
            out += f" {ver.group(0).lstrip('v')}"
        return out

    # ---- Tool use: BFCL, tau-bench, MCP, Toolathlon ----------------------
    # tau is special-cased: after normalization the family shows up as every
    # combination of {"tau", "tau2", "tau 2", "tau bench", "tau 2 bench",
    # "taubench"} plus an optional version and sub-benchmark.
    m = re.match(r"^tau(?: ?(\d+(?:\.\d+)?))?(?: ?bench)?"
                 r"(?: (retail|airline|telecom|banking|verified))?$", stem)
    if m:
        num = m.group(1) or "1"
        head = "tau-bench" if num == "1" else f"tau{num}-bench"
        return f"{head} {m.group(2).capitalize()}" if m.group(2) else head

    m = re.match(rf"^(bfcl|mcp atlas|mcp mark|toolathlon)"
                 rf"(?: {_VER})?(?: ([a-z]+))?$", stem)
    if m:
        kind, rest = m.group(1), m.group(2)
        ver = re.search(_VER, stem)
        out = _TOOL_NAMES[kind]
        if ver:
            out += f" v{ver.group(0).lstrip('v')}"
        if rest and rest != "verified":
            out += f" {rest.capitalize()}"
        return out

    # ---- Computer use ----------------------------------------------------
    m = re.match(rf"^(osworld|androidworld|agentworld)(?: (verified))?(?: {_VER})?$", stem)
    if m:
        out = {"osworld": "OSWorld", "androidworld": "AndroidWorld",
               "agentworld": "AgentWorld"}[m.group(1)]
        if m.group(2):
            out += "-Verified"
        ver = re.search(_VER, stem)
        if ver:
            out += f" {ver.group(0).lstrip('v')}"
        return out

    return None


def family_of(canon: str | None) -> str | None:
    """Which agentic FAMILIES bucket a canonical name belongs to."""
    if not canon:
        return None
    low = _norm(canon)
    if low.startswith("terminal bench") or low.startswith("terminalbench"):
        return "terminal"
    if "swe" in low:
        return "swe"
    if low.startswith(("browsecomp", "webarena", "webvoyager")):
        return "web"
    if low.startswith(("bfcl", "tau", "mcp", "toolathlon")):
        return "tool"
    if low.startswith(("osworld", "androidworld", "agentworld")):
        return "computer"
    return None


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "Terminal Bench 2.0", "TerminalBench", "Terminal-Bench v2.1",
        "SWE Verified", "SWE-Bench Verified (OpenHands)", "SWE_Bench Pro",
        "τ²-Bench Retail", "TAU-Retail", "BFCLv3", "BFCL_v3_MultiTurn (Acc)",
        "MCPAtlas", "WebArena-Verified", "OSWorld 2.0", "Arena-Hard", "MMLU-Pro",
    ]
    for arg in args:
        c = canonical(arg)
        print(f"{arg!r:38} -> {c!r:34} family={family_of(c)!r}")
