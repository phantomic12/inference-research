#!/usr/bin/env python3
"""Build the static-site data payload from data/ and docs/.

This is the site's only data source. It reads data/**/*.json, resolves every
bare-slug cross-reference into a qualified record id, computes back-references,
rewrites the docs/ wikilinks into site-relative links, and emits a client-side
search index into site/src/data/.

    python tools/build_site_data.py            # writes site/src/data/
    python tools/build_site_data.py --check    # report problems, write nothing

Markdown -> HTML happens in the Astro build with `marked`; this script only
preprocesses the link syntax, so it needs no third-party Python packages.

Does NOT touch data/, schemas/, or tools/index.py, and never regenerates
docs/00-index.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DIRS, REF_FIELDS, TITLES, TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "src" / "data"

# ---------------------------------------------------------------------------
# schema knowledge: which fields hold bare record slugs, and how each type's
# fields read. Field choices and the summarise() logic follow tools/index.py's
# card() renderer, which already encodes the right per-type summary.
# ---------------------------------------------------------------------------

# Fields whose values are bare record slugs, and the type they should point at,
# are REF_FIELDS above: imported from registry.py, which is the single definition
# shared with tools/validate.py. This tool computed 110 dangling references that
# validate.py never checked, and a second copy of this table is how that
# divergence happened in the first place.

REF_LABELS = {
    "sources": "Sources",
    "interconnect": "Interconnect",
    "engine_id": "Engine",
    "accelerator_ids": "Accelerators",
    "interconnect_ids": "Interconnect",
    "format_id": "Format",
    "affected_by_hardware": "Hardware-dependent",
    "affects": "Affects",
    "hardware_relevance": "Hardware relevance",
    "native_support": "Native support",
    "emulated_support": "Emulated support",
}

FIELD_META: dict[str, list[tuple[str, str]]] = {
    "accelerator": [
        ("vendor", "Vendor"), ("architecture", "Architecture"),
        ("release_year", "Release year"), ("process_nm", "Process (nm)"),
        ("form_factors", "Form factors"), ("vram_gb", "VRAM (GB)"),
        ("memory_type", "Memory type"), ("memory_bus_bit", "Memory bus (bit)"),
        ("memory_bandwidth_gbps", "Memory bandwidth (GB/s)"),
        ("memory_bandwidth_basis", "Bandwidth basis"),
        ("flops", "Peak FLOPS"), ("tdp_w", "TDP (W)"),
        ("interconnect", "Interconnect"), ("unified_memory", "Unified memory"),
        ("consumer", "Consumer part"),
    ],
    "flop": [
        ("class", "Class"), ("bound_by", "Bound by"),
        ("arithmetic_intensity", "Arithmetic intensity"),
        ("scales_with", "Scales with"),
        ("affected_by_hardware", "Hardware-dependent"),
        ("workarounds", "Workarounds"),
    ],
    "engine": [
        ("repo", "Repo"), ("languages", "Languages"), ("license", "License"),
        ("first_release_year", "First release"), ("design_goal", "Design goal"),
        ("scheduling", "Scheduling"), ("supported_backends", "Backends"),
        ("supported_formats", "Formats"), ("notable_features", "Notable features"),
        ("hardware_caveats", "Hardware caveats"), ("best_for", "Best for"),
        ("avoid_for", "Avoid for"), ("docs_url", "Docs"),
    ],
    "quantization": [
        ("scheme", "Scheme"), ("bits", "Bits"),
        ("weight_group_size", "Weight group size"),
        ("activation_scheme", "Activation scheme"),
        ("native_support", "Native support"), ("emulated_support", "Emulated support"),
        ("kernels", "Kernels"), ("quality_delta", "Quality delta"),
    ],
    "interconnect": [
        ("kind", "Kind"), ("version", "Version"),
        ("bandwidth_gbps", "Bandwidth (GB/s, per link, unidirectional)"),
        ("bandwidth_basis", "Bandwidth basis"), ("link_count", "Link count"),
        ("topology", "Topology"), ("scale_up", "Scale-up"), ("scale_out", "Scale-out"),
        ("switching", "Switching"),
    ],
    "benchmark": [
        ("metric", "Metric"), ("value", "Value"), ("unit", "Unit"),
        ("model", "Model"), ("engine_id", "Engine"),
        ("accelerator_ids", "Accelerators"), ("interconnect_ids", "Interconnect"),
        ("format_id", "Format"), ("methodology", "Methodology"),
        ("measured_by", "Measured by"), ("reproducible", "Reproducible"),
    ],
    "gotcha": [
        ("class", "Class"), ("severity", "Severity"), ("affects", "Affects"),
        ("concepts", "Concepts"), ("symptom", "Symptom"),
        ("root_cause", "Root cause"), ("workaround", "Workaround"),
    ],
    "supply": [
        ("kind", "Kind"), ("vendor", "Vendor"), ("region", "Region"),
        ("accelerator_ids", "Accelerators"), ("channels", "Channels"),
        ("price_usd", "Price (USD)"), ("price_basis", "Price basis"),
        ("availability", "Availability"), ("lead_time_weeks", "Lead time (weeks)"),
        ("export_controlled", "Export controlled"),
    ],
    "model": [
        ("family", "Family"), ("vendor", "Vendor"), ("release_year", "Release year"),
        ("architecture", "Architecture"), ("params_b", "Params (B)"),
        ("active_params_b", "Active params (B)"), ("num_experts", "Experts"),
        ("experts_per_token", "Experts per token"),
        ("shared_expert", "Shared expert"), ("hidden_size", "Hidden size"),
        ("num_layers", "Layers"), ("num_kv_heads", "KV heads"),
        ("head_dim", "Head dim"), ("gqa_ratio", "GQA ratio"),
        ("max_position_embeddings", "Max context"),
        ("context_scaling", "Context scaling"), ("activation", "Activation"),
        ("attention_variant", "Attention variant"),
        ("kv_cache_bytes_per_token", "KV cache (bytes/token)"),
        ("flops_per_token_active", "FLOPs per active token"),
        ("open_weights", "Open weights"), ("license", "License"),
    ],
    "paper": [
        ("arxiv_id", "arXiv"), ("venue", "Venue"), ("year", "Year"),
        ("category", "Category"), ("adoption", "Adoption"),
        ("authors", "Authors"), ("affiliations", "Affiliations"),
        ("problem", "Problem"), ("mechanism", "Mechanism"),
        ("speedup_reported", "Speedup reported"),
        ("hardware_relevance", "Hardware relevance"),
        ("code_url", "Code"), ("open_weights", "Open weights"),
    ],
    "source": [
        ("url", "URL"), ("publisher", "Publisher"), ("kind", "Kind"),
        ("published", "Published"), ("accessed", "Accessed"),
        ("archived_url", "Archived URL"),
    ],
}

FACETS: dict[str, list[str]] = {
    "accelerator": ["vendor", "architecture", "release_year"],
    "flop": ["class", "bound_by"],
    "engine": ["license"],
    "quantization": ["scheme"],
    "interconnect": ["kind"],
    "benchmark": ["metric", "measured_by"],
    "gotcha": ["class", "severity"],
    "supply": ["kind", "availability"],
    "model": ["architecture", "family", "vendor"],
    "paper": ["category", "venue", "adoption", "year"],
    "source": ["kind", "publisher"],
}

# Long-form fields: render as prose, not as a wrapped scalar.
PROSE = {
    "notes", "arithmetic_intensity", "best_for", "avoid_for", "hardware_caveats",
    "notable_features", "symptom", "root_cause", "workaround", "quality_delta",
    "memory_bandwidth_basis", "problem", "mechanism", "speedup_reported",
    "topology", "switching", "methodology", "price_basis", "bandwidth_basis",
}
# Fields holding a single URL.
URL_FIELDS = {"url", "archived_url", "docs_url", "code_url"}

COMMON = ("id", "type", "name", "status", "confidence", "updated")
STATUS_ORDER = ["contested", "draft", "verified", "deprecated"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_records() -> tuple[dict[str, dict], list[dict]]:
    """Return ({qualified id -> record}, [problem])."""
    records: dict[str, dict] = {}
    problems: list[dict] = []
    for t in TYPES:
        d = ROOT / "data" / DIRS[t]
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            rid = f"{DIRS[t]}/{f.stem}"
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append({"rid": rid, "level": "ERROR", "msg": f"unparseable: {exc}"})
                continue
            if not isinstance(rec, dict):
                problems.append({"rid": rid, "level": "ERROR", "msg": "not a JSON object"})
                continue
            if rec.get("type") != t:
                problems.append({"rid": rid, "level": "warn",
                                 "msg": f"type says {rec.get('type')!r}, lives in data/{DIRS[t]}/"})
            if rec.get("id") != f.stem:
                problems.append({"rid": rid, "level": "warn",
                                 "msg": f"id says {rec.get('id')!r}, filename says {f.stem!r}"})
            if not rec.get("name"):
                problems.append({"rid": rid, "level": "warn", "msg": "no name"})
            if rec.get("status") not in ("verified", "draft", "contested", "deprecated"):
                problems.append({"rid": rid, "level": "warn",
                                 "msg": f"unexpected status {rec.get('status')!r}"})
            conf = rec.get("confidence")
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
                problems.append({"rid": rid, "level": "warn", "msg": f"bad confidence {conf!r}"})
            for s in rec.get("sources") or []:
                if not isinstance(s, str):
                    problems.append({"rid": rid, "level": "warn", "msg": f"non-string source {s!r}"})
            rec["_rid"] = rid
            rec["_type"] = t
            records[rid] = rec
    return records, problems


def build_bare(records: dict[str, dict]) -> dict[str, str]:
    """Bare slug -> qualified record id, for resolving cross-reference fields."""
    bare: dict[str, str] = {}
    for rid, rec in records.items():
        stem = rid.split("/", 1)[1]
        bare.setdefault(stem, rid)
        if isinstance(rec.get("id"), str) and rec["id"]:
            bare.setdefault(rec["id"], rid)
    return bare


# ---------------------------------------------------------------------------
# summary line — mirrors tools/index.py card()
# ---------------------------------------------------------------------------

def _s(v) -> str:
    return "" if v is None else str(v)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clip(v, n: int) -> str:
    s = "" if v is None else str(v)
    return (s[:n] + "…") if len(s) > n else s


def summarise(rec: dict) -> dict:
    t = rec["_type"]
    g = rec.get
    head = ""
    tail: list[str] = []
    facets: dict[str, str] = {}

    if t == "accelerator":
        head = " ".join(str(x) for x in (g("vendor"), g("architecture"), g("release_year")) if x) or "unspecified"
        if _num(g("vram_gb")):
            tail.append(f"{g('vram_gb')} GB {g('memory_type') or ''}".strip())
        if _num(g("memory_bandwidth_gbps")):
            tail.append(f"{g('memory_bandwidth_gbps')} GB/s")
        if _num(g("tdp_w")):
            tail.append(f"{g('tdp_w')} W")
        facets = {"vendor": _s(g("vendor")), "architecture": _s(g("architecture")),
                  "release_year": _s(g("release_year"))}
    elif t == "flop":
        head = f"{g('bound_by')} bound"
        tail = [_clip(g("arithmetic_intensity"), 280)]
        facets = {"class": _s(g("class")), "bound_by": _s(g("bound_by"))}
    elif t == "engine":
        head = ", ".join(g("supported_backends") or []) or "no backends listed"
        tail = [_clip("; ".join(g("best_for") or []), 280)]
        facets = {"license": _s(g("license"))}
    elif t == "quantization":
        head = _s(g("bits")) or "?"
        tail = [_clip(g("quality_delta"), 240)]
        facets = {"scheme": _s(g("scheme"))}
    elif t == "interconnect":
        head = f"{g('bandwidth_gbps')} GB/s"
        tail = [_clip(g("topology"), 280)]
        facets = {"kind": _s(g("kind"))}
    elif t == "benchmark":
        head = f"{g('value')} {g('unit') or ''}".strip()
        tail = [f"{g('model') or '?'} on {', '.join(g('accelerator_ids') or []) or '?'}"
                f" via {g('engine_id') or '?'}"]
        facets = {"metric": _s(g("metric")), "measured_by": _s(g("measured_by"))}
    elif t == "gotcha":
        head = f"[{g('severity')}] {g('class')}"
        tail = [_clip(g("symptom"), 280)]
        facets = {"class": _s(g("class")), "severity": _s(g("severity"))}
    elif t == "supply":
        bits = [str(x) for x in (g("vendor"), g("region")) if x]
        head = " · ".join(bits) or _s(g("kind"))
        price = g("price_usd")
        tail = [f"${price:,.0f}" if _num(price) else "", _s(g("availability"))]
        tail = [x for x in tail if x]
        if g("price_basis"):
            tail.append(_clip(g("price_basis"), 80))
        facets = {"kind": _s(g("kind")), "availability": _s(g("availability"))}
    elif t == "model":
        arch = g("architecture") or "?"
        tot, act = g("params_b"), g("active_params_b")
        head = str(arch)
        if arch == "moe" and _num(tot) and _num(act):
            head += f" {act:g}/{tot:g}B active"
        elif _num(tot):
            head += f" {tot:g}B"
        if _num(g("kv_cache_bytes_per_token")):
            tail.append(f"{g('kv_cache_bytes_per_token'):,.0f} B/token KV")
        if g("attention_variant"):
            tail.append(str(g("attention_variant")))
        if _num(g("max_position_embeddings")):
            tail.append(f"{int(g('max_position_embeddings')):,} ctx")
        facets = {"architecture": str(arch), "family": _s(g("family")), "vendor": _s(g("vendor"))}
    elif t == "paper":
        venue = _s(g("venue")).replace("-preprint", "") or "?"
        arx = f"arXiv:{g('arxiv_id')}" if g("arxiv_id") else "no arXiv"
        head = f"{g('category') or '?'} · {venue}"
        tail = [x for x in [_s(g("year")), arx] if x]
        if g("adoption"):
            tail.append(str(g("adoption")))
        if g("problem"):
            tail.append(_clip(g("problem"), 200))
        facets = {"category": _s(g("category")), "venue": _s(g("venue")),
                  "adoption": _s(g("adoption")), "year": _s(g("year"))}
    else:
        head = _s(g("kind"))
        tail = [_s(g("publisher"))]
        facets = {"kind": _s(g("kind")), "publisher": _s(g("publisher"))}

    return {"head": head, "tail": [x for x in tail if x], "facets": facets}


def ordered_fields(rec: dict) -> list[tuple[str, str]]:
    """(key, label) in display order; common fields and sources are excluded."""
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for k, label in FIELD_META.get(rec["_type"], []):
        if k in rec:
            ordered.append((k, label))
            seen.add(k)
    for k in sorted(rec):
        if k in COMMON or k in seen or k.startswith("_") or k == "sources":
            continue
        ordered.append((k, k.replace("_", " ")))
    return ordered


# ---------------------------------------------------------------------------
# cross-references
# ---------------------------------------------------------------------------

def build_refs(records: dict[str, dict], bare: dict[str, str]):
    outgoing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    dangling: list[dict] = []

    for rid, rec in records.items():
        for s in rec.get("sources") or []:
            if not isinstance(s, str):
                continue
            target = bare.get(s)
            if target:
                outgoing[rid]["sources"].append(target)
                incoming[target].append((rid, "sources"))
            else:
                dangling.append({"from": rid, "field": "sources", "value": s})
        for field in REF_FIELDS.get(rec["_type"], {}):
            v = rec.get(field)
            vals = [x for x in v if isinstance(x, str)] if isinstance(v, list) else \
                   ([v] if isinstance(v, str) and v else [])
            for x in vals:
                target = bare.get(x)
                if target:
                    outgoing[rid][field].append(target)
                    incoming[target].append((rid, field))
                else:
                    dangling.append({"from": rid, "field": field, "value": x})
    return {k: dict(v) for k, v in outgoing.items()}, incoming, dangling


# ---------------------------------------------------------------------------
# docs -> link-rewritten markdown (rendered to HTML in the Astro build)
# ---------------------------------------------------------------------------

# [[target]] or [[target|label]]. The target may itself contain slashes
# ([[accelerators/nvidia-h100-sxm]]), a doc stem ([[05-known-traps]]) or a bare
# record slug, so the target group is matched loosely and resolved by lookup
# rather than by pattern.
WIKILINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\]]*))?\]\]")
DOC_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:\.{1,2}/)*(?:docs/)?([A-Za-z0-9._-]+\.md)(#[^)]*)?\)")


def rewrite_docs(text: str, bare: dict[str, str], records: dict[str, dict],
                 doc_slugs: set[str]) -> str:
    """Resolve the doc wikilink grammar into site URLs.

    The docs use several forms, all of which have to land somewhere real:
      [[type/slug]]          typed record id
      [[type/slug|label]]    same, with display text
      [[05-known-traps]]     another doc
      [[slug|label]]         bare record slug, type inferred
    Anything that resolves to neither stays as inline code, so an unresolved
    reference is visible as such instead of becoming a broken link.
    """
    def wl(m: re.Match) -> str:
        target_txt, label = m.group(1), m.group(2)
        display = label or None

        if target_txt in doc_slugs:
            text_ = label or next(
                (d for d in doc_slugs if d == target_txt), target_txt)
            return f"[{text_}](/docs/{target_txt}/)"

        rid: str | None = None
        if "/" in target_txt:
            dir_, stem = target_txt.split("/", 1)
            if dir_ == "data":
                dir_, stem = "sources" if stem.startswith("sources/") else "", stem
            stem = stem[:-5] if stem.endswith(".json") else stem
            rid = bare.get(stem)
            if rid is None:
                cand = f"{dir_}/{stem}"
                rid = cand if cand in records else None
        else:
            rid = bare.get(target_txt)

        if rid and rid in records:
            name = display or str(records[rid].get("name") or rid)
            name = name.replace("[", "(").replace("]", ")")
            return f"[{name}](/r/{rid}/)"
        return f"`{label or target_txt}`"

    body = WIKILINK.sub(wl, text)

    def doc(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        stem = target[:-3]
        if stem in doc_slugs:
            return f"[{label}](/docs/{stem}/)"
        return f"`{label or target}`" if not label else label

    return DOC_MD_LINK.sub(doc, body)


def load_docs(records: dict[str, dict], bare: dict[str, str]) -> list[dict]:
    d = ROOT / "docs"
    files = [f for f in sorted(d.glob("*.md")) if f.name != "00-index.md"]
    doc_slugs = {f.stem for f in files}
    out = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+?)\s*$", raw, re.M)
        out.append({
            "slug": f.stem,
            "title": (m.group(1).strip() if m else f.stem),
            "order": int(f.stem[:2]) if f.stem[:2].isdigit() else 99,
            "chars": len(raw),
            "markdown": rewrite_docs(raw, bare, records, doc_slugs),
        })
    out.sort(key=lambda x: (x["order"], x["slug"]))
    return out


# ---------------------------------------------------------------------------
# client-side search index
# ---------------------------------------------------------------------------

STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on",
    "with", "that", "this", "it", "as", "by", "at", "from", "be", "was", "were",
    "not", "but", "which", "than", "so", "if", "can", "will", "does", "do",
}
TOKEN = re.compile(r"[a-z0-9][a-z0-9._+-]*")

# Searched fields. The brief requires name + notes + mechanism + problem; the
# enum facets are included because they are what a reader actually types.
SEARCH_FIELDS = [
    "problem", "mechanism", "symptom", "root_cause", "workaround",
    "best_for", "avoid_for", "hardware_caveats", "notable_features",
    "arithmetic_intensity", "quality_delta", "speedup_reported", "methodology",
    "memory_bandwidth_basis", "bandwidth_basis", "price_basis", "topology",
    "switching", "notes", "repo", "architecture", "family", "scheme",
    "attention_variant", "context_scaling", "workarounds", "scales_with",
    "concepts", "design_goal", "scheduling", "supported_backends",
    "supported_formats", "kernels", "channels", "region", "license", "vendor",
    "memory_type", "class", "metric", "kind", "severity", "availability",
    "adoption", "category", "venue", "publisher", "url", "model",
]
NOTE_CAP = 4000  # chars of a field indexed; full text stays on the detail page


def _flatten(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return " ".join(_flatten(x) for x in v)
    if isinstance(v, dict):
        return " ".join(f"{k} {_flatten(val)}" for k, val in v.items())
    return str(v)


def build_search(records: dict[str, dict], docs_out: list[dict]) -> dict:
    """Inverted token index over records and doc pages.

    Two buckets because the two hit differently: a record match is a fact, a
    doc match is a place in the prose. Both are ranked by tf-idf in the client.
    """
    rids = sorted(records)
    docs = []
    tokens: dict[str, list[int]] = defaultdict(list)
    for i, rid in enumerate(rids):
        rec = records[rid]
        s = summarise(rec)
        blob = " ".join(_flatten(rec[f])[:NOTE_CAP] for f in SEARCH_FIELDS if f in rec)
        counts: Counter[str] = Counter()
        for tok in TOKEN.findall(blob.lower()):
            if len(tok) < 2 or tok in STOP or tok.isdigit():
                continue
            counts[tok] += 1
        # name and id are the highest-signal terms, so they get a bonus weight
        for tok in set(TOKEN.findall((str(rec.get("name") or "") + " " + rid).lower())):
            if len(tok) < 2 or tok in STOP or tok.isdigit():
                continue
            counts[tok] += 8
        for tok, _c in counts.items():
            tokens[tok].append(i)
        problem = rec.get("problem")
        docs.append({
            "i": i,
            "r": rid,
            "t": rec["_type"],
            "n": str(rec.get("name") or rid),
            "s": rec.get("status") or "",
            "c": rec.get("confidence"),
            "d": rec.get("updated"),
            "h": s["head"],
            "p": ((problem if isinstance(problem, str) else "") or (s["tail"][0] if s["tail"] else ""))[:260],
            "v": s["facets"],
        })

    doc_slugs = [d["slug"] for d in docs_out]
    for k, slug in enumerate(doc_slugs):
        d = docs_out[k]
        body = WIKILINK.sub(" ", re.sub(r"```.*?```", " ", d["markdown"], flags=re.S))
        counts = Counter(t for t in TOKEN.findall(body.lower())
                         if len(t) >= 2 and t not in STOP and not t.isdigit())
        for tok in set(TOKEN.findall(d["title"].lower())):
            counts[tok] = counts.get(tok, 0) + 6
        for tok in counts:
            tokens[tok].append(100_000 + k)
        docs.append({
            "i": 100_000 + k,
            "r": f"doc:{slug}",
            "t": "doc",
            "n": d["title"],
            "s": "",
            "c": None,
            "d": None,
            "h": f"Synthesis doc · {d['chars']:,} chars",
            "p": "",
            "v": {},
        })
    return {"docs": docs, "tokens": {k: sorted(v) for k, v in sorted(tokens.items())}}


def build_compare(records: dict[str, dict]) -> dict:
    """Per-type compare tables.

    Field values are copied verbatim from the records; the site only aligns them
    into a matrix. Which fields a type has, and in what order, comes from the
    actual record set so the table cannot drift from the data.
    """
    labels: dict[str, dict[str, str]] = {}
    for t in TYPES:
        recs = [r for r in records.values() if r["_type"] == t]
        if not recs:
            labels[t] = {}
            continue
        present = [k for r in recs for k in r
                   if k not in COMMON and k != "sources" and not k.startswith("_")]
        labels[t] = {k: dict(FIELD_META.get(t, [])).get(k, k.replace("_", " "))
                     for k in present}

    def cell(rec: dict, key: str):
        v = rec.get(key)
        if v in (None, [], ""):
            return ""
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, list):
            parts = []
            for x in v:
                if isinstance(x, dict):
                    parts.append(", ".join(f"{_s(k)} {_shortval(val)}" for k, val in x.items()))
                else:
                    parts.append(_s(x))
            return "; ".join(parts)
        if isinstance(v, dict):
            return ", ".join(f"{k}: {_shortval(val)}" for k, val in v.items())
        return _shortval(v)

    out: dict[str, dict] = {}
    for t in TYPES:
        recs = [records[rid] for rid in sorted(records) if records[rid]["_type"] == t]
        out[t] = {
            "labels": labels[t],
            "rows": [{"rid": r["_rid"], "name": str(r.get("name") or r["_rid"]),
                      "status": r.get("status"),
                      "cells": {k: cell(r, k) for k in labels[t]}} for r in recs],
        }
    return out


def _shortval(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report problems, write nothing")
    args = ap.parse_args()

    records, problems = load_records()
    bare = build_bare(records)
    outgoing, incoming, dangling = build_refs(records, bare)
    docs_out = load_docs(records, bare)
    search = build_search(records, docs_out)

    counts = Counter(r["_type"] for r in records.values())
    status_counts = Counter(r.get("status") for r in records.values())
    conf = [float(r["confidence"]) for r in records.values() if _num(r.get("confidence"))]

    slim: dict[str, dict] = {}
    for rid, rec in sorted(records.items()):
        s = summarise(rec)
        ref_fields = REF_FIELDS.get(rec["_type"], {})
        slim[rid] = {
            "rid": rid,
            "type": rec["_type"],
            "name": str(rec.get("name") or rid),
            "status": rec.get("status"),
            "confidence": rec.get("confidence"),
            "updated": rec.get("updated"),
            "head": s["head"],
            "tail": s["tail"],
            "facets": s["facets"],
            "fields": [
                {"key": k, "label": lab, "value": rec.get(k),
                 "prose": k in PROSE, "ref": k in ref_fields, "url": k in URL_FIELDS}
                for k, lab in ordered_fields(rec)
                if rec.get(k) not in (None, [], "")
            ],
            "notes": str(rec.get("notes") or ""),
            "refs": [{"field": f, "label": REF_LABELS.get(f, f.replace("_", " ")), "targets": v}
                     for f, v in sorted(outgoing.get(rid, {}).items())],
            "backrefs": [{"from": src, "via": via} for src, via in sorted(incoming.get(rid, []))],
        }

    meta = {
        "total": len(records),
        "counts": dict(counts),
        "types": [{"type": t, "title": TITLES[t], "dir": DIRS[t], "count": counts.get(t, 0),
                   "primary": t != "source", "facets": FACETS.get(t, [])}
                  for t in TYPES],
        "status": dict(status_counts),
        "status_order": STATUS_ORDER,
        "mean_confidence": round(sum(conf) / len(conf), 3) if conf else None,
        "dangling": dangling,
        "docs": [{"slug": d["slug"], "title": d["title"], "order": d["order"]} for d in docs_out],
    }

    if args.check:
        print(f"records {len(records)}  docs {len(docs_out)}  "
              f"dangling refs {len(dangling)}  data problems {len(problems)}")
        for p in problems[:40]:
            print(f"  {p['level']}: {p['rid']}: {p['msg']}")
        if dangling:
            print("\ndangling references (first 25):")
            for d in dangling[:25]:
                print(f"  {d['from']} . {d['field']} -> {d['value']}")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    def dumps(o) -> str:
        return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

    # One file per record. Astro then bundles only the record a page actually
    # needs, instead of shipping the whole 6 MB corpus as JS to every reader.
    rdir = OUT / "records"
    by_type: dict[str, list[dict]] = defaultdict(list)
    written = 0
    for rid, entry in slim.items():
        path = rdir / f"{rid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(entry), encoding="utf-8")
        written += 1
        by_type[entry["type"]].append({
            "rid": rid, "name": entry["name"], "status": entry["status"],
            "confidence": entry["confidence"], "updated": entry["updated"],
            "head": entry["head"], "tail": entry["tail"], "facets": entry["facets"],
            "nbackrefs": len(entry["backrefs"]),
            "nrefs": sum(len(g["targets"]) for g in entry["refs"]),
        })

    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # id -> {name, type, status}. Small enough to ship to the client, and needed
    # wherever a link is rendered without the full record in hand.
    (OUT / "names.json").write_text(dumps({
        rid: {"n": str(rec.get("name") or rid), "t": rec["_type"], "s": rec.get("status")}
        for rid, rec in sorted(records.items())
    }), encoding="utf-8")
    (OUT / "docs.json").write_text(dumps(docs_out), encoding="utf-8")
    (OUT / "search-index.json").write_text(dumps(search), encoding="utf-8")

    # Compare matrices are written to public/ as well as src/data/, because the
    # compare page fetches them at runtime: only the chosen type's matrix is ever
    # downloaded, so a 3 MB corpus never reaches the browser wholesale.
    public_cmp = ROOT / "site" / "public" / "c"
    public_cmp.mkdir(parents=True, exist_ok=True)
    for t, payload in build_compare(records).items():
        (public_cmp / f"{t}.json").write_text(dumps(payload), encoding="utf-8")
        slimmed = {
            "labels": payload["labels"],
            "rows": [
                {"r": r["rid"], "n": r["name"], "s": r["status"],
                 "c": {k: v for k, v in r["cells"].items() if v}}
                for r in payload["rows"]
            ],
        }
        (public_cmp / f"{t}-slim.json").write_text(dumps(slimmed), encoding="utf-8")

    idir = OUT / "index"
    idir.mkdir(parents=True, exist_ok=True)
    for t in TYPES:
        rows = sorted(by_type.get(t, []), key=lambda r: (r["name"].lower(), r["rid"]))
        (idir / f"{t}.json").write_text(dumps(rows), encoding="utf-8")

    # Tiny list of types with data, for client-side pickers.
    (ROOT / "site" / "public" / "type-list.json").write_text(
        dumps([t for t in TYPES if counts.get(t, 0) > 0]), encoding="utf-8")

    # Search facets. Pagefind's Windows CLI cannot be told to record per-page
    # metadata, so /search/ resolves type and status from this map instead.
    public = ROOT / "site" / "public"
    (public / "facets.json").write_text(dumps({
        rid: {"t": rec["_type"], "s": rec.get("status")}
        for rid, rec in sorted(records.items())
    }), encoding="utf-8")
    # Page titles, so a search hit can be labelled by record name rather than by
    # whatever the <title> element happened to be.
    (public / "titles.json").write_text(dumps({
        rid: str(rec.get("name") or rid) for rid, rec in sorted(records.items())
    }), encoding="utf-8")

    nodes = [{"i": rid, "t": rec["_type"], "n": str(rec.get("name") or rid),
              "s": rec.get("status")} for rid, rec in sorted(records.items())]
    edges = sorted({(rid, tgt, field) for rid, groups in outgoing.items()
                    for field, targets in groups.items() for tgt in targets
                    if tgt != rid and not rid.startswith("sources/")})
    (OUT / "graph.json").write_text(
        dumps({"nodes": nodes, "edges": [list(e) for e in edges]}), encoding="utf-8")

    total = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(f"records {len(records)}  docs {len(docs_out)}  edges {len(edges)}  "
          f"dangling refs {len(dangling)}  data problems {len(problems)}")
    print(f"  {written} per-record files + {len(TYPES)} type indexes, "
          f"{total / 1024 / 1024:.1f} MB total under site/src/data/")
    for name in ("meta.json", "docs.json", "search-index.json", "graph.json"):
        print(f"  site/src/data/{name}  {(OUT / name).stat().st_size / 1024:.0f} KB")
    for p in problems[:20]:
        print(f"  {p['level']}: {p['rid']}: {p['msg']}")


if __name__ == "__main__":
    main()