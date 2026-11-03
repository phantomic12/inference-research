#!/usr/bin/env python3
"""Rank the models people actually run, from the quant repos they download.

  quant_popularity.py --fetch            # pull quant repo metadata from the HF API
  quant_popularity.py --build            # rank + join evidence -> joined/quant_popularity.json
  quant_popularity.py --queue 40         # write the review queue for the top unproven models

Why quant downloads and not base-model downloads: a base repo's download counter includes
every fine-tune pipeline, every eval harness, every CI job that pulls weights once. A GGUF,
AWQ, MLX or FP8 repo is only downloaded by someone who intends to run that model on their own
hardware. Summing the quant repos of a base model is therefore a much closer proxy for "models
people actually run" than the base model's own counter, which is dominated by machines.

Two accuracy rules, both learnt the hard way:

1. Attribute to the ROOT model, not the intermediate repo. `SC117/X-abliterated-GGUF` declares
   `base_model:quantized:ISTA-DASLab/X-GGUF`, which declares `base_model:quantized:Qwen/Qwen3.8-
   Flash-Next`. Counting the first hop credits a re-quant of a re-quant to the re-quant.
   The base_model tags are walked up to the first repo that is not itself a quant build,
   capped at MAX_HOPS so a cycle cannot hang the run.

2. Per-level popularity is NOT available. The HF API reports downloads per REPO, not per file,
   so "Q4_K_M is the most-downloaded level" cannot be measured from here. Repo-level numbers
   rank the MODELS; which level inside a repo people take is unknown unless the provider
   publishes per-file figures. Anything else would be invented.

Evidence is joined from the review verdicts on disk (/opt/data/review/*.jsonl), which record
base-vs-quant comparisons read off provider cards. A model with no reviewed quant is queued,
not assumed good.

Outputs:
  /opt/data/quant_pop/quant_popularity.json  ranked models with formats, downloads, evidence
                                            (--publish also drops it in site/public/joined/)
  /opt/data/quant_pop/review_queue.json      models to review next, most-run first
  /opt/data/quant_pop/raw_quants.json        cache of the API pull (re-run --fetch to refresh)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "public" / "joined"
CACHE = pathlib.Path("/opt/data/quant_pop")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) inference-research quant survey"}
MAX_HOPS = 4
REVIEW_GLOBS = ("done_*.jsonl", "ldone_*.jsonl", "ddone_*.jsonl", "wdone_*.jsonl", "jdone_*.jsonl")

# HF tag filters, one API sweep each. `gguf` alone is not enough: the AWQ/GPTQ/MLX/FP8 builds
# are what vLLM/SGLang/MLX users run, and they do not carry the gguf tag.
FORMAT_FILTERS = ["gguf", "awq", "gptq", "mlx", "fp8", "exl3", "nvfp4", "int4", "auto-round"]
FORMAT_OF = [
    ("gguf", "GGUF"), ("awq", "AWQ"), ("gptq", "GPTQ"), ("mlx", "MLX"), ("fp8", "FP8"),
    ("exl3", "EXL3"), ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("w4a16", "W4A16"),
    ("int4", "INT4"), ("int8", "INT8"), ("bnb", "BNB"), ("bitsandbytes", "BNB"),
    ("autoround", "AutoRound"), ("gptqmodel", "GPTQModel"),
]
QUANT_NAME_HINT = re.compile(
    r"(gguf|awq|gptq|mlx|fp8|fp4|nvfp4|mxfp4|w4a16|int4|int8|bnb|bitsandbytes|autoround|"
    r"exl[23]|q[2-8]_k|q[2-8]_0|iq[1-4]|abliterat|quantiz|4bit|8bit|6bit|5bit|3bit|2bit)", re.I)


def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers.get("Link")


def fetch_all(limit_per_filter: int, pages: int) -> dict[str, dict]:
    """Quant repo metadata by id. One sweep per format filter, following Link rel=next."""
    seen: dict[str, dict] = {}
    for filt in FORMAT_FILTERS:
        url = (f"https://huggingface.co/api/models?filter={filt}&sort=downloads"
               f"&direction=-1&limit={limit_per_filter}&full=true")
        for page in range(pages):
            try:
                data, link = _get(url)
            except Exception as e:
                print(f"  {filt}: page {page} failed: {str(e)[:90]}", file=sys.stderr)
                break
            for m in data:
                mid = m.get("id")
                if not mid:
                    continue
                rec = seen.setdefault(mid, {"id": mid, "downloads": 0, "likes": 0, "tags": [],
                                            "pipeline_tag": None, "filters": []})
                rec["downloads"] = max(rec["downloads"], m.get("downloads") or 0)
                rec["likes"] = max(rec["likes"], m.get("likes") or 0)
                rec["tags"] = rec["tags"] or (m.get("tags") or [])
                rec["pipeline_tag"] = rec["pipeline_tag"] or m.get("pipeline_tag")
                if filt not in rec["filters"]:
                    rec["filters"].append(filt)
            if not link or 'rel="next"' not in link:
                break
            m2 = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if not m2:
                break
            url = m2.group(1)
            time.sleep(0.2)
        print(f"  {filt}: {len(seen)} repos so far", file=sys.stderr)
    return seen


def base_of(rec: dict) -> str | None:
    """The declared base model, preferring the explicit 'quantized:' relation."""
    tags = rec.get("tags") or []
    quant = [t.split("base_model:quantized:", 1)[1] for t in tags if t.startswith("base_model:quantized:")]
    if quant:
        return quant[0]
    plain = [t.split("base_model:", 1)[1] for t in tags
             if t.startswith("base_model:") and not t.startswith("base_model:quantized:")]
    plain = [p for p in plain if not p.startswith(("adapter:", "merge:", "finetune:"))]
    return plain[0] if plain else None


def format_of(rec: dict) -> str:
    ident = rec["id"].lower()
    tags = " ".join(rec.get("tags") or []).lower()
    for key, label in FORMAT_OF:
        if key in ident or key in tags or key in (rec.get("filters") or []):
            return label
    return "other"


def looks_like_quant(repo_id: str, rec: dict | None) -> bool:
    if rec is not None:
        return bool(set(rec.get("filters") or []) & set(FORMAT_FILTERS))
    base = repo_id.rsplit("/", 1)[-1]
    return bool(QUANT_NAME_HINT.search(base))


def walk_root(repo_id: str, all_repos: dict[str, dict], extra: dict[str, str]) -> tuple[str, int]:
    """Follow base_model up to the first repo that is not itself a quant build."""
    cur, hops = repo_id, 0
    while hops < MAX_HOPS:
        parent = extra.get(cur) or (base_of(all_repos[cur]) if cur in all_repos else None)
        if not parent:
            return cur, hops
        if not looks_like_quant(parent, all_repos.get(parent)) and parent not in extra:
            return parent, hops + 1
        cur, hops = parent, hops + 1
    return cur, hops


HF_URL = re.compile(r"huggingface\.co/(?!api|datasets|spaces|docs|blog)([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)")


def load_catalog() -> dict[str, dict]:
    """Catalog records keyed by the Hugging Face repo they cite.

    Model ids in this repo are slugs (`qwen3-8b`), not HF ids (`Qwen/Qwen3-8B`), so the join
    goes through the source records: each model lists sources, and the HF one carries the repo
    URL. Name equality is only a fallback, because two labs ship same-named repos.
    """
    sources: dict[str, str] = {}
    for f in (ROOT / "data" / "sources").glob("*.json"):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = s.get("url")
        if isinstance(url, str):
            m = HF_URL.search(url)
            if m:
                sources[s.get("id") or f.stem] = m.group(1).lower()

    out: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for f in sorted((ROOT / "data" / "models").glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict) or not r.get("id"):
            continue
        for sid in (r.get("sources") or []):
            hf = sources.get(sid)
            if hf:
                out.setdefault(hf, r)
        by_name[(r.get("name") or "").lower()] = r
    out["__by_name__"] = by_name  # type: ignore[assignment]
    return out


def load_evidence() -> dict[str, list[dict]]:
    """Review verdicts keyed by repo id, from every batch file on disk."""
    out: dict[str, list[dict]] = defaultdict(list)
    for pat in REVIEW_GLOBS:
        for f in pathlib.Path("/opt/data/review").glob(pat):
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rid = r.get("id") or r.get("repo")
                if rid:
                    out[rid.lower()].append(r)
    return out


# What counts as quality evidence. Everything else a card publishes — refusal batteries,
# throughput, context ceilings, file sizes — is not quant damage and must not enter the average:
# a refusal row going 100 -> 1 is the abliteration working, and averaging it in produced a
# "9900% retention" on the first run of this tool.
_LOWER_BETTER = re.compile(r"\b(ppl|perplex\w*|kld?|kl divergence|wer|cer)\b", re.I)
_NOT_QUALITY = re.compile(r"(refus|jailbreak|harmful|safety|uncensor|abliterat|heretic|"
                          r"tok/s|tokens/s|t/s|throughput|context|latency|ttft|tpot|"
                          r"\bgib\b|\bgbs?\b|size|memory|vram|params)", re.I)
_INTERVENTION = re.compile(r"(abliterat|heretic|obliterat|uncensor|decensor|refusal-?free)", re.I)


def quality_rows(rec: dict) -> list[dict]:
    keep = []
    for r in (rec.get("rows") or []):
        bench = str(r.get("bench") or "")
        if _NOT_QUALITY.search(bench):
            continue
        try:
            base, quant = float(r["base"]), float(r["quant"])
        except Exception:
            continue
        if not base:
            continue
        keep.append({**r, "bench": bench, "base": base, "quant": quant})
    return keep


def retention(rec: dict) -> float | None:
    """Mean quality retention across a verdict's quality rows, 100 = no loss.

    Recomputes from the raw base/quant cells rather than trusting a stored delta: batches
    disagree on sign convention, and a wrong sign here would invert the whole ranking. For a
    lower-is-better metric the ratio is inverted so that below 100 always means damage.
    """
    vals = []
    for r in quality_rows(rec):
        base, quant = r["base"], r["quant"]
        if _LOWER_BETTER.search(r["bench"]):
            if quant <= 0:
                continue
            vals.append(base / quant * 100.0)
        else:
            vals.append(quant / base * 100.0)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def bit_bucket(bits) -> float | None:
    try:
        b = float(bits)
    except Exception:
        return None
    return b if b > 0 else None


DOMAIN_OK = {"text-generation", "image-text-to-text", "any-to-any", "text2text-generation", None}
# a repo with no pipeline_tag still declares what it is in its tags; audio/video/vision encoders
# ship "GGUF" too and would otherwise crowd a table about text models
OFF_DOMAIN_TAGS = {"text-to-audio", "text-to-speech", "automatic-speech-recognition", "audio",
                   "text-to-image", "image-to-image", "image-to-video", "text-to-video",
                   "sentence-similarity", "feature-extraction", "text-ranking", "fill-mask",
                   "audio-classification", "voice-activity-detection", "music-generation"}


def _off_domain(m: dict) -> bool:
    tags = {str(x).lower() for x in (m.get("tags_all") or [])}
    if not tags:
        return False
    return bool(tags & OFF_DOMAIN_TAGS) and "text-generation" not in tags


def build() -> dict:
    cache = CACHE / "raw_quants.json"
    all_repos = {r["id"]: r for r in json.loads(cache.read_text())}
    extra = {r["id"]: r["base"] for r in json.loads(cache.read_text()) if r.get("base")}
    catalog = load_catalog()
    by_name = catalog.pop("__by_name__", {})
    evidence = load_evidence()

    by_model: dict[str, dict] = {}
    unattributed = 0
    for rid, rec in all_repos.items():
        root, hops = walk_root(rid, all_repos, extra)
        if root == rid:
            unattributed += 1
            continue
        m = by_model.setdefault(root, {"id": root, "downloads": 0, "likes": 0, "repos": [],
                                       "formats": set(), "max_hops": 0, "pipeline_tag": None,
                                       "tags_all": set()})
        m["tags_all"] |= {str(x).lower() for x in (rec.get("tags") or [])}
        m["downloads"] += int(rec.get("downloads") or 0)
        m["likes"] += int(rec.get("likes") or 0)
        m["formats"].add(format_of(rec))
        m["max_hops"] = max(m["max_hops"], hops)
        m["pipeline_tag"] = m["pipeline_tag"] or rec.get("pipeline_tag")
        m["repos"].append({"id": rid, "downloads": int(rec.get("downloads") or 0),
                           "likes": int(rec.get("likes") or 0), "format": format_of(rec),
                           "hops": hops,
                           # a lab publishing the FP8 of its own model is the reference
                           # deployment, not a third-party quant of it; the distinction changes
                           # how "100% retention" should be read
                           "official": rid.split("/")[0].lower() == root.split("/")[0].lower()})

    rows = []
    dropped_domain = 0
    for mid, m in by_model.items():
        m["repos"].sort(key=lambda r: -r["downloads"])
        cat = catalog.get(mid.lower()) or by_name.get(mid.rsplit("/", 1)[-1].lower())
        # an audio/video/image model also ships "GGUF" repos and would otherwise sit in a table
        # about models people run for text; keep the LLM/VLM domains only
        if cat is None and ((m["pipeline_tag"] or None) not in DOMAIN_OK or _off_domain(m)):
            dropped_domain += 1
            continue
        reviewed, best = [], None
        levels: dict[float, dict] = {}
        cards_read = 0
        for repo in m["repos"]:
            for v in evidence.get(repo["id"].lower(), []):
                # a verdict with no base column still means the card was READ. Re-queuing it
                # spends a second review to learn the same "no numbers published".
                cards_read += 1
                if v.get("ref") != "base":
                    continue
                qrows = quality_rows(v)
                if not qrows:
                    continue
                ret = retention(v)
                intervention = bool(_INTERVENTION.search(repo["id"]) or _INTERVENTION.search(str(v.get("verdict") or "")))
                entry = {"repo": repo["id"], "format": v.get("format"), "bits": v.get("bits"),
                         "official": repo.get("official"),
                         "retention": ret, "quality_rows": len(qrows), "rows": len(v["rows"]),
                         "base": v.get("base"), "verdict": v.get("verdict"),
                         "intervention": intervention,
                         # a build that beats its own base is a card-side artifact (pooled rows,
                         # a distilled teacher, a ±pp gate), not a quant doing the impossible
                         "exceeds_base": bool(ret is not None and ret > 102.0)}
                reviewed.append(entry)
                if intervention or entry["exceeds_base"] or ret is None:
                    continue
                b = bit_bucket(v.get("bits"))
                if b is not None and (b not in levels or ret > (levels[b].get("retention") or 0)):
                    levels[b] = {"bits": b, "retention": ret, "format": v.get("format"),
                                 "repo": repo["id"], "verdict": v.get("verdict")}
                if best is None or ret > (best.get("retention") or 0):
                    best = entry
        rows.append({
            "id": mid,
            "name": (cat or {}).get("name") or mid,
            "in_catalog": bool(cat),
            "catalog_id": (cat or {}).get("id"),
            "pipeline_tag": m["pipeline_tag"],
            "quant_downloads": m["downloads"],
            "quant_likes": m["likes"],
            "quant_repo_count": len(m["repos"]),
            "official_downloads": sum(x["downloads"] for x in m["repos"] if x.get("official")),
            "official_repos": sum(1 for x in m["repos"] if x.get("official")),
            # the choice signal: what people picked instead of the lab's own build
            "community_downloads": sum(x["downloads"] for x in m["repos"] if not x.get("official")),
            "formats": sorted(m["formats"]),
            "top_repo": m["repos"][0] if m["repos"] else None,
            "repos": m["repos"][:12],
            "derivation_hops": m["max_hops"],
            "reviewed": cards_read,
            "with_base_evidence": len({e["repo"] for e in reviewed}),
            "best_evidence": best,
            "evidence": reviewed[:8],
            # the answer to "what is the most performant quant at a given size" for this model
            "levels": [levels[b] for b in sorted(levels)],
            "best_at_4bit": min(
                (l for l in levels.values() if l["bits"] <= 4.5),
                key=lambda l: -l["retention"], default=None),
        })

    rows.sort(key=lambda r: -r["quant_downloads"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "generated": time.strftime("%Y-%m-%d"),
        "method": {
            "popularity": "Sum of HF download counts over the quant repos a base model has, walked "
                          "up from each quant repo's base_model tag to the first non-quant ancestor. "
                          "Base-repo download counters are not used: they count every automated pull, "
                          "while a quant repo is downloaded by someone intending to run it.",
            "levels": "Per-level (Q4_K_M vs Q5_K_M) popularity is not measurable: the API reports "
                      "downloads per repo, not per file. Repo numbers rank models; which level is "
                      "most used is unknown unless a provider publishes per-file figures.",
            "evidence": "Quality comes only from reviewed verdicts that carry an explicit base column "
                        "and at least one numeric row. Retention is recomputed from the raw base/quant "
                        "cells with lower-is-better metrics flipped, never read from a stored delta.",
            "caveat": "Downloads are lifetime totals and are not deduplicated across mirrors, so they "
                      "rank interest, not installs. A model can rank high on one popular repo.",
        },
        "counts": {"models": len(rows), "repos": len(all_repos), "unattributed": unattributed,
                   "dropped_non_llm": dropped_domain,
                   "with_evidence": sum(1 for r in rows if r["with_base_evidence"]),
                   "cards_read": sum(1 for r in rows if r["reviewed"]),
                   "in_catalog": sum(1 for r in rows if r["in_catalog"])},
        "rows": rows,
    }


def write_queue(data: dict, n: int) -> dict:
    """The top models by runs that have no base-vs-quant evidence yet."""
    queue = [r for r in data["rows"] if r["reviewed"] == 0][:n]
    payload = {
        "generated": data["generated"],
        "why": "Ranked by quant-repo downloads: the models people actually run. None of these has a "
               "reviewed base-vs-quant comparison yet, so the next review batch starts here.",
        "entries": [{"rank": r["rank"], "model": r["id"], "catalog_id": r["catalog_id"],
                     "quant_downloads": r["quant_downloads"], "formats": r["formats"],
                     "repos_to_review": [x["id"] for x in r["repos"][:6]],
                     "top_repo": (r["top_repo"] or {}).get("id")} for r in queue],
    }
    (CACHE / "review_queue.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull quant repo metadata from HF")
    ap.add_argument("--build", action="store_true", help="build the joined view from the cache")
    ap.add_argument("--queue", type=int, default=0, metavar="N", help="write a review queue of N models")
    ap.add_argument("--limit", type=int, default=1000, help="repos per format sweep")
    ap.add_argument("--pages", type=int, default=2, help="pages per format sweep")
    ap.add_argument("--publish", action="store_true",
                    help="also write site/public/joined/quant_popularity.json (no page reads it yet)")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    if a.fetch:
        print("Fetching quant repos from the HF API...", file=sys.stderr)
        repos = fetch_all(a.limit, a.pages)
        payload = [{"id": r["id"], "downloads": r["downloads"], "likes": r["likes"],
                    "pipeline_tag": r["pipeline_tag"], "tags": r["tags"],
                    "filters": r["filters"], "base": base_of(r)} for r in repos.values()]
        (CACHE / "raw_quants.json").write_text(json.dumps(payload, ensure_ascii=False))
        print(f"  {len(payload)} quant repos cached -> {CACHE/'raw_quants.json'}", file=sys.stderr)

    if not (CACHE / "raw_quants.json").exists():
        sys.exit("no cache: run --fetch first")

    data = build()
    for entry in data["rows"]:
        for e in entry.get("evidence") or []:
            if e.get("verdict"):
                e["verdict"] = e["verdict"][:400]
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "quant_popularity.json").write_text(blob, encoding="utf-8")
    if a.publish:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "quant_popularity.json").write_text(blob, encoding="utf-8")
    c = data["counts"]
    print(f"models {c['models']} | repos {c['repos']} | unattributed {c['unattributed']} | "
          f"with evidence {c['with_evidence']} | in catalog {c['in_catalog']}")
    for r in data["rows"][:10]:
        print(f"  {r['rank']:2d}. {r['id'][:48]:<48} {r['quant_downloads']:>12,}  "
              f"{'/'.join(r['formats'])[:22]:<22} evidence={r['reviewed']}")

    if a.queue:
        q = write_queue(data, a.queue)
        print(f"\nqueue: {len(q['entries'])} models -> {CACHE/'review_queue.json'}")


if __name__ == "__main__":
    main()
