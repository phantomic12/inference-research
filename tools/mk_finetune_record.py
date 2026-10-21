#!/usr/bin/env python3
"""Create a first-class model record for a fine-tune, merge or abliteration.

The catalog tracks quants and format ports in `data/derivatives.jsonl`, but a
post-trained model that beats its base on a published benchmark is a contender,
not a repack: it has its own behaviour, its own licence and its own numbers. This
tool builds the mechanical half of such a record from Hugging Face - architecture
inherited from the base's config.json, parameter counts from the safetensors
index, licence and dates from the repo metadata - and takes the evidence half
from a JSON file the caller wrote after reading the card.

It will not invent a benchmark row. A record with no benchmarks is refused unless
--allow-empty is passed, and an existing record is never overwritten.

    tools/mk_finetune_record.py org/Repo-Name --bench /tmp/bench.json
    tools/mk_finetune_record.py org/Repo-Name --base Qwen/Qwen3.6-35B-A3B --dry-run

bench.json is a list of rows read off the card:

    [{"name": "SWE-bench-Live", "score": 48.0,
      "config": "Card table, column 'Tiel'; base column 44.2. n=25 single run."}, ...]

Derivative kind is inferred: an 'abliterat' in the name means abliteration, a
'merge' means merge, everything else is a finetune; --kind overrides.
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "data" / "models"
SOURCES = REPO / "data" / "sources"
REPACK = re.compile(r"(-GGUF|-AWQ|-GPTQ|-FP8|-NVFP4|-bnb|-MLX|-EXL[23]|-\d+bpw|I?Q\d_|APEX)", re.I)
TODAY = dt.date.today().isoformat()


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")


def jget(url):
    return json.loads(get(url))


def config_of(repo):
    c = jget("https://huggingface.co/%s/raw/main/config.json" % repo)
    return c.get("text_config") or c


def arch_of(base_repo):
    """Everything the derivative inherits unchanged because it keeps the base's config."""
    cfg = config_of(base_repo)
    raw_cfg = jget("https://huggingface.co/%s/raw/main/config.json" % base_repo)
    layer_types = cfg.get("layer_types") or []
    full_idx = [i for i, x in enumerate(layer_types) if "full" in x]
    n_full = len(full_idx) if full_idx else cfg.get("num_hidden_layers", 0) // (
        cfg.get("full_attention_interval") or 1)
    # Not every config carries head_dim (LFM2 and some hybrid stacks derive it from
    # hidden_size / heads), so fall back rather than crashing on the base.
    heads = cfg.get("num_attention_heads") or 1
    kvh = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or (cfg.get("hidden_size") // heads if heads else None)
    kvb = (n_full * 2 * kvh * hd * 2) if hd else None  # fp16 KV
    return {
        "num_layers": cfg.get("num_hidden_layers"),
        "hidden_size": cfg.get("hidden_size"),
        "num_kv_heads": kvh,
        "head_dim": hd,
        "gqa_ratio": heads // kvh if kvh else None,
        "num_experts": cfg.get("num_experts"),
        "experts_per_token": cfg.get("num_experts_per_tok"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "kv_bytes": kvb,
        "n_full": n_full,
        "full_idx": full_idx,
        "layer_types": ["full_attention" if "full" in x else "linear_attention" for x in layer_types],
        "act": cfg.get("hidden_act"),
        "has_vision": bool(raw_cfg.get("vision_config")),
        "architecture": "moe" if cfg.get("num_experts") else "dense",
    }


def find_base_record(base_repo):
    """Locate the catalog record whose source points at this HF repo."""
    want = base_repo.lower().rstrip("/")
    cand = re.sub(r"[^a-z0-9]+", "-", want.split("/")[-1])
    f = MODELS / (cand + ".json")
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("type") == "model":
            return d, f
    for f in MODELS.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("type") != "model":
            continue
        for sid in d.get("sources") or []:
            sf = SOURCES / (sid + ".json")
            if not sf.exists():
                continue
            url = (json.loads(sf.read_text()).get("url") or "").lower()
            if want in url:
                return d, f
    return None, None


def slug_of(hf_id):
    name = hf_id.split("/")[-1].lower()
    name = REPACK.sub("", name)
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id")
    ap.add_argument("--base", help="base HF repo id; defaults to the card's base_model tag")
    ap.add_argument("--bench", help="JSON file: [{name, score, config}, ...] read off the card")
    ap.add_argument("--kind", choices=["finetune", "merge", "abliteration", "distillation", "rl", "other"])
    ap.add_argument("--task", default="text-generation",
                    choices=["text-generation", "vision-language", "decision", "embedding",
                             "reranking", "speech-to-text", "text-to-speech", "image-generation", "other"],
                    help="decision = per-option probabilities in one forward pass; excluded from "
                         "the worth-running ranking and measured on the Decision Index")
    ap.add_argument("--allow-empty", action="store_true", help="write a record with no benchmarks")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    info = jget("https://huggingface.co/api/models/%s?blobs=false" % a.hf_id)
    tags = info.get("tags") or []
    card = info.get("cardData") or {}
    lic = None
    for t in tags:
        if t.startswith("license:"):
            lic = t.split(":", 1)[1]
    base = a.base
    if not base:
        bm = card.get("base_model")
        base = (bm[0] if isinstance(bm, list) and bm else bm) if bm else None
        if not base:
            for t in tags:
                if t.startswith("base_model:"):
                    base = t.split(":", 1)[1]
                    break
    if not base:
        sys.exit("no base model: pass --base or use a repo that tags base_model")

    base_rec, base_file = find_base_record(base)
    if base_rec is None:
        sys.exit("base %s is not a catalog record - add the base first, or this derivative "
                 "belongs in data/derivatives.jsonl" % base)
    arch = arch_of(base)
    params = round(((info.get("safetensors") or {}).get("total") or 0) / 1e9, 3) or base_rec.get("params_b")
    created = (info.get("createdAt") or "")[:10]
    year = int(created[:4]) if created[:4].isdigit() else None
    kind = a.kind or ("abliteration" if "abliterat" in a.hf_id.lower()
                      else "merge" if "merge" in a.hf_id.lower() else "finetune")

    bench = []
    if a.bench:
        bench = json.loads(pathlib.Path(a.bench).read_text())
    if not bench and not a.allow_empty:
        sys.exit("no benchmarks: a record needs at least one published row that beats the base "
                 "on the same test. Pass --bench, or --allow-empty to record it as a stub.")

    rid = slug_of(a.hf_id)
    out = MODELS / (rid + ".json")
    if out.exists():
        sys.exit("%s already exists" % out)

    src_id = "hf-" + re.sub(r"[^a-z0-9]+", "-", a.hf_id.lower())
    record = {
        "id": rid, "type": "model", "name": a.hf_id.split("/")[-1],
        "family": base_rec.get("family"), "vendor": a.hf_id.split("/")[0],
        "release_year": year, "architecture": arch["architecture"], "task": a.task,
        "params_b": params, "active_params_b": base_rec.get("active_params_b"),
        "num_experts": arch["num_experts"], "experts_per_token": arch["experts_per_token"],
        "shared_expert": base_rec.get("shared_expert"),
        "hidden_size": arch["hidden_size"], "num_layers": arch["num_layers"],
        "num_kv_heads": arch["num_kv_heads"], "head_dim": arch["head_dim"], "gqa_ratio": arch["gqa_ratio"],
        "max_position_embeddings": arch["max_position_embeddings"],
        "context_scaling": "Inherited unchanged from %s (config.json max_position_embeddings %s). "
                           "This derivative does not publish a context-extension recipe of its own."
                           % (base, arch["max_position_embeddings"]),
        "context_scaling_method": "native",
        "activation": arch["act"],
        "attention_variant": "Inherited unchanged from %s: %d full GQA layers (kv heads %d, head_dim %d) "
                             "and %d other layers, per the base config.json."
                             % (base, arch["n_full"], arch["num_kv_heads"], arch["head_dim"],
                                (arch["num_layers"] or 0) - arch["n_full"]),
        "layer_types": arch["layer_types"], "attention_layer_indices": arch["full_idx"],
        "kv_cache_bytes_per_token": arch["kv_bytes"],
        "flops_per_token_active": (6.0e9 * (base_rec.get("active_params_b") or 0)) or None,
        "open_weights": True, "license": lic,
        "base_model": base, "derivative_kind": kind,
        "notes": ("ADDED %s. A %s of %s, not a new base model: same config.json, same %s parameters "
                  "%s. Recorded as a contender because it publishes benchmark evidence against its own "
                  "base; the evidence rows below are card-sourced and marked vendor_claim. Licence %s, "
                  "repo created %s." % (TODAY, kind, base, params, "(identical tensor set)" if params == base_rec.get("params_b") else "", lic, created or "unknown")),
        "status": "verified", "confidence": 0.5, "updated": TODAY,
        "sources": [src_id],
        "benchmarks": [{"name": b["name"], "score": b["score"], "scale": b.get("scale", 100),
                        "higher_is_better": b.get("higher_is_better", True), "source": src_id,
                        "vendor_claim": True, "config": b.get("config", "Card row; no harness printed.")}
                       for b in bench],
    }
    source = {
        "id": src_id, "type": "source", "name": "%s model card" % a.hf_id, "status": "verified",
        "confidence": 0.85, "updated": TODAY, "url": "https://huggingface.co/%s" % a.hf_id,
        "publisher": a.hf_id.split("/")[0], "kind": "repo", "published": created or None,
        "accessed": TODAY, "archived_url": None,
        "notes": "Fetched %s. Repo created %s, licence %s, base_model %s (kind: %s). "
                 "Safetensors total %s. Tags: %s."
                 % (TODAY, created or "unknown", lic, base, kind,
                    (info.get("safetensors") or {}).get("total"), ", ".join(tags[:8])),
    }
    if a.dry_run:
        print(json.dumps({"record": record, "source": source}, indent=2)[:4000])
        return
    SOURCES.joinpath(src_id + ".json").write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s and %s (%d benchmark rows)" % (out, SOURCES / (src_id + ".json"), len(bench)))


if __name__ == "__main__":
    main()
