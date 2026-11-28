#!/usr/bin/env python3
"""Pick the Phase-3 agentic backfill queue and split it into 15 shards.

Why a script and not a vibe: the skill's first rule is "pick the work queue from
the catalog, not from vibes", and the applier's SKIP list has to reconcile against
exactly the same queue or unscored models disappear silently.

In scope: models that already have a capability score (so they are ranked in
worth_running) but carry no agentic axis, are chat/agentic-shaped rather than
embedding/ASR/rerankers, and are not GGUF quant bundles.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = 15

# Family substrings that are not agentic chat models: embedding, rerank, ASR/TTS,
# OCR, image-gen, retrieval. They cannot earn an agentic score and would only add
# noise to the fan-out.
EXCLUDE_FAMILY = (
    "embed", "rerank", "retriev", "whisper", "asr", "tts", "audio", "voice",
    "ocr", "diffusion", "flux", "stable-diffusion", "clip", "siglip", "bge",
    "e5-", "gtr", "nomic", "jina", "colbert", "splade", "lilt", "donut",
    "trocr", "sensevoice", "parakeet", "coqui", "bark", "vae", "controlnet",
)
EXCLUDE_ID = ("gguf-quant-set", "quant-set")


def main() -> None:
    rows = json.loads((ROOT / "site/public/joined/worth_running.json").read_text())["rows"]
    queue = []
    skipped = []
    for r in rows:
        if r.get("capability") is None:
            continue
        if (r.get("axes") or {}).get("agentic") is not None:
            continue
        rec = json.loads((ROOT / "data/models" / f"{r['id']}.json").read_text())
        fam = (rec.get("family") or "").lower()
        rid = r["id"].lower()
        if any(x in fam for x in EXCLUDE_FAMILY) or any(x in rid for x in EXCLUDE_ID):
            skipped.append((r["id"], fam))
            continue
        queue.append({
            "id": r["id"],
            "name": r.get("name"),
            "family": rec.get("family"),
            "release_year": r.get("release_year"),
            "active_params_b": r.get("active_params_b"),
            "existing_benchmarks": [
                b.get("name") for b in (rec.get("benchmarks") or [])
            ],
        })

    # Largest active-param models first: they are the ones a "what should I run"
    # platform is asked about, and sharding in a stable sorted order means the
    # shards are balanced by cost rather than by alphabet.
    queue.sort(key=lambda q: (-(q["active_params_b"] or 0), -(q["release_year"] or 0), q["id"]))

    shards: list[list[dict]] = [[] for _ in range(SHARDS)]
    for i, item in enumerate(queue):
        shards[i % SHARDS].append(item)

    out = ROOT / "cache" / "agentic-backfill"
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.json").write_text(json.dumps(queue, indent=1), encoding="utf-8")
    for i, s in enumerate(shards, 1):
        (out / f"shard-{i:02d}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")

    print(f"queue: {len(queue)} models in scope")
    print(f"skipped as non-agentic: {len(skipped)}")
    for sid, fam in skipped[:10]:
        print(f"   - {sid} ({fam})")
    print(f"shards: {SHARDS}, sizes {[len(s) for s in shards]}")
    for i, s in enumerate(shards, 1):
        print(f"  shard-{i:02d}: " + ", ".join(x["id"] for x in s))


if __name__ == "__main__":
    main()
