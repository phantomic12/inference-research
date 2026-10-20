#!/usr/bin/env python3
"""Build the full derivatives database from cached HF API responses."""
import json, re, pathlib

CACHE = pathlib.Path("/opt/data/cache/hf_derivs")
R = pathlib.Path(__file__).resolve().parent.parent
OUT = R / "data/derivatives.jsonl"


KINDS = [
    ("gguf", r"(?i)gguf|lmstudio|ggml"),
    ("awq", r"(?i)awq"),
    ("gptq", r"(?i)gptq"),
    ("fp8", r"(?i)fp8"),
    ("nvfp4", r"(?i)nvfp4|fp4"),
    ("lowbit", r"(?i)iq1|iq2|iq3|q2_k|q3_k|q4_k|2bit|3bit|ternary|1\.58"),
    ("quant4", r"(?i)bnb|4bit|8bit|int4|int8|w4a16"),
    ("abliterated", r"(?i)abliterat|uncensor|decensor"),
    ("lora", r"(?i)lora|adapter"),
    ("port", r"(?i)xenova|onnx|litert|mlx-community|comfy|openvino|coreml|ggml-org"),
]


def classify(rid, tags):
    s = rid + " " + " ".join(tags or [])
    return [k for k, pat in KINDS if re.search(pat, s)]


def main():
    seen = {}
    for f in CACHE.glob("*.json"):
        base = f.stem.replace("__", "/")
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        for r in rows:
            rid = r.get("id")
            if not rid or rid in seen:
                continue
            seen[rid] = {
                "id": rid,
                "base": base,
                "kinds": classify(rid, r.get("tags")),
                "downloads": r.get("downloads") or 0,
                "likes": r.get("likes") or 0,
                "created": (r.get("createdAt") or "")[:10],
                "pipeline": r.get("pipeline_tag"),
            }
    rows = sorted(seen.values(), key=lambda x: -x["downloads"])
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT.name, len(rows), "rows")
    return rows


if __name__ == "__main__":
    main()
