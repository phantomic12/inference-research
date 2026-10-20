#!/usr/bin/env python3
"""Mine HF for derivatives of tracked models: quants and fine-tunes."""
import json, re, sys, time, urllib.parse, urllib.request, pathlib

R = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path("/opt/data/cache/hf_derivs")
UA = {"User-Agent": "inference-research/1.0"}


KINDS = [
    ("gguf", r"(?i)gguf|lmstudio|ggml"),
    ("awq", r"(?i)awq"),
    ("gptq", r"(?i)gptq"),
    ("quant4", r"(?i)bnb|4bit|8bit|int4|int8|w4a16"),
    ("fp8", r"(?i)fp8"),
    ("nvfp4", r"(?i)nvfp4|fp4"),
    ("lowbit", r"(?i)(iq1|iq2|q2_|q3_|2bit|ternary|1\.58)"),
    ("abliterated", r"(?i)abliterat|uncensor|refusal"),
    ("lora", r"(?i)lora|adapter"),
    ("port", r"(?i)xenova|onnx|litert|mlx-community|comfy|openvino|coreml|ggml-org"),
]


def classify(name):
    hits = [k for k, pat in KINDS if re.search(pat, name)]
    return hits or ["finetune"]


def hf_repos(model):
    """Every HF repo the record cites, canonical first."""
    out = []
    for sid in model.get("sources") or []:
        sp = R / "data/sources" / (str(sid) + ".json")
        if not sp.exists():
            continue
        url = json.loads(sp.read_text()).get("url") or ""
        m = re.search(r"huggingface\.co/([\w.-]+/[\w.-]+?)(?:/|$|\?)", url)
        rid = m.group(1) if m else None
        if rid and rid.split("/")[0] in ("api", "datasets", "spaces", "blog", "papers"):
            rid = None
        if rid and rid not in out:
            out.append(rid)
    return out[:4]


def fetch(repo, limit=100):
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / (repo.replace("/", "__") + ".json")
    if cf.exists() and time.time() - cf.stat().st_mtime < 86400:
        return json.loads(cf.read_text())
    q = {"filter": "base_model:" + repo, "limit": limit,
         "sort": "downloads", "direction": -1,
         "expand[]": ["downloads", "likes", "pipeline_tag", "tags"]}
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(q, doseq=True)
    for _ in range(3):
        try:
            d = json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=30).read())
            cf.write_text(json.dumps(d))
            return d
        except Exception:
            time.sleep(2)
    return None


def main(limit_models=None):
    out = {}
    models = sorted((R / "data/models").glob("*.json"))
    for i, f in enumerate(models):
        m = json.loads(f.read_text())
        repos = hf_repos(m)
        if not repos:
            continue
        seen, merged = set(), []
        for repo in repos:
            got = fetch(repo) or []
            for x in got:
                if x["id"] not in seen:
                    seen.add(x["id"])
                    merged.append(x)
        repo = repos[0]
        d = merged
        kinds, top = {}, []
        for x in d:
            for k in classify(x["id"]):
                kinds[k] = kinds.get(k, 0) + 1
            top.append({"id": x["id"], "downloads": x.get("downloads") or 0,
                        "kinds": classify(x["id"])})
        top.sort(key=lambda z: -z["downloads"])
        out[m["id"]] = {"repo": repo, "n": len(d), "kinds": kinds, "top": top[:8]}
        if limit_models and i + 1 >= limit_models:
            break
        if i % 20 == 0:
            print(i, m["id"], len(d))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data = main(n)
    jf = R / "site/public/joined/derivatives.json"
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(data, indent=1))
    tot = sum(v["n"] for v in data.values())
    print("models mined:", len(data), "| derivatives:", tot)
    kc = {}
    for v in data.values():
        for k, c in v["kinds"].items():
            kc[k] = kc.get(k, 0) + c
    print(sorted(kc.items(), key=lambda z: -z[1]))


def markdown(data):
    tot = sum(v["n"] for v in data.values())
    kc = {}
    for v in data.values():
        for k, c in v["kinds"].items():
            kc[k] = kc.get(k, 0) + c
    L = ["# Fine-tunes and quants on HF", "",
         "Every derivative HF reports for the %d models we track: %d repos total."
         % (len(data), tot), "",
         "| kind | repos |", "|---|---:|"]
    for k, c in sorted(kc.items(), key=lambda z: -z[1]):
        L.append("| %s | %d |" % (k, c))
    return "\n".join(L) + "\n"


def sections(data):
    L = []
    nogguf = sorted(k for k, v in data.items() if "gguf" not in v["kinds"])
    L += ["", "## Models you cannot run quantized today", "",
          "%d of %d tracked models have no GGUF. For a low-VRAM box these are"
          " unbuildable without writing the conversion yourself." % (len(nogguf), len(data)), ""]
    for k in nogguf[:40]:
        L.append("- [[models/%s]]" % k)
    lb = sorted((k for k, v in data.items() if "lowbit" in v["kinds"]))
    L += ["", "## Have 2-3 bit builds (%d)" % len(lb), ""]
    for k in lb[:40]:
        L.append("- [[models/%s]] (%d low-bit repos)" % (k, data[k]["kinds"]["lowbit"]))
    return L


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data = main(n)
    jf = R / "site/public/joined/derivatives.json"
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(data, indent=1))
    md = markdown(data) + "\n".join(sections(data)) + "\n"
    (R / "docs/32-finetunes-and-quants.md").write_text(md)
    print("models", len(data), "derivatives", sum(v["n"] for v in data.values()))
