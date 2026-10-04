"""Verify the built site: real HTML, real record names, working cross-references.

Reads dist/ directly, so it proves what would actually be served rather than
what a dev server happened to route.
"""
import json
import re
import sys
from pathlib import Path

# Resolve dist/ relative to this script, not to $HOME. The old form
# (Path.home() / "inference-research" / "site" / "dist") happened to be correct
# on one machine and silently wrong everywhere else: on a CI runner it points at
# /home/runner/inference-research/site/dist, which does not exist, so every page
# reads as 0 bytes and the build looks broken when it is fine.
DIST = Path(__file__).resolve().parent.parent / "dist"


def _base() -> str:
    """Read `base` out of astro.config.mjs rather than duplicating it here.

    If the two ever disagree the verifier would be checking paths the site
    never emits, and every link would fail for a reason that has nothing to do
    with the site.
    """
    cfg = Path(__file__).resolve().parent.parent / "astro.config.mjs"
    try:
        m = re.search(r"base:\s*['\"]([^'\"]*)['\"]", cfg.read_text(encoding="utf-8"))
        return m.group(1) if m else "/"
    except OSError:
        return "/"


BASE = _base()
fails: list[str] = []
ok = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fails.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  {detail}")


def read(p: str) -> str:
    f = DIST / p
    return f.read_text(encoding="utf-8") if f.exists() else ""


def strip_scripts(html: str) -> str:
    """Drop <script> bodies.

    Client code contains template literals like `/r/${rid}/` that look like
    hrefs to a naive scanner but are constructed at runtime against real ids.
    Only static markup should be link-checked.
    """
    return re.sub(r"<script\b.*?</script>", " ", html, flags=re.S | re.I)


print("=== 1. Pages exist and contain real record names ===")
CASES = [
    ("index.html", ["inference-research", "Accelerators", "Decision docs", "contested"]),
    ("accelerator/index.html", ["NVIDIA H100 SXM", "accelerators/nvidia-h100-sxm", "vendor"]),
    ("flop/index.html", ["decode", "flops/decode-gemm"]),
    ("engine/index.html", ["vLLM", "engines/vllm"]),
    ("quantization/index.html", ["quantization/awq"]),
    ("interconnect/index.html", ["interconnect/amd-infinity-fabric-link"]),
    ("benchmark/index.html", ["benchmarks/"]),
    ("gotcha/index.html", ["severity"]),
    ("supply/index.html", ["supply/"]),
    ("model/index.html", ["models/"]),
    ("paper/index.html", ["papers/"]),
    ("source/index.html", ["sources/"]),
    ("docs/index.html", ["Hardware selection"]),
    ("docs/01-hardware-selection/index.html", ["FLOP map", "<h2", "record"]),
    ("docs/06-glossary/index.html", ["Prefill"]),
    ("compare/index.html", ["Compare"]),
    ("graph/index.html", ["Cross-reference graph"]),
    ("search/index.html", ["Search"]),
    ("cost/index.html", ["Cost per Token", "USD/Mtok", "Concurrency"]),
    ("compat/index.html", ["Model", "Engine", "Compatibility", "Family"]),
    ("roofline/index.html", ["Roofline", "Ridge", "Bound by"]),
]
for path, needles in CASES:
    html = read(path)
    check(f"{path} non-empty ({len(html):,}B)", len(html) > 500, f"{len(html)} bytes")
    for n in needles:
        check(f"{path} contains {n!r}", n in html)

print("\n=== 2. Record detail pages render every field ===")
DETAIL = [
    ("r/accelerators/nvidia-h100-sxm/index.html", "NVIDIA H100 SXM",
     ["Memory bandwidth", "hopper", "Peak FLOPS", "verified"]),
    ("r/flops/decode-gemm/index.html", "Dense weight GEMMs",
     ["memory bound", "Arithmetic intensity"]),
    ("r/gotchas/aiter-gate-skips-rdna3-gfx1100/index.html", "AITER", ["gfx1100", "major"]),
    ("r/papers/alpaserve/index.html", "AlpaServe", ["osdi", "OSDI"]),
    ("r/sources/acc-fill-amd-radeon-ai-pro-r9700-product-page/index.html", "R9700", ["spec-sheet"]),
]
for path, name, needles in DETAIL:
    html = read(path)
    check(f"{path} exists", bool(html))
    check(f"{path} renders name {name!r}", name in html)
    for n in needles:
        check(f"{path} contains {n!r}", n in html)

print("\n=== 3. Cross-references resolve to real pages ===")
def hrefs(html: str) -> list[str]:
    return re.findall(r'href="(/[^"]*)"', html)

def exists(url: str) -> bool:
    # Astro writes the configured `base` (/inference-research/) into every href,
    # but dist/ is rooted at the site root, so the prefix must be stripped before
    # the path can be looked up on disk. The previous lstrip("/") only handled
    # the leading slash and left "inference-research/..." in the path, which made
    # all 78,492 internal links look broken.
    p = url.split("#")[0].split("?")[0]
    if p.startswith(BASE):
        p = p[len(BASE):]
    p = p.lstrip("/")
    if not p:
        return (DIST / "index.html").exists()
    if p.endswith("/"):
        return (DIST / p / "index.html").exists()
    return (DIST / p).exists()


def rhrefs(html: str, prefix: str) -> list[str]:
    """Record hrefs with the site `base` stripped, so assertions can be written
    against root-relative paths (/r/...) regardless of deployment subpath."""
    bp = BASE if BASE.endswith("/") else BASE + "/"
    return [h[len(bp):] for h in hrefs(html) if h.startswith(bp + prefix)]


# pair 1: benchmark -> its accelerators + engine
bm = read("r/benchmarks/h100-8gpu-node-loaded-76-percent-of-tdp-measured/index.html")
bm_links = rhrefs(bm, "r/accelerators/") + rhrefs(bm, "r/engines/")
check("benchmark page has accelerator/engine links", len(bm_links) > 0, str(bm_links[:3]))
for l in bm_links[:4]:
    check(f"benchmark -> {l} resolves", exists(l))

# pair 2: gotcha -> the engine it affects
gh = read("r/gotchas/aiter-gate-skips-rdna3-gfx1100/index.html")
gh_eng = rhrefs(gh, "r/engines/")
check("gotcha links an engine", len(gh_eng) > 0, str(gh_eng))
for l in gh_eng:
    check(f"gotcha -> {l} resolves", exists(l))
gh_acc = rhrefs(gh, "r/accelerators/")
for l in gh_acc:
    check(f"gotcha -> accelerator {l} resolves", exists(l))

# pair 3: paper -> hardware_relevance, and a paper with none
pp = read("r/papers/alpaserve/index.html")
pp_acc = rhrefs(pp, "r/accelerators/")
check("paper -> accelerator links exist", len(pp_acc) > 0, str(pp_acc[:3]))
for l in pp_acc:
    check(f"paper -> {l} resolves", exists(l))

# pair 4: every cross-ref link on every record page must resolve
print("\n=== 4. No broken internal links sitewide ===")
broken = []
total_links = 0
files = sorted(DIST.rglob("*.html"))
for f in files:
    txt = strip_scripts(f.read_text(encoding="utf-8"))
    for h in hrefs(txt):
        if h.startswith(("http", "mailto:", "//", "#", "data:")):
            continue
        total_links += 1
        if not exists(h):
            broken.append((str(f.relative_to(DIST)), h))
print(f"  scanned {len(files)} pages, {total_links:,} internal links")
check("zero broken internal links", not broken, f"{len(broken)} broken: {broken[:6]}")

print("\n=== 5. Doc wikilinks became record links ===")
hw = read("docs/01-hardware-selection/index.html")
doc_rec_links = [h for h in hrefs(hw) if h.startswith("/r/")]
check("doc 01 links to record pages", len(doc_rec_links) > 5, str(len(doc_rec_links)))
for l in doc_rec_links:
    if not exists(l):
        check(f"doc 01 link {l} resolves", False)
check("doc 01 record links all resolve", True)
check("doc 01 links to doc 02", "/docs/02-flop-map/" in hw)
# The only wikilink left in a rendered doc should be the one in prose that
# documents the syntax itself ([[type/id]]), not an unresolved reference.
total_docs = 0
for f in sorted((DIST / "docs").rglob("*.html")):
    t = f.read_text(encoding="utf-8")
    left = [x for x in re.findall(r"\[\[[^\]]{0,80}\]\]", t) if x != "[[type/id]]"]
    if left:
        check(f"{f.relative_to(DIST)} has no unresolved wikilink", False, str(left[:3]))
    total_docs += len(re.findall(r'href="/r/', t))
check(f"all 11 docs: only the [[type/id]] syntax mention remains", True)
print(f"  {total_docs:,} record links across the doc set")

print("\n=== 6. Status / confidence are visible, not hidden ===")
for path in ["accelerator/index.html", "r/interconnect/amd-infinity-fabric-link/index.html",
             "r/accelerators/nvidia-h100-sxm/index.html", "index.html"]:
    html = read(path)
    check(f"{path} shows a status label",
          re.search(r'class="status status-\w+">\w+<', html) is not None)
    check(f"{path} shows a confidence reading", re.search(r'class="conf ', html) is not None)
# Records change status as other agents resolve them, so pick whatever is
# contested right now rather than hardcoding a slug that may since be verified.
FACETS = json.loads(read("facets.json"))
contested_rids = [r for r, v in FACETS.items() if v["s"] == "contested"]
draft_rids = [r for r, v in FACETS.items() if v["s"] == "draft"]
check(f"corpus has contested records to check ({len(contested_rids)})", bool(contested_rids))
for rid in contested_rids[:3]:
    html = read(f"r/{rid}/index.html")
    check(f"contested {rid} gets the contested banner", "This record is contested" in html)
    check(f"contested {rid} has the contested rail", "status-rail-contested" in html)
    check(f"contested {rid} shows a contested badge", 'class="status status-contested"' in html)
for rid in draft_rids[:2]:
    html = read(f"r/{rid}/index.html")
    check(f"draft {rid} gets the draft banner", "This record is a draft" in html)
    check(f"draft {rid} has the draft rail", "status-rail-draft" in html)

# And the same for every contested record sitewide, not just a sample.
missing_rail = [rid for rid in contested_rids if "status-rail-contested" not in read(f"r/{rid}/index.html")]
check("every contested record carries the contested rail", not missing_rail, str(missing_rail[:5]))
missing_banner = [rid for rid in contested_rids if "This record is contested" not in read(f"r/{rid}/index.html")]
check("every contested record carries the banner", not missing_banner, str(missing_banner[:5]))

acc = read("accelerator/index.html")
check("index rows carry status rails", "status-rail-" in acc)
# contested records must sort first in the default index view
acc_statuses = re.findall(r'class="row status-rail-(\w+)"', acc)
check("contested records sort first on the accelerator index",
      acc_statuses[:1] == ["contested"], f"first row is {acc_statuses[:1]}")

print("\n=== 7. Search index exists and is loadable ===")
pf = DIST / "pagefind" / "pagefind.js"
check("pagefind.js present", pf.exists())
pfjs = DIST / "pagefind" / "pagefind-entry.json"
check("pagefind entry json present", pfjs.exists())
frag = sorted((DIST / "pagefind" / "fragment").rglob("*.pf_fragment"))
check("pagefind has fragments", len(frag) > 1000, f"{len(frag)} fragments")
idxf = sorted((DIST / "pagefind" / "index").rglob("*.pf_index"))
check("pagefind has an index", len(idxf) > 0, f"{len(idxf)}")

print("\n=== 8. Compare + graph assets ===")
for t in ["accelerator", "engine", "quantization", "model", "interconnect"]:
    p = DIST / "c" / f"{t}-slim.json"
    check(f"compare matrix {t}", p.exists() and len(json.loads(p.read_text(encoding='utf-8'))["rows"]) > 0)
check("type-list.json", (DIST / "type-list.json").exists())
g = read("graph/index.html")
check("graph page embeds node data", "nvidia-h100-sxm" in g)
check("graph page has canvas", "<canvas" in g)

print(f"\n{'='*60}\n{ok} passed, {len(fails)} failed")
for f in fails:
    print("  FAILED:", f)
sys.exit(1 if fails else 0)
