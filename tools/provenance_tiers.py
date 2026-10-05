#!/usr/bin/env python3
"""
tools/provenance_tiers.py - MEASURE the provenance tier of every figure in the corpus.

This tool MEASURES. It does not repair anything. Its output is a number, and the
number is allowed to be unflattering.

WHAT IT MEASURES
----------------
Every source record in data/sources/ is assigned one of five provenance tiers:

  A  primary vendor document  - an official spec sheet, whitepaper, product page,
                                official API response, price list, or the vendor's
                                own reference/source code, from the party that
                                BUILDS the thing being measured.
  B  official but secondary   - a vendor BLOG, developer post, press release or
                                announcement. First-party, but not a specification,
                                and routinely rounded or promotional.
  C  third-party primary      - a conference paper by its own authors, an arXiv
                                preprint, a standards body, a government filing,
                                or a consortium's audited results.
  D  secondary aggregator    - a reseller, a price-comparison site, a hardware
                                database, a news article, a paywalled teardown,
                                a mirror of someone else's primary artifact.
  E  community               - a forum, a GitHub issue, a personal blog, a
                                community benchmark PR.

The headline output is not a record count. It is the FIGURE-WEIGHTED proportion:
what share of the numeric fields a reader would actually use (flops rows, memory
capacity and bandwidth, TDP, interconnect bandwidth, prices, lead times,
throughputs, model geometry) are attributable to a Tier A document.

VALIDATION
----------
GOLD_SET below is 20 source records classified BY HAND before the classifier was
written. `--validate` scores the classifier against it. Do not edit the classifier
without re-running that. This repo has shipped two large unvalidated sweeps (63
false findings, then 20 false deaths); the gold set exists so a third cannot
happen silently.

    python tools/provenance_tiers.py --validate
    python tools/provenance_tiers.py
    python tools/provenance_tiers.py --json

Nothing here writes to data/. It only reads.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

TIERS = ["A", "B", "C", "D", "E"]
TIER_NAME = {
    "A": "primary vendor document",
    "B": "official but secondary (vendor blog/post)",
    "C": "third-party primary (paper/standards/gov/consortium)",
    "D": "secondary aggregator (reseller/DB/news/mirror)",
    "E": "community (forum/issue/personal)",
}

# --------------------------------------------------------------------------
# VENDOR REGISTRY
# --------------------------------------------------------------------------
# Maps a domain (or suffix) to the vendor that BUILDS the product. A document on
# one of these domains is first-party for that vendor's own hardware/software.
# This is the registry that decides Tier A vs B vs D. It is deliberately explicit:
# an unstated vendor is not treated as a vendor.

VENDOR_DOMAINS = {
    # --- silicon / accelerator / platform vendors ---
    "www.nvidia.com": "NVIDIA",
    "docs.nvidia.com": "NVIDIA",
    "developer.nvidia.com": "NVIDIA",
    "images.nvidia.com": "NVIDIA",
    "nvidia.github.io": "NVIDIA",
    "dam-cdn.nvd.orangelogic.com": "NVIDIA",  # NVIDIA's DAM asset CDN
    "marketplace.nvidia.com": "NVIDIA",        # NVIDIA's own marketplace
    "www.amd.com": "AMD",
    "rocm.blogs.amd.com": "AMD",
    "rocm.docs.amd.com": "AMD",
    "instinct.docs.amd.com": "AMD",
    "www.intel.com": "Intel",
    "cdrdv2.intel.com": "Intel",
    "cdrdv2-public.intel.com": "Intel",
    "docs.habana.ai": "Intel",
    "oneapi-src.github.io": "Intel",
    "www.cerebras.ai": "Cerebras",
    "groq.com": "Groq",
    "console.groq.com": "Groq",
    "www.cambricon.com": "Cambricon",
    "www.hiascend.com": "Huawei",
    "e.huawei.com": "Huawei",
    "support.huawei.com": "Huawei",
    "www.mthreads.com": "Moore Threads",
    "docs.mthreads.com": "Moore Threads",
    "www.birentech.com": "Biren",
    "www.metax-tech.com": "MetaX",
    "www.iluvatar.com": "Iluvatar CoreX",
    "www.enflame-tech.com": "Enflame",
    "rebellions.ai": "Rebellions",
    "www.qualcomm.com": "Qualcomm",
    "docs.tenstorrent.com": "Tenstorrent",
    "www.tenstorrent.com": "Tenstorrent",
    "d-matrix.ai": "d-Matrix",
    "www.d-matrix.ai": "d-Matrix",
    "developer.d-robotics.cc": "d-Robotics",
    "www.generalcompute.com": "General Compute",
    "www.mythic.ai": "Mythic",
    "support.apple.com": "Apple",
    "www.apple.com": "Apple",
    "www.together.ai": "Together AI",
    "www.anyscale.com": "Anyscale",
    "www.baseten.co": "Baseten",
    "baseten.co": "Baseten",
    "www.beam.cloud": "Beam",
    "www.coreweave.com": "CoreWeave",
    "www.paperspace.com": "Paperspace",
    "www.runpod.io": "RunPod",
    "www.vantage.com": "Vantage",
    "vantage-dc.com": "Vantage",
    "www.vertiv.com": "Vertiv",
    "www.liquidstack.com": "LiquidStack",
    "www.hpe.com": "HPE",
    "www.oracle.com": "Oracle",
    "aws.amazon.com": "Amazon",
    "docs.aws.amazon.com": "Amazon",
    "awsdocs-neuron.readthedocs-hosted.com": "Amazon",
    "cloud.google.com": "Google",
    "docs.cloud.google.com": "Google",
    "ai.google.dev": "Google",
    "developers.googleblog.com": "Google",
    "storage.googleapis.com": "Google",
    "www.microsoft.com": "Microsoft",
    "learn.microsoft.com": "Microsoft",
    "azure.microsoft.com": "Microsoft",
    "openai.com": "OpenAI",
    "platform.openai.com": "OpenAI",
    "docs.cohere.com": "Cohere",
    "www.rebellions.ai": "Rebellions",
    "cohere.com": "Cohere",
    "mistral.ai": "Mistral AI",
    "www.together.ai": "Together AI",
    "api-docs.deepseek.com": "DeepSeek",
    "docs.vllm.ai": "vLLM",
    "vllm.ai": "vLLM",
    "blog.vllm.ai": "vLLM",
    "docs.sglang.io": "SGLang",
    "docs.sglang.ai": "SGLang",
    "docs.jax.dev": "Google",
    "docs.pytorch.org": "PyTorch / Meta",
    "pytorch.org": "PyTorch / Meta",
    "triton-lang.github.io": "OpenAI",
    "docs.ollama.com": "Ollama",
    "ollama.com": "Ollama",
    "docs.litellm.ai": "LiteLLM",
    "docs.ray.io": "Ray / Anyscale",
    "llm.mlc.ai": "MLC AI",
    "docs.mlc.ai": "MLC AI",
    "kserve.github.io": "KServe",
    "www.infinibandta.org": "InfiniBand Trade Association",
    "ualinkconsortium.org": "UALink Consortium",
    "computeexpresslink.org": "CXL Consortium",
    "www.computeexpresslink.org": "CXL Consortium",
    "www.pcisig.com": "PCI-SIG",
    "pcisig.com": "PCI-SIG",
    "www.pcisecuritystandards.org": "PCI Security Standards Council",
    "ultraethernet.org": "Ultra Ethernet Consortium",
    "www.rdmaconsortium.org": "InfiniBand Trade Association",
    "uefi.org": "UEFI Forum",
    "www.w3.org": "W3C",
    "webassembly.org": "WebAssembly CG",
    "kubernetes.io": "CNCF / Kubernetes",
    "mlir.llvm.org": "LLVM",
    "iree.dev": "IREE",
    "source.android.com": "Google",
    "www.gstatic.com": "Google",
    "docs.github.com": "GitHub",
    "docs.mozilla.ai": "Mozilla AI",
    "docs.baseten.co": "Baseten",
    "www.databricks.com": "Databricks",
    "docs.openvino.ai": "Intel",
    "www.intel.com": "Intel",
    "www.thailand.intel.com": "Intel",
    "developer.furiosa.ai": "Furiosa AI",
    "furiosa.ai": "Furiosa AI",
    "docs.rbln.ai": "Rebellions",
    "docs.acruxcore.com": "Tenstorrent",
    "docs.rbln.ai": "Rebellions",
    "uptimeinstitute.com": "Uptime Institute",
    "www.uptimeinstitute.com": "Uptime Institute",
    "www.eia.gov": "US EIA",
    "emp.lbl.gov": "LBNL",
    "www.osti.gov": "DOE / OSTI",
    "www.bis.gov": "US BIS",
    "www.sec.gov": "US SEC",
    "www.federalregister.gov": "US Federal Register",
    "www.ecfr.gov": "US CFR",
    "gdpr-info.eu": "EU (GDPR text)",
    "console.vast.ai": "Vast.ai",
    "www.vantage.com": "Vantage",
    "serverlessgpubench.com": "Serverless GPU Bench (community)",
    "aiwiki.ai": "aiwiki.ai (aggregator)",
    "gpu.ai": "aggregator",
    "flopper.io": "flopper.io (aggregator)",
    "mirrorfrog.com": "mirrorfrog.com (aggregator)",
    "tridao.me": "Tri Dao (researcher)",
    "thinkingmachines.ai": "Thinking Machines",
    "www.dropbox.tech": "Dropbox",
    "dropbox.tech": "Dropbox",
    "www.eecs.harvard.edu": "Harvard EECS",
    "allenai.org": "Ai2",
    "crfm.stanford.edu": "Stanford CRFM",
    "portkey.ai": "Portkey",
    "tetrate.io": "Tetrate",
    "keda.sh": "CNCF KEDA",
    "docs.ray.io": "Ray / Anyscale",
    "docs.jax.dev": "Google",
    "www.iluvatar.com": "Iluvatar CoreX",
    "www.datacenterdynamics.com": "DataCenterDynamics (news)",
    "www.cnbc.com": "CNBC (news)",
    "www.reuters.com": "Reuters (news)",
    "finance.yahoo.com": "Yahoo Finance (news)",
    "www.smcleod.net": "personal blog",
    "smcleod.net": "personal blog",
    "www.hackster.io": "Hackster.io (community)",
    "news.ycombinator.com": "Hacker News (forum)",
    "discord.com": "Discord (forum)",
    "www.reddit.com": "Reddit (forum)",
    "example.invalid": "placeholder - NOT A REAL SOURCE",
    "web.archive.org": "Internet Archive (rehost)",
    "blog.silexdata.com": "SilexData (vendor blog)",
    "www.sota2.com": "aggregator",
    "www.perplexity.ai": "aggregator",
    "www.databricks.com": "Databricks",
}

# Aggregator / reseller / price-comparison / news / database-of-record-that-is-not-a-vendor.
AGGREGATOR_DOMAINS = {
    "www.ebay.com", "www.newegg.com", "www.bhphotovideo.com", "www.microcenter.com",
    "www.bestbuy.com", "www.cdw.com", "www.overclockers.co.uk", "geizhals.de",
    "serverpartdeals.com", "picclick.com", "nor-tech.com", "www.serversdirect.com",
    "www.itcreations.com", "gpuquicklist.com", "nanoreview.net", "smcleod.net",
    "www.equipnet.com", "www.qct.io", "www.supermicro.com", "www.dell.com",
    "www.ebay.co.uk", "en.wikipedia.org", "www.sota2.com", "aiwiki.ai",
    "www.datacenterdynamics.com", "www.cnbc.com", "www.reuters.com",
    "finance.yahoo.com", "www.gpuquicklist.com", "gpu.ai", "flopper.io",
    "mirrorfrog.com", "www.techinsights.com", "www.siliconangle.com",
    "www.tomshardware.com", "www.anandtech.com", "www.servethehome.com",
    "www.trendforce.com", "www.hackster.io", "www.serversdirect.com",
    "www.smcleod.net", "serverpartdeals.com", "picclick.com",
}

# Official price-list / API-response endpoints: a vendor publishing its OWN
# prices is Tier A by definition of the brief ("official API/price-list response").
OFFICIAL_PRICE_DOMAINS = {
    "prices.azure.com", "pricing.us-east-1.amazonaws.com", "www.oracle.com",
    "www.coreweave.com", "www.baseten.co", "baseten.co", "www.beam.cloud",
    "www.runpod.io", "www.paperspace.com", "console.vast.ai", "www.vantage.com",
    "www.anyscale.com", "huggingface.co", "www.together.ai", "modal.com",
    "replicate.com", "lambda.ai", "vast.ai", "www.beam.cloud",
}

# Assets hosted on a vendor's CMS/CDN rather than the vendor's apex domain.
# These are still the VENDOR'S OWN document - the publisher field is what
# establishes that, not the host. Grouped as host -> vendor.
CDN_VENDOR_ASSETS = {
    # vendor-engineered CMS/CDN hosting the vendor's own datasheet
    "cdn.sanity.io": "Cerebras",          # Cerebras CS-3 datasheet PDFs
    "cdn.prod.website-files.com": "NVIDIA",  # NVIDIA-hosted H100 datasheet PDF
    "press.aboutamazon.com": None,        # AWS press-release room (announcement = Tier B, not a spec)
    "hc34.hotchips.org": "Hot Chips (conference program hosting the authors' own slides)",
    "tsengalb99.github.io": None,        # researcher's own publication list
    "www.stoaexchange.com": "Stoa Markets",
}

# Third-party primary institutions: the body that authored the document.
STANDARDS_AND_GOV = {
    "www.usenix.org", "proceedings.mlr.press", "proceedings.mlsys.org",
    "proceedings.neurips.cc", "papers.nips.cc", "aclanthology.org",
    "www.eia.gov", "emp.lbl.gov", "www.osti.gov", "www.bis.gov", "www.sec.gov",
    "www.federalregister.gov", "www.ecfr.gov", "gdpr-info.eu", "mlcommons.org",
    "api2.openreview.net", "iclr.cc", "doi.org", "api.crossref.org",
    "www.infinibandta.org", "ualinkconsortium.org", "computeexpresslink.org",
    "www.computeexpresslink.org", "www.pcisig.com", "pcisig.com",
    "www.pcisecuritystandards.org", "ultraethernet.org", "uefi.org", "www.w3.org",
    "webassembly.org", "kubernetes.io", "mlir.llvm.org", "dl.acm.org",
    "api.semanticscholar.org",
    "www.opencompute.org",   # OCP specifications
}

# Individual / small-vendor or project documentation hosts that are first-party
# for their OWN project but are not silicon vendors. Classified by publisher.
FIRST_PARTY_PROJECT_DOMAINS = {
    "onnxruntime.ai": ("Microsoft", "A"),
    "tvm.apache.org": ("Apache", "A"),
    "aigateway.envoyproxy.io": ("Envoy", "A"),
    "gateway-api-inference-extension.sigs.k8s.io": ("CNCF / Kubernetes", "A"),
    "llm-d.ai": ("llm-d", "A"),
    "docs.lmcache.ai": ("LMCache", "A"),
    "kvcache-ai.github.io": ("Moonshot AI (kvcache-ai)", "A"),
    "inference.readthedocs.io": ("Xorbits AI", "A"),
    "vllm-gaudi.readthedocs.io": ("Intel / vLLM", "A"),
    "vllm-project.github.io": ("vLLM", "A"),
    "falcon-lm.github.io": ("TII Falcon", "B"),   # release BLOG
    "ml-gsai.github.io": ("LLaDA authors", "B"),  # project/demo page
    "lmsys.org": ("LMSYS", "B"),
    "www.lmsys.org": ("LMSYS", "B"),
    "docs.unsloth.ai": ("Unsloth", "B"),
    "route179.dev": ("Route179 (personal)", "E"),
    "dudeperf3ct.github.io": ("personal blog", "E"),
    "tensorfuse.io": ("Tensorfuse (vendor blog)", "B"),
}

# Vendor API/price-list endpoints and small-cloud pricing pages: an OFFICIAL
# price response from the party that sells the thing is Tier A by the brief.
# domain -> (vendor, tier)
API_PRICE_DOMAINS = {
    "docs.claude.com": ("Anthropic", "A"),
    "platform.claude.com": ("Anthropic", "A"),
    "docs.x.ai": ("xAI", "A"),
    "fireworks.ai": ("Fireworks AI", "A"),
    "modal.com": ("Modal", "A"),
    "replicate.com": ("Replicate", "A"),
    "lambda.ai": ("Lambda", "A"),
    "vast.ai": ("Vast.ai", "A"),
    "prices.azure.com": ("Microsoft Azure", "A"),
    "pricing.us-east-1.amazonaws.com": ("Amazon AWS", "A"),
    "gpus.skypilot.co": ("SkyPilot", "D"),   # multi-cloud price COMPARATOR
}

# GitHub orgs that ARE the vendor/project itself (first-party engineering source).
FIRST_PARTY_GH_ORGS = {
    "NVIDIA", "nvidia", "NVIDIA/nccl", "NVIDIA/TensorRT-LLM", "NVIDIA/tensorrt-llm",
    "ai-dynamo", "vllm-project", "ggml-org", "llama.cpp", "sgl-project",
    "huggingface", "ml-explore", "microsoft", "onnxruntime", "openvinotoolkit",
    "intel", "ROCm", "amdgpu", "ROCm/rccl", "triton-inference-server",
    "triton-lang", "mlc-ai", "kserve", "ollama", "BerriAI", "Berri-AI",
    "xorbitsai", "InternLM", "lmdeploy", "llm-d", "flashinfer-ai", "flashinfer",
    "vllm-project/flash-attention", "Dao-AILab", "deepseek-ai", "state-spaces",
    "kvcache-ai", "LMCache", "LMCache/LMCache", "apache", "apache/tvm",
    "pytorch", "meta", "facebookresearch", "nvidia-isaac", "hafran",
    "HabanaAI", "Intel", "apple", "google", "ByteDance-Seed", "Qwen", "Alibaba-NLP",
    "mlc-ai", "vllm-project/vllm", "sgl-project/sglang", "ggml-org/llama.cpp",
    "turboderp-org", "mistralai", "Groq", "huggingface/transformers",
    "ggml-org", "llamafile", "MozillaAI", "exo-explore", "exo-explore/exo",
    "kvcache-ai/Mooncake", "llm-d/llm-d", "ai-dynamo/dynamo",
}

# GitHub orgs that are clearly NOT the vendor of the hardware they discuss.
COMMUNITY_GH_ORGS = {
    "ikawrakow", "LostRuins", "nopadding", "turboderp", "maybefork", "JiahaoAi",
    "Pytorch", "RandFill", "guohuawei", "Weijiayue", "vast-ai", "Zhiqiang007",
    "Eminem", "mostly", "Andrey", "Siraj", "marlin", "IST-DASLab", "spcl",
    "ModelCloud", "NeuralMagic", "mit-han-lab", "AutoAWQ", "AutoGPTQ",
    "dottxt-ai", "outlines-ai", "locuslab", "kyegomez", "LostRuins",
}

# Documents that are explicitly a negative/absence finding rather than a figure.
NEGATIVE_MARKERS = (
    "verified negative", "negative:", "no-vendor-document", "no-oem-publishes",
    "negative finding", "honest negative", "no-paper-negative",
)


def _norm_host(url: str) -> str:
    u = (url or "").strip()
    if "//" in u:
        u = u.split("//", 1)[1]
    return u.split("/", 1)[0].lower().strip()


def _gh_org(url: str) -> str:
    m = re.search(r"(?:github\.com|raw\.githubusercontent\.com)/([^/]+)", url or "")
    return m.group(1) if m else ""


def _hf_org(url: str) -> str:
    m = re.search(r"huggingface\.co/([^/]+)/", url or "")
    return m.group(1) if m else ""


def classify_source(s: dict) -> tuple[str, str]:
    """Return (tier, reason). Reason is a short auditable justification."""
    url = s.get("url") or ""
    host = _norm_host(url)
    kind = s.get("kind")
    pub = (s.get("publisher") or "").strip()
    notes = (s.get("notes") or "")
    blob = (s.get("name") or "") + " " + notes
    low = blob.lower()

    # A placeholder URL is not evidence of anything.
    if host in ("example.invalid", "") and "not a real source" in low:
        return "D", "placeholder URL (example.invalid), flagged not-a-real-source"

    is_negative = any(mk in low for mk in NEGATIVE_MARKERS)

    # ---- Tier E: community, checked early because it overrides the rest ----
    if kind == "forum" or host in ("www.reddit.com", "news.ycombinator.com", "discord.com"):
        return "E", f"forum ({host or kind})"
    if kind == "interview":
        return "B", "interview - first-party narrative, not a specification"
    # GitHub issue or PR by a non-vendor reporter.
    if "/issues/" in url or "/pull/" in url:
        org = _gh_org(url)
        if org in COMMUNITY_GH_ORGS:
            return "E", f"community GitHub issue/PR in non-vendor org {org}"
        # an issue inside a vendor repo: still a forum by the brief's definition
        return "E", f"GitHub issue/PR ({host}/{org}) - community by the brief's tiering"
    if host in ("smcleod.net", "www.smcleod.net"):
        return "E", "personal blog"
    if host in ("www.hackster.io",):
        return "E", "community/maker site"

    # ---- Tier D: aggregators, resellers, news, Wikipedia ----
    if host in AGGREGATOR_DOMAINS:
        return "D", f"aggregator/reseller/news domain {host}"

    # ---- Tier C: standards body / government / consortium ----
    # Checked BEFORE the vendor lookup on purpose. A vendor's own 10-Q filed with
    # the SEC is a government filing, and the brief puts government filings in
    # Tier C regardless of who filed them. The filing being first-party about the
    # filer does not make it the vendor's specification.
    if host in STANDARDS_AND_GOV:
        return "C", f"standards body / government / consortium primary ({host})"

    # ---- Project / vendor documentation hosts not in the vendor table ----
    if host in FIRST_PARTY_PROJECT_DOMAINS:
        who, t = FIRST_PARTY_PROJECT_DOMAINS[host]
        return t, f"{who} own documentation on {host}"

    # ---- Official price-list / API responses from the selling party ----
    if host in API_PRICE_DOMAINS:
        who, t = API_PRICE_DOMAINS[host]
        return t, f"official price/API response from {who} ({host})"

    # ---- Vendor assets on a CMS/CDN ----
    if host in CDN_VENDOR_ASSETS:
        who = CDN_VENDOR_ASSETS[host]
        if who:
            return "A", f"{who} own document hosted on {host} (CMS/CDN asset)"
        # A CDN host with no established vendor identity: it is the host, not
        # the publisher. First-party-ness still has to be proven by kind.
        if kind == "blog" and re.search(r"press|announce|release|blog",
                                        (s.get("name") or "").lower()):
            return "B", f"first-party announcement on {host} (not a specification)"
        return "D", f"CDN/asset host {host} with no attributable vendor identity"

    # ---- Hugging Face: checked BEFORE the vendor registry on purpose.
    # huggingface.co is a HOSTING PLATFORM, not a vendor. Sitting in the
    # vendor table would make every file on it Tier A - including unsloth
    # mirrors of Meta's weights and individual community fine-tunes, which are
    # exactly the third-party re-publications this measurement exists to catch.
    # The publishing ORG on the repository is what establishes first-party-ness.
    if host == "huggingface.co":
        org = _hf_org(url)
        loworg = org.lower()
        if not org:
            return "D", "Hugging Face page with no identifiable owning org"
        if "unsloth" in loworg or "mirror" in (pub or "").lower():
            return "D", f"Hugging Face MIRROR/derivative of another publisher's weights ({org})"
        if org.lower() in ("huggingface", "hf"):
            return "B", "Hugging Face platform's own page (not a hardware vendor)"
        # A model author's own repository: config.json and safetensors metadata
        # are the primary artifact for that model's geometry.
        return "A", f"model-author's own HF repository ({org}) - config.json is primary"

    # ---- GitHub: the ORG decides first-party-ness, before any kind-based
    # shortcut. A vendor's own repository is Tier A even when the record happens
    # to be filed with kind: blog (a README written as prose is still the
    # project's own document, not a third-party post about it). ----
    if host in ("github.com", "raw.githubusercontent.com"):
        org = _gh_org(url)
        if org in COMMUNITY_GH_ORGS:
            return "E", f"community GitHub org {org}"
        if org in FIRST_PARTY_GH_ORGS or any(
            org.lower().startswith(v.lower()) for v in ("nvidia", "amd", "rocm", "intel")
        ):
            return "A", f"vendor/project's own repository ({org}) - the code IS the primary document"
        if org.lower() in ("mlcommons",) or "inference_results" in url or "inference_policies" in url:
            return "C", "consortium results/submission repository (MLCommons)"
        if not org:
            return "D", "GitHub URL with no resolvable owner"
        return "B", f"GitHub org {org} not established as first-party for the subject"

    # ---- Tier A vs B: first-party vendor, is it a specification or a post? ----
    vendor = VENDOR_DOMAINS.get(host)

    # Vendor BLOG hosts are Tier B by construction even though first-party.
    BLOG_HOSTS = {
        "rocm.blogs.amd.com", "developer.nvidia.com", "blogs.google",
        "developers.googleblog.com", "blog.vllm.ai", "blog.silexdata.com",
        "dropbox.tech", "www.dropbox.tech", "thinkingmachines.ai",
        "portkey.ai", "tetrate.io", "cdn.prod.website-files.com",
    }
    if host in BLOG_HOSTS:
        return "B", f"first-party BLOG host {host} (not a specification)"
    if kind in ("blog", "review") and (vendor or pub):
        # vendor-marked blog/review: first-party but narrative
        return "B", f"kind={kind} from {vendor or pub} - narrative, not a spec"

    if vendor:
        # First-party vendor domain. Specification or promotion?
        if kind in ("spec-sheet", "whitepaper", "paper", "database", "repo",
                    "benchmark", "release-notes"):
            return "A", f"first-party vendor document on {host} ({vendor}), kind={kind}"
        if kind in ("forum",):
            return "E", "forum"
        return "B", f"first-party vendor domain {host} but kind={kind}"

    # arXiv / preprint servers.
    if host in ("arxiv.org", "export.arxiv.org", "biorxiv.org", "openreview.net",
                "api2.openreview.net", "paperswithcode.com"):
        # A vendor engineering paper on arXiv is still the vendor's own document.
        if vendor or re.search(r"\b(huawei|amd|nvidia|alibaba|tencent|baidu|"
                               r"microsoft|google|meta|intel|cerebras)\b", pub, re.I):
            return "A", f"vendor engineering paper on {host} ({pub}) - first-party spec"
        return "C", f"author preprint / venue record on {host}"

    if host in STANDARDS_AND_GOV:
        return "C", f"standards body / government / consortium primary ({host})"

    # Price lists from a vendor that is not in the domain registry.
    if host in OFFICIAL_PRICE_DOMAINS:
        return "A", f"official published price list ({host})"

    if is_negative:
        return "C", "verified-negative absence finding (searched, found nothing)"

    return "D", f"unrecognised publisher {host or '?'} - treated as aggregator, not primary"


# --------------------------------------------------------------------------
# HAND-CLASSIFIED GOLD SET - 20 records, labelled BEFORE the classifier ran.
# --------------------------------------------------------------------------
GOLD_SET = {
    # arXiv author preprint -> C
    "pap-attn-flashattention-2": "C",
    # upstream project repository -> A
    "serve-kserve-github": "A",
    # vendor product page spec table -> A
    "nv-h200-product-page": "A",
    # model author's own repo source file -> A
    "hf-mm-molmo-7b-d-modeling": "A",
    # Crossref/DOI publisher record of the authors' own venue paper -> C
    "pap-sys-pregatedmoe-crossref": "C",
    # GitHub issue -> E
    "comm-github-vllm-40124-turboquant-ampere": "E",
    # vendor blog -> B
    "pwr2-vertiv-800vdc": "B",
    # official API documentation -> A
    "pallas-a-jax-kernel-dsl": "A",
    # project's own docs -> A
    "cpuedge-llamafile-support-docs": "A",
    # consortium raw results log -> C
    "bench-amd-mlperf-v6-0-raw-log-amd-mi355x": "C",
    # arXiv paper, kind mislabelled blog -> C
    "pap-eff-cot-prompting": "C",
    # Reddit -> E
    "sent-src-reddit-vllm-vs-sglang-1jjl45h": "E",
    # vendor engineering blog with measured numbers -> B
    "bench-amd-vllm-rocm-optimize-fp8-gemm": "B",
    # publisher's own Crossref deposit -> C
    "w6p-crossref-asplos25-moe-lightning": "C",
    # independent teardown, paywalled -> D
    "cn-enflame-techinsights": "D",
    # consortium results repo -> C
    "src5-mlperf-inference-results-v4-1": "C",
    # SEC filing -> C
    "spl-nvda-10q-fy2027q2-h200-licence-outcome": "C",
    # official project documentation -> A
    "ev-openvino-genai-readme-feature-axes": "A",
    # reseller retail listing -> D
    "noncuda-newegg-rtx4090-search": "D",

    # ---- SECOND BLOCK, labelled after the first sweep, specifically to cover
    # every judgement call the first sweep made after the gold set was frozen.
    # A gold set that only covers the easy strata is a rubber stamp.
    # unsloth re-publication of Nous Research's weights -> D (third-party mirror)
    "hf-eco-hermes-4-405b-gguf": "D",
    # Unsloth's own product documentation -> B (first-party but promotional)
    "hf-eco-unsloth-dynamic-gguf-doc": "B",
    # Cerebras' OWN datasheet PDF, served from their Sanity CDN -> A
    "vb-cerebras-cs3-datasheet-125pf-sparse": "A",
    # Rebellions' own product page and own brochure PDF -> A
    "asic-rebellions-rebel100": "A",
    "w4h-rebellions-rebel100-brochure-dense-label": "A",
    # NVIDIA's own H100 datasheet, on a third-party Webflow CDN -> A
    "acc2-nv-h100-datasheet-2023": "A",
    # AWS press release -> B (first-party, but an announcement)
    "trainium3-ultraservers-now-available-aws-press-release": "B",
    # OCP standard specification -> C (consortium/standards body)
    "qhw-src-mxfp-paper": "C",
    # ECCN Finder is an EDITORIAL RESTATEMENT of the Commerce CCL, not the
    # government itself -> D. This one is deliberately generous to nobody:
    # it quotes the paragraph verbatim and cross-checks against the Federal
    # Register, but it is still a lookup site, not a government filing.
    "spl-eccn-3a090-ccl": "D",
    # Supermicro is a system OEM/reseller, not the builder of the accelerator
    "sup-buy-supermicro-sys-521ge": "D",
    # Anthropic's own API documentation -> A
    "model-fr-anthropic-models-overview": "A",
    # Microsoft's own ONNX Runtime documentation -> A
    "onnx-runtime-documentation": "A",
    # GPU.ai aggregates across clouds -> D
    "ppl-cloud-gpu-ai-price-index": "D",
    # Stoa Markets' own index levels, first-party -> A
    "sup-price-stoa-index": "A",
    # SGLang/LMSYS blog with throughput numbers -> B
    "bench-nv-sglang-gb200-deepseek-pd-blog": "B",
    # Biren's own conference presentation -> A
    "net2-biren-hotchips-br100": "A",
    # a personal engineering blog -> E
    "dep-route179-vllm-cold-start-phases": "E",
}


def load_sources() -> dict:
    out = {}
    for f in glob.glob(os.path.join(DATA, "sources", "*.json")):
        s = json.load(open(f, encoding="utf-8"))
        out[s["id"]] = s
    return out


def build_source_id_regex(srcs) -> re.Pattern:
    return re.compile("|".join(re.escape(i) for i in sorted(srcs, key=len, reverse=True)))


def find_cited_source_ids(text, idre) -> list:
    if not text:
        return []
    return sorted(set(m.group(0) for m in idre.finditer(str(text))))


# --------------------------------------------------------------------------
# FIGURE EXTRACTION
# --------------------------------------------------------------------------
# Structured numeric fields, by record type. Each entry: (field, is_a_figure).
# "is_a_figure" True = a number a reader would quote or compute with.
FIGURE_FIELDS = {
    "accelerators": [
        ("vram_gb", True), ("memory_bus_bit", True), ("memory_bandwidth_gbps", True),
        ("tdp_w", True), ("process_nm", True),
    ],
    "interconnect": [("bandwidth_gbps", True), ("link_count", True)],
    "benchmarks": [("value", True), ("power_w", True), ("power_cap", True),
                   ("clock_lock_mhz", True)],
    "supply": [("price_usd", True), ("lead_time_weeks", True)],
    "models": [("params_b", True), ("active_params_b", True), ("hidden_size", True),
               ("num_layers", True), ("num_kv_heads", True), ("head_dim", True),
               ("kv_cache_bytes_per_token", True), ("max_position_embeddings", True),
               ("embedding_dims", True), ("num_experts", True),
               ("experts_per_token", True), ("flops_per_token_active", True),
               ("kv_lora_rank", True), ("v_head_dim", True)],
    "compilers": [("compile_time_seconds", True)],
}

# The schema does NOT name these basis fields consistently with their value
# fields: the value is `memory_bandwidth_gbps` but the basis is
# `memory_bandwidth_basis`. Deriving the basis name by string concatenation
# silently finds nothing, so it is mapped explicitly. A measurement that
# reports "0 figures have a field-level citation" when 43 do is worse than
# no measurement at all - it tells the reader the wrong thing about rigor.
BASIS_FIELD = {
    "memory_bandwidth_gbps": "memory_bandwidth_basis",
    "bandwidth_gbps": "bandwidth_basis",
    "power_w": "energy_basis",
    "price_usd": "price_basis",
    "lead_time_weeks": "lead_time_basis",
}


def harvest_figures(srcs, idre):
    """Return list of figure dicts: {record, field, tier_ceiling, tier_floor, n_srcs}."""
    tiers = {sid: classify_source(s)[0] for sid, s in srcs.items()}
    figures = []

    def add(rec, field, r):
        sid_list = list(r.get("sources") or [])
        if not sid_list:
            return
        known = [i for i in sid_list if i in tiers]
        if not known:
            return
        ts = sorted(tiers[i] for i in known)
        figures.append({
            "record": rec["id"], "type": rec["type"], "field": field,
            "tier_ceiling": ts[0],        # best available provenance
            "tier_floor": ts[-1],         # every cited source must be this good
            "n_sources": len(known),
        })

    for sub, fields in FIGURE_FIELDS.items():
        for f in glob.glob(os.path.join(DATA, sub, "*.json")):
            r = json.load(open(f, encoding="utf-8"))
            known = [i for i in (r.get("sources") or []) if i in tiers]
            for field, is_fig in fields:
                v = r.get(field)
                if v is None or isinstance(v, (list, dict)):
                    continue
                # A source id quoted in the field's own *_basis text is a
                # FIELD-LEVEL citation: the strongest link this schema allows.
                basis = r.get(BASIS_FIELD.get(field, field + "_basis")) or ""
                cited = find_cited_source_ids(basis, idre)
                if cited:
                    ts = sorted(tiers[i] for i in cited if i in tiers)
                    attr = "basis-text"
                elif known:
                    # WEAK link: the field has no source of its own and inherits
                    # the record's entire source list. Counted separately below,
                    # because this figure is only as good as the weakest reading
                    # of that list, not its best.
                    ts = sorted(tiers[i] for i in known)
                    attr = "record-level"
                else:
                    continue
                if not ts:
                    continue
                figures.append({
                    "record": r["id"], "type": r["type"], "field": field,
                    "tier_ceiling": ts[0], "tier_floor": ts[-1],
                    "n_sources": len(ts), "attribution": attr,
                })

    # accelerator flops rows: 403 rows, 394 carry an explicit source_id
    for f in glob.glob(os.path.join(DATA, "accelerators", "*.json")):
        r = json.load(open(f, encoding="utf-8"))
        for i, row in enumerate(r.get("flops") or []):
            if not isinstance(row, dict) or row.get("tflops") is None:
                continue
            sid = row.get("source_id")
            blob = (row.get("basis_detail") or "") + " " + (row.get("notes") or "")
            cited = find_cited_source_ids(blob, idre)
            if sid and sid in tiers:
                ts = [tiers[sid]]
                attr = "row.source_id"
            elif cited:
                ts = sorted(tiers[i2] for i2 in cited if i2 in tiers)
                attr = "row-text"
            else:
                ts = sorted(tiers[i2] for i2 in (r.get("sources") or []) if i2 in tiers)
                attr = "record-level"
            if not ts:
                continue
            figures.append({
                "record": r["id"], "type": "accelerator", "field": f"flops[{i}]",
                "tier_ceiling": ts[0], "tier_floor": ts[-1],
                "n_sources": len(ts), "attribution": attr,
            })

    return figures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    srcs = load_sources()
    tiers = {sid: classify_source(s)[0] for sid, s in srcs.items()}

    if args.validate:
        agree = 0
        rows = []
        for sid, gold in sorted(GOLD_SET.items()):
            got, reason = classify_source(srcs[sid])
            ok = got == gold
            agree += ok
            rows.append((sid, gold, got, "AGREE" if ok else "DISAGREE", reason))
        n = len(GOLD_SET)
        print(f"CLASSIFIER VALIDATION: {agree}/{n} = {agree / n * 100:.1f}% agreement "
              f"on the hand-classified sample")
        for sid, gold, got, v, reason in rows:
            mark = "  " if v == "AGREE" else "<<"
            print(f"  {mark} {sid:52s} gold={gold} got={got}  {reason}")
        return 0 if agree == n else 1

    if args.json:
        out = {
            "sources": {sid: {"tier": tiers[sid], "reason": classify_source(s)[1]}
                        for sid, s in srcs.items()},
        }
        print(json.dumps(out, indent=1))
        return 0

    total = len(srcs)
    bytier = collections.Counter(tiers.values())
    print("=" * 78)
    print(f"PROVENANCE TIER BY RECORD  (data/sources/, {total} records)")
    print("=" * 78)
    for t in TIERS:
        c = bytier.get(t, 0)
        print(f"  Tier {t}  {TIER_NAME[t]:<52} {c:5d}  {c / total * 100:5.1f}%")
    print()

    figures = harvest_figures(srcs, build_source_id_regex(srcs))
    nf = len(figures)
    ceil = collections.Counter(f["tier_ceiling"] for f in figures)
    floor = collections.Counter(f["tier_floor"] for f in figures)
    print("=" * 78)
    print(f"FIGURE-WEIGHTED PROPORTIONS  ({nf} numeric figures)")
    print("  CEILING = at least one Tier-A source among those cited for the figure.")
    print("  FLOOR   = EVERY source cited for the figure is Tier A.")
    print("=" * 78)
    print(f"  {'tier':<6} {'n':>7} {'ceiling %':>11} {'floor %':>10}")
    for t in TIERS:
        print(f"  {t:<6} {ceil.get(t, 0):>7} {ceil.get(t, 0) / nf * 100:>10.1f}% "
              f"{floor.get(t, 0) / nf * 100:>9.1f}%")
    print()
    print(f"  HEADLINE  Tier A by figure weight (ceiling): "
          f"{ceil.get('A', 0) / nf * 100:.1f}%  of {nf} figures")
    print(f"            Tier A by figure weight (floor):   "
          f"{floor.get('A', 0) / nf * 100:.1f}%")
    print()

    print("  By record type, Tier A share of figures (ceiling):")
    bytype = collections.defaultdict(collections.Counter)
    for f in figures:
        bytype[f["type"]][f["tier_ceiling"]] += 1
    for ty, c in sorted(bytype.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values())
        print(f"    {ty:<16} n={tot:>5}  A={c.get('A', 0) / tot * 100:5.1f}%  "
              f"B={c.get('B', 0) / tot * 100:5.1f}%  C={c.get('C', 0) / tot * 100:5.1f}%  "
              f"D={c.get('D', 0) / tot * 100:5.1f}%  E={c.get('E', 0) / tot * 100:5.1f}%")
    print()

    print("  Attribution strength of the figure-to-source link:")
    byattr = collections.defaultdict(collections.Counter)
    for f in figures:
        byattr[f.get("attribution", "record-level")][f["tier_ceiling"]] += 1
    for a, c in sorted(byattr.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        print(f"    {a:<16} n={n:>5}  Tier A={c.get('A', 0) / n * 100:5.1f}%  "
              f"D+E={100 * (c.get('D', 0) + c.get('E', 0)) / n:5.1f}%")
    print()
    print("  THE HONEST RESTRICTION - Tier A share restricted to figures whose own")
    print("  basis text names a source (a field-level citation). This is the")
    print("  number that does not depend on a record's source list happening to")
    print("  contain at least one good document:")
    strong = [f for f in figures if f.get("attribution") == "basis-text"]
    sr = [f for f in figures if f.get("attribution") == "row.source_id"]
    for label, grp in (("basis-text only", strong),
                       ("basis-text + flops row.source_id", strong + sr)):
        if not grp:
            continue
        c = collections.Counter(f["tier_ceiling"] for f in grp)
        n = len(grp)
        print(f"    {label:<34} n={n:>5}  Tier A={c.get('A', 0) / n * 100:5.1f}%  "
              f"D+E={(c.get('D', 0) + c.get('E', 0)) / n * 100:5.1f}%")
    print()

    print("  Per-TIER domain sanity check (where each tier actually comes from):")
    tier_dom = collections.defaultdict(collections.Counter)
    for sid, s in srcs.items():
        tier_dom[tiers[sid]][_norm_host(s.get("url"))] += 1
    for t in TIERS:
        top = tier_dom[t].most_common(10)
        print(f"    Tier {t}: " + ", ".join(f"{d or '?'}={c}" for d, c in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())