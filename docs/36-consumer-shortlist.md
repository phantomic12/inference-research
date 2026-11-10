# Consumer-hardware shortlist (2026-11-10)

The smartest new open-weight models that fit hardware a person can buy, with the quant to run and
what that quant costs. Regenerate the inputs with `tools/build_joined_views.py` (capability +
footprint) and `tools/quant_popularity.py --build` (quant evidence); this file is a dated snapshot,
not a generated artifact.

## Filters

- **New**: `release_year == 2026` only.
- **Generative text/vision models only**: embedders, rerankers, ASR/TTS and image generators are
  excluded because chat/math/code benchmarks cannot rank them.
- **Runnable**: fits at 4-bit weights + 8K context inside the class — `single_user_gib <= class`.
  For MoE, "offload" requires total weights **≤ 64 GiB** and ≤ 12B active params, which is what
  `--cpu-moe` / K-Transformers can stream from 64 GB of system RAM.
- **Smart**: `capability` = evidence-weighted share of the best-known score per benchmark, so 88.7
  means ~89% of the frontier number on the benchmarks it was measured on. It is **relative, and
  only as good as the card that published it**.

## 16 GiB class — the value slot (fits a 16 GB card, or 12 GB + 4 GB shared)

| model | params | 4-bit footprint | cap | headline | quant to run | quant cost |
|---|---|---|---|---|---|---|
| **VeriLoop E2** | 27.8B dense | 14.4 GiB | **88.7** | GPQA 93.9, AIME-26 98.3, SWE-Pro 76.2, TB2.1 88.8 | `tsinghua-sigs-robot-lab/VeriLoop-E2-GGUF` | lossless, median −0.8% over 14 benchmarks |
| **Raon-1 mini preview** | 28.8B / 4.3B active | 14.5 GiB | 86.3 | AIME-26 93.3, GPQA 84.3, SWE-Verified 70.8, LCB 81.6 | `KRAFTON/Raon-1-mini-preview-gguf` | unmeasured |
| **Swift-1.5-Qwen3.8-27b** | 27.8B dense | 14.4 GiB | 84.0 | LCB 8x, ×7 coding/agentic | `unsloth/Qwen3.8-27B-GGUF` family | unmeasured |
| **Qwen3.8-27B** | 27.8B dense | 14.4 GiB | 81.5 | GPQA 89.2, LCB 90.3, SWE-V 77.2 | `pearsonkyle/…-GPTQ-W4A16` (4.00 bpw) | **100.5%** — parity |
| | | | | | `ISTA-DASLab/…-GSQ-RCO-GGUF` (3.50 bpw) | 96.6% |
| **Gemma 4 26B A4B** | 25.8B / 4.0B active | 14.6 GiB | 76.3 | multimodal | `bg-digitalservices/…-NVFP4` (4.00 bpw) | 97.6% |
| **LFM2.5-24B-A2B** | 24.0B / 2.5B active | 12.2 GiB | 68.3 | 16 benchmarks | `GestaltLabs/…-NVFP4-W4A16` (4.00 bpw) | 100% |
| Nemotron-3.5-Lightning-30B-A3B | 30.0B / 3.0B active | 15.0 GiB | 61.7 | — | 3-bit GPTQ | **82%** — avoid |

VeriLoop E2 is the top of this class on capability and it is a **Qwen3.8-27B fine-tune** (Tsinghua
SIGS Robot Lab). Its 93.9 GPQA / 98.3 AIME are the lab's own runs — but each score links a public
evaluation-evidence dataset with prompts and outputs, which is more than most cards offer. Nothing
independent has reproduced them. Its quant coverage is thin in the community (a 404-download NVFP4,
a 67-download W4A16), so the official GGUF is the build to use.

## 24 GiB class (fits a 24 GB card; the 3090 / MI50 tier)

| model | params | 4-bit footprint | cap | headline | quant to run | quant cost |
|---|---|---|---|---|---|---|
| **Gemma 4 31B** | 31.3B dense | 22.5 GiB | **86.9** | MMLU-Pro 85.2, GPQA 84.3, LCB 80.0, IFEval 98.9 | `QuantTrio/…-AWQ` or `unsloth/…-GGUF` at Q4 | ⚠ only measured build is 2.85 bpw = **59%** |
| **Nanbeige4.3-33B** | 33.1B dense | 18.2 GiB | 84.6 | TB2.1 80.9, DeepSWE 65.5, SWE-Pro 60.3 (19 benchmarks) | `bartowski/…-GGUF` | unmeasured |
| **Sage-A1.5** | 36.0B / 3.0B active | 18.1 GiB | 82.3 | 16 benchmarks | — | unmeasured |
| **Qwen3.5-35B-A3B** | 35.0B / 3.0B active | 17.7 GiB | 80.6 | 15 benchmarks | `apolo13x/…-NVFP4` (4.00 bpw) | 99% |

Gemma 4 31B is the smartest 24 GB model and the **one confirmed quant trap**: the only build with a
published base comparison (`pearsonkyle/gemma4-31b-imatrix-mtp-GGUF`, 2.85 bpw) retains 59% — that
is a broken quant, not a small one. Run it at ≥4 bpw or not at all.

## MoE offload class (≤64 GiB weights, streams from RAM)

| model | params | 4-bit footprint | cap | quant |
|---|---|---|---|---|
| **Ornith-1.5-122B** | 125.1B / 9.3B active | 62.7 GiB | 86.3 | `unsloth/…-GGUF` (unmeasured) |
| **Qwen3.5-122B-A10B** | 121.3B / 10.0B active | 60.9 GiB | 83.1 | catalog retention: lossless |
| **Nemotron 3.5 Super VL 120B** | 120.0B / 12.0B active | 60.1 GiB | 79.4 | 3-bit NVFP4, 99% |

These need 64 GB of RAM and a GPU for attention+shared experts. If the box has 32 GB, the honest
answer is the 16/24 GiB class — an offloaded 122B runs at a few tokens/sec and loses more to
scheduling than a 27B loses in quality.

## 8 GiB class

`granite-4.1-3b` (82.7, 7 benchmarks, not thin) is the only one there that is both small and
solidly evidenced. `LFM3-VL-2B` shows 86.6 but that number is built from vision benchmarks where
it happens to top a narrow field (`thin: true`, math axis pinned at 100 by a single MathVista
result) — it is a good document/UI VLM, not a general reasoner. `Qwen3.5-4B` (78.7) is the safest
all-rounder at this size.

## What this list does not claim

- **Every capability number here rests on vendor-run benchmarks.** No independent reproduction
  exists for these releases; `thin` marks the ones with too few benchmarks to trust the axis
  weighting.
- **Quant retention exists for a minority.** Where it says "unmeasured", nobody published a
  base-vs-quant comparison — that is a statement about the ecosystem, not a quality claim.
- **Capability is relative to the best-known per benchmark**, so it rewards models measured on
  benchmarks where the frontier is weak. Compare within a class, not across years.
