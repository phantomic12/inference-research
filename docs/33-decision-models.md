# Decision models

A decision model is not a text generator. It reads a state - text, JSON, an image, a video - together with a schema of typed questions, and returns a probability for every allowed option of every question in a **single forward pass**: zero output tokens, no parsing, and one logit per option that a softmax turns into a calibrated answer. That is a different job from writing a reply, so it gets a different ranking and different evidence.

Chat benchmarks cannot rank these models. MMLU, SWE-bench and Terminal-Bench all score generated text; a decision model produces none. Nor does a refusal table mean anything, because the model answers with probabilities rather than prose. The shared suite is the **Decision Index** (Jev Decision Index, maintained by multimodalart and contributors), currently **edition 0.4.1**. It is **chance-corrected**: 0 is random guessing and 100 is perfect, so an index of 51.2 is not "half right" but "well above chance on a five-way, unevenly-weighted task mix".

Within the Decision Index the weight is fixed by the suite, not by us: knowledge 26%, language 26%, retrieval 20%, tools 18%, artifacts 10%. **ECE** is expected calibration error - the gap between the confidence the model prints and the accuracy it achieves - so lower is better, and a model with a strong index and a bad ECE is confidently wrong.


## Ranking - Decision Index 0.4.1 (current board)

| # | Model | Index | Know. | Lang. | Retr. | Tools | Arts | Median ms | p95 ms | ECE | Params | Base |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | [vLLM-SR d3-ultra (Decision 3.0, 27B)](https://huggingface.co/vllm-sr/d3-ultra) | **70.18** | 62.3 | 72.3 | 71.8 | 87.8 | 63.5 | 200 | 674 | 0.067 | 26.1B | `Qwen/Qwen3.8-27B` |
| 5 | [Perplexity Decider v1.2 (27B)](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) | **67.98** | 55.3 | 71.9 | 68.1 | 85.4 | 56.2 | 144 | 1552 | 0.055 | 26.1B | `Qwen/Qwen3.8-27B` |

`*` = self-reported. 0 of 2 rows are the author's own run and the upstream board has not reproduced them; every other row is board-validated on one NVIDIA RTX PRO 6000.


## Previous board edition (not comparable)

These rows were scored on Decision Index 0.2.1. The 0.4.1 board rescored the panel (GSM8K rebuilt, ForecastBench retired, private-test halves added), so these indices are **not on the same scale** as the table above and must not be ranked against it or subtracted from it. They are kept because each is the only published number for that model.

| Model | Index (0.2.1) | Know. | Lang. | Retr. | Tools | Arts | Median ms | p95 ms | ECE | Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide) * | **62.48** | — | — | — | — | — | — | — | — | `google/gemma-4-26B-A4B-it` |
| [Clef](https://huggingface.co/Cloudflare/clef) * | **61.21** | 51.2 | 61.2 | 62.5 | 81.2 | 47.8 | 209 | 239 | — | `Qwen/Qwen3.8-27B` |
| Jev Closed | **57.91** | 51.4 | 62.0 | 55.4 | 75.1 | 37.7 | 524 | 536 | 0.074 | — |
| [Surogate Rune 26B-A4B v3 [bf16]](https://huggingface.co/surogate/rune-26b-a4b-GGUF) | **57.44** | 43.4 | 63.1 | 63.5 | 71.2 | 41.9 | 121 | 683 | 0.120 | `google/gemma-4-26B-A4B-it` |
| Decider chat · Gemma-4-31B | **57.33** | 44.3 | 60.4 | 63.1 | 75.6 | 38.3 | 109 | 1115 | 0.047 | `google/gemma-4-31B` |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) * | **57.07** | 50.4 | 52.6 | 52.3 | 82.0 | 49.9 | 39 | 122 | — | `Qwen/Qwen3.5-9B` |
| [AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) | **56.40** | 40.9 | 63.5 | 54.9 | 79.3 | 39.4 | 101 | 1053 | 0.018 | `Qwen/Qwen3.8-27B` |
| simple-jev · Qwen3.8-27B (featherless) | **55.74** | 36.6 | 62.1 | 63.2 | 76.2 | 36.5 | 373 | 2797 | 0.113 | `Qwen/Qwen3.8-27B` |
| [Jebadiah 27B](https://huggingface.co/frontier-infra/jebadiah-27b) | **54.67** | 38.8 | 60.7 | 53.9 | 78.1 | 38.7 | 110 | 591 | 0.014 | `Qwen/Qwen3.8-27B` |
| [Eikos-27B [FP8]](https://huggingface.co/caiovicentino1/Eikos-27B-FP8) | **53.13** | 39.9 | 54.3 | 55.9 | 74.4 | 39.8 | 130 | 510 | 0.057 | `Qwen/Qwen3.8-27B` |
| reflex 27B [FP8 · wide choice] | **52.16** | 35.1 | 54.2 | 57.8 | 74.1 | 39.6 | 108 | 1240 | 0.024 | `Qwen/Qwen3.8-27B` |
| Decider chat · Qwen3.6-27B | **51.35** | 37.0 | 57.1 | 52.2 | 71.4 | 35.1 | 84 | 966 | 0.021 | `Qwen/Qwen3.6-27B` |
| [Winnow-12B [Q8_0]](https://huggingface.co/EldanRing/Winnow-12B) | **50.02** | 33.8 | 56.0 | 54.0 | 71.0 | 30.0 | 72 | 356 | 0.168 | `google/gemma-4-12B` |
| JoshuaSP diffusiongemma (open-jev) | **49.47** | 32.7 | 53.5 | 58.4 | 70.2 | 26.7 | 260 | 537 | 0.215 | `google/diffusiongemma-26B-A4B-it` |
| Jevfire | **49.37** | 30.5 | 53.3 | 56.2 | 72.3 | 32.2 | 90 | 448 | 0.052 | `Qwen/Qwen3.8-27B` |
| [Decider 35B-A3B [NVFP4]](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4) | **47.11** | 31.8 | 55.5 | 54.7 | 56.5 | 32.6 | 101 | 211 | 0.023 | `Qwen/Qwen3.5-35B-A3B-Base` |
| [JPT-9B](https://huggingface.co/kirp/jpt-9b) | **46.89** | 31.7 | 56.7 | 44.6 | 67.0 | 28.6 | 149 | 462 | 0.077 | `Qwen/Qwen3.5-9B-Base` |
| [Decision 1.0 Lux](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) | **43.49** | 30.9 | 48.0 | 50.0 | 57.2 | 26.4 | 51 | 348 | 0.076 | `Qwen/Qwen3.5-9B-Base` |
| [JPT-4B](https://huggingface.co/kirp/jpt-4b) | **43.04** | 28.7 | 52.5 | 45.0 | 57.2 | 25.8 | 139 | 406 | 0.079 | `Qwen/Qwen3.5-4B-Base` |
| [Jet v6.2](https://huggingface.co/michaljach/jet) | **42.60** | 28.7 | 43.9 | 48.2 | 62.9 | 27.0 | 47 | 371 | 0.141 | `Qwen/Qwen3.5-4B-Base` |
| [Xor](https://huggingface.co/juspay/xor) | **41.48** | 34.3 | 55.6 | 23.0 | 55.1 | 35.6 | 140 | 420 | 0.015 | `Qwen/Qwen3.6-35B-A3B` |
| [Hopper (G) 1.2](https://huggingface.co/HopitAI/hopper-g) | **40.77** | 25.2 | 44.8 | 45.4 | 60.1 | 26.0 | 23 | 238 | 0.093 | `Qwen/Qwen3.5-4B-Base` |
| [Decider 4B](https://huggingface.co/Mapika/decider-4b) | **40.70** | 25.7 | 46.0 | 44.7 | 58.6 | 25.0 | 13 | 189 | 0.084 | `Qwen/Qwen3.5-4B-Base` |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | **40.53** | 25.5 | 50.5 | 35.8 | 60.8 | 26.1 | 55 | 628 | 0.161 | `google/gemma-4-12B` |
| djev | **40.28** | 26.4 | 48.8 | 41.8 | 51.7 | 30.4 | 84 | 211 | 0.212 | `google/diffusiongemma-26B-A4B-it` |
| [Winnow-E4B [Q8_0]](https://huggingface.co/EldanRing/Winnow-E4B) | **39.89** | 22.3 | 45.1 | 43.8 | 62.5 | 22.8 | 45 | 188 | 0.058 | `google/gemma-4-E4B` |
| [Bespoke Nimble 9B v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) | **39.57** | 29.5 | 45.1 | 34.9 | 59.2 | 24.7 | 77 | 621 | 0.024 | `Qwen/Qwen3.5-9B-Base` |
| [JevK5](https://huggingface.co/alibiserikbay/JevK5) | **38.81** | 23.5 | 43.3 | 46.2 | 52.1 | 27.8 | 22 | 226 | 0.027 | `Qwen/Qwen3.5-4B-Base` |
| [lev](https://huggingface.co/interfaze-ai/lev) | **38.54** | 22.0 | 38.5 | 47.9 | 57.5 | 28.0 | 71 | 688 | 0.064 | `Qwen/Qwen3.5-4B-Base` |
| [Kev 9B](https://huggingface.co/jaredpalmer/kev-9b) | **38.48** | 26.2 | 41.7 | 43.7 | 54.5 | 22.4 | 51 | 188 | 0.138 | `Qwen/Qwen3.5-9B-Base` |
| [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | **37.81** | 23.7 | 41.9 | 40.3 | 55.7 | 26.1 | 44 | 115 | 0.028 | `Qwen/Qwen3.5-4B-Base` |
| razorback16 openjev diffusiongemma (NVFP4, vLLM) [one read, NVFP4] | **37.25** | 26.2 | 44.9 | 37.0 | 47.2 | 28.5 | 38 | 118 | 0.232 | `google/diffusiongemma-26B-A4B-it` |
| [NeoHorse-Jev-4B](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B) | **36.75** | 22.6 | 39.4 | 43.9 | 58.9 | 11.8 | 49 | 274 | 0.104 | `Qwen/Qwen3.5-4B-Base` |
| [Solomon v1.1](https://huggingface.co/DoccyHealth/Solomon) | **36.43** | 22.1 | 57.3 | 26.0 | 47.0 | 21.2 | 224 | 1177 | 0.081 | `Qwen/Qwen3.8-27B` |
| [Kev 4B](https://huggingface.co/jaredpalmer/kev-4b) | **34.64** | 22.9 | 35.3 | 41.0 | 52.6 | 17.9 | 52 | 141 | 0.176 | `Qwen/Qwen3.5-4B-Base` |
| [Decision 1.0 Nox](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox) | **34.36** | 21.4 | 36.6 | 44.3 | 49.6 | 14.4 | 53 | 242 | 0.140 | `Qwen/Qwen3.5-4B-Base` |
| Jobe | **32.35** | 17.7 | 34.5 | 36.7 | 48.8 | 25.7 | 54 | 2910 | 0.123 | `Qwen/Qwen3.5-4B-Base` |
| mmastrac diffusiongemma (vLLM PR 57250) | **32.24** | 24.9 | 45.9 | 24.2 | 34.0 | 28.8 | 127 | 275 | 0.202 | `google/diffusiongemma-26B-A4B-it` |
| [open-jev (pngwn)](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b) | **29.91** | 19.9 | 35.5 | 29.1 | 45.7 | 14.1 | 112 | 724 | 0.059 | `Qwen/Qwen3.5-4B-Base` |
| [Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental) | **29.24** | 23.5 | 39.2 | 15.5 | 41.8 | 22.9 | 36 | 166 | 0.104 | `Qwen/Qwen3.5-4B-Base` |
| [Decider 2B [FP8]](https://huggingface.co/Mapika/decider-2b) | **28.97** | 14.9 | 32.6 | 37.3 | 42.4 | 14.6 | 8 | 135 | 0.077 | `Qwen/Qwen3.5-2B-Base` |
| openvons | **28.42** | 20.8 | 30.1 | 25.4 | 44.6 | 20.3 | 21 | 113 | 0.371 | `Qwen/Qwen3-4B-Instruct-2507` |
| [this-that 1.2](https://huggingface.co/flock-io/this-that-model-1.2) | **28.14** | 15.1 | 31.9 | 32.8 | 45.0 | 11.8 | 44 | 57 | 0.198 | `Mapika/decider-2b` |
| [Metask-Jev-4B](https://huggingface.co/wayfind/metask-jev-4b-policy-mix) | **26.89** | 21.0 | 39.4 | 12.9 | 37.6 | 18.3 | 57 | 369 | 0.103 | `Qwen/Qwen3.5-4B-Base` |
| SemIf | **25.94** | 17.1 | 33.8 | 15.7 | 39.1 | 24.9 | 113 | 215 | 0.099 | `Qwen/Qwen3.5-4B-Base` |
| [Decision 1.0 Sol](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol) | **25.32** | 11.5 | 25.4 | 32.7 | 46.1 | 8.2 | 38 | 101 | 0.115 | `Qwen/Qwen3.5-2B-Base` |
| mini-jev | **20.98** | 17.6 | 28.6 | 8.7 | 28.4 | 21.1 | 67 | 178 | 0.346 | `Qwen/Qwen3-4B-Instruct-2507` |
| [Bosun v3.1 1.7B](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b) | **20.10** | 4.7 | 12.4 | 42.8 | 34.8 | 7.5 | 40 | 174 | 0.130 | `Qwen/Qwen3-1.7B-Base` |
| [Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B) | **19.38** | 13.8 | 20.6 | 15.7 | 33.3 | 12.7 | 34 | 58 | 0.061 | `Qwen/Qwen3.5-2B-Base` |
| [JPT-0.8B](https://huggingface.co/kirp/jpt-0.8b) | **19.22** | 9.9 | 23.0 | 16.4 | 36.2 | 8.0 | 118 | 341 | 0.069 | `Qwen/Qwen3.5-0.8B-Base` |
| [Decision 1.0 Eos](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) | **18.41** | 7.4 | 14.6 | 30.6 | 32.1 | 7.5 | 40 | 84 | 0.083 | `Qwen/Qwen3.5-0.8B-Base` |
| [Kev 0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | **14.60** | 6.1 | 11.3 | 19.1 | 31.5 | 5.1 | 42 | 96 | 0.074 | `Qwen/Qwen3.5-0.8B-Base` |
| [Bosun v3.1 0.6B](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b) | **14.32** | 2.8 | 3.8 | 34.2 | 28.4 | 5.6 | 39 | 107 | 0.142 | `Qwen/Qwen3-0.6B-Base` |
| [Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental) | **12.85** | 7.3 | 14.1 | 4.9 | 32.0 | 5.1 | 28 | 134 | 0.126 | `Qwen/Qwen3.5-0.8B-Base` |
| [Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) | **11.94** | 8.0 | 10.3 | 6.4 | 28.4 | 7.2 | 34 | 39 | 0.025 | `Qwen/Qwen3.5-0.8B-Base` |
| [MoJev](https://huggingface.co/MoLeMo-Lab/mojev) | **11.69** | 5.6 | 9.1 | 16.2 | 21.9 | 6.7 | 46 | 55 | 0.125 | `Qwen/Qwen3.5-0.8B-Base` |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | **11.21** | 2.4 | 7.5 | 25.3 | 14.8 | 8.9 | 23 | 229 | 0.088 | `fastino/gliner2-large-v1` |
| [Lavoir](https://huggingface.co/moganai/lavoir) | **8.69** | 3.5 | 10.4 | 12.0 | 13.0 | 3.2 | 20 | 50 | 0.149 | `answerdotai/ModernBERT-large` |
| jeff | **8.04** | 3.0 | 10.4 | 21.3 | 1.2 | 0.8 | 22 | 68 | 0.097 | `knowledgator/gliformer-large-v1` |
| [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | **7.40** | 3.1 | 5.2 | 11.3 | 14.0 | 4.3 | 47 | 131 | 0.323 | `Qwen/Qwen3-8B-Base` |
| [GLiNER 2.5 base full fine-tune](https://huggingface.co/fastino/gliner2.5-base-v1) | **6.76** | 2.1 | 7.5 | 9.7 | 11.2 | 3.1 | 14 | 62 | 0.367 | — |
| [LFM2.5-2.6B-RLCD](https://huggingface.co/monotykamary/LFM2.5-2.6B-RLCD) | **6.76** | 7.4 | 5.4 | 10.5 | 4.4 | 5.3 | 39 | 95 | 0.255 | `LiquidAI/LFM2.5-2.6B-Base` |
| [Decision 1.0 Kai](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai) | **6.52** | 3.8 | 2.9 | 9.3 | 14.2 | 3.1 | 30 | 107 | 0.185 | `jhu-clsp/mmBERT-base` |
| [Laya full fine-tune](https://huggingface.co/convaiinnovations/laya) | **6.04** | 3.6 | 9.0 | 7.1 | 5.8 | 2.9 | 6 | 223 | 0.140 | — |
| [Julia 1](https://huggingface.co/SupersonicLabs/Julia-1) | **5.54** | 6.6 | 1.4 | 9.9 | 7.2 | 1.7 | 6 | 15 | 0.420 | `jhu-clsp/mmBERT-small` |
| system-one-gemma | **5.07** | 1.3 | 1.1 | 17.2 | 3.3 | 4.1 | 32 | 748 | 0.239 | `google/gemma-3-270m` |
| [Decision 1.0 Lex](https://huggingface.co/llm-semantic-router/decision-1.0-lex) | **4.54** | 1.9 | 2.4 | 6.3 | 10.9 | 1.9 | 30 | 106 | 0.169 | `jhu-clsp/mmBERT-base` |
| [GLiNER 2.5 multilingual full fine-tune](https://huggingface.co/fastino/gliner2.5-multi-v1) | **4.26** | 0.9 | 6.4 | 5.3 | 5.9 | 2.5 | 14 | 92 | 0.254 | — |
| [GLiNER 2.5 small full fine-tune](https://huggingface.co/fastino/gliner2.5-small-v1) | **3.82** | 1.8 | 3.8 | 4.3 | 7.0 | 2.7 | 14 | 35 | 0.161 | — |
| [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | **3.78** | 1.4 | 4.7 | 0.9 | 7.0 | 7.3 | 44 | 75 | 0.205 | `Qwen/Qwen2.5-1.5B` |
| [Lumma-Fev-0.6B](https://huggingface.co/FrontiersMind/Lumma-fev-0.6b) | **2.97** | 2.3 | 0.9 | 8.9 | 1.1 | 1.4 | 21 | 220 | 0.236 | `FrontiersMind/Lumma-0.6B-Base` |
| [Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) | **1.87** | 0.4 | 1.5 | 0.0 | 6.6 | 2.0 | 11 | 60 | 0.154 | `knowledgator/gliclass-modern-base-v2.0` |
| [Lumma-Fev-0.1B](https://huggingface.co/FrontiersMind/Lumma-fev-0.1b) | **1.78** | 1.5 | 0.6 | 2.5 | 2.6 | 2.6 | 21 | 113 | 0.157 | `FrontiersMind/Nandi-Mini-150M` |
| [LFM2.5-350M-RLCD full fine-tune](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD) | **1.38** | 0.5 | 1.0 | 0.0 | 3.8 | 3.0 | 27 | 174 | 0.568 | — |


## How to read this against the worth-running table

- **Do not compare an index to a capability score.** They are different scales answering different questions, and a 61.2 here is not "61.2%% of GPT" - it is a chance-corrected composite over 38 decision benchmarks.
- **Latency is part of the score, not a footnote.** A decision model returns in one pass, so the useful figure is median milliseconds per decision, and the 38 ms - 524 ms spread below is a bigger practical difference than the index spread. p95 matters more than median for an agent loop that stalls.
- **Params decide where it runs.** The 3B - 9B band is the sweet spot for edge and single-GPU deployment; 26B - 36B costs a mid-range GPU for a few index points.
- **Coverage is not exhaustive.** Only models currently scored in the Decision Index appear here. Community decision models that the suite has not scored are absent rather than bottom-ranked.


## Ingestion contract

Decision models are records with `task: decision` and a `base_model`. They carry their Decision Index row as benchmarks named for the edition (e.g. `Decision Index 0.4.1 full score (balanced_skill)`), the five axis sub-scores, and an `edition` field so two board versions never share one rank column, and they are **excluded from the worth-running ranking by construction** - not down-weighted, excluded - because that view averages text benchmarks. Adding one means: record the index, its five sub-scores, median and p95 latency, and ECE, and mark `vendor_claim: true` only when the number is the author's own run.

