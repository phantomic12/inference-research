# 27 — Paper reading map: the 282 paper records by topic and adoption

<!-- written 2026-10-12 against 281 paper records. Counts are a snapshot; regenerate the tables from data/papers/ if they drift. 2026-10-18: quantization 36 -> 37, research-only 132 -> 133, for [[papers/qmoe-sub-1-bit-compression]]. -->
<!-- 2026-10-18: moe 15 -> 16, research-only 133 -> 134, for [[papers/specprefetch-window-aware-expert-prefetch]]. The tables below remain the 2026-10-12 snapshot plus the QMoE row. -->
<!-- 2026-10-18: moe 16 -> 17, research-only 134 -> 135, for [[papers/seqmoe-graph-compatible-moe-offload]]. -->

The repo holds a paper record for every method that shapes an inference decision, but
until now almost none were cited from `docs/`. This map makes them navigable: one
section per `category`, papers ordered by `adoption` (what a reader can actually run
today first), then by year. The adoption field is the record's own claim — see each
record's `adoption` evidence before treating it as settled.

## How to read it

| adoption | papers | meaning |
|---|---|---|
| `in-production` | 76 | the method ships in a production system named in the record |
| `in-upstream-engine` | 67 | merged into an open engine (vLLM, SGLang, llama.cpp, TRT-LLM, ...) |
| `research-only` | 135 | paper results only; buys nothing without custom kernels or code |
| `abandoned` | 5 | superseded or unmaintained |
| `unknown` | 1 | not established from sources |

**Where to start.** For serving, the canonical spine is Orca's iteration-level
scheduling ([[papers/orca-osdi22-iteration-scheduling]]), PagedAttention
([[papers/pagedattention-vllm]]) and Sarathi-Serve's chunked prefill
([[papers/sarathi-serve]]). For memory, MQA/GQA/MLA
([[papers/multi-query-attention]], [[papers/grouped-query-attention]],
[[papers/deepseek-v2-mla]]). For quantization, LLM.int8(), GPTQ, AWQ and the MX formats
([[papers/llmint8]], [[papers/paper-gptq]], [[papers/paper-awq]], [[papers/mx-formats]]).
For decoding, the two speculative-decoding originals
([[papers/leviathan-speculative-decoding]], [[papers/chen-speculative-sampling-deepmind]]).

**What the map shows that no single record does.** Pruning (7) and long-context
(7) have **zero** papers past `research-only`; scheduling has one
([[papers/sarathi-serve]]). Treat a speedup claim from those categories as a paper
result until an engine record says otherwise.

## Topics

### quantization (37)

4 in-production, 4 in-upstream-engine, 29 research-only. Synthesis: [23-quantization-selection.md](23-quantization-selection.md), [04-quant-selection.md](04-quant-selection.md).

| paper | year | adoption |
|---|---|---|
| [[papers/llmint8|LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale]] | 2022 | `in-production` |
| [[papers/paper-gptq|GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers]] | 2022 | `in-production` |
| [[papers/mx-formats|Microscaling Data Formats for Deep Learning]] | 2023 | `in-production` |
| [[papers/paper-awq|AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration]] | 2024 | `in-production` |
| [[papers/paper-smoothquant|SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models]] | 2023 | `in-upstream-engine` |
| [[papers/marlin-mixed-precision-autoregressive-inference|MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models]] | 2024 | `in-upstream-engine` |
| [[papers/paper-aqlm|Extreme Compression of Large Language Models via Additive Quantization]] | 2024 | `in-upstream-engine` |
| [[papers/paper-quarot|QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs]] | 2024 | `in-upstream-engine` |
| [[papers/zeroquant|ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers]] | 2022 | `research-only` |
| [[papers/bitnet|BitNet: Scaling 1-bit Transformers for Large Language Models]] | 2023 | `research-only` |
| [[papers/moqe|Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness]] | 2023 | `research-only` |
| [[papers/paper-zeroquant-v2|ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation]] | 2023 | `research-only` |
| [[papers/quip|QuIP: 2-Bit Quantization of Large Language Models With Guarantees]] | 2023 | `research-only` |
| [[papers/qmoe-sub-1-bit-compression|QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models]] | 2023 | `research-only` |
| [[papers/mx-ftptq|Post Training Quantization of Large Language Models with Microscaling Formats]] | 2024 | `research-only` |
| [[papers/paper-atom|Atom: Low-bit Quantization for Efficient and Accurate LLM Serving]] | 2024 | `research-only` |
| [[papers/paper-bitnet-b158|The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits]] | 2024 | `research-only` |
| [[papers/paper-flute|Fast Matrix Multiplications for Lookup Table-Quantized LLMs]] | 2024 | `research-only` |
| [[papers/paper-llm-qat|LLM-QAT: Data-Free Quantization Aware Training for Large Language Models]] | 2024 | `research-only` |
| [[papers/paper-lqer|LQER: Low-Rank Quantization Error Reconstruction for LLMs]] | 2024 | `research-only` |
| [[papers/paper-omniquant|OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models]] | 2024 | `research-only` |
| [[papers/paper-quip-sharp|QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks]] | 2024 | `research-only` |
| [[papers/paper-squeezellm|SqueezeLLM: Dense-and-Sparse Quantization]] | 2024 | `research-only` |
| [[papers/quantmoe-bench|QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts]] | 2024 | `research-only` |
| [[papers/spqr|SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression]] | 2024 | `research-only` |
| [[papers/eaquant|EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization]] | 2025 | `research-only` |
| [[papers/moequant|MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance]] | 2025 | `research-only` |
| [[papers/nvfp4-pretraining|Pretraining Large Language Models with NVFP4]] | 2025 | `research-only` |
| [[papers/paper-spinquant|SpinQuant: LLM Quantization with Learned Rotations]] | 2025 | `research-only` |
| [[papers/qserve|QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving]] | 2025 | `research-only` |
| [[papers/sageattention|SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration]] | 2025 | `research-only` |
| [[papers/sageattention2|SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization]] | 2025 | `research-only` |
| [[papers/alphaq|AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization]] | 2026 | `research-only` |
| [[papers/flashattention-4-fp4|Hardware-Aware FP4 FlashAttention-4]] | 2026 | `research-only` |
| [[papers/tetra|Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per Parameter]] | 2026 | `research-only` |
| [[papers/thinquant|ThinQuant: Scalable Rotation Learning for Weight and Activation Quantization of LLMs]] | 2026 | `research-only` |
| [[papers/vsraq|Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models]] | 2026 | `research-only` |

### serving-systems (31)

7 in-production, 2 in-upstream-engine, 22 research-only. Synthesis: [12-serving-workflows.md](12-serving-workflows.md), [21-deployment-topology.md](21-deployment-topology.md).

| paper | year | adoption |
|---|---|---|
| [[papers/clipper|Clipper: A Low-Latency Online Prediction Serving System]] | 2017 | `in-production` |
| [[papers/deepspeed-inference|DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale]] | 2022 | `in-production` |
| [[papers/orca-osdi22-iteration-scheduling|Orca: A Distributed Serving System for Transformer-Based Generative Models]] | 2022 | `in-production` |
| [[papers/pagedattention-vllm|Efficient Memory Management for Large Language Model Serving with PagedAttention]] | 2023 | `in-production` |
| [[papers/deepspeed-fastgen|DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference]] | 2024 | `in-production` |
| [[papers/mooncake-kimi|Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving]] | 2024 | `in-production` |
| [[papers/paper-sglang|SGLang: Efficient Execution of Structured Language Model Programs]] | 2024 | `in-production` |
| [[papers/punica-multi-tenant-lora-serving|Punica: Multi-Tenant LoRA Serving]] | 2024 | `in-upstream-engine` |
| [[papers/flashinfer-paper|FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving]] | 2025 | `in-upstream-engine` |
| [[papers/clockwork|Serving DNNs like Clockwork: Performance Predictability from the Bottom Up]] | 2020 | `research-only` |
| [[papers/alpaserve|AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving]] | 2023 | `research-only` |
| [[papers/flexgen-single-gpu-high-throughput|FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU]] | 2023 | `research-only` |
| [[papers/w4p-s-lora-multi-tenant-adapter-serving|S-LoRA: Serving Thousands of Concurrent LoRA Adapters]] | 2023 | `research-only` |
| [[papers/hetegen-heterogeneous-parallel-inference|HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices]] | 2024 | `research-only` |
| [[papers/infinigen|InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management]] | 2024 | `research-only` |
| [[papers/llm-in-a-flash-limited-memory|LLM in a flash: Efficient Large Language Model Inference with Limited Memory]] | 2024 | `research-only` |
| [[papers/muxserve|MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving]] | 2024 | `research-only` |
| [[papers/paper-distserve|DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving]] | 2024 | `research-only` |
| [[papers/parrot|Parrot: Efficient Serving of LLM-based Applications with Semantic Variable]] | 2024 | `research-only` |
| [[papers/powerinfer-consumer-gpu-serving|PowerInfer: Fast Large Language Model Inference with a Consumer-grade GPU]] | 2024 | `research-only` |
| [[papers/splitwise|Splitwise: Efficient generative LLM inference using phase splitting]] | 2024 | `research-only` |
| [[papers/w4p-loongserve-elastic-sequence-parallelism|LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism]] | 2024 | `research-only` |
| [[papers/vattention|vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention]] | 2025 | `research-only` |
| [[papers/w4p-ess-latent-cache-offload-deepseek|ESS: An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp]] | 2025 | `research-only` |
| [[papers/flashinfer-bench|FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems]] | 2026 | `research-only` |
| [[papers/joule-point-energy-optimal-power-cap|The Joule Point: an Energy-Optimal Operating Point for AI Inference]] | 2026 | `research-only` |
| [[papers/k8s-predictive-autoscaling-llm|Decomposing Predictive Kubernetes Autoscaling for Large Language Model Serving Under Long Startup Delays]] | 2026 | `research-only` |
| [[papers/sustainability-dimensions-llm-serving|Beyond Energy: When Sustainability Dimensions Reshape LLM Serving Decisions (PRISM)]] | 2026 | `research-only` |
| [[papers/token-latency-fairness|Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving]] | 2026 | `research-only` |
| [[papers/where-does-the-energy-go-blackwell-profiling|Where Does the Energy Go? Profiling LLM Agent Inference on Blackwell GPUs]] | 2026 | `research-only` |
| [[papers/xwind-renewable-site-inference-router|XWind: A Cross-site Router for Large Language Model Inference Serving at Renewable Energy Farms]] | 2026 | `research-only` |

### attention (29)

4 in-production, 11 in-upstream-engine, 10 research-only, 4 abandoned. Synthesis: [02-flop-map.md](02-flop-map.md), [13-model-compatibility.md](13-model-compatibility.md).

| paper | year | adoption |
|---|---|---|
| [[papers/mistral-7b-gqa-swa|Mistral 7B]] | 2023 | `in-production` |
| [[papers/minimax-01|MiniMax-01: Scaling Foundation Models with Lightning Attention]] | 2025 | `in-production` |
| [[papers/moba|MoBA: Mixture of Block Attention for Long-Context LLMs]] | 2025 | `in-production` |
| [[papers/paper-minimax-m1|MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention]] | 2025 | `in-production` |
| [[papers/linear-transformers-fast-weight-programmers|Linear Transformers Are Secretly Fast Weight Programmers]] | 2021 | `in-upstream-engine` |
| [[papers/flashattention|FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness]] | 2022 | `in-upstream-engine` |
| [[papers/flashattention-2|FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning]] | 2024 | `in-upstream-engine` |
| [[papers/flashattention-3|FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision]] | 2024 | `in-upstream-engine` |
| [[papers/flashdecoding|FlashDecoding++: Faster Large Language Model Inference on GPUs]] | 2024 | `in-upstream-engine` |
| [[papers/flexattention|FlexAttention: A Programming Model for Generating Optimized Attention Kernels]] | 2025 | `in-upstream-engine` |
| [[papers/gated-deltanet|Gated Delta Networks: Improving Mamba2 with Delta Rule]] | 2025 | `in-upstream-engine` |
| [[papers/kimi-linear|Kimi Linear: An Expressive, Efficient Attention Architecture]] | 2025 | `in-upstream-engine` |
| [[papers/nsa-native-sparse|Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-deepseek-sparse-attention-dsa|DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (introducing DeepSeek Sparse Attention, DSA)]] | 2025 | `in-upstream-engine` |
| [[papers/flashattention-4|FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling]] | 2026 | `in-upstream-engine` |
| [[papers/linear-transformers|Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention]] | 2020 | `research-only` |
| [[papers/sau|Sparse Attention with Linear Units]] | 2021 | `research-only` |
| [[papers/blockwise-parallel-transformer|Blockwise Parallel Transformer for Large Context Models]] | 2023 | `research-only` |
| [[papers/hyena|Hyena Hierarchy: Towards Larger Convolutional Language Models]] | 2023 | `research-only` |
| [[papers/striped-attention|Striped Attention: Faster Ring Attention for Causal Transformers]] | 2023 | `research-only` |
| [[papers/distflashattn|DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training]] | 2024 | `research-only` |
| [[papers/hawk-griffin-gated-linear-recurrence|Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models]] | 2024 | `research-only` |
| [[papers/ring-attention|Ring Attention with Blockwise Transformers for Near-Infinite Context]] | 2024 | `research-only` |
| [[papers/simple-linear-attention|Simple linear attention language models balance the recall-throughput tradeoff]] | 2024 | `research-only` |
| [[papers/diff-transformer|Differential Transformer]] | 2025 | `research-only` |
| [[papers/linformer|Linformer: Self-Attention with Linear Complexity]] | 2020 | `abandoned` |
| [[papers/performer|Rethinking Attention with Performers]] | 2021 | `abandoned` |
| [[papers/transnormerllm|TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer]] | 2023 | `abandoned` |
| [[papers/lightning-attention-2|Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models]] | 2024 | `abandoned` |

### kv-cache (19)

5 in-upstream-engine, 14 research-only. Synthesis: [09-model-memory-envelope.md](09-model-memory-envelope.md).

| paper | year | adoption |
|---|---|---|
| [[papers/multi-query-attention|Fast Transformer Decoding: One Write-Head is All You Need]] | 2019 | `in-upstream-engine` |
| [[papers/grouped-query-attention|GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints]] | 2023 | `in-upstream-engine` |
| [[papers/cacheblend-rag-kv-fusion|CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion]] | 2024 | `in-upstream-engine` |
| [[papers/deepseek-v2-mla|DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model]] | 2024 | `in-upstream-engine` |
| [[papers/paper-lmcache|LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference]] | 2025 | `in-upstream-engine` |
| [[papers/h2o|H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models]] | 2023 | `research-only` |
| [[papers/scissorhands|Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time]] | 2023 | `research-only` |
| [[papers/kivi|KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache]] | 2024 | `research-only` |
| [[papers/paper-cachegen|CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving]] | 2024 | `research-only` |
| [[papers/paper-kvquant|KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization]] | 2024 | `research-only` |
| [[papers/paper-pensieve|Stateful Large Language Model Serving with Pensieve]] | 2024 | `research-only` |
| [[papers/paper-zipcache|ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification]] | 2024 | `research-only` |
| [[papers/quest|Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference]] | 2024 | `research-only` |
| [[papers/snapkv-paper|SnapKV: LLM Knows What You are Looking for Before Generation]] | 2024 | `research-only` |
| [[papers/w4p-memserve-elastic-memory-pool|MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool]] | 2024 | `research-only` |
| [[papers/w4p-minicache-depth-kv-compression|MiniCache: KV Cache Compression in Depth Dimension for Large Language Models]] | 2024 | `research-only` |
| [[papers/paper-shadowkv|ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference]] | 2025 | `research-only` |
| [[papers/pyramidkv-paper|PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling]] | 2025 | `research-only` |
| [[papers/w4p-magicpig-lsh-attention-sampling|MagicPIG: LSH Sampling for Efficient LLM Generation]] | 2025 | `research-only` |

### speech (18)

8 in-production, 6 in-upstream-engine, 4 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/conformer|Conformer: Convolution-augmented Transformer for Speech Recognition]] | 2020 | `in-production` |
| [[papers/wav2vec2|wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]] | 2020 | `in-production` |
| [[papers/soundstream|SoundStream: An End-to-End Neural Audio Codec]] | 2021 | `in-production` |
| [[papers/whisper|Whisper: Robust Speech Recognition via Large-Scale Weak Supervision]] | 2023 | `in-production` |
| [[papers/whisper-streaming|Turning Whisper into a Real-Time Transcription System (Whisper-Streaming)]] | 2023 | `in-production` |
| [[papers/careless-whisper|Careless Whisper: Speech-to-Text Hallucination Harms]] | 2024 | `in-production` |
| [[papers/encodec|High Fidelity Neural Audio Compression (EnCodec)]] | 2024 | `in-production` |
| [[papers/qwen2-audio|Qwen2-Audio Technical Report]] | 2024 | `in-production` |
| [[papers/wav2vec|wav2vec: Unsupervised Pre-training for Speech Recognition]] | 2019 | `in-upstream-engine` |
| [[papers/emformer|Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition]] | 2021 | `in-upstream-engine` |
| [[papers/fastemit|FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization]] | 2021 | `in-upstream-engine` |
| [[papers/hubert|HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units]] | 2021 | `in-upstream-engine` |
| [[papers/data2vec|data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language]] | 2022 | `in-upstream-engine` |
| [[papers/zipformer|Zipformer: A faster and better encoder for automatic speech recognition]] | 2024 | `in-upstream-engine` |
| [[papers/speechgpt|SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities]] | 2023 | `research-only` |
| [[papers/audio-flamingo|Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities]] | 2024 | `research-only` |
| [[papers/hubert-ee|HuBERT-EE: Early Exiting HuBERT for Efficient Speech Recognition]] | 2024 | `research-only` |
| [[papers/salmonn|SALMONN: Towards Generic Hearing Abilities for Large Language Models]] | 2024 | `research-only` |

### multimodal (18)

13 in-production, 3 in-upstream-engine, 2 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/clip|Learning Transferable Visual Models From Natural Language Supervision (CLIP)]] | 2021 | `in-production` |
| [[papers/blip2|BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models]] | 2023 | `in-production` |
| [[papers/llava|Visual Instruction Tuning (LLaVA)]] | 2023 | `in-production` |
| [[papers/qwen-vl|Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, and Text Recognition]] | 2023 | `in-production` |
| [[papers/siglip|Sigmoid Loss for Language Image Pre-Training (SigLIP)]] | 2023 | `in-production` |
| [[papers/internvl|InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks]] | 2024 | `in-production` |
| [[papers/llava-next|LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models]] | 2024 | `in-production` |
| [[papers/llava-onevision|LLaVA-OneVision: Easy Visual Task Transfer]] | 2024 | `in-production` |
| [[papers/paligemma|PaliGemma: A versatile 3B VLM for transfer]] | 2024 | `in-production` |
| [[papers/pixtral|Pixtral 12B]] | 2024 | `in-production` |
| [[papers/dinov2|DINOv2: Learning Robust Visual Features without Supervision]] | 2025 | `in-production` |
| [[papers/internvl-x|InternVL-X: Advancing and Accelerating the InternVL Series with Efficient Visual Token Compression]] | 2025 | `in-production` |
| [[papers/molmo|Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models]] | 2025 | `in-production` |
| [[papers/dynamicvit|DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification]] | 2021 | `in-upstream-engine` |
| [[papers/mae|Masked Autoencoders Are Scalable Vision Learners]] | 2022 | `in-upstream-engine` |
| [[papers/visionzip|VisionZip: Longer is Better but Not Necessary in Vision Language Models]] | 2025 | `in-upstream-engine` |
| [[papers/flamingo|Flamingo: a Visual Language Model for Few-Shot Learning]] | 2022 | `research-only` |
| [[papers/sparsevlm|SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference]] | 2025 | `research-only` |

### speculative-decoding (18)

3 in-production, 7 in-upstream-engine, 8 research-only. Synthesis: [12-serving-workflows.md](12-serving-workflows.md).

| paper | year | adoption |
|---|---|---|
| [[papers/chen-speculative-sampling-deepmind|Accelerating Large Language Model Decoding with Speculative Sampling]] | 2023 | `in-production` |
| [[papers/leviathan-speculative-decoding|Fast Inference from Transformers via Speculative Decoding]] | 2023 | `in-production` |
| [[papers/production-token-embedding-speculators|Accelerating Production LLMs with Combined Token/Embedding Speculators]] | 2024 | `in-production` |
| [[papers/eagle-2-dynamic-draft-trees|EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees]] | 2024 | `in-upstream-engine` |
| [[papers/eagle-feature-level-speculative|EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty]] | 2024 | `in-upstream-engine` |
| [[papers/lookahead-parallel-decoding|Break the Sequential Dependency of LLM Inference Using Lookahead Decoding]] | 2024 | `in-upstream-engine` |
| [[papers/medusa-multiple-decoding-heads|Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads]] | 2024 | `in-upstream-engine` |
| [[papers/specinfer-tree-speculative-serving|SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification]] | 2024 | `in-upstream-engine` |
| [[papers/eagle-3-training-time-test|EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-sam-decoding-suffix-automaton|SAM Decoding: Speculative Decoding via Suffix Automaton]] | 2025 | `in-upstream-engine` |
| [[papers/layerskip-self-speculative|LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding]] | 2024 | `research-only` |
| [[papers/ouroboros-phrase-by-phrase-drafting|Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative Decoding]] | 2024 | `research-only` |
| [[papers/recursive-speculative-decoding-rsd|Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement]] | 2024 | `research-only` |
| [[papers/rest-retrieval-speculative-decoding|REST: Retrieval-Based Speculative Decoding]] | 2024 | `research-only` |
| [[papers/sequoia-hardware-aware-specdec|Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding]] | 2024 | `research-only` |
| [[papers/w4p-cascade-speculative-drafting|Cascade Speculative Drafting for Even Faster LLM Inference (CS Drafting)]] | 2024 | `research-only` |
| [[papers/w4p-specexec-consumer-speculative|SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices]] | 2024 | `research-only` |
| [[papers/magicdec-long-context-specdec|MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding]] | 2025 | `research-only` |

### moe (17)

4 in-production, 5 in-upstream-engine, 7 research-only. Synthesis: [13-model-compatibility.md](13-model-compatibility.md), [17-deployment-shape.md](17-deployment-shape.md).

| paper | year | adoption |
|---|---|---|
| [[papers/deepseek-v3-report|DeepSeek-V3 Technical Report]] | 2024 | `in-production` |
| [[papers/deepseekmoe|DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models]] | 2024 | `in-production` |
| [[papers/mixtral|Mixtral of Experts]] | 2024 | `in-production` |
| [[papers/w4p-apple-ptmoe-server-model|Apple Intelligence Foundation Language Models: Tech Report 2025 (PT-MoE server model)]] | 2025 | `in-production` |
| [[papers/gshard|GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding]] | 2021 | `in-upstream-engine` |
| [[papers/deepspeed-moe|DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale]] | 2022 | `in-upstream-engine` |
| [[papers/switch-transformers|Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity]] | 2022 | `in-upstream-engine` |
| [[papers/olmoe|OLMoE: Open Mixture-of-Experts Language Models]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-kimi-k2-moe-agentic|Kimi K2: Open Agentic Intelligence]] | 2025 | `in-upstream-engine` |
| [[papers/megablocks|MegaBlocks: Efficient Sparse Training with Mixture-of-Experts]] | 2023 | `research-only` |
| [[papers/tutel-flex|Tutel / Flex: Adaptive Mixture-of-Experts at Scale]] | 2023 | `research-only` |
| [[papers/pre-gated-moe|Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert LLM Inference]] | 2024 | `research-only` |
| [[papers/w4p-fiddler-cpu-gpu-moe-orchestration|Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models]] | 2025 | `research-only` |
| [[papers/w4p-megascale-infer-disaggregated-expert-parallelism|MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism]] | 2025 | `research-only` |
| [[papers/w4p-moe-lightning-hierarchical-roofline|MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs]] | 2025 | `research-only` |
| [[papers/specprefetch-window-aware-expert-prefetch|SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models]] | 2026 | `research-only` |
| [[papers/seqmoe-graph-compatible-moe-offload|SeqMoE: Toward Full-Load Performance via Predictive and Graph-Compatible MoE Offloading]] | 2026 | `research-only` |

### other (12)

6 in-production, 3 in-upstream-engine, 3 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/carbon-emissions-large-neural-training|Carbon Emissions and Large Neural Network Training]] | 2021 | `in-production` |
| [[papers/carbon-intensity-ai-cloud-instances|Measuring the Carbon Intensity of AI in Cloud Instances]] | 2022 | `in-production` |
| [[papers/meth-llm-as-judge-mt-bench|Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena]] | 2023 | `in-production` |
| [[papers/mobilellm-on-device|MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases]] | 2024 | `in-production` |
| [[papers/phi-3-small-capable|Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone]] | 2024 | `in-production` |
| [[papers/meth-arena-hard-benchbuilder|From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline]] | 2025 | `in-production` |
| [[papers/sedd-score-entropy-discrete-diffusion|Structured Denoising Diffusion Models in Discrete State-Spaces (SEDD)]] | 2021 | `in-upstream-engine` |
| [[papers/dream-7b|Dream 7B: Diffusion Large Language Models]] | 2025 | `in-upstream-engine` |
| [[papers/llada-large-language-diffusion-models|Large Language Diffusion Models (LLaDA)]] | 2025 | `in-upstream-engine` |
| [[papers/diffusion-lm|Diffusion-LM Improves Controllable Text Generation]] | 2022 | `research-only` |
| [[papers/meth-detecting-pretraining-data|Detecting Pretraining Data from Large Language Models]] | 2024 | `research-only` |
| [[papers/block-diffusion-bd3-lm|Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)]] | 2025 | `research-only` |

### embedding (11)

10 in-production, 1 in-upstream-engine.

| paper | year | adoption |
|---|---|---|
| [[papers/sbert|Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks]] | 2019 | `in-production` |
| [[papers/simcse|SimCSE: Simple Contrastive Learning of Sentence Embeddings]] | 2021 | `in-production` |
| [[papers/e5|Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)]] | 2022 | `in-production` |
| [[papers/matryoshka|Matryoshka Representation Learning]] | 2022 | `in-production` |
| [[papers/gte|Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE)]] | 2023 | `in-production` |
| [[papers/mteb|MTEB: Massive Text Embedding Benchmark]] | 2023 | `in-production` |
| [[papers/nomic-embed|Nomic Embed: Training a Reproducible Long Context Text Embedder]] | 2024 | `in-production` |
| [[papers/paper-bge-m3|M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings]] | 2024 | `in-production` |
| [[papers/embedder-dilemma|The Embedder's Dilemma: LLMs Are Better, but at What Cost?]] | 2026 | `in-production` |
| [[papers/giga-embeddings|Giga-Embeddings: Mixture-of-Experts Encoders for High-Throughput Text Embeddings]] | 2026 | `in-production` |
| [[papers/llm2vec|LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders]] | 2024 | `in-upstream-engine` |

### ssm (11)

7 in-upstream-engine, 4 research-only. Synthesis: [13-model-compatibility.md](13-model-compatibility.md).

| paper | year | adoption |
|---|---|---|
| [[papers/rwkv|RWKV: Reinventing RNNs for the Transformer Era]] | 2023 | `in-upstream-engine` |
| [[papers/jamba-1-5|Jamba-1.5: Hybrid Transformer-Mamba Models at Scale]] | 2024 | `in-upstream-engine` |
| [[papers/jamba-hybrid|Jamba: A Hybrid Transformer-Mamba Language Model]] | 2024 | `in-upstream-engine` |
| [[papers/mamba|Mamba: Linear-Time Sequence Modeling with Selective State Spaces]] | 2024 | `in-upstream-engine` |
| [[papers/mamba-2|Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality]] | 2024 | `in-upstream-engine` |
| [[papers/mambavision|MambaVision: A Hybrid Mamba-Transformer Vision Backbone]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-nemotron-h-hybrid-mamba-transformer|Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models]] | 2025 | `in-upstream-engine` |
| [[papers/s4|Efficiently Modeling Long Sequences with Structured State Spaces]] | 2022 | `research-only` |
| [[papers/retnet|Retentive Network: A Successor to Transformer for Large Language Models]] | 2023 | `research-only` |
| [[papers/recurrentgemma-griffin|RecurrentGemma: Moving Past Transformers for Efficient Open Language Models]] | 2024 | `research-only` |
| [[papers/zamba|Zamba: A Compact 7B SSM Hybrid Model]] | 2024 | `research-only` |

### retrieval (8)

4 in-production, 2 in-upstream-engine, 2 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/colbert|ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT]] | 2020 | `in-production` |
| [[papers/dpr|Dense Passage Retrieval for Open-Domain Question Answering]] | 2020 | `in-production` |
| [[papers/splade|SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking]] | 2021 | `in-production` |
| [[papers/colbertv2|ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction]] | 2022 | `in-production` |
| [[papers/contriever|Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever)]] | 2021 | `in-upstream-engine` |
| [[papers/plaid|PLAID: An Efficient Engine for Late Interaction Retrieval]] | 2022 | `in-upstream-engine` |
| [[papers/w4p-cachecraft-rag-chunk-cache|Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation]] | 2025 | `research-only` |
| [[papers/w4p-turborag-precomputed-chunk-kv|TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text]] | 2025 | `research-only` |

### distillation (8)

4 in-production, 4 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/orca|Orca: Progressive Learning from Complex Explanation Traces of GPT-4]] | 2023 | `in-production` |
| [[papers/llama-3-herd|The Llama 3 Herd of Models]] | 2024 | `in-production` |
| [[papers/phi-4|Phi-4 Technical Report]] | 2024 | `in-production` |
| [[papers/paper-deepseek-r1|DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning]] | 2025 | `in-production` |
| [[papers/distilling-step-by-step|Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes]] | 2023 | `research-only` |
| [[papers/gkd|On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]] | 2024 | `research-only` |
| [[papers/minillm|MiniLLM: On-Policy Distillation of Large Language Models]] | 2024 | `research-only` |
| [[papers/wizardlm|WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions]] | 2024 | `research-only` |

### long-context (7)

7 research-only. Synthesis: [09-model-memory-envelope.md](09-model-memory-envelope.md).

| paper | year | adoption |
|---|---|---|
| [[papers/gist-tokens|Learning to Compress Prompts with Gist Tokens]] | 2023 | `research-only` |
| [[papers/infini-attention|Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention]] | 2024 | `research-only` |
| [[papers/minference|MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention]] | 2024 | `research-only` |
| [[papers/selfextend|LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning]] | 2024 | `research-only` |
| [[papers/streamingllm|Efficient Streaming Language Models with Attention Sinks]] | 2024 | `research-only` |
| [[papers/duoattention|DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads]] | 2025 | `research-only` |
| [[papers/titans|Titans: Learning to Memorize at Test Time]] | 2025 | `research-only` |

### scheduling (7)

1 in-upstream-engine, 5 research-only, 1 unknown. Synthesis: [12-serving-workflows.md](12-serving-workflows.md), [19-production-operations.md](19-production-operations.md).

| paper | year | adoption |
|---|---|---|
| [[papers/sarathi-serve|Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve]] | 2024 | `in-upstream-engine` |
| [[papers/fastserve-llm|Fast Distributed Inference Serving for Large Language Models]] | 2023 | `research-only` |
| [[papers/llumnix-dynamic-scheduling|Llumnix: Dynamic Scheduling for Large Language Model Serving]] | 2024 | `research-only` |
| [[papers/tetriinfer|Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads (TetriInfer)]] | 2024 | `research-only` |
| [[papers/w4p-preble-distributed-prompt-scheduling|Preble: Efficient Distributed Prompt Scheduling for LLM Serving]] | 2025 | `research-only` |
| [[papers/rapid-power-aware-disaggregated-inference|Power Aware Dynamic Reallocation For Inference (RAPID)]] | 2026 | `research-only` |
| [[papers/packserve-agentic-scheduling|PackServe: SLO-Aware Request Scheduling for Agentic LLM Serving at Scale]] | 2026 | `unknown` |

### pruning (7)

7 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/llm-pruner|LLM-Pruner: On the Structural Pruning of Large Language Models]] | 2023 | `research-only` |
| [[papers/sparsegpt|SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot]] | 2023 | `research-only` |
| [[papers/llm-surgeon|The LLM Surgeon]] | 2024 | `research-only` |
| [[papers/sheared-llama|Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning]] | 2024 | `research-only` |
| [[papers/shortgpt-layer-pruning|ShortGPT: Layers in Large Language Models are More Redundant Than You Expect]] | 2024 | `research-only` |
| [[papers/slicegpt|SliceGPT: Compress Large Language Models by Deleting Rows and Columns]] | 2024 | `research-only` |
| [[papers/wanda|A Simple and Effective Pruning Approach for Large Language Models]] | 2024 | `research-only` |

### inference-time-compute (6)

4 in-production, 2 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/chain-of-thought-prompting|Chain-of-Thought Prompting Elicits Reasoning in Large Language Models]] | 2022 | `in-production` |
| [[papers/self-consistency-marginalise-paths|Self-Consistency Improves Chain of Thought Reasoning in Language Models]] | 2023 | `in-production` |
| [[papers/llms-cannot-self-correct-reasoning|Large Language Models Cannot Self-Correct Reasoning Yet]] | 2024 | `in-production` |
| [[papers/process-supervision-prm800k|Let's Verify Step by Step]] | 2024 | `in-production` |
| [[papers/mixture-of-depths|Mixture-of-Depths: Dynamically allocating compute in transformer-based language models]] | 2024 | `research-only` |
| [[papers/s1-budget-forcing|s1: Simple test-time scaling]] | 2025 | `research-only` |

### position-encoding (6)

6 in-upstream-engine. Synthesis: [13-model-compatibility.md](13-model-compatibility.md).

| paper | year | adoption |
|---|---|---|
| [[papers/ntk-aware-scaled-rope|A Length-Extrapolatable Transformer]] | 2022 | `in-upstream-engine` |
| [[papers/position-interpolation|Extending Context Window of Large Language Models via Positional Interpolation]] | 2023 | `in-upstream-engine` |
| [[papers/longrope|LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens]] | 2024 | `in-upstream-engine` |
| [[papers/qwen2-vl-mrope|Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution]] | 2024 | `in-upstream-engine` |
| [[papers/yarn|YaRN: Efficient Context Window Extension of Large Language Models]] | 2024 | `in-upstream-engine` |
| [[papers/longrope2|LongRoPE2: Near-Lossless LLM Context Window Scaling]] | 2025 | `in-upstream-engine` |

### routing (4)

1 in-production, 3 research-only.

| paper | year | adoption |
|---|---|---|
| [[papers/routellm|RouteLLM: Learning to Route LLMs with Preference Data]] | 2025 | `in-production` |
| [[papers/expert-choice-routing|Mixture-of-Experts with Expert Choice Routing]] | 2022 | `research-only` |
| [[papers/frugalgpt|FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance]] | 2023 | `research-only` |
| [[papers/energy-aware-routing|Measured Joules, Learned Routes: Learning to Route for Energy-Efficient LLM Serving]] | 2026 | `research-only` |

### structured-output (3)

3 in-upstream-engine. Synthesis: [03-engine-selection.md](03-engine-selection.md).

| paper | year | adoption |
|---|---|---|
| [[papers/jsonschemabench-structured-output-eval|JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-xgrammar-structured-generation-engine|XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models]] | 2025 | `in-upstream-engine` |
| [[papers/w4p-xgrammar2-dynamic-agentic-structured-generation|XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs]] | 2026 | `in-upstream-engine` |

### security (3)

2 in-production, 1 research-only. Synthesis: [22-safety-and-security.md](22-safety-and-security.md).

| paper | year | adoption |
|---|---|---|
| [[papers/llama-guard-safeguard-model|Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations]] | 2023 | `in-production` |
| [[papers/llama-guard-3-1b-int4|Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations]] | 2024 | `in-production` |
| [[papers/meth-ai-safety-benchmark-v05|Introducing v0.5 of the AI Safety Benchmark from MLCommons]] | 2024 | `research-only` |

### cluster (2)

1 in-upstream-engine, 1 abandoned. Synthesis: [21-deployment-topology.md](21-deployment-topology.md).

| paper | year | adoption |
|---|---|---|
| [[papers/w4p-aibrix-multitenant-serving-control-plane|AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure]] | 2025 | `in-upstream-engine` |
| [[papers/salus|Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications]] | 2019 | `abandoned` |

### agentic (1)

1 in-production.

| paper | year | adoption |
|---|---|---|
| [[papers/react-reasoning-and-acting|ReAct: Synergizing Reasoning and Acting in Language Models]] | 2023 | `in-production` |

### sparsity (1)

1 in-production.

| paper | year | adoption |
|---|---|---|
| [[papers/sparse-tensor-cores|Accelerating Sparse Deep Neural Networks]] | 2021 | `in-production` |

## Related documents

- [README.md](README.md) — the reading guide for all docs.
- [25-citation-integrity.md](25-citation-integrity.md) — why a paper record's number is
  checked against the paper before it is cited.
