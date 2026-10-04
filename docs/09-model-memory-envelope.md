# 09 — Model memory envelope: weights, KV, and the weight-vs-KV streaming crossover

Every figure below is derived from a record in `data/models/` and an accelerator record in
`data/accelerators/`. Model dimensions were read from the records; where a record was silent or
internally inconsistent, the derivation is shown from the authoritative `config.json` instead and
marked. Nothing here is a measurement. Section 7 says what is still missing.

**The central result.** `docs/02-flop-map.md` §8 item 2 records that *"No measured crossover batch
for any real model on any real part."* That remains true — this document does not close it. What
this document closes is the *analytic* side, and the analytic side turns out to say something the
roofline framing hides:

> On H100 SXM at bf16, **61 of the 62 models here with a recorded active-parameter figure have a
> maximum achievable decode arithmetic intensity below the 295.4 flop/byte ridge at every context
> length the repo records, and are therefore memory-bound at every batch size.** Six of those put the
> ridge out of reach inside **100 tokens** of context and 37 more inside **1,000**. The batch-295
> crossover that `docs/02-flop-map.md` reports is real but it is a *ceiling on a best case*, not the
> operating point for almost any model in the repo.

---

## 1. What is computed, and from what

### 1.1 The ridge, per part

`ridge = 1000 · dense_bf16_TFLOPs / memory_bandwidth_GB_per_s`, computed from each accelerator
record's own `flops` and `memory_bandwidth_gbps` fields. This is the convention already stated in
`docs/02-flop-map.md` §3.1, recomputed here and consistent with it for H100.

| part | bf16 dense TFLOPS | bandwidth GB/s | **ridge flop/byte** | vram_gb |
|---|---|---|---|---|
| [[accelerators/nvidia-h100-sxm]] | 989.5 | 3350 | **295.4** | 80 |
| [[accelerators/nvidia-h200-sxm]] | 989.5 | 4800 | **206.1** | 141 |
| [[accelerators/nvidia-b200]] | 2500 | 8000 | **312.5** | 186 |
| [[accelerators/amd-instinct-mi300x]] | 1307.4 | 5300 | **246.7** | 192 |
| [[accelerators/amd-instinct-mi325x]] | 1307.4 | 6000 | **217.9** | 256 |
| [[accelerators/amd-instinct-mi355x]] | 2500 | 8000 | **312.5** | 288 |
| [[accelerators/nvidia-a100-80gb-sxm4]] | 312 | 2039 | **153.0** | 80 |
| [[accelerators/nvidia-l40s]] | 362 | 864 | **419.0** | 48 |
| [[accelerators/nvidia-h20]] | 148 | 4000 | **37.0** | 96 |

`docs/02-flop-map.md` §8 item 1 asks for ridge points on parts other than H100 and does not have
them. **The accelerator records already contain every input; §3.1's table above is the answer to
that open question and needs no new data.** The H100 295.4 and the A100 153.0 and B200 312.5
figures match `docs/02-flop-map.md` §3.1 exactly, which is the cross-check.

Note the two rows that are decision-relevant and counterintuitive: **[[accelerators/nvidia-l40s]] has
the second-highest ridge in the repo (419) and 864 GB/s of bandwidth** — a high ridge is not the
same as good decode, per the wall-width warning quoted in `docs/02-flop-map.md` §3.1. And
**[[accelerators/nvidia-h20]] has a ridge of 37**, i.e. 148 dense bf16 TFLOPS against 4000 GB/s: it
is memory-rich and compute-starved by design, so on this document's arithmetic it is *the* decode
part and is a bad prefill part.

Unless a column says otherwise, every crossover in this document is **H100 SXM, dense bf16, R = 295.37**.

### 1.2 The three quantities

For each model, from its record:

| symbol | meaning | source field |
|---|---|---|
| `P` | active parameters per token, in units of 1e9 | `active_params_b` |
| `W_act` | resident weight bytes per token of the *active* set, bf16 | `2 · P · 1e9` |
| `k` | KV cache bytes per token | `kv_cache_bytes_per_token` |
| `g` | GQA group ratio = query heads / KV heads | `gqa_ratio` |
| `L` | current sequence length in tokens | — |

**(a) Weight-vs-KV streaming crossover, `B_eq`.** Weight streaming dominates while the weights are
larger than the KV read. Total bytes per decode step = `W + k·L·B`; total bytes attributable to
weights alone = `W`. Weight streaming stops dominating when `k·L·B > W`, i.e.

> **`B_eq = W / (k · L)`**

and the batch where KV streaming *equals* weight streaming, on an active-weight basis, is
**`B_eq = W_act / k = 2·P·1e9 / k`** — a context length of one token. This is the ratio the brief
asks for, and it is enormous for every model here (smallest value in the set: 17,405 for
[[models/smollm2-1-7b-instruct]]). At `L = 8k` divide it by 8,192. `docs/02-flop-map.md` §3.3's
"batch 295" figure is *not* this number and does not conflict with it: 295 is the ridge crossing of
the weight GEMM alone, while `B_eq` is where attention traffic overtakes weight traffic.

**(b) The ridge crossing, `B*`.** From `docs/02-flop-map.md` §3.3 and §3.5: the weight GEMM has
AI = `B`, decode attention has AI = `g` (the GQA ratio, flat in both `B` and `L`). A decode step's
combined arithmetic intensity at batch `B` and context `L` is

> **AI(B, L) = (2·P·B + g·k·L·B) / (W + k·L·B)**

Setting `AI = R` and solving for `B`:

> **`B* = R·W / (2P − (R − g)·k·L)`**

with `P` in units of 1e9 and `k·L` in bytes. Three readings follow directly:

1. **`g` appears in the denominator, not outside it.** With `g` omitted the formula reduces to the
   `B* = R·W/(2P − R·k·L)` that treats attention as AI = 1, which is exactly the error
   `docs/02-flop-map.md` §3.5 documents as CORRECTED. Any document computing `B*` for a GQA model
   without `g` is repeating a correction the repo already made.
2. **`B*` is unbounded only while `2P > (R−g)·k·L`.** When that fails the denominator goes negative
   and there is **no batch size at which the step reaches the ridge** — attention traffic is already
   dominant and streaming it alone keeps AI below `R`.
3. **The condition for a crossover to exist at all is a statement about context length, not batch.**
   Define

   > **`L_wall = 2·P·1e9 / ((R − g)·k)`**

   `L_wall` is the context length above which no batch size can reach the ridge. It is **59 tokens for
   [[models/smollm2-1-7b-instruct]]** and **3,577 for [[models/deepseek-v3]]** — and `docs/02-flop-map.md`
   §3.5 already states the conclusion for the general case: "**NO standard attention configuration
   reaches the compute roof, at any context length, at any batch size, at any dtype.**" `L_wall` is
   that statement, per model, with a number attached.

**(c) KV cache size at a context length** is `k · L`, in GB at `L` = 8k / 32k / 128k / 1M, and the
context at which KV exceeds 1 / 8 / 24 / 80 GB is `threshold_bytes / k`. All four thresholds are in
the per-family tables in §3.

### 1.3 Field I would add to `schemas/model.schema.json`

I could not edit `schemas/` per the brief, so recording the request here. Three additions, all
backward compatible:

1. **`kv_cache_bytes_per_token_basis`** (enum: `gqa` | `mha` | `mla-latent` | `hybrid-subset` |
   `decoder-self-attention` | `none-fixed-state` | `not-applicable`). Fourteen records in this table
   deviate from the `num_layers × num_kv_heads × head_dim × 2 × 2` formula the `model` schema
   currently documents as *the* derivation, and nothing in the schema marks which formula a given
   record used. `kv_cache_bytes_per_token` alone is not self-describing.
2. **`kv_caching_layer_count`** (int|null) and **`state_bytes_per_sequence`** (number|null) — for
   hybrid models. [[models/nemotron-h-8b-base-8k]] holds 16 KiB/token of KV **plus a fixed 96 MiB per
   sequence** of Mamba-2 state, and [[models/falcon-h1-7b-base]] holds 44 KiB/token **plus 66 MiB**.
   The fixed part is invisible to any consumer sizing memory from `kv_cache_bytes_per_token` alone,
   and it is the part that scales with concurrency.
3. **`attention_head_count`** (int|null), needed to make `gqa_ratio` self-checking. `gqa_ratio` is
   recorded on 62 of 64 rows here, but it is a derived quantity and two records carry a value that
   their own head counts do not reproduce ([[models/deepseek-v3]] records `gqa_ratio: 1.0` while
   `num_kv_heads: 128` — correct for MLA, where the latent is expanded, but a naive
   `num_kv_heads` reader will compute `g = 1` and get the wrong answer for the right reason).

---

## 2. Corrections found in existing records

Two records carry an arithmetic error in the derivation printed in their own `notes`, and both are
in the same failure mode: **the printed product does not equal the printed result**. Verified against
`config.json` on Hugging Face for each.

### 2.1 ‡ [[models/phi-3-mini-4k-instruct]] and [[models/phi-3-5-mini-instruct]] — off by 16x

Both records print:

> `kv_cache_bytes_per_token = num_layers * num_kv_heads * head_dim * 2 * bytes = 32 * 32 * 96 * 2 * 2 = 24576 bytes/token at bf16`

`32 · 32 · 96 · 2 · 2` = **393,216**, not 24,576. The recorded field value (24,576) is wrong too, so
this is not a typo in the prose — the number is wrong.

*Correct derivation*, from `microsoft/Phi-3-mini-4k-instruct/config.json` (fetched; `num_hidden_layers`
32, `num_key_value_heads` 32, `hidden_size` 3072, `num_attention_heads` 32, no `head_dim` field, so
`head_dim = 3072/32 = 96`):

> **kv_cache_bytes_per_token = 32 × 32 × 96 × 2 × 2 = 393,216 B/token = 384 KiB/token**

Confirmed identical for `microsoft/Phi-3.5-mini-instruct/config.json` — every geometry field matches,
so both records take the same corrected value.

The likely origin of the error is visible in the record's own next sentence: "This is MHA, not GQA:
32 KV heads means the KV cache is 4x what a GQA-8 variant of the same width would need." 384 KiB
*is* 4x the 96 KiB a GQA-8 model of the same width would need (32 × 8 × 96 × 2 × 2 = 98,304 B = 96
KiB). The reasoning was right; the multiplication was not carried through.

**Serving impact.** [[models/phi-3-5-mini-instruct]] advertises 131,072 tokens of context. At the
recorded 24,576 B/token that is 3.2 GB of KV per sequence; at the corrected 393,216 B/token it is
**51.5 GB**. The correction moves this model from "the KV is comparable to the weights" (7.64 GB
weights) to "**the KV is 6.7x the weights at full advertised context**", and it moves the context at
which KV passes 24 GB from 976,562 tokens (well past anything servable) to **61,035 tokens — inside
the model's own advertised window.** The 8k/32k/128k columns in §3 use the corrected figure.

**This makes Phi-3 the worst KV-per-parameter dense model in the repo.** [[models/phi-3-5-mini-instruct]]
is 3.8B parameters and carries 384 KiB/token — the *same* per-token KV as [[models/olmo-3-7b]], a
7.3B model, and 4x that of [[models/llama-3-1-8b]] at nearly half the size. Its `L_wall` is **66
tokens**; at any context beyond a paragraph, no batch size reaches the ridge.

### 2.2 [[models/gte-qwen2-1-5b-instruct]] — off by 2x

The record prints:

> `kv_cache_bytes_per_token = 28 * 2 * 128 * 2 * 2 = 14336 bytes/token at bf16`

`28 · 2 · 128 · 2 · 2` = **28,672**. Verified against
`Alibaba-NLP/gte-Qwen2-1.5B-instruct/config.json` (`num_hidden_layers` 28, `num_key_value_heads` 2,
`hidden_size` 1536, `num_attention_heads` 12, so `head_dim = 1536/12 = 128`):

> **kv_cache_bytes_per_token = 28 × 2 × 128 × 2 × 2 = 28,672 B/token = 28 KiB/token**

Low stakes — the record's own text correctly says the number is irrelevant to embedding use, since
embedding never grows the cache — but the field value is wrong by 2x and a serving stack that sizes
memory by the KV formula would be wrong.

### 2.3 Two derivations that are right but whose *field* is a worst case

Not errors, but they change what a reader should do with the number, and both records say so:

- **[[models/step-3-5-flash]]** records 196,608 B/token from "48 × 8 × 128 × 2 × 2 with all 48 layer
  slots counted." `config.json` gives `num_hidden_layers: 45`, `num_attention_groups: 8`,
  `sliding_window: 512`, and a `layer_types` array of length 48 that is 12 `full_attention` +
  36 `sliding_attention`. The true steady-state cache is
  `12·8·128·2·2·L + 36·8·128·2·2·512`, which **saturates at ~688 KB per sequence regardless of
  length** — the record states this in its own notes. The 196,608 field is a per-token figure that
  only applies below the 512-token window. Its `L_wall` of 389 tokens is therefore also wrong; the
  saturated model has no `L_wall`.
- **[[models/gemma-3-27b]]** records 507,904 B/token (62 × 16 × 128 × 2 × 2) with `sliding_window`
  1024 at a 5:1 local/global ratio. Same shape of caveat, and the record says the effective figure
  is roughly 10/62 of the naive one. For capacity planning the naive figure is the safe upper bound;
  for step-time bandwidth it is ~6x too high.

Both appear in §3 with the record value, marked here rather than in the table, because the record
value is the conservative one.

### 2.4 Where the naive formula is right to be wrong

Confirmed correct, and worth stating because the schema's documented formula does not apply:

- **MLA models.** [[models/deepseek-v3]] (70,272 = 61 × 576 × 2), [[models/deepseek-v2]] (69,120 =
  60 × 576 × 2), [[models/kimi-k2]] (70,272), [[models/glm-5-3]] (89,856 = 78 × 576 × 2),
  [[models/glm-4-7-flash]] (54,144 = 47 × 576 × 2). The naive GQA figure overstates by **57x**
  (DeepSeek-V3), and `docs/02-flop-map.md` §2 already carries MLA's "85.3x cache reduction."
- **Hybrid attention-SSM models**, where only a subset of layers cache:
  [[models/kimi-linear-48b-a3b]] (8,064 = 7 layers), [[models/granite-4-0-h-tiny]] (8,192 = 4 layers),
  [[models/minimax-m1]] (40,960 = 10 of 80 layers), [[models/qwen3-5-35b-a3b]] (20,480 = 10 of 40),
  [[models/qwen3-8-flash-next]] (24,576 = 12 of 48), [[models/minicpm-v-4-6]] (12,288 = 6 of 24),
  [[models/nemotron-h-8b-base-8k]] (16,384 = 4 of 52), [[models/kimi-k3]] (27,648 = 24 of 93).
- **Encoder-decoder.** [[models/whisper-large-v3]]'s 163,840 is decoder self-attention over
  `decoder_layers = 32`; `num_layers = 32` in that record is the **encoder** depth, which has no KV
  cache at all — a bidirectional single-pass forward, nothing to cache. [[models/whisper-large-v3-turbo]]
  is the one with the 4-layer decoder (20,480). Confirmed against the safetensors/config split; the
  turbo record's own "FIELD-PAIR WARNING" already states this.
- **Records where `kv_cache_bytes_per_token: 0` is a positive statement, not a missing value.**
  [[models/mamba-codestral-7b]] and [[models/rwkv5-eagle-7b]] (fixed recurrent state, flat in
  context), [[models/llada-8b-instruct]] (diffusion, no AR cache), [[models/parakeet-ctc-1-1b]] and
  [[models/canary-1b-flash]] (encoder-only, no decoder). For each of these **the crossover question
  does not arise**: there is no KV stream for weight streaming to be compared against.

### 2.5 Independently re-derived from `config.json` — all agree

Ten records were recomputed from first principles against their Hugging Face `config.json` and
every one matches the record's stored `kv_cache_bytes_per_token` to the byte:
[[models/llama-3-1-8b]] (131,072), [[models/llama-3-3-70b]] (327,680),
[[models/qwen3-235b-a22b]] (192,512), [[models/qwen3-30b-a3b]] (98,304), [[models/qwen2-5-72b]]
(327,680), [[models/mixtral-8x22b]] (229,376), [[models/glm-4-5]] (376,832), [[models/gpt-oss-120b]]
(73,728), [[models/qwen3-8b]] (147,456), [[models/olmo-3-7b]] (524,288),
[[models/smollm2-1-7b-instruct]] (196,608), [[models/granite-4-0-h-tiny]] (81,920 naive vs 8,192 hybrid —
the hybrid derivation is correct), [[models/kimi-linear-48b-a3b]] (248,832 naive vs 8,064 hybrid —
correct).

### 2.6 Three Gemma 4 records with `kv_cache_bytes_per_token: null` — now derivable

`docs/02-flop-map.md` and [[models/gemma-4-31b]] both record that Gemma 4's KV was left null
because "the geometry is not a single formula": sliding layers use `head_dim` 256 while global layers
use `global_head_dim` 512. **Two geometries in one stack is not a reason to record null — it is a
reason to record the sum.** `config.json` for each of the three exposes a `layer_types` array that
enumerates the pattern explicitly (5 `sliding_attention` per `full_attention`), plus
`num_key_value_heads` and `num_global_key_value_heads`. Fetched and summed:

> **kv = 2 · [ (n_layers − n_global) · num_key_value_heads · head_dim + n_global · num_global_key_value_heads · global_head_dim ]**

| record | layers | global layers | sliding KV elts/layer | global KV elts/layer | **kv B/token** |
|---|---|---|---|---|---|
| [[models/gemma-4-12b]] | 48 | 6 | 8 × 256 = 2048 | 1 × 512 = 512 | **178,176** (174.0 KiB) |
| [[models/gemma-4-26b-a4b]] | 30 | 5 | 8 × 256 = 2048 | 2 × 512 = 1024 | **112,640** (110.0 KiB) |
| [[models/gemma-4-31b]] | 60 | 10 | 16 × 256 = 4096 | 4 × 512 = 2048 | **450,560** (440.0 KiB) |

Marked `*` in §3. These are **derivable values, not measurements**, and they are marked as derived
rather than asserted into the records — per the brief I have not edited `data/models/`. Note the
result: [[models/gemma-4-31b]] at 440 KiB/token is the **6th-worst KV-per-token in the whole repo**,
and [[models/gemma-4-26b-a4b]] at 110 KiB/token is 3.2x cheaper than its 25.81B total suggests,
because a 4B-active MoE with a 2-head global geometry caches very little per token.

### 2.7 Records that cannot be computed, and why

Recorded as `no record` rather than estimated:

- [[models/claude-opus-5-5]], [[models/claude-sonnet-5-5]], [[models/claude-haiku-4-5]],
  [[models/gpt-4-1]], [[models/o3]] — architecture undisclosed. Every structural field is null by
  design and the records say so explicitly. Anthropic and OpenAI publish no layer count, head count,
  or attention type, so no KV-per-token figure is derivable and inventing one would be the exact
  failure mode this repo exists to prevent.
- [[models/deepseek-v4-pro]], [[models/deepseek-v4-flash]] — `kv_lora_rank` is absent from both
  configs and the compressed-cache geometry (`compress_ratios`, `sliding_window` 128, `hc_mult`)
  is not resolvable from the fields present. The records warn against carrying the V3 latent formula
  forward, which is correct.
- [[models/parakeet-tdt-0-6b-v2]], [[models/paligemma-3b-mix-224]] — no machine-readable per-layer
  dimensions available; both records already flag incomplete verification.
- [[models/grok-1]], [[models/glm-5-3]] — `kv_cache_bytes_per_token` and `gqa_ratio` are present, but
  **`active_params_b` is null**, so `W_act`, `B_eq` and `L_wall` cannot be computed. [[models/glm-5-3]]'s
  own notes decline to state an activated figure, and estimate ~40.9B from config as "indicative only."
  I have carried that discipline: `no record`, not 40.9.

---

## 3. Per-family tables

All KV figures bf16. `active weights GB` = `2 · active_params_b · 1e9 / 1e9`. `B_eq` and `L_wall` are
as defined in §1.2. `*` = KV derived in §2.6, not in the record. `‡` = corrected in §2.1.

A note on reading `active weights GB`: for a MoE this is **not** the memory the card must hold.
[[models/gpt-oss-120b]] shows 10.20 GB active but 116.8B total = **233.6 GB** of bf16 weights that all
have to be resident — the record's own note says "all 117B must be resident even though only 5.1B is
active," and explains that this is exactly why the released MXFP4 build fits on one 80GB device.
The `B_eq (total basis)` column in §4 carries the resident-weights version of the crossover.


#### Dense — 34 models

| model | total B | active B | active weights GB (bf16) | KV KiB/tok | KV GB @8k | @32k | @128k | KV>1GB @ tok | KV>24GB @ tok | KV>80GB @ tok | B_eq (active basis) | L_wall tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [[models/deepseek-r1-distill-qwen-7b]] | 7.615616512 | 7.615616512 | 15.23 | 56.0 | 0.47 | 1.88 | 7.5 | 17,438 | 418,526 | 1,395,089 | 265,611 | 921 |
| [[models/gemma-3n-e2b]] | 5.44 | 1.91 | 3.82 | 60.0 | 0.50 | 2.01 | 8.1 | 16,276 | 390,625 | 1,302,083 | 62,174 | 213 |
| [[models/gemma-2-2b]] | 2.614 | 2.614 | 5.23 | 104.0 | 0.87 | 3.49 | 14.0 | 9,390 | 225,360 | 751,201 | 49,091 | 167 |
| [[models/gemma-4-26b-a4b]]* | 25.81 | 4.0 | 8.00 | 110.0 | 0.92 | 3.69 | 14.8 | 8,877 | 213,068 | 710,227 | 71,022 | 242 |
| [[models/falcon3-7b-instruct]] | 7.455550464 | 7.455550464 | 14.91 | 112.0 | 0.94 | 3.76 | 15.0 | 8,719 | 209,263 | 697,544 | 130,014 | 444 |
| [[models/olmo-2-1b]] | 1.485 | 1.485 | 2.97 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 22,659 | 76 |
| [[models/phi-4-mini-instruct]] | 3.836 | 3.836 | 7.67 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 58,532 | 200 |
| [[models/llama-3-1-8b]] | 8.03 | 8.03 | 16.06 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 122,528 | 420 |
| [[models/deepseek-r1-distill-llama-8b]] | 8.030261248 | 8.030261248 | 16.06 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 122,532 | 420 |
| [[models/ministral-3-8b]] | 8.92 | 8.92 | 17.84 | 136.0 | 1.14 | 4.56 | 18.3 | 7,180 | 172,334 | 574,448 | 128,102 | 439 |
| [[models/ministral-8b]] | 8.02 | 8.02 | 16.04 | 144.0 | 1.21 | 4.83 | 19.3 | 6,781 | 162,760 | 542,534 | 108,778 | 373 |
| [[models/qwen3-8b]] | 8.19 | 8.19 | 16.38 | 144.0 | 1.21 | 4.83 | 19.3 | 6,781 | 162,760 | 542,534 | 111,083 | 381 |
| [[models/qwen3-vl-8b-instruct]] | 8.767 | 8.767 | 17.53 | 144.0 | 1.21 | 4.83 | 19.3 | 6,781 | 162,760 | 542,534 | 118,910 | 408 |
| [[models/whisper-large-v3]] | 1.543 | 1.543 | 3.09 | 160.0 | 1.34 | 5.37 | 21.5 | 6,103 | 146,484 | 488,281 | 18,835 | 63 |
| [[models/granite-3-3-8b-instruct]] | 8.17086464 | 8.17086464 | 16.34 | 160.0 | 1.34 | 5.37 | 21.5 | 6,103 | 146,484 | 488,281 | 99,742 | 342 |
| [[models/pixtral-12b-2409]] | 12.0 | 12.0 | 24.00 | 160.0 | 1.34 | 5.37 | 21.5 | 6,103 | 146,484 | 488,281 | 146,484 | 502 |
| [[models/magistral-small]] | 24.01 | 24.01 | 48.02 | 160.0 | 1.34 | 5.37 | 21.5 | 6,103 | 146,484 | 488,281 | 293,090 | 1,005 |
| [[models/gemma-4-12b]]* | 11.96 | 11.96 | 23.92 | 174.0 | 1.46 | 5.84 | 23.4 | 5,612 | 134,698 | 448,994 | 134,249 | 457 |
| [[models/smollm2-1-7b-instruct]] | 1.711 | 1.711 | 3.42 | 192.0 | 1.61 | 6.44 | 25.8 | 5,086 | 122,070 | 406,901 | 17,405 | 59 |
| [[models/deepseek-r1-distill-qwen-32b]] | 32.8 | 32.8 | 65.60 | 256.0 | 2.15 | 8.59 | 34.4 | 3,814 | 91,552 | 305,175 | 250,244 | 861 |
| [[models/seed-oss-36b]] | 36.6 | 36.6 | 73.20 | 256.0 | 2.15 | 8.59 | 34.4 | 3,814 | 91,552 | 305,175 | 279,235 | 978 |
| [[models/llama-3-3-70b]] | 70.55 | 70.55 | 141.10 | 320.0 | 2.68 | 10.74 | 43.0 | 3,051 | 73,242 | 244,140 | 430,603 | 1,498 |
| [[models/deepseek-r1-distill-llama-70b]] | 70.553706496 | 70.553706496 | 141.11 | 320.0 | 2.68 | 10.74 | 43.0 | 3,051 | 73,242 | 244,140 | 430,625 | 1,498 |
| [[models/qwen2-5-72b]] | 72.7 | 72.7 | 145.40 | 320.0 | 2.68 | 10.74 | 43.0 | 3,051 | 73,242 | 244,140 | 443,725 | 1,544 |
| [[models/gemma-2-9b]] | 9.24 | 9.24 | 18.48 | 336.0 | 2.82 | 11.27 | 45.1 | 2,906 | 69,754 | 232,514 | 53,710 | 183 |
| [[models/gemma-3-12b-it]] | 12.187 | 12.187 | 24.37 | 384.0 | 3.22 | 12.88 | 51.5 | 2,543 | 61,035 | 203,450 | 61,986 | 211 |
| [[models/phi-3-5-mini-instruct]] ‡ | 3.821 | 3.821 | 7.64 | 384.0 | 3.22 | 12.88 | 51.5 | 2,543 | 61,035 | 203,450 | 19,434 | 66 |
| [[models/phi-3-mini-4k-instruct]] ‡ | 3.821 | 3.821 | 7.64 | 384.0 | 3.22 | 12.88 | 51.5 | 2,543 | 61,035 | 203,450 | 19,434 | 66 |
| [[models/gemma-4-31b]]* | 31.27 | 31.27 | 62.54 | 440.0 | 3.69 | 14.76 | 59.1 | 2,219 | 53,267 | 177,556 | 138,805 | 473 |
| [[models/gemma-1-7b]] | 8.538 | 8.538 | 17.08 | 448.0 | 3.76 | 15.03 | 60.1 | 2,179 | 52,315 | 174,386 | 37,222 | 126 |
| [[models/gemma-3-27b]] | 27.43 | 27.43 | 54.86 | 496.0 | 4.16 | 16.64 | 66.6 | 1,968 | 47,253 | 157,510 | 108,012 | 368 |
| [[models/hermes-4-405b]] | 405.8533888 | 405.8533888 | 811.71 | 504.0 | 4.23 | 16.91 | 67.6 | 1,937 | 46,502 | 155,009 | 1,572,782 | 5,629 |
| [[models/olmo-3-7b]] | 7.3 | 7.3 | 14.60 | 512.0 | 4.29 | 17.18 | 68.7 | 1,907 | 45,776 | 152,587 | 27,847 | 94 |
| [[models/qwen2-audio-7b-instruct]] | 8.397 | 8.397 | 16.79 | 512.0 | 4.29 | 17.18 | 68.7 | 1,907 | 45,776 | 152,587 | 32,032 | 108 |

#### MoE — 20 models

| model | total B | active B | active weights GB (bf16) | KV KiB/tok | KV GB @8k | @32k | @128k | KV>1GB @ tok | KV>24GB @ tok | KV>80GB @ tok | B_eq (active basis) | L_wall tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [[models/gpt-oss-20b]] | 20.9 | 3.6 | 7.20 | 48.0 | 0.40 | 1.61 | 6.4 | 20,345 | 488,281 | 1,627,604 | 146,484 | 509 |
| [[models/glm-4-7-flash]] | 31.2 | 3 | 6.00 | 52.9 | 0.44 | 1.77 | 7.1 | 18,469 | 443,262 | 1,477,541 | 110,815 | 376 |
| [[models/deepseek-v2]] | 236 | 21 | 42.00 | 67.5 | 0.57 | 2.27 | 9.1 | 14,467 | 347,222 | 1,157,407 | 607,638 | 2,064 |
| [[models/kimi-k2]] | 1026 | 32 | 64.00 | 68.6 | 0.58 | 2.30 | 9.2 | 14,230 | 341,530 | 1,138,433 | 910,746 | 3,093 |
| [[models/deepseek-v3]] | 671 | 37 | 74.00 | 68.6 | 0.58 | 2.30 | 9.2 | 14,230 | 341,530 | 1,138,433 | 1,053,051 | 3,577 |
| [[models/deepseek-v3-2]] | 685.4 | 37 | 74.00 | 68.6 | 0.58 | 2.30 | 9.2 | 14,230 | 341,530 | 1,138,433 | 1,053,051 | 3,577 |
| [[models/deepseek-r1]] | 671 | 37 | 74.00 | 68.6 | 0.58 | 2.30 | 9.2 | 14,230 | 341,530 | 1,138,433 | 1,053,051 | 3,577 |
| [[models/gpt-oss-120b]] | 116.8 | 5.1 | 10.20 | 72.0 | 0.60 | 2.42 | 9.7 | 13,563 | 325,520 | 1,085,069 | 138,346 | 481 |
| [[models/glm-5-3]] | 753.3 | no record | — | 87.8 | 0.74 | 2.94 | 11.8 | 11,128 | 267,094 | 890,313 | no record | 56,957 |
| [[models/qwen3-30b-a3b]] | 30.5 | 3.3 | 6.60 | 96.0 | 0.81 | 3.22 | 12.9 | 10,172 | 244,140 | 813,802 | 67,138 | 233 |
| [[models/mixtral-8x7b]] | 46.7 | 13.0 | 26.00 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 198,364 | 680 |
| [[models/hunyuan-a13b]] | 80 | 13 | 26.00 | 128.0 | 1.07 | 4.29 | 17.2 | 7,629 | 183,105 | 610,351 | 198,364 | 680 |
| [[models/qwen3-235b-a22b]] | 235 | 22 | 44.00 | 188.0 | 1.58 | 6.31 | 25.2 | 5,194 | 124,667 | 415,558 | 228,557 | 818 |
| [[models/step-3-5-flash]] | 196.8 | 11 | 22.00 | 192.0 | 1.61 | 6.44 | 25.8 | 5,086 | 122,070 | 406,901 | 111,897 | 389 |
| [[models/llama-4-scout]] | 108.6 | 17.0 | 34.00 | 192.0 | 1.61 | 6.44 | 25.8 | 5,086 | 122,070 | 406,901 | 172,932 | 595 |
| [[models/llama-4-maverick]] | 401.6 | 17.0 | 34.00 | 192.0 | 1.61 | 6.44 | 25.8 | 5,086 | 122,070 | 406,901 | 172,932 | 595 |
| [[models/mixtral-8x22b]] | 140.6 | 39.0 | 78.00 | 224.0 | 1.88 | 7.52 | 30.1 | 4,359 | 104,631 | 348,772 | 340,053 | 1,175 |
| [[models/minimax-m2-7]] | 229 | 10 | 20.00 | 248.0 | 2.08 | 8.32 | 33.3 | 3,937 | 94,506 | 315,020 | 78,755 | 272 |
| [[models/grok-1]] | 314.0 | no record | — | 256.0 | 2.15 | 8.59 | 34.4 | 3,814 | 91,552 | 305,175 | no record | 8,278 |
| [[models/glm-4-5]] | 355 | 32 | 64.00 | 368.0 | 3.09 | 12.35 | 49.4 | 2,653 | 63,688 | 212,296 | 169,836 | 599 |

#### Hybrid attention-SSM — 10 models

| model | total B | active B | active weights GB (bf16) | KV KiB/tok | KV GB @8k | @32k | @128k | KV>1GB @ tok | KV>24GB @ tok | KV>80GB @ tok | B_eq (active basis) | L_wall tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [[models/kimi-linear-48b-a3b]] | 48 | 3 | 6.00 | 7.9 | 0.07 | 0.26 | 1.1 | 124,007 | 2,976,190 | 9,920,634 | 744,047 | 2,527 |
| [[models/granite-4-0-h-tiny]] | 6.939037248 | 1.0 | 2.00 | 8.0 | 0.07 | 0.27 | 1.1 | 122,070 | 2,929,687 | 9,765,625 | 244,140 | 835 |
| [[models/minicpm-v-4-6]] | 1.3 | 1.3 | 2.60 | 12.0 | 0.10 | 0.40 | 1.6 | 81,380 | 1,953,125 | 6,510,416 | 211,588 | 726 |
| [[models/nemotron-h-8b-base-8k]] | 8.100852736 | 8.100852736 | 16.20 | 16.0 | 0.13 | 0.54 | 2.1 | 61,035 | 1,464,843 | 4,882,812 | 988,873 | 3,393 |
| [[models/qwen3-5-35b-a3b]] | 35 | 3 | 6.00 | 20.0 | 0.17 | 0.67 | 2.7 | 48,828 | 1,171,875 | 3,906,250 | 292,968 | 1,019 |
| [[models/ling-3-0-flash]] | 124 | 5.1 | 10.20 | 22.5 | 0.19 | 0.76 | 3.0 | 43,402 | 1,041,666 | 3,472,222 | 442,708 | 1,503 |
| [[models/qwen3-8-flash-next]] | 180 | 6 | 12.00 | 24.0 | 0.20 | 0.81 | 3.2 | 40,690 | 976,562 | 3,255,208 | 488,281 | 1,723 |
| [[models/kimi-k3]] | 2780 | 104 | 208.00 | 27.0 | 0.23 | 0.91 | 3.6 | 36,168 | 868,055 | 2,893,518 | 7,523,148 | 25,556 |
| [[models/minimax-m1]] | 456 | 45.9 | 91.80 | 40.0 | 0.34 | 1.34 | 5.4 | 24,414 | 585,937 | 1,953,125 | 2,241,210 | 7,798 |
| [[models/falcon-h1-7b-base]] | 7.585648736 | 7.585648736 | 15.17 | 44.0 | 0.37 | 1.48 | 5.9 | 22,194 | 532,670 | 1,775,568 | 336,720 | 1,163 |

---

## 4. The crossover, and what it is actually for

`B_eq` on an active-weights basis is a ratio of two numbers in the record and needs no accelerator at
all:

> **`B_eq = 2 · active_params_b · 1e9 / kv_cache_bytes_per_token`**

This is the number that says how much context you need before KV streaming matters at all. Its
physical reading: **at batch `B_eq`, one step streams as many KV bytes as it streams weight bytes**,
on a single-token-per-sequence basis. Multiply by `L` for the real batch requirement at context `L`.

### 4.1 `B_eq` ranked — smallest first (weight streaming stops dominating soonest)

| model | `B_eq` active basis | `B_eq` total-weights basis | KV KiB/tok | active weights GB | total weights GB |
|---|---|---|---|---|---|
| [[models/smollm2-1-7b-instruct]] | **17,405** | 17,405 | 192.0 | 3.42 | 3.42 |
| [[models/whisper-large-v3]] | **18,835** | 18,835 | 160.0 | 3.09 | 3.09 |
| [[models/phi-3-5-mini-instruct]] ‡ | **19,434** | 19,434 | 384.0 | 7.64 | 7.64 |
| [[models/phi-3-mini-4k-instruct]] ‡ | **19,434** | 19,434 | 384.0 | 7.64 | 7.64 |
| [[models/olmo-2-1b]] | **22,659** | 22,659 | 128.0 | 2.97 | 2.97 |
| [[models/olmo-3-7b]] | **27,847** | 27,847 | 512.0 | 14.60 | 14.60 |
| [[models/qwen2-audio-7b-instruct]] | **32,032** | 32,032 | 512.0 | 16.79 | 16.79 |
| [[models/gemma-1-7b]] | **37,222** | 37,222 | 448.0 | 17.08 | 17.08 |
| [[models/gemma-2-2b]] | **49,091** | 49,091 | 104.0 | 5.23 | 5.23 |
| [[models/gemma-2-9b]] | **53,710** | 53,710 | 336.0 | 18.48 | 18.48 |
| [[models/phi-4-mini-instruct]] | **58,532** | 58,532 | 128.0 | 7.67 | 7.67 |
| [[models/gemma-3-12b-it]] | **61,986** | 61,986 | 384.0 | 24.37 | 24.37 |
| [[models/gemma-3n-e2b]] | **62,174** | 177,083 | 60.0 | 3.82 | 10.88 |

**Read this ranking carefully — the brief's framing inverts it.** The question asked was which models
"stay memory-bound longest, so care most about bandwidth not FLOPS." The lowest-`B_eq` models are the
ones where **KV streaming overtakes weight streaming soonest**, which means they are the models where
**bandwidth matters most and weight-streaming optimization matters least** — but *not* the models most
stuck on the memory wall. `B_eq` is a crossover *between two memory streams*; both sides are memory
traffic. The models that are stuck on the memory wall are the ones with a **low `L_wall`**, which is a
different and much more severe condition (§4.2).

The two are nonetheless mechanically linked, and the reason is visible in the top rows: a low `B_eq`
requires either small `P` or large `k`, and **large `k` at small `P` is precisely what pushes
`L_wall` down.** [[models/smollm2-1-7b-instruct]] tops the `B_eq` table *and* has the lowest `L_wall`
in the repo (59 tokens). The first eleven rows of this table and the first eleven rows of §4.2 are
largely the same models.

### 4.2 `L_wall` ranked — the models that cannot reach the compute roof at any batch

`L_wall = 2·P·1e9 / ((295.37 − g)·k)`. Above this context length, no batch size makes a decode step
compute-bound on H100 bf16.

| model | **`L_wall` (tokens)** | KV KiB/tok | `g` | AI ceiling @8k | @32k | @128k |
|---|---|---|---|---|---|---|
| [[models/smollm2-1-7b-instruct]] | **59** | 192.0 | 1 | 3.1 | 1.5 | 1.1 |
| [[models/whisper-large-v3]] | **63** | 160.0 | 1 | 3.3 | 1.6 | 1.1 |
| [[models/phi-3-5-mini-instruct]] ‡ | **66** | 384.0 | 1 | 3.4 | 1.6 | 1.1 |
| [[models/phi-3-mini-4k-instruct]] ‡ | **66** | 384.0 | 1 | 3.4 | 1.6 | 1.1 |
| [[models/olmo-2-1b]] | **76** | 128.0 | 1 | 3.8 | 1.7 | 1.2 |
| [[models/olmo-3-7b]] | **94** | 512.0 | 1 | 4.4 | 1.8 | 1.2 |
| [[models/qwen2-audio-7b-instruct]] | **108** | 512.0 | 1 | 4.9 | 2.0 | 1.2 |
| [[models/gemma-1-7b]] | **126** | 448.0 | 1 | 5.5 | 2.1 | 1.3 |
| [[models/gemma-2-2b]] | **167** | 104.0 | 2 | 8.0 | 3.5 | 2.4 |
| [[models/gemma-2-9b]] | **183** | 336.0 | 2 | 8.6 | 3.6 | 2.4 |
| [[models/phi-4-mini-instruct]] | **200** | 128.0 | 3 | 10.1 | 4.8 | 3.4 |
| [[models/gemma-3-12b-it]] | **211** | 384.0 | 2 | 9.6 | 3.9 | 2.5 |
| [[models/gemma-3n-e2b]] | **213** | 60.0 | 4 | 11.6 | 5.9 | 4.5 |
| [[models/qwen3-30b-a3b]] | **233** | 96.0 | 8 | 16.2 | 10.0 | 8.5 |
| [[models/gemma-4-26b-a4b]]* | **242** | 110.0 | 2 | 10.7 | 4.2 | 2.5 |
| [[models/minimax-m2-7]] | **272** | 248.0 | 6 | 15.6 | 8.4 | 6.6 |
| [[models/gemma-3-27b]] | **368** | 496.0 | 2 | 15.2 | 5.3 | 2.8 |
| [[models/glm-4-7-flash]] | **376** | 52.9 | 1 | 14.5 | 4.4 | 1.8 |
| [[models/gemma-4-12b]]* | **457** | 174.0 | 2 | 18.4 | 6.1 | 3.0 |
| [[models/gemma-4-31b]]* | **473** | 440.0 | 2 | 18.9 | 6.2 | 3.1 |

The "AI ceiling" columns are `AI(B→∞, L) = 2P/(k·L) + g` — the highest arithmetic intensity
achievable at that context length *at any batch size*. Compare to **R = 295.4**. Every value in the
table is **6x to 50x below the ridge**. [[models/gemma-3-27b]] at 128k context has an AI ceiling of
**2.8 flop/byte against a ridge of 295.4 — 106x below it, at infinite batch.** This is
`docs/02-flop-map.md` §3.5's "NO standard attention configuration reaches the compute roof" as a
per-model table.

Three of the twenty are *also* the worst models in the repo on raw KV bytes (§5): [[models/gemma-1-7b]]
(448 KiB/token at 8.5B parameters), [[models/gemma-3-12b-it]] (384 KiB), [[models/gemma-3-27b]]
(496 KiB). **The Gemma family is consistently the worst KV geometry in this repo, and it is worst
because of `head_dim: 256` with few KV heads** — Gemma 2 and 3 use head dimensions of 256 where
Llama and Qwen use 128, doubling per-head KV for the same hidden size. That is the mechanism, and it
is in the records (`head_dim: 256` on every Gemma 2/3 record, `head_dim: 128` on every Llama/Qwen 3).

### 4.3 `B_eq` on a resident-weights basis — where MoE sparsity changes the answer

For a MoE, the weights a card must hold are the **total**, not the active set. Recomputing `B_eq` with
`W = 2 · params_b · 1e9`:

| model | `B_eq` active basis | `B_eq` resident basis | ratio |
|---|---|---|---|
| [[models/gpt-oss-120b]] | 138,346 | **3,168,402** | 22.9x |
| [[models/llama-4-maverick]] | 172,932 | **4,085,286** | 23.6x |
| [[models/glm-4-7-flash]] | 110,815 | **1,152,482** | 10.4x |
| [[models/qwen3-235b-a22b]] | 228,557 | **2,441,406** | 10.7x |
| [[models/minimax-m2-7]] | 78,755 | **1,803,490** | 22.9x |
| [[models/llama-4-scout]] | 172,932 | **1,104,736** | 6.4x |
| [[models/step-3-5-flash]] | 111,897 | **2,001,953** | 17.9x |
| [[models/qwen3-5-35b-a3b]] | 292,968 | **3,417,968** | 11.7x |
| [[models/deepseek-v3]] | 1,053,051 | **19,097,222** | 18.1x |
| [[models/kimi-k2]] | 910,746 | **29,200,819** | 32.1x |
| [[models/kimi-linear-48b-a3b]] | 744,047 | **11,904,761** | 16.0x |

**This is the MoE crossover penalty stated as a ratio, and it is the reason the
[[flops/moe-experts]] record's `AI = B·(k/E)` matters at the memory level and not only the FLOP
level.** [[models/kimi-k2]] streams 2.05 TB of weights to do 32B active; on the active basis the
crossover looks like batch 911k, on the resident basis it is **29.2 million**. There is no serving
configuration that reaches the latter.

The exception that proves the mechanism: **[[models/glm-4-7-flash]] has the smallest MoE
active-to-total ratio in the table (31.2B total / 3B active = 10.4x)** and therefore the smallest
crossover penalty — and it got there by adopting MLA, which took its KV from a hypothetical 368
KiB/token down to 52.9. [[models/glm-4-5]] and [[models/glm-4-7-flash]] are the cleanest
before/after pair in the repo: **same vendor, same 368-vs-53 KiB question, 7x KV reduction against a
32B-active model.**

### 4.4 The `B*` numbers that exist, and why almost none do

Evaluating `B* = R·W / (2P − (R−g)·k·L)` on the active-weights basis:

| model | `B*` @ L=1k | @ 8k | @ 32k | @ 128k |
|---|---|---|---|---|
| [[models/falcon-h1-7b-base]] | 2,461 | none | none | none |
| [[models/mixtral-8x22b]] | 2,296 | none | none | none |
| [[models/llama-3-3-70b]] | 932 | none | none | none |
| [[models/qwen2-5-72b]] | 876 | none | none | none |
| [[models/ling-3-0-flash]] | 925 | none | none | none |
| [[models/qwen3-8-flash-next]] | 728 | none | none | none |
| [[models/deepseek-v2]] | 586 | none | none | none |
| [[models/kimi-linear-48b-a3b]] | 496 | none | none | none |
| [[models/kimi-k2]] | 441 | none | none | none |
| [[models/nemotron-h-8b-base-8k]] | 423 | none | none | none |
| [[models/deepseek-v3]] / [[models/deepseek-r1]] / [[models/deepseek-v3-2]] | 413 | none | none | none |
| [[models/hermes-4-405b]] | 361 | none | none | none |
| [[models/minimax-m1]] | 340 | none | none | none |
| [[models/kimi-k3]] | 307 | 434 | none | none |
| [[models/grok-1]] † | 337 | 28,178 | none | none |
| [[models/glm-5-3]] † | 300 | 344 | 695 | none |
| **the other 46 models** | **none** | **none** | **none** | **none** |

† computed on a **total-weights** basis because `active_params_b` is null in both records (§7.4); the
other rows are active-weights basis.

"none" = the denominator `2P − (R−g)·k·L` is ≤ 0, i.e. **no batch size reaches the ridge at that
context length.** **19 of the 64 rows have a `B*` at 1k tokens of context, and only 2 of those survive
to 8k** ([[models/kimi-k3]] at 434, [[models/grok-1]] at 28,178). **Exactly one survives to 32k** —
[[models/glm-5-3]] at **batch 695**, on a 753B model whose weights cannot fit on the card the ridge
was computed for. **The other 45 rows have no `B*` at any context length the repo records.**
`active_params_b` is non-null for the other 62 rows.

Where `B*` does exist at 1k it is **not far above the 295 that `docs/02-flop-map.md` quotes**:
[[models/deepseek-v3]] at 413, [[models/kimi-k3]] at 307, [[models/glm-5-3]] at 300. That is a useful
sanity result — the roofline figure is approximately right *at short context*, and the whole
model-specific contribution is showing **how quickly it stops being applicable.** The six widest
models in the table ([[models/hermes-4-405b]] 361, [[models/minimax-m1]] 340, [[models/grok-1]] 337,
[[models/llama-3-3-70b]] 932, [[models/qwen2-5-72b]] 876) get *closer* to 295 as they get wider,
because a larger `P` pushes the attention term down relative to the weight term — which is the same
mechanism `docs/02-flop-map.md` §5 describes as "**a WIDER model defers the attention crossover
further out**," now expressed as a batch instead of a context length.

### 4.5 What this means for the gap in `docs/02-flop-map.md` §8.2

The gap says no measured crossover exists and that
[[flops/decode-gemm]] warns real kernels cross *earlier* than the roofline. The per-model picture
sharpens what is being asked for: **a single "measured crossover batch" is not a well-defined
quantity for these models**, because for 47 of 64 there is no analytic crossover to be early
relative to. What is measurable, and what the repo lacks, is:

1. **Achieved-vs-peak bandwidth at batch 1** per model — bounded above by the AI ceiling column in
   §4.2. For [[models/gemma-3-27b]] the ceiling is 2.8 flop/byte, so a measured step time should be
   within ~3x of `bytes/3.35 TB/s` **no matter what batch size is used**, and any throughput gain
   from batching must come from amortizing *bandwidth* (fewer weight passes), never from reaching
   compute.
2. **The batch at which aggregate decode throughput stops scaling** — which for these models is set
   by memory capacity and kernel efficiency, not by the ridge. The only existing measurement of this
   shape in the repo is [[benchmarks/mpt7b-a100-bs1-per-user-decode-tps]] against
   [[benchmarks/mpt7b-a100-bs64-aggregate-output-tps]]: batch 1 → 64 raises aggregate throughput
   **13.9x** (57.6 → 800 tok/s) while per-user decode speed **falls 4.6x** (57.6 → 12.5 tok/s). That
   is a 13.9x throughput gain over a 64x batch increase — i.e. **efficiency per token fell 4.6x**, which
   is the empirical signature of staying memory-bound the whole way. MPT-7B is a 7B model whose
   `B_eq` is ~89,000 and whose `L_wall` is well under 1,000 tokens, so this is exactly the predicted
   regime and the prediction holds. But it is one model, one engine (FasterTransformer, static
   batching), one part, and a 40GB A100 the repo has **no accelerator record for** — the benchmarks
   say so themselves. It is evidence, not a measurement of the crossover.

I have not added a `benchmark` record for it, because both numbers already exist as records and a
third would be a duplicate, not a measurement.

---

## 5. The 10 models with the worst long-context memory envelope

Ranked by the context length at which KV cache for **one** sequence passes **80 GB** — beyond this a
single request cannot fit on an 80GB card, whatever the weights do. 80GB is
[[accelerators/nvidia-h100-sxm]]'s `vram_gb`.

| # | model | KV KiB/tok | **KV > 80 GB @ token** | advertised `max_position_embeddings` | KV @ advertised max ctx |
|---|---|---|---|---|---|
| 1 | [[models/olmo-3-7b]] | 512.0 | **152,587** | 65,536 | 34.4 GB |
| 2 | [[models/qwen2-audio-7b-instruct]] | 512.0 | **152,587** | 8,192 | 4.3 GB |
| 3 | [[models/hermes-4-405b]] | 504.0 | **155,009** | 131,072 | 67.6 GB |
| 4 | [[models/gemma-3-27b]] | 496.0 | **157,510** | 131,072 | 66.6 GB |
| 5 | [[models/gemma-1-7b]] | 448.0 | **174,386** | 8,192 | 3.8 GB |
| 6 | [[models/gemma-4-31b]]* | 440.0 | **177,556** | 262,144 | 118.1 GB |
| 7 | [[models/gemma-3-12b-it]] | 384.0 | **203,450** | 131,072 | 51.5 GB |
| 8 | [[models/phi-3-5-mini-instruct]] ‡ | 384.0 | **203,450** | 131,072 | 51.5 GB |
| 9 | [[models/phi-3-mini-4k-instruct]] ‡ | 384.0 | **203,450** | 4,096 | 1.6 GB |
| 10 | [[models/glm-4-5]] | 368.0 | **212,296** | 131,072 | 49.4 GB |

Four observations that change how this table is used:

1. **Only one of the ten — [[models/gemma-4-31b]]*, at 118.1 GB — actually breaks an 80GB card at its
   own advertised context.** Two more come close enough to matter operationally:
   [[models/hermes-4-405b]] (67.6 GB) and [[models/gemma-3-27b]] (66.6 GB) each advertise 128k and
   would leave ~12 GB for weights, activations and workspace — and for [[models/hermes-4-405b]] the
   weights alone are 811.71 GB in bf16, so it needs quantization and sharding regardless.
   The table ranks **KV geometry**, not usability: rows 1, 2, 5 and 9 advertise contexts far *below*
   their own 80GB threshold and are comfortably servable.
2. **[[models/glm-4-5]] is the only MoE in the top ten and the only model in it that could have been
   fixed.** Its 368 KiB/token is the direct consequence of plain GQA, and the record says so
   explicitly: "GLM-4.5 does 32B active in 368 KiB/token — 5.4x the KV for slightly less compute"
   versus [[models/deepseek-v3]]'s 68.6 KiB. Its own successor [[models/glm-4-7-flash]] fixed it with
   MLA at 52.9 KiB. **The fix is one architectural decision, already made by the same vendor one
   generation later.**
3. **The Gemma pattern is `head_dim: 256`.** Every Gemma 2/3 model in the top ten has 256-wide heads
   against Llama's and Qwen's 128, and [[models/gemma-4-31b]]* stacks that on top of 60 layers. This is a design choice with a long history, not a bug, but it is the single most
   consequential memory decision across the models in this repo.
4. **[[models/phi-3-5-mini-instruct]] would not have appeared here at all** with its recorded
   (erroneous) 24,576 B/token — that value puts it **56th** of 64 in this ranking. The §2.1 correction
   moves it to **8th**. The comparison that makes the correction matter: at 3.82B parameters it carries
   **exactly the same 384 KiB/token as [[models/olmo-3-7b]], a 7.3B model**, and **3x the KV of
   [[models/phi-4-mini-instruct]] at 3.84B** — a near-identical sibling one release later that added
   GQA (8 KV heads instead of 32). Only 4 of the 64 rows in §3 carry more KV per token.

### 5.1 Models whose *advertised* full context exceeds 80 GB of KV for one sequence

Six, on the raw `max_position_embeddings` figure. Three of the six are excluded below because the
record itself says the advertised context is not a serving figure:

| model | `max_position_embeddings` | **KV @ full context** | KV KiB/tok |
|---|---|---|---|
| [[models/llama-4-maverick]] | 1,048,576 | **206.2 GB** | 192.0 |
| [[models/seed-oss-36b]] | 524,288 | **137.4 GB** | 256.0 |
| [[models/gemma-4-31b]]* | 262,144 | **118.1 GB** | 440.0 |
| [[models/glm-5-3]] | 1,048,576 | **94.2 GB** | 87.8 |

The two excluded rows are the same failure mode in both cases, and both records warn about it in their
own notes:

- **[[models/llama-4-scout]]** carries the same 10,485,760 `max_position_embeddings` as Maverick. At
  192 KiB/token that is **2,061.6 GB** of KV. Its own record calls the figure "a nominal
  training/rope-range figure, NOT a usable serving context" and notes real deployments land at 10M
  only with heavy KV offload. Not reported as a finding.
- **[[models/minimax-m1]]**'s 10,240,000 gives **419.4 GB** naively, but that over-counts: the record's
  40,960 B/token is already net of the hybrid pattern (10 full-attention layers of 80), so the figure
  is arithmetically correct and genuinely unusable — the record notes the context was "built for very
  long RL/test-time-compute runs," not for KV-resident serving.

---

## 6. The cheap end, for contrast

The bottom of the table is where the architectural decisions pay, and the spread is **two orders of
magnitude in KV per token** across models of similar size:

| model | total B | active B | KV KiB/tok | KV @128k | vs worst in repo |
|---|---|---|---|---|---|
| [[models/kimi-linear-48b-a3b]] | 48 | 3 | **7.9** | 1.06 GB | 63x cheaper than [[models/gemma-3-27b]] |
| [[models/granite-4-0-h-tiny]] | 6.94 | 1.0 | **8.0** | 1.07 GB | 62x |
| [[models/minicpm-v-4-6]] | 1.3 | 1.3 | **12.0** | 1.61 GB | 41x |
| [[models/nemotron-h-8b-base-8k]] | 8.10 | 8.10 | **16.0** | 2.15 GB | 31x |
| [[models/qwen3-5-35b-a3b]] | 35 | 3 | **20.0** | 2.68 GB | 25x |
| [[models/kimi-k3]] | 2780 | 104 | **27.0** | 3.62 GB | 18x |
| [[models/minimax-m1]] | 456 | 45.9 | **40.0** | 5.37 GB | 12x |
| — | | | | | |
| [[models/deepseek-v3]] (MLA) | 671 | 37 | **68.6** | 9.21 GB | 7.2x |
| [[models/gpt-oss-120b]] (sliding) | 116.8 | 5.1 | **72.0** | 9.66 GB | 6.9x |
| [[models/glm-4-5]] (plain GQA) | 355 | 32 | **368.0** | 49.4 GB | 1.35x |
| [[models/olmo-3-7b]] (MHA) | 7.3 | 7.3 | **512.0** | 68.7 GB | 1.0x |

**The three structural levers, ranked by effect, all visible in the table:**

1. **Hybrid attention** ([[models/kimi-k3]]: 27 KiB/token at 2.78T total / 104B active).
   7.5M total parameters and the **lowest-but-few** KV in the repo. Its record states the ratio
   directly: "KV per token is LOWER than Qwen3-8B's 144 KiB (a dense 8B) and less than half of
   DeepSeek-V3's 68.6 KiB."
2. **MLA** ([[models/deepseek-v3]]: 68.6 KiB/token for 37B active, against [[models/glm-4-5]]'s
   368 KiB for 32B active — **5.4x less KV for more compute**). Confirmed by
   [[flops/mla-latent-attention]]'s 85.3x reduction against the naive GQA figure.
3. **Aggressive GQA** ([[models/gpt-oss-120b]]: 72 KiB/token at g=8, on a 116.8B MoE that fits one
   80GB device *only* because its weights ship MXFP4).

And the counter-example in the same table: **[[models/olmo-3-7b]] is a 7.3B model with 512 KiB/token
— the worst in the repo — because it is MHA (32 query / 32 KV heads, g=1).** A model small enough to
be irrelevant on every other axis is undone entirely by KV geometry. That is the clearest possible
statement of what `kv_cache_bytes_per_token` is for.

**One caution on the hybrid rows.** These models' small KV figures come with a fixed per-sequence
state that the table above does not show: [[models/nemotron-h-8b-base-8k]] holds **96 MiB per
sequence** of Mamba-2 state (48 layers × 2 MiB), [[models/falcon-h1-7b-base]] holds **66 MiB**,
[[models/mamba-codestral-7b]] holds **128 MiB**. Both numbers are in those records' notes. The fixed
part scales with **concurrency** rather than context, so at 100 concurrent sequences it is 9.4 GB for
Nemotron-H — larger than its entire 128k KV budget. `kv_cache_bytes_per_token` alone is an incomplete
memory model for every hybrid model in this repo, which is why §1.3 asks for
`state_bytes_per_sequence`.

---

## 7. What this document does not establish

1. **No measurement.** Every crossover here is analytic, exactly as `docs/02-flop-map.md` §8.2 says.
   [[flops/decode-gemm]]'s warning stands unchanged: "Real kernels hit their crossover **earlier**
   than the ideal roofline." §4.4 shows that for 47 of 64 models there is no analytic crossover to
   be early relative to, which makes that warning **harder** to apply, not easier.
2. **The ridge is anchored to H100 SXM bf16.** §1.1 gives the per-part ridges but the crossover
   arithmetic is only worked for H100. On [[accelerators/nvidia-a100-80gb-sxm4]] (R = 153.0) the
   `L_wall` values roughly **double**; on [[accelerators/nvidia-l40s]] (R = 419.0) they roughly
   halve. No model in this table has an `L_wall` above ~57,000 tokens, so **no part in the repo's
   accelerator set makes a single one of these models compute-bound at a realistic serving context.**
   That is a stronger and more uncomfortable version of the claim in §8.
3. **The AI expression assumes the ideal weight-GEMM AI = B.** A real MoE decode step has
   `AI = B·(k/E)` per [[flops/moe-experts]] — so for the 20 MoE rows, the true AI is **lower than
   computed here** and the true `L_wall` is correspondingly **higher**. [[models/deepseek-v3]]'s
   k/E = 1/32 means the real AI ceiling at 8k is not 129.5 but ~5, and its real `L_wall` is not 3,577
   but ~115,000. The MoE rows in §4.2 are therefore **optimistic**, and the correct reading is that
   MoE models are *more* memory-bound than the table shows.
4. **`active_params_b` is null on 2 records** — [[models/grok-1]] and [[models/glm-5-3]] — so
   `W_act`, `B_eq` and `L_wall` are `no record` for them and only the total-weights `B*` is given
   (§4.4). [[models/glm-5-3]]'s own note declines to state an activated figure and estimates ~40.9B
   from config as "indicative only"; **that estimate is not used here.** No model dimension was
   invented anywhere in this document — where a record is silent, the table says `no record`.
5. **The `*` Gemma 4 rows are derivations, not records.** §2.6 gives the arithmetic and the config
   fields. They are marked everywhere they appear.
6. **fp8/int8 KV is not modelled** in §3–§5, but it can be bounded. Per [[flops/decode-attention]],
   halving KV bytes doubles attention AI to `2g`, so with `k' = k/2` and `g' = 2g` the wall moves to
   `L_wall' = 2P/((R−2g)·k/2)` — **`L_wall` grows by `2(R−g)/(R−2g)`, which is 2.007x at `g = 1` and
   2.122x at `g = 16`.** That lifts [[models/deepseek-v3]] from `L_wall` 3,577 to 7,179 tokens and
   [[models/smollm2-1-7b-instruct]] from 59 to 119 — real, and still far below any serving context.
   So on H100 bf16 the KV-dtype lever is a **capacity** lever, not a compute-bound lever: it halves
   every 80GB threshold in §5 and changes no conclusion about the roofline. Same conclusion the
   [[flops/decode-attention]] record reaches by a different route.

---

## Related documents

- [02-flop-map.md](02-flop-map.md) — the ridge points, the flop classes, and §8's list of what the
  records cannot answer. This document is the per-model completion of its §3.1 table and its §8.2 gap.
- [01-hardware-selection.md](01-hardware-selection.md) — which part to buy for each regime; §1.1 here
  gives the ridge input that selection needs.
- [04-quant-selection.md](04-quant-selection.md) — the one lever that changes the arithmetic intensity
  without touching the model, and the reason most of the large models in §5 are quantized in practice.
- [05-known-traps.md](05-known-traps.md) — the software failures behind §4's crossovers not
  materialising.