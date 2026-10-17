# 27 — Affordable 35B-class fit matrix

Which recorded accelerators can hold a ~35B-class model, and what does the
cheapest one cost per unit of decode throughput.

> **Refresh note (docs/fit-matrix-oct17, Oct-17, this pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `2925c06` (200 accelerator records, 218 supply records, 3,050 benchmarks — #179 on top of `cd120be`). The selection rule still yields **148 qualifying / 43 priced / 105 unpriced** (68 of the unpriced fit 35B-A3B, 37 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #179 adds no accelerator and no purchase price. 48 of the 55 new rows are llama.cpp Vulkan #10879 **no-flash-attention** Llama-2-7B Q4_0 pp512/tg128 pairs on cards this doc already covers (5090, 4090, 5080, 3090, 4080 Super, 5070 Ti, 4070 family, 5060 Ti, 3060, 3080 10GB, 5090 Laptop, 7900 XTX, 9070 XT, R9700, MI50, A100 80GB PCIe, GB10, M4 Max, Tesla P100/P40/M40). A 7B dense model is the wrong byte count for the 20 GB ceiling, so none of them enters §4b and none moves a rank. The seven that are the workload are a second protocol on a part the table already prices:
>
> - **Strix Halo 8060S, rank 37, and the long-context column does not collapse.** [[accelerators/amd-radeon-8060s-ryzen-ai-max]] (128 GB unified, 256 GB/s, 35B ceiling 256/20 = **12.8** tok/s *(derived)*, $1,900 → $148.44/tok-s *(derived)*). The homelab page's single-run llama-bench (`-fa 1 -r 1 --mmap 0`) measures Qwen3-30B-A3B UD-Q4_K_XL under Vulkan AMDVLK 2025.Q2.1 at **81.79** tok/s at depth 0 and **85.11** at depth 130560 ([[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-amdvlk-tg128]], [[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-amdvlk-tg128-d130560]]). Against the 12.8 floor that is **6.4×** and **6.6×** *(derived)* (81.79/12.8 = 6.39, 85.11/12.8 = 6.65). The same build's tuned-ROCm column (ROCm 7.0.0-0, hipBLASLt) is **64.17** at depth 0 and **67.66** at depth 130560 ([[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-rocm-tuned-tg128]], [[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-rocm-tuned-tg128-d130560]]) → **5.0×** and **5.3×** *(derived)* (64.17/12.8 = 5.01, 67.66/12.8 = 5.29). Depth 130560 is ~128k of primed context, and both columns *hold* — AMDVLK is slightly faster at the long depth than at zero. That contradicts the 5-run wiki column already on file for the same model on the same box: AMDVLK there is 130.56 at depth 0 and **27.32** at depth 131072 ([[benchmarks/wiki-strix-8060s-qwen3-30b-a3b-udq4-amdvlk-tg128-d0]], [[benchmarks/wiki-strix-8060s-qwen3-30b-a3b-udq4-amdvlk-tg128-d128k]]), and the plain ROCm b6702 column already filed is 64.17 at depth 0 collapsing to **4.98** at depth 130560 ([[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-rocm-tg128-d130560]]). Same silicon, three protocols, three answers at ~128k (85.11, 27.32, 4.98). The single-run rows are not averaged into the 5-run ones, and they do not replace the 72.35 cell §4b ranks on ([[benchmarks/lhl-8060s-qwen3-30b-a3b-ud-q4-k-xl-vulkan-tg1]], 5.7×) or the 130.56 depth-0 row the §4b prose calls 10.2× (130.56/12.8 = 10.20, *(derived)*). What they do is bound the claim: the floor stays a floor at 128k on this build, and the collapse §4b warns about is a backend effect, not a bandwidth one. $1,900/85.11 = **$22.32 per measured tok/s** *(derived)* at depth 130560 — still above the 3090's $5.31 at its 4k cell, and a different context depth, so it does not re-rank. pp512 on the new AMDVLK column is 741.6 at depth 0 and 755.14 at depth 130560 ([[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-amdvlk-pp512]], [[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-amdvlk-pp512-d130560]]); the tuned-ROCm pp512 at depth 130560 is 659.07 ([[benchmarks/shhl-8060s-qwen3-30b-a3b-udq4-rocm-tuned-pp512-d130560]]). The depth-0 tuned pp512 of 651.93 was already filed and was not duplicated.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16, this pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `bd282fe` (200 accelerator records, 218 supply records, 2,995 benchmarks — #176 and #177 on top of `5b1cadc`). The selection rule now yields **148 qualifying / 43 priced / 105 unpriced** (68 of the unpriced fit 35B-A3B, 37 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #177 is a services telemetry re-read and touches no accelerator, supply, or benchmark. #176 adds no purchase price. What it does add is a bandwidth that was previously null: [[accelerators/furiosaai-rngd-renegade]] moved from `memory_bandwidth_gbps: null` / `vram_gb: null` to **48 GB HBM3 at 1,500 GB/s** (vendor product page, cited on the record). 48 ≥ 24, 48 ≥ 20 and 48 ≥ 48, so the fit flags are **Y/Y/Y**; the 35B ceiling is 1500/20 = **75.0 tok/s** *(derived)*, which slots it in §4 between [[accelerators/biren-br100]] (80.0) and [[accelerators/huawei-ascend-910b]] (70.0). It stays unpriced: the only Furiosa supply row, [[supply/furiosaai-rngd-eval-only-channel]], has `price_usd: null`, `kind: oem-direct`, and its `accelerator_ids` still point at the Warboy, not the RNGD. The seven new benchmarks are vendor ranges and multi-card aggregates (4× EXAONE 3.5 32B batch-1 decode, 4× Qwen3-32B aggregate TPS, a Hot Chips ~10B range, a GPT-J 6B queries/s figure). None is a single-card batch-1 A3B decode, so none enters §4b and none moves a rank.
>
> **Refresh note (docs/fit-matrix-oct16-3090-a3b, Oct-16, this pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `5b1cadc` (200 accelerator records, 218 supply records, 2,988 benchmarks — the Phase-C Oct-16 p20 llama.cpp Vulkan #10879 rows for 12 previously uncovered cards, #172, on top of `ece538c`). The selection rule now yields **147 qualifying / 43 priced / 104 unpriced** (67 of the unpriced fit 35B-A3B, 37 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order), and #172 adds no purchase price. What it does add to the *gate* is five bandwidth-qualified cards that fit nothing in this doc's model class, all `-/-/-`, all unpriced, so they extend the §4 close rather than the table: [[accelerators/amd-radeon-rx-6800]] (16 GB, 512 GB/s, 35B ceiling 25.6 tok/s *(derived)*), [[accelerators/nvidia-rtx-2070-super]] and [[accelerators/nvidia-rtx-3060-ti]] (both 8 GB, 448 GB/s, ceiling 22.4 *(derived)*), [[accelerators/amd-radeon-rx-6750-xt]] (12 GB, 432 GB/s, ceiling 21.6 *(derived)*) and [[accelerators/amd-radeon-rx-6700-xt]] (12 GB, 384 GB/s, ceiling 19.2 *(derived)*). The other seven new records miss the gate: [[accelerators/amd-radeon-rx-7600]] (288 GB/s), [[accelerators/amd-radeon-rx-6650-xt]] (280), [[accelerators/amd-radeon-rx-6600-xt]] (256) and [[accelerators/amd-radeon-rx-6600]] (224) are under 300 GB/s with no `<= $3,000` purchase record, and [[accelerators/intel-arc-b570]], [[accelerators/nvidia-gtx-1080-ti]] and [[accelerators/nvidia-rtx-a2000]] have a null bandwidth. The 44 new benchmarks are Llama-2-7B Q4_0, so none of them is an A3B measurement. What this pass actually lands is a hole in §4b: the part the headline calls the honest mainstream buy, [[accelerators/nvidia-rtx-3090]], had a full Qwen3-30B-A3B Q4_K depth sweep in the corpus since #147 and was never cited. The 4090, 6000 Ada and A6000 sweeps from the same ingest were in the same state. None of them moves a rank — they are the same lab, the same model, a longer context than the one cell §4b already quoted.
>
> - **RTX 3090, rank 11, now has the sweep.** [[accelerators/nvidia-rtx-3090]] (24 GB, 936 GB/s, 35B ceiling 46.8 tok/s *(derived)*, $950 → $20.30/tok-s *(derived)*). Hardware Corner's March 2026 llama.cpp table measures Qwen3 30B A3B (Q4_K), batch 1, at **179.0 / 98.3 / 58.8 / 32.8** tok/s for 4k / 16k / 32k / 64k ([[benchmarks/hc26-3090-qwen3-30b-a3b-tg-4k]], [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-16k]], [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-32k]], [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-64k]]). Against the 46.8 floor that is **3.8× / 2.1× / 1.3× / 0.70×** *(derived)* (179.0/46.8 = 3.82, 98.3/46.8 = 2.10, 58.8/46.8 = 1.26, 32.8/46.8 = 0.70). The floor is crossed between 32k and 64k — the same context shape §4b already records on the PRO 5000, just one bucket earlier. A second 4k cell on the same card, same quant, is **153.6** ([[benchmarks/hc26-3090-qwen3-30b-a3b-q4k-4k-tg]], 153.6/46.8 = **3.3×** *(derived)*); the two pages do not agree at 4k, so both are kept and neither is averaged. At the 179.0 cell the $950 sticker is **$5.31 per measured tok/s** *(derived, 950/179.0)*, under the 3090 Ti's $9.67 at its 16k cell — different context depths, so that is not a re-rank, it is why the 3090 now leads the *measured* cheap-fit read at short context and loses it once the KV is deep.
> - **RTX 4090, same sweep, higher multiplier.** [[accelerators/nvidia-rtx-4090]] already had the 16k cell at 109.57 (2.2×). The March table's other depths are **207.3 / 139.2 / 105.1 / 68.2** at 4k / 16k / 32k / 64k ([[benchmarks/hc26-4090-qwen3-30b-a3b-tg-4k]], [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-16k]], [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-32k]], [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-64k]]) against the 50.4 ceiling → **4.1× / 2.8× / 2.1× / 1.4×** *(derived)* (207.3/50.4 = 4.11, 139.2/50.4 = 2.76, 105.1/50.4 = 2.09, 68.2/50.4 = 1.35). The 16k cell (139.2) is a different page from the 109.57 already in §4b; both stand. The card never crosses the floor inside 64k. $2,000 / 207.3 = **$9.65 per measured tok/s** *(derived)* at 4k — essentially the 3090 Ti's 16k figure, at a quarter of the context.
> - **RTX 6000 Ada and RTX A6000, unpriced, same shape.** [[accelerators/nvidia-rtx-6000-ada]] (48 GB, 960 GB/s, ceiling 48.0 *(derived)*, §4) goes **199.6 / 120.1 / 74.3 / 40.8** at 4k / 16k / 32k / 64k ([[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-4k]], [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-16k]], [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-32k]], [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-64k]]) → **4.2× / 2.5× / 1.5× / 0.85×** *(derived)* (199.6/48.0 = 4.16, 120.1/48.0 = 2.50, 74.3/48.0 = 1.55, 40.8/48.0 = 0.85). The floor is crossed between 32k and 64k, same as the 3090. [[accelerators/nvidia-rtx-a6000]] (48 GB, 768 GB/s, ceiling 38.4) already had 75.35 at 16k; the 4k cell is **131.2** ([[benchmarks/hc26-a6000-qwen3-30b-a3b-q4k-4k-tg]], 131.2/38.4 = **3.4×** *(derived)*). Neither has a purchase price, so neither enters §3.
> - **Two siblings, not a new rank.** The 3090 Ti's 32k cell is **91.97** ([[benchmarks/hwcorner-rtx-3090-ti-qwen3-30b-q4kxl-decode-32k]], 91.97/50.4 = **1.8×** *(derived)*), next to the 98.25 already quoted at 16k — the Ti does not have the 3090's 4k cell, so the two cards are not comparable at short context from this corpus. The 5090's second 4k page is **226.1** ([[benchmarks/hc26-5090-qwen3-30b-a3b-q4k-4k-tg]], 226.1/89.6 = **2.5×** *(derived)*) beside the 234.3 already in §4b; same quant, same depth, two pages, 4% apart. Not averaged.
> - **Dense 27B on the same 3090, under the floor, as the ceiling predicts.** Qwen3.5 27B (Q4_K) decodes at **33.5** tok/s at 4k and **28.8** at 64k ([[benchmarks/hc26-3090-qwen35-27b-q4k-4k-tg]], [[benchmarks/hc26-3090-qwen35-27b-q4k-64k-tg]]) against the 27B ceiling of 936/16 = **58.5** tok/s *(derived)* → 33.5/58.5 = **0.57×** at 4k *(derived)*. That is the dense half-the-floor §4b already records on the PRO 5000, now on the card this doc actually recommends. The MoE-vs-dense gap on one card is 179.0 vs 33.5 at the same 4k bucket — the reason the 35B-A3B column, not the 27B column, is what a 24 GB card is for. The named 35B target in MXFP4 on the same card is **111.2** at 4k ([[benchmarks/hc26-3090-qwen35-35b-mxfp4-4k-tg]], 111.2/46.8 = **2.4×** *(derived)*), below the Q4_K 30B-A3B cell because MXFP4 is a different byte count, not because the card got slower.
> - **Arc Pro B60, priced, fits, and the measured 35B-A3B is slow.** [[accelerators/intel-arc-pro-b60]] (rank 16, 24 GB, 456 GB/s, ceiling 22.8 *(derived)*, $650 → $28.51/tok-s *(derived)*) has a SYCL llama.cpp row of Qwen3.5-35B-A3B at **Q8** — not Q4 — decoding at **8.3** tok/s ([[benchmarks/comm-arc-pro-b60-qwen35-35b-a3b-q8-sycl-tg]]). 8.3/22.8 = **0.36×** the Q4 floor *(derived)*. Q8 reads about twice the bytes the floor assumes, so the ratio is a lower bound on a Q4 run and still well under it; the record's own note calls it a capacity win, not a speed win. It does not enter the §4b short-context table: wrong quant, and one number with no depth. $650/8.3 = **$78.31 per measured tok/s** *(derived)*, an order of magnitude worse than the 3090's 4k cell.

> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `a2e890e`
 (188 accelerator records, 218 supply records, 2,944 benchmarks — the Phase-C Oct-16 p19 official DGX Spark llama-bench rows, #169, on top of `1b2f366`). The selection rule still yields **142 qualifying / 43 priced / 99 unpriced** (67 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #169 adds no purchase price and no accelerator. What it does add is the first **maintainer** llama.cpp decode series on a part this doc already treats as outside the gate, and the model is the 30B-A3B class:
>
> - **DGX Spark GB10, measured, and it does not qualify.** [[accelerators/nvidia-gb10]] is the same shape as the Jetson AGX Thor row already in the note below: 128 GB unified LPDDR5x at **273 GB/s**. That clears the 24 GB fit bar and misses the 300 GB/s bandwidth floor, and there is no purchase-price supply record at `<= $3,000`, so the selection rule keeps it out of the 142. The official ggml table (build 11fb327bf, CUDA, flash attention, batch 1) runs **Qwen3-Coder-30B-A3B-Instruct Q8_0** — 30.25 GiB on disk, not the 20 GB Q4 the ceiling assumes — at tg32 **61.06** tok/s with no prior context ([[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-tg32-d0-b7941]]), then **54.77 / 48.02 / 40.23 / 30.21** after 4k / 8k / 16k / 32k of depth ([[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-tg32-d4096-b7941]], [[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-tg32-d8192-b7941]], [[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-tg32-d16384-b7941]], [[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-tg32-d32768-b7941]]). The same arithmetic the ceiling column uses gives 273/20 = **13.7** tok/s *(derived)*, so the depth-0 row lands at 61.06/13.7 = **4.5×** that floor *(derived)* and the 32k row at 30.21/13.7 = **2.2×** *(derived)*. Q8 reads more bytes than the Q4 the floor assumes, so 4.5× is a lower bound on how far a Q4 A3B would sit above it, not a Q4 measurement. pp2048 at depth 0 is **2,986.97** tok/s ([[benchmarks/ggml-cuda-gb10-spark-qwen3-coder-30b-a3b-pp2048-d0-b7941]]). It does not enter §3 or §4b: no price, and the part is outside the selection rule. The bandwidth match with Thor is exact (both 273 GB/s) and the decode is not — Thor's third-party Qwen3.6-35B-A3B **FP8** row is 14.6 tok/s, this Q8_0 A3B row is 61.06 — different quant, different engine, same bus.
> - **The same box, three other MoEs, none of them the 35B-A3B target.** gpt-oss-20b MXFP4 decodes at **83.43** tg32 ([[benchmarks/ggml-cuda-gb10-spark-gptoss20b-tg32-d0-b7941]], 11.27 GiB), 83.43/13.7 = **6.1×** the 20 GB ceiling *(derived)* — the wrong byte count for a 20B MXFP4, same caveat the RTX 4080/5080 note already uses. gpt-oss-120b MXFP4 is **58.72** ([[benchmarks/ggml-cuda-gb10-spark-gptoss120b-tg32-d0-b7941]], 59.02 GiB) → **4.3×** *(derived)*, and it is the model that actually uses the 128 GB pool. GLM-4.7-Flash Q8_0, which llama-bench labels `deepseek2 30B.A3B`, is **48.68** ([[benchmarks/ggml-cuda-gb10-spark-glm47-flash-tg32-d0-b7941]], 29.65 GiB) → **3.6×** *(derived)*. None of the three is a Q4 35B-A3B, so none enters §4b. The earlier third-party gpt-oss-120b CUDA cell on this part stays where it was (52.87, a different build); the maintainer number is 58.72.

> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `2e37ebe` (188 accelerator records, 218 supply records, 2,884 benchmarks — the Phase-C Oct-16 p18 llama.cpp #15013 rows, #167, on top of `ed08dce`). The selection rule now yields **142 qualifying / 43 priced / 99 unpriced** (67 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #167 adds no purchase price. What it does add is four accelerator records, all of which clear the 8 GB / 300 GB/s gate and none of which has a supply row, plus the first llama.cpp Llama-2-7B Q4_0 tg128 cell on a part §4 already listed:
>
> - **RTX 4090 D, new, unpriced, fits 35B/27B and not 70B.** [[accelerators/nvidia-rtx-4090-d]] is the China-only AD102-250 cut: 24 GB, **the same 1,008 GB/s** as [[accelerators/nvidia-rtx-4090]], TGP 425 W instead of 450 W. 35B ceiling 1,008/20 = **50.4** tok/s *(derived)* — identical to the priced 4090 at rank 22. It slots into §4 between the MI50/MI60 pair (51.2) and the RTX 6000 Ada (48.0). The scoreboard row is Llama-2-7B Q4_0, not an A3B model, so it does not enter §4b: tg128 **191.57** tok/s ([[benchmarks/llamacpp-cuda-rtx-4090-d-llama27b-q40-fa-tg128]]) and pp512 **12,506.97** ([[benchmarks/llamacpp-cuda-rtx-4090-d-llama27b-q40-fa-pp512]]). Dividing the 7B decode by the 20 GB 35B denominator — the only ceiling this doc computes, and the wrong byte count for a 3.56 GiB model — gives 191.57/50.4 = **3.80×** *(derived)*. That ratio is a small-model calibration, not a claim the card decodes a 35B-A3B at 192 tok/s. No street price, so it cannot take the 4090's rank.
> - **RTX 5090 Laptop, new, unpriced, same fit shape.** [[accelerators/nvidia-rtx-5090-laptop]] is a GB203 mobile part, 24 GB GDDR7, **896 GB/s**, TGP up to 175 W. 35B ceiling 896/20 = **44.8** tok/s *(derived)*, tied with [[accelerators/nvidia-rtx-pro-4500-blackwell]] (same bus, same bandwidth, a 32 GB workstation card). It does not fit 70B. tg128 **158.14** ([[benchmarks/llamacpp-cuda-rtx-5090-laptop-llama27b-q40-fa-tg128]]), pp512 **7,641.89** ([[benchmarks/llamacpp-cuda-rtx-5090-laptop-llama27b-q40-fa-pp512]]). 158.14/44.8 = **3.53×** the 20 GB ceiling *(derived)*, same caveat: the model is Llama-2-7B. The scoreboard prints a 384-bit bus; the accelerator record keeps the 256-bit spec and does not trust that width. Not a desktop 5090 (1,792 GB/s, rank 12) and not priced as a bare GPU — the part ships inside a laptop.
> - **Quadro RTX 8000 and Quadro RTX 6000, new, unpriced, same 672 GB/s.** Turing TU102, not the RTX 6000 Ada and not the RTX PRO 6000 Blackwell. [[accelerators/nvidia-quadro-rtx-8000]] is **48 GB** so it fits 70B (`Y/Y/Y`); [[accelerators/nvidia-quadro-rtx-6000]] is **24 GB** and does not (`Y/Y/-`). Both ceiling at 672/20 = **33.6** tok/s *(derived)*, tied with [[accelerators/nvidia-rtx-pro-4000-blackwell]]. The 8000's extra 24 GB buys the stretch model and buys no bandwidth. tg128 is **91.88** on the 8000 ([[benchmarks/llamacpp-cuda-quadro-rtx-8000-llama27b-q40-fa-tg128]]) and **93.11** on the 6000 ([[benchmarks/llamacpp-cuda-quadro-rtx-6000-llama27b-q40-fa-tg128]]) — within 2% of each other, which is what equal bandwidth predicts. 93.11/33.6 = **2.77×** and 91.88/33.6 = **2.73×** *(derived)*, again against a 7B model, so neither enters §4b. pp512 is 2,934.33 and 2,941.85 ([[benchmarks/llamacpp-cuda-quadro-rtx-8000-llama27b-q40-fa-pp512]], [[benchmarks/llamacpp-cuda-quadro-rtx-6000-llama27b-q40-fa-pp512]]).
> - **RTX PRO 4500 Blackwell, already in §4, now has its tg128 cell.** [[accelerators/nvidia-rtx-pro-4500-blackwell]] (32 GB, 896 GB/s, ceiling 44.8 *(derived)*, unpriced). The pp512 sibling was already stored; #167 adds tg128 **168.9** ([[benchmarks/llamacpp-cuda-rtx-pro-4500-blackwell-llama27b-q40-fa-tg128]]). 168.9/44.8 = **3.77×** the 20 GB ceiling *(derived)*. Same Llama-2-7B protocol as the three new cards, so the four tg128 numbers are comparable to each other and not to the §4b A3B column. 168.9 vs the 5090 Laptop's 158.14 on the same 896 GB/s is a power-cap gap (workstation card vs a 95–175 W mobile SKU), not a bandwidth gap.

> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `1c2a269` (184 accelerator records, 218 supply records, 2,876 benchmarks — the Phase-C Oct-16 p17 tt-metal matrix rows, #165, on top of `3438151`). The selection rule still yields **138 qualifying / 43 priced / 95 unpriced** (63 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #165 adds no purchase price and no accelerator. What it does add is the first tt-metal decode numbers on three parts this doc already ranks, and none of them is an A3B-class model, so none of them enters §4b or moves a rank:
>
> - **Wormhole n300, rank 24, and the only one of the three that fits 35B-A3B.** [[accelerators/tenstorrent-wormhole-n300]] (24 GB, 576 GB/s, 35B ceiling 28.8 tok/s *(derived)*, $1,449 → $50.31/tok-s *(derived)*). The vendor's own tt-metal matrix measures **Llama-3.2-11B-Vision** at **18.1** tok/s per user, TP=2 across both chips, batch 16, input length 128 ([[benchmarks/ttmetal-n300-llama32-11b-vision-tp2-b16-tpsu]]; aggregate 289.6 = 18.1 × 16, not a per-request rate). TTFT on that row is 714 ms ([[benchmarks/ttmetal-n300-llama32-11b-vision-tp2-b16-ttft]]). Dividing the per-user figure by the 20 GB 35B ceiling — the only ceiling this doc computes, and the wrong byte count for an 11B vision model at batch 16 — gives 18.1/28.8 = **0.63×** *(derived)*. That is under the floor, which is what a concurrent per-user rate on a smaller model looks like against a batch-1 dense-equivalent denominator. It does not re-rank the row.
> - **Wormhole n150, rank 30, does not fit 35B-A3B.** [[accelerators/tenstorrent-wormhole-n150]] is 12 GB / 288 GB/s, ceiling 14.4 tok/s *(derived)*, $999 → $69.38/tok-s *(derived)*, fits `-/-/-`. The vendor matrix's LLM row is **Mamba-2.8B** at **14.1** tok/s per user at batch 32 ([[benchmarks/ttmetal-n150-mamba-2p8b-b32-tpsu]]; aggregate 451.2). 14.1/14.4 = **0.98×** the 20 GB ceiling *(derived)* — a coincidence of denominators, not evidence the 12 GB card holds a 24 GB model. The speech row on the same card, Whisper distil-large-v3 at **105.0** tok/s batch 1 ([[benchmarks/ttmetal-n150-whisper-distil-large-v3-b1-tpsu]], TTFT 163 ms [[benchmarks/ttmetal-n150-whisper-distil-large-v3-b1-ttft]]), is an encoder-decoder audio model, not a 35B decode, so it is not divided into the ceiling.
> - **Blackhole p150, rank 25, fits 35B/27B and not 70B, and the new row is speech.** [[accelerators/tenstorrent-blackhole-p150]] (32 GB, 512 GB/s, ceiling 25.6 tok/s *(derived)*, $1,399 → $54.65/tok-s *(derived)*). Whisper distil-large-v3 decodes at **263.4** tok/s batch 1 ([[benchmarks/ttmetal-p150-whisper-distil-large-v3-b1-tpsu]], TTFT 63 ms [[benchmarks/ttmetal-p150-whisper-distil-large-v3-b1-ttft]]). The record's own notes flag the Blackhole software as still under active optimization, and a speech model is not the 35B-A3B workload, so 263.4 is not a measured override of the 25.6 tok/s floor and does not enter §4b.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `18b33eb` (184 accelerator records, 218 supply records, 2,869 benchmarks — the Phase-C Oct-16 p16 Hardware Corner long-context cells, #163, on top of `d4b79fb`). The selection rule still yields **138 qualifying / 43 priced / 95 unpriced** (63 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #163 adds no purchase price and no accelerator. What it does add, and what this pass lands, is the rest of the context sweep on one §4 part and the first gpt-oss-20B depth rows on two parts that do not fit 35B-A3B:
>
> - **RTX PRO 5000 Blackwell, still unpriced, still in §4.** [[accelerators/nvidia-rtx-pro-5000-blackwell]] (48 GB, 1,344 GB/s, 35B ceiling 67.2 tok/s *(derived)*). The §4b row already quotes the short-context end of the same lab: **188.8** tok/s at 4k ([[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-4k-tg]], 2.8×) and **148.5** at 16k ([[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-16k-tg]], 2.2×). #163 extends that one Qwen3-30B-A3B series: **123.2** at 32k ([[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-32k]], 123.2/67.2 = **1.8×** *(derived)*), **91.9** at 64k ([[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-64k]], **1.4×**), **61.4** at 128k ([[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-128k]], **0.91×**), **37.0** at 256k ([[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-256k]], **0.55×**). The floor is crossed between 64k and 128k — a context effect, the same shape §4b already flags on the PRO 6000's 256k row, not a reason to reorder the bandwidth ranking. The dense cross-check on the same card: Qwen3-32B decode is **42.0 / 36.4 / 28.9** at 16k/32k/64k ([[benchmarks/hc26-pro5000-qwen3-32b-tg-16k]], [[benchmarks/hc26-pro5000-qwen3-32b-tg-32k]], [[benchmarks/hc26-pro5000-qwen3-32b-tg-64k]]) against the 27B ceiling of 1,344/16 = **84.0** tok/s *(derived)*, so 42.0/84.0 = **0.50×** at 16k *(derived)* — the same half-the-floor the prior refresh already recorded for the Ollama `qwen2.5:32b` rows, and it keeps falling as context grows. The stretch 70B was already on the card at 24.0 (4k) and 22.4 (16k) against a 33.6 tok/s ceiling; #163 does not extend it. It does not enter §3: still no purchase price.
> - **RTX 4080 and RTX 5080, gpt-oss-20B, and neither fits 35B-A3B.** Both are 16 GB, so both are in the 32-row near-miss list, not §3 or §4. [[accelerators/nvidia-rtx-4080]] (716.8 GB/s, 35B ceiling 35.8 tok/s *(derived)*) now has a gpt-oss-20B MXFP4 sweep at **120.2 / 106.0 / 81.3** tok/s for 16k/32k/64k ([[benchmarks/hc26-4080-gptoss-20b-tg-16k]], [[benchmarks/hc26-4080-gptoss-20b-tg-32k]], [[benchmarks/hc26-4080-gptoss-20b-tg-64k]]). Dividing by the 20 GB 35B denominator — the only ceiling this doc computes, and it is the wrong byte count for a 20B MXFP4 — gives 120.2/35.8 = **3.4×** at 16k *(derived)*. [[accelerators/nvidia-rtx-5080]] (960 GB/s, ceiling 48.0) does the same at **161.3 / 149.3 / 130.9** for 16k/32k/64k ([[benchmarks/hc26-5080-gptoss-20b-tg-16k]], [[benchmarks/hc26-5080-gptoss-20b-tg-32k]], [[benchmarks/hc26-5080-gptoss-20b-tg-64k]]), 161.3/48.0 = **3.4×** at 16k *(derived)*, down from the already-recorded 4k cell of 172.4. Neither card holds the 24 GB the 35B-A3B fit flag requires, so these rows calibrate the roofline, they do not add a rank.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `16ea04a` (184 accelerator records, 218 supply records, 2,813 benchmarks — the Phase-C Oct-16 p13 Ollama / Thor backfill, #161, on top of `37c67d2`). The selection rule still yields **138 qualifying / 43 priced / 95 unpriced** (63 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #161 adds no purchase price. What it does add, and what this pass lands, is the first **Qwen3.6-35B-A3B** measurement in the corpus plus dense cross-checks on two parts this doc already lists as unpriced:
>
> - **Jetson AGX Thor, measured, and it does not qualify.** [[accelerators/nvidia-jetson-agx-thor]] is 128 GB unified LPDDR5X at **273 GB/s**. That clears the 24 GB fit bar and misses the 300 GB/s bandwidth floor, and there is no purchase-price supply record at `<= $3,000`, so the selection rule keeps it out of the 138. A third-party SGLang run of **Qwen3.6-35B-A3B FP8** decodes at **14.6** tok/s ([[benchmarks/thor-sglang-qwen36-35b-fp8-decode]], ~36 GB footprint, 24-in/200-out, batch 1). The same arithmetic the ceiling column uses gives 273/20 = **13.7** tok/s *(derived)*, so the FP8 row lands at 14.6/13.7 = **1.07×** that floor *(derived)*. That is not a Q4-MoE comparison — FP8 reads roughly twice the bytes of the 20 GB Q4 the floor assumes — and it is the first time the named 35B-A3B target itself, not the Qwen3-30B-A3B stand-in, shows up against a part this doc would otherwise never mention. It does not enter §3 or §4b: no price, and the part is outside the selection rule. Warm TTFT on the same run is 101 ms ([[benchmarks/thor-sglang-qwen36-35b-fp8-ttft]]); prompt length is unknown, so that number is not a prefill rate.
> - **RTX PRO 4000 / PRO 5000 Blackwell, dense, not A3B.** Still unpriced, still in §4. [[accelerators/nvidia-rtx-pro-4000-blackwell]] (24 GB, 672 GB/s, 35B ceiling 33.6 *(derived)*) has an Ollama `qwen2.5:32b` eval of **24.6** tok/s ([[benchmarks/cfgk-pro4000-ollama-qwen25-32b-eval]]). The 27B-dense ceiling on that card is 672/16 = **42.0** tok/s *(derived)*, so 24.6/42.0 = **0.59×** *(derived)* — under the floor, which is what a ~19.4 GB 32B resident looks like against a 16 GB denominator, and the page's own quant is unnamed. [[accelerators/nvidia-rtx-pro-5000-blackwell]] (48 GB) gets the same tag at **42.9** / **42.92** tok/s from two labs ([[benchmarks/cfgk-pro5000-ollama-qwen25-32b-eval]], [[benchmarks/dm-pro5000-ollama-qwen25-32b-eval]]) against a 27B ceiling of 1,344/16 = **84.0** *(derived)* → 0.51×. The stretch 70B is now measured on that card too: Ollama `llama3.3:70b` at **21.42** tok/s ([[benchmarks/dm-pro5000-ollama-llama33-70b-eval]], 42 GB resident) and `qwen2.5:72b` at **18.4** ([[benchmarks/cfgk-pro5000-ollama-qwen25-72b-eval]]), against a 70B ceiling of 1,344/40 = **33.6** *(derived)* → 0.64× and 0.55×. The 72 GB sibling still has no public row. None of these move §3.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `37c67d2` (184 accelerator records, 218 supply records, 2,801 benchmarks — the llama.cpp Llama-2-7B scoreboard backfill, #159, on top of `b7921d5`). The selection rule still yields **138 qualifying / 43 priced / 95 unpriced** (63 of the unpriced fit 35B-A3B, 32 do not). The ranked $/tok-s table is unchanged: every priced cell was re-diffed against the generator and matches (price, ceiling, fit flags, supply id, order). #159 adds no purchase price and no A3B-class measurement — its rows are Llama-2-7B Q4_0 — so it cannot move this ranking. What this pass does land is one measured A3B row the previous table never cited:
>
> - **RTX PRO 5000 Blackwell, measured.** [[accelerators/nvidia-rtx-pro-5000-blackwell]] (48 GB, 1,344 GB/s, §4, unpriced) has a third-party Qwen3-30B-A3B Q4_K decode of **188.8** tok/s at 4k ([[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-4k-tg]]) against a 67.2 tok/s ceiling *(derived)* — **2.8×** the floor (188.8/67.2 = 2.81, *(derived)*). The same card at 16k is 148.5 ([[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-16k-tg]]), 2.2×. That is the first measured A3B number for a part this doc lists only as "fits, can't price it," and it sits between the RTX 5090 (234.3) and the RTX PRO 6000 Blackwell (198.7) on the same lab's 4k protocol. The 72 GB sibling still has no public row. It does not enter §3: there is still no bare-card price.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16, prior pass).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `b7921d5` (184 accelerator records, 218 supply records, 2,742 benchmarks). The selection rule now yields **138 qualifying / 43 priced / 95 unpriced** (63 of the unpriced fit 35B-A3B, 32 do not). Two things actually moved; the rest of the ranked order did not:
>
> - **One new priced row.** [[accelerators/amd-epyc-9554-genoa]] (vram_gb null, unified DDR5 host) is joined to the whole 384 GB self-build [[supply/used-epyc-9004-genoa-build-384gb]] at $7,500. The 384 GB figure is the configured pool of that supply record (`UNIFIED_MEM_GB` in the tool), so the `Y/Y/Y` flag and the sticker describe the same machine. Ceiling 23.0 tok/s *(derived)* → **$325.52/tok-s *(derived)***, rank 42, one slot above the H100 PCIe. The $2,000 CPU-only row is not joined (it prices no RAM).
> - **Six fit-but-unpriced rows the previous table dropped.** [[accelerators/nvidia-rtx-pro-5000-72gb-blackwell]], [[accelerators/nvidia-rtx-pro-5000-blackwell]], [[accelerators/nvidia-l40]], [[accelerators/nvidia-rtx-a5000]], [[accelerators/nvidia-rtx-pro-4000-blackwell]], [[accelerators/amd-radeon-pro-v620]]. They were already in `data/accelerators/`; §4 now lists them. The near-miss (does-not-fit) count went from 19 to 32 for the same reason — those ids are named in the §4 close, not given their own table.
> - **A stale generator join was removed before this table was written.** The tool briefly joined [[supply/used-mac-studio-m1-ultra-64gb]] ($1,850) and [[supply/used-mac-studio-m3-ultra-256gb]] ($6,250) onto records whose `vram_gb` is 128 and 512. That would have ranked M1 Ultra at $46.25/tok-s and M3 Ultra at $152.63/tok-s for fit flags the sticker does not buy. Both stay dropped; the table keeps the capacity-matched rows ($2,500 / 128 GB and $9,000 / 512 GB).
>
> Ranked order of the other 42 priced rows is unchanged from the note below, including the RTX 5070 Ti measured cell at **111.11** tok/s.
>
> **Refresh note (docs/fit-matrix-oct14, Oct-16).** Re-ran `python tools/fit_matrix_35b.py` on local `main` `069e36f` (162 accelerator records, 214 supply records). Selection rule still yields **118 qualifying / 42 priced / 76 unpriced** (57 of the unpriced fit 35B-A3B, 19 do not). Ranked $/tok-s order is unchanged from the p6 table already on this tree. The only numeric edit is §4b: the RTX 5070 Ti measured decode cell is restored to the benchmark record's own value, **111.11** tok/s ([[benchmarks/hwcorner-rtx-5070-ti-qwen3-30b-a3b-q4kxl-decode-16k]]), not the rounded 111.1 the p6 prose left behind. `2.5×` vs the 44.8 tok/s ceiling is unchanged (111.11/44.8 = 2.48, still 2.5× *(derived)*).
>
> **Refresh note (docs/fit-matrix-oct16-p6, Oct-16).** Re-run of
> `python tools/fit_matrix_35b.py` against local `main` `e128197` (162
> accelerator records, 214 supply records, 1,595 benchmarks — the Phase-C
> Oct-16 p7 benchmark backfill on top of `origin/main`). The selection rule
> yields **118 qualifying accelerators**, of which **42 have a usable
> purchase price** and 76 are bandwidth-qualified but unpriced (57 of those
> fit 35B-A3B, 19 do not). The ranked $/tok-s table is unchanged from the
> Oct-16 `docs/fit-matrix-oct14` refresh; what this pass does is land that
> table on the tree that actually contains the Atlas join and the four §4b
> rows, and quote two measured values at the record's own precision. Three
> things moved since the Oct-15 pass-3 doc:
>
> - **One new priced row.** [[accelerators/huawei-atlas-300i-pro]] (24 GB,
>   204.8 GB/s) is the card-level record of the same 310P-class silicon as
>   [[accelerators/huawei-ascend-310p]]. Joined at $265 from
>   [[supply/taobao-used-huawei-atlas-300i-pro-24gb]] — identical sticker and
>   identical derived ceiling, so the two tie at ranks 13/14.
> - **Two supply prices were re-read at their recorded low end**, which
>   reorders the top of §3. [[supply/ebay-used-amd-mi100-32gb]] is $720 (the
>   record's buy-target; the prior doc used the $900 top of the same band),
>   so [[accelerators/amd-instinct-mi100]] moves from rank 5 at $14.66/tok-s
>   to **rank 4 at $11.73/tok-s *(derived)***.
>   [[supply/used-mac-studio-m1-ultra-128gb]] is $2,500 (midpoint of the
>   $2,200–3,000 band), so [[accelerators/apple-m1-ultra]] moves from $65.00
>   to **$62.50/tok-s *(derived)***.
> - **Capacity-mismatched Apple joins dropped.** The prior doc priced
>   [[accelerators/apple-m3-ultra]] (record `vram_gb` 512) off the 96 GB
>   [[supply/used-mac-studio-m3-ultra-96gb]] at $3,700. The generator now
>   joins the 512 GB machine ([[supply/used-mac-studio-m3-ultra-512gb]],
>   $9,000), which is the memory the fit flag is computed against. That
>   moves M3 Ultra from rank 32 ($90.35) to rank 41 (**$219.78/tok-s
>   *(derived)***).
>
> §4 grew from 52 to 57 fit-but-unpriced rows (DGX-1, RTX A6000, A40, RTX
> 5000 Ada, M1 Max, M2 Max). §4b adds four measured A3B decode rows that
> landed after the last refresh (RTX PRO 6000 Blackwell, A100 40 GB SXM4,
> A100 40 GB PCIe, RTX A6000).

> *Prior refresh (docs/fit-matrix-oct14, third refresh):* re-run on the
> Phase-C Oct-15 pass-3 tree. Ranked table then 41 priced rows; §4 at 52.

**Scope, set by the commissioning brief.** The target class is a **35B-A3B
Mixture-of-Experts** model (~3B active parameters at decode), a **~27B dense**
model, and **Llama-3.3-70B at Q4** as the stretch goal. The question is narrow:
*does it fit, and what is the cheapest $/(tok/s) for the 35B-A3B job.* This doc
does not try to rank serving quality, concurrency, or prefill — those need the
measured-benchmark join in `docs/08-cost-per-token.md`, not a roofline.

**Every number in the tables is either copied verbatim from a record or marked
`(derived)`.** Derived means the arithmetic is shown below and a reader can
re-run it. Nothing here is a measurement.

## 1. The selection rule (which records got a row)

*This build is a re-run of `tools/fit_matrix_35b.py` on `main` `bd282fe`
(200 accelerators, 218 supply, 2,995 benchmarks). The selection rule is
unchanged. The counts moved: **148 qualifying / 43 priced / 105 unpriced**
(68 of the unpriced fit 35B-A3B, 37 do not). The one new row is unpriced.
See the refresh note above.*

An accelerator record gets a row when **all** of these hold:

- `vram_gb >= 8` — below this there is no useful quantised-7B headroom either.
- `memory_bandwidth_gbps >= 300` **OR** it has a linked `supply` record with a
  purchase price `<= $3,000`.

The `<= $3,000` arm exists so that a cheap-but-slow card (the used-Tesla slice)
is not filtered out by the bandwidth floor before its price can speak. That rule
yields **148 qualifying accelerators**, of which **43 have a usable purchase
price** and 99 are bandwidth-qualified but have no bare-card/system price in
`data/supply/` (67 of those fit 35B-A3B; 32 qualify and do not).

**One exception to the `vram_gb` gate.** A `unified_memory` part (an APU/SoC
whose GPU shares the CPU's memory pool) leaves `vram_gb` null *on purpose* —
there is no dedicated VRAM to report. The `>= 8` gate would silently drop it.
For those, `tools/fit_matrix_35b.py` substitutes the **configured memory of the
cheapest linked supply record** (a whole system, so its RAM is the usable pool).
Two records need the override: [[accelerators/amd-radeon-8060s-ryzen-ai-max]]
(Strix Halo, 128 GB configured) and [[accelerators/amd-epyc-9554-genoa]] (the
384 GB DDR5 pool of [[supply/used-epyc-9004-genoa-build-384gb]]). The Apple
Ultra parts already carry an explicit `vram_gb`.

## 2. The three fit columns and the one ceiling

Model footprints used for the fit flags (these are the brief's numbers, kept as
stated):

| workload | weights | + KV / runtime headroom | needs `vram_gb >=` |
|---|---|---|---|
| **35B-A3B MoE, Q4** | ~20 GB | 4 GB | **24 GB** |
| **27B dense, Q4** | ~16 GB | 4 GB | **20 GB** |
| **70B dense, Q4** | ~40 GB | 8 GB | **48 GB** |

`fit = vram_gb >= (weights + headroom)`, read off each accelerator record's own
`vram_gb`. A `Y` means *resident on a single device* — no multi-GPU span, no
offload. The 35B-A3B threshold of 24 GB is why every 24 GB card in the table is
a `Y` by exactly zero margin.

**The ceiling.** `tok/s ceiling (derived) = memory_bandwidth_gbps /
model_weights_gb`. This is the standard decode roofline: at batch-1 each output
token must read the resident weights once, so a perfectly bandwidth-bound decode
cannot exceed bandwidth / bytes-read. It is a **ceiling, not a forecast** — real
decode lands well under it because of kernel inefficiency, KV reads, and the
fact that a MoE reads only its active experts. It is also *optimistic for MoE*:
a 35B-A3B that activates ~3B/token reads closer to the active-expert bytes than
the full 20 GB resident, so the same ceiling formula on a dense-equivalent basis
overstates the *value* of a big resident pool for MoE. The doc divides by the
full resident weights (20 GB) anyway, which makes the ceiling a **conservative
lower bound on the achievable rate** and keeps the arithmetic identical across
the three columns.

*How* conservative shows up the moment a low-bandwidth part enters the table:
[[accelerators/amd-radeon-8060s-ryzen-ai-max]] (Strix Halo, 256 GB/s shared)
derives to a 12.8 tok/s ceiling on the 20 GB column, but the corpus's own
measurement of the **exact target model** —
[[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]] — records **42.04
tok/s** on it, ~3.3× the derived ceiling, because the MoE reads only ~3B active
experts per token, not the 20 GB resident. The roofline ranks capex against a
*dense-equivalent* number; it understates real MoE throughput most where
bandwidth is scarcest. Treat every ceiling cell as the floor, not the forecast.

**$/tok-s (derived) = cheapest purchase price ÷ the 35B-A3B ceiling.** It ranks
"how little silicon do I buy to move 20 GB of weights." It is a *capex-efficiency*
figure, not a cost-per-token-served figure — it ignores power, the chassis/baseboard
several of these parts need (§5), and that $/tok-s for a 14 tok/s card and a 240
tok/s card are not the same product.

**A third calibration sweep landed with the Phase-C pass-5 ingest.** The
hardware-corner 16k-context sweep now holds **five** third-party measured decode
rows on **Qwen3-30B-A3B (MoE) Q4_K XL** — the closest corpus proxy to the
35B-A3B target — at a ~16,384-token context, batch-1 per-request:

|| part | measured tok/s | derived ceiling (BW ÷ 20 GB) | measured ÷ ceiling |
||---|---|---|---|
|| [[accelerators/nvidia-rtx-3090-ti]] | **98.25** ([[benchmarks/hwcorner-rtx-3090-ti-qwen3-30b-a3b-q4kxl-decode-16k]]) | 50.4 | 1.95× |
|| [[accelerators/nvidia-rtx-4090]] | **109.57** ([[benchmarks/hwcorner-rtx-4090-qwen3-30b-a3b-q4kxl-decode-16k]]) | 50.4 | 2.17× |
|| [[accelerators/nvidia-rtx-6000-ada]] | **104.88** ([[benchmarks/hwcorner-rtx-6000-ada-qwen3-30b-a3b-q4kxl-decode-16k]]) | 48.0 | 2.19× |
|| [[accelerators/nvidia-rtx-4080]] | **85.04** ([[benchmarks/hwcorner-rtx-4080-qwen3-30b-a3b-q4kxl-decode-16k]]) | 35.8 | 2.37× |
|| [[accelerators/nvidia-rtx-5070-ti]] | **111.11** ([[benchmarks/hwcorner-rtx-5070-ti-qwen3-30b-a3b-q4kxl-decode-16k]]) | 44.8 | 2.48× |

Two of the five (3090 Ti, 4090) are also *priced* rows in §3 — the ranking on
them can now be sanity-checked against a measured number: both beat their dense
roofline by ~2× at 16k context, corroborating the direction the Strix Halo
anchors set (MoE reads ~3B active bytes, not the resident 20 GB). The ordering
is *not* pure bandwidth — the 5070 Ti (896 GB/s) edges the 4090 (1,008 GB/s) at
16k, so generation/architecture effects are real at long context — but every
measured value sits *above* its derived ceiling, i.e. the ceilings in the table
remain floors, and the $/tok-s ranking *undervalues* nothing below a measured
anchor. Caveat: the hwcorner sweep measures 30B-A3B, not the 35B-A3B target —
the active-parameter ratio is the same shape (~3B active of ~30B vs ~3B of ~35B
resident), so the direction transfers, not the exact figure.

## 3. Ranked: purchase-priced parts by $/tok-s for the 35B-A3B job

Sorted by derived $/tok-s, lowest first. `Fits` columns are 35B/27B/70B.
Only rows with a usable purchase price appear; the `← supply` column is the
record the price was read from. A `Y` on Fits is bolded. Every ceiling and
every $/tok-s cell is *(derived)*: `ceiling = memory_bandwidth_gbps / 20` and
`$/tok-s = price / ceiling`.

| rank | accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* | price | ← supply | $/tok-s *(derived)* |
|---|---|---|---|---|---|---|---|---|
| 1 | [[accelerators/nvidia-tesla-p100]] | 16 | 732 | -/-/- | 36.6 | $115 | [[supply/ebay-used-tesla-p100-16gb]] | **3.14** |
| 2 | [[accelerators/intel-arc-b580]] | 12 | 456 | -/-/- | 22.8 | $199.99 | [[supply/newegg-refreshed-arc-b580]] | 8.77 |
| 3 | [[accelerators/nvidia-v100-sxm2-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $500 | [[supply/ebay-used-tesla-v100-sxm2-32gb]] | 11.11 |
| 4 | [[accelerators/amd-instinct-mi100]] | 32 | 1228 | **Y/Y/-** | 61.4 | $720 | [[supply/ebay-used-amd-mi100-32gb]] | 11.73 |
| 5 | [[accelerators/nvidia-tesla-m40]] | 24 | 288 | **Y/Y/-** | 14.4 | $189 | [[supply/used-nvidia-tesla-m40-market]] | 13.12 |
| 6 | [[accelerators/intel-gaudi2]] | 96 | 2460 | **Y/Y/Y** | 123.0 | $2,000 | [[supply/ebay-used-gaudi2-hl225-oam-module]] | 16.26 |
| 7 | [[accelerators/nvidia-v100-pcie-32gb]] | 32 | 900 | **Y/Y/-** | 45.0 | $750 | [[supply/ebay-used-tesla-v100-pcie-32gb]] | 16.67 |
| 8 | [[accelerators/nvidia-tesla-p40]] | 24 | 347 | **Y/Y/-** | 17.4 | $290 | [[supply/ebay-used-tesla-p40-24gb]] | 16.71 |
| 9 | [[accelerators/amd-radeon-rx-7900-xtx]] | 24 | 960 | **Y/Y/-** | 48.0 | $875 | [[supply/ebay-used-rx-7900-xtx-market-oct2026]] | 18.23 |
| 10 | [[accelerators/nvidia-rtx-3090-ti]] | 24 | 1008 | **Y/Y/-** | 50.4 | $950 | [[supply/used-rtx-3090-ti-market-oct2026]] | 18.85 |
| 11 | [[accelerators/nvidia-rtx-3090]] | 24 | 936 | **Y/Y/-** | 46.8 | $950 | [[supply/ebay-used-rtx3090-oct2026]] | 20.30 |
| 12 | [[accelerators/nvidia-rtx-5090]] | 32 | 1792 | **Y/Y/-** | 89.6 | $1,999 | [[supply/nvidia-marketplace-geforce]] | 22.31 |
| 13 | [[accelerators/huawei-ascend-310p]] | 24 | 204.8 | **Y/Y/-** | 10.2 | $265 | [[supply/taobao-used-huawei-ascend-310p]] | 25.88 |
| 14 | [[accelerators/huawei-atlas-300i-pro]] | 24 | 204.8 | **Y/Y/-** | 10.2 | $265 | [[supply/taobao-used-huawei-atlas-300i-pro-24gb]] | 25.88 |
| 15 | [[accelerators/amd-radeon-rx-9070-xt]] | 16 | 640 | -/-/- | 32.0 | $829 | [[supply/used-rx-9070-xt-market]] | 25.91 |
| 16 | [[accelerators/intel-arc-pro-b60]] | 24 | 456 | **Y/Y/-** | 22.8 | $650 | [[supply/new-arc-pro-b60-24gb-retail-oct2026]] | 28.51 |
| 17 | [[accelerators/cambricon-mlu370]] | 48 | 307.2 | **Y/Y/Y** | 15.4 | $450 | [[supply/ebay-used-cambricon-mlu370]] | 29.30 |
| 18 | [[accelerators/amd-instinct-mi250]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $4,950 | [[supply/ebay-used-amd-mi250x-128gb-oam]] | 30.94 |
| 19 | [[accelerators/amd-instinct-mi250x]] | 128 | 3200 | **Y/Y/Y** | 160.0 | $4,950 | [[supply/ebay-used-amd-mi250x-128gb-oam]] | 30.94 |
| 20 | [[accelerators/intel-arc-pro-b70]] | 32 | 608 | **Y/Y/-** | 30.4 | $949 | [[supply/newegg-new-arc-pro-b70-32gb]] | 31.22 |
| 21 | [[accelerators/nvidia-a100-40gb-sxm4]] | 40 | 1555 | **Y/Y/-** | 77.8 | $2,499 | [[supply/ebay-used-a100-40gb-sxm4]] | 32.14 |
| 22 | [[accelerators/nvidia-rtx-4090]] | 24 | 1008 | **Y/Y/-** | 50.4 | $2,000 | [[supply/ebay-used-consumer-gpu-market]] | 39.68 |
| 23 | [[accelerators/amd-instinct-mi210]] | 64 | 1600 | **Y/Y/Y** | 80.0 | $4,000 | [[supply/ebay-used-amd-mi210-64gb]] | 50.00 |
| 24 | [[accelerators/tenstorrent-wormhole-n300]] | 24 | 576 | **Y/Y/-** | 28.8 | $1,449 | [[supply/new-tenstorrent-wormhole-n300]] | 50.31 |
| 25 | [[accelerators/tenstorrent-blackhole-p150]] | 32 | 512 | **Y/Y/-** | 25.6 | $1,399 | [[supply/new-tenstorrent-blackhole-p150]] | 54.65 |
| 26 | [[accelerators/qualcomm-ai100]] | 32 | 137 | **Y/Y/-** | 6.8 | $400 | [[supply/used-qualcomm-cloud-ai100-pcie]] | 58.39 |
| 27 | [[accelerators/cambricon-mlu270]] | 16 | 102.4 | -/-/- | 5.1 | $317 | [[supply/ebay-used-cambricon-mlu270-s4]] | 61.91 |
| 28 | [[accelerators/apple-m1-ultra]] | 128 | 800 | **Y/Y/Y** | 40.0 | $2,500 | [[supply/used-mac-studio-m1-ultra-128gb]] | 62.50 |
| 29 | [[accelerators/tenstorrent-grayskull-e150]] | 8 | 118 | -/-/- | 5.9 | $400 | [[supply/used-tenstorrent-grayskull-e150]] | 67.80 |
| 30 | [[accelerators/tenstorrent-wormhole-n150]] | 12 | 288 | -/-/- | 14.4 | $999 | [[supply/new-tenstorrent-wormhole-n150]] | 69.38 |
| 31 | [[accelerators/nvidia-h100-sxm]] | 80 | 3350 | **Y/Y/Y** | 167.5 | $12,999 | [[supply/ebay-used-h100-sxm5-module]] | 77.61 |
| 32 | [[accelerators/nvidia-a100-40gb-pcie]] | 40 | 1555 | **Y/Y/-** | 77.8 | $7,000 | [[supply/ebay-used-a100-40gb-pcie-card]] | 90.03 |
| 33 | [[accelerators/apple-m2-ultra]] | 192 | 800 | **Y/Y/Y** | 40.0 | $3,700 | [[supply/used-mac-studio-m2-ultra-192gb]] | 92.50 |
| 34 | [[accelerators/nvidia-rtx-pro-6000-blackwell]] | 96 | 1792 | **Y/Y/Y** | 89.6 | $8,650 | [[supply/new-rtx-pro-6000-blackwell-96gb]] | 96.54 |
| 35 | [[accelerators/nvidia-a100-80gb-sxm4]] | 80 | 2039 | **Y/Y/Y** | 102.0 | $10,449.95 | [[supply/ebay-used-accelerator-market]] | 102.50 |
| 36 | [[accelerators/nvidia-a100-80gb-pcie]] | 80 | 1935 | **Y/Y/Y** | 96.8 | $12,099 | [[supply/ebay-used-a100-80gb-pcie]] | 125.05 |
| 37 | [[accelerators/amd-radeon-8060s-ryzen-ai-max]] | 128* | 256 | **Y/Y/Y** | 12.8 | $1,900 | [[supply/new-minisforum-ms-s1-max-128gb]] | 148.44 |
| 38 | [[accelerators/amd-ryzen-ai-max-395-framework-desktop]] | 128 | 256 | **Y/Y/Y** | 12.8 | $2,099 | [[supply/new-beelink-gtr9-pro-strix-halo-128gb-2]] | 163.98 |
| 39 | [[accelerators/apple-m3-max]] | 128 | 400 | **Y/Y/Y** | 20.0 | $3,400 | [[supply/used-macbook-pro-m3-max-128gb]] | 170.00 |
| 40 | [[accelerators/apple-m4-max]] | 128 | 546 | **Y/Y/Y** | 27.3 | $4,700 | [[supply/used-macbook-pro-m4-max-128gb]] | 172.16 |
| 41 | [[accelerators/apple-m3-ultra]] | 512 | 819 | **Y/Y/Y** | 41.0 | $9,000 | [[supply/used-mac-studio-m3-ultra-512gb]] | 219.78 |
| 42 | [[accelerators/amd-epyc-9554-genoa]] | 384* | 460.8 | **Y/Y/Y** | 23.0 | $7,500 | [[supply/used-epyc-9004-genoa-build-384gb]] | 325.52 |
| 43 | [[accelerators/nvidia-h100-pcie]] | 80 | 2000 | **Y/Y/Y** | 100.0 | $43,000 | [[supply/used-h100-80gb-pcie-market]] | 430.00 |

\* Two rows are `unified_memory` and the capacity shown is not the
record's `vram_gb` (that field is null). The 8060s Strix Halo 128 GB is the
configured LPDDR5x pool of the cheapest linked supply record
([[supply/new-minisforum-ms-s1-max-128gb]]). The EPYC 9554 384 GB is the
configured DDR5 pool of [[supply/used-epyc-9004-genoa-build-384gb]] — a whole
self-build, not a bare CPU. The sibling
`amd-ryzen-ai-max-395-framework-desktop` record carries an explicit
`vram_gb=128` for the same Strix Halo silicon, priced at the Beelink
whole-system row. See §1 and §6.

### The headline read

- **The cheapest *fits* answer is the used V100 SXM2 — and MI100 just passed
  the M40.** [[accelerators/nvidia-tesla-p100]] at **$115** and
  [[accelerators/intel-arc-b580]] at **$199.99** still rank #1/#2 on $/tok-s
  and still fit nothing: 16 GB / 12 GB are both `-/-/-`. They are in the table
  because a sub-$3,000 price is allowed to speak. The first row that fits
  **35B-A3B** is [[accelerators/nvidia-v100-sxm2-32gb]] at **$500** (rank 3,
  45.0 tok/s ceiling *(derived)*). Re-reading
  [[supply/ebay-used-amd-mi100-32gb]] at its recorded $720 low end (was $900)
  moves [[accelerators/amd-instinct-mi100]] to **rank 4 at $11.73/tok-s
  *(derived)*** — ahead of the $189 M40, and still the cheapest part with both
  a 35B `Y` and >1 TB/s (61.4 tok/s ceiling *(derived)*, faster than a 3090
  for less money). All four are traps unless you read §5.
- **The honest mainstream answer is still the RTX 3090, matched by its Ti.**
  Both sit at $950 used ([[supply/ebay-used-rtx3090-oct2026]],
  [[supply/used-rtx-3090-ti-market-oct2026]]) for 24 GB. The Ti's extra 72 GB/s
  buys a 50.4 vs 46.8 tok/s ceiling *(derived)* at the same price, so it edges
  the 3090 (ranks 10 and 11). Both are real PCIe cards with a fan. The
  [[accelerators/amd-radeon-rx-7900-xtx]] still undercuts both ($875 used,
  rank 9) but needs the ROCm path.
- **New priced row: the Atlas 300I Pro ties the Ascend 310P.**
  [[accelerators/huawei-atlas-300i-pro]] and [[accelerators/huawei-ascend-310p]]
  are die-vs-card records of the same 24 GB / 204.8 GB/s part, both at $265,
  both a 10.2 tok/s ceiling *(derived)* and $25.88/tok-s *(derived)*. A `Y` on
  35B here means the bytes fit, not that it serves — see §5.
- **The stretch goal (70B, needs ≥48 GB) did not get a cheaper *usable*
  answer this pass.** Cheapest single-accelerator 70B sticker is still the
  MLU370 at $450 (307 GB/s — a capacity fit, not a usable one), then Gaudi 2
  at $2,000 (OAM, §5) and the MI250/X pair at $4,950 (same OAM caveat). The
  cheapest *drop-in PCIe* 70B card with >1 TB/s remains
  [[accelerators/nvidia-rtx-pro-6000-blackwell]] at $8,650 (rank 34, 89.6
  tok/s ceiling *(derived)*). [[accelerators/apple-m3-ultra]] stays on the
  512 GB Studio ([[supply/used-mac-studio-m3-ultra-512gb]], $9,000, rank 41,
  $219.78/tok-s *(derived)*) — the memory the `Y/Y/Y` flag is computed
  against. The new row is a host, not a GPU:
  [[accelerators/amd-epyc-9554-genoa]] at $7,500 for a 384 GB DDR5 self-build
  (rank 42, 23.0 tok/s ceiling *(derived)*, $325.52/tok-s *(derived)*). It
  fits 70B because the pool is 384 GB, and it is slow because 12-channel
  DDR5 is 460.8 GB/s. Strix Halo at $1,900 still fits 70B on paper and is
  still bandwidth-capped (§5).

## 4. Qualified but no purchase price — the "fits, can't price it" list

These records satisfy the bandwidth/fit rule and **fit 35B-A3B** but have
**no bare-card or system purchase price** in `data/supply/` (they may have a
*rental* price in the cloud records — that is a different join, see
`tools/cost_per_token.py`). Listed so a future supply pass can close them;
sorted by the 35B ceiling *(derived)*. 68 rows. This pass adds
[[accelerators/furiosaai-rngd-renegade]] (48 GB, 1,500 GB/s, ceiling 75.0
*(derived)*, **Y/Y/Y**, no purchase price — see the refresh note). The prior
pass added the four
#167 records: [[accelerators/nvidia-rtx-4090-d]] (1,008 GB/s, tied on ceiling
with the priced 4090), [[accelerators/nvidia-rtx-5090-laptop]] (896 GB/s,
tied with the PRO 4500 Blackwell), [[accelerators/nvidia-quadro-rtx-8000]]
(48 GB, fits 70B) and [[accelerators/nvidia-quadro-rtx-6000]] (24 GB, does
not) — both 672 GB/s, tied with the PRO 4000 Blackwell.

| accelerator | VRAM | BW GB/s | Fits | 35B ceil *(derived)* | note |
|---|---|---|---|---|---|
| [[accelerators/huawei-ascend-910c-cloudmatrix-384]] | 49,152 | 1,228,800 | **Y/Y/Y** | 61440.0 | 384-socket supernode; whole-rack figures, no unit price |
| [[accelerators/amd-instinct-mi350x]] | 288 | 8,000 | **Y/Y/Y** | 400.0 | integrator/OEM channel only ([[supply/amd-instinct-oem-systems]]) |
| [[accelerators/amd-instinct-mi355x]] | 288 | 8,000 | **Y/Y/Y** | 400.0 | integrator + cloud-hour only |
| [[accelerators/nvidia-gb200-nvl72]] | 186 | 8,000 | **Y/Y/Y** | 400.0 | rack unit, not a bare card |
| [[accelerators/nvidia-gb300]] | 279 | 8,000 | **Y/Y/Y** | 400.0 | rack unit, not a bare card |
| [[accelerators/nvidia-b200]] | 180 | 7,700 | **Y/Y/Y** | 385.0 | enterprise/cloud channel only |
| [[accelerators/nvidia-b300]] | 270 | 7,700 | **Y/Y/Y** | 385.0 | enterprise/cloud channel only |
| [[accelerators/google-tpu-ironwood]] | 192 | 7,380 | **Y/Y/Y** | 369.0 | cloud only |
| [[accelerators/amd-instinct-mi325x]] | 256 | 6,000 | **Y/Y/Y** | 300.0 | integrator/OEM only |
| [[accelerators/amd-instinct-mi300a]] | 128 | 5,300 | **Y/Y/Y** | 265.0 | APU variant, integrator only |
| [[accelerators/amd-instinct-mi300x]] | 192 | 5,300 | **Y/Y/Y** | 265.0 | integrator + cloud-hour ([[supply/w5s-oracle-oci-amd-mi300x-mi355x-gpu-hour]]) |
| [[accelerators/aws-trainium3]] | 144 | 4,900 | **Y/Y/Y** | 245.0 | AWS only |
| [[accelerators/nvidia-h200-nvl]] | 141 | 4,800 | **Y/Y/Y** | 240.0 | integrator only |
| [[accelerators/nvidia-h200-sxm]] | 141 | 4,800 | **Y/Y/Y** | 240.0 | cloud/rental only |
| [[accelerators/rebellions-rebel100]] | 144 | 4,800 | **Y/Y/Y** | 240.0 | no price record |
| [[accelerators/amd-instinct-mi350p]] | 144 | 4,000 | **Y/Y/Y** | 200.0 | integrator only |
| [[accelerators/nvidia-gh200]] | 96 | 4,000 | **Y/Y/Y** | 200.0 | unified-memory Grace+Hopper superchip; integrator/cloud only |
| [[accelerators/nvidia-h20]] | 96 | 4,000 | **Y/Y/Y** | 200.0 | export part; the 37-ridge decode specialist of [[09-model-memory-envelope]] |
| [[accelerators/nvidia-h100-nvl]] | 94 | 3,938 | **Y/Y/Y** | 196.9 | cloud/integrator only |
| [[accelerators/intel-gaudi3]] | 128 | 3,700 | **Y/Y/Y** | 185.0 | integrator only |
| [[accelerators/xpu-intel-dc-gpu-max-1550]] | 128 | 3,276 | **Y/Y/Y** | 163.8 | no price |
| [[accelerators/huawei-ascend-910c]] | 128 | 3,200 | **Y/Y/Y** | 160.0 | no price record |
| [[accelerators/aws-trainium2]] | 96 | 2,875 | **Y/Y/Y** | 143.8 | AWS only |
| [[accelerators/google-tpu-v5p]] | 95 | 2,765 | **Y/Y/Y** | 138.2 | cloud only |
| [[accelerators/cambricon-mlu590]] | 192 | 2,400 | **Y/Y/Y** | 120.0 | no price record |
| [[accelerators/google-tpu-v6e]] | 32 | 1,638 | **Y/Y/-** | 81.9 | cloud only |
| [[accelerators/biren-br100]] | 64 | 1,600 | **Y/Y/Y** | 80.0 | no price record |
| [[accelerators/furiosaai-rngd-renegade]] | 48 | 1,500 | **Y/Y/Y** | 75.0 | eval-only, no retail price ([[supply/furiosaai-rngd-eval-only-channel]] is `price_usd: null` and still points at the Warboy); bandwidth is a vendor claim |
| [[accelerators/huawei-ascend-910b]] | 64 | 1,400 | **Y/Y/Y** | 70.0 | no price record |
| [[accelerators/nvidia-rtx-pro-5000-72gb-blackwell]] | 72 | 1,344 | **Y/Y/Y** | 67.2 | no bare-card price in supply/ |
| [[accelerators/nvidia-rtx-pro-5000-blackwell]] | 48 | 1,344 | **Y/Y/Y** | 67.2 | no bare-card price in supply/ |
| [[accelerators/google-tpu-v4]] | 32 | 1,200 | **Y/Y/-** | 60.0 | cloud only |
| [[accelerators/local-apple-m5-ultra]] | 512 | 1,200 | **Y/Y/Y** | 60.0 | no bare-part price |
| [[accelerators/amd-radeon-instinct-mi50x]] | 32 | 1,024 | **Y/Y/-** | 51.2 | no price record |
| [[accelerators/amd-radeon-instinct-mi60]] | 32 | 1,024 | **Y/Y/-** | 51.2 | no price record |
| [[accelerators/nvidia-rtx-4090-d]] | 24 | 1,008 | **Y/Y/-** | 50.4 | China-only 4090 cut; same 1,008 GB/s, no bare-card price |
| [[accelerators/nvidia-rtx-6000-ada]] | 48 | 960 | **Y/Y/Y** | 48.0 | cloud/rental only; 48 GB fit, thin BW |
| [[accelerators/nvidia-a30]] | 24 | 933 | **Y/Y/-** | 46.6 | enterprise-distributor only |
| [[accelerators/google-tpu-v3]] | 32 | 900 | **Y/Y/-** | 45.0 | cloud only |
| [[accelerators/nvidia-dgx-1-8x-v100]] | 256 | 900 | **Y/Y/Y** | 45.0 | 8-GPU server; the $4,000 chassis rows are system prices, not joined (see §6) |
| [[accelerators/nvidia-rtx-5090-laptop]] | 24 | 896 | **Y/Y/-** | 44.8 | mobile GB203, not the desktop 5090; no bare-GPU price |
| [[accelerators/nvidia-rtx-pro-4500-blackwell]] | 32 | 896 | **Y/Y/-** | 44.8 | no price record |
| [[accelerators/amd-radeon-pro-w7900]] | 48 | 864 | **Y/Y/Y** | 43.2 | no price record |
| [[accelerators/nvidia-l40]] | 48 | 864 | **Y/Y/Y** | 43.2 | no bare-card price in supply/; Ada sibling of the L40S |
| [[accelerators/nvidia-l40s]] | 48 | 864 | **Y/Y/Y** | 43.2 | cloud/rental only; second-highest ridge in the repo per docs/09 |
| [[accelerators/aws-inferentia2]] | 32 | 820 | **Y/Y/-** | 41.0 | AWS only |
| [[accelerators/apple-m4-ultra]] | 192 | 819 | **Y/Y/Y** | 41.0 | no bare-part price; whole-Mac-Studio silicon |
| [[accelerators/aws-trainium1]] | 32 | 800 | **Y/Y/-** | 40.0 | AWS only |
| [[accelerators/moore-threads-s4000]] | 48 | 768 | **Y/Y/Y** | 38.4 | no price record |
| [[accelerators/nvidia-rtx-a5000]] | 24 | 768 | **Y/Y/-** | 38.4 | no bare-card price in supply/ |
| [[accelerators/nvidia-rtx-a6000]] | 48 | 768 | **Y/Y/Y** | 38.4 | no bare-card price in supply/ |
| [[accelerators/nvidia-a40]] | 48 | 696 | **Y/Y/Y** | 34.8 | no bare-card price in supply/ |
| [[accelerators/nvidia-quadro-rtx-6000]] | 24 | 672 | **Y/Y/-** | 33.6 | Turing TU102, not the 6000 Ada; no price record |
| [[accelerators/nvidia-quadro-rtx-8000]] | 48 | 672 | **Y/Y/Y** | 33.6 | 48 GB Turing; fits 70B, same BW as the 6000; no price record |
| [[accelerators/nvidia-rtx-pro-4000-blackwell]] | 24 | 672 | **Y/Y/-** | 33.6 | no bare-card price in supply/ |
| [[accelerators/amd-radeon-ai-pro-r9700]] | 32 | 640 | **Y/Y/-** | 32.0 | no price record |
| [[accelerators/cambricon-mlu370-x8]] | 48 | 614.4 | **Y/Y/Y** | 30.7 | dual-die card; no price record |
| [[accelerators/local-apple-m5-max]] | 128 | 614 | **Y/Y/Y** | 30.7 | no bare-part price |
| [[accelerators/nvidia-a10]] | 24 | 600 | **Y/Y/-** | 30.0 | cloud/distributor only |
| [[accelerators/nvidia-rtx-5000-ada]] | 32 | 576 | **Y/Y/-** | 28.8 | no bare-card price in supply/ |
| [[accelerators/qualcomm-ai200]] | 128 | 548 | **Y/Y/Y** | 27.4 | no price record |
| [[accelerators/amd-radeon-pro-v620]] | 32 | 512 | **Y/Y/-** | 25.6 | no price record |
| [[accelerators/huawei-ascend-950dt]] | 96 | 500 | **Y/Y/Y** | 25.0 | no price record |
| [[accelerators/huawei-atlas-300i-duo]] | 96 | 408 | **Y/Y/Y** | 20.4 | no price record (the 24 GB Pro sibling is priced; this Duo is not) |
| [[accelerators/apple-m1-max]] | 64 | 400 | **Y/Y/Y** | 20.0 | no bare-part price; laptop/desktop silicon |
| [[accelerators/apple-m2-max]] | 96 | 400 | **Y/Y/Y** | 20.0 | no bare-part price |
| [[accelerators/local-apple-m5-pro]] | 64 | 307 | **Y/Y/Y** | 15.3 | no bare-part price |
| [[accelerators/nvidia-l4]] | 24 | 300 | **Y/Y/-** | 15.0 | cloud/distributor only |

Thirty-seven further accelerators qualify on the bandwidth-or-price rule and do
**not** fit 35B-A3B (all unpriced). The highest-bandwidth near-misses, all
`-/-/-` except the two 20 GB parts which fit 27B only: [[accelerators/amd-radeon-pro-vii]]
(16 GB, 1024 GB/s), [[accelerators/nvidia-rtx-5080]] (16 GB, 960),
[[accelerators/nvidia-rtx-3080-ti]] (12 GB, 912),
[[accelerators/nvidia-rtx-5070-ti]] (16 GB, 896),
[[accelerators/google-tpu-v5e]] (16 GB, 819),
[[accelerators/amd-radeon-rx-7900-xt]] (20 GB, 800, `-/Y/-`),
[[accelerators/nvidia-rtx-3080-10gb]] (10 GB, 760),
[[accelerators/nvidia-rtx-4080-super]] / [[accelerators/nvidia-rtx-4080]],
[[accelerators/google-tpu-v2]] (16 GB, 700),
[[accelerators/local-nvidia-rtx-5070]] (12 GB, 672),
[[accelerators/nvidia-rtx-4070-ti-super]],
[[accelerators/amd-radeon-rx-9070]] (16 GB, 640),
[[accelerators/nvidia-rtx-a4500]] (20 GB, 640, `-/Y/-`), and
[[accelerators/nvidia-rtx-4000-ada]] (20 GB, 360 GB/s, `-/Y/-`). Five of the
thirty-seven are new with #172 and sit at the bottom of that list, all
`-/-/-`: [[accelerators/amd-radeon-rx-6800]] (16 GB, 512 GB/s, 35B ceiling
25.6 tok/s *(derived)*), [[accelerators/nvidia-rtx-2070-super]] and
[[accelerators/nvidia-rtx-3060-ti]] (both 8 GB, 448 GB/s, 22.4),
[[accelerators/amd-radeon-rx-6750-xt]] (12 GB, 432 GB/s, 21.6) and
[[accelerators/amd-radeon-rx-6700-xt]] (12 GB, 384 GB/s, 19.2). The full
148-row list, including these 37, is `python tools/fit_matrix_35b.py`.

## 4b. Roofline vs. measured — the A3B-MoE cross-check

The 35B ceiling column divides by the **full resident** 20 GB, which makes it a
dense-equivalent *floor* for a MoE that reads only ~3B active per token (§2).
The Phase-C benchmark backfill landed **22+ third-party `decode_tok_s` records
on real A3B-class MoE models** (Qwen3-30B-A3B / Qwen3.5/3.6-35B-A3B, Q4-class
quants, batch-1 per-request) — enough to check how far off the floor each part
actually lands. `measured ÷ ceiling` below is derived arithmetic; the measured
numbers themselves come from the cited records.

| accelerator | BW GB/s | ceiling *(derived)* | measured tok/s | × over floor *(derived)* | benchmark record |
|---|---|---|---|---|---|
| [[accelerators/nvidia-rtx-5090]] | 1792 | 89.6 | 234.3 | 2.6× | [[benchmarks/hc-rtx5090-qwen3-30b-a3b-q4kxl-4k-tg128]] *(4k; a second page on the same card, same quant, same depth is 226.1 → 2.5× [[benchmarks/hc26-5090-qwen3-30b-a3b-q4k-4k-tg]], 226.1/89.6 = 2.52 *(derived)*. The two are not averaged.)* |
| [[accelerators/nvidia-rtx-pro-5000-blackwell]] | 1344 | 67.2 | 188.8 | 2.8× | [[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-4k-tg]] *(4k; 16k is 148.5 → 2.2×, [[benchmarks/hc-rtxpro5000-qwen3-30b-a3b-16k-tg]]. #163 continues the same series: 123.2 at 32k → 1.8× [[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-32k]], 91.9 at 64k → 1.4× [[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-64k]], 61.4 at 128k → 0.91× [[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-128k]], 37.0 at 256k → 0.55× [[benchmarks/hc26-pro5000-qwen3-30ba3b-tg-256k]]. All ÷ 67.2, *(derived)*. The floor is crossed between 64k and 128k. Unpriced — §4, not §3.)* |
| [[accelerators/nvidia-rtx-4090]] | 1008 | 50.4 | 207.3 | 4.1× | [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-4k]] *(4k; same series 139.2 at 16k → 2.8× [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-16k]], 105.1 at 32k → 2.1× [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-32k]], 68.2 at 64k → 1.4× [[benchmarks/hc26-4090-qwen3-30b-a3b-tg-64k]]. All ÷ 50.4, *(derived)*. The 109.57 cell at 16k [[benchmarks/hwcorner-rtx-4090-qwen3-30b-a3b-q4kxl-decode-16k]] is a different page and still stands. The card stays above the floor through 64k.)* |
| [[accelerators/nvidia-rtx-3090-ti]] | 1008 | 50.4 | 98.25 | 1.9× | [[benchmarks/hwcorner-rtx-3090-ti-qwen3-30b-a3b-q4kxl-decode-16k]] *(16k; the 32k cell on the same card is 91.97 → 1.8×, [[benchmarks/hwcorner-rtx-3090-ti-qwen3-30b-q4kxl-decode-32k]], 91.97/50.4 = 1.82 *(derived)*. No 4k cell in the corpus.)* |
| [[accelerators/nvidia-rtx-3090]] | 936 | 46.8 | 179.0 | 3.8× | [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-4k]] *(4k; same Q4_K series is 98.3 at 16k → 2.1× [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-16k]], 58.8 at 32k → 1.3× [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-32k]], 32.8 at 64k → 0.70× [[benchmarks/hc26-3090-qwen3-30b-a3b-tg-64k]]. All ÷ 46.8, *(derived)*. The floor is crossed between 32k and 64k. A second 4k page on this card is 153.6 → 3.3× [[benchmarks/hc26-3090-qwen3-30b-a3b-q4k-4k-tg]]; the two are not averaged.)* |
| [[accelerators/nvidia-rtx-6000-ada]] | 960 | 48.0 | 199.6 | 4.2× | [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-4k]] *(4k; same series 120.1 at 16k → 2.5× [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-16k]], 74.3 at 32k → 1.5× [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-32k]], 40.8 at 64k → 0.85× [[benchmarks/hc26-6000ada-qwen3-30b-a3b-tg-64k]]. All ÷ 48.0, *(derived)*. The floor is crossed between 32k and 64k. The 104.88 cell at 16k [[benchmarks/hwcorner-rtx-6000-ada-qwen3-30b-a3b-q4kxl-decode-16k]] is a different page and still stands. Unpriced — §4.)* |
| [[accelerators/nvidia-rtx-5070-ti]] | 896 | 44.8 | 111.11 | 2.5× | [[benchmarks/hwcorner-rtx-5070-ti-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/nvidia-rtx-4080]] | 716.8 | 35.8 | 85.04 | 2.4× | [[benchmarks/hwcorner-rtx-4080-qwen3-30b-a3b-q4kxl-decode-16k]] |
| [[accelerators/local-apple-m5-max]] | 614 | 30.7 | 128.0 | 4.2× | [[benchmarks/ss-m5max-48-qwen35-35b-mlx-4bit-tg]] *(MLX path; llama.cpp Q4_K_M row lands 89.4 → 2.9×, [[benchmarks/ss-m5max-48-qwen35-35b-llamacpp-q4km-tg]])* |
| [[accelerators/apple-m4-max]] | 546 | 27.3 | 113.33 | 4.2× | [[benchmarks/mlx-m4-max-qwen3-30b-a3b-4bit-decode-tok-s]] |
| [[accelerators/apple-m4-max]] | 546 | 27.3 | 92.09 | 3.4× | [[benchmarks/ss-m4max-40core-qwen3-30b-mlx-q4-tg]] *(second, community run on the same part)* |
| [[accelerators/amd-epyc-9454p-genoa]] | 460.8 | 23.0 | 63.1 | 2.7× | [[benchmarks/comm-epyc-9454p-qwen3-moe-30b-decode]] |
| [[accelerators/local-apple-m5-pro]] | 307 | 15.3 | 41.9 | 2.7× | [[benchmarks/ss-m5pro-64-qwen35-35b-llamacpp-q4kl-tg]] |
| [[accelerators/amd-radeon-8060s-ryzen-ai-max]] | 256 | 12.8 | 72.35 | 5.7× | [[benchmarks/lhl-8060s-qwen3-30b-a3b-ud-q4-k-xl-vulkan-tg1]] *(best of 20 Strix Halo runs; the pass-7 row [[benchmarks/kyuz0-strix-8060s-qwen36-35b-a3b-q4kxl-vulkan-radv-tg128]] at 60.43 → 4.7× still stands)* |
| [[accelerators/nvidia-rtx-pro-6000-blackwell]] | 1792 | 89.6 | 198.7 | 2.2× | [[benchmarks/hc26-rtxpro6000-qwen3-30b-a3b-q4k-tg-4k]] *(4k context; the same card's 256k row is 38.4 and is a context effect, not a roofline one)* |
| [[accelerators/nvidia-a100-40gb-sxm4]] | 1555 | 77.8 | 148 | 1.9× | [[benchmarks/llamabench-a100sxm40-ornith15-35b-a3b-q4km-tg128]] |
| [[accelerators/nvidia-a100-40gb-pcie]] | 1555 | 77.8 | 136 | 1.7× | [[benchmarks/llamabench-a100pcie40-qwen36-35b-a3b-q4kxl-tg128]] |
| [[accelerators/nvidia-rtx-a6000]] | 768 | 38.4 | 131.2 | 3.4× | [[benchmarks/hc26-a6000-qwen3-30b-a3b-q4k-4k-tg]] *(4k; the 16k cell already quoted is 75.35 → 2.0× [[benchmarks/hwcorner-rtx-a6000-qwen3-30b-a3b-q4kxl-decode-16k]]. 131.2/38.4 = 3.42, *(derived)*. Unpriced — §4.)* |

Three reads:

- **The ordering is the same at one context depth, and it is not at 4k.**
  The 16k cells this section was built on sit in a ~2–2.6× band and do not
  reorder the bandwidth ranking. The 4k cells now cited do reorder *within*
  a generation: the 3090 at 179.0 (3.8×) and the 6000 Ada at 199.6 (4.2×)
  both clear the 4090's *16k* number and do not clear the 4090's own 4k
  number (207.3). The RTX 5090 still leads the short-context list (234.3).
  The RTX PRO 5000 Blackwell cell (188.8 tok/s, 2.8×) still slots between
  the 5090 and the PRO 6000 on bandwidth and on the 4k measurement, which
  is what the floor predicts. What the new sweep adds is the depth at which
  the floor stops being a floor: 3090 and 6000 Ada cross it between 32k and
  64k, the PRO 5000 between 64k and 128k.
- **The floor is ~2–6× low, and *where* it is lowest is systematic.** The
  tightest-margin parts (Strix Halo now at 5.7× on its best run, M5 Max and
  M4 Max at 4.2×) are exactly the bandwidth-starved ones — the MoE reads only
  ~3B/token, so a 256 GB/s pool sustains ~72 tok/s real where the
  dense-equivalent floor says 12.8. The discrete GPUs cluster at ~2–2.6×
  because KV reads and kernel overhead bite closer to the roofline. The
  engine matters too: the same M5 Max measures 128.0 tok/s under MLX but
  89.4 under llama.cpp — a 1.4× swing on one part, wider than some
  cross-part gaps. **Do not use the ceiling column to compare a MoE target
  across architectures** — it ranks $/tok-s correctly only because the error
  is monotonic in bandwidth.
- **The cheapest *measured* A3B answer depends on the context depth.** At the
  16k cell this section used to quote, [[accelerators/nvidia-rtx-3090-ti]] at
  $950 with 98.25 tok/s is still ~$9.67/measured-tok/s *(derived)*, and its
  1.9× margin was the worst of the set that had been cited. The 3090's own
  sweep, now in the table, changes the short-context read and not the deep
  one: **179.0 tok/s at 4k** on the same $950 sticker is **$5.31/measured-tok/s
  *(derived, 950/179.0)***, the cheapest verified fit at that depth, at
  **3.8×** its floor — no longer the worst margin. By 64k the same card is at
  32.8 tok/s and **under** the floor (0.70×), while the 4090 is still above it
  (68.2, 1.4×) and costs $9.65/measured-tok/s at its 4k cell *(derived,
  2000/207.3)*. The Ti still has no 4k cell here, so it cannot be compared to
  the 3090 at short context. Strix Halo's 5.7× multiplier (10.2× on the
  depth-0 wiki row, which is a different protocol and not the cell this
  paragraph ranks on) is why [[gotchas/gotcha-strix-halo-moe-not-dense]] calls
  the MoE target the workload that part is *for*.

## 5. Why the cheap rows are cheap — read before buying

The $/tok-s ranking is honest arithmetic on **asking prices for the part as
listed**, and four of the top six are cheap for a *reason the price doesn't
show*. These are the corpus's own gotcha records, not new claims:

- **SXM/OAM is not a card.** [[accelerators/nvidia-v100-sxm2-32gb]] (#3),
  [[accelerators/intel-gaudi2]] (#6), [[accelerators/amd-instinct-mi250]] and
  [[accelerators/amd-instinct-mi250x]] (#18/#19) are modules. Per
  [[gotchas/gotcha-sxm-module-needs-carrier-baseboard]], the SXM discount is a
  *form-factor* discount — the module needs a carrier
  baseboard + heatsink and has no PCIe edge. Gaudi2 is worse:
  [[gotchas/gotcha-gaudi2-hl225-oam-only-chassis-required]] records that HL-225
  has *no PCIe path at all* and needs a UBB chassis + fabric. Add that cost to
  the $/tok-s before comparing it to a PCIe row.
- **Pascal/Maxwell has no FP16 tensor path.** [[accelerators/nvidia-tesla-m40]]
  (#5) and [[accelerators/nvidia-tesla-p40]] (#8) look like steals at $189/$290,
  but [[gotchas/gotcha-pascal-p40-m40-no-fp16-tensor-dp4a-only]] is explicit:
  the M40's low sticker is a trap (no DP4A, slower than P40 on the only LLM path
  that matters), both are passive cards needing a shroud+blower, and their
  EPS-12V power connector is *not* a PCIe 8-pin. The P40's DP4A INT8 path is the
  floor buy; the M40 is not a cheaper P40.
- **MI100's $720 needs a ROCm tax.** [[gotchas/gotcha-mi100-multigpu-rocm-segfault]]
  records that MI100 is the best $/(GB·TB/s) in the used slice but multi-GPU
  segfaults on bare-metal ROCm and wants a PCIe-passthrough VM — and gfx908 is
  maintenance-tier on the ROCm matrix.
- **A "fits" flag is not a speed promise — and the new edge-NPU rows prove it.**
  The M40 fits 35B-A3B (24 GB) but its 288 GB/s ceiling is 14.4 tok/s *(derived)* —
  barely interactive. The cheap edge-NPU fits are slower still:
  [[accelerators/huawei-ascend-310p]] and [[accelerators/huawei-atlas-300i-pro]]
  (both 204.8 GB/s → 10.2 tok/s ceiling *(derived)*),
  [[accelerators/qualcomm-ai100]] (137 GB/s → 6.8 tok/s) and
  [[accelerators/cambricon-mlu370]] (307.2 GB/s → 15.4 tok/s) are NPUs/AIDC parts
  whose low sticker reflects an immature or edge-targeted software stack, not a
  serving answer — a `Y` on 70B (MLU370) means "the bytes fit," not "it serves."
  Conversely the unified-memory parts (Apple, Strix Halo) fit large models but
  are bandwidth-capped; [[gotchas/gotcha-strix-halo-moe-not-dense]] measures a
  dense-70B at ~5 tok/s on the ~256 GB/s Strix Halo wall, while the same record
  class measures the 35B-A3B **MoE** target at 42 tok/s
  ([[benchmarks/ryzen-ai-max-8060s-qwen35-35b-a3b-decode-tok-s]]). The MoE
  target is exactly the workload those parts *are* good for.

## 6. Assumptions and what this table deliberately ignores

- **Price class:** every purchase price is an *asking* price (used-market,
  reseller, or MSRP), not a realised transaction price. The repo stamps each
  with a `PRICE CLASS` and a `RE-VERIFY BY` date; the Oct-2026 used records are
  freshest. `$`/`tok-s` inherits that uncertainty.
- **Unit price ≠ system price.** Prices are per bare card/module where the
  record gives one. Several rows price a *whole system* instead — the part is
  not sold bare: [[supply/used-mac-studio-m1-ultra-128gb]] ($2,500),
  [[supply/used-mac-studio-m2-ultra-192gb]] ($3,700),
  [[supply/used-mac-studio-m3-ultra-512gb]] ($9,000 — the 512 GB config
  the accelerator record's `vram_gb` describes; the 96 GB Studio at
  $3,700 is a different machine and is not joined) are whole Mac Studios, and
  all six Strix Halo rows ([[supply/new-minisforum-ms-s1-max-128gb]] $1,900,
  [[supply/new-framework-desktop-strix-halo-128gb]],
  [[supply/new-beelink-gtr9-pro-strix-halo-128gb]],
  [[supply/new-gmktec-evo-x2-strix-halo-128gb]],
  [[supply/new-beelink-gtr9-pro-strix-halo-128gb-2]],
  [[supply/new-gmktec-evo-x3-strix-halo-128gb]]) are whole mini-PCs/boards, plus
  the DGX-1 (below), two whole **MacBook Pro laptops**
  ([[supply/used-macbook-pro-m3-max-128gb]] $3,400 and
  [[supply/used-macbook-pro-m4-max-128gb]] $4,700), and — new this pass — a
  whole **EPYC self-build**, [[supply/used-epyc-9004-genoa-build-384gb]]
  ($7,500: used CPU + SP5 board + 384 GB DDR5 + chassis, joined to
  [[accelerators/amd-epyc-9554-genoa]]). Dividing a whole-system price by a
  per-device ceiling is an apples/orange flagged here, not hidden — the
  laptop rows price the screen, battery and logic board as well as the
  unified pool, and the Genoa row prices the DRAM and the motherboard, which
  is why both sit at the bottom of the ranking despite a `Y/Y/Y` fit. These
  parts are listed because no bare-silicon price exists, and for the
  unified-memory rows the *system* is the product.
  **Capacity-mismatched Apple SKUs stay unjoined.**
  [[supply/used-mac-studio-m1-ultra-64gb]] ($1,850) is a 64 GB chassis against
  a record whose `vram_gb` is 128; [[supply/used-mac-studio-m3-ultra-256gb]]
  ($6,250) and [[supply/used-mac-studio-m3-ultra-96gb]] ($3,700) price less
  pool than the M3 Ultra record's 512 GB. Joining either would rank a cheaper
  machine against a fit flag it does not have.
- **Derived, not measured.** `docs/08-cost-per-token.md` joins *measured*
  benchmarks to *rental* prices; this doc joins *roofline ceilings* to
  *purchase* prices. They answer different questions and neither substitutes
  for the other. A `Y` here and a cheap $/tok-s do not mean the part serves the
  model well at concurrency.
- **Deliberately dropped joins** (the raw `accelerator_ids` link would
  over-attribute, per the same trap `tools/cost_per_token.py` documents):
  - `ebay-used-accelerator-market` ($10,449.95) is **one A100 80GB SXM4**
    listing; its H100/H200 ids are "obtainable," not "priced at $10,450." Not
    joined to H100/H200.
  - `ebay-used-a100-40gb-sxm4` ($2,499) prices a **40GB** module but points at
    the 80GB id (no 40GB record when written). Mapped to
    [[accelerators/nvidia-a100-40gb-sxm4]], not the 80GB part.
  - `microcenter-refurb-rtx-3090` is two tiers: 3090 $1,099.99 / 3090 Ti
    $1,239.99 — each joined at its own price.
  - `new-retail-rtx-5090-amazon` headline is the **5090** band; the 5080 sibling
    band lives in `notes` and is not the headline price. 5080 not joined.
  - `new-retail-rtx-4090-3090-consumer` headline $3,299 is the **4090** median;
    the 3090 median is $1,514 in notes. 3090 not joined at $3,299.
- **Whole-system rows kept for context but not mis-joined:**
  `used-dgx1-8x-v100` and `used-dgx1-8xv100-32gb-chassis` ($4,000) are **8-GPU**
  servers → ~$500/GPU-equivalent
  (derived), consistent with the $500 bare-SXM2 row but *system* prices.
  `gaudi2-8x-oam-tray-supermicro-820gh` ($11,995) is an **8-module** OAM tray →
  ~$1,500/module-equivalent (derived), a system price not a module price.
  `ebay-used-a100-40gb-rack-liquidator` ($249,000) is a 32-GPU rack;
  `used-8x-h100-sxm-node-system` ($180,000) is an 8-GPU server. None are
  bare-card prices.
- **Edge/SBC whole-board prices are not accelerator prices.**
  `new-raspberry-pi-ai-hat-2` ($200, Hailo-10H + a Pi 5),
  `new-rockchip-rk3588-sbc` ($140, RK3588 board) and
  `used-google-coral-edge-tpu` ($90) price a *whole single-board computer*, not
  the NPU alone — and their accelerator records carry no `vram_gb >= 8` anyway
  (Hailo-10H has 8 GB but no `memory_bandwidth_gbps`; the other two are
  `vram_gb`-null unified-memory SoCs), so they would not qualify even if joined.
  Left unjoined so the price column stays a GPU/part price.

## 7. Reproduce

    python tools/fit_matrix_35b.py            # the full 148-row matrix, all three sections (GB10 is outside the gate; see the Oct-16 p19 note)
    python tools/validate.py                  # record integrity (no new records added here)

Every figure above is generated by that tool against the committed `data/`
records; where the tool's arithmetic disagrees with a table cell, the tool is
right and the cell is stale. The tool also carries the two judgement calls the
table depends on, documented inline: the curated `PRICE_JOIN` (which corrects
the raw `accelerator_ids` over-attribution the same way
`tools/cost_per_token.py` does) and `UNIFIED_MEM_GB` (the configured-memory
override that keeps `vram_gb`-null unified-memory parts like Strix Halo from
being silently filtered out by the `>= 8 GB` gate).
