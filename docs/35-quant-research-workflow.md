# Quant research workflow

How to answer "what is the most performant quant of the models people actually run" with data
that survives being checked. Two questions, deliberately kept apart:

1. **Which models do people run?** — measured, not guessed.
2. **What does a given quant level cost?** — measured from a published base-vs-quant
   comparison, never inferred from a name or a bit count.

Everything below is a tool you can re-run. Nothing here is a one-off hand count.

## Stage 1 — popularity from quant downloads

```bash
python tools/quant_popularity.py --fetch --limit 1000 --pages 2
python tools/quant_popularity.py --build --queue 25
```

`--fetch` sweeps the Hugging Face API once per format filter (`gguf`, `awq`, `gptq`, `mlx`,
`fp8`, `exl3`, `nvfp4`, `int4`, `auto-round`), 1000 repos per page, following `Link: rel=next`.
It caches to `/opt/data/quant_pop/raw_quants.json` so the build step is offline and repeatable.

**Why quant downloads and not base-model downloads.** A base repo's counter includes every
fine-tune pipeline, eval harness and CI job that pulls the weights. A quant repo is downloaded
by someone who intends to run that model on their own hardware. Summing a base model's quant
repos is a much closer proxy for "models people run" than the base counter, which is dominated
by machines. `Qwen/Qwen3.8-27B` leads at ~55.6M quant downloads; a base-model leaderboard would
have put `sentence-transformers/all-MiniLM-L6-v2` (182M) on top, which nobody runs as an LLM.

**Attribute to the root model, not the intermediate repo.** `SC117/X-abliterated-GGUF` declares
`base_model:quantized:ISTA-DASLab/X-GGUF`, which declares `base_model:quantized:Qwen/Qwen3.8-
Flash-Next`. Crediting the first hop files a re-quant of a re-quant under the re-quant. The tool
walks `base_model` up to the first non-quant ancestor, capped at 4 hops so a cycle cannot hang.

**Per-level popularity is not measurable.** The API reports downloads per *repo*, not per file,
so "Q4_K_M is the most-downloaded level" cannot be read from it. Repo numbers rank **models**.
Which level inside a repo people take is unknown unless a provider publishes per-file figures.
Do not state a per-level popularity ranking.

## Stage 2 — evidence, and what counts as evidence

Quality comes only from a reviewed verdict that has **an explicit base column and at least one
numeric quality row**. Four tiers, in descending trust:

| tier | shape | usable? |
|---|---|---|
| A | card table with a `base`/`BF16`/`Original` column and per-benchmark numbers | yes — recompute the delta |
| B | prose with both numbers ("PPL 4.010 → 4.107") | yes, as one row |
| C | a delta-only or `Recovery %` column with no base number | **no** — the base is absent |
| D | "quality is nearly identical" with no figures | **no** |

The review itself is the `quantization-quality-review` skill: one verdict per repo, written as
JSONL to `/opt/data/review/`. `tools/quant_popularity.py` joins those verdicts to the popularity
table by repo id and recomputes retention from the raw `base`/`quant` cells.

## Stage 3 — the accuracy rules that actually bite

Each of these came from a wrong number produced by an earlier version of this pipeline.

1. **Recompute every delta from `base`/`quant`; never trust a stored `delta_pct`.** Batches
   disagree on sign convention, and one wrong sign inverts a ranking.
2. **Flip lower-is-better metrics** (PPL, KLD, WER, CER) so that `< 100%` always means damage.
   Without the flip a perplexity rise reads as a quality *gain*.
3. **Exclude refusal, jailbreak, safety and abliteration rows from retention.** They measure the
   intervention, not the quant. The first run of this tool averaged a refusal battery
   (100 → 1) into the mean and reported **9900% retention**.
4. **Exclude throughput, context, size and memory rows.** `tok/s`, `GiB`, `VRAM` are not scores;
   pooling them produced "+337% retention" in an earlier batch.
5. **Cap at 102% and bucket anything above as `exceeds_base`.** A build cannot beat its own BF16
   by 5%. Real causes: pooled rows, a distilled teacher, a ±pp gate in the card. Excluded from
   the ladder, kept in `evidence` so the claim stays visible.
6. **Exclude abliterated / heretic / uncensored repos from the ladder** (`intervention: true`).
   They are a different product, not a quant level of the base.
7. **One benchmark per level.** A card publishing a KL ladder and a PPL ladder yields several
   rows for the same bpw; keep the most informative and never average across metrics.
8. **A name is not evidence.** `*-Q4_K_M` tells you the recipe, not the damage. No verdict, no
   row.

## Stage 4 — reading the output

`site/public/joined/quant_popularity.json` carries, per model:

- `quant_downloads`, `quant_repo_count`, `formats`, `top_repo` — the popularity side
- `levels[]` — one entry per measured bpw: `{bits, retention, format, repo, verdict}`
- `best_at_4bit` — the highest-retention build at ≤4.5 bpw (the practical local band)
- `evidence[]` — every reviewed verdict, including the ones excluded from the ladder, each
  flagged `intervention` / `exceeds_base` so a reader can see what was set aside and why
- `reviewed` — 0 means **unproven**, not "good". 2,981 of 3,027 models are currently 0.

`/opt/data/quant_pop/review_queue.json` is the work queue: the most-run models with no evidence,
with the 6 repos to review first for each.

## Stage 5 — refresh

- **Weekly**: `--fetch --build` (download counts move daily; the cache makes the rebuild free).
- **Monthly**: re-review the queue's top 25. Cards change; a verdict older than a quarter is
  stale by this repo's own rule.
- **On a new model release**: run `--fetch`, then check whether its quant repos are already
  reviewed; if not, queue it. A model with 5M downloads and no verdict is the highest-value
  review in the system.

## Known gaps, stated rather than hidden

- **2,599 quant repos are unattributed** — no `base_model` tag, and the name heuristic cannot
  place them. They are counted in `counts.unattributed`, not silently dropped.
- **Coverage is the headline finding**: 46 of 3,027 models (1.5%) have a usable base-vs-quant
  comparison. A ranking over that subset is a ranking of the legible minority, not the field.
- **Retention is directional, not precise.** Two builds at 97% may have been measured on
  different benchmarks by different providers. Compare within a model's ladder, not across
  models, unless the benchmark names match.
- **Downloads are lifetime totals**, not installs, and are not deduplicated across mirrors.
