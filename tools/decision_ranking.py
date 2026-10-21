#!/usr/bin/env python3
"""Render the decision-model view from data/decision_index.jsonl.

Decision models are a separate class, not text generators. A decision model takes
a state (text, JSON, image, video) plus a schema of typed questions and returns a
probability for every allowed option of every question in ONE forward pass, with
no output tokens and no parsing. Chat benchmarks do not apply to them: there is no
generation to score, and a refusal table is meaningless. The shared suite is the
Decision Index (Jev Decision Index 0.2.1, maintained by multimodalart and
contributors), which is chance-corrected: 0 is random guessing, 100 is perfect.

The catalog therefore ranks them here and nowhere else - the worth-running view
excludes task=decision rather than comparing them to generative models.

    python3 tools/decision_ranking.py > docs/33-decision-models.md
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "decision_index.jsonl"
AXES = [("knowledge", "Know."), ("language", "Lang."), ("retrieval", "Retr."),
        ("tools", "Tools"), ("artifacts", "Arts")]


def load():
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def fmt(v, nd=1, dash="—"):
    return dash if v is None else ("%.*f" % (nd, v))


def main():
    rows = sorted(load(), key=lambda r: r["rank"])
    n_self = sum(1 for r in rows if r.get("self_reported"))
    out = []
    out.append("# Decision models\n")
    out.append(
        "A decision model is not a text generator. It reads a state - text, JSON, an "
        "image, a video - together with a schema of typed questions, and returns a "
        "probability for every allowed option of every question in a **single forward "
        "pass**: zero output tokens, no parsing, and one logit per option that a "
        "softmax turns into a calibrated answer. That is a different job from writing "
        "a reply, so it gets a different ranking and different evidence.\n")
    out.append(
        "Chat benchmarks cannot rank these models. MMLU, SWE-bench and Terminal-Bench "
        "all score generated text; a decision model produces none. Nor does a refusal "
        "table mean anything, because the model answers with probabilities rather than "
        "prose. The shared suite is the **Decision Index 0.2.1** (Jev Decision Index, "
        "maintained by multimodalart and contributors), which is **chance-corrected**: "
        "0 is random guessing and 100 is perfect, so an index of 51.2 is not \"half "
        "right\" but \"well above chance on a five-way, unevenly-weighted task mix\".\n")
    out.append(
        "Within the Decision Index the weight is fixed by the suite, not by us: "
        "knowledge 26%, language 26%, retrieval 20%, tools 18%, artifacts 10%. "
        "**ECE** is expected calibration error - the gap between the confidence the "
        "model prints and the accuracy it achieves - so lower is better, and a model "
        "with a strong index and a bad ECE is confidently wrong.\n")
    out.append("\n## Ranking\n")
    hdr = ("| # | Model | Index | Know. | Lang. | Retr. | Tools | Arts | "
           "Median ms | p95 ms | ECE | Params | Base |")
    out.append(hdr)
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        name = r["name"]
        if r.get("hf"):
            name = "[%s](https://huggingface.co/%s)" % (name, r["hf"])
        elif r.get("project"):
            # A GitHub implementation rather than a weights repo: Mapika/decider,
            # simple-jev and jevfire ship as code against a stock base model.
            name = "[%s](%s)" % (name, r["project"])
        if r.get("self_reported"):
            name += " *"
        a = r.get("axes") or {}
        out.append("| %d | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["rank"], name, fmt(r.get("index"), 2),
            *[fmt(a.get(k)) for k, _ in AXES],
            fmt(r.get("median_ms"), 0), fmt(r.get("p95_ms"), 0), fmt(r.get("ece"), 3),
            fmt(r.get("params_b"), 1) + "B" if r.get("params_b") else "—",
            ("`%s`" % r["base"]) if r.get("base") else "—"))
    out.append("\n`*` = self-reported. %d of %d rows are the author's own run and the "
               "upstream board has not reproduced them; every other row is "
               "board-validated on one NVIDIA RTX PRO 6000.\n" % (n_self, len(rows)))
    out.append("\n## How to read this against the worth-running table\n")
    out.append(
        "- **Do not compare an index to a capability score.** They are different "
        "scales answering different questions, and a 61.2 here is not \"61.2%% of "
        "GPT\" - it is a chance-corrected composite over 38 decision benchmarks.\n"
        "- **Latency is part of the score, not a footnote.** A decision model returns "
        "in one pass, so the useful figure is median milliseconds per decision, and "
        "the 38 ms - 524 ms spread below is a bigger practical difference than the "
        "index spread. p95 matters more than median for an agent loop that stalls.\n"
        "- **Params decide where it runs.** The 3B - 9B band is the sweet spot for "
        "edge and single-GPU deployment; 26B - 36B costs a mid-range GPU for a few "
        "index points.\n"
        "- **Coverage is not exhaustive.** Only models currently scored in the "
        "Decision Index appear here. Community decision models that the suite has not "
        "scored are absent rather than bottom-ranked.\n")
    out.append("\n## Ingestion contract\n")
    out.append(
        "Decision models are records with `task: decision` and a `base_model`. They "
        "carry their Decision Index row as benchmarks named `Decision Index 0.2.1` "
        "and the five axis sub-scores, and they are **excluded from the worth-running "
        "ranking by construction** - not down-weighted, excluded - because that view "
        "averages text benchmarks. Adding one means: record the index, its five "
        "sub-scores, median and p95 latency, and ECE, and mark `vendor_claim: true` "
        "only when the number is the author's own run.\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
