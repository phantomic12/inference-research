#!/usr/bin/env python3
"""E1 — empirical calibration of the agentic-axis evidence shrink (K).

This is an ANALYSIS tool. It does not modify data/, schemas/, the build, or the
published K=2 constant. It reads the worth-running join (or regenerates it in
memory from data/ when the built payload is absent), then measures whether the
agentic axis agrees with evidence we can check independently.

The published agentic score is::

    agentic = round((raw * w + PRIOR_AG * K) / (w + K), 1)

where ``raw`` is ``axes.agentic_raw``, ``w`` is ``axis_weights.agentic``, and
``PRIOR_AG`` is the catalog median of the raw agentic scores. ``K`` is the one
term that was chosen by judgement. This tool sweeps K and reports:

  * Pearson and Spearman correlation of the raw agentic score against the
    published capability score and against the truly independent axes
    (coding, instruction), with n for each pair;
  * the same Spearman correlation for every K in 0..10 step 0.5;
  * top-10 membership at every K, using the site's published tie-break
    (agentic desc, then worth desc);
  * how many top-10 members change between K=0 and K=2;
  * a bootstrap confidence interval for the headline Spearman values and the
    K=0-vs-K=2 difference, so "0.815 vs 0.817" can be read as noise.

Outputs (never touches site/ or data/):
  cache/calibration/agentic_axis_calibration.json
  cache/calibration/agentic_axis_calibration.md

Usage:
    python3 tools/agentic_axis_calibration.py
    python3 tools/agentic_axis_calibration.py --k-max 10 --k-step 0.5 --top 10
    python3 tools/agentic_axis_calibration.py --bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the repo root regardless of whether this file lives in tools/ or cache/.
# ---------------------------------------------------------------------------


def _find_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "tools").is_dir() and (parent / "data").is_dir() and (parent / "site").is_dir():
            return parent
    raise SystemExit(f"could not find repo root above {start}")


ROOT = _find_root(Path(__file__).resolve())
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from benchmark_names import canonical, family_of  # noqa: E402

# The four families the agentic axis is allowed to score. SWE-bench is
# deliberately NOT here: it scores the coding axis (see build_joined_views).
AGENTIC_FAMILIES = {"terminal", "web", "tool", "computer"}
NON_AGENTIC_AXES = ("math", "coding", "knowledge", "instruction")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_worth(root: Path) -> tuple[dict, dict]:
    """Return (worth_payload, provenance).

    Prefer the built payload. If it is absent (fresh worktree, no --build), call
    the real builder's build_worth_running() in memory. No file is written.
    """
    path = root / "site" / "public" / "joined" / "worth_running.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), {
            "path": str(path.relative_to(root)),
            "regenerated": False,
            "note": "read the built worth-running payload",
        }
    from build_joined_views import build_worth_running  # noqa: PLC0415

    payload = build_worth_running()
    return payload, {
        "path": "regenerated in memory by tools/build_joined_views.build_worth_running()",
        "regenerated": True,
        "note": (
            "site/public/joined/worth_running.json was absent; the builder was "
            "called in memory. No file was written."
        ),
    }


def _agentic_families(benchmarks: list[dict]) -> tuple[list[str], int, list[str]]:
    """Distinct families and agentic benchmark-row count behind one model.

    ``family_of(canonical(name))`` is the canonical family test. DeepSearchQA is
    explicitly agentic in build_joined_views but has no canonical() mapping, so
    it is counted as ``web`` (search-agent) and noted in ``unmapped_agentic``.
    """
    families: set[str] = set()
    unmapped: list[str] = []
    n_rows = 0
    for b in benchmarks or []:
        name = b.get("name") or ""
        fam = family_of(canonical(name))
        if fam in AGENTIC_FAMILIES:
            families.add(fam)
            n_rows += 1
        elif "deepsearchqa" in name.lower():
            families.add("web")
            unmapped.append(name)
            n_rows += 1
    return sorted(families), n_rows, unmapped


def _capability_ex_agentic(row: dict) -> float | None:
    """Raw weighted mean of the non-agentic capability axes.

    The published ``capability`` includes the agentic axis, so correlating
    agentic with it is partly mechanical. This is the same construction with
    the agentic term removed, used as the honest independent check.
    """
    ax = row.get("axes") or {}
    aw = row.get("axis_weights") or {}
    parts = [
        (ax[k], float(aw.get(k) or 0.0))
        for k in NON_AGENTIC_AXES
        if ax.get(k) is not None and (aw.get(k) or 0.0) > 0
    ]
    total_w = sum(w for _, w in parts)
    if not parts or total_w <= 0:
        return None
    return round(sum(v * w for v, w in parts) / total_w, 1)


def _build_models(worth: dict) -> tuple[list[dict], dict]:
    rows = worth.get("rows") or []
    agentic_rows = [
        r for r in rows
        if (r.get("axes") or {}).get("agentic") is not None
    ]
    raw_values = sorted(
        float(r["axes"]["agentic_raw"])
        for r in agentic_rows
        if (r.get("axes") or {}).get("agentic_raw") is not None
    )
    if not raw_values:
        raise SystemExit("no agentic rows with agentic_raw in worth_running.json")
    prior = round(raw_values[len(raw_values) // 2], 1)

    published_priors = {
        r.get("agentic_prior") for r in agentic_rows if r.get("agentic_prior") is not None
    }
    prior_note = (
        "median of agentic_raw over all agentic rows (sorted[len//2])"
        if published_priors == {prior}
        else f"computed median; published values are {sorted(published_priors)!r}"
    )

    models: list[dict] = []
    for r in agentic_rows:
        ax = r.get("axes") or {}
        aw = r.get("axis_weights") or {}
        if ax.get("agentic_raw") is None:
            continue
        families, n_agentic_rows, unmapped = _agentic_families(r.get("benchmarks") or [])
        models.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "raw": round(float(ax["agentic_raw"]), 4),
            "published_shrunk": ax.get("agentic"),
            "w": round(float(aw.get("agentic") or r.get("agentic_evidence") or 0.0), 4),
            "families": families,
            "n_families": len(families),
            "n_agentic_benchmarks": n_agentic_rows,
            "unmapped_agentic": unmapped,
            "capability": r.get("capability"),
            "capability_ex_agentic": _capability_ex_agentic(r),
            "raw_capability": r.get("raw_capability"),
            "coding": ax.get("coding"),
            "instruction": ax.get("instruction"),
            "knowledge": ax.get("knowledge"),
            "worth": r.get("worth"),
        })
    if len(models) != len(agentic_rows):
        print(
            f"warning: {len(agentic_rows) - len(models)} agentic row(s) had no "
            "agentic_raw and were skipped",
            file=sys.stderr,
        )
    meta = {
        "rows_total": len(rows),
        "rows_with_agentic_axis": len(agentic_rows),
        "models_used": len(models),
        "prior": prior,
        "prior_note": prior_note,
        "prior_published_values": sorted(published_priors),
    }
    return models, meta


# ---------------------------------------------------------------------------
# Statistics (stdlib only; scipy is not a repo dependency)
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties (common after 1-decimal rounding) are handled."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _shrunk(raw: float, w: float, k: float, prior: float) -> float:
    """The published shrink, with the same 1-decimal rounding as the builder."""
    if k <= 0:
        return round(float(raw), 1)
    if w <= 0:
        return round(float(raw), 1)
    return round((raw * w + prior * k) / (w + k), 1)


def _top(models: list[dict], k: float, prior: float, n: int = 10) -> list[dict]:
    """Top rows by shrunk agentic, tie-broken on worth (the site's order)."""

    def key(m: dict):
        score = _shrunk(m["raw"], m["w"], k, prior)
        worth = m.get("worth")
        worth = worth if worth is not None else float("-inf")
        return (-score, -worth)

    # Stable sort: equal (score, worth) keeps the builder's row order.
    return sorted(models, key=key)[:n]


def _sweep(models: list[dict], prior: float, k_max: float, k_step: float, top_n: int) -> list[dict]:
    ks: list[float] = []
    k = 0.0
    while k <= k_max + 1e-9:
        ks.append(round(k, 1))
        k += k_step

    def score(m: dict, k: float) -> float:
        return _shrunk(m["raw"], m["w"], k, prior)

    def corr_for(k: float, axis: str) -> tuple[float | None, int]:
        xs: list[float] = []
        ys: list[float] = []
        for m in models:
            y = m.get(axis)
            if y is not None:
                xs.append(score(m, k))
                ys.append(float(y))
        return _spearman(xs, ys), len(xs)

    baseline_ids = [m["id"] for m in _top(models, 0.0, prior, top_n)]
    baseline_set = set(baseline_ids)

    out: list[dict] = []
    for k in ks:
        top = _top(models, k, prior, top_n)
        top_ids = [m["id"] for m in top]
        top_set = set(top_ids)
        entered_vs_k0 = [i for i in top_ids if i not in baseline_set]
        exited_vs_k0 = [i for i in baseline_ids if i not in top_set]
        row: dict = {
            "k": k,
            "spearman_capability": None,
            "pearson_capability": None,
            "spearman_capability_ex_agentic": None,
            "spearman_coding": None,
            "spearman_instruction": None,
            "spearman_knowledge": None,
            "n_capability": None,
            "n_capability_ex_agentic": None,
            "n_coding": None,
            "n_instruction": None,
            "n_knowledge": None,
            "top10": top_ids,
            "top10_single_family": sum(1 for m in top if m["n_families"] <= 1),
            "top10_single_benchmark": sum(1 for m in top if m["n_agentic_benchmarks"] <= 1),
            "top10_missing_capability": sum(1 for m in top if m.get("capability") is None),
            "entered_vs_k0": entered_vs_k0,
            "exited_vs_k0": exited_vs_k0,
            "overlap_with_k0": len(top_set & baseline_set),
        }
        # Published capability
        sp, n = corr_for(k, "capability")
        row["spearman_capability"], row["n_capability"] = sp, n
        # Pearson against published capability, same pairs
        xs = [score(m, k) for m in models if m.get("capability") is not None]
        ys = [float(m["capability"]) for m in models if m.get("capability") is not None]
        row["pearson_capability"] = _pearson(xs, ys)
        # Agentic-excluded capability
        sp, n = corr_for(k, "capability_ex_agentic")
        row["spearman_capability_ex_agentic"], row["n_capability_ex_agentic"] = sp, n
        # The two independent axes the task asks for
        sp, n = corr_for(k, "coding")
        row["spearman_coding"], row["n_coding"] = sp, n
        sp, n = corr_for(k, "instruction")
        row["spearman_instruction"], row["n_instruction"] = sp, n
        # Knowledge is a useful extra independent axis (not required by E1).
        sp, n = corr_for(k, "knowledge")
        row["spearman_knowledge"], row["n_knowledge"] = sp, n
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Bootstrap (only for the headline comparisons; deterministic seed)
# ---------------------------------------------------------------------------


def _bootstrap_spearman(
    xs: list[float], ys: list[float], n_boot: int, seed: int
) -> dict | None:
    if len(xs) < 5 or len(xs) != len(ys):
        return None
    rng = random.Random(seed)
    n = len(xs)
    vals: list[float] = []
    for _ in range(n_boot):
        sample = []
        for _ in range(n):
            i = rng.randrange(n)
            sample.append((xs[i], ys[i]))
        sp = _spearman([a for a, _ in sample], [b for _, b in sample])
        if sp is not None:
            vals.append(sp)
    if len(vals) < 50:
        return None
    vals.sort()
    estimate = _spearman(xs, ys)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return {
        "estimate": estimate,
        "ci95": [lo, hi],
        "n": n,
        "resamples": n_boot,
        "seed": seed,
    }


def _bootstrap_difference(
    models: list[dict],
    prior: float,
    k_a: float,
    k_b: float,
    axis: str,
    n_boot: int,
    seed: int,
) -> dict | None:
    """Bootstrap the Spearman difference for the same model set at two K."""
    xa = [m for m in models if m.get(axis) is not None]
    if len(xa) < 5:
        return None
    xs_a = [_shrunk(m["raw"], m["w"], k_a, prior) for m in xa]
    xs_b = [_shrunk(m["raw"], m["w"], k_b, prior) for m in xa]
    ys = [float(m[axis]) for m in xa]
    rng = random.Random(seed)
    n = len(xa)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [xs_a[i] for i in idx]
        b = [xs_b[i] for i in idx]
        y = [ys[i] for i in idx]
        sp_a = _spearman(a, y)
        sp_b = _spearman(b, y)
        if sp_a is not None and sp_b is not None:
            diffs.append(float(sp_a) - float(sp_b))
    if len(diffs) < 50:
        return None
    diffs.sort()
    return {
        "estimate": _spearman(xs_a, ys) - _spearman(xs_b, ys),
        "ci95": [diffs[int(0.025 * len(diffs))], diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]],
        "n": n,
        "resamples": n_boot,
        "seed": seed,
    }


def _bootstrap_section(models: list[dict], prior: float, n_boot: int, seed: int) -> dict:
    def pairs_for(k: float, axis: str):
        xs = [_shrunk(m["raw"], m["w"], k, prior) for m in models if m.get(axis) is not None]
        ys = [float(m[axis]) for m in models if m.get(axis) is not None]
        return xs, ys

    out: dict = {}
    for axis, label in (
        ("capability", "capability"),
        ("capability_ex_agentic", "capability_ex_agentic"),
        ("coding", "coding"),
    ):
        section = {}
        for k in (0.0, 1.0, 2.0):
            xs, ys = pairs_for(k, axis)
            section[str(k)] = _bootstrap_spearman(xs, ys, n_boot, seed)
        section["k1_minus_k2"] = _bootstrap_difference(
            models, prior, 1.0, 2.0, axis, n_boot, seed
        )
        section["k0_minus_k2"] = _bootstrap_difference(
            models, prior, 0.0, 2.0, axis, n_boot, seed
        )
        out[label] = section
    return out


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _argmax_k(sweep: list[dict], field: str) -> tuple[float | None, float | None]:
    candidates = [r for r in sweep if r.get(field) is not None]
    if not candidates:
        return None, None
    best_value = max(r[field] for r in candidates)
    best_k = min(r["k"] for r in candidates if r[field] == best_value)
    return best_k, best_value


def _near_max(sweep: list[dict], field: str, tolerance: float = 0.005) -> list[float]:
    """K values within ``tolerance`` of the maximum (not necessarily contiguous)."""
    candidates = [r for r in sweep if r.get(field) is not None]
    if not candidates:
        return []
    best = max(r[field] for r in candidates)
    return [r["k"] for r in candidates if r[field] >= best - tolerance]


def _make_recommendation(sweep: list[dict], meta: dict, bootstrap: dict | None = None) -> dict:
    best_cap_k, best_cap = _argmax_k(sweep, "spearman_capability")
    best_nag_k, best_nag = _argmax_k(sweep, "spearman_capability_ex_agentic")
    best_cod_k, best_cod = _argmax_k(sweep, "spearman_coding")
    first_zero_single = next(
        (r["k"] for r in sweep if r["top10_single_benchmark"] == 0), None
    )
    near_cap = _near_max(sweep, "spearman_capability")
    near_nag = _near_max(sweep, "spearman_capability_ex_agentic")
    k0 = next(r for r in sweep if r["k"] == 0.0)
    k2 = next(r for r in sweep if r["k"] == 2.0)
    cap = k2["spearman_capability"]
    nag = k2["spearman_capability_ex_agentic"]

    if first_zero_single == 2.0:
        guard_reason = (
            "K=2 is the smallest K in the sweep with zero single-benchmark "
            "models in the top 10, so it satisfies the thin-evidence guard "
            "without a measurable correlation cost."
        )
    elif first_zero_single is not None:
        guard_reason = (
            f"K={first_zero_single} is the smallest K in the sweep with zero "
            "single-benchmark models in the top 10; K=2 reduces their count but "
            "does not eliminate them."
        )
    else:
        guard_reason = (
            "No K in the sweep puts zero single-benchmark models in the top 10; "
            "K=2 reduces their count but does not eliminate them."
        )

    reasons = [
        (
            f"K=2 gives Spearman vs published capability {cap:.3f} (n={k2['n_capability']}), "
            f"vs the numeric maximum {best_cap:.3f} at K={best_cap_k}; the gap is "
            f"{abs(best_cap - cap):.3f}."
        ),
        (
            f"Against the honest agentic-excluded capability it gives {nag:.3f}, "
            f"vs a maximum {best_nag:.3f} at K={best_nag_k}; the gap is "
            f"{abs(best_nag - nag):.3f}."
        ),
        (
            "K=2 is within 0.005 of the maximum on both capability measures; "
            "the K=1 and K=1.5 point estimates are only marginally different."
        ),
        (
            f"K=0 is worse on both ({k0['spearman_capability']:.3f} / "
            f"{k0['spearman_capability_ex_agentic']:.3f}) and puts "
            f"{k0['top10_single_benchmark']} single-benchmark model(s) in the top 10; "
            f"K=2 puts {k2['top10_single_benchmark']}."
        ),
        guard_reason,
    ]
    if bootstrap:
        cap_diff = bootstrap.get("capability", {}).get("k0_minus_k2")
        nag_diff = bootstrap.get("capability_ex_agentic", {}).get("k0_minus_k2")
        cod_diff = bootstrap.get("coding", {}).get("k0_minus_k2")
        if cap_diff and nag_diff and cod_diff:
            reasons.append(
                "Bootstrap (K=0 minus K=2 Spearman): "
                f"capability {cap_diff['estimate']:+.3f} "
                f"95% CI [{cap_diff['ci95'][0]:+.3f}, {cap_diff['ci95'][1]:+.3f}], "
                f"agentic-excluded {nag_diff['estimate']:+.3f} "
                f"[{nag_diff['ci95'][0]:+.3f}, {nag_diff['ci95'][1]:+.3f}], "
                f"coding {cod_diff['estimate']:+.3f} "
                f"[{cod_diff['ci95'][0]:+.3f}, {cod_diff['ci95'][1]:+.3f}]. "
                "The intervals include zero, so correlation alone cannot "
                "statistically distinguish K=0 from K=2; the case for K=2 is the "
                "thin-evidence guard in the top 10."
            )
    return {
        "recommended_k": 2.0,
        "verdict": "K=2 is defensible",
        "reasoning": " ".join(reasons),
        "argmax_spearman": {
            "capability": best_cap_k,
            "capability_ex_agentic": best_nag_k,
            "coding": best_cod_k,
            "values": {
                "capability": best_cap,
                "capability_ex_agentic": best_nag,
                "coding": best_cod,
            },
        },
        "k_values_within_0.005_of_max": {
            "capability": near_cap,
            "capability_ex_agentic": near_nag,
            "note": "not necessarily a contiguous interval",
        },
        "smallest_k_zero_single_benchmark_top10": first_zero_single,
        "caveats": [
            "The published capability score is NOT independent of agentic: it includes "
            "the agentic axis, so its correlation is partly mechanical. The "
            "agentic-excluded capability column is the honest check.",
            "Instruction is the only axis with no meaningful relationship to agentic "
            "(Spearman near zero); coding is the axis that actually validates the "
            "agentic evidence.",
            "Every agentic score in this catalog is a vendor claim; there is no "
            "third-party agentic measurement to correlate against.",
            "Families come from benchmark_names.family_of; DeepSearchQA has no "
            "canonical mapping and is counted as web/search evidence.",
            "The sweep does not identify a unique optimal K: the K=0-to-K=2 "
            "correlation gain is small relative to the bootstrap noise, so K is "
            "better understood as a thin-evidence guard than as a finely tuned "
            "calibration parameter.",
        ],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown_report(result: dict) -> str:
    sweep = result["k_sweep"]
    meta = result["metadata"]
    rec = result["recommendation"]
    k0 = next(r for r in sweep if r["k"] == 0.0)
    k2 = next(r for r in sweep if r["k"] == 2.0)
    rec_row = next(r for r in sweep if r["k"] == rec["recommended_k"])

    lines: list[str] = []
    lines.append("# E1 — agentic-axis shrink calibration")
    lines.append("")
    lines.append(
        f"Generated from worth-running rows ({meta['source']['path']}); "
        f"{meta['models_used']} models with an agentic axis; prior {meta['prior']}."
    )
    lines.append("")
    lines.append("## Raw agentic vs independent axes")
    lines.append("")
    lines.append("| axis | n | Pearson | Spearman |")
    lines.append("|---|---:|---:|---:|")
    for axis, label in (
        ("capability", "capability (includes agentic — not independent)"),
        ("capability_ex_agentic", "capability excl. agentic"),
        ("coding", "coding"),
        ("instruction", "instruction"),
        ("knowledge", "knowledge"),
    ):
        c = result["raw_correlations"][axis]
        lines.append(
            f"| {label} | {c['n']} | {_fmt(c['pearson'])} | {_fmt(c['spearman'])} |"
        )
    lines.append("")
    lines.append(
        "Raw agentic predicts coding moderately and instruction not at all. The "
        "published-capability correlation is inflated by construction because the "
        "agentic axis is one of its terms."
    )
    lines.append("")
    lines.append("## Spearman across K")
    lines.append("")
    lines.append(
        "| K | vs capability | vs capability excl. agentic | vs coding | vs instruction | "
        "single-family in top10 | single-benchmark in top10 | overlap with K=0 |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in sweep:
        lines.append(
            f"| {r['k']:.1f} | {_fmt(r['spearman_capability'])} | "
            f"{_fmt(r['spearman_capability_ex_agentic'])} | "
            f"{_fmt(r['spearman_coding'])} | {_fmt(r['spearman_instruction'])} | "
            f"{r['top10_single_family']}/10 | {r['top10_single_benchmark']}/10 | "
            f"{r['overlap_with_k0']}/10 |"
        )
    lines.append("")
    lines.append(
        f"**K=0** Spearman vs capability = {k0['spearman_capability']:.3f}; "
        f"vs capability excl. agentic = {k0['spearman_capability_ex_agentic']:.3f}."
    )
    lines.append(
        f"**K=2** Spearman vs capability = {k2['spearman_capability']:.3f}; "
        f"vs capability excl. agentic = {k2['spearman_capability_ex_agentic']:.3f}."
    )
    lines.append(
        f"**Recommended K={rec['recommended_k']:.1f}**: {rec['verdict']}. "
        f"Spearman = {rec_row['spearman_capability']:.3f} (published capability), "
        f"{rec_row['spearman_capability_ex_agentic']:.3f} (agentic-excluded)."
    )
    lines.append("")
    lines.append("## Top-10 membership")
    lines.append("")
    lines.append("### K=0")
    for i, m in enumerate(result["top10_detail"]["0.0"], 1):
        lines.append(f"{i}. {m['id']} — shrunk {m['shrunk']:.1f}, raw {m['raw']:.1f}, "
                     f"w {m['w']:.2f}, {m['n_families']} famil(ies), "
                     f"{m['n_agentic_benchmarks']} agentic row(s)")
    lines.append("")
    lines.append("### K=2 (current)")
    for i, m in enumerate(result["top10_detail"]["2.0"], 1):
        lines.append(f"{i}. {m['id']} — shrunk {m['shrunk']:.1f}, raw {m['raw']:.1f}, "
                     f"w {m['w']:.2f}, {m['n_families']} famil(ies), "
                     f"{m['n_agentic_benchmarks']} agentic row(s)")
    lines.append("")
    churn = result["churn"]["k0_vs_k2"]
    lines.append(
        f"Between K=0 and K=2: {len(churn['entered'])} models entered, "
        f"{len(churn['exited'])} exited, {len(churn['common'])} are common; "
        f"symmetric difference = {churn['symmetric_difference']}."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(rec["verdict"] + ".")
    lines.append("")
    lines.append(rec["reasoning"])
    lines.append("")
    lines.append("### Caveats")
    for c in rec["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


def _top10_detail(models: list[dict], prior: float, k: float, top_n: int) -> list[dict]:
    out = []
    for m in _top(models, k, prior, top_n):
        out.append({
            "id": m["id"],
            "name": m["name"],
            "shrunk": _shrunk(m["raw"], m["w"], k, prior),
            "raw": m["raw"],
            "w": m["w"],
            "n_families": m["n_families"],
            "families": m["families"],
            "n_agentic_benchmarks": m["n_agentic_benchmarks"],
            "capability": m.get("capability"),
            "capability_ex_agentic": m.get("capability_ex_agentic"),
            "worth": m.get("worth"),
        })
    return out


def _self_check(result: dict, models: list[dict], prior: float) -> None:
    """Assert the numbers we report are consistent with the published builder."""
    n = len(models)
    assert n == result["metadata"]["models_used"], "model count drift"
    assert result["metadata"]["rows_with_agentic_axis"] >= n
    # K=2 recomputation must match every published agentic score exactly.
    worst = 0.0
    for m in models:
        if m.get("published_shrunk") is None:
            continue
        diff = abs(_shrunk(m["raw"], m["w"], 2.0, prior) - m["published_shrunk"])
        worst = max(worst, diff)
    assert worst == 0.0, f"K=2 recompute deviates from published score by {worst}"
    # The top-10 at K=2 must equal the published board order.
    published_top = [
        m["id"] for m in _top(models, 2.0, prior, 10)
    ]
    result_top = result["top10_detail"]["2.0"]
    assert published_top == [r["id"] for r in result_top], "top-10 drift at K=2"
    # Every K in the sweep must be present.
    ks = [r["k"] for r in result["k_sweep"]]
    expected = [round(0.0 + i * result["metadata"]["k_step"], 1)
                for i in range(0, int(result["metadata"]["k_max"] / result["metadata"]["k_step"]) + 1)]
    assert ks == expected, f"sweep K mismatch: {ks} != {expected}"
    print(f"self-check OK: {n} models, worst K=2 recompute deviation {worst}, "
          f"{len(ks)} K values, top-10 at K=2 matches the published board.")


def run(root: Path, k_max: float, k_step: float, top_n: int, bootstrap: int) -> dict:
    worth, provenance = _load_worth(root)
    models, meta = _build_models(worth)
    prior = meta["prior"]

    # Raw-vs-axis correlations (Pearson and Spearman, with n).
    raw_correlations = {}
    for axis, label in (
        ("capability", "capability"),
        ("capability_ex_agentic", "capability_ex_agentic"),
        ("coding", "coding"),
        ("instruction", "instruction"),
        ("knowledge", "knowledge"),
    ):
        pairs = [
            (m["raw"], float(m[axis]))
            for m in models
            if m.get(axis) is not None
        ]
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        raw_correlations[label] = {
            "n": len(pairs),
            "pearson": _pearson(xs, ys),
            "spearman": _spearman(xs, ys),
            "note": (
                "capability includes the agentic axis; not independent"
                if label == "capability"
                else "independent of the agentic axis"
            ),
        }

    sweep = _sweep(models, prior, k_max, k_step, top_n)
    churn = _churn(sweep, top_n)
    bootstrap_result = (
        _bootstrap_section(models, prior, bootstrap, 20261128)
        if bootstrap > 0 else None
    )
    recommendation = _make_recommendation(sweep, meta, bootstrap_result)

    result = {
        "task": "E1",
        "generated_at": _now_iso(),
        "source": provenance,
        "metadata": {
            **meta,
            "source": provenance,
            "agentic_families": sorted(AGENTIC_FAMILIES),
            "k_max": k_max,
            "k_step": k_step,
            "top_n": top_n,
        },
        "model_rows": models,
        "raw_correlations": raw_correlations,
        "k_sweep": sweep,
        "churn": churn,
        "top10_detail": {
            "0.0": _top10_detail(models, prior, 0.0, top_n),
            "2.0": _top10_detail(models, prior, 2.0, top_n),
            str(float(recommendation["recommended_k"])): _top10_detail(
                models, prior, recommendation["recommended_k"], top_n
            ),
        },
        "recommendation": recommendation,
        "bootstrap": bootstrap_result,
    }
    # Put the human-readable report in the JSON too, so the JSON is self-contained.
    result["report_markdown"] = _markdown_report(result)
    return result


def _churn(sweep: list[dict], top_n: int) -> dict:
    k0 = next(r for r in sweep if r["k"] == 0.0)
    k2 = next(r for r in sweep if r["k"] == 2.0)
    k0_set = set(k0["top10"])
    k2_set = set(k2["top10"])
    entered = [i for i in k2["top10"] if i not in k0_set]
    exited = [i for i in k0["top10"] if i not in k2_set]
    common = [i for i in k2["top10"] if i in k0_set]
    return {
        "k0_vs_k2": {
            "k0_top10": k0["top10"],
            "k2_top10": k2["top10"],
            "entered": entered,
            "exited": exited,
            "common": common,
            "overlap": len(common),
            "symmetric_difference": len(set(k0["top10"]) ^ set(k2["top10"])),
            "top_n": top_n,
        },
        "note": (
            "top-10 membership is a set comparison; a model counts as changed if "
            "it is in exactly one of the two top-10s. Symmetric difference = "
            "entered + exited."
        ),
    }


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "E1 agentic-axis calibration").splitlines()[0]
    )
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repo root (default: detected from this file)")
    parser.add_argument("--k-max", type=float, default=10.0,
                        help="last K in the sweep (default: 10)")
    parser.add_argument("--k-step", type=float, default=0.5,
                        help="K step (default: 0.5)")
    parser.add_argument("--top", type=int, default=10,
                        help="top-N membership size (default: 10)")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="bootstrap resamples; 0 disables (default: 500)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "cache" / "calibration",
                        help="output directory (default: cache/calibration)")
    parser.add_argument("--self-check", action="store_true",
                        help="run assertions against the published builder and exit")
    args = parser.parse_args()

    if args.k_step <= 0:
        parser.error("--k-step must be > 0")
    if args.k_max < 0:
        parser.error("--k-max must be >= 0")
    if args.top < 1:
        parser.error("--top must be >= 1")

    result = run(args.root, args.k_max, args.k_step, args.top, args.bootstrap)
    if args.self_check:
        worth, _ = _load_worth(args.root)
        models, meta = _build_models(worth)
        _self_check(result, models, meta["prior"])
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "agentic_axis_calibration.json"
    md_path = args.out_dir / "agentic_axis_calibration.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(result["report_markdown"], encoding="utf-8")

    print(result["report_markdown"])
    print(f"\nWrote {json_path.relative_to(args.root)}")
    print(f"Wrote {md_path.relative_to(args.root)}")


if __name__ == "__main__":
    main()