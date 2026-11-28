# E1 — agentic-axis shrink calibration

Generated from worth-running rows (regenerated in memory by tools/build_joined_views.build_worth_running()); 131 models with an agentic axis; prior 75.1.

## Raw agentic vs independent axes

| axis | n | Pearson | Spearman |
|---|---:|---:|---:|
| capability (includes agentic — not independent) | 112 | 0.743 | 0.778 |
| capability excl. agentic | 112 | 0.343 | 0.392 |
| coding | 90 | 0.525 | 0.564 |
| instruction | 56 | 0.017 | 0.006 |
| knowledge | 88 | 0.283 | 0.303 |

Raw agentic predicts coding moderately and instruction not at all. The published-capability correlation is inflated by construction because the agentic axis is one of its terms.

## Spearman across K

| K | vs capability | vs capability excl. agentic | vs coding | vs instruction | single-family in top10 | single-benchmark in top10 | overlap with K=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.778 | 0.392 | 0.564 | 0.006 | 5/10 | 3/10 | 10/10 |
| 0.5 | 0.814 | 0.417 | 0.581 | -0.023 | 4/10 | 2/10 | 6/10 |
| 1.0 | 0.817 | 0.418 | 0.579 | -0.026 | 3/10 | 1/10 | 5/10 |
| 1.5 | 0.816 | 0.417 | 0.578 | -0.035 | 3/10 | 1/10 | 5/10 |
| 2.0 | 0.814 | 0.414 | 0.573 | -0.041 | 3/10 | 0/10 | 4/10 |
| 2.5 | 0.812 | 0.409 | 0.574 | -0.046 | 3/10 | 0/10 | 4/10 |
| 3.0 | 0.810 | 0.406 | 0.572 | -0.050 | 3/10 | 0/10 | 4/10 |
| 3.5 | 0.810 | 0.405 | 0.573 | -0.051 | 3/10 | 0/10 | 4/10 |
| 4.0 | 0.811 | 0.409 | 0.578 | -0.053 | 3/10 | 0/10 | 4/10 |
| 4.5 | 0.810 | 0.408 | 0.577 | -0.051 | 3/10 | 0/10 | 4/10 |
| 5.0 | 0.810 | 0.409 | 0.577 | -0.047 | 3/10 | 0/10 | 4/10 |
| 5.5 | 0.811 | 0.409 | 0.578 | -0.050 | 3/10 | 0/10 | 4/10 |
| 6.0 | 0.812 | 0.412 | 0.577 | -0.050 | 3/10 | 0/10 | 4/10 |
| 6.5 | 0.809 | 0.410 | 0.579 | -0.051 | 3/10 | 0/10 | 4/10 |
| 7.0 | 0.811 | 0.413 | 0.576 | -0.051 | 3/10 | 0/10 | 4/10 |
| 7.5 | 0.812 | 0.414 | 0.579 | -0.049 | 3/10 | 0/10 | 4/10 |
| 8.0 | 0.810 | 0.412 | 0.578 | -0.052 | 3/10 | 0/10 | 4/10 |
| 8.5 | 0.812 | 0.415 | 0.578 | -0.049 | 3/10 | 0/10 | 4/10 |
| 9.0 | 0.811 | 0.413 | 0.575 | -0.052 | 3/10 | 0/10 | 4/10 |
| 9.5 | 0.808 | 0.409 | 0.572 | -0.054 | 3/10 | 0/10 | 4/10 |
| 10.0 | 0.808 | 0.409 | 0.572 | -0.048 | 3/10 | 0/10 | 4/10 |

**K=0** Spearman vs capability = 0.778; vs capability excl. agentic = 0.392.
**K=2** Spearman vs capability = 0.814; vs capability excl. agentic = 0.414.
**Recommended K=2.0**: K=2 is defensible. Spearman = 0.814 (published capability), 0.414 (agentic-excluded).

## Top-10 membership

### K=0
1. smollm3-3b — shrunk 100.0, raw 100.0, w 0.26, 1 famil(ies), 1 agentic row(s)
2. minimax-m1 — shrunk 100.0, raw 100.0, w 0.07, 1 famil(ies), 1 agentic row(s)
3. hunmin-397b-a17b-cua — shrunk 100.0, raw 100.0, w 0.75, 1 famil(ies), 1 agentic row(s)
4. raon-1-mini-preview — shrunk 99.6, raw 99.6, w 0.50, 2 famil(ies), 2 agentic row(s)
5. kimi-k3 — shrunk 98.5, raw 98.5, w 1.74, 2 famil(ies), 2 agentic row(s)
6. ling-3-1-flash — shrunk 97.9, raw 97.9, w 4.13, 3 famil(ies), 5 agentic row(s)
7. qwen3-8-27b — shrunk 97.8, raw 97.8, w 1.88, 3 famil(ies), 4 agentic row(s)
8. glm-5-3 — shrunk 97.4, raw 97.4, w 0.82, 1 famil(ies), 2 agentic row(s)
9. qwen3-8-flash-next — shrunk 97.3, raw 97.3, w 1.25, 2 famil(ies), 3 agentic row(s)
10. hunyuan-a13b — shrunk 97.2, raw 97.2, w 0.35, 1 famil(ies), 2 agentic row(s)

### K=2 (current)
1. ling-3-1-flash — shrunk 90.5, raw 97.9, w 4.13, 3 famil(ies), 5 agentic row(s)
2. qwen3-8-27b — shrunk 86.1, raw 97.8, w 1.88, 3 famil(ies), 4 agentic row(s)
3. kimi-k3 — shrunk 86.0, raw 98.5, w 1.74, 2 famil(ies), 2 agentic row(s)
4. mimo-v2-6-pro-rl — shrunk 85.5, raw 92.5, w 2.99, 3 famil(ies), 4 agentic row(s)
5. deepseek-v4-pro-0813 — shrunk 84.6, raw 96.7, w 1.57, 2 famil(ies), 2 agentic row(s)
6. deepseek-v4-1-flash-1017 — shrunk 84.2, raw 96.2, w 1.53, 1 famil(ies), 2 agentic row(s)
7. ornith-1-5-122b — shrunk 83.8, raw 89.1, w 3.31, 3 famil(ies), 4 agentic row(s)
8. qwen3-8-flash-next — shrunk 83.6, raw 97.3, w 1.25, 2 famil(ies), 3 agentic row(s)
9. arex-2 — shrunk 82.3, raw 93.8, w 1.26, 1 famil(ies), 2 agentic row(s)
10. veriloop-e2 — shrunk 81.9, raw 90.7, w 1.53, 1 famil(ies), 3 agentic row(s)

Between K=0 and K=2: 6 models entered, 6 exited, 4 are common; symmetric difference = 12.

## Recommendation

K=2 is defensible.

K=2 gives Spearman vs published capability 0.814 (n=112), vs the numeric maximum 0.817 at K=1.0; the gap is 0.003. Against the honest agentic-excluded capability it gives 0.414, vs a maximum 0.418 at K=1.0; the gap is 0.004. K=2 is within 0.005 of the maximum on both capability measures; the K=1 and K=1.5 point estimates are only marginally different. K=0 is worse on both (0.778 / 0.392) and puts 3 single-benchmark model(s) in the top 10; K=2 puts 0. K=2 is the smallest K in the sweep with zero single-benchmark models in the top 10, so it satisfies the thin-evidence guard without a measurable correlation cost. Bootstrap (K=0 minus K=2 Spearman): capability -0.036 95% CI [-0.100, +0.011], agentic-excluded -0.022 [-0.090, +0.039], coding -0.009 [-0.067, +0.046]. The intervals include zero, so correlation alone cannot statistically distinguish K=0 from K=2; the case for K=2 is the thin-evidence guard in the top 10.

### Caveats
- The published capability score is NOT independent of agentic: it includes the agentic axis, so its correlation is partly mechanical. The agentic-excluded capability column is the honest check.
- Instruction is the only axis with no meaningful relationship to agentic (Spearman near zero); coding is the axis that actually validates the agentic evidence.
- Every agentic score in this catalog is a vendor claim; there is no third-party agentic measurement to correlate against.
- Families come from benchmark_names.family_of; DeepSearchQA has no canonical mapping and is counted as web/search evidence.
- The sweep does not identify a unique optimal K: the K=0-to-K=2 correlation gain is small relative to the bootstrap noise, so K is better understood as a thin-evidence guard than as a finely tuned calibration parameter.
