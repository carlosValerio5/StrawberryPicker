# 0005 — Pre-registered acceptance criteria

- **Status:** accepted. Written before any audit numbers or model results exist.
- **Date:** 2026-11-02

## Context

If we choose what "good" means after seeing the numbers, any result can be made to look
like a pass. This ADR fixes the bar now. The referees are people other than the model
builder (two independent labelers), the USDA definitions
([0004](0004-usda-label-rubric.md)), and a test set the model never sees
([test-set protocol](../labeling/test-set-protocol.md)).

## Decision

All confidence intervals are 95 % percentile bootstrap intervals with 2,000 resamples.
Berries are resampled, not crops.

### A. The labels have to be reliable first

Krippendorff's α between the two labelers' **raw** maturity labels (ordinal) must be
**≥ 0.67**.

Below that, the rubric isn't usable. We revise it and relabel, and no model gets a
verdict. α for each defect (nominal) is reported but isn't a gate.

### B. Maturity model on the locked phone test set (adjudicated labels)

The model passes only if all three hold:
1. **Human-level agreement:** α(model, labeler) for **each** labeler is at least the
   lower bound of the confidence interval of α(labeler 1, labeler 2).
2. **No neglected class:** recall is **≥ 0.70** for each of `unripe`, `ripe` and
   `overripe`.
3. **Honest confidence:** expected calibration error (15 equal-width bins, top-label
   confidence) is **≤ 0.10**.

Quadratic weighted kappa, macro-F1 and the confusion matrix are reported alongside.
Results are also broken down by lighting condition.

### C. Quality on the locked phone test set

- **Rejects must be caught:** recall of `reject` grade (mold or rot), against the
  adjudicated grade, is **≥ 0.80**.
- The other defects get per-defect precision and recall with intervals. They aren't
  gates, because the test set will hold too few examples to judge them.

### D. Keep, relabel or drop each public source

For each source, a human reviews cleanlab's top 100 suspected label errors (or all of
them, if fewer). The estimated error rate is:

`e = (confirmed errors ÷ reviewed) × (cleanlab-flagged ÷ source size)`

| e | Action |
|---|---|
| ≤ 10 % | keep |
| 10–25 % | relabel: suspect rows become `needs_review` |
| > 25 % | drop the source |

### E. Diagnostics: always reported, never gates

- **Leave-one-source-out macro-F1.** A drop of more than 0.15 compared with
  in-distribution results means the report must explain it.
- **Source-identification probe accuracy:** shortcut risk.
- **Learning curves** at 10, 25, 50 and 100 % of the training data.

### F. Changing the bar

A threshold may change only through a new ADR, and only **before** results on the
locked set are seen. After that, a new threshold applies only to the next model version.
