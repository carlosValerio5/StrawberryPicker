# 0006 — Estimate each source's label-error rate from a random sample

- **Status:** **proposed**. It amends ADR [0005](0005-acceptance-criteria.md) §D, which
  §F allows only before locked-set results exist. None exist yet. The project owner
  accepts or rejects it.
- **Date:** 2026-11-02

## Context

ADR 0005 §D estimates a source's error rate as
`(confirmed ÷ reviewed) × (cleanlab-flagged ÷ source size)`. It assumes confident
learning finds most label errors.

The first audit run showed that it doesn't:
- **Strawberry-DS:** confident learning flagged 11 of 1,027 maturity crops (1.1 %). The
  contact sheets showed boxes with **no fruit at all** in about 6–12 % of a random sample.
- **Why:** confident learning can only suggest one of the *existing* classes. A box of
  soil labelled `ripe` looks like no class, so it's rarely flagged.

So the §D formula is a **lower bound**, and it could keep a source that should be relabelled.

## Decision (proposed)

1. **Error rate:**
   - The audit writes a seeded random sample of 50 crops per source to
     `reports/dataset-audit/random_sample/`.
   - A reviewer marks each crop, using the [rubric](../labeling/rubric.md), as `ok`,
     `wrong_class`, `not_fruit` or `unsure`.
   - `e = (wrong_class + not_fruit) ÷ (reviewed − unsure)`, reported with a 95 % Wilson
     interval.
2. **The keep/relabel/drop thresholds in §D stay the same** (≤ 10 %, 10–25 %, > 25 %),
   applied to that `e`.
3. **The confident-learning list (`reports/dataset-audit/review/`) stays a tool for
   *finding and fixing* errors**, not for estimating their rate.

## Consequences

- One review session per source: about 50 crops at a glance, plus the flagged list.
- With n = 50 the interval is wide (about ±8 points near 10 %). A source near a
  threshold gets a second sample of 50 before it's dropped.
- `not_fruit` errors are evidence for the "is this a strawberry?" gate discussed earlier.
  A detector, or a gate class, would remove these crops instead of training on them.
