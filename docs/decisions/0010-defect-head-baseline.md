# 0010 — Defect head baseline: one linear probe per defect

- **Status:** proposed
- **Date:** 2026-11-04

## Context

Architecture milestone 3 and ADR [0005](0005-acceptance-criteria.md) §C need a quality
model: multi-label defects (data contract §2a) and a grade derived from them (§2b). The
maturity baseline (DINOv2-small CLS embeddings + a logistic probe) already caches an
embedding for every crop, and crop ids match across gold tasks.

The defect data is lopsided. Gold-eligible rows today:

| Defect | Rows | Sources |
|---|---|---|
| `rot` | ~723 | `strawberry_disease_2025`, `fresh_rotten_fruits_2022` (after review) |
| `mold` | ~683 | `strawberry_disease_2025`, `kamat_2025` (after review; held out) |
| `bruise`, `deformed`, `pest_damage`, `mechanical_damage` | **0** | none |

Many sources rate ripeness only. Their rows have `defects: null` (not assessed), which is
not the same as `[]` (assessed, no defects).

## Decision

1. **One probe per defect.** Each defect gets a standardised, class-balanced logistic
   regression on the frozen DINOv2-small crop embedding (`models/defect_probe.py`). The
   probes are independent, so a crop can carry several defects.
2. **Unassessed is never negative.** A probe is fitted only on rows whose defects were
   assessed. `defects: null` rows are left out of training and scoring.
3. **Train only what the data supports.** A defect is trained only if the training split
   has at least `min_positive` (30) positives **and** 30 negatives. Untrained defects are
   listed in the run's `metrics.json` and are never predicted. Today that leaves `bruise`,
   `deformed`, `pest_damage` and `mechanical_damage` untrained, so the model can't output
   `standard` grade yet.
4. **Thresholds come from the gold `val` split only**, by rules fixed here, before results:
   - `rot` and `mold` (the reject defects): the highest threshold that keeps val recall
     at least **0.90**. That leaves a margin above the 0.80 reject-recall bar of §C, since
     the phone set will be a different domain.
   - every other defect: the threshold that maximises val F1.
   - no val positives: 0.5.
   The test split and the proxy never move a threshold.
5. **Grade is derived, never learned:** `grade_from_defects` on the thresholded defects.
6. **Same holdout as maturity:** `kamat_2025` is never trained on (ADR 0007). Proxy scores
   rank models; they never give a §C verdict. `reject_verdict` is for the locked phone set.

## Consequences

- `scripts/train_quality.py` writes `models/quality_cls/<run_id>/` with per-defect
  precision and recall, reject-grade recall and the grade confusion matrix, all with
  group-bootstrap intervals, in-distribution and per proxy source.
- The rot and mold rows come mostly from field disease photos and DSLR shop shots. Expect
  a domain gap on phone photos; only the phone set measures it.
- Rot vs mold is a fine distinction (rubric: visible mycelium). Both make a `reject`, so
  confusing them costs nothing at the grade level. Their per-defect numbers will look worse
  than the reject recall.
- `standard`-grade defects need new labels (our own photos, or a new source) before they
  can be trained. Revisit this ADR when they exist.
- The phone-set verdict (`scripts/evaluate_phone_set.py`) doesn't call the defect head yet.
  Wiring it in is a follow-up.
