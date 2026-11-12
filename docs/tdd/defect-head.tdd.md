# TDD evidence: defect head baseline (ADR 0010)

- **Source plan:** "next steps that need no manual work", item 6 (defect head), approved in
  chat on 2026-11-04.
- **Branch:** `feat/defect-head`.

## User journeys

1. As an ML engineer, I train one probe per defect from the quality gold release and get
   per-defect precision/recall and reject-grade recall, in-distribution and on the Kamat
   proxy, without waiting for the phone set.
2. As the owner, I can see which defects the model can't predict yet, and why (too few rows).
3. Later, the phone-set verdict calls `evaluate_quality` + `reject_verdict` for ADR 0005 §C.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| `defects: null` is unassessed, `[]` is clean; unknown defect names fail | `716738a` | `46145fc` |
| Thresholds: recall-target (highest that meets it), max-F1, 0.5 without val positives, bad rule fails | `716738a` | `46145fc` |
| Rare defects stay untrained and listed; unassessed rows are neither positives nor negatives | `716738a` | `46145fc` |
| Thresholded defects → grade by `grade_from_defects` | `716738a` | `46145fc` |
| Thresholds depend on val only (scrambling test rows changes nothing) | `716738a` | `46145fc` |
| §C report: per-defect P/R and reject recall with group intervals, no verdict, JSON-ready | `716738a` | `46145fc` |
| `reject_verdict` uses the pre-registered 0.80 bar | `716738a` | `46145fc` |
| Held-out sources are never trained on; proxy scored on all its assessed rows | `716738a` | `46145fc` |
| Head round-trips through `linear_probe.save_run` / `load_probe` | `716738a` | `46145fc` |
| `configs/quality_cls.yaml` is valid | `716738a` | `46145fc` |
| Report for a set with no assessed rows (e.g. an empty proxy) | added with the fix in `46145fc` | `46145fc` |

All tests use synthetic embeddings; no data, weights or GPU.

## Not done

- **No real run yet.** Needs `data/gold/quality_cls_v0.1` (`scripts/build_gold.py quality_cls 0.1`)
  and the cached DINOv2 embeddings (`ml` extra). No numbers exist, so the ADR stays proposed.
- **Kamat proxy is probably empty** for quality: Kamat's only defect label (`rotten` → mold)
  is `needs_review` until the overripe task is labeled and landed.
- **Phone-set wiring:** `evaluation/phone_set.py` doesn't call the defect head. Follow-up:
  embed the phone crops, `head.predict`, then `evaluate_quality(predicted, truth, photo_ids,
  head.defects)` and `reject_verdict(report)` against the adjudicated defects.
- `bruise`, `deformed`, `pest_damage`, `mechanical_damage` have no training rows.
