# 0008 — Isolate the target berry in each crop

- **Status:** proposed. The policy is chosen by the comparison below, before any
  phone-set result exists (ADR 0005 §F).
- **Date:** 2026-11-04

## Context

Detection sources (Strawberry-DS, Yurdakul, Kamat) photograph clusters, and the phone
test set is shot as multi-berry punnets and plates (ADR 0007). `crop_box` cuts a square
1.3× the box's longest side, so a crop of one berry often shows part of its neighbours,
which can be at another ripeness stage. The DINOv2 CLS embedding pools the whole crop,
so the probe sees a mix of both berries.

Nobody has measured how often this happens or whether it explains the errors. The proxy
errors noted so far are pale, partly pink berries, which is a coverage gap (ADR 0007),
not necessarily neighbour leakage.

## Options

All of them are implemented behind `CropPolicy` (`src/strawberry_picker/data/crops.py`),
set by `gold.crop_padding` and `gold.crop_neighbours` in `configs/data.yaml`:

| Option | What it does | Cost / risk |
|---|---|---|
| `keep` (status quo) | Crop as-is | Neighbours leak in |
| lower padding (e.g. 0.05) | Less context, fewer neighbours | Loses the calyx and shape cues; doesn't help with touching berries |
| `mask` | Paint other annotated boxes grey (`fill`), except inside the target's own box | Needs every berry boxed. Unboxed berries still leak; grey patches are a pattern the model can learn, so test crops must be masked the same way |
| `drop` | Leave training crops with neighbour overlap > `max_overlap` out of gold (listed in `dropped.parquet`) | Removes crowded examples, and those are exactly the phone-set conditions. Training only: at evaluation and inference every berry is cropped as `keep` |
| `segment` | Keep only the target's SAM mask and fill the rest | Removes neighbours **and** background. Background is a strong shortcut (source probe 0.979), so this may help on its own. Needs the `ml` extra and ~375 MB of weights (SAM, Apache-2.0). Only the gold build passes a segmenter so far |

## Pre-registered comparison

1. **Measure first:** run `scripts/crop_overlap.py` (silver only) and `scripts/dataset_audit.py`.
   Record the share of crops with overlap > 0, > 10 % and > 25 % per source, and the
   out-of-fold error rate by overlap bucket.
   - If the error rate doesn't climb with overlap, keep `keep`, and close this ADR as
     rejected with the numbers.
2. **Otherwise compare** `keep`, padding 0.05, `mask`, `drop` (max_overlap 0.10) and, if the
   weights are available, `segment`. Use one gold release per candidate, all with the same
   seed and splits.
   - Rank them by macro-F1 on the **Kamat proxy** (held out, ADR 0007), with its
     group-bootstrap interval.
   - Report leave-one-source-out macro-F1 (ADR 0005 §E) alongside.
   - Pick the best. Prefer the simpler policy when intervals overlap. Order of simplicity:
     keep < padding < drop < mask < segment.
3. **Never use the phone set** to choose. `scripts/evaluate_phone_set.py` reads the chosen
   policy from the run's gold release (`crop_policy.json`) and crops the locked set the same
   way. No policy changes after the first phone-set result for that model version.

## Decision

Pending the comparison above. Until then, `configs/data.yaml` stays at `keep` with
padding 0.15. That reproduces `maturity_cls_v0.1` exactly.

## Consequences

- Every gold release now records `neighbour_bboxes`, `neighbour_overlap` and `crop_policy`
  per row, plus `crop_policy.json`. Training (`cached_embeddings`) reads the policy from the
  manifest, and embedding caches are keyed by policy. A masked release can't reuse unmasked
  vectors.
- The chosen policy ships as a new release (`maturity_cls_v0.2`). v0.1 is immutable. Cutting
  it needs the real silver layer.
- Inference crops through `crop_fruit` with `policy.for_evaluation()`, using the detector's
  other boxes as neighbours.
- Non-goal: images labelled per image that show a pile of berries (Fresh/Rotten). They
  have no boxes, so no policy can isolate a berry in them. That's a separate cleaning step.
