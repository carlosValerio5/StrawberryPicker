# TDD evidence: inference pipeline (ADR 0009)

- **Source plan:** next-step 2 ("detect → crop → classify") agreed in chat on 2026-11-04.
- **Branch:** `feat/inference`.

## User journeys

1. As the owner, I point one command at a folder of phone photos and get maturity for
   **every berry** in each photo, not one label for the whole photo.
2. As the owner, I see which box got which label on an annotated copy of the photo.
3. As anyone, I can't run predictions on the locked phone set by accident.
4. As an ML engineer, I can plug in the defect head later without changing the pipeline.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| One prediction per kept box, best first; score threshold + NMS from `proposals`; never-trained classes get probability 0 | `c80d7ac` | `bd1da9b` |
| Crops use gold's padded square at the given padding; a custom `crop_fn` replaces it | `c80d7ac` | `bd1da9b` |
| No detections → empty list, embedder never called | `c80d7ac` | `bd1da9b` |
| `heads` add fields per berry (defect-head hook); a head with the wrong count fails | `c80d7ac` | `bd1da9b` |
| Images decoded like silver (EXIF orientation, RGB); HEIC and unreadable files → clear error | `c80d7ac` | `bd1da9b` |
| Locked phone set refused: the folder itself, a file inside, a parent folder, a symlink to it | `c80d7ac` | `bd1da9b` |
| Padding resolved from the run config, else the gold dataset card, else an error asking for `--padding` | `c80d7ac` | `bd1da9b` |
| Run folder loading; per-image records where a bad file doesn't stop the batch; annotated copies | `c80d7ac` | `bd1da9b` |

17 new tests, all with a fake detector, embedder and probe. Full suite: 224 passed, coverage
99 %. The real OWLv2 and DINOv2 loaders are `# pragma: no cover`, as before.

## Evidence that isn't a unit test

`scripts/predict.py` was run against a temporary data dir:
- `predict.py <data>/bronze` → refused: "is or contains the locked phone test set".
- `predict.py <folder>` with no model run → "no model run yet".
- `predict.py missing.jpg` → "not found".

## Not done

- No end-to-end run on real photos: it needs the `ml` extra, the OWLv2 and DINOv2 weights
  and a trained run in `models/maturity_cls/`.
- The detector's score threshold (0.2) isn't measured on our photos, and detector quality
  has no gate yet (ADR 0009, Consequences).
- No defect head plugged in. The `heads` hook is ready for it.
- `train_maturity.py` doesn't write `crop_padding` into the run config yet, so older runs
  fall back to their gold dataset card. That's left for the crop-policy branch, which owns
  crop settings.

## Round 2: HEIC, crop policy, calibrated probe, defect head

Tests written first (RED: import error for `resolve_policy`, then the new behaviours failing),
then the code (GREEN). All tests use fakes; none reads real data or weights.

| Behavior | Test |
|---|---|
| HEIC goes through `images.open_rgb` (no rejection); a corrupt HEIC gives a per-image error | `test_load_image_decodes_heic_through_open_rgb`, `test_load_image_corrupt_heic_gives_a_clear_error` |
| Crops are cut by `crop_fruit` under the run's policy; `mask` paints the other kept berries, not the target | `test_predict_image_cuts_crops_with_the_policy_and_masks_neighbours` |
| `keep` leaves neighbours in; `drop` crops every berry like `keep` at inference | `test_predict_image_keep_policy_leaves_neighbours_in_the_crop`, `test_predict_image_drop_policy_crops_every_berry_like_keep` |
| Policy resolution: run config, then gold `crop_policy.json`, then legacy policy for pre-ADR 0011 releases | `test_resolve_policy_prefers_run_config_then_gold_release` |
| Legacy `crop_padding` in an old run config is still honoured | `test_legacy_crop_padding_in_run_config_is_still_honoured` |
| `calibration.json` gives a temperature-scaled probe and softened probabilities; no file, raw probabilities | `test_load_predictor_applies_the_run_calibration`, `test_load_predictor_without_calibration_file_keeps_raw_probabilities` |
| Defect head from a saved quality run adds `defects` and `grade` to every berry | `test_load_predictor_adds_defects_and_grade_from_the_quality_run` |
| No quality run: no defect fields | `test_load_predictor_without_quality_run_adds_no_defect_fields` |
| A quality run on another backbone or crop policy is refused | `test_quality_run_must_share_the_maturity_backbone_and_crops` |

Not done in this round:
- No run on real photos. The OWLv2 and DINOv2 loaders were exercised only by loading an
  existing maturity run and a non-image file; no detector or classifier output was checked.
- No quality run exists yet in `models/quality_cls/`, so the defect head is tested with a
  fake probe only. `scripts/predict.py` picks up the newest one automatically once it exists.
- The detector score threshold (0.2) and detector quality still have no measurement (ADR 0009).
