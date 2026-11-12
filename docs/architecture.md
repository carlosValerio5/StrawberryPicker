# Architecture

## Pipeline

```
image ──► [1] detect strawberries ──► per-fruit crops ──► [2] maturity head  ──► unripe/ripe/overripe
                                                      └─► [3] defect head    ──► multi-label defects ──► grade
```

1. **Detection / localization.** Find each fruit and output its box (and optionally a mask).
   Skip this step for single-fruit, centered images. Until a detector is trained, inference
   uses zero-shot OWLv2 ([ADR 0009](decisions/0009-inference-detector.md), `strawberry_picker.inference`).
2. **Maturity classification.** Single label, ordinal (`docs/data-contract.md` §1).
3. **Defect classification.** Multi-label (§2a). The grade is derived by rule, not learned.

Heads 2 and 3 can share one frozen backbone with two lightweight heads. That's
the recommended first baseline because it trains fast on small labeled sets.

**Calibration.** ADR 0005 §B3 needs ECE ≤ 0.10, and the first baseline scored 0.130 on
the Kamat proxy. The maturity probe is therefore temperature-scaled
(`models/calibration.py`, Guo et al. 2017): one scalar `T`, fitted by negative
log-likelihood on the gold **val** split of the training sources. It's never fitted on
test, a held-out proxy source or the phone set. `T` is stored in the run's
`calibration.json`. Argmax predictions don't change, so recall and kappa stay the same;
only the confidences move. `calibration: none` in `configs/maturity_cls.yaml` turns it off.

## Candidate open-source models

Check licenses before adopting. Code and weights can have different licenses.

| Role | Model | License | Notes |
|------|-------|---------|-------|
| Backbone (frozen features) | DINOv2 (ViT-S/B) | Apache-2.0 | Strong linear-probe baseline on small data |
| Backbone (fine-tune) | ConvNeXt / EfficientNet via `timm` | Apache-2.0 (most weights) | Cheap to fine-tune, edge-friendly |
| Zero-shot baseline | SigLIP / CLIP | Apache-2.0 / MIT | "a photo of an unripe strawberry" before any training |
| Detector | RT-DETR (HF `transformers`) | Apache-2.0 | Permissive end-to-end detector |
| Detector | YOLO (Ultralytics) | **AGPL-3.0** | Great tooling, but the copyleft terms need an ADR first |
| Auto-labeling | OWLv2 / Grounding DINO | Apache-2.0 | Bootstrap boxes from text prompts, then fix by hand |
| Masks | SAM 2 | Apache-2.0 | Precise fruit masks for surface-area color ratios |
| Crop isolation | SAM (`facebook/sam-vit-base`, HF `transformers`) | Apache-2.0 (code and weights) | Box-prompted masks for the `segment` crop policy (ADR 0011); `data.crops.sam_segmenter` |
| Label-error audit | Confident learning, own code in `audit/label_issues.py` | — | `cleanlab` is **AGPL-3.0**, so not a dependency; the method is ~30 lines (Northcutt et al. 2021) |

## Suggested milestones

1. Ingest 1–2 public strawberry ripeness datasets into bronze, then build silver and `maturity_cls_v0.1` gold.
2. Zero-shot SigLIP baseline, then a DINOv2 linear probe. Report macro-F1 and the confusion matrix.
3. Add the defect head once defect labels exist. Baseline: one DINOv2 linear probe per defect
   with val-chosen thresholds (ADR [0010](decisions/0010-defect-head-baseline.md),
   `scripts/train_quality.py`). Only `rot` and `mold` have data so far.
4. Add detection for multi-fruit / in-field images.
