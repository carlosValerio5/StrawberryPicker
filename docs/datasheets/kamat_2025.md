# Datasheet: `kamat_2025`

| | |
|---|---|
| **Title** | Comprehensive Dataset on Ripening Stages of Strawberries and Avocados: From Unripe to Rotten |
| **Origin** | Kamat et al., *Data in Brief* (2025) |
| **URL / DOI** | https://data.mendeley.com/datasets/zysvgmxcyz/1, 10.17632/zysvgmxcyz.1 |
| **License** | CC BY 4.0 |
| **Bronze pull** | `2026-11-02`, the whole-dataset ZIP (2.11 GB) |

## Composition (verified)

- **What's in the deposit:**
  - 758 full-resolution phone JPEGs
  - 1,332 YOLO label files
  - a bundled **YOLOv5 code checkout** (with `.git`), which is excluded
- **No `data.yaml`.** 8 class indices are used.
- **574 label files have no image** in the deposit, so 43 % of the images are missing.

| Index | Our name | Boxes with an image | Our label | Contact-sheet check |
|---|---|---|---|---|
| 0 | unripe | 112 | `unripe`, `needs_review` | **about 1 in 4 sampled boxes is a flower or bud** |
| 1 | partially ripe | 79 | `unripe` | pale, partly pink. 2 of 40 sampled were rotten avocados |
| 2 | ripe | 69 | `ripe` | correct |
| 3 | rotten | 88 | `overripe` + `mold`, `needs_review` | mostly white fuzzy mold (rubric: `mold`, not `rot`) |
| 4–7 | avocado_4 … avocado_7 | 576 | dropped | 4 and 7 viewed: avocados |

## What the audit found

- Many photos are **hand-held phone shots of picked fruit**, the closest public match
  to [our target domain](../target-domain.md).
- Some backgrounds were **replaced with flat colours in software**, a shortcut risk.
- **Leave-one-source-out:** held-out macro-F1 0.876. Adding Kamat raised the held-out
  Strawberry-DS score from 0.792 to 0.873.
- Confident learning flagged 5 of 145 maturity crops. See `reports/dataset-audit/review/`.

## Uses

Ripe and partly-ripe examples in the target domain. The overripe and mold candidates
(`rotten`) count only after review.
