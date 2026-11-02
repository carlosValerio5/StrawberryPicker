# Datasheet: `strawberry_ds_2023`

| | |
|---|---|
| **Title** | Strawberry-DS: annotated strawberry fruit images at various developmental stages |
| **Origin** | Elhariri, El-Bendary, Saleh. *Data in Brief* (2023) |
| **URL / DOI** | https://data.mendeley.com/datasets/z6dtfdpzz8/1, 10.17632/z6dtfdpzz8.1 |
| **License** | CC BY 4.0 (commercial use allowed with attribution) |
| **Bronze pull** | `2026-11-02`: `Strawberry (2).zip`, 293 MB. sha256 matches Mendeley's |

## Motivation and collection

Ripening-stage detection in an Egyptian greenhouse. 3840×2160 photos with several
fruits each, exported from Roboflow (YOLO format, "no augmentation applied").

## Composition (verified)

247 images, 1,083 boxes after silver: train 173, valid 50, test 27 images.

| Class (index in `data.yaml`) | Boxes | Our label |
|---|---|---|
| Early-Turning (0) | 34 | `unripe` |
| Green (1) | 450 | `unripe` |
| Late-Turning (2) | 56 | `ripe`, `needs_review` |
| Red (3) | 225 | `ripe` |
| Turning (4) | 42 | `unripe` |
| White (5) | 276 | `unripe` |

## What the audit found

- **The class order is alphabetical in `data.yaml`, not the paper's stage order.** With
  the paper's order, red berries became `unripe` and white ones `ripe`. Fixed and
  confirmed on contact sheets.
- **About 6–12 % of sampled boxes show no visible fruit** (soil or leaves). The
  label-error review should confirm the rate.
- `Late-Turning` is mostly ≥ ¾ pink or orange, so it leans `ripe` under the rubric. It
  stays `needs_review` because a few are borderline.
- `Turning` (⅓–⅔ colored) gives us real borderline `unripe` examples, which Riseholme lacks.
- One original was saved twice and appears in two of the source's own splits.

## Uses

Our main source for the ¾ boundary. On-plant greenhouse photos, so the domain is far
from hand-held phone photos.
