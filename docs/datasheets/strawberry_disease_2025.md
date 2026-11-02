# Datasheet: `strawberry_disease_2025`

| | |
|---|---|
| **Title** | Strawberry Disease Detection Dataset with Hybrid Data Augmentation |
| **Origin** | Extends Afzaal et al. (2021), *Sensors* |
| **URL / DOI** | https://data.mendeley.com/datasets/k2ptxmfjhj/1, 10.17632/k2ptxmfjhj.1 |
| **License** | **CC BY-NC 4.0**: non-commercial only. `commercial_use: false`, so `--commercial-only` gold builds exclude it |
| **Bronze pulls** | `2026-11-02` (test + valid only, per file) and `2026-11-02.2` (the whole-dataset ZIP, 545 MB; the one in use) |

## Motivation and collection

Instance segmentation of strawberry diseases on leaves, flowers and fruit, from field
photos. It ships two copies of one split:
- **traditional:** geometric and photometric augmentation.
- **hybrid:** the same plus 1,018 **DALL·E 3 images, train only, not marked**.

## Composition (verified from the COCO files)

| Split (traditional) | Images | Annotations |
|---|---|---|
| train | 3,672 | 6,921 |
| valid | 456 | 908 |
| test | 463 | 864 |

| COCO category | Our label |
|---|---|
| Anthracnose Fruit Rot | `defects: [rot]` |
| Gray Mold | `defects: [mold]`, `needs_review` |
| Powderly Mildew Fruit (sic) | `defects: [mold]` |
| Asymptomatic Strawberry | `defects: []` |
| Angular Leafspot, Leaf Spot, Powderly Mildew Leaf, Blossom Blight, `fruit` (id 0, unused) | dropped |

## What the audit found

- **The category names differ from the paper.** "Asymptomatic" replaces "Healthy", and
  "Powderly" is misspelled. The registry follows the files.
- **About half the sampled "Gray Mold" boxes are infected flowers or dried calyces**,
  not fruit, hence `needs_review`.
- **Augmented copies are in every split**: black-cornered rotations and hue shifts.
  Near-duplicate removal dropped 319 images. Copies rotated by arbitrary angles can't be
  caught by hashing. All 1,980 images kept come from distinct originals (by Roboflow
  filename id).
- The hybrid copy is excluded entirely, so no synthetic images are used.
- **No maturity labels.** It's used only for the defect head.

## Uses

Mold and rot examples for the quality head, non-commercial only. Its hue-shifted
augmentations make it unsuitable for any color or ripeness signal.
