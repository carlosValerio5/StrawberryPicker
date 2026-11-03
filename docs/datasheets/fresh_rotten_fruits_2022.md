# Datasheet: `fresh_rotten_fruits_2022`

| | |
|---|---|
| **Title** | Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality |
| **URL / DOI** | https://data.mendeley.com/datasets/bdd69gyhv8, 10.17632/bdd69gyhv8.1 |
| **License** | CC BY 4.0 |
| **Bronze pull** | `2026-11-02`: `Original Image.zip`, 2.79 GB. sha256 matches Mendeley's. `Augmented Image.zip` is not used |

## Composition (verified)

- 16 folders (8 fruits × fresh/rotten), 200 images each. Only the two strawberry folders
  are mapped.
- Image-level labels with no boxes, so the whole photo is the crop.
- One TIFF reads with Pillow's "Truncated File Read" warning but decodes.

| Folder | Images | Our label | Contact-sheet check |
|---|---|---|---|
| FreshStrawberry | 200 | `defects: []` | Means *not rotten*: a few sampled berries are clearly deformed. About 1 in 4 images is a **pile of many berries** |
| RottenStrawberry | 200 | `overripe` + `rot`, `needs_review` | Shrivelled, dried, dark; past peak. Piles as above |

## What the audit found

- Picked fruit on plain white paper, which is close to a phone photo in a kitchen.
- Confident learning flagged 0 of 199 defect crops. That doesn't mean the labels are
  clean: piles and deformed "fresh" berries look like no *other* class (ADR 0006).

## Uses

The largest pool of past-peak strawberries, 199 overripe candidates. They count only
after review. Piles should be excluded or boxed before training a per-fruit classifier.
