# Datasheet: `owner_phone_2026`

| | |
|---|---|
| **Title** | Owner iPhone photos (project capture, 2026-11-10) |
| **URL / DOI** | None. Project's own capture; the photos stay out of git |
| **License** | Project-owned. Commercial use OK |
| **Bronze pull** | `2026-11-12`: `Fresas-20261112T065557Z-1-001.zip`, landed as `data/bronze/owner_phone_2026/2026-11-12/` |
| **Registry** | `configs/sources.yaml`, `owner_phone_2026` |

## Composition (verified)

- Photos taken by the project owner on their own iPhone, stored as HEIC. Full size is
  4032×3024. Two photos are 5712×4284.
- One session on 2026-11-10, 09:00–11:00. One photo is dated 2026-11-08.
- The owner sorted the photos into two folders:
  - `Fresas/overripe`: 94 photos, mostly one berry each.
  - `Fresas/defective ` (57): the folder name has a trailing space, and the registry
    key keeps it.
- 3 file names appear in both folders (IMG_1622, IMG_1638, IMG_1639).

| Folder | Photos | Our label | Notes |
|---|---|---|---|
| `overripe` | 94 | `overripe` (maturity) | Sorted by the owner. Defects are not assessed |
| `defective ` | 57 | `needs_review` | Mixed defect types. The owner labels maturity and each defect per berry in `scripts/label.py`. Task: `reports/labeling/needs-review_owner_phone_2026.csv`, 58 crops |

## Boxes and rows

- OWLv2 box proposals (ADR [0008](../decisions/0008-box-proposals-for-image-labels.md),
  `min_score` 0.3): 279 boxes on 151 photos.
- Silver has 147 rows. 144 of them got a tight single-berry box. 2 overripe photos were
  held as piles and set to `needs_review`.
- Silver rows: 89 overripe ready, 2 overripe held, 56 defective pending review.
- Rejected: 3 exact duplicates and 1 near-duplicate.
  - The 3 exact duplicates are IMG_1622, IMG_1638 and IMG_1639. Each is in both folders.
    The defective copy is kept, so the owner's review sets both maturity and defects.
  - The near-duplicate is defective IMG_1486, which is ≈ IMG_1485.

## What the audit found

- HEIC is decoded with `pi-heif` (BSD-3-Clause, decode only). See
  [architecture.md](../architecture.md). EXIF orientation is applied before crops, as
  for every source.
- Of the 147 silver rows, 144 give one tight crop each. The 2 overripe photos that
  show piles are held out as `needs_review`.

## Caveats

1. **Single session.** Photos of the same berries can fall into different gold splits
   (`group_by: image`). In-distribution scores on this source are therefore optimistic.
   The cross-source Kamat proxy is the honest number.
2. **One labeler.** Every label is per photo, from one person. There is no second
   labeler.
3. **No unripe or ripe photos.** This source adds overripe and defects only.
4. **Not the locked phone test set.** This is a training source. It is not
   `bronze/phone_testset` (ADR [0007](../decisions/0007-lean-phone-test-set.md)), so it
   can't give an ADR 0005 verdict.

## Uses

It is the third overripe setting, and it is the target domain: hand-held phone photos
at close range. Overripe from Fresh/Rotten (shrivelled, dried) and from Kamat (red with
white mold) do not transfer to each other (findings
[#17 and #18](../datasets.md#part-3-reviewed-overripe-and-the-first-3-class-model-2026-11-03)).
This source adds a third, phone-camera overripe setting. The defective photos wait for
the owner's per-berry review before they can go to gold.
