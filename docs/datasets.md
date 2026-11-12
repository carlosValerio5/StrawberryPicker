# Datasets

The public datasets we can train on, how each maps to our classes
([data-contract.md](data-contract.md)), and what has to be done to them first.
The machine-readable version of this page is [`configs/sources.yaml`](../configs/sources.yaml).

Research done 2026-11-02. Counts are **original** images, before the dataset authors'
own augmented copies.

## Catalog

| Source (`configs/sources.yaml`) | Gives us | Size | Setting | License | Status |
|---|---|---|---|---|---|
| [`riseholme_2021`](https://github.com/ctyeong/Riseholme-2021) | unripe 2,406 · ripe 462 · anomalous 153 (+ 499 occluded, dropped) | single-fruit crops, ~63×66 px | UK polytunnels, robot camera | **not stated** | enabled |
| [`yurdakul_2026`](https://www.kaggle.com/datasets/mahyeks/multi-class-strawberry-ripeness-detection-dataset) | unripe 776 · ripe 273 · semi-ripe 120 (review) | 566 images, 1,169 boxes | Türkiye greenhouses | not stated (paper CC BY 4.0) | needs verify |
| [`strawberry_ds_2023`](https://data.mendeley.com/datasets/z6dtfdpzz8/1) | 6 stages → unripe / ripe | 247 images, multi-fruit | Egypt greenhouse, 3840×2160 | CC BY 4.0 | **enabled** (audited) |
| [`kamat_2025`](https://data.mendeley.com/datasets/zysvgmxcyz/1) | unripe 114+238 · ripe 189 · rotten 170 | ~711 strawberry samples | India, field, **pastel backgrounds** | CC BY 4.0 | **enabled** (audited; 43 % of images missing from the deposit) |
| [`fresh_rotten_fruits_2022`](https://data.mendeley.com/datasets/bdd69gyhv8) | no defects 200 · rot 200 | 512×512 | shops and fields, DSLR | CC BY 4.0 | **enabled** (audited) |
| [`owner_phone_2026`](datasheets/owner_phone_2026.md) | overripe 94 · defective 57 (review) | 151 photos, 279 boxes | owner's iPhone, hand-held, HEIC | project-owned | **enabled** (defects pending review) |
| [`strawberry_disease_2025`](https://data.mendeley.com/datasets/k2ptxmfjhj/1) | mold, rot, healthy (fruit classes only) | 2,459 originals in 8 classes | field, COCO polygons, **1,018 DALL·E images** | **CC BY-NC 4.0** | **enabled**, test + valid only (audited) |

"Needs verify" means the label mapping is written, but the source stays `enabled: false`
until someone downloads it and confirms the class order and file layout. A wrong YOLO
class order would silently mislabel every box.

**Rejected:**
- **StrawDI_Db1:** no ripeness labels, non-commercial, no redistribution. Useful later
  for a detector.
- **Strawberry Grading:** grades size, not quality, and has 130 images.
- **BRIN good/bad (2018):** unclear license.
- **TrainingDataPro:** a commercial sample.
- **Roboflow Universe:** uneven quality and heavy duplication. Only add it after
  near-duplicate removal.

## Download inventory (checked 2026-11-02 from the repositories' metadata APIs)

Sizes come from the GitHub and Mendeley APIs. Nothing has been downloaded yet. Mendeley's
file listing stops at 1,000 files per folder, so counts shown as "1,000" are lower bounds.

| Source | What to download | Size | Findings from the file listing |
|---|---|---|---|
| `riseholme_2021` | GitHub ZIP of `main` | ~27 MB | The repo has no license file |
| `strawberry_ds_2023` | `Strawberry (2).zip` | 293 MB | A single archive. Class order is checked after unpacking |
| `strawberry_disease_2025` | the `traditional-augmentation` folder plus `README.txt` | ~400 MB (the listing showed ~147 MB because train was cut off at 1,000 files; its COCO file lists **3,672** images) | Two copies of the same originals. The *hybrid* copy adds DALL·E 3 images, which are only in train and **aren't marked**, so it's excluded entirely. The traditional copy still has geometric and photometric augmentations. Filenames are Roboflow exports (`<id>_jpg.rf.<hash>.jpg`) |
| `kamat_2025` | the `images/` and `labels/` folders | ~2.1 GB | 758+ full-resolution JPEGs (~2.8 MB each), plus a bundled **YOLOv5 code checkout** (excluded). **No dataset `data.yaml`**, so the class order has to be confirmed on contact sheets |
| `fresh_rotten_fruits_2022` | `Original Image.zip` | 2.79 GB | Holds all 8 fruits; we use about 400 strawberry images. `Augmented Image.zip` (803 MB) is skipped |
| `yurdakul_2026` | the Kaggle dataset ZIP | see Kaggle | Needs a Kaggle login, so the owner downloads it by hand |

## Audit findings (2026-11-02)

Each source's labels were checked against its own files and against contact sheets
(`scripts/contact_sheets.py`, ~50 random crops per label, judged with the
[rubric](labeling/rubric.md)). Pulls checked: `riseholme_2021`, `strawberry_ds_2023`,
and `strawberry_disease_2025` (test and valid only), all from 2026-11-02.

| # | Finding | Evidence | Action |
|---|---|---|---|
| 1 | **Strawberry-DS class order was wrong in our registry.** The export's `data.yaml` is alphabetical (`Early-Turning, Green, Late-Turning, Red, Turning, White`), not the paper's stage order. Enabled as written, every `Red` box would have become `unripe` and every `White` box `ripe` | `data.yaml`. Under the old order the "Red" sheet showed white berries and the "Turning" sheet showed red ones | `class_names` fixed and source enabled |
| 2 | **Near-duplicate removal merged different berries.** The 64-bit dHash matched 71 Riseholme crops, and 10 of those pairs were ripe↔unripe. On ~60 px blurry crops, different fruit hashes almost the same. Distances on a 256-bit hash were 47–94 bits; real copies sit at ≤ 13 | `near_dup_pairs` review. All 19 matches left after the fix are true flips or hue-shifted copies | dHash widened to 16×16, threshold 24/256 (`configs/data.yaml`). Regression test added |
| 3 | **Disease COCO names differ from the paper:** `Asymptomatic Strawberry` (not "Healthy"), `Powderly Mildew Fruit/Leaf` (sic), plus an unused `fruit` category | `_annotations.coco.json` | Registry renamed |
| 4 | **"Asymptomatic Strawberry" is healthy fruit**, mostly green or white | contact sheet | `needs_review` removed |
| 5 | **About half of the sampled "Gray Mold" boxes are infected flowers or dried calyces**, not fruit | contact sheet | `needs_review: true`. It stays out of gold until reviewed |
| 6 | **The disease "traditional" copy has augmented images in every split**: rotations with black corners, hue jitter. Hue jitter matters for ripeness. This source has no maturity labels, but its defect crops carry shifted colors | contact sheets | Noted. Dihedral dHash only catches 90° rotations and flips |
| 7 | **About 6–12 % of sampled Strawberry-DS boxes show no visible fruit** (soil, leaf) | `Red`, `Late-Turning` sheets | Left for the label-error audit (ADR 0005 D) |
| 8 | **Riseholme has no borderline fruit.** `Unripe` is white or green, `Ripe` is fully red. It can't teach the model where the ¾ line falls; Strawberry-DS `Turning`/`Late-Turning` and our own photos have to | contact sheets | None. This is about coverage, not an error |
| 9 | **Disease `train/` images are missing from our pull.** Mendeley's file listing stops at 1,000 files per folder | COCO lists 3,672 train images. 999 were listed | `train/**` excluded until a full pull. Needs the whole-dataset ZIP |

Silver after part 1 of the audit (3 sources): **5,038 fruit rows from 3,556 images**.
- Maturity rows that can go to gold: **unripe 3,065, ripe 686, overripe 0**.
- Defect rows that can go to gold: **rot 176, mold 173**. The rest are healthy or unassessed.
- 325 rows are `needs_review`.

### Part 2: Kamat, Fresh/Rotten, the full disease set, and statistics (2026-11-02)

Datasheets for every source are in [`docs/datasheets/`](datasheets/).

| # | Finding | Evidence | Action |
|---|---|---|---|
| 10 | **Kamat ships no class list and uses 8 indices.** 0–3 are strawberry, as named. 4 and 7 were checked as avocado | contact sheets; box counts match the paper | 8 names registered, avocados dropped |
| 11 | **About 1 in 4 sampled Kamat "unripe" boxes is a flower or bud** | contact sheet | `unripe` → `needs_review` |
| 12 | **Kamat "rotten" is mostly white fuzzy mold** | contact sheet | mapped to `overripe` + `mold`, `needs_review` |
| 13 | **574 of Kamat's 1,332 label files have no image** in the deposit | file listing | none; 342 fruit rows are usable |
| 14 | **Fresh/Rotten labels are per image, and many images are piles of berries** (about 1 in 4 on the contact sheets; OWLv2 later counted 46 %, see ADR 0008). "Fresh" means not rotten; some are deformed | contact sheets | Fresh kept as `defects: []` (caveat noted); Rotten → `overripe` + `rot`, `needs_review` |
| 15 | **Disease `train/` is now complete** (pull `2026-11-02.2`, the whole-dataset ZIP). 319 augmented copies were removed | silver rejections | `exclude` for train removed |
| 14a | **Follow-up to 14: piles are now found by a detector, then confirmed by a person.** `scripts/flag_multi_fruit.py` counts berries (OWLv2) in every whole-image row; `make_label_task.py multi-fruit` sends those with 2+ berries to `label.py --multi-fruit`. An agreed `s` (several berries) keeps the source label but sets `needs_review`, so gold skips the image | `reports/dataset-audit/multi_fruit/report.md` (not run yet: needs the ml extra and weights) | Run it, then label the task |
| 16 | **Confident learning misses boxes that aren't fruit.** It flagged 1.3 % of Strawberry-DS, while the contact sheets showed 6–12 % empty boxes | audit report vs contact sheets | **ADR [0006](decisions/0006-random-sample-error-rate.md) (proposed)**: estimate error rates from a random sample |

**Statistical audit** (`scripts/dataset_audit.py`, DINOv2-small embeddings, linear
probes, group-aware CV; full report in `reports/dataset-audit/report.md`):

- **Sources are almost perfectly identifiable:** balanced accuracy 0.979 against a
  chance level of 0.200. Any shortcut a source offers (backgrounds, camera, crop size)
  is easy to learn, so evaluation has to be on held-out sources and on our own phone set.
- **Leave-one-source-out maturity macro-F1:**
  - in-distribution: 0.952
  - held-out Riseholme: 0.807 (−0.145, just under the 0.15 line in ADR 0005 §E)
  - held-out Strawberry-DS: 0.873
  - held-out Kamat: 0.876

  Adding Kamat raised the held-out Strawberry-DS score from 0.792 to 0.873.
- **The learning curve is flat:** macro-F1 0.940 with 10 % of the training data and
  0.940 with all of it. More of these datasets won't help; *different* data will (our
  phone photos and overripe fruit).
- **Confident learning flagged:**
  - Riseholme: 26 of 2,724
  - Strawberry-DS: 13 of 1,027
  - Kamat: 5 of 145
  - Disease: 1 of 4,519
  - Fresh/Rotten: 0 of 199

  Most Riseholme flags are pink berries labelled `Unripe`, which are `ripe` under the
  USDA rubric. Lists are in `reports/dataset-audit/review/`. Random samples for ADR 0006
  are in `reports/dataset-audit/random_sample/`.

**Silver now (5 sources):** 9,781 fruit rows from 5,672 images.
- Gold-eligible maturity: **unripe 3,144, ripe 752, overripe 0**.
- 287 overripe candidates (Fresh/Rotten 199, Kamat 88) are waiting for review.
- Gold-eligible defects: **rot 723, mold 683**.

### Part 3: reviewed overripe and the first 3-class model (2026-11-03)

All 287 overripe candidates were reviewed on the phone labeler by one labeler. Landed as
`bronze/human_review/2026-11-02`.

- **Verdicts:** 278 confirmed overripe (Fresh/Rotten 196, Kamat 82), 9 actually ripe, 0 not
  a strawberry, 0 unsure.
- **Defects marked:** rot 101, mold 93, mechanical damage 37, bruise 6, pest damage 6, and
  **deformed 143**. That last count needs a second look: the rubric's `deformed` means
  misshapen growth, not shrivelling with age.
- **Gold `maturity_cls_v0.2`:** unripe 3,144, ripe 761, **overripe 278**.

**Finding #17: overripe is learned as a source shortcut.** All 139 overripe *training*
examples come from Fresh/Rotten (whole photos on white paper, often piles), which has
almost no other class.

| Run | Overripe recall, in-distribution | Overripe recall, Kamat proxy (never trained on) | Proxy QWK | Proxy ECE |
|---|---|---|---|---|
| `20261103T023318Z` (v0.1, no overripe) | n/a | n/a | 0.674 | 0.130 |
| `20261103T040357Z` (v0.2) | 1.00 (23/23) | **0.20** [0.11, 0.32] (16/82) | **0.429** | **0.339** |

On the proxy, 61 of 82 overripe berries were called ripe. The in-distribution 1.00 is
same-source and shouldn't be cited on its own. Overripe needs examples from more than one
setting before any model's overripe numbers mean anything.

**Finding #18: cropping didn't fix it, because the two sources' "overripe" look different.**

OWLv2 single-berry crops (ADR [0008](decisions/0008-box-proposals-for-image-labels.md),
gold `maturity_cls_v0.3`) removed the white-paper background and the piles. That left 62
Fresh/Rotten overripe crops (38 of them in train). Then each source was held out in turn:

| Trained with overripe from | Held out | Overripe recall on the held-out source | Misses called |
|---|---|---|---|
| Fresh/Rotten (38; run `20261103T050445Z`) | Kamat | 0.15 [0.08, 0.25] | ripe (64/82) |
| Kamat (77) | Fresh/Rotten | 0.27 | ripe (44/62) |

Each source shows one narrow kind of overripe: Fresh/Rotten berries are shrivelled and
dried, while Kamat's are red with white fuzzy mold. A probe that relies on color calls
both kinds "ripe". The fix is overripe examples from more settings, not more cropping.
Whether the decay signs (mold, rot, shrivelling) belong in the maturity label or only in
the defect head is an open taxonomy question.

The OWLv2 crops are kept anyway: they remove the pile-label errors (46 % of Fresh/Rotten
images held out).

### Part 4: owner phone photos (2026-11-12)

The project owner's own iPhone photos, one session on 2026-11-10. Datasheet:
[`datasheets/owner_phone_2026.md`](datasheets/owner_phone_2026.md).

- **Folders:** `Fresas/overripe` 94 photos, `Fresas/defective ` 57 photos (trailing space
  in the name).
- **Mapping:** overripe → `overripe` (owner-sorted, defects not assessed). Defective →
  `needs_review`. The owner labels maturity and each defect per berry, in
  `reports/labeling/needs-review_owner_phone_2026.csv` (58 crops).
- **Boxes:** OWLv2 (ADR [0008](decisions/0008-box-proposals-for-image-labels.md), min score
  0.3) found 279 boxes on 151 photos.
- **Silver:** 147 rows. 89 overripe ready, 2 overripe held as piles, 56 defective pending
  review. 3 exact duplicates were rejected (IMG_1622, IMG_1638, IMG_1639; the defective
  copy is kept, so the owner's review sets both maturity and defects). 1 near-duplicate
  was rejected (defective IMG_1486 ≈ IMG_1485).
- **Caveats:** one session, so in-distribution scores are optimistic (use the Kamat
  proxy). One labeler. No unripe or ripe photos. Training source only, not the locked
  phone test set (ADR [0007](decisions/0007-lean-phone-test-set.md)).

Findings #17 and #18 showed that overripe from Fresh/Rotten and from Kamat don't transfer
to each other. This source adds a third overripe setting, hand-held phone photos like our
target use. It is training data, so it can't give an ADR 0005 verdict.

## Expected class balance after mapping (approximate)

| unripe | ripe | overripe | rot / mold | bruise, deformed, pest, mechanical |
|---|---|---|---|---|
| ~3,500 | ~1,000 | **~170** (Kamat "rotten", needs review) | ~1,000+ | **0** |

Overripe and most defect types are the gaps. See
[labeling/overripe-timelapse.md](labeling/overripe-timelapse.md) for how we fill the first.

## Transformations and why

| # | What | Why | Where |
|---|---|---|---|
| 1 | **Crop one fruit per image**: a square box padded by 15 %. Neighbours that fall inside the crop are kept, masked or dropped by `CropPolicy` (ADR 0011, proposed) | Classifiers learn one label per image. Detection datasets have several fruits per photo | `crops.crop_box`, `crops.crop_fruit` |
| 2 | **Map labels per source**. Ambiguous classes get `needs_review` | Every dataset draws ripeness boundaries differently (Yurdakul's semi-ripe spans 10–90 % red, ours is ≥75 %) | `configs/sources.yaml` |
| 3 | **Remove exact and near duplicates**: SHA-256 of pixels, then dHash over all 8 flips/rotations | Several sources ship their own augmented copies. Copies split across train and test inflate the scores | `silver.build_silver`, `dedup` |
| 4 | **Drop synthetic images** | DALL·E images aren't real fruit and would teach the model artifacts | `exclude` globs per source |
| 5 | **Apply EXIF orientation, convert to RGB, re-encode as JPEG** | Phone photos are often stored sideways. Box coordinates assume the displayed orientation | `silver.build_silver` |
| 6 | **Keep a 32 px minimum side** (not 64) | Riseholme crops average ~63 px. A 64 px cut would discard about half of them. The model upscales | `configs/data.yaml` |
| 7 | **Split by group, hashing the group id** | Fruit from the same session or plant must stay in one split. Hashing keeps old groups in place when data is added | `gold.assign_splits` |
| 8 | **No hue or saturation augmentation in training** | Ripeness *is* color. Use geometric changes and mild brightness/contrast only | training config (next plan) |
| 9 | **Weight classes and report per-class macro-F1** | Unripe outnumbers overripe about 20:1 | training config (next plan) |
| 10 | **Evaluate per source and with one source held out** | Kamat's pastel backgrounds and per-source capture setups let a model learn "which dataset" instead of "how ripe" | evaluation (next plan) |

## Workflow

```bash
# 1. Download the source manually (Mendeley "Download All", GitHub ZIP, Kaggle CLI)
uv run python scripts/ingest_source.py riseholme_2021 ~/Downloads/Riseholme-2021-main.zip
# 2. Rebuild silver from every enabled source
uv run python scripts/build_silver.py
# 3. Release an immutable gold version
uv run python scripts/build_gold.py maturity_cls 0.1
```

To enable a "needs verify" source:
1. Ingest it.
2. Compare its `data.yaml`, folders or COCO categories against `class_names` and `labels`,
   then run `uv run python scripts/contact_sheets.py <source>` and check every sheet
   against the [rubric](labeling/rubric.md).
3. Fill in the `exclude` globs for augmented or synthetic files.
4. Set `enabled: true`.
5. Rebuild silver and read `rejected.parquet` before cutting gold.
