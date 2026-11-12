# Data contract

The labels and manifest schemas every layer and model agrees on.
Code mirror: `src/strawberry_picker/schema.py`. **Keep both in sync.**

> Status: **v0.** Class definitions follow the USDA strawberry grade standard
> (ADR [0004](decisions/0004-usda-label-rubric.md)). Record changes in an ADR.

## 1. Maturity (single label, ordinal)

The exact decision rules, anchored to the USDA grade standard, are in
[labeling/rubric.md](labeling/rubric.md) (ADR [0004](decisions/0004-usda-label-rubric.md)).
The rubric wins wherever this short summary differs from it.

| Value | Meaning (visual cue) |
|-------|----------------------|
| `unripe`   | Less than ¾ of the surface is pink or red: green, white, or turning |
| `ripe`     | At least ¾ of the surface is pink or red, with no overripe signs |
| `overripe` | Dark red/purplish, dull, wrinkled, slumped or leaking. Past peak |

The order matters (`unripe < ripe < overripe`), so ordinal-aware losses and
metrics are allowed. Partly-red "turning" fruit counts as `unripe`
(ADR [0002](decisions/0002-three-maturity-classes.md)).

## 2. Quality

### 2a. Defects (multi-label, zero or more)

| Value | Meaning |
|-------|---------|
| `bruise`            | Soft, darkened, or water-soaked area from impact or pressure |
| `mold`              | Visible fungal growth (e.g. grey mold / *Botrytis*) |
| `rot`               | Decay or collapse without visible mycelium |
| `deformed`          | Misshapen fruit (poor pollination, cat-facing) |
| `pest_damage`       | Bites, holes, or slug/insect/bird damage |
| `mechanical_damage` | Cuts, crushing, or torn skin from handling |

### 2b. Grade (derived from defects, not labeled directly)

| Grade | Rule |
|-------|------|
| `reject`   | any of `mold`, `rot` |
| `standard` | any other defect |
| `premium`  | no defects |

`strawberry_picker.schema.grade_from_defects` implements this rule.

## 3. Silver manifest — `data/silver/manifest.parquet`

One row per unique image.

| Column | Type | Notes |
|--------|------|-------|
| `image_id`       | string    | first 16 hex chars of sha256 over image size + EXIF-corrected RGB pixels |
| `source_name`    | string    | bronze source folder name, as registered in `configs/sources.yaml` |
| `bronze_path`    | string    | path relative to `data/bronze/` |
| `source_label`   | string    | the label exactly as the source wrote it, before mapping (for audits) |
| `license`        | string    | SPDX id or free text, from the source registry |
| `commercial_use` | bool      | false if the source license forbids commercial use |
| `width`          | int       | pixels, after EXIF orientation |
| `height`         | int       | pixels, after EXIF orientation |
| `maturity`       | string?   | one of §1, or null if the source doesn't label ripeness |
| `defects`        | list[str]? | subset of §2a. Empty means none found. Null means not assessed |
| `needs_review`   | bool      | the source label maps ambiguously. Gold excludes the row until a human confirms it |
| `bbox`           | list[float]? | `[x_min, y_min, x_max, y_max]` in pixels. Null means the whole image is one fruit |
| `group_id`       | string    | split-leakage group, set by the source's `group_by`: `<source>/<image_id>` (default), `<source>`, or `<source>/<folder>` |
| `ingested_at`    | timestamp | UTC time of the silver build |
| `reviewed_by`    | string?   | Labelers (`alice+bob`) whose review in `bronze/human_review/` set this row. Null means the source's mapped label. With `needs_review` still true, reviewers disagreed, were unsure, marked it not a strawberry, or marked a whole-image row as several berries (its source label is kept) |

Multi-fruit images with per-fruit boxes are stored as one row per fruit, sharing
the same `image_id`. Images dropped from bronze are listed in
`data/silver/rejected.parquet` with a reason: `excluded`, `label_dropped`,
`corrupt`, `too_small`, `duplicate` (identical pixels), or `near_duplicate`
(perceptual-hash match, e.g. a flipped or re-encoded copy).

## 4. Gold manifest — `data/gold/<version>/manifest.parquet`

The silver columns plus:

| Column | Type | Notes |
|--------|------|-------|
| `split`     | string | `train` / `val` / `test`, assigned per `group_id` |
| `crop_id`   | string | `<image_id>_<n>`, where `n` is the fruit's position within its image in silver |
| `crop_path` | string | crop file relative to the dataset folder |
| `neighbour_bboxes` | list of `[x0, y0, x1, y1]` | the other fruit boxes in the same silver image, including rows this release leaves out ([ADR 0011](decisions/0011-crop-isolation.md)) |
| `neighbour_overlap` | float | share of the crop those boxes cover, outside the fruit's own box (0–1) |
| `crop_policy` | string | the `CropPolicy` JSON the crop was cut with; the release also has `crop_policy.json` |

Crops the `drop` policy leaves out are listed, with the same columns, in `dropped.parquet`,
and counted under `dropped` in `stats.json`.

Rows with `needs_review = true`, or without the task's label, are not in gold.
