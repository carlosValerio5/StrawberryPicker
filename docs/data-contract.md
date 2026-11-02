# Data contract

The labels and manifest schemas every layer and model agrees on.
Code mirror: `src/strawberry_picker/schema.py`. **Keep both in sync.**

> Status: **v0, proposed.** Confirm the classes with the team (and ideally a
> grower or QA spec) before labeling at scale. Record changes in an ADR.

## 1. Maturity (single label, ordinal)

| Value | Meaning (visual cue) |
|-------|----------------------|
| `unripe`   | Not ready to pick: green, white, or only partly red (< ~75 % of the surface) |
| `ripe`     | ≥ ~75 % evenly red, firm, glossy. Ready to pick |
| `overripe` | Dark red/purplish, dull, soft or leaking. Past peak |

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
| `image_id`     | string    | first 16 hex chars of sha256 over image size + EXIF-corrected RGB pixels |
| `source_name`  | string    | bronze source folder name |
| `bronze_path`  | string    | path relative to `data/bronze/` |
| `license`      | string    | SPDX id or free text, copied from `SOURCE.md` |
| `width`        | int       | pixels |
| `height`       | int       | pixels |
| `maturity`     | string?   | one of §1, or null if unknown |
| `defects`      | list[str] | subset of §2a. Empty means none. Null means not assessed |
| `bbox`         | list[float]? | `[x_min, y_min, x_max, y_max]` in pixels, if the image has one fruit annotated |
| `group_id`     | string    | split-leakage group (session/plant/source). Defaults to `source_name` |
| `ingested_at`  | timestamp | UTC, ISO-8601 |

Multi-fruit images with per-fruit boxes are stored as one row per fruit, sharing
the same `image_id`.

## 4. Gold manifest — `data/gold/<version>/manifest.parquet`

The silver columns plus:

| Column | Type | Notes |
|--------|------|-------|
| `split` | string | `train` / `val` / `test` |
