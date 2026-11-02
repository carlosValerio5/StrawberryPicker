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
| [`strawberry_ds_2023`](https://data.mendeley.com/datasets/z6dtfdpzz8/1) | 6 stages → unripe / ripe | 247 images, multi-fruit | Egypt greenhouse, 3840×2160 | CC BY 4.0 | needs verify |
| [`kamat_2025`](https://data.mendeley.com/datasets/zysvgmxcyz/1) | unripe 114+238 · ripe 189 · rotten 170 | ~711 strawberry samples | India, field, **pastel backgrounds** | CC BY 4.0 | needs verify |
| [`fresh_rotten_fruits_2022`](https://data.mendeley.com/datasets/bdd69gyhv8) | no defects 200 · rot 200 | 512×512 | shops and fields, DSLR | CC BY 4.0 | needs verify |
| [`strawberry_disease_2025`](https://data.mendeley.com/datasets/k2ptxmfjhj/1) | mold, rot, healthy (fruit classes only) | 2,459 originals in 8 classes | field, COCO polygons, **1,018 DALL·E images** | **CC BY-NC 4.0** | needs verify |

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

## Expected class balance after mapping (approximate)

| unripe | ripe | overripe | rot / mold | bruise, deformed, pest, mechanical |
|---|---|---|---|---|
| ~3,500 | ~1,000 | **~170** (Kamat "rotten", needs review) | ~1,000+ | **0** |

Overripe and most defect types are the gaps. See
[labeling/overripe-timelapse.md](labeling/overripe-timelapse.md) for how we fill the first.

## Transformations and why

| # | What | Why | Where |
|---|---|---|---|
| 1 | **Crop one fruit per image**: a square box padded by 15 % | Classifiers learn one label per image. Detection datasets have several fruits per photo | `gold.crop_box` |
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
2. Compare its `data.yaml`, folders or COCO categories against `class_names` and `labels`.
3. Fill in the `exclude` globs for augmented or synthetic files.
4. Set `enabled: true`.
5. Rebuild silver and read `rejected.parquet` before cutting gold.
