# TDD evidence — dataset pipeline

- **Source plan:** the dataset research plan approved in chat on 2026-11-02 (no `.plan.md`
  file). Its phases 1, 2 and 4 are implemented here. Phase 3 is manual work: the
  time-lapse capture and label review are documented, not coded.
- **Branch:** `feat/dataset-pipeline`

## Decisions taken without an explicit answer

The plan left three questions open. The defaults recommended in the plan were used and
recorded in [ADR 0003](../decisions/0003-dataset-selection.md):
1. Kamat "rotten" → overripe + rot, flagged `needs_review`.
2. Overripe comes from our own time-lapse capture.
3. Academic use is assumed, and non-commercial rows are tagged.

The zero-shot "is it a strawberry?" gate discussed afterwards was **not** added. The user
hadn't confirmed it.

## User journeys

1. As a data engineer, I register a public dataset with its license and label mapping,
   so a bad mapping fails before any data is processed.
2. As a data engineer, I land a downloaded archive in bronze without ever overwriting an
   earlier pull.
3. As an ML engineer, I rebuild silver from every enabled source and get one manifest
   in our taxonomy, plus an audit list of everything dropped and why.
4. As an ML engineer, I release an immutable, leakage-safe gold dataset of per-fruit
   crops for maturity or quality training.

## RED → GREEN → refactor

| Stage | Commit | Command | Result |
|---|---|---|---|
| RED | `a1bec81` | `uv run pytest` | 6 test modules fail at import, because the modules and functions don't exist. `test_schema.py`: 2 failed (missing silver/gold columns), 57 passed |
| GREEN | `49802f2` | `uv run pytest` | 141 passed. The contract-sync test caught the undocumented columns until `data-contract.md` was updated |
| Refactor | `3813e8f` | `make check`, `uv run pytest --cov` | ruff clean, 145 passed, coverage 99.24 % (gate is 80 %) |

## What the tests guarantee

| # | Guarantee | Test | Type |
|---|---|---|---|
| 1 | Registry mappings are typed. Invalid maturity or defect values, unknown formats, missing fields and YOLO without `class_names` are rejected at load | `tests/test_sources.py` | unit |
| 2 | An unmapped source label stops the build, unless the source opts into `unmapped: drop` | `test_sources.py::test_unknown_label_*`, `test_silver.py::test_build_silver_fails_loudly_on_unmapped_label` | unit / integration |
| 3 | The repo registry is valid, and only verified sources are enabled | `test_sources.py::test_repo_registry_*` | unit |
| 4 | Folder-per-class, YOLO (boxes, polygons, `images/`↔`labels/`) and COCO readers produce pixel `xyxy` boxes. Malformed rows and out-of-range class indices raise | `tests/test_readers.py` | unit |
| 5 | dHash survives JPEG re-encoding and resizing. Flipped and rotated copies cluster with their original, and distinct images don't | `tests/test_dedup.py` | unit |
| 6 | Ingest creates `<source>/<date>/` with `SOURCE.md`, unpacks zip and tar safely, and refuses to overwrite a pull | `tests/test_bronze.py` | integration |
| 7 | Silver keeps one row per fruit with lineage, preserves "no defects" versus "not assessed" in parquet, and applies EXIF orientation | `test_silver.py::test_build_silver_manifest_rows`, `…writes_images_and_parquet`, `…applies_exif_orientation` | integration |
| 8 | Silver rejects excluded, label-dropped, corrupt, too-small, duplicate and near-duplicate files, each with a reason | `test_silver.py::test_build_silver_rejections` | integration |
| 9 | Silver never modifies bronze, and is reproducible | `test_silver.py::test_build_silver_never_touches_bronze`, `…is_reproducible` | integration |
| 10 | Crops are padded squares that are shifted or clipped to stay inside the image | `test_gold.py::test_crop_*` | unit |
| 11 | Splits are deterministic, follow the ratios, never move existing groups when data is added, and never split a group | `test_gold.py::test_splits_*`, `…adding_groups_never_moves_existing_ones`, `…groups_never_straddle_splits` | unit / integration |
| 12 | Gold excludes `needs_review` and unlabeled rows, honours `--commercial-only`, and writes stats and a dataset card | `test_gold.py::test_build_*` | integration |
| 13 | A released gold version is immutable, and a failed build leaves no partial output | `test_gold.py::test_released_version_is_immutable`, `…failed_build_leaves_nothing_behind` | integration |
| 14 | `schema.py` and `data-contract.md` stay in sync, gold columns included | `tests/test_schema.py` | unit |

## End-to-end smoke run (manual)

I built a synthetic Riseholme-shaped zip and ran the three scripts against a scratch
`STRAWBERRY_DATA_DIR`:
- **Ingest:** 49 images.
- **Silver:** 45 rows. 4 "Occluded" images were rejected as `label_dropped`.
- **Gold:** `maturity_cls_v0.1` with 29/6/7 crops across train/val/test.

The repo's `data/` was not touched.

## Coverage and known gaps

`uv run pytest --cov` reports 99.24 %. Uncovered:
- `bronze.py:62`: the notes section of the source card.
- `readers.py:107`: the unreachable unsupported-format raise.
- `paths.py:38,42`: `reports_dir` / `configs_dir`.

Not covered by automated tests:
- the CLI scripts (thin wrappers; smoke-run manually).
- the real downloads: no real dataset has been ingested yet, and five of the six sources
  stay disabled until their layouts are verified.
