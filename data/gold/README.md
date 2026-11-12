# Gold — model-ready, versioned

Frozen datasets that a training run consumes directly. Built from silver by
`strawberry_picker.data.gold`. **Once a version is released, don't change it.**
Cut a new version instead.

## Layout

```
gold/
└── <task>_v<MAJOR>.<MINOR>/      # e.g. maturity_cls_v1.0, quality_det_v0.1
    ├── DATASET_CARD.md           # task, sources, class balance, split strategy, known issues
    ├── manifest.parquet          # silver rows + split, crop_id, crop_path, neighbour overlap
    ├── crop_policy.json          # how crops were cut (ADR 0011); evaluation reuses it
    ├── dropped.parquet           # only if the drop policy left crops out
    ├── train/  val/  test/       # task-specific format, see below
    └── stats.json                # counts per class per split, plus dropped crops
```

Build one with `uv run python scripts/build_gold.py <task> <MAJOR.MINOR>`.

`<task>` is one of:

| Task | Folder format inside each split |
|------|---------------------------------|
| `maturity_cls` | ImageFolder (single label): `<split>/<maturity>/<crop_id>.jpg` |
| `quality_cls` | Multi-label: `<split>/images/<crop_id>.jpg`, with `defects` taken from `manifest.parquet` |
| `*_det` (future) | YOLO: `<split>/images/*.jpg` + `<split>/labels/*.txt`, plus a `data.yaml` |
| `*_seg` (future) | COCO: `<split>/images/*.jpg` + `<split>/annotations.json` |

Each crop is a square around one fruit, with 15 % padding (`configs/data.yaml`). Other
berries inside the crop are kept, masked or dropped by `gold.crop_neighbours` (ADR 0011).
Rows with `needs_review` or without the task's label are left out.

## Rules

- Splits are **group-aware**. Each `group_id` is hashed to a split, so images from the
  same session or plant stay together. Adding data in a new version never moves an
  existing group to another split.
- The `test` split is held out. Don't use it for tuning.
- Bump MINOR for added data with the same classes. Bump MAJOR for a taxonomy or split-strategy change.
