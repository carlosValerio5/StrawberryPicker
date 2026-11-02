# Gold — model-ready, versioned

Frozen datasets that a training run consumes directly. Built from silver by
`strawberry_picker.data.gold`. **Once a version is released, don't change it.**
Cut a new version instead.

## Layout

```
gold/
└── <task>_v<MAJOR>.<MINOR>/      # e.g. maturity_cls_v1.0, quality_det_v0.1
    ├── DATASET_CARD.md           # task, sources, class balance, split strategy, known issues
    ├── manifest.parquet          # silver rows + `split` column (train/val/test)
    ├── train/  val/  test/       # task-specific format, see below
    └── stats.json                # counts per class per split (written by the build)
```

`<task>` is one of:

| Task | Folder format inside each split |
|------|---------------------------------|
| `maturity_cls` / `quality_cls` | ImageFolder: `<split>/<class_name>/<image_id>.jpg` |
| `*_det` | YOLO: `<split>/images/*.jpg` + `<split>/labels/*.txt`, plus a `data.yaml` |
| `*_seg` | COCO: `<split>/images/*.jpg` + `<split>/annotations.json` |

## Rules

- Splits are **group-aware**. Images from the same source session or the same plant
  stay in the same split, so near-duplicate photos can't leak across splits.
- The `test` split is held out. Don't use it for tuning.
- Bump MINOR for added data with the same classes. Bump MAJOR for a taxonomy or split-strategy change.
