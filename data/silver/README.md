# Silver — cleaned, validated, unified

One consistent view over all bronze sources. Fully regenerable from bronze by
`strawberry_picker.data.silver`. Never hand-edit.

## Layout

```
silver/
├── images/
│   └── <image_id>.jpg          # image_id = strawberry_picker.data.silver.compute_image_id
├── manifest.parquet            # one row per image, schema: strawberry_picker.schema.SilverRecord
└── rejected.parquet            # images dropped from bronze and why (corrupt, duplicate, unlabeled…)
```

## What the bronze → silver step does

Run it with `uv run python scripts/build_silver.py`. It reads the latest pull of
every `enabled` source in `configs/sources.yaml`.

1. Skip files that match the source's `exclude` globs, such as synthetic or augmented copies.
2. Map each source's labels to our taxonomy using `configs/sources.yaml`. A label
   with no mapping stops the build. Fix the registry rather than guessing.
3. Decode and validate. Drop corrupt files and images under `min_side_px`.
4. Apply EXIF orientation, convert to RGB, and save as JPEG at a consistent quality.
5. Deduplicate: identical pixels first, then near-duplicates (dHash over all 8
   flips/rotations), both within and across sources.
6. Record lineage: every row keeps `source_name`, `bronze_path`, `source_label`,
   `license`, and `commercial_use`.

Every dropped file is listed in `rejected.parquet` with a reason (`excluded`,
`label_dropped`, `corrupt`, `too_small`, `duplicate`, `near_duplicate`). Check that
list before cutting a gold release.

Images are **not** resized or cropped here. That's a gold/model concern.
