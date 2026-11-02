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

1. Decode and validate. Drop corrupt or unreadable files and log them in `rejected.parquet`.
2. Apply EXIF orientation, convert to RGB, and save as JPEG at a consistent quality.
3. Deduplicate by content hash, both within and across sources.
4. Map each source's labels to our taxonomy (`docs/data-contract.md`). Any label
   that can't be mapped is set to `null`. Don't guess.
5. Record lineage: every row keeps `source_name`, `bronze_path`, and `license`.

Images are **not** resized or cropped here. That's a gold/model concern.
