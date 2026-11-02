# Bronze — raw, immutable

Everything exactly as it arrived: public datasets, field photos, scraped images.
**Never modify, rename, or delete files here.** If a source is bad, exclude it
in the silver transform instead.

## Layout

```
bronze/
└── <source_name>/              # snake_case, e.g. kaggle_strawberry_ripeness, field_2026_11
    └── <ingest_date>/          # YYYY-MM-DD, the day it was pulled
        ├── SOURCE.md           # where it came from, license, URL/DOI, who pulled it
        ├── <original files>    # untouched images, annotations, archives
        └── ...
```

## Rules

- One folder per source, one sub-folder per pull. A re-download goes in a new date folder.
- `SOURCE.md` is **required**. Include the license, because some datasets ban
  commercial use and that limits which gold datasets can use them.
- Keep original annotation files (COCO, YOLO, VOC, CSV) next to the images. The silver
  step maps them to our taxonomy.
