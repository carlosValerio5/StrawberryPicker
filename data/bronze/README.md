# Bronze — raw, immutable

Everything exactly as it arrived: public datasets, field photos, scraped images.
**Never modify, rename, or delete files here.** If a source is bad, exclude it
in the silver transform instead.

## Layout

```
bronze/
└── <source_name>/              # snake_case, e.g. kaggle_strawberry_ripeness, field_2026_11
    └── <ingest_date>/          # YYYY-MM-DD, the day it was pulled (YYYY-MM-DD.2 for a 2nd pull that day)
        ├── SOURCE.md           # where it came from, license, URL/DOI, who pulled it
        ├── <original files>    # untouched images, annotations, archives
        └── ...
```

## Adding a pull

Register the source in `configs/sources.yaml` first. Then:

```bash
uv run python scripts/ingest_source.py <source_name> path/to/download.zip
```

That unpacks or copies the download into `<source_name>/<today>/` and writes
`SOURCE.md` from the registry. It refuses to overwrite an existing pull.

## Rules

- One folder per source, one sub-folder per pull. A re-download goes in a new date folder.
- `SOURCE.md` is **required**. Include the license, because some datasets ban
  commercial use and that limits which gold datasets can use them.
- Keep original annotation files (COCO, YOLO, VOC, CSV) next to the images. The silver
  step maps them to our taxonomy.
