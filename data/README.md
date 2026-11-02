# Data — Medallion Layout

All data in this folder is **git-ignored**. Only these README files are tracked,
so every collaborator (human or AI agent) sees the same structure and contract.
Share the actual data through the team's storage (drive, bucket, DVC remote).
Don't commit it to git.

```
data/
├── bronze/   Raw, immutable, as-acquired      (append-only)
├── silver/   Cleaned, validated, unified      (reproducible from bronze)
└── gold/     Model-ready, versioned datasets  (reproducible from silver)
```

Data flows in **one direction only**: `bronze → silver → gold`. Each layer is
derived from the previous one by code in `src/strawberry_picker/data/`.
Never hand-edit silver or gold. Fix the transform and regenerate.

| Layer  | Contains | Who writes it | Mutable? |
|--------|----------|---------------|----------|
| bronze | Original images + source metadata, exactly as downloaded/captured | `data/bronze.py` (ingest) or a human dropping files | Append-only, never modified |
| silver | Deduplicated, decoded, orientation-fixed, standardized images + one unified `manifest.parquet` with labels mapped to our taxonomy | `data/silver.py` | Regenerated |
| gold   | Train/val/test splits in the format a model consumes, frozen per version | `data/gold.py` | Immutable once a version is released |

See each layer's README for its exact layout and naming rules, and
[`docs/data-contract.md`](../docs/data-contract.md) for the label taxonomy
(maturity + quality classes) and manifest schemas.

## Overriding the location

Set `STRAWBERRY_DATA_DIR=/path/to/data` (see `.env.example`) to keep the data
outside the repo, for example on an external drive or a mounted bucket. The code
resolves every layer through `strawberry_picker.paths`, so never hardcode
`data/...` paths.
