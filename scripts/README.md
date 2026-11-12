# Scripts

Thin command-line entry points. Each script parses arguments, loads a config with
`strawberry_picker.config.load_config`, and calls into `src/strawberry_picker/`.
**Keep logic out of here.** Run with `uv run python scripts/<name>.py`.

| Script | What it does |
|---|---|
| `ingest_source.py <source> <download>` | Unpack a download into `data/bronze/<source>/<today>/` and write `SOURCE.md` |
| `contact_sheets.py <source>` | Tile ~50 random crops per source label into `reports/dataset-audit/<source>/contact-sheets/`. Use it to check a source before enabling it |
| `build_silver.py` | Rebuild `data/silver/` from every enabled source |
| `build_gold.py <task> <version>` | Release an immutable gold dataset |
| `dataset_audit.py` | Statistical audit, review lists and random samples → `reports/dataset-audit/` |
| `crop_overlap.py` | How much of each crop shows other berries, per source and label (silver only, no ml extra) → `reports/dataset-audit/neighbours/` (ADR 0008) |
| `make_label_task.py <kind>` | Turn a backlog (`overripe`, `needs-review`, `random-sample`, `flagged`) into a labeling task |
| `label.py <task.csv> --labeler <you>` | Labeling page on 127.0.0.1, by keyboard or touch (`--blind`, `--sample 50`). `--lan` serves it to phones on the same Wi-Fi behind a fresh token; each phone enters its own name |
| `ingest_reviews.py labels_*.csv` | Land finished labels in `bronze/human_review/`; the next `build_silver.py` applies them |
| `propose_boxes.py <photo folder>` | OWLv2 boxes for phone photos → `task.csv` (downloads ~600 MB weights on first use) |
| `ingest_phone_set.py <folder>` | Land the finished phone set in `bronze/phone_testset/` (locked: never in silver/gold) |
| `train_maturity.py` | Train the DINOv2 linear-probe baseline → `models/maturity_cls/<run_id>/` |
| `evaluate_phone_set.py` | The ADR 0005 verdict on the locked phone set, or the adjudication task if labelers disagree |
