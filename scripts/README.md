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
