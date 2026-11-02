# Scripts

Thin command-line entry points. Each script parses arguments, loads a config with
`strawberry_picker.config.load_config`, and calls into `src/strawberry_picker/`.
**Keep logic out of here.** Run with `uv run python scripts/<name>.py`.
