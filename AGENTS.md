# AGENTS.md — guide for AI coding agents

This is the **single source of truth** for any model working in this repo
(Claude Code, Codex, Cursor, Gemini CLI, Copilot…). `CLAUDE.md` just imports
this file. Keep this file current when you change structure or conventions.

## What this project is

StrawberryPicker looks at an image of a strawberry and reports its **maturity**
(how ripe it is) and its **quality** (visible defects and an overall grade).
It's built on open-source vision models: detection to find each fruit, then
classification heads on each crop. See `docs/architecture.md`.

## Repo map

```
AGENTS.md / CLAUDE.md      ← you are here
README.md                  human quick-start
pyproject.toml             deps (uv), ruff + pytest config
Makefile                   setup / lint / format / test / check
configs/                   YAML configs: data + one file per training task
  sources.yaml             bronze source registry + label mappings (validated on load)
data/                      MEDALLION DATA — git-ignored except READMEs
  bronze/ silver/ gold/    raw → cleaned → model-ready (read each README)
docs/
  target-domain.md         where the model runs (hand-held phone photos); judge datasets against it
  architecture.md          pipeline + candidate open-source models
  data-contract.md         label taxonomy + manifest schemas  ← read before touching labels
  datasets.md              public dataset catalog, mappings, transformations, audit findings
  datasheets/              one datasheet per source (what we verified, what the audit found)
  labeling/                rubric.md (USDA-anchored label rules), test-set protocol, capture procedures
  decisions/               ADRs — one file per significant decision
  tdd/                     TDD evidence reports
models/                    trained weights/checkpoints — git-ignored
notebooks/                 exploration only; naming rule in notebooks/README.md
reports/                   generated metrics/figures — git-ignored
scripts/                   thin CLIs: ingest_source.py, contact_sheets.py, build_silver.py, build_gold.py,
                           dataset_audit.py
src/strawberry_picker/
  paths.py                 ONLY place that resolves filesystem locations
  schema.py                taxonomy enums + SilverRecord (code mirror of data-contract.md)
  config.py                YAML config loading
  data/                    bronze.py (ingest) → silver.py (clean) → gold.py (split/export)
    sources.py             registry parsing; readers.py (imagefolder/YOLO/COCO); dedup.py (dHash)
  audit/                   dataset audits → reports/dataset-audit/: contact_sheets.py (raw bronze),
                           embeddings.py, label_issues.py (confident learning), probes.py, report.py
  models/                  model definitions / backbones / heads
  evaluation/              metrics, confusion matrices, reports
  inference/               predict on new images
tests/                     pytest; must not need real data or GPUs
```

## Commands

```bash
make setup      # uv sync (core + dev). `make setup-ml` adds torch/timm/transformers
make check      # ruff lint + pytest. Run before you say a task is done.
make format     # ruff format + autofix
```

Use `uv run <cmd>` to run anything inside the project env.

## Hard rules

1. **Never commit data or weights.** `data/**`, `models/**`, and `reports/**` are
   git-ignored on purpose. If `git status` shows an image, parquet, or `.pt` file,
   stop and fix `.gitignore`. Don't force-add it.
2. **Bronze is append-only.** Never modify, rename, or delete anything under `data/bronze/`.
3. **Data flows one way:** bronze → silver → gold. Never hand-edit silver or gold.
   Change the transform in `src/strawberry_picker/data/` and regenerate.
4. **Released gold versions are immutable.** To change one, cut a new version.
5. **Resolve paths through `strawberry_picker.paths`.** Never hardcode `data/...`
   or absolute paths. The data dir can live outside the repo (`STRAWBERRY_DATA_DIR`).
6. **Label meaning comes from `docs/labeling/rubric.md`** (USDA-anchored). Pass/fail bars
   are fixed in ADR 0005 *before* results are seen. Never tune against the locked phone test set.
7. **The taxonomy lives in two places that must stay in sync:**
   `docs/data-contract.md` (for humans) and `src/strawberry_picker/schema.py` (for code).
   Change both in the same commit, and add an ADR if classes change.
8. **Check model licenses** before adding a dependency or pretrained weights, and
   record it in `docs/architecture.md`. Avoid AGPL/non-commercial weights unless an
   ADR accepts the tradeoff.
9. **Secrets** go in `.env` (git-ignored). Add new variable names to `.env.example`.
10. **Tests must run without real data or a GPU.** Use tiny synthetic images built
   in fixtures (see `tests/conftest.py`).

## Conventions

- Python ≥ 3.11, type hints on public functions, ruff-clean (line length 100).
- Logic goes in `src/`. Notebooks and `scripts/` only call into it.
- Each training run reads one config from `configs/` and writes to
  `models/<task>/<run_id>/` (weights + the resolved config + metrics.json).
- Significant decisions (new model family, taxonomy change, split strategy) get an
  ADR in `docs/decisions/NNNN-short-title.md`. Copy the format of `0001`.
