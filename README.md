# 🍓 StrawberryPicker

Look at a strawberry and tell how **ripe** it is and what its **quality** is, using
open-source vision models.

- **Maturity:** `unripe` · `ripe` · `overripe`
- **Quality:** visible defects (bruise, mold, rot, …), giving a `premium` / `standard` / `reject` grade

The classes are defined in [docs/data-contract.md](docs/data-contract.md), and the
model plan is in [docs/architecture.md](docs/architecture.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/carlosValerio5/StrawberryPicker.git
cd StrawberryPicker
make setup        # core + dev deps
make setup-ml     # + torch / timm / transformers (only where you train)
cp .env.example .env
make check        # lint + tests
```

## Data (medallion layout)

Data is **never committed**. The folders and their READMEs are tracked so
everyone uses the same structure:

```
data/bronze/   raw, immutable, as-acquired      → data/bronze/README.md
data/silver/   cleaned, deduplicated, one taxonomy → data/silver/README.md
data/gold/     versioned train/val/test sets    → data/gold/README.md
```

Start with [data/README.md](data/README.md). Get the actual data from the team's
shared storage, or point `STRAWBERRY_DATA_DIR` at your copy.

## Repo layout

| Path | What lives there |
|------|------------------|
| `src/strawberry_picker/` | Library code: `paths`, `schema`, `config`, `data/`, `models/`, `evaluation/`, `inference/` |
| `configs/` | YAML configs for the data pipeline and each training task |
| `docs/` | Architecture, data contract, ADRs (`docs/decisions/`) |
| `notebooks/` | Exploration only |
| `scripts/` | Thin CLI wrappers around `src/` |
| `models/`, `reports/` | Generated weights and outputs (git-ignored) |
| `tests/` | pytest, runs without real data or a GPU |

## Working with AI agents

[AGENTS.md](AGENTS.md) holds the conventions and hard rules for every coding
agent (Claude Code loads it through `CLAUDE.md`). Read it too. It's the shortest
complete description of how this repo works.
