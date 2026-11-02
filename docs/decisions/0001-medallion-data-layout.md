# 0001 — Medallion data layout, data kept out of git

- **Status:** accepted
- **Date:** 2026-11-02

## Context

Several people and AI agents will work with image datasets from different
sources, licenses, and label schemes. Images are large and some licenses forbid
redistribution, so they can't live in git. Collaborators still need one shared
picture of where data lives and how it moves.

## Decision

- Use a three-layer medallion layout under `data/`: **bronze** (raw, append-only),
  **silver** (cleaned, deduplicated, unified taxonomy), and **gold** (versioned,
  split, model-ready).
- Git-ignore all data. Track only the README for each layer, which defines its contract.
- Every layer below bronze is reproducible from code in `src/strawberry_picker/data/`.
- All path resolution goes through `strawberry_picker.paths`, with
  `STRAWBERRY_DATA_DIR` as an override.

## Consequences

- Data sharing happens outside git (shared drive / bucket). A DVC remote is a
  likely follow-up ADR once the datasets stabilize.
- Lineage (`source_name`, `bronze_path`, `license`) is carried into silver and gold,
  so the license of any training set can be audited.
