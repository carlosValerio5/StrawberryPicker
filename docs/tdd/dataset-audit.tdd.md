# TDD evidence: dataset audit, part 1

- **Source:** the "deeper" dataset prompt agreed in chat on 2026-11-02. Its steps 1–3
  and 5–7 are written as docs. Step 4 (embeddings, cleanlab) and step 6 (learning curves,
  results on held-out sources) haven't started.
- **Branch:** `feat/dataset-audit`.
- **Data:** with the owner's approval, Riseholme, Strawberry-DS and Disease (test + valid)
  were downloaded and ingested into bronze. The checksum of `Strawberry (2).zip` matches
  Mendeley's sha256.

## User journeys

1. As a data engineer, I see a sample of every source label before enabling a source,
   so a wrong class order or a wrong label meaning is caught by eye.
2. As an ML engineer, near-duplicate removal drops copies of an image, never different
   fruit. Above all, it never merges a ripe berry with an unripe one.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| Contact sheets: sampling, grid, crops, unreadable files, mapping summary, empty-pull error | `921e5f7`: module missing | `df627dd`: 8 passed |
| Contact sheets honor `exclude` globs | test failed (excluded files sampled) | `9b4c25b`: 9 passed (one commit for both stages) |
| Low-texture distinct crops are not near-duplicates; thresholds are in 256-bit units | `7377a74`: 7 failures. The 64-bit hash merged the synthetic twins (distance 5 ≤ 24) | `32d5624`: all passed, coverage 99.35 % |

## Evidence that isn't a unit test

- **Contact-sheet review** of all 18 labels across the three sources. Findings #1 and
  #3–#8 are in [datasets.md § Audit findings](../datasets.md#audit-findings-2026-11-02).
- **Near-duplicate review:**
  - **Before the fix:** 94 matches. 71 were Riseholme pairs with 256-bit distances of
    47–94 bits, and 10 of them were ripe↔unripe.
  - **After the fix:** 19 matches, all disease-set copies, every one checked by eye.
  - **Silver:** grew from 4,957 to 5,038 rows.

## Gaps

- The rubric's reference images aren't picked yet. They'll come from these contact
  sheets.
- Not run yet:
  - the domain-shift probe and UMAP
  - cleanlab label-error estimates
  - per-source datasheets
  - learning curves
- These sources aren't downloaded yet:
  - Kamat
  - Fresh/Rotten
  - Yurdakul
  - the Disease `train/` images
- The locked phone test set doesn't exist yet. Only the owner can capture it.

## Part 2: statistics, remaining sources, CI parity

| Behavior | RED | GREEN |
|---|---|---|
| Crop embeddings, confident learning, source/held-out-source/learning-curve probes | `326050c`: modules missing | `4273bf2`: 167 passed |
| Audit report, review sheets, embedding cache | `d60cea8`: module missing | `dc9af31`: 170 passed |
| Same-day bronze revisions `<date>.<n>` | RED commit, then `latest_pull` ignored them | GREEN commit: all passed |
| Seeded random-sample review sheets | test failed (function missing) | `caa56d8`: all passed, coverage 99.28 % |

- **Confident learning** is implemented in-repo. `cleanlab` is AGPL-3.0 (AGENTS.md
  rule 7; noted in `docs/architecture.md`).
- **CI parity:** scikit-learn is in the dev group, so the audit tests run in CI without
  torch. A clean core + dev environment passes at 99.28 % coverage.
- **Real-data runs:**
  - Kamat (sha256 n/a: whole-dataset ZIP) and Fresh/Rotten (sha256 matches) were
    ingested, contact-sheeted and enabled.
  - Disease `train/` landed as pull `2026-11-02.2`.
  - `scripts/dataset_audit.py` ran over 5 sources. Findings #10–#16 and the statistics
    are in `docs/datasets.md`.
- **Not done:**
  - the 2-D embedding plot (UMAP / t-SNE). The source probe answers the same question
    with a number.
  - human verdicts on the review lists and random samples (owner).
  - acceptance of ADR 0006 (owner).
