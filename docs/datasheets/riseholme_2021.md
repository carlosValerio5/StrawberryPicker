# Datasheet: `riseholme_2021`

Short answers to the *Datasheets for Datasets* questions (Gebru et al., 2021). Only facts
we checked are listed. Assumptions are marked.

| | |
|---|---|
| **Title** | Riseholme Strawberry Classification 2021 |
| **Origin** | University of Lincoln. Choi, Would, Salazar-Gomez, Cielniak, ICRA 2022 |
| **URL** | https://github.com/ctyeong/Riseholme-2021 |
| **License** | **None stated** (no license file). Academic use is assumed (ADR 0003). Ask the authors before redistributing |
| **Bronze pull** | `2026-11-02`, GitHub ZIP of `main`, 28.4 MB |

## Motivation and collection

Built to detect fruit anomalies from a robot. Images are crops from a camera mounted on
a robot in UK polytunnels, with fruit still on the plant.

## Composition (verified)

| Folder | Images | Our label |
|---|---|---|
| `Data/Normal/Unripe` | 2,406 | `unripe`, no defects |
| `Data/Normal/Ripe` | 462 | `ripe`, no defects |
| `Data/Normal/Occluded` | 499 | dropped (no ripeness label) |
| `Data/Anomalous` | 153 | `needs_review` (the defect type isn't given) |

Crops are tiny: the median side is about 60 px, and 152 are under our 32 px minimum.

## What the audit found

- **Labels match the folders**, judging from contact sheets of 50 random crops per label.
- **No borderline fruit.** `Unripe` is white or green, `Ripe` fully red. The source can't
  teach where the ¾ line falls.
- **Domain:** far from hand-held phone photos (robot camera, polytunnel light, ~60 px).
- It exposed the 64-bit dHash bug: small blurry crops of different berries looked like
  duplicates. Fixed. See `docs/datasets.md` finding #2.
- The statistical audit's numbers are in `reports/dataset-audit/report.md`.

## Uses

Fine for unripe/ripe color learning. Not useful for overripe or defects until
`Anomalous` has been reviewed by hand.
