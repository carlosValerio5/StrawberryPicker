# Model card: StrawberryPicker v1

Branch `release/v1`. Every number here is a development number. Read
[Not a verdict](#not-a-verdict) first.

## What it does

From one phone photo, for each strawberry it finds:

| Output | Values |
|---|---|
| Maturity | `unripe` / `ripe` / `overripe`, temperature-calibrated probabilities |
| Defects | `mold`, `rot` |
| Grade | rule-derived: `premium` / `standard` / `reject` |

Pipeline:

| Step | Component | License / note |
|---|---|---|
| Detect | OWLv2 `google/owlv2-base-patch16-ensemble`, score ≥ 0.2, NMS 0.5 | Apache-2.0 |
| Crop | ADR [0011](decisions/0011-crop-isolation.md) policy, mode `keep`, padding 0.15 | |
| Embed | DINOv2-small, frozen | Apache-2.0 |
| Classify | logistic-regression probes (maturity, defects) | |

Input: JPEG, PNG, and HEIC (decoded with pi-heif).

## How to run

```bash
make setup-ml   # torch, timm, transformers (once)
uv run python scripts/predict.py <photo or folder> --out preds.json --annotate
```

- The first run downloads the weights.
- Speed: about 5 minutes for 2 photos on a Mac. Most of that is model loading and OWLv2
  on CPU.

Runs used by v1:

| Head | Run | Gold version |
|---|---|---|
| Maturity | `models/maturity_cls/20261112T170730Z` | `maturity_cls_v0.5`, temperature T = 1.66 fitted on gold val |
| Defects | `models/quality_cls/20261112T170718Z` | `quality_cls_v0.2` |

## Training data

Sources and labels are in [datasets.md](datasets.md).

**Maturity (training counts):**

| Class | Count | Sources |
|---|---|---|
| unripe | 2,161 | Riseholme, Strawberry-DS |
| ripe | 488 | Riseholme, Strawberry-DS |
| overripe | 38 | Fresh/Rotten |

Held out, never trained on for maturity: `kamat_2025` and `owner_phone_2026`
(see [Finding #19](datasets.md#part-4-owner-phone-photos-2026-11-12)).

**Defects:**

| Probe | Positives | Status |
|---|---|---|
| mold | 502 | trained |
| rot | 554 | trained |
| deformed | 29 | untrained (min 30) |
| mechanical_damage | 4 | untrained |
| bruise | 0 | untrained |
| pest_damage | 0 | untrained |

## Results

### Maturity

| Set | n | QWK [95% CI] | Recall unripe / ripe / overripe | ECE |
|---|---|---|---|---|
| In-distribution test (same sources as training; optimistic) | 589 | 0.938 [0.90, 0.97] | 0.98 / 0.94 / 1.00 (overripe n=10) | 0.011 |
| Kamat proxy (phone shots, never trained on) | 233 | 0.424 [0.30, 0.55] | 0.72 / 0.85 / 0.15 [0.08, 0.25] | 0.308 |
| Owner phone proxy (never trained on for maturity) | 144 | -0.023 [-0.11, 0.07] | n/a / 0.58 / 0.36 [0.28, 0.44] | 0.350 |

On owner photos, 80 of 125 overripe berries are called ripe.

### Defects

| Set | Reject recall (mold or rot) [95% CI] |
|---|---|
| In-distribution | 0.92 [0.87, 0.96] |
| Kamat proxy | 0.35 [0.20, 0.49] |

ADR [0005](decisions/0005-acceptance-criteria.md) §C bar of 0.80 applies only to the locked
phone set. That set does not exist yet (see below).

## Limitations

- **Overripe is not reliably detected on phone photos.** Recall is 0.15 on Kamat and 0.36
  on the owner's photos. Cause: [Findings #17–#19](datasets.md#audit-findings-2026-11-02).
- **Confidence is not calibrated off-distribution.** ECE is 0.308 on Kamat and 0.350 on the owner photos, so
  the probabilities are not trustworthy outside the training sources.
- **Defects: only mold and rot.** The other types have too few positives to train.
- **Kamat labels are noisy.** About 1 in 4 sampled "unripe" boxes are flowers or buds
  (Finding #11).
- **Owner source is one session, one labeler.** Its in-distribution scores are optimistic.
- **Crop size is a shortcut.** Training crop sizes separate the classes by source, not by
  ripeness (Finding #19).

## Not a verdict

- No locked phone test set exists (ADR [0007](decisions/0007-lean-phone-test-set.md) §6),
  so there is no ADR 0005 pass/fail for v1.
- Every number on this page is a development number.
- Thresholds were not changed after seeing these results.
