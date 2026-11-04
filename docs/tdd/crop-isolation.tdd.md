# TDD evidence: crop isolation (ADR 0008)

- **Source plan:** "next steps" 1, 3 and 5, approved in chat on 2026-11-04.
- **Branch:** `feat/crop-isolation`.

## User journeys

1. As an ML engineer, I run one command on silver and see, per source, how many training
   crops show other berries, and whether the probe errs more on those crops.
2. As an ML engineer, I set `gold.crop_neighbours` and cut a release where neighbours are
   masked, or crowded crops are dropped and listed, never silently lost.
3. As the owner, I get a phone-set verdict on crops cut exactly the way the model's training
   crops were cut.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| `crop_box` moves to `data/crops.py`; `gold.crop_box` still works | `c66b391` | `db68c7d` |
| Neighbour overlap: union, inside the crop, outside the target box | `c66b391` | `db68c7d` |
| `CropPolicy` keep/mask/drop/segment: validation, JSON round trip, `data.yaml` settings, drop → keep at evaluation | `c66b391` | `db68c7d` |
| Mask fills neighbour pixels only; segment keeps only the mask; segment needs a segmenter | `c66b391` | `db68c7d` |
| `with_neighbours` lists the other boxes of the same image | `c66b391` | `db68c7d` |
| Gold: overlap, neighbours and policy per row, `crop_policy.json`; mask greys the neighbour; drop lists and counts dropped crops; the `padding` argument still works | `c66b391` | `db68c7d` |
| Embedding cache keyed by policy; a gold manifest's policy is used without arguments; legacy caches still read | `c66b391` | `db68c7d` |
| Audit: `predicted` column on issues; overlap table, per-source shares, error by overlap bucket, report files | `c66b391` | `db68c7d` |
| Phone set crops with the training policy; rejected proposals aren't neighbours; drop never skips a test berry | `c66b391` | `db68c7d` |
| `resolve_policy`: run config → gold `crop_policy.json` → legacy release → `data.yaml` | `c66b391` | `db68c7d` |
| Edge cases: mixed policies refused, segment mask shape checked, NaN boxes skipped | added with GREEN (`db68c7d`) | `db68c7d` |

250 tests pass, with 99 % coverage, and no real data or GPU is needed.

## Evidence that isn't a unit test

- **Smoke run on synthetic silver** (6 images, 9 boxes, `STRAWBERRY_DATA_DIR` in the scratchpad):
  - `scripts/crop_overlap.py` wrote `neighbours.md`, which showed overlap 0.083 for the two-berry
    images and 0 for the single-fruit ones.
  - `scripts/build_gold.py maturity_cls 0.1` released with `crop_policy.json` = keep, 0.15.

  The outputs were deleted afterwards.

## Not done (needs real data)

- The overlap numbers per source, and the error-by-overlap table, on the real silver layer.
- The ADR 0008 comparison: one gold release per policy, scored on the Kamat proxy.
- `segment`: the SAM weights (~375 MB) were never downloaded. Only the gold build passes a
  segmenter. Embedding and phone-set evaluation raise a clear error under `segment` until
  that's wired in.
