# Locked phone test set — capture and labeling protocol

The one dataset that decides whether the model works ([target-domain.md](../target-domain.md)).
It's built before any training. Training code must never read it, and nobody tunes
against it. The pass/fail rules it feeds are fixed in ADR
[0005](../decisions/0005-acceptance-criteria.md).

## What to collect

- **About 200 berries, one berry per photo, one photo per berry.** One berry per photo
  tests the classifier without a detector. One photo per berry means no berry can count
  twice.
- **Target mix:** ~70 unripe, ~70 ripe, ~60 overripe, with as many visible defects as
  you can find. Buy several batches (different shops or days). Unripe berries are the
  hardest to buy, so look for white-shouldered berries in cheap punnets.
- **Overripe:** leave berries out for 2–5 days. **Use different batches from the
  [overripe time-lapse](overripe-timelapse.md)**, which is training data. A berry in
  both sets would leak.

## How to shoot

1. **JPEG, not HEIC.** On iPhone: Settings → Camera → Formats → *Most Compatible*.
   Our pipeline can't read HEIC.
2. Main lens at 1×. No portrait mode, no filters, no Live Photo edits. Leave flash as
   it would normally be used, and note it if it fires.
3. The berry is centered and fills roughly 30–60 % of the frame, with its side and
   shoulders visible (not top-down on the cap).
4. Spread photos across the lighting and background combinations, so no condition holds
   more than ~25 % of the set:

| Lighting | Backgrounds |
|---|---|
| `window` (daylight), `warm_led`, `cool_led_or_fluorescent`, `flash` | `hand`, `white_plate`, `punnet`, `table` |

5. Fill in `captures.csv` as you go. One row per photo, dates as `YYYY-MM-DD`:

```csv
file,berry_id,batch,captured_on,lighting,background,cap_on
IMG_0001.jpg,B001,batch-2026-11-05,2026-11-05,window,white_plate,true
```

## How to label

- **Two labelers:** the owner and a classmate. Each labels **every photo
  independently**, using only the [rubric](rubric.md). Neither sees the other's labels
  or any model output. Use Label Studio (image classification, choices plus a
  multi-select for defects), or a spreadsheet with this layout:

```csv
file,maturity,defects,unsure
IMG_0001.jpg,ripe,,false
IMG_0002.jpg,overripe,bruise|rot,false
```

- `maturity` is one of `unripe`, `ripe`, `overripe`. `defects` is pipe-separated, and
  empty means none seen. `unsure` is `true` or `false`.
- Keep both raw label files. **Agreement is computed on the raw files** (Krippendorff's
  α, ordinal for maturity and nominal for each defect).
- **Only then adjudicate.** The two labelers meet, look at every disagreement and every
  `unsure` row, and agree on one label. The adjudicated labels become ground truth. A
  photo still unresolved after adjudication is dropped and counted in the report.

## Where it lives

- **Bronze:** the photos plus `captures.csv` and both raw label files go into
  `data/bronze/phone_testset_<yyyy_mm>/<date>/` like any other pull (never edited).
- **Gold:** it is released as its own gold dataset. It is never mixed into a training
  release.
- **Code:** reading the capture and label CSVs, and keeping it out of training, is
  built in the next step. Until then this page is the contract.
