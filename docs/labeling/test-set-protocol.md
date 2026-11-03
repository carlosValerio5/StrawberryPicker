# Locked phone test set — capture and labeling protocol (v2)

The one dataset that decides whether the model works ([target-domain.md](../target-domain.md)).
Training code must never read it, and nobody tunes against it. The pass/fail rules are in
ADR [0005](../decisions/0005-acceptance-criteria.md). This lean version is set by ADR
[0007](../decisions/0007-lean-phone-test-set.md). **Owner time: about 45 minutes.**

## 1. Shoot (about 15 minutes)

- **15–20 photos with at least 120 berries in total.** Punnets, plates and a few berries
  in a hand all count.
- **Mix of ripeness:**
  - **Unripe:** look for white-shouldered berries.
  - **Overripe:** buy the shop's reduced "use today" berries, or leave a punnet out for a
    few days.
  - **Different batches only:** none of these berries may also appear in the
    [time-lapse](overripe-timelapse.md) training photos.
- **JPEG, not HEIC.** On iPhone: Settings → Camera → Formats → *Most Compatible*.
  Use the 1× lens, no portrait mode, no filters.
- **Lighting and backgrounds:** spread photos across `window`, `warm_led`,
  `cool_led_or_fluorescent` and `flash`, and backgrounds `hand`, `white_plate`, `punnet`
  and `table`, so no condition holds more than ~25 % of the photos.
- **Capture log:** fill in `captures.csv`, one row per **photo**, with dates as
  `YYYY-MM-DD`:

```csv
file,batch,captured_on,lighting,background
IMG_0001.jpg,batch-2026-11-05,2026-11-05,window,punnet
```

## 2. Proposed boxes (automatic)

```bash
uv run python scripts/propose_boxes.py path/to/photos
```

OWLv2 draws a box around every likely strawberry and writes a labeling task. Nobody draws
boxes by hand.

## 3. Label (about 10 minutes per person)

```bash
uv run python scripts/label.py <task.csv> --labeler alice --blind
```

The page shows one crop at a time, with the [rubric](rubric.md) beside it:

| Key | Meaning |
|---|---|
| `1` / `2` / `3` | unripe / ripe / overripe |
| `b` `m` `r` `d` `p` `c` | toggle bruise / mold / rot / deformed / pest_damage / mechanical_damage |
| `Enter` | save and go to the next crop |
| `n` | not a strawberry (the box is wrong) |
| `u` | unsure: goes to adjudication |
| `Backspace` | undo the last one |

- **Labeler 1** labels every crop.
- **Labeler 2** labels the blind random 50 (`--sample 50`), without seeing labeler 1's
  answers.
- **Agreement and adjudication:** Krippendorff's α is computed on the 50. Then both
  labelers settle every disagreement and every `unsure`.

## 4. Where it lives

- **Bronze:** the photos, `captures.csv`, the proposals, and both label files are one
  bronze pull of the source `phone_testset_<yyyy_mm>`, which is never edited.
- **Locked:** the source is registered with `locked: true`. Gold training releases refuse
  locked sources, and only the evaluation script reads them.
