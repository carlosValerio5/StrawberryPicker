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

Add `--lan` to label from phones: open the printed link on a phone on the same Wi-Fi
(not a guest or campus network that isolates devices), type your name once, and tap. Each
phone labels under its own name, into the same file. On a laptop, the keys below also work.

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
- **Agreement and adjudication:** run `uv run python scripts/evaluate_phone_set.py --pull <folder>`.
  If any crop is disputed or `unsure`, it writes `adjudication.csv` instead of a verdict.
  Both labelers label it together with `--labeler adjudicated`. Then re-run it for the
  verdict. Krippendorff's α is computed on the double-labeled crops.

## 4. Where it lives

- **Bronze:** when labeling is finished, land the whole folder (photos, `captures.csv`,
  `task.csv` and the label files) with
  `uv run python scripts/ingest_phone_set.py <folder>`. It becomes a pull of
  `data/bronze/phone_testset/`, which is never edited.
- **Locked:** `phone_testset` is **never added to the source registry**, so silver, gold and
  training can't read it. Only `scripts/evaluate_phone_set.py` does.
