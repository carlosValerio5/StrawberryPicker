# Overripe time-lapse capture

Public datasets have almost no overripe strawberries. This procedure produces them
cheaply and with honest labels: we watch the same fruit go past its peak.

## Setup

- 3–4 punnets of strawberries from different stores or varieties. Pick ripe fruit.
- 2–3 plain backgrounds (white, wood, a green tray) so the model can't learn the background.
- A phone on a fixed stand, about 25 cm above the table, with daylight or one fixed lamp.
- Number each fruit with a small paper tag outside the frame, so each fruit stays
  identifiable across days.

## Daily (days 0–10)

1. Photograph each fruit alone, filling about a third of the frame. Take 2 angles.
2. Name files `fruit<NN>_day<DD>_<angle>.jpg`, for example `fruit07_day04_a.jpg`.
3. Keep the fruit at room temperature so it ages within about a week.

## Labeling

- Label each fruit and day independently in a CSV: `file, maturity, defects`.
  - Ripe: still glossy and firm.
  - Overripe: dull, dark, soft, or leaking.
  - Record mold or rot as defects when they appear.
- Don't label by day number alone. Fruit ages at different speeds.

## Ingest

- Put each fruit's photos in its own folder, `<label>/fruit<NN>/`.
- Register the source as `own_timelapse_2026_11` in `configs/sources.yaml` with
  `format: imagefolder` and **`group_by: parent_dir`**. That keeps every photo of one
  fruit in the same split. Otherwise the test set would contain the same fruit the
  model trained on.
