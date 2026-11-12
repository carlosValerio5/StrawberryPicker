# TDD evidence: flag whole-image rows that show several berries

- **Source plan:** next step 4 of the "several strawberries per image" plan, agreed in chat
  on 2026-11-04.
- **Branch:** `feat/multi-fruit-flags`.

## User journeys

1. As the owner, I run one command and learn how many whole-image rows of each source and
   label actually show a pile of berries, without the detector changing any label.
2. As the owner, I clear those flags in the labeling tool: `s` for "several berries" or a
   normal label if it really is one berry. The next silver build keeps piles out of gold.
3. As an ML engineer, I re-run the count with another threshold without re-running the
   detector.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| "Several berries" label (`multi_fruit`): valid without maturity, refused when the task doesn't offer it, older label files keep their header and read as `False` | `173f660` | `fd8d701` |
| An agreed `multi_fruit` review keeps the source label and sets `needs_review`; a split vote holds the row; old review files still load | `173f660` | `fd8d701` |
| Detector counts confident berries after NMS, only in rows with `bbox` null | `cba96bb` | `8436cf1` |
| Raw detections cached per image and detector: new thresholds and already-seen images never re-detect | `cba96bb` | `8436cf1` |
| Per-source/label summary, `flags.csv`, `summary.csv`, `report.md` | `cba96bb` | `8436cf1` |
| `multi-fruit` task: flagged, unreviewed rows only, whole image shown, detector count in `given` | `cba96bb` | `8436cf1` |

The full suite (218 tests) runs without real data, weights or a GPU. The detector is a
fake keyed on image width.

## Design notes

- **The human decides.** The detector only proposes. A row leaves gold only through an
  agreed review in `bronze/human_review/`, the same path as every other review.
- **Keep the label.** A pile of `RottenStrawberry` is probably rotten berry by berry. The
  row keeps its mapped label, so a later pass that boxes each berry can reuse it.
- **Not offered on the phone set.** `label.py` shows the `s` key only with `--multi-fruit`.
  The phone set is labeled per proposed box, where a box that isn't one berry is
  rejected with `n` (ADR 0007).
- **Schema:** `LABEL_COLUMNS` gains a trailing `multi_fruit`. The silver manifest
  columns don't change. `docs/data-contract.md` now lists the new reason in `reviewed_by`.

## Not done

- Not run on real silver: needs the ml extra and the OWLv2 weights (~600 MB).
- `min_score` 0.2 and `iou` 0.5 match `propose_boxes.py`. They aren't tuned. A pile of
  touching berries may count as one box, so the count is a lower bound.
- `evaluation/phone_set.load_pull_labels` still selects `LABEL_COLUMNS` strictly, so a
  phone-set label file written *before* this change would fail to load. None exists yet.
