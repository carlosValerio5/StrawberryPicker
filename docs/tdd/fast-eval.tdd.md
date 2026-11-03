# TDD evidence: lean evaluation and labeling (ADR 0007)

- **Source plan:** "unblock without 200 photos", approved in chat on 2026-11-02.
- **Branch:** `feat/fast-eval`.

## User journeys

1. As the owner, I clear a review backlog (overripe candidates, random samples) in minutes
   with the keyboard, and the next silver build uses my labels.
2. As the owner, I photograph about 15 punnets. Boxes are proposed for me, two of us label,
   and one command gives the ADR 0005 verdict, or tells us what to settle first.
3. As an ML engineer, I train the baseline and score it in-distribution and on a source it
   never saw, without waiting for the phone set.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| ADR 0005 metrics; α checked against Krippendorff (2011): 0.743 nominal, 0.815 ordinal | `ba3501c` | `325a75e` |
| Labeling tool: taxonomy validation, append-only store, undo, sampling, HTTP API | `18b961d` | `d067776` |
| Human reviews: agreement required, undo replayed, rows never dropped | `a6ce8cd` | `41b5c6a` (unit tests) |
| Silver applies reviews on rebuild; `reviewed_by` column | `41b5c6a` (integration test) | `8c12402` |
| Task builder; review files landed in bronze | `dd3e0c3` | `d87e888` |
| OWLv2 proposals (NMS); phone-set ground truth and labeler agreement | `5ba58ed` | `2ca53b6` |
| Relative task paths; phone set landed outside the registry | `6b3b17c` | `f63e0c6` |
| Shared embedding cache; maturity report; linear-probe run artifacts | `c2c5cec` | `b98afe0` |
| Baseline never trains on held-out sources | `a091e15` | `e7f8c24` |
| One-command ADR 0005 verdict | `23290a6` | `4cee765` |
| Unresolved items → adjudication task | test added in `d5254c1` (failed first) | `d5254c1` |

The full test suite runs without real data or a GPU. Pyright's pandas-stub warnings are
noise, as before.

## Evidence that isn't a unit test

- **Labeling page in a real browser** (in-app browser, 287-crop overripe task):
  - keys `3`, `r`, Enter saved "overripe + rot"
  - `n` saved "not a strawberry"
  - Backspace undid it
  - the labels file held exactly those three append-only rows

  The smoke-test labels stayed in the scratchpad.
- **Baseline run** `models/maturity_cls/20261103T023318Z` (gold `maturity_cls_v0.1`, 3,896
  crops; trained on 2,648: unripe 2,161, ripe 487; Kamat held out):

  | Set | n (groups) | QWK [95 % CI] | macro-F1 | ECE | Recall unripe / ripe |
  |---|---|---|---|---|---|
  | In-distribution (gold test) | 579 (461) | 0.910 [0.860, 0.953] | 0.955 | 0.020 | 0.98 / 0.94 |
  | Kamat proxy (never trained on) | 145 (113) | 0.674 [0.554, 0.783] | 0.834 | **0.130** | **0.72** / 0.97 |

  The proxy errors are mostly pale, partly pink "partially ripe" berries called ripe:
  borderline fruit that the training sources barely contain.
- **`evaluate_phone_set.py`** was smoke-tested on a stand-in pull (12 Kamat crops, scripted
  labels). The plumbing works. That verdict file was deleted, because it isn't a result.

## Not done

- OWLv2 weights (~600 MB) haven't been downloaded. `propose_boxes.py` downloads them on its
  first real run.
- No phone photos exist yet, so there's no ADR 0005 verdict.
- Overripe still has 0 gold rows until the overripe task is labeled and landed.
- The defect head (ADR 0005 §C) isn't built.

## Phone labeling (follow-up)

| Behavior | RED | GREEN |
|---|---|---|
| Token on every route (query → HttpOnly cookie, or `X-Token`); per-phone labeler names with their own progress and undo; names validated; touch page | `913c433` | the next commit |

- **Checked at phone size** (375×812, in-app browser, dark mode):
  - 403 without the token
  - name prompt
  - Overripe, Mold and Rot taps highlight
  - Save advances the count, Undo restores it
  - the labels file held exactly the expected rows
- **Two quick taps at once:** in the emulator, two taps fired back to back (no pause) lost
  the second one. With a normal pause, every tap registered. I haven't checked this on a
  real phone.
- **Address:** `--lan` first chose a VPN address (`10.2.0.2`), unreachable from Wi-Fi. It
  now prefers macOS `en0`/`en1`, with a `--host` override. Checked from this Mac at the
  Wi-Fi address: 403 without the token, 200 with it.
