# TDD evidence: temperature scaling for the maturity probe

- **Source plan:** next-step 6 ("fix the proxy calibration"), approved in chat on 2026-11-04.
  The baseline run `20261103T023318Z` had ECE 0.130 on the Kamat proxy; ADR 0005 §B3's bar is 0.10.
- **Branch:** `feat/calibration`.

## User journeys

1. As an ML engineer, I train the baseline and get a probe whose confidences are fitted on
   gold val, with ECE before and after recorded in `metrics.json`.
2. As the owner, `evaluate_phone_set.py` scores the calibrated probabilities with no extra
   flag, and older runs keep working unchanged.

## RED → GREEN

| Behavior | RED | GREEN |
|---|---|---|
| Overconfident probe: fitted T > 1 and ECE drops below 0.10; underconfident: T < 1 | `379cace` | `0ce04d9` |
| Argmax, row sums and never-trained (zero) classes unchanged; T = 1 is the identity; T ≤ 0 rejected | `379cace` | `0ce04d9` |
| Run round trip: `calibration.json` holds T, `probe.joblib` stays plain sklearn, `load_probe` re-wraps | `379cace` | `0ce04d9` |
| Runs without `calibration.json` load as T = 1 | `379cace` (regression guard: passed on RED, as intended) | — |
| `run_baseline` fits T on val of training sources only (a held-out source's val rows are ignored) | `379cace` | `0ce04d9` |
| ECE before/after for in-distribution and each proxy; `calibration: none`; unknown method rejected; no val rows → T = 1 | `379cace` | `0ce04d9` |
| Phone-set scoring uses the calibrated probabilities (ECE 0.45 → < 0.10, same confusion matrix) | `379cace` | `0ce04d9` |

The full suite runs without real data or a GPU.

## Not done

- No re-run on real gold yet: this container has no data. The next
  `scripts/train_maturity.py` run will report the fitted T and the proxy ECE after
  calibration. That number ranks models (ADR 0007). It doesn't give an ADR 0005 verdict.
- Overripe has no gold rows, so T is fitted on unripe/ripe only. Refit once overripe lands.
