# TDD evidence: precision, macro-F1 and phone-set defect scoring

- **Source:** the owner asked for the metrics the v1 card left out, especially for the phone
  datasets (2026-11-12).
- **Branch:** `feat/v1-metrics`.

## RED → GREEN

| Behaviour | RED | GREEN |
|---|---|---|
| `per_class_precision` matches scikit-learn, with `NaN` for a class never predicted | `d06e17a` | `6a34e1f` |
| The maturity report carries per-class precision with group-bootstrap intervals | `d06e17a` | `6a34e1f` |
| The defect report carries reject precision with group-bootstrap intervals | `d06e17a` | `6a34e1f` |
| The phone-set evaluation scores a defect head: per-defect precision/recall, reject recall/precision, and check C gets a verdict; without a head, check C stays n/a | see `git log` | see `git log` |

## Evidence that isn't a unit test

- Both v1 configs were re-run with the extended evaluator. Each re-run has the same config and
  reproduces every metric v1 reported exactly, so the new precision numbers describe the
  shipped v1 models. The re-runs are `maturity_cls/20261113T014511Z` and
  `quality_cls/20261113T014518Z`.
- The results are in `docs/model-card-v1.md` (Results).
