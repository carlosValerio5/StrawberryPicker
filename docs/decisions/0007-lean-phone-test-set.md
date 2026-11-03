# 0007 — A lean phone test set and a development proxy

- **Status:** accepted (the project owner approved the plan on 2026-11-02). Amends
  ADR [0005](0005-acceptance-criteria.md) §A–B and the
  [test-set protocol](../labeling/test-set-protocol.md), as §F allows before any
  locked-set results exist.
- **Date:** 2026-11-02

## Context

The protocol asked for ~200 single-berry phone photos, each labeled in full by two
people. That takes a day the team doesn't have, and every model decision was waiting on it.

## Decision

1. **Photos:** at least **120 berries** taken from **15–20 multi-berry photos** (punnets,
   plates, hands) instead of 200 single-berry photos.
2. **Boxes:** OWLv2 (zero-shot, Apache-2.0) proposes the boxes. A person keeps or rejects
   each box while labeling.
3. **Overripe berries** may be bought reduced-price ("use today") the same day, instead
   of being aged at home.
4. **Labeling:**
   - **One labeler labels every berry.**
   - **A second labeler labels a blind random 50.**
   - ADR 0005 §A's α ≥ 0.67 is computed on those 50.
   - The full set uses labeler 1's labels, after adjudicating the 50 overlapping ones.
5. **Statistics:** the photo is the sampling unit. Bootstrap intervals resample
   **photos**, not berries, because berries in one photo share light and background.
6. **Development proxy:**
   - Until the phone set exists, **`kamat_2025` is held out of training** and scored as
     a proxy target. It's hand-held phone shots of picked fruit, never trained on.
   - Proxy scores rank models against each other. **They never give an ADR 0005 verdict**:
     Kamat's labels are noisy, and about a quarter of its "unripe" boxes are flowers.

## Consequences

- Owner time drops from about a day to about 45 minutes: 15 minutes of photos, plus about
  10 minutes of labeling per person in `scripts/label.py`.
- **Error bars are wider:** with ~40 berries per class, recall intervals are about
  ±15 points. A borderline result is settled by shooting more photos, never by moving the bar.
- Kamat leaves the training pool, which costs 145 maturity crops, until the phone set
  replaces it as the referee.
