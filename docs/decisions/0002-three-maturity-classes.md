# 0002 — Three maturity classes

- **Status:** accepted
- **Date:** 2026-11-02

## Context

The v0 taxonomy proposed four maturity classes: `unripe`, `turning`, `ripe`,
`overripe`. The boundary between `turning` and the classes next to it is fuzzy,
so labelers would disagree on it and the class would be hard to fill with enough
examples.

## Decision

Maturity has three ordinal classes: `unripe < ripe < overripe`. Fruit that
would have been `turning` (partly red) is labeled `unripe`, because it isn't
ready to pick. The quality classes (defects and grade) are unchanged.

## Consequences

- When the bronze → silver step maps public datasets that use a "turning",
  "half-ripe", or "partially ripe" label, it maps that label to `unripe`.
- If a picking decision later needs a finer ripeness scale, reintroduce the
  class with a new ADR and a MAJOR bump of the gold datasets.
