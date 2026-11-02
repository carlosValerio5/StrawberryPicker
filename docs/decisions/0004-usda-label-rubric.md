# 0004 — Label definitions follow the USDA strawberry grade standard

- **Status:** accepted
- **Date:** 2026-11-02

## Context

The v0 taxonomy ([data-contract.md](../data-contract.md)) gave each class a short visual
cue, written by us. Six public datasets draw their ripeness lines differently, and our
own labelers need a rule they can apply the same way. Without an outside reference, the
claim that a label is right only means that we agree with ourselves.

The USDA AMS *United States Standards for Grades of Strawberries* (effective 2006-02-23)
defines color, overripe, undeveloped, damage and serious damage. Pickers, packers and
inspectors already use those definitions.

## Decision

1. **The classes don't change.** Maturity stays `unripe` / `ripe` / `overripe`, the
   six defects stay, and the grade rule stays.
2. **Each class is defined by [labeling/rubric.md](../labeling/rubric.md)**, anchored to
   the USDA text:
   - **`ripe`:** at least ¾ of the surface is pink or red (§51.3115). "Pink" now counts,
     where the v0 cue said "evenly red".
   - **`overripe`:** a visual proxy for USDA's "dead ripe, becoming soft" (§51.3120).
     Two or more visible signs are required.
   - **Defects:** each points to the USDA damage wording (§51.3115, §51.3121, §51.3123).
3. **Labelers may answer `unsure`**, and those rows are adjudicated rather than guessed.

## Consequences

- Public-source mappings are re-checked against the rubric during the audit. Strawberry-DS
  "Late-Turning" and Yurdakul "semi-ripe" are the borderline cases.
- Our grade stays defect-only. USDA grades also use color, severity, cap and size. The
  rubric lists these gaps. Adding severity or a "missing cap" flag would be a taxonomy
  change and needs its own ADR.
- Softness, the core of USDA "overripe", can't be seen in a photo. Disagreement on
  `overripe` in the test set shows how much the visual proxy costs.
