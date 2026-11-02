# 0003 — Dataset selection and ambiguous label mappings

- **Status:** accepted (defaults, open to change)
- **Date:** 2026-11-02

## Context

The dataset research ([datasets.md](../datasets.md)) found six usable public sources,
but none of them use our exact taxonomy. Three questions were left open when the plan
was approved. These defaults were taken so work could proceed.

## Decision

1. **Kamat "rotten" maps to `maturity: overripe`, `defects: [rot]`, with `needs_review`.**
   It's the only public source of past-peak fruit, but rotten and overripe aren't the
   same thing. The rows stay out of gold until a human confirms each crop.
2. **Overripe gets our own data.** We photograph store-bought fruit daily as it ages
   ([labeling/overripe-timelapse.md](../labeling/overripe-timelapse.md)). Until then,
   gold maturity datasets will be short of overripe examples.
3. **Assume academic use.** Sources with unknown or non-commercial licenses may be used
   for research. Each silver row carries `commercial_use`, and
   `build_gold --commercial-only` excludes them for any commercial build.
4. **Unverified sources ship disabled.** Mappings are written from the papers, but a
   source is only enabled after its downloaded layout and class order are checked.

## Consequences

- `needs_review` rows (semi-ripe, late-turning, rotten, Riseholme anomalous) need a
  review pass before they add to gold. Until then they're visible in silver only.
- Revisit decision 3 before any non-academic use. Riseholme and Yurdakul have no stated
  dataset license.
