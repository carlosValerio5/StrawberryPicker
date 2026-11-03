# 0008 — Machine box proposals for image-level labels

- **Status:** accepted (owner approved boxing Fresh/Rotten with OWLv2 on 2026-11-03)
- **Date:** 2026-11-03

## Context

`fresh_rotten_fruits_2022` labels whole photos: berries on white paper, often in piles.
Every overripe training example came from it, and the retrained probe learned *the
source* instead of overripeness. Overripe recall was 1.00 on Fresh/Rotten but 0.20 on the
Kamat proxy (finding #17 in `docs/datasets.md`).

## Decision

1. **Proposals as data:**
   - OWLv2 (`google/owlv2-base-patch16-ensemble`, Apache-2.0) boxes "a photo of a
     strawberry" on every labeled image of the source.
   - Every box scoring ≥ 0.05 after NMS (IoU 0.3) is landed as machine annotations in
     `bronze/owlv2_boxes/<date>/<source>.csv`, by `scripts/propose_source_boxes.py`.
   - Images with no box at all get a zero-score row, so "searched, found nothing" is
     recorded.
2. **Silver rule** (`box_proposals: {source, min_score}` in the registry):
   - **Exactly one** box ≥ `min_score`: that box becomes the crop. The crop id stays
     `<image>_0`, so existing human reviews still apply.
   - **Two or more** (a pile) or **none**: the image is held out of gold, even after a
     review, because one label can't be pinned to one berry.
3. **`min_score = 0.3` for Fresh/Rotten**, chosen by drawing the boxes:
   - all 24 sampled multi-box images were real groups or piles
   - all 16 sampled single-box images were one berry with a tight box
   - no-box images were shrivelled berries or piles that OWLv2 doesn't recognise

## Consequences

- About 46 % of Fresh/Rotten strawberry images are piles, and 25 have no confident box;
  all of these leave gold.
- Overripe training examples from this source drop from 139 to a few dozen. They're
  cleaner, but it's a small class. Overripe photos from more settings are still the real fix.
- The rule is generic: any image-labeled source can opt in.
- **Result:** the crops did *not* fix overripe transfer. Kamat-proxy overripe recall was
  0.15 (it was 0.20 with whole photos). Overripe doesn't transfer in either direction
  between the two sources (finding #18), so the shortcut is in what the sources show, not
  in the background. The crops stay because they remove the pile-label errors.
