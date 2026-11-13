# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
- **Primary audience:** growers and packhouse managers, watching a live demo on a presenter's
  laptop (often on a projector). They are judging whether a phone photo can do part of a quality
  inspector's job.
- **Operator:** the presenter, who drives the demo: they open a sample tray or drop a fresh phone
  photo, then point at individual berries.

## Product Purpose
StrawberryPicker looks at a photo of strawberries and reports, for every berry it finds:
- its maturity (unripe, ripe or overripe)
- its visible defects (v1 detects mold and rot)
- a grade (premium, standard or reject)

The demo has to make one belief stick: a phone photo can sort a tray by ripeness and flag rot
before it ships. Success is a grower who understands the result of a photo within seconds,
without an explanation.

## Positioning
- **Input:** a single ordinary phone photo, with no sorting line, special camera or lab.
- **Output:** a result per berry, with the reason behind each reject visible.
- **How it works:** open-source vision models (OWLv2 finds each berry, DINOv2 features feed small
  classifiers). Labels are anchored to the USDA strawberry grade rubric (`docs/labeling/rubric.md`).

## Operating Context
- **The demo:** a desktop browser at 1280×720 to 1440 wide, served locally by `scripts/demo.py` on
  127.0.0.1, and it may run offline.
- **Inputs:** iPhone HEIC and JPEG photos.
- **Timing:** model warm-up takes minutes at server start, and a live photo takes tens of seconds on
  CPU. Precomputed sample trays open instantly.
- **Industry practice it echoes:** commercial produce QC (photo → grade, lot-level pass/fail with
  the reason that triggered it).

## Capabilities and Constraints
- **Maturity classes** are unripe, ripe and overripe. **Grades** are premium (no defect), standard
  (a minor defect) and reject (mold or rot). These come from `docs/data-contract.md` and
  `src/strawberry_picker/schema.py`.
- **Defects:** v1 has trained heads for mold and rot only. Bruise, deformed, pest and mechanical
  damage are not detected yet.
- **Display:** results are shown cleanly. Confidence is available on demand (hover or tap), not
  up front, as the owner decided.
- **Locked test set:** the demo must never read `data/bronze/phone_testset`.
- **Dependencies:** no new runtime dependencies. The server is stdlib, and the page is vanilla
  HTML/CSS/JS with no build step.

## Brand Commitments
- The name is **StrawberryPicker**.
- All on-screen text is in English.

## Evidence on Hand
- **Real photos:** owner iPhone photos (`owner_phone_2026`, HEIC, overripe and defective trays)
  and the public datasets catalogued in `docs/datasets.md`.
- **Model results:** v1 development results are in `docs/model-card-v1.md`.
- **What does not exist:** customers, testimonials, deployments, prices and accuracy guarantees.
  Do not invent any of them, and do not quote accuracy numbers as marketing claims.

## Product Principles
1. **The photo is the hero.** Every result is anchored to the berry it describes, in place.
2. **Verdict first, evidence on demand.** A grower reads the tray verdict, then drills into any berry.
3. **Every reject shows its reason.** No unexplained grades.
4. **Truthful by construction.** Show only what the model actually outputs; never imply defect
   types it cannot see.

## Accessibility & Inclusion
- **Contrast:** it must read at projector distance and under washed-out projector contrast, so
  text needs at least 4.5:1 and every color-coded state needs a text label.
- **Input:** it must be fully usable by keyboard and pointer.
