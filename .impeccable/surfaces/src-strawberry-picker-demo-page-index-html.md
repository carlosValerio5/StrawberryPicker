---
version: 1
slug: "src-strawberry-picker-demo-page-index-html"
primary_target: "src/strawberry_picker/demo/page/index.html"
related_targets: []
---

# Surface brief: inspection demo (src/strawberry_picker/demo/page/index.html)

Scope: the live demo page served by `scripts/demo.py`. Mode: **Operate** (the presenter runs a tray inspection on stage while growers and packhouse managers watch a projector).
Audience/job: a grower should read a tray's result within seconds, then see why any berry was rejected.
Content: real photos (precomputed sample trays plus live upload) and the real model output only (maturity, mold/rot, grade). No invented claims or accuracy numbers.
Constraints: offline-capable, vanilla HTML/CSS/JS, legible at 1280×720 on a projector, confidence shown on demand only (owner decision).
Scene: a meeting room with the lights half on and a projector washing out blacks, so the ground is light, the ink is dense and contrast is high.

## Direction contract

THESIS: Every photo is filed as a one-tray market report, the way USDA Market News reports a commodity: dateline, a monumental verdict headline, the condition line, then one lot line per berry. It refuses the AgTech KPI-card dashboard (top bar, four stat tiles, a table).

OWN-WORLD: Green-bar continuous fanfold paper. The ground is a cool paper white with pale green bands two rows deep, a punched tractor-feed strip down each edge, and a perforated fold line. The ink is dense green-black, the lot table is set in a dot-matrix-like mono with tabular figures, and headlines are a condensed wire-service grotesk. Ripeness has three inks, each with its own line form: unripe chartreuse dashed, ripe strawberry-red solid, overripe aubergine doubled. A reject is a vermilion rubber stamp, slightly askew, with ink grain. Layering is the only rank: a berry's detail is a torn-off slip laid over the report.

STORY: The presenter opens a sample tray or drops a phone photo. The report prints: the verdict says ship, sort or reject the tray and why, and every berry is outlined on the photo and listed as a lot line. Hovering a line or an outline links the two and lays the berry's slip down (crop, ripeness, defects, grade, confidence on demand). The grower believes a phone photo can sort a tray.

FIRST VIEWPORT: At 1280×720, the left tractor strip, then the tray photo as the report plate (about 55% width, full height), its outlines carrying lot codes. On the right are the dateline ("STRAWBERRYPICKER INSPECTION REPORT · tray file · time") and the verdict headline at display scale ("REJECT TRAY", with the stamp). Under it sit the condition line with its reasons, the ripeness and grade tallies as a printed tally bar, then green-bar lot lines that scroll inside the sheet. The primary action ("File a new photo", plus the sample tray picker) is a printed form field at the top of the sheet. Before any report, the sheet shows an unprinted form: an empty plate with drop instructions and the sample tray index.

FORM: Market News Wire (USDA Market News terminal report on green-bar fanfold), position 6 on my ordered list, seed key 2ada607b. Raises: line-form state (from Emission-line rail), layering as the only rank (from Acetate tab board), an honest raw/model view toggle (from Bitmap specimen), right-aligned tabular figures (from Cassette j-card). Signature interaction: the report "prints" line by line, with the paper feeding down one band per lot line, when a result arrives (about 600 ms total, skipped under reduced motion), and the hover link between outline and lot line. Motion grammar: paper feed (translateY in band steps, ease-out) and stamp impact (scale 1.15 to 1 with a slight rotate); nothing else moves.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved
- Font sourcing (self-hosted OFL faces) needs the owner's OK to download.
