# 0012 — Inspection demo: a stdlib server and a vanilla page, shown as a tray report

- **Status:** accepted (the owner approved the plan and chose the visual direction on 2026-11-12)
- **Date:** 2026-11-12

## Context

v1 runs end to end (`scripts/predict.py`), but its only visible output is a JSON file and an
annotated JPEG. The owner needs to show the model to growers and packhouse managers in a live
laptop/projector demo. The repo's one UI, the labeling page (`labeling/tool.py`), is an internal
tool for labeling crops, so it is the wrong surface.

Commercial produce-QC apps (Clarifruit/Clarifresh) set the expectations for this kind of tool:
- a photo is graded in seconds
- the verdict for a lot comes with the tolerance that triggered it
- defects roll up per lot

USDA strawberry grades are also lot tolerances. So the demo reports a **tray verdict with its
reasons**, then one line per berry.

## Decision

- **Server.** A new package `src/strawberry_picker/demo/` runs a **stdlib** `ThreadingHTTPServer`,
  the same pattern as the labeling tool, with no web framework.
  - `summary.py` turns per-berry predictions into the tray verdict, counts and plain-English
    reasons. It is pure logic with no ML.
  - The model loads once in a background thread, so the page shows "warming up" instead of
    hanging.
  - Live photos queue on one worker thread.
- **Precomputed samples.** `scripts/demo.py --precompute` writes sample trays to `reports/demo/`
  (git-ignored), so a presenter is never waiting on CPU inference. It calls
  `inference.ensure_not_locked` first, so the locked phone set can never become a sample (rule 6).
- **Page.** It lives in `demo/page/` as **vanilla HTML/CSS/JS** with no build step.
  - The page draws the berry outlines itself (SVG over the served photo), so they stay crisp and
    clickable and are linked to the lot table. Pillow's `annotate()` is not used here.
  - **Clean results.** By the owner's decision, the page shows the class, defects and grade.
    Confidence is available on demand per berry, not up front.
  - **Truthful scope.** The page names the defects actually checked (mold, rot) and never shows
    accuracy claims.
- **Visual world.** "Market News Wire" was chosen through the impeccable design workflow. Each
  photo is filed as a one-tray market report on green-bar fanfold paper. The direction contract is
  in `.impeccable/surfaces/`, and the resulting system is in `DESIGN.md`.
- **Fonts.** Two **SIL OFL 1.1** fonts are self-hosted so the demo works offline: Archivo and
  Martian Mono, as `woff2` from Fontsource 5.1.1, with their license files next to them. They are
  recorded in `docs/architecture.md`.

## Consequences

- No new Python dependencies. The fonts are the only new third-party files, about 125 KB together.
- The page has no JS test harness. Tests cover the server routes, the summary rules and that the
  page assets are served. The look is checked by screenshots in the impeccable finish review.
- A real photo still takes tens of seconds on CPU. The precomputed samples are the presenter's
  default path, and live upload is the "try your own" moment.
- If the summary rules change, stored samples pick the change up on load, because the server
  recomputes each sample's summary.
