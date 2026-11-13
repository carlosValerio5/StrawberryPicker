# TDD evidence: inspection demo (ADR 0012)

- **Source plan:** "StrawberryPicker inspection demo", approved in chat on 2026-11-12.
- **Branch:** `feat/inspection-demo`.

## User journeys

1. As a presenter, I open a sample tray and the report appears instantly, even while the model is
   still warming up.
2. As a presenter, I drop a phone photo (HEIC, JPEG or PNG) and get a report for it, with a visible
   reading state while the model works.
3. As a grower watching, I read the tray's verdict and why it fired, then point at any berry to see
   its ripeness, defects and grade.
4. As anyone, I can't turn the locked phone set into a demo sample by accident.

## Behaviour covered by tests (`tests/test_demo.py`, no ML, no GPU)

| Behaviour | Tests |
|---|---|
| Tray summary: empty photo, all premium, mixed with reject, standard from ripeness or a minor defect, no defect head, grade derived from defects | `test_summary_*` |
| Page routes serve files with the right content type; traversal and missing files are 404 | `test_page_routes_*`, `test_traversal_*`, `test_missing_page_dir_is_404` |
| The shipped page links its assets, every element id `app.js` uses exists, and each font ships with its OFL license | `test_the_shipped_page_and_its_assets_are_served` |
| Status reports warming → ready and names the runs; a failed load is reported | `test_status_*`, `test_a_failed_load_is_reported` |
| Upload → job → done with berries and summary; large photos are downscaled and boxes follow | `test_upload_then_poll_until_done`, `test_large_photos_*` |
| Junk/empty body → 400, oversize → 413, upload while warming → 503, prediction failure → job error | `test_junk_*`, `test_oversize_*`, `test_upload_while_warming_is_503`, `test_a_failing_prediction_*` |
| Precomputed samples load as finished jobs; a broken sample is skipped; precompute refuses the locked set | `test_precomputed_samples_*` and the precompute tests |

Full suite: 350 passed, coverage 99 % (`demo/` at 99–100 %).

## Evidence that isn't a unit test

- **Live run.** `scripts/demo.py --precompute` was run on 7 owner photos (`owner_phone_2026`).
  The page was then checked in a browser at 1440×900, 1280×720 (projector) and 390 wide, covering:
  - the empty state with the contact sheet
  - a reject tray and a two-berry tray
  - keyboard stepping through berries
  - the berry slip
  - the model-view toggle
- **Design review.** The impeccable finish review checked the page against the direction contract
  (`.impeccable/surfaces/`). The screenshots stay local in `.impeccable/review/`, which is
  git-ignored.
