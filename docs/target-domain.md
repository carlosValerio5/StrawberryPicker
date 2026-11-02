# Target domain — where the model has to work

Every "is this dataset fit?" question is answered against this page. A dataset only
fits if it looks like these photos, or if we can measure how far it is from them.

Confirmed with the project owner on 2026-11-02:
- Phone photo, hand-held.
- Test photos come from the owner's phone.
- A classmate is the second labeler.
- No Brix measurement for now.

Everything else on this page is an assumption. Correct it if it's wrong.

## The photo the model receives

| Property | Expected | Assumed or confirmed |
|---|---|---|
| Device | Phone main camera, hand-held | confirmed |
| Subject | Picked strawberries: one berry, a few on a plate, or a punnet | assumed |
| Distance | 15–50 cm. A single berry fills roughly 10–60 % of the frame | assumed |
| Lighting | Indoor and mixed: window daylight, warm LED, fluorescent, shop lights. Flash sometimes | assumed |
| Background | Hand, white plate, plastic punnet, table or cutting board, shop shelf | assumed |
| Cap (calyx) | Usually on, sometimes removed | assumed |
| File format | JPEG. iPhones default to **HEIC**, which Pillow and our readers can't open (`IMAGE_SUFFIXES` has no `.heic`) | confirmed risk |
| Resolution | 12 MP or more. Berries are large in the frame, unlike Riseholme's ~63 px crops | assumed |

## What this rules in and out

- **Out of scope:**
  - berries still on the plant, with leaves, occlusion and outdoor sun
  - fixed conveyor cameras
  - photos taken at night under a single colored light source
- **Several berries per photo** happen often, so the full app still needs a detector.
  The classifier is judged on its own first, using one berry per test photo (see the
  [test-set protocol](labeling/test-set-protocol.md)).
- **Color is the signal**, and phone auto white balance shifts reds under warm LEDs.
  Lighting is recorded for every test photo, so errors can be broken down by light.

## How far each public source is from this (before measuring)

| Source | Closeness | Main gap |
|---|---|---|
| `fresh_rotten_fruits_2022` | close | DSLR, shop and field shots. Already-picked fruit |
| `kamat_2025` | medium | Pastel backgrounds replaced in software, a likely shortcut |
| `strawberry_ds_2023`, `yurdakul_2026` | far | Fruit still on the plant in greenhouses, several per image |
| `strawberry_disease_2025` | far | Mostly on the plant, field disease photos |
| `riseholme_2021` | farthest | ~63 px robot crops on the plant, polytunnel light |

This table is a guess. The domain-shift audit (UMAP of DINOv2 embeddings, plus a
classifier that predicts which source an image came from) replaces it with numbers.
Only the locked phone test set decides whether the model works.
