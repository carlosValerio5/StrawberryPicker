# 0009 — OWLv2 as the stopgap detector for inference

- **Status:** proposed
- **Date:** 2026-11-04

## Context

The maturity classifier was trained on one-berry crops. Phone photos usually show several
berries (punnets, plates, hands; see [target domain](../target-domain.md)), so a whole photo
given to the classifier mixes several ripeness signals and gets a meaningless answer. Until
now `src/strawberry_picker/inference/` was empty, so there was no supported way to run a
trained model on a new photo at all.

Step 1 of the [architecture](../architecture.md) pipeline (detection) has no trained model
yet. OWLv2 is already in the repo: `labeling/proposals.py` uses it to propose boxes for the
phone test set, and a person accepts or rejects each one.

## Decision

1. Inference is **detect → crop → classify**, in `strawberry_picker.inference`:
   - Boxes below a score of 0.2 are dropped, then non-maximum suppression at IoU 0.5. These
     are the same thresholds as `labeling.proposals.propose`, through the same functions.
   - Each box is cropped with gold's padded square (`data.gold.crop_box`) at the **padding
     the run was trained with**. That's `crop_padding` in the run config, or failing that the
     padding in its gold release's dataset card. Crops at inference must match crops in
     training.
   - Images are decoded the way silver decodes them (EXIF orientation, then RGB). HEIC
     gives a clear error that tells the user to export as JPEG.
2. **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache-2.0, zero-shot, query
   "a photo of a strawberry") is the default detector until a trained one exists.
3. The follow-up is **RT-DETR** (Apache-2.0, Hugging Face `transformers`), fine-tuned on
   the box-annotated sources (Strawberry-DS, Kamat, Yurdakul once verified). It replaces
   OWLv2 when it beats it on held-out box sources. It still has to be judged on photos
   that look like ours, not only on greenhouse images.
4. **YOLO (Ultralytics) stays out** because its code and weights are AGPL-3.0. Adopting it
   needs its own ADR that accepts the copyleft terms.
5. `scripts/predict.py` refuses any path that is, contains, or lies inside
   `bronze/phone_testset` (AGENTS.md rule 6).

## Known limits of OWLv2 here

- **Score threshold is a guess.** 0.2 came from the labeling workflow, where a person
  rejects false boxes. In inference nobody does. It hasn't been measured on our photos.
- **False positives:** zero-shot "strawberry" is expected to fire on flowers, calyces, leaves, red
  objects and printed strawberries on punnet labels. The classifier will still give each
  one a maturity.
- **Missed and merged berries** in tight clusters, where one box can cover two touching
  fruits. That brings back the multi-berry crop problem this ADR is meant to remove.
- **Latency:** OWLv2 base is a ViT-B at 960 px, with ~600 MB of weights. Expect seconds
  per photo on a laptop CPU (not measured yet). That's fine for batch scoring, too slow
  for a phone app.

## Consequences

- A trained run can be applied to new photos with one command, one berry at a time.
- Detector errors and classifier errors are now mixed in end-to-end results. The locked
  phone set still scores the classifier on human-accepted boxes (ADR 0007), so its verdict
  isn't affected by this choice.
- Detector quality (recall of berries, false boxes per photo) has no gate yet. It needs
  one before the app is judged end to end.
- `predict_image` takes extra `heads` (`embeddings -> fields per berry`). The defect head
  plugs in there and reuses the same embeddings.
