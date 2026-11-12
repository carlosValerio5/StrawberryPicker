"""Box every labeled image of a source with OWLv2: scripts/propose_source_boxes.py <SRC>.

Writes bronze/owlv2_boxes/<today>/<SRC>.csv (every box scoring >= 0.05 after NMS). Turn
it on with `box_proposals: {source: owlv2_boxes, min_score: ...}` in configs/sources.yaml.
Needs the ml extra; downloads OWLv2 (Apache-2.0, ~600 MB) on first use.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from strawberry_picker.data.bronze import ingest_box_proposals, latest_pull
from strawberry_picker.data.readers import read_annotations
from strawberry_picker.data.sources import load_sources
from strawberry_picker.labeling.proposals import non_max_suppression, owlv2_detector
from strawberry_picker.paths import Layer, layer_dir

MODEL = "google/owlv2-base-patch16-ensemble"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--revision", type=int)
    args = parser.parse_args()

    spec = load_sources()[args.source]
    pull = latest_pull(spec.name)
    if pull is None:
        parser.error(f"{spec.name} has no bronze pull")
    bronze = layer_dir(Layer.BRONZE)
    images = sorted(
        {a.image_path for a in read_annotations(spec, pull) if spec.labels.get(a.source_label)}
    )
    print(f"{len(images)} labeled images in {pull}", flush=True)
    detect = owlv2_detector(MODEL)
    rows = []
    for n, path in enumerate(images, 1):
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        boxes, scores = detect(image)
        keep = scores >= 0.05
        boxes, scores = boxes[keep], scores[keep]
        for i in non_max_suppression(boxes, scores, iou=0.3):
            rows.append(
                (path.relative_to(bronze).as_posix(), float(scores[i]), *map(float, boxes[i]))
            )
        if n % 50 == 0:
            print(f"{n}/{len(images)}", flush=True)
    frame = pd.DataFrame(rows, columns=["bronze_path", "score", "x0", "y0", "x1", "y1"])
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / f"{spec.name}.csv"
        frame.round({"score": 4}).to_csv(csv, index=False)
        landed = ingest_box_proposals(csv, "owlv2_boxes", MODEL, args.date, args.revision)
    per_image = frame.groupby("bronze_path").size()
    print(f"{len(frame)} boxes on {per_image.size} images -> {landed}")
    print(np.percentile(frame.groupby("bronze_path").score.max(), [5, 50, 95]).round(3))


if __name__ == "__main__":
    main()
