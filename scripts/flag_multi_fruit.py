"""Count berries in whole-image silver rows: uv run python scripts/flag_multi_fruit.py.

Needs the ml extra (`make setup-ml`); OWLv2 weights (~600 MB, Apache-2.0) download on first
use. Raw detections are cached in reports/dataset-audit/multi_fruit/detections.parquet, so
re-running with another --min-score or --iou is instant. Nothing is relabeled: send the
flags to a person with `scripts/make_label_task.py multi-fruit`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from strawberry_picker.audit.multi_fruit import flag_multi_fruit, write_outputs
from strawberry_picker.labeling.proposals import owlv2_detector
from strawberry_picker.paths import Layer, layer_dir, reports_dir

MODEL = "google/owlv2-base-patch16-ensemble"
QUERY = "a photo of a strawberry"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.2, help="detector confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS overlap for duplicates")
    parser.add_argument("--min-count", type=int, default=2, help="berries that make a pile")
    args = parser.parse_args()

    silver = layer_dir(Layer.SILVER)
    out = reports_dir() / "dataset-audit" / "multi_fruit"
    flags = flag_multi_fruit(
        pd.read_parquet(silver / "manifest.parquet"),
        silver / "images",
        out / "detections.parquet",
        lambda: owlv2_detector(MODEL, QUERY),
        detector=f"{MODEL}|{QUERY}",
        min_score=args.min_score,
        iou=args.iou,
    )
    paths = write_outputs(flags, out, min_count=args.min_count)
    print(paths["report"].read_text())
    print(
        f"next: uv run python scripts/make_label_task.py multi-fruit --min-count {args.min_count}"
    )


if __name__ == "__main__":
    main()
