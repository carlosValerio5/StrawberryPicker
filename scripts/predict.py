"""Per-berry maturity, defects and grade for new photos.

Usage: uv run python scripts/predict.py <image or folder>...

Detects every berry (OWLv2, ADR 0009), crops it under the run's crop policy (ADR 0011), and
classifies the crop (maturity, calibrated per ADR 0005). The defect head (ADR 0010) adds
``defects`` and ``grade`` to every berry from the same embeddings.

Uses the newest models/maturity_cls/* run unless --run says otherwise, and the newest
models/quality_cls/* run as the defect head unless --quality-run or --no-quality says
otherwise. Prints one JSON record per image, or writes them to --out. --annotate also saves
each photo with boxes and labels to reports/predictions/<run_id>/.
Refuses data/bronze/phone_testset: the locked set is read only by evaluate_phone_set.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strawberry_picker.inference import (
    LockedSetError,
    ensure_not_locked,
    image_paths,
    load_predictor,
    newest_run,
    predict_paths,
)
from strawberry_picker.paths import reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="images and/or folders of images")
    parser.add_argument("--run", type=Path, help="maturity run folder (default: newest)")
    parser.add_argument(
        "--quality-run", type=Path, help="defect run folder (default: newest quality_cls run)"
    )
    parser.add_argument("--no-quality", action="store_true", help="skip the defect head")
    parser.add_argument("--padding", type=float, help="crop padding (default: the run's policy)")
    parser.add_argument("--min-score", type=float, default=0.2, help="detector score threshold")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--annotate", action="store_true", help="save annotated copies")
    args = parser.parse_args()

    try:
        ensure_not_locked(args.inputs)  # before any model is loaded
        paths = image_paths(args.inputs)
    except (LockedSetError, FileNotFoundError) as error:
        parser.error(str(error))
    if not paths:
        parser.error("no images found")
    run = args.run or newest_run("maturity_cls")
    if run is None:
        parser.error("no model run yet: uv run python scripts/train_maturity.py")
    quality_run = None
    if not args.no_quality:
        quality_run = args.quality_run or newest_run("quality_cls")

    try:
        predictor = load_predictor(
            run, quality_run=quality_run, padding=args.padding, min_score=args.min_score
        )
    except ValueError as error:
        parser.error(str(error))
    annotate_dir = reports_dir() / "predictions" / run.name if args.annotate else None
    records = predict_paths(paths, predictor, annotate_dir=annotate_dir)
    text = json.dumps(
        {
            "run": str(run),
            "quality_run": str(quality_run) if quality_run else None,
            "images": records,
        },
        indent=2,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"{len(records)} images -> {args.out}")
    else:
        print(text)
    if annotate_dir is not None:
        print(f"annotated images -> {annotate_dir}")


if __name__ == "__main__":
    main()
