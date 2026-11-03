"""Propose fruit boxes for a folder of phone photos: uv run python scripts/propose_boxes.py DIR.

Writes DIR/task.csv for scripts/label.py. Needs the ml extra; the first run downloads
OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0, ~600 MB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from strawberry_picker.labeling.proposals import owlv2_detector, propose
from strawberry_picker.labeling.tool import write_task

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--min-score", type=float, default=0.2)
    args = parser.parse_args()

    photos = [p for p in args.folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES]
    if not photos:
        heic = [p for p in args.folder.iterdir() if p.suffix.lower() == ".heic"]
        hint = " (found HEIC: set the camera to 'Most Compatible')" if heic else ""
        parser.error(f"no JPEG/PNG photos in {args.folder}{hint}")
    task = propose(photos, owlv2_detector(), min_score=args.min_score)
    task["image_path"] = [Path(p).name for p in task.image_path]  # relative: survives ingest
    path = write_task(task, args.folder / "task.csv")
    print(f"{len(task)} boxes in {len(photos)} photos -> {path}")
    print(f"next: uv run python scripts/label.py {path} --labeler <you> --blind")


if __name__ == "__main__":
    main()
