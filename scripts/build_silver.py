"""Rebuild data/silver from bronze: uv run python scripts/build_silver.py."""

from __future__ import annotations

import logging

from strawberry_picker.config import load_config
from strawberry_picker.data.silver import build_silver
from strawberry_picker.data.sources import load_sources


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_config("data")["silver"]
    result = build_silver(
        load_sources(),
        min_side_px=settings["min_side_px"],
        jpeg_quality=settings["jpeg_quality"],
        near_duplicate_max_distance=settings["near_duplicate_max_distance"],
    )
    print(f"{len(result.manifest)} fruit rows from {result.manifest.image_id.nunique()} images")
    print(result.manifest.groupby("source_name").size().to_string())
    if not result.rejected.empty:
        print("\nrejected:")
        print(result.rejected.groupby(["source_name", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
