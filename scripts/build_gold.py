"""Release a gold dataset: uv run python scripts/build_gold.py maturity_cls 0.1."""

from __future__ import annotations

import argparse

from strawberry_picker.config import load_config
from strawberry_picker.data.crops import CropPolicy, NeighbourMode, sam_segmenter
from strawberry_picker.data.gold import GoldTask, build_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[t.value for t in GoldTask])
    parser.add_argument("version", help="MAJOR.MINOR, e.g. 0.1")
    parser.add_argument(
        "--commercial-only", action="store_true", help="drop sources that forbid commercial use"
    )
    args = parser.parse_args()

    settings = load_config("data")["gold"]
    policy = CropPolicy.from_settings(settings)  # ADR 0008: crop_padding + crop_neighbours
    out = build_gold(
        args.task,
        args.version,
        ratios=settings["split_ratios"],
        seed=settings["seed"],
        policy=policy,
        segment_fn=sam_segmenter() if policy.neighbours is NeighbourMode.SEGMENT else None,
        jpeg_quality=settings["jpeg_quality"],
        commercial_only=args.commercial_only,
    )
    print(f"released {out}")
    print((out / "stats.json").read_text())


if __name__ == "__main__":
    main()
