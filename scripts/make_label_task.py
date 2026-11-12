"""Turn a review backlog into a labeling task: uv run python scripts/make_label_task.py overripe.

Kinds:
  overripe       overripe candidates still waiting for review (Kamat, Fresh/Rotten)
  needs-review   every unreviewed needs_review row (optionally --source)
  random-sample  the ADR 0006 random sample of --source (from scripts/dataset_audit.py)
  flagged        the confident-learning suspects of --source (from scripts/dataset_audit.py)
  multi-fruit    whole-image rows the detector counts --min-count+ berries in (from
                 scripts/flag_multi_fruit.py); label with `scripts/label.py --multi-fruit`
"""

from __future__ import annotations

import argparse

import pandas as pd

from strawberry_picker.audit.embeddings import with_crop_ids
from strawberry_picker.labeling.tasks import crop_task, multi_fruit_task
from strawberry_picker.labeling.tool import write_task
from strawberry_picker.paths import Layer, layer_dir, reports_dir

KINDS = ("overripe", "needs-review", "random-sample", "flagged", "multi-fruit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--source", help="source name (required for random-sample / flagged)")
    parser.add_argument("--min-count", type=int, default=2, help="multi-fruit: berries in a pile")
    args = parser.parse_args()

    silver = with_crop_ids(pd.read_parquet(layer_dir(Layer.SILVER) / "manifest.parquet"))
    unreviewed = silver.needs_review & silver.reviewed_by.isna()
    if args.kind == "multi-fruit":
        flags = pd.read_csv(reports_dir() / "dataset-audit" / "multi_fruit" / "flags.csv")
        if args.source:
            flags = flags[flags.source_name == args.source]
        task = multi_fruit_task(
            silver.drop(columns="crop_id"), flags, layer_dir(Layer.BRONZE), min_count=args.min_count
        )
        if task.empty:
            parser.error("nothing to label")
        name = args.kind + (f"_{args.source}" if args.source else "")
        path = write_task(task, reports_dir() / "labeling" / f"{name}.csv")
        print(f"{len(task)} images -> {path}")
        print(f"next: uv run python scripts/label.py {path} --labeler <you> --multi-fruit")
        return
    if args.kind in ("random-sample", "flagged"):
        if not args.source:
            parser.error(f"{args.kind} needs --source")
        folder = "random_sample" if args.kind == "random-sample" else "review"
        listed = pd.read_csv(reports_dir() / "dataset-audit" / folder / f"{args.source}.csv")
        rows = silver.set_index("crop_id").loc[listed.crop_id].reset_index()
    elif args.kind == "overripe":
        rows = silver[unreviewed & (silver.maturity == "overripe")]
    else:
        rows = silver[unreviewed]
    if args.source and args.kind in ("overripe", "needs-review"):
        rows = rows[rows.source_name == args.source]
    if rows.empty:
        parser.error("nothing to label")

    name = args.kind + (f"_{args.source}" if args.source else "")
    task = crop_task(rows.drop(columns="crop_id"), layer_dir(Layer.BRONZE))
    path = write_task(task, reports_dir() / "labeling" / f"{name}.csv")
    print(f"{len(task)} crops -> {path}")
    print(f"next: uv run python scripts/label.py {path} --labeler <you>")


if __name__ == "__main__":
    main()
