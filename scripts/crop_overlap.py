"""Neighbour overlap in training crops (ADR 0011): uv run python scripts/crop_overlap.py.

Needs only the silver manifest: no images, no model, no ml extra. If an earlier
`scripts/dataset_audit.py` run left out-of-fold predictions in
reports/dataset-audit/label_issues.csv, the error rate is also broken down by overlap.
Writes reports/dataset-audit/neighbours/.
"""

from __future__ import annotations

import argparse

import pandas as pd

from strawberry_picker.audit.neighbours import overlap_table, write_neighbour_report
from strawberry_picker.config import load_config
from strawberry_picker.data.crops import CropPolicy
from strawberry_picker.paths import Layer, layer_dir, reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding", type=float, help="override crop_padding from data.yaml")
    args = parser.parse_args()

    settings = dict(load_config("data")["gold"])
    if args.padding is not None:
        settings["crop_padding"] = args.padding
    manifest = pd.read_parquet(layer_dir(Layer.SILVER) / "manifest.parquet")
    out = reports_dir() / "dataset-audit"
    issues_path = out / "label_issues.csv"
    issues = pd.read_csv(issues_path) if issues_path.exists() else None

    table = overlap_table(manifest, CropPolicy.from_settings(settings))
    report = write_neighbour_report(table, out, issues=issues)
    print(report.read_text())


if __name__ == "__main__":
    main()
