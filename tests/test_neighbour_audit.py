from __future__ import annotations

import pandas as pd
import pytest

from strawberry_picker.audit.neighbours import (
    error_by_overlap,
    overlap_summary,
    overlap_table,
    write_neighbour_report,
)
from strawberry_picker.data.crops import CropPolicy


def _silver() -> pd.DataFrame:
    def row(image_id, source, maturity, bbox, **kw):
        return {
            "image_id": image_id,
            "source_name": source,
            "maturity": maturity,
            "defects": [],
            "needs_review": False,
            "bbox": bbox,
            "width": 100,
            "height": 100,
            **kw,
        }

    return pd.DataFrame(
        [
            # Two touching berries: each crop sees a strip of the other.
            row("dense", "farm", "ripe", [10.0, 10.0, 50.0, 50.0]),
            row("dense", "farm", "unripe", [45.0, 10.0, 85.0, 50.0]),
            # A box that still needs review is a neighbour, but gets no audit row itself.
            row("dense", "farm", "ripe", [10.0, 52.0, 50.0, 92.0], needs_review=True),
            row("lone", "farm", "ripe", [30.0, 30.0, 60.0, 60.0]),
            row("whole", "shop", "ripe", None),
        ]
    )


def test_overlap_table_has_one_row_per_audited_crop() -> None:
    table = overlap_table(_silver(), CropPolicy(padding=0.15))

    assert list(table.crop_id) == ["dense_0", "dense_1", "lone_0", "whole_0"]
    assert list(table.columns) == [
        "crop_id",
        "source_name",
        "label",
        "n_neighbours",
        "neighbour_overlap",
    ]
    by_crop = table.set_index("crop_id")
    assert by_crop.loc["dense_0", "n_neighbours"] == 2
    assert by_crop.loc["dense_0", "neighbour_overlap"] > 0.10
    assert by_crop.loc["lone_0", "neighbour_overlap"] == 0.0
    assert by_crop.loc["whole_0", "neighbour_overlap"] == 0.0


def test_overlap_summary_shares_per_source_and_label() -> None:
    table = pd.DataFrame(
        {
            "crop_id": list("abcd"),
            "source_name": ["s", "s", "s", "t"],
            "label": ["ripe", "ripe", "ripe", "ripe"],
            "n_neighbours": [0, 1, 2, 0],
            "neighbour_overlap": [0.0, 0.05, 0.30, 0.0],
        }
    )
    summary = overlap_summary(table).set_index(["source_name", "label"])

    s = summary.loc[("s", "ripe")]
    assert s.n == 3
    assert s.share_any == pytest.approx(2 / 3)
    assert s.share_over_10 == pytest.approx(1 / 3)
    assert s.share_over_25 == pytest.approx(1 / 3)
    assert s.mean_overlap == pytest.approx(0.35 / 3)
    assert summary.loc[("t", "ripe")].share_any == 0.0


def test_error_by_overlap_buckets_out_of_fold_mistakes() -> None:
    table = pd.DataFrame(
        {
            "crop_id": list("abcdef"),
            "source_name": "s",
            "label": "ripe",
            "n_neighbours": 0,
            "neighbour_overlap": [0.0, 0.0, 0.05, 0.2, 0.4, 0.5],
        }
    )
    issues = pd.DataFrame(
        {
            "crop_id": list("abcdef"),
            "task": "maturity",
            "given": "ripe",
            "predicted": ["ripe", "ripe", "ripe", "unripe", "unripe", "ripe"],
            "is_issue": [False, False, False, True, True, False],
        }
    )
    buckets = error_by_overlap(table, issues).set_index("bucket")

    assert list(buckets.index) == ["0", "0-10%", "10-25%", ">25%"]
    assert buckets.loc["0", "n"] == 2 and buckets.loc["0", "error_rate"] == 0.0
    assert buckets.loc["10-25%", "error_rate"] == 1.0
    assert buckets.loc[">25%", "error_rate"] == 0.5
    assert buckets.loc[">25%", "flagged_rate"] == 0.5


def test_write_neighbour_report(tmp_path) -> None:
    table = overlap_table(_silver(), CropPolicy())
    out = write_neighbour_report(table, tmp_path)

    assert out == tmp_path / "neighbours" / "neighbours.md"
    assert (tmp_path / "neighbours" / "per_crop.csv").is_file()
    assert (tmp_path / "neighbours" / "summary.csv").is_file()
    assert not (tmp_path / "neighbours" / "error_by_overlap.csv").exists()
    assert "farm" in out.read_text()

    issues = pd.DataFrame(
        {
            "crop_id": table.crop_id,
            "task": "maturity",
            "given": "ripe",
            "predicted": "ripe",
            "is_issue": False,
        }
    )
    write_neighbour_report(table, tmp_path, issues=issues)
    assert (tmp_path / "neighbours" / "error_by_overlap.csv").is_file()
    assert "error rate" in out.read_text().lower()
