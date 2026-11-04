from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strawberry_picker.audit.multi_fruit import (
    FLAG_COLUMNS,
    flag_multi_fruit,
    summarize,
    write_outputs,
)
from strawberry_picker.labeling.tasks import multi_fruit_task

# The fake detector reads the image width: each silver image gets a different scene.
SCENES = {
    40: ([[2, 2, 30, 30]], [0.9]),  # one berry
    60: (
        [[0, 0, 20, 20], [1, 1, 21, 21], [30, 5, 55, 30], [40, 30, 50, 40]],
        [0.8, 0.7, 0.6, 0.1],  # a duplicate box (NMS) and a low-score box: 2 berries
    ),
    50: ([], []),  # the detector sees nothing
}


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        boxes, scores = SCENES[image.width]
        return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores, dtype=float)


def _silver() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "c", "d"],
            "source_name": ["fresh", "fresh", "ds", "ds", "rise"],
            "source_label": ["FreshStrawberry", "RottenStrawberry", "Red", "Green", "Ripe"],
            "bronze_path": ["f/a.jpg", "fresh/1/b.jpg", "ds/c.jpg", "ds/c.jpg", "r/d.jpg"],
            "maturity": [None, "overripe", "ripe", "unripe", "ripe"],
            "defects": [[], ["rot"], None, None, []],
            "bbox": [None, None, [1.0, 1.0, 9.0, 9.0], [10.0, 1.0, 19.0, 9.0], None],
            "needs_review": [False, True, False, False, False],
            "reviewed_by": [None, None, None, None, None],
        }
    )


@pytest.fixture
def images(tmp_path, make_image):
    folder = tmp_path / "silver" / "images"
    for image_id, width in {"a": 40, "b": 60, "c": 70, "d": 50}.items():
        make_image(folder / f"{image_id}.jpg", size=(width, 45))
    return folder


def _flag(images, tmp_path, detector, silver=None, **kwargs) -> pd.DataFrame:
    return flag_multi_fruit(
        _silver() if silver is None else silver,
        images,
        tmp_path / "cache.parquet",
        lambda: detector,
        detector="fake",
        **kwargs,
    )


def test_counts_confident_berries_in_whole_image_rows_only(images, tmp_path) -> None:
    flags = _flag(images, tmp_path, FakeDetector()).set_index("crop_id")

    assert list(flags.reset_index().columns) == list(FLAG_COLUMNS)
    assert set(flags.index) == {"a_0", "b_0", "d_0"}  # boxed rows already are one fruit
    assert flags.n_detected.to_dict() == {"a_0": 1, "b_0": 2, "d_0": 0}
    assert flags.loc["b_0", "max_score"] == pytest.approx(0.8)
    assert np.isnan(flags.loc["d_0", "max_score"])
    assert json.loads(flags.loc["b_0", "boxes"]) == [[0, 0, 20, 20], [30, 5, 55, 30]]
    assert flags.loc["b_0", "source_label"] == "RottenStrawberry"


def test_detections_are_cached_and_thresholds_reuse_them(images, tmp_path) -> None:
    first = FakeDetector()
    _flag(images, tmp_path, first)
    assert first.calls == 3

    def no_weights():
        raise AssertionError("cached images must not load the detector")

    lenient = flag_multi_fruit(
        _silver(), images, tmp_path / "cache.parquet", no_weights, detector="fake", min_score=0.05
    ).set_index("crop_id")
    assert lenient.loc["b_0", "n_detected"] == 3  # low-score box now counts, no re-detection

    new_row = {**_silver().iloc[0].to_dict(), "image_id": "e", "bronze_path": "fresh/1/e.jpg"}
    silver = pd.concat([_silver(), pd.DataFrame([new_row])], ignore_index=True)
    (images / "e.jpg").write_bytes((images / "a.jpg").read_bytes())
    second = FakeDetector()
    _flag(images, tmp_path, second, silver=silver)
    assert second.calls == 1  # only the new image

    other = FakeDetector()
    flag_multi_fruit(_silver(), images, tmp_path / "cache.parquet", lambda: other, detector="v2")
    assert other.calls == 3  # a different detector has its own cache entries


def test_summary_per_source_and_label(images, tmp_path) -> None:
    summary = summarize(_flag(images, tmp_path, FakeDetector()), min_count=2)
    rows = summary.set_index(["source_name", "source_label"])

    assert rows.loc[("fresh", "RottenStrawberry")].to_dict() == {
        "images": 1,
        "flagged": 1,
        "flagged_share": 1.0,
    }
    assert rows.loc[("fresh", "FreshStrawberry"), "flagged"] == 0
    assert rows.loc[("rise", "Ripe"), "images"] == 1


def test_write_outputs(images, tmp_path) -> None:
    flags = _flag(images, tmp_path, FakeDetector())
    paths = write_outputs(flags, tmp_path / "out", min_count=2)

    assert pd.read_csv(paths["flags"]).crop_id.tolist() == ["a_0", "b_0", "d_0"]
    assert pd.read_csv(paths["summary"]).flagged.sum() == 1
    assert "RottenStrawberry" in paths["report"].read_text()


def test_multi_fruit_task_lists_flagged_unreviewed_rows(images, tmp_path) -> None:
    silver = _silver()
    flags = _flag(images, tmp_path, FakeDetector())
    flags.loc[flags.crop_id == "a_0", "n_detected"] = 4  # pretend a is a pile too
    silver.loc[silver.image_id == "a", "reviewed_by"] = "alice"  # ...already reviewed

    task = multi_fruit_task(silver, flags, tmp_path / "bronze", min_count=2)

    assert task.item_id.tolist() == ["b_0"]
    assert task.bbox.tolist() == [""]  # the reviewer sees the whole image
    assert task.image_path.iloc[0] == str(tmp_path / "bronze" / "fresh/1/b.jpg")
    assert task.given.iloc[0] == "overripe +rot (detector: 2 berries)"
    assert multi_fruit_task(silver, flags, tmp_path / "bronze", min_count=5).empty


def test_no_whole_image_rows_needs_no_detector(images, tmp_path) -> None:
    boxed = _silver()[lambda f: f.bbox.notna()]

    def no_weights():
        raise AssertionError("nothing to detect")

    flags = flag_multi_fruit(boxed, images, tmp_path / "c.parquet", no_weights, detector="x")

    assert flags.empty and list(flags.columns) == list(FLAG_COLUMNS)
    assert summarize(flags).empty
