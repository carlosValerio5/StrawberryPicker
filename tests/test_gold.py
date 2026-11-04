from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from strawberry_picker.data.gold import SPLITS, assign_splits, build_gold, crop_box
from strawberry_picker.schema import GOLD_COLUMNS, SILVER_COLUMNS

from conftest import pattern_image

ALL_TRAIN = {"train": 1.0, "val": 0.0, "test": 0.0}


def _row(image_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "image_id": image_id,
        "source_name": "toy",
        "bronze_path": f"toy/2026-11-02/{image_id}.png",
        "source_label": "Ripe",
        "license": "CC-BY-4.0",
        "commercial_use": True,
        "width": 100,
        "height": 80,
        "maturity": "ripe",
        "defects": [],
        "needs_review": False,
        "bbox": None,
        "group_id": f"toy/{image_id}",
        "ingested_at": datetime(2026, 11, 2, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def _write_silver(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    images = data_dir / "silver" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for seed, image_id in enumerate(dict.fromkeys(r["image_id"] for r in rows)):
        row = next(r for r in rows if r["image_id"] == image_id)
        pattern_image(seed, (row["width"], row["height"])).save(images / f"{image_id}.jpg")
    pd.DataFrame(rows, columns=list(SILVER_COLUMNS)).to_parquet(
        data_dir / "silver" / "manifest.parquet"
    )


@pytest.fixture
def silver_rows(tmp_data_dir) -> list[dict[str, Any]]:
    rows = [
        _row("img1"),
        _row("img2", maturity="unripe", source_label="unripe", bbox=[10.0, 10.0, 30.0, 30.0]),
        _row("img2", bbox=[50.0, 20.0, 90.0, 60.0]),
        _row("img3", maturity=None, defects=["mold"], source_label="Gray Mold"),
        _row("img4", maturity="unripe", needs_review=True),
        _row(
            "img5",
            maturity="overripe",
            defects=["rot"],
            license="CC-BY-NC-4.0",
            commercial_use=False,
        ),
    ]
    _write_silver(tmp_data_dir, rows)
    return rows


# --- crop_box ---------------------------------------------------------------------------


def test_crop_without_bbox_is_full_image() -> None:
    assert crop_box(None, 100, 80) == (0, 0, 100, 80)


def test_crop_is_padded_square_around_bbox() -> None:
    assert crop_box((40, 30, 60, 50), 100, 80, padding=0.25) == (35, 25, 65, 55)
    assert crop_box((40, 30, 60, 40), 100, 80, padding=0.0) == (40, 25, 60, 45)


def test_crop_is_shifted_inside_image() -> None:
    assert crop_box((0, 0, 20, 20), 100, 80, padding=0.25) == (0, 0, 30, 30)
    assert crop_box((85, 65, 100, 80), 100, 80, padding=0.0) == (85, 65, 100, 80)


def test_crop_larger_than_image_is_clipped() -> None:
    assert crop_box((0, 0, 100, 80), 100, 80, padding=0.25) == (0, 0, 100, 80)


# --- assign_splits ----------------------------------------------------------------------

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_splits_are_deterministic_and_complete() -> None:
    groups = [f"g{i}" for i in range(50)]
    first = assign_splits(groups, RATIOS, seed=42)
    assert first == assign_splits(reversed(groups), RATIOS, seed=42)
    assert set(first) == set(groups)
    assert set(first.values()) <= set(SPLITS)


def test_splits_follow_ratios() -> None:
    splits = assign_splits([f"g{i}" for i in range(4000)], RATIOS, seed=42)
    counts = pd.Series(splits).value_counts(normalize=True)
    for split, ratio in RATIOS.items():
        assert counts[split] == pytest.approx(ratio, abs=0.03)


def test_adding_groups_never_moves_existing_ones() -> None:
    old = assign_splits([f"g{i}" for i in range(100)], RATIOS, seed=42)
    new = assign_splits([f"g{i}" for i in range(300)], RATIOS, seed=42)
    assert all(new[g] == split for g, split in old.items())


def test_seed_changes_assignment() -> None:
    groups = [f"g{i}" for i in range(100)]
    assert assign_splits(groups, RATIOS, seed=1) != assign_splits(groups, RATIOS, seed=2)


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.7, "val": 0.2},
        {"train": 0.7, "val": 0.15, "holdout": 0.15},
        {"train": 1.2, "val": -0.2},
    ],
)
def test_invalid_ratios_rejected(ratios) -> None:
    with pytest.raises(ValueError):
        assign_splits(["g1"], ratios, seed=42)


# --- build_gold -------------------------------------------------------------------------


def test_build_maturity_dataset_layout(silver_rows, tmp_data_dir) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)

    assert out == tmp_data_dir / "gold" / "maturity_cls_v0.1"
    manifest = pd.read_parquet(out / "manifest.parquet")
    assert tuple(manifest.columns) == GOLD_COLUMNS
    assert sorted(manifest.crop_path) == [
        "train/overripe/img5_0.jpg",
        "train/ripe/img1_0.jpg",
        "train/ripe/img2_1.jpg",
        "train/unripe/img2_0.jpg",
    ]
    for crop_path in manifest.crop_path:
        assert (out / crop_path).is_file()


def test_build_excludes_unlabeled_and_review_rows(silver_rows) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)
    manifest = pd.read_parquet(out / "manifest.parquet")
    assert "img3" not in set(manifest.image_id)  # no maturity label
    assert "img4" not in set(manifest.image_id)  # mapping still needs human review


def test_build_crops_each_fruit(silver_rows) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42, padding=0.15)
    with Image.open(out / "train" / "unripe" / "img2_0.jpg") as crop:
        assert crop.size == (26, 26)
    with Image.open(out / "train" / "ripe" / "img1_0.jpg") as full:
        assert full.size == (100, 80)


def test_build_writes_stats_and_card(silver_rows) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)

    stats = json.loads((out / "stats.json").read_text())
    assert stats["train"] == {"total": 4, "maturity": {"unripe": 1, "ripe": 2, "overripe": 1}}
    assert stats["test"]["total"] == 0

    card = (out / "DATASET_CARD.md").read_text()
    for expected in ("maturity_cls_v0.1", "toy", "CC-BY-4.0", "CC-BY-NC-4.0", "seed"):
        assert expected in card


def test_commercial_only_drops_non_commercial_rows(silver_rows) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42, commercial_only=True)
    manifest = pd.read_parquet(out / "manifest.parquet")
    assert manifest.commercial_use.all()
    assert "img5" not in set(manifest.image_id)


def test_build_quality_dataset(silver_rows) -> None:
    out = build_gold("quality_cls", "0.1", ratios=ALL_TRAIN, seed=42)

    manifest = pd.read_parquet(out / "manifest.parquet")
    assert sorted(manifest.crop_path) == [
        "train/images/img1_0.jpg",
        "train/images/img2_0.jpg",
        "train/images/img2_1.jpg",
        "train/images/img3_0.jpg",
        "train/images/img5_0.jpg",
    ]
    stats = json.loads((out / "stats.json").read_text())["train"]
    assert stats["total"] == 5
    assert stats["grade"] == {"premium": 3, "standard": 0, "reject": 2}
    assert stats["defects"]["mold"] == 1
    assert stats["defects"]["rot"] == 1


def test_groups_never_straddle_splits(tmp_data_dir) -> None:
    rows = [_row(f"same{i}", group_id="toy/session-1") for i in range(10)]
    rows += [_row(f"solo{i}") for i in range(30)]
    _write_silver(tmp_data_dir, rows)

    out = build_gold("maturity_cls", "0.1", ratios=RATIOS, seed=42)

    manifest = pd.read_parquet(out / "manifest.parquet")
    assert manifest.groupby("group_id").split.nunique().max() == 1


def test_released_version_is_immutable(silver_rows, tmp_data_dir) -> None:
    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)
    before = sorted(p.relative_to(out) for p in out.rglob("*"))

    with pytest.raises(FileExistsError):
        build_gold("maturity_cls", "0.1", ratios=RATIOS, seed=7)

    assert sorted(p.relative_to(out) for p in out.rglob("*")) == before
    assert [p.name for p in (tmp_data_dir / "gold").iterdir()] == ["maturity_cls_v0.1"]


def test_build_rejects_bad_inputs(tmp_data_dir, silver_rows) -> None:
    with pytest.raises(ValueError):
        build_gold("ripeness", "0.1", ratios=ALL_TRAIN, seed=42)
    with pytest.raises(ValueError):
        build_gold("maturity_cls", "latest", ratios=ALL_TRAIN, seed=42)


def test_build_requires_rows(tmp_data_dir) -> None:
    _write_silver(tmp_data_dir, [_row("img1", maturity=None, defects=None)])
    with pytest.raises(ValueError, match="no rows"):
        build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)
    assert not (tmp_data_dir / "gold").exists() or not any((tmp_data_dir / "gold").iterdir())


def test_build_requires_silver(tmp_data_dir) -> None:
    with pytest.raises(FileNotFoundError):
        build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)


def test_failed_build_leaves_nothing_behind(silver_rows, tmp_data_dir) -> None:
    (tmp_data_dir / "silver" / "images" / "img2.jpg").unlink()  # silver is inconsistent

    with pytest.raises(FileNotFoundError):
        build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)

    assert list((tmp_data_dir / "gold").iterdir()) == []  # no release, no .partial


# --- crop policy (ADR 0008) -------------------------------------------------------------


@pytest.fixture
def crowded_silver(tmp_data_dir) -> list[dict[str, Any]]:
    """Two touching berries, plus a needs-review box that is still a neighbour."""
    rows = [
        _row("crowd", maturity="unripe", bbox=[10.0, 10.0, 40.0, 40.0]),
        _row("crowd", bbox=[38.0, 10.0, 70.0, 40.0]),
        _row("crowd", needs_review=True, bbox=[10.0, 45.0, 40.0, 75.0]),
        _row("solo"),
    ]
    _write_silver(tmp_data_dir, rows)
    return rows


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as crop:
        return np.asarray(crop.convert("RGB"), dtype=int)


def test_default_policy_records_overlap_and_policy(crowded_silver) -> None:
    from strawberry_picker.data.crops import LEGACY_POLICY, CropPolicy

    out = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)

    manifest = pd.read_parquet(out / "manifest.parquet").set_index("crop_id")
    # crop_box → (6, 6, 45, 45); the neighbour shows in columns 40-45 → 5 × 30 / 39².
    assert manifest.loc["crowd_0", "neighbour_overlap"] == pytest.approx(150 / 39**2)
    assert manifest.loc["solo_0", "neighbour_overlap"] == 0.0
    assert len(manifest.loc["crowd_0", "neighbour_bboxes"]) == 2  # incl. the review box
    assert CropPolicy.from_json(manifest.loc["crowd_0", "crop_policy"]) == LEGACY_POLICY
    assert CropPolicy.from_json((out / "crop_policy.json").read_text()) == LEGACY_POLICY
    assert json.loads((out / "stats.json").read_text())["dropped"] == {"total": 0}


def test_mask_policy_greys_out_the_neighbour(crowded_silver) -> None:
    from strawberry_picker.data.crops import CropPolicy

    policy = CropPolicy(padding=0.15, neighbours="mask", fill=(0, 0, 255))
    kept = build_gold("maturity_cls", "0.1", ratios=ALL_TRAIN, seed=42)
    masked = build_gold("maturity_cls", "0.2", ratios=ALL_TRAIN, seed=42, policy=policy)

    before = _pixels(kept / "train" / "unripe" / "crowd_0.jpg")
    after = _pixels(masked / "train" / "unripe" / "crowd_0.jpg")
    # Crop columns ≥ 40 - 6 = 34 are the neighbour's only; columns < 34 are target/context.
    # JPEG blurs a couple of pixels across the edge, so sample away from it.
    assert np.abs(after[6:32, 37:] - [0, 0, 255]).max() < 30
    assert np.abs(after[:, :24] - before[:, :24]).max() < 15
    assert CropPolicy.from_json((masked / "crop_policy.json").read_text()) == policy


def test_drop_policy_reports_every_dropped_crop(crowded_silver) -> None:
    from strawberry_picker.data.crops import CropPolicy

    policy = CropPolicy(neighbours="drop", max_neighbour_overlap=0.05)
    out = build_gold("maturity_cls", "0.3", ratios=ALL_TRAIN, seed=42, policy=policy)

    manifest = pd.read_parquet(out / "manifest.parquet")
    assert set(manifest.crop_id) == {"solo_0"}
    dropped = pd.read_parquet(out / "dropped.parquet")
    assert set(dropped.crop_id) == {"crowd_0", "crowd_1"}
    assert (dropped.neighbour_overlap > 0.05).all()
    assert not (out / "train" / "unripe" / "crowd_0.jpg").exists()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["dropped"] == {"total": 2, "train": 2}
    assert stats["train"]["total"] == 1
    card = (out / "DATASET_CARD.md").read_text()
    assert "Dropped crops" in card and "drop" in card


def test_padding_argument_still_works_without_a_policy(crowded_silver) -> None:
    from strawberry_picker.data.crops import CropPolicy

    out = build_gold("maturity_cls", "0.4", ratios=ALL_TRAIN, seed=42, padding=0.0)
    assert CropPolicy.from_json((out / "crop_policy.json").read_text()) == CropPolicy(padding=0.0)
    with Image.open(out / "train" / "unripe" / "crowd_0.jpg") as crop:
        assert crop.size == (30, 30)
