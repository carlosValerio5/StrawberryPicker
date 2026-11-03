from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from strawberry_picker.data.silver import REJECTED_COLUMNS, build_silver, compute_image_id
from strawberry_picker.schema import SILVER_COLUMNS

from conftest import pattern_image

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
BUILD_KWARGS = {
    "min_side_px": 32,
    "jpeg_quality": 95,
    "near_duplicate_max_distance": 24,
    "now": NOW,
}


def test_image_id_is_deterministic_and_content_based(red_image: Image.Image) -> None:
    assert compute_image_id(red_image) == compute_image_id(red_image.copy())
    assert len(compute_image_id(red_image)) == 16


def test_image_id_ignores_color_mode(red_image: Image.Image) -> None:
    assert compute_image_id(red_image.convert("RGBA")) == compute_image_id(red_image)


def test_image_id_distinguishes_content_and_shape(red_image: Image.Image) -> None:
    green = Image.new("RGB", (8, 8), (40, 160, 40))
    reshaped = Image.new("RGB", (4, 16), (200, 20, 30))
    ids = {compute_image_id(img) for img in (red_image, green, reshaped)}
    assert len(ids) == 3


# --- bronze → silver build -------------------------------------------------------------


@pytest.fixture
def bronze_sources(tmp_data_dir, make_image, make_spec):
    """Two enabled toy sources (folder-per-class + YOLO) and one disabled source."""
    bronze = tmp_data_dir / "bronze"
    pull = bronze / "toy_folder" / "2026-11-02"
    make_image(pull / "Ripe" / "a.png", seed=1)
    make_image(pull / "Unripe" / "b.png", seed=2)
    make_image(pull / "Unripe" / "b_copy.png", seed=2)  # identical pixels → exact duplicate
    make_image(
        pull / "Ripe" / "c_rot.png", image=pattern_image(1).transpose(Image.Transpose.ROTATE_90)
    )
    make_image(pull / "Occluded" / "d.png", seed=4)  # mapped to "drop"
    make_image(pull / "Ripe" / "tiny.png", seed=5, size=(10, 10))
    (pull / "Ripe" / "broken.jpg").write_bytes(b"not a jpeg")
    make_image(pull / "Synthetic" / "e.png", seed=6)  # excluded by glob, label never mapped
    make_image(bronze / "toy_folder" / "2026-10-01" / "Ripe" / "old.png", seed=7)  # older pull

    yolo = bronze / "toy_yolo" / "2026-11-02"
    make_image(yolo / "field.png", seed=8, size=(200, 100))
    (yolo / "field.txt").write_text("0 0.25 0.5 0.1 0.2\n1 0.5 0.5 0.5 0.25\n")

    make_image(bronze / "toy_disabled" / "2026-11-02" / "Ripe" / "z.png", seed=9)

    return {
        "toy_folder": make_spec(
            "toy_folder",
            license="CC-BY-NC-4.0",
            commercial_use=False,
            exclude=["Synthetic/*"],
            labels={
                "Ripe": {"maturity": "ripe", "defects": []},
                "Unripe": {"maturity": "unripe", "defects": []},
                "Occluded": "drop",
            },
        ),
        "toy_yolo": make_spec(
            "toy_yolo",
            format="yolo",
            class_names=["unripe", "ripe"],
            group_by="source",
            labels={
                "unripe": {"maturity": "unripe"},
                "ripe": {"maturity": "ripe", "needs_review": True},
            },
        ),
        "toy_disabled": make_spec("toy_disabled", enabled=False),
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_build_silver_manifest_rows(bronze_sources, tmp_data_dir) -> None:
    result = build_silver(bronze_sources, **BUILD_KWARGS)
    manifest = result.manifest

    assert tuple(manifest.columns) == SILVER_COLUMNS
    assert sorted(manifest.bronze_path) == [
        "toy_folder/2026-11-02/Ripe/a.png",
        "toy_folder/2026-11-02/Unripe/b.png",
        "toy_yolo/2026-11-02/field.png",
        "toy_yolo/2026-11-02/field.png",
    ]

    a = manifest.set_index("bronze_path").loc["toy_folder/2026-11-02/Ripe/a.png"]
    assert a.maturity == "ripe"
    assert list(a.defects) == []
    assert a.source_label == "Ripe"
    assert a.license == "CC-BY-NC-4.0"
    assert not a.commercial_use
    assert not a.needs_review
    assert (a.width, a.height) == (64, 48)
    assert a.group_id == f"toy_folder/{a.image_id}"
    assert a.ingested_at == NOW

    field = manifest[manifest.source_name == "toy_yolo"].reset_index(drop=True)
    assert field.image_id.nunique() == 1
    assert list(field.source_label) == ["unripe", "ripe"]
    assert [tuple(b) for b in field.bbox] == [(40.0, 40.0, 60.0, 60.0), (50.0, 37.5, 150.0, 62.5)]
    assert list(field.needs_review) == [False, True]
    assert field.defects.isna().all()
    assert set(field.group_id) == {"toy_yolo"}


def test_build_silver_rejections(bronze_sources) -> None:
    rejected = build_silver(bronze_sources, **BUILD_KWARGS).rejected

    assert tuple(rejected.columns) == REJECTED_COLUMNS
    reasons = dict(zip(rejected.bronze_path, rejected.reason, strict=True))
    assert reasons == {
        "toy_folder/2026-11-02/Unripe/b_copy.png": "duplicate",
        "toy_folder/2026-11-02/Ripe/c_rot.png": "near_duplicate",
        "toy_folder/2026-11-02/Occluded/d.png": "label_dropped",
        "toy_folder/2026-11-02/Ripe/tiny.png": "too_small",
        "toy_folder/2026-11-02/Ripe/broken.jpg": "corrupt",
        "toy_folder/2026-11-02/Synthetic/e.png": "excluded",
    }
    near = rejected.set_index("reason").loc["near_duplicate"]
    assert near.detail == "toy_folder/2026-11-02/Ripe/a.png"


def test_build_silver_writes_images_and_parquet(bronze_sources, tmp_data_dir) -> None:
    silver = tmp_data_dir / "silver"
    (silver / "images").mkdir(parents=True)
    (silver / "images" / "stale.jpg").write_bytes(b"left over from an old build")

    result = build_silver(bronze_sources, **BUILD_KWARGS)

    stored = {p.name for p in (silver / "images").iterdir()}
    assert stored == {f"{image_id}.jpg" for image_id in result.manifest.image_id.unique()}
    with Image.open(silver / "images" / stored.pop()) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"

    defects = pq.read_table(silver / "manifest.parquet").column("defects").to_pylist()
    assert sorted(defects, key=str) == [None, None, [], []]  # "none found" ≠ "not assessed"
    assert len(pd.read_parquet(silver / "rejected.parquet")) == len(result.rejected)


def test_build_silver_never_touches_bronze(bronze_sources, tmp_data_dir) -> None:
    before = _snapshot(tmp_data_dir / "bronze")
    build_silver(bronze_sources, **BUILD_KWARGS)
    assert _snapshot(tmp_data_dir / "bronze") == before


def test_build_silver_is_reproducible(bronze_sources) -> None:
    first = build_silver(bronze_sources, **BUILD_KWARGS)
    second = build_silver(bronze_sources, **BUILD_KWARGS)
    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    pd.testing.assert_frame_equal(first.rejected, second.rejected)


def test_build_silver_applies_exif_orientation(tmp_data_dir, make_spec) -> None:
    path = tmp_data_dir / "bronze" / "toy_source" / "2026-11-02" / "Ripe" / "phone.jpg"
    path.parent.mkdir(parents=True)
    image = pattern_image(3, size=(64, 48))
    exif = image.getexif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    image.save(path, exif=exif)

    row = build_silver({"toy_source": make_spec()}, **BUILD_KWARGS).manifest.iloc[0]

    assert (row.width, row.height) == (48, 64)


def test_build_silver_fails_loudly_on_unmapped_label(tmp_data_dir, make_image, make_spec) -> None:
    make_image(tmp_data_dir / "bronze" / "toy_source" / "2026-11-02" / "Mystery" / "x.png")
    with pytest.raises(ValueError, match="Mystery"):
        build_silver({"toy_source": make_spec()}, **BUILD_KWARGS)


def test_build_silver_with_no_pulls_writes_empty_manifest(tmp_data_dir, make_spec) -> None:
    result = build_silver({"toy_source": make_spec()}, **BUILD_KWARGS)
    assert result.manifest.empty
    assert tuple(result.manifest.columns) == SILVER_COLUMNS
    assert (tmp_data_dir / "silver" / "manifest.parquet").is_file()


def test_parent_dir_grouping_keeps_sessions_together(tmp_data_dir, make_image, make_spec) -> None:
    pull = tmp_data_dir / "bronze" / "toy_source" / "2026-11-02"
    make_image(pull / "Ripe" / "session1" / "a.png", seed=1)
    make_image(pull / "Ripe" / "session1" / "b.png", seed=2)
    make_image(pull / "Ripe" / "session2" / "c.png", seed=3)
    spec = make_spec(
        group_by="parent_dir",
        labels={"Ripe/session1": {"maturity": "ripe"}, "Ripe/session2": {"maturity": "ripe"}},
    )

    manifest = build_silver({"toy_source": spec}, **BUILD_KWARGS).manifest

    assert sorted(manifest.group_id) == [
        "toy_source/Ripe/session1",
        "toy_source/Ripe/session1",
        "toy_source/Ripe/session2",
    ]


def test_build_silver_applies_agreed_human_reviews(bronze_sources, tmp_data_dir) -> None:
    from strawberry_picker.labeling.tool import LabelStore, parse_label

    first = build_silver(bronze_sources, **BUILD_KWARGS).manifest
    assert first.reviewed_by.isna().all()
    field = first[first.source_name == "toy_yolo"]
    ripe_crop = f"{field.image_id.iloc[0]}_1"  # the needs_review "ripe" box
    review = tmp_data_dir / "bronze" / "human_review" / "2026-11-05" / "labels.csv"
    LabelStore(review, "alice").record(parse_label({"item_id": ripe_crop, "maturity": "overripe"}))

    second = build_silver(bronze_sources, **BUILD_KWARGS).manifest

    reviewed = second[second.source_name == "toy_yolo"].iloc[1]
    assert (reviewed.maturity, reviewed.needs_review, reviewed.reviewed_by) == (
        "overripe",
        False,
        "alice",
    )
    assert reviewed.source_label == "ripe"  # what the source said is kept for audits
    assert len(second) == len(first)
