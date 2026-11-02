from __future__ import annotations

import random

import pandas as pd
import pytest
from PIL import Image

from strawberry_picker.audit.contact_sheets import (
    HEADER_PX,
    build_contact_sheets,
    label_slug,
    label_summary,
    render_sheet,
    sample_per_label,
)
from strawberry_picker.data.readers import Annotation


def _annotations(tmp_path, make_image, label: str, n: int, start: int = 0) -> list[Annotation]:
    return [
        Annotation(make_image(tmp_path / label / f"{i}.png", seed=i), label, None)
        for i in range(start, start + n)
    ]


def test_label_slug_is_filename_safe() -> None:
    assert label_slug("Normal/Ripe") == "Normal_Ripe"
    assert label_slug("partially ripe") == "partially_ripe"
    assert label_slug("Gray Mold ") == "Gray_Mold"


def test_sample_per_label_caps_and_is_deterministic(tmp_path, make_image) -> None:
    many = _annotations(tmp_path, make_image, "A", 7)
    few = _annotations(tmp_path, make_image, "B", 2, start=100)
    shuffled = many + few
    random.Random(1).shuffle(shuffled)

    sample = sample_per_label(many + few, per_label=5, seed=0)

    assert sorted(sample) == ["A", "B"]
    assert len(sample["A"]) == 5 and set(sample["A"]) <= set(many)
    assert sample["B"] == sorted(few, key=lambda a: a.image_path)
    assert sample_per_label(shuffled, per_label=5, seed=0) == sample  # input order is irrelevant
    assert sample_per_label(many, per_label=5, seed=1)["A"] != sample["A"]


def test_render_sheet_grid_size(tmp_path, make_image) -> None:
    annotations = _annotations(tmp_path, make_image, "A", 12)

    sheet = render_sheet(annotations, title="toy · A", tile_px=32, columns=5)

    assert sheet.mode == "RGB"
    assert sheet.size == (5 * 32, HEADER_PX + 3 * 32)


def test_render_sheet_crops_the_box(tmp_path, make_image) -> None:
    image = Image.new("RGB", (200, 100), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 100, 100))
    path = make_image(tmp_path / "two_halves.png", image=image)

    sheet = render_sheet(
        [Annotation(path, "A", (10.0, 10.0, 90.0, 90.0))], title="t", tile_px=40, columns=1
    )

    assert sheet.getpixel((20, HEADER_PX + 20)) == (255, 0, 0)


def test_render_sheet_survives_unreadable_images(tmp_path, make_image) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    good = _annotations(tmp_path, make_image, "A", 1)

    sheet = render_sheet([Annotation(broken, "A", None), *good], title="t", tile_px=16, columns=2)

    assert sheet.size == (32, HEADER_PX + 16)


def test_label_summary_reports_mapping_without_raising(tmp_path, make_image, make_spec) -> None:
    spec = make_spec(
        labels={
            "Ripe": {"maturity": "ripe", "defects": []},
            "Leaf": "drop",
            "Odd": {"maturity": "unripe", "needs_review": True},
        }
    )
    image = make_image(tmp_path / "x.png")
    annotations = [
        Annotation(image, "Ripe", (0.0, 0.0, 5.0, 5.0)),
        Annotation(image, "Ripe", (5.0, 5.0, 9.0, 9.0)),
        Annotation(image, "Leaf", None),
        Annotation(image, "Odd", None),
        Annotation(image, "Surprise", None),
    ]

    summary = label_summary(spec, annotations).set_index("source_label")

    assert summary.loc["Ripe", "annotations"] == 2
    assert summary.loc["Ripe", "images"] == 1
    assert summary.loc["Ripe", "mapped_to"] == "maturity=ripe defects=[]"
    assert summary.loc["Leaf", "mapped_to"] == "drop"
    assert summary.loc["Odd", "mapped_to"] == "maturity=unripe needs_review"
    assert summary.loc["Surprise", "mapped_to"] == "UNMAPPED"


def test_build_contact_sheets_writes_one_sheet_per_label(tmp_path, make_image, make_spec) -> None:
    pull = tmp_path / "bronze" / "toy" / "2026-11-02"
    for i in range(3):
        make_image(pull / "Normal" / "Ripe" / f"r{i}.png", seed=i)
    make_image(pull / "Normal" / "Unripe" / "u0.png", seed=9)
    before = sorted(p.relative_to(pull) for p in pull.rglob("*"))
    spec = make_spec(
        labels={
            "Normal/Ripe": {"maturity": "ripe", "defects": []},
            "Normal/Unripe": {"maturity": "unripe", "defects": []},
        }
    )
    out = tmp_path / "reports"

    summary = build_contact_sheets(spec, pull, out, per_label=2, tile_px=16, columns=2)

    assert sorted(p.name for p in out.iterdir()) == [
        "Normal_Ripe.jpg",
        "Normal_Unripe.jpg",
        "summary.csv",
    ]
    assert Image.open(out / "Normal_Ripe.jpg").size == (32, HEADER_PX + 16)
    written = pd.read_csv(out / "summary.csv").set_index("source_label")
    assert written.loc["Normal/Ripe", "annotations"] == 3
    assert written.loc["Normal/Ripe", "sheet"] == "Normal_Ripe.jpg"
    assert list(summary.columns) == ["source_label", "annotations", "images", "mapped_to", "sheet"]
    assert sorted(p.relative_to(pull) for p in pull.rglob("*")) == before  # bronze untouched


def test_build_contact_sheets_fails_loudly_when_nothing_is_found(tmp_path, make_spec) -> None:
    spec = make_spec(root="wrong/root")

    with pytest.raises(ValueError, match="no annotations"):
        build_contact_sheets(spec, tmp_path, tmp_path / "reports")

    assert not (tmp_path / "reports").exists()


def test_build_contact_sheets_skips_excluded_files(tmp_path, make_image, make_spec) -> None:
    pull = tmp_path / "pull"
    make_image(pull / "Ripe" / "keep.png", seed=1)
    make_image(pull / "Ripe" / "aug" / "copy.png", seed=2)
    make_image(pull / "Unripe" / "aug" / "only_copy.png", seed=3)
    spec = make_spec(
        labels={
            "Ripe": {"maturity": "ripe"},
            "Ripe/aug": {"maturity": "ripe"},
            "Unripe/aug": "drop",
        },
        exclude=["*/aug/*"],
    )

    summary = build_contact_sheets(spec, pull, tmp_path / "out", tile_px=8, columns=1)

    assert summary.source_label.tolist() == ["Ripe"]
    assert summary.annotations.tolist() == [1]
