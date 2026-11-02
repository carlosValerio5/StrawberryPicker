from __future__ import annotations

import json

import pytest

from strawberry_picker.data.readers import (
    Annotation,
    read_annotations,
    read_coco,
    read_imagefolder,
    read_yolo,
    yolo_to_xyxy,
)


def test_imagefolder_uses_relative_folder_as_label(tmp_path, make_image) -> None:
    a = make_image(tmp_path / "Normal" / "Ripe" / "a.png", seed=1)
    b = make_image(tmp_path / "Anomalous" / "b.jpg", seed=2)
    make_image(tmp_path / "stray.png", seed=3)  # no folder → no label
    (tmp_path / "Normal" / "Ripe" / "notes.txt").write_text("not an image")

    assert list(read_imagefolder(tmp_path)) == [
        Annotation(b, "Anomalous", None),
        Annotation(a, "Normal/Ripe", None),
    ]


def test_yolo_to_xyxy() -> None:
    assert yolo_to_xyxy(0.5, 0.5, 0.5, 0.25, width=200, height=100) == (50.0, 37.5, 150.0, 62.5)


def test_yolo_sibling_label_file(tmp_path, make_image) -> None:
    image = make_image(tmp_path / "img.jpg", size=(200, 100))
    (tmp_path / "img.txt").write_text("1 0.5 0.5 0.5 0.25\n\n0 0.25 0.5 0.1 0.2\n")

    assert list(read_yolo(tmp_path, ["unripe", "ripe"])) == [
        Annotation(image, "ripe", (50.0, 37.5, 150.0, 62.5)),
        Annotation(image, "unripe", (40.0, 40.0, 60.0, 60.0)),
    ]


def test_yolo_images_labels_split_dirs(tmp_path, make_image) -> None:
    image = make_image(tmp_path / "train" / "images" / "x.png", size=(100, 100))
    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    (labels / "x.txt").write_text("0 0.5 0.5 1.0 1.0\n")

    assert list(read_yolo(tmp_path, ["ripe"])) == [
        Annotation(image, "ripe", (0.0, 0.0, 100.0, 100.0))
    ]


def test_yolo_polygon_rows_become_bounding_boxes(tmp_path, make_image) -> None:
    image = make_image(tmp_path / "p.png", size=(100, 100))
    (tmp_path / "p.txt").write_text("0 0.1 0.2 0.5 0.2 0.3 0.6\n")

    assert list(read_yolo(tmp_path, ["ripe"])) == [
        Annotation(image, "ripe", (10.0, 20.0, 50.0, 60.0))
    ]


def test_yolo_skips_unlabeled_and_empty(tmp_path, make_image) -> None:
    make_image(tmp_path / "no_label.png")
    make_image(tmp_path / "empty.png")
    (tmp_path / "empty.txt").write_text("")
    assert list(read_yolo(tmp_path, ["ripe"])) == []


def test_yolo_rejects_unknown_class_index(tmp_path, make_image) -> None:
    make_image(tmp_path / "bad.png")
    (tmp_path / "bad.txt").write_text("3 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="class index 3"):
        list(read_yolo(tmp_path, ["ripe"]))


def test_yolo_rejects_malformed_row(tmp_path, make_image) -> None:
    make_image(tmp_path / "bad.png")
    (tmp_path / "bad.txt").write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match="bad.txt"):
        list(read_yolo(tmp_path, ["ripe"]))


def test_coco_boxes_and_categories(tmp_path, make_image) -> None:
    split = tmp_path / "train"
    image = make_image(split / "f1.jpg")
    (split / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 0, "name": "diseases"}, {"id": 2, "name": "Gray Mold"}],
                "images": [{"id": 7, "file_name": "f1.jpg"}],
                "annotations": [
                    {"id": 2, "image_id": 7, "category_id": 2, "bbox": [10, 5, 20, 30]},
                    {"id": 1, "image_id": 7, "category_id": 2, "bbox": [0, 0, 4, 4]},
                ],
            }
        )
    )
    assert list(read_coco(tmp_path)) == [
        Annotation(image, "Gray Mold", (0.0, 0.0, 4.0, 4.0)),
        Annotation(image, "Gray Mold", (10.0, 5.0, 30.0, 35.0)),
    ]


def test_coco_ignores_unrelated_json(tmp_path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"name": "not coco"}))
    assert list(read_coco(tmp_path)) == []


def test_read_annotations_dispatches_on_format_and_root(tmp_path, make_image, make_spec) -> None:
    image = make_image(tmp_path / "repo-main" / "Data" / "Ripe" / "a.png")
    spec = make_spec(root="repo-main/Data")
    assert list(read_annotations(spec, tmp_path)) == [Annotation(image, "Ripe", None)]

    yolo_dir = tmp_path / "yolo"
    yolo_image = make_image(yolo_dir / "y.png", size=(10, 10))
    (yolo_dir / "y.txt").write_text("0 0.5 0.5 1 1\n")
    yolo_spec = make_spec(format="yolo", class_names=["Ripe"], root="yolo")
    assert list(read_annotations(yolo_spec, tmp_path)) == [
        Annotation(yolo_image, "Ripe", (0.0, 0.0, 10.0, 10.0))
    ]


def test_read_annotations_dispatches_coco(tmp_path, make_image, make_spec) -> None:
    image = make_image(tmp_path / "coco" / "f.jpg")
    (tmp_path / "coco" / "ann.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "Ripe"}],
                "images": [{"id": 1, "file_name": "f.jpg"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8]}],
            }
        )
    )
    spec = make_spec(format="coco", root="coco")
    assert list(read_annotations(spec, tmp_path)) == [
        Annotation(image, "Ripe", (0.0, 0.0, 8.0, 8.0))
    ]
