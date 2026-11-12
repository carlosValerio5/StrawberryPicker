"""Turn a source's on-disk annotation format into one ``Annotation`` per fruit.

Boxes are returned in pixel ``(x_min, y_min, x_max, y_max)`` coordinates of the image
*as displayed*, i.e. after EXIF orientation, which is how annotation tools show it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from strawberry_picker.data.sources import AnnotationFormat, SourceSpec

BBox = tuple[float, float, float, float]
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
)


@dataclass(frozen=True, slots=True)
class Annotation:
    image_path: Path
    source_label: str
    bbox: BBox | None  # None = the whole image is the fruit


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def read_imagefolder(root: Path) -> Iterator[Annotation]:
    """``<root>/<label>/.../<image>``: the folder path below ``root`` is the label."""
    for path in _images(root):
        if path.parent != root:
            yield Annotation(path, path.parent.relative_to(root).as_posix(), None)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, *, width: int, height: int) -> BBox:
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def _yolo_label_path(image_path: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        candidates.append(Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt"))
    return next((c for c in candidates if c.is_file()), None)


def read_yolo(root: Path, class_names: Sequence[str]) -> Iterator[Annotation]:
    """YOLO detection (``cls cx cy w h``) or segmentation (``cls x1 y1 x2 y2 …``) rows."""
    for image_path in _images(root):
        label_path = _yolo_label_path(image_path)
        if label_path is None:
            continue
        rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        for values in rows:
            if len(values) < 5 or len(values) % 2 == 0:
                raise ValueError(f"{label_path}: malformed YOLO row {' '.join(values)!r}")
            index, coords = int(values[0]), [float(v) for v in values[1:]]
            if not 0 <= index < len(class_names):
                raise ValueError(f"{label_path}: class index {index} outside class_names")
            if len(coords) == 4:
                bbox = yolo_to_xyxy(*coords, width=width, height=height)
            else:
                xs, ys = coords[0::2], coords[1::2]
                bbox = (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)
            yield Annotation(image_path, class_names[index], bbox)


def read_coco(root: Path, pattern: str = "**/*.json") -> Iterator[Annotation]:
    """Every COCO file under ``root``; non-COCO json files are ignored."""
    for annotation_file in sorted(root.glob(pattern)):
        data = json.loads(annotation_file.read_text())
        if not all(key in data for key in ("images", "annotations", "categories")):
            continue
        categories = {c["id"]: c["name"] for c in data["categories"]}
        images = {i["id"]: annotation_file.parent / i["file_name"] for i in data["images"]}
        for ann in sorted(data["annotations"], key=lambda a: a["id"]):
            x, y, w, h = (float(v) for v in ann["bbox"])
            yield Annotation(
                images[ann["image_id"]], categories[ann["category_id"]], (x, y, x + w, y + h)
            )


def read_annotations(spec: SourceSpec, pull_dir: Path) -> Iterator[Annotation]:
    root = pull_dir / spec.root
    match spec.format:
        case AnnotationFormat.IMAGEFOLDER:
            return read_imagefolder(root)
        case AnnotationFormat.YOLO:
            return read_yolo(root, spec.class_names)
        case AnnotationFormat.COCO:
            return read_coco(root)
    raise ValueError(f"{spec.name}: unsupported format {spec.format!r}")
