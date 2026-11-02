"""Per-label contact sheets: a random sample of each source label, tiled into one image.

This is how a source's labels are checked before it's enabled. If the ``unripe`` sheet
is full of red fruit, the mapping or the YOLO class order is wrong. Sheets apply the
source's ``exclude`` globs, so they show what silver will see, and never modify bronze.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from strawberry_picker.data.gold import crop_box
from strawberry_picker.data.readers import Annotation, read_annotations
from strawberry_picker.data.sources import SourceSpec

HEADER_PX = 24
SUMMARY_COLUMNS = ("source_label", "annotations", "images", "mapped_to", "sheet")
_BACKGROUND = (255, 255, 255)
_UNREADABLE = (128, 128, 128)  # grey tile: the file couldn't be decoded


def label_slug(label: str) -> str:
    """Filename-safe version of a source label: ``Normal/Ripe`` → ``Normal_Ripe``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _order(annotation: Annotation) -> tuple[str, tuple[float, ...]]:
    return annotation.image_path.as_posix(), annotation.bbox or ()


def sample_per_label(
    annotations: Iterable[Annotation], per_label: int, seed: int
) -> dict[str, list[Annotation]]:
    """Up to ``per_label`` annotations per label, the same for any input order and ``seed``."""
    by_label: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        by_label[annotation.source_label].append(annotation)
    sample = {}
    for label in sorted(by_label):
        items = sorted(by_label[label], key=_order)
        if len(items) > per_label:
            items = sorted(random.Random(f"{seed}:{label}").sample(items, per_label), key=_order)
        sample[label] = items
    return sample


def _tile(annotation: Annotation, tile_px: int, padding: float) -> Image.Image:
    try:
        with Image.open(annotation.image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except OSError:
        return Image.new("RGB", (tile_px, tile_px), _UNREADABLE)
    crop = image.crop(crop_box(annotation.bbox, *image.size, padding=padding))
    return ImageOps.pad(crop, (tile_px, tile_px), color=_BACKGROUND)


def render_sheet(
    annotations: Sequence[Annotation],
    *,
    title: str,
    tile_px: int = 128,
    columns: int = 10,
    padding: float = 0.15,
) -> Image.Image:
    """Tile one padded crop per annotation under a one-line ``title`` header."""
    rows = max(1, math.ceil(len(annotations) / columns))
    sheet = Image.new("RGB", (columns * tile_px, HEADER_PX + rows * tile_px), _BACKGROUND)
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0))
    for i, annotation in enumerate(annotations):
        row, col = divmod(i, columns)
        sheet.paste(_tile(annotation, tile_px, padding), (col * tile_px, HEADER_PX + row * tile_px))
    return sheet


def _describe_mapping(spec: SourceSpec, label: str) -> str:
    if label not in spec.labels:
        return "drop (unmapped)" if spec.drop_unmapped else "UNMAPPED"
    mapping = spec.labels[label]
    if mapping is None:
        return "drop"
    parts = []
    if mapping.maturity is not None:
        parts.append(f"maturity={mapping.maturity}")
    if mapping.defects is not None:
        parts.append(f"defects=[{','.join(mapping.defects)}]")
    if mapping.needs_review:
        parts.append("needs_review")
    return " ".join(parts) or "no labels"


def label_summary(spec: SourceSpec, annotations: Iterable[Annotation]) -> pd.DataFrame:
    """Per source label: annotation and image counts, and what the registry maps it to."""
    frame = pd.DataFrame(
        [(a.source_label, a.image_path.as_posix()) for a in annotations],
        columns=["source_label", "image"],
    )
    summary = (
        frame.groupby("source_label")
        .agg(annotations=("image", "size"), images=("image", "nunique"))
        .reset_index()
    )
    summary["mapped_to"] = summary.source_label.map(lambda label: _describe_mapping(spec, label))
    return summary


def build_contact_sheets(
    spec: SourceSpec,
    pull_dir: Path,
    out_dir: Path,
    *,
    per_label: int = 50,
    seed: int = 0,
    tile_px: int = 128,
    columns: int = 10,
    padding: float = 0.15,
) -> pd.DataFrame:
    """Write ``<label>.jpg`` per source label and ``summary.csv`` into ``out_dir``."""
    annotations = [
        a
        for a in read_annotations(spec, pull_dir)
        if not any(fnmatch(a.image_path.relative_to(pull_dir).as_posix(), p) for p in spec.exclude)
    ]
    if not annotations:
        raise ValueError(f"{spec.name}: no annotations under {pull_dir / spec.root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = label_summary(spec, annotations).set_index("source_label")
    for label, sample in sample_per_label(annotations, per_label, seed).items():
        row = summary.loc[label]
        title = f"{spec.name} | {label} | {len(sample)} of {row.annotations} | {row.mapped_to}"
        sheet = f"{label_slug(label)}.jpg"
        render_sheet(sample, title=title, tile_px=tile_px, columns=columns, padding=padding).save(
            out_dir / sheet, quality=90
        )
        summary.loc[label, "sheet"] = sheet
    summary = summary.reset_index()[list(SUMMARY_COLUMNS)]
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary
