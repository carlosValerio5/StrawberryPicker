"""Predict maturity on a new photo: optional OWLv2 boxes, then DINOv2 + linear probe."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from strawberry_picker.config import load_config
from strawberry_picker.data.gold import crop_box
from strawberry_picker.labeling.proposals import DetectFn, keep_box, non_max_suppression
from strawberry_picker.models.linear_probe import load_probe, predict_all_classes
from strawberry_picker.paths import models_dir

EmbedFn = Callable[[list[Image.Image]], np.ndarray]

_LABEL_RGB: dict[str, tuple[int, int, int]] = {
    "unripe": (46, 160, 67),
    "ripe": (200, 40, 40),
    "overripe": (120, 60, 140),
}


@dataclass(frozen=True, slots=True)
class FruitPrediction:
    bbox: tuple[float, float, float, float] | None
    label: str
    confidence: float
    probs: dict[str, float]
    detection_score: float | None = None


@dataclass(frozen=True, slots=True)
class MaturityRun:
    path: Path
    probe: Any
    classes: tuple[str, ...]
    backbone: str
    config: dict[str, Any]


def latest_maturity_run(task: str = "maturity_cls") -> Path | None:
    runs = sorted((models_dir() / task).glob("*/probe.joblib"))
    return runs[-1].parent if runs else None


def load_maturity_run(run_dir: Path | None = None) -> MaturityRun:
    path = run_dir or latest_maturity_run()
    if path is None:
        raise FileNotFoundError("no model run yet: uv run python scripts/train_maturity.py")
    config = load_config(path / "config.yaml")
    return MaturityRun(
        path=path,
        probe=load_probe(path),
        classes=tuple(config["classes"]),
        backbone=str(config["model"]["backbone"]),
        config=config,
    )


def _filter_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    *,
    min_score: float,
    iou: float,
    min_side_frac: float,
    max_aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = scores >= min_score
    boxes, scores = boxes[mask], scores[mask]
    if len(boxes) == 0:
        return boxes, scores
    order = non_max_suppression(boxes, scores, iou=iou)
    kept_boxes, kept_scores = [], []
    for i in order:
        if keep_box(boxes[i], width, height, min_side_frac=min_side_frac, max_aspect=max_aspect):
            kept_boxes.append(boxes[i])
            kept_scores.append(scores[i])
    if not kept_boxes:
        return np.zeros((0, 4)), np.zeros(0)
    return np.asarray(kept_boxes, dtype=float), np.asarray(kept_scores, dtype=float)


def predict_image(
    image: Image.Image,
    probe: Any,
    embed_fn: EmbedFn,
    classes: Sequence[str],
    *,
    detect_fn: DetectFn | None = None,
    min_score: float = 0.2,
    iou: float = 0.5,
    min_side_frac: float = 0.25,
    max_aspect: float = 2.0,
    padding: float = 0.15,
) -> list[FruitPrediction]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    boxes: list[tuple[float, float, float, float] | None] = [None]
    det_scores: list[float | None] = [None]

    if detect_fn is not None:
        raw_boxes, raw_scores = detect_fn(rgb)
        filtered, filtered_scores = _filter_boxes(
            np.asarray(raw_boxes, dtype=float),
            np.asarray(raw_scores, dtype=float),
            rgb.width,
            rgb.height,
            min_score=min_score,
            iou=iou,
            min_side_frac=min_side_frac,
            max_aspect=max_aspect,
        )
        if len(filtered):
            boxes = [tuple(map(float, box)) for box in filtered]
            det_scores = [float(s) for s in filtered_scores]

    crops = [
        rgb.crop(crop_box(list(box) if box is not None else None, *rgb.size, padding=padding))
        for box in boxes
    ]
    probs = predict_all_classes(probe, embed_fn(crops), len(classes))
    out: list[FruitPrediction] = []
    for box, det, row in zip(boxes, det_scores, probs, strict=True):
        idx = int(row.argmax())
        out.append(
            FruitPrediction(
                bbox=box,
                label=str(classes[idx]),
                confidence=float(row[idx]),
                probs={str(c): float(p) for c, p in zip(classes, row, strict=True)},
                detection_score=det,
            )
        )
    return out


def annotate_image(
    image: Image.Image,
    predictions: Sequence[FruitPrediction],
    *,
    line_width: int = 4,
) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    canvas = rgb.copy()
    draw = ImageDraw.Draw(canvas)
    size = max(28, rgb.width // 18)
    font = ImageFont.load_default()
    for candidate in (
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if candidate.exists():
            font = ImageFont.truetype(str(candidate), size=size)
            break

    def _banner(x: float, y: float, text: str, color: tuple[int, int, int]) -> None:
        pad = 8
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
            fill=color,
        )
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    for pred in predictions:
        color = _LABEL_RGB.get(pred.label, (255, 200, 0))
        text = f"{pred.label} {pred.confidence:.0%}"
        if pred.bbox is None:
            draw.rectangle((0, 0, rgb.width - 1, rgb.height - 1), outline=color, width=line_width)
            _banner(16, 16, text, color)
            continue
        x0, y0, x1, y1 = pred.bbox
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        _banner(x0 + 4, max(4, y0 + 4), text, color)
    return canvas


def predictions_table(predictions: Sequence[FruitPrediction]) -> list[list[Any]]:
    rows = []
    for i, pred in enumerate(predictions, start=1):
        probs = " · ".join(f"{k}={v:.0%}" for k, v in pred.probs.items())
        det = "" if pred.detection_score is None else f"{pred.detection_score:.2f}"
        rows.append([i, pred.label, f"{pred.confidence:.0%}", det, probs])
    return rows
