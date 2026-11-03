"""Proposed fruit boxes for multi-berry phone photos, so nobody draws boxes by hand.

``propose`` takes any ``detect_fn(image) -> (boxes_xyxy, scores)``; tests use a fake, the
protocol uses ``owlv2_detector`` (zero-shot, Apache-2.0). The labeler rejects bad boxes with
``n`` (not a strawberry) in ``scripts/label.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from strawberry_picker.labeling.tool import TASK_COLUMNS

DetectFn = Callable[[Image.Image], tuple[np.ndarray, np.ndarray]]


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, *, iou: float) -> np.ndarray:
    """Indices of kept boxes, best first; a box overlapping a kept one by > ``iou`` is dropped."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept: list[int] = []
    for i in order:
        if kept:
            k = np.array(kept)
            x0 = np.maximum(boxes[i, 0], boxes[k, 0])
            y0 = np.maximum(boxes[i, 1], boxes[k, 1])
            x1 = np.minimum(boxes[i, 2], boxes[k, 2])
            y1 = np.minimum(boxes[i, 3], boxes[k, 3])
            overlap = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
            if (overlap / (areas[i] + areas[k] - overlap)).max() > iou:
                continue
        kept.append(int(i))
    return np.array(kept, dtype=int)


def propose(
    photos: Sequence[Path], detect_fn: DetectFn, *, min_score: float = 0.2, iou: float = 0.5
) -> pd.DataFrame:
    """A labeling task with one row per proposed fruit: ids ``<photo stem>_<n>``, best first."""
    rows = []
    for photo in sorted(photos, key=lambda p: Path(p).name):
        with Image.open(photo) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        boxes, scores = detect_fn(image)
        boxes, scores = np.asarray(boxes, dtype=float), np.asarray(scores, dtype=float)
        confident = scores >= min_score
        boxes, scores = boxes[confident], scores[confident]
        stem = Path(photo).stem
        for n, i in enumerate(non_max_suppression(boxes, scores, iou=iou)):
            rows.append(
                (f"{stem}_{n}", str(photo), json.dumps([float(v) for v in boxes[i]]), "", stem)
            )
    return pd.DataFrame(rows, columns=list(TASK_COLUMNS))


def owlv2_detector(  # pragma: no cover - needs the ml extra and model weights
    model_name: str = "google/owlv2-base-patch16-ensemble",
    query: str = "a photo of a strawberry",
) -> DetectFn:
    """Zero-shot OWLv2 boxes for ``query``. Weights: Apache-2.0."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def detect(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        inputs = processor(text=[[query]], images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        side = max(image.size)  # OWLv2 pads to a square before predicting
        result = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=torch.tensor([[side, side]])
        )[0]
        boxes = result["boxes"].cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)
        return boxes, result["scores"].cpu().numpy()

    return detect
