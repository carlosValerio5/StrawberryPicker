"""Crop geometry: the one place that decides which pixels a fruit's crop contains (ADR 0011).

Detection sources put several berries in one photo, and a padded square around one berry
often shows part of its neighbours. A :class:`CropPolicy` says what to do about them:

- ``keep``: crop as-is (how ``maturity_cls_v0.1`` was cut);
- ``mask``: paint other annotated boxes with ``fill``, except where they overlap the
  target's own box;
- ``drop``: leave crops whose neighbour overlap exceeds ``max_neighbour_overlap`` out of a
  *training* release. Evaluation and inference never skip a berry, so there it crops as
  ``keep`` (:meth:`CropPolicy.for_evaluation`);
- ``segment``: keep only the target's own pixels from a ``segment_fn`` mask (SAM), fill
  the rest. Removes neighbours and background together.

Gold, the audit embeddings, the phone-set evaluation and inference all crop through
:func:`crop_fruit`, so a model is scored on crops cut the way it was trained.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from strawberry_picker.paths import Layer, layer_dir

BBox = tuple[float, float, float, float]
Rect = tuple[float, float, float, float]
# ``segment_fn(image, bbox) -> bool array (height, width)``: True on the target fruit.
SegmentFn = Callable[[Image.Image, Sequence[float]], np.ndarray]

POLICY_FILE = "crop_policy.json"  # written into every gold release


class NeighbourMode(StrEnum):
    KEEP = "keep"
    MASK = "mask"
    DROP = "drop"
    SEGMENT = "segment"


@dataclass(frozen=True)
class CropPolicy:
    padding: float = 0.15
    neighbours: NeighbourMode = NeighbourMode.KEEP
    max_neighbour_overlap: float = 0.10  # only used by ``drop``
    fill: tuple[int, int, int] = (128, 128, 128)  # mid grey: no hue, so no ripeness signal

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "neighbours", NeighbourMode(self.neighbours))
        except ValueError:
            modes = [m.value for m in NeighbourMode]
            raise ValueError(
                f"neighbours must be one of {modes}, got {self.neighbours!r}"
            ) from None
        fill = tuple(int(v) for v in self.fill)
        if len(fill) != 3 or not all(0 <= v <= 255 for v in fill):
            raise ValueError(f"fill must be three 0-255 values, got {self.fill!r}")
        object.__setattr__(self, "fill", fill)
        if self.padding < 0:
            raise ValueError(f"padding must be >= 0, got {self.padding}")
        if not 0 <= self.max_neighbour_overlap <= 1:
            raise ValueError(
                f"max_neighbour_overlap must be in [0, 1], got {self.max_neighbour_overlap}"
            )

    @classmethod
    def from_settings(cls, gold: Mapping[str, Any]) -> CropPolicy:
        """From ``configs/data.yaml`` ``gold``: ``crop_padding`` plus ``crop_neighbours``."""
        neighbours = gold.get("crop_neighbours") or {}
        return cls(
            padding=float(gold.get("crop_padding", cls.padding)),
            neighbours=neighbours.get("mode", NeighbourMode.KEEP),
            max_neighbour_overlap=float(neighbours.get("max_overlap", cls.max_neighbour_overlap)),
            fill=tuple(neighbours.get("fill", cls().fill)),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CropPolicy:
        return cls(**{**raw, "fill": tuple(raw.get("fill", cls().fill))})

    @classmethod
    def from_json(cls, text: str) -> CropPolicy:
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "neighbours": self.neighbours.value, "fill": list(self.fill)}

    def to_json(self) -> str:
        """Canonical JSON: equal policies give equal strings (used as a cache key)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def keeps(self, overlap: float) -> bool:
        """Whether a training crop with this neighbour overlap goes into gold."""
        return self.neighbours is not NeighbourMode.DROP or overlap <= self.max_neighbour_overlap

    def for_evaluation(self) -> CropPolicy:
        """The policy for scoring and inference: every berry gets a crop, so drop → keep."""
        if self.neighbours is NeighbourMode.DROP:
            return replace(self, neighbours=NeighbourMode.KEEP)
        return self


LEGACY_POLICY = CropPolicy()  # how every release cut before ADR 0011 was cropped


# --- geometry ---------------------------------------------------------------------------


def crop_box(
    bbox: Iterable[float] | None, width: int, height: int, padding: float = 0.15
) -> tuple[int, int, int, int]:
    """Square crop around a fruit with ``padding`` × its longest side of context on each side.

    The square is shifted to stay inside the image, and clipped when it can't fit.
    ``bbox=None`` (single-fruit image) returns the whole image.
    """
    if bbox is None:
        return (0, 0, width, height)
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) * (1 + 2 * padding)

    def axis(center: float, limit: int) -> tuple[int, int]:
        if side >= limit:
            return 0, limit
        start = round(min(max(center - side / 2, 0), limit - side))
        return start, min(limit, start + round(side))

    left, right = axis((x0 + x1) / 2, width)
    top, bottom = axis((y0 + y1) / 2, height)
    return (left, top, right, bottom)


def _intersect(a: Rect, b: Rect) -> Rect | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _union_area(rects: Sequence[Rect]) -> float:
    """Exact area of a union of rectangles (coordinate compression)."""
    if not rects:
        return 0.0
    xs = np.unique([v for r in rects for v in (r[0], r[2])])
    ys = np.unique([v for r in rects for v in (r[1], r[3])])
    covered = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for x0, y0, x1, y1 in rects:
        covered[
            np.searchsorted(ys, y0) : np.searchsorted(ys, y1),
            np.searchsorted(xs, x0) : np.searchsorted(xs, x1),
        ] = True
    return float((covered * np.outer(np.diff(ys), np.diff(xs))).sum())


def _boxes(values: Iterable[Any] | None) -> list[Rect]:
    """Boxes as float tuples; ``None``/NaN entries (whole-image fruit) are skipped."""
    if values is None or isinstance(values, float):
        return []
    return [
        tuple(float(v) for v in box)  # type: ignore[misc]
        for box in values
        if box is not None and not isinstance(box, float)
    ]


def neighbour_overlap(
    crop: Sequence[float], target_bbox: Sequence[float] | None, other_bboxes: Iterable[Any]
) -> float:
    """Fraction of ``crop`` covered by other fruit boxes, outside the target's own box.

    Overlapping neighbours count once (union, not sum). This is the share of the crop that
    ``mask`` paints over.
    """
    crop_rect: Rect = tuple(float(v) for v in crop)  # type: ignore[assignment]
    area = (crop_rect[2] - crop_rect[0]) * (crop_rect[3] - crop_rect[1])
    seen = [r for r in (_intersect(b, crop_rect) for b in _boxes(other_bboxes)) if r]
    if area <= 0 or not seen:
        return 0.0
    covered = _union_area(seen)
    if target_bbox is not None:
        target: Rect = tuple(float(v) for v in target_bbox)  # type: ignore[assignment]
        covered -= _union_area([r for r in (_intersect(s, target) for s in seen) if r])
    return covered / area


def _rasterize(rect: Rect, width: int, height: int) -> tuple[slice, slice]:
    x0, y0 = max(0, round(rect[0])), max(0, round(rect[1]))
    x1, y1 = min(width, round(rect[2])), min(height, round(rect[3]))
    return slice(y0, max(y0, y1)), slice(x0, max(x0, x1))


def crop_fruit(
    image: Image.Image,
    bbox: Sequence[float] | None,
    neighbours: Iterable[Any] | None,
    policy: CropPolicy,
    *,
    segment_fn: SegmentFn | None = None,
) -> Image.Image:
    """The crop of one fruit under ``policy``. ``drop`` crops like ``keep`` here; filtering is
    the caller's job (:meth:`CropPolicy.keeps`)."""
    box = crop_box(bbox, image.width, image.height, policy.padding)
    mode = policy.neighbours
    if mode in (NeighbourMode.KEEP, NeighbourMode.DROP):
        return image.crop(box)

    pixels = np.array(image.convert("RGB"))
    if mode is NeighbourMode.SEGMENT:
        if segment_fn is None:
            raise ValueError("the segment policy needs a segment_fn (e.g. sam_segmenter())")
        target = bbox if bbox is not None else (0.0, 0.0, float(image.width), float(image.height))
        keep = np.asarray(segment_fn(image, tuple(target)), dtype=bool)
        if keep.shape != pixels.shape[:2]:
            raise ValueError(f"segment mask is {keep.shape}, image is {pixels.shape[:2]}")
        pixels[~keep] = policy.fill
    else:  # MASK
        paint = np.zeros(pixels.shape[:2], dtype=bool)
        for other in _boxes(neighbours):
            paint[_rasterize(other, image.width, image.height)] = True
        if bbox is not None:
            target_rect: Rect = tuple(float(v) for v in bbox)  # type: ignore[assignment]
            paint[_rasterize(target_rect, image.width, image.height)] = False
        pixels[paint] = policy.fill
    return Image.fromarray(pixels).crop(box)


def with_neighbours(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add ``neighbour_bboxes``: the other fruit boxes of the same image, for every row.

    Pass the *whole* silver manifest: a box that still needs review, or has no label for
    this task, is still a berry in the picture.
    """
    out = manifest.copy()
    own = [_boxes([bbox]) for bbox in out.bbox]  # zero or one box per row
    neighbours: list[list[list[float]]] = [[] for _ in range(len(out))]
    for rows in out.groupby("image_id", sort=False).indices.values():
        for r in rows:
            neighbours[r] = [list(box) for o in rows if o != r for box in own[o]]
    out["neighbour_bboxes"] = neighbours
    return out


# --- which policy a model was trained with ----------------------------------------------


def resolve_policy(run_config: Mapping[str, Any] | None) -> CropPolicy:
    """The crop policy a training run used, so evaluation can cut test crops the same way.

    Order: an explicit ``crop_policy`` in the run's resolved config; else the
    ``crop_policy.json`` of its gold ``dataset``; a release older than ADR 0011 (no file)
    used :data:`LEGACY_POLICY`; with no run or release, ``configs/data.yaml``.
    """
    config = run_config or {}
    if config.get("crop_policy"):
        return CropPolicy.from_dict(config["crop_policy"])
    dataset = config.get("dataset")
    if dataset:
        release = layer_dir(Layer.GOLD) / str(dataset)
        if (release / POLICY_FILE).is_file():
            return CropPolicy.from_json((release / POLICY_FILE).read_text())
        if release.is_dir():
            return LEGACY_POLICY
    from strawberry_picker.config import load_config

    return CropPolicy.from_settings(load_config("data").get("gold", {}))


def sam_segmenter(model_name: str = "facebook/sam-vit-base") -> SegmentFn:  # pragma: no cover
    """Box-prompted SAM masks (weights Apache-2.0). Needs the ``ml`` extra and ~375 MB weights.

    SAM 2 (``Sam2Model``, also Apache-2.0) drops in the same way on transformers versions
    that ship it.
    """
    import torch
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def segment(image: Image.Image, bbox: Sequence[float]) -> np.ndarray:
        rgb = image.convert("RGB")
        inputs = processor(rgb, input_boxes=[[list(map(float, bbox))]], return_tensors="pt")
        outputs = model(**inputs.to(device), multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][0, 0].numpy().astype(bool)

    return segment
