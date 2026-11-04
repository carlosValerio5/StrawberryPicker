"""Photo → one prediction per berry: detect, crop the way gold does, embed, classify.

Every model is injected, as in ``labeling/proposals.py``: tests pass fakes, real use passes
``owlv2_detector`` (ADR 0009) and ``dinov2_embedder``. The classifier only ever saw one-berry
crops, so a whole multi-berry photo must never reach it directly; this module is the only
supported way to run a maturity run on a new photo.

Extension point: ``heads`` are callables ``(embeddings) -> [dict per berry]``. A defect head
plugs in here and adds ``defects``/``grade`` to every berry, reusing the same embeddings.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from strawberry_picker.config import load_config
from strawberry_picker.data.gold import crop_box, dataset_dir
from strawberry_picker.data.readers import IMAGE_SUFFIXES
from strawberry_picker.labeling.proposals import DetectFn, non_max_suppression
from strawberry_picker.models.linear_probe import load_probe, predict_all_classes
from strawberry_picker.paths import Layer, layer_dir

BBox = tuple[float, float, float, float]
EmbedFn = Callable[[list[Image.Image]], np.ndarray]
CropFn = Callable[[BBox, int, int], tuple[int, int, int, int]]
HeadFn = Callable[[np.ndarray], Sequence[dict[str, Any]]]

HEIC_SUFFIXES = frozenset({".heic", ".heif"})
LOCKED_SOURCE = "phone_testset"  # AGENTS.md rule 6: read only by evaluate_phone_set.py


class UnsupportedImageError(ValueError):
    """The file can't be decoded (HEIC, corrupt, not an image)."""


class LockedSetError(PermissionError):
    """Refused to read the locked phone test set."""


@dataclass(frozen=True)
class BerryPrediction:
    bbox: BBox
    score: float  # detector confidence
    maturity: str
    probabilities: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)  # filled by heads (defects, grade…)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(v) for v in self.bbox],
            "score": float(self.score),
            "maturity": self.maturity,
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            **self.extras,
        }


def load_image(path: Path) -> Image.Image:
    """Decode like silver does: EXIF orientation applied, then RGB."""
    path = Path(path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        raise UnsupportedImageError(
            f"{path.name}: HEIC can't be read. Export the photo as JPEG, or set the iPhone "
            "camera to 'Most Compatible' (Settings > Camera > Formats)."
        )
    try:
        with Image.open(path) as raw:
            return ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise UnsupportedImageError(f"{path.name}: not a readable image ({error})") from error


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def ensure_not_locked(paths: Iterable[Path]) -> None:
    """Raise if any path is, contains, or lies inside ``bronze/phone_testset``."""
    locked = (layer_dir(Layer.BRONZE) / LOCKED_SOURCE).resolve()
    for path in paths:
        resolved = Path(path).resolve()
        if _inside(resolved, locked) or _inside(locked, resolved):
            raise LockedSetError(
                f"{path} is or contains the locked phone test set; only "
                "scripts/evaluate_phone_set.py may read it (AGENTS.md rule 6)."
            )


def image_paths(inputs: Iterable[Path]) -> list[Path]:
    """Images named directly, plus every image under named folders (HEIC kept, to report)."""
    suffixes = IMAGE_SUFFIXES | HEIC_SUFFIXES
    found: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            found += [p for p in item.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
        elif item.is_file():
            found.append(item)
        else:
            raise FileNotFoundError(f"not found: {item}")
    return sorted(set(found))


def predict_image(
    image: Image.Image,
    *,
    detect_fn: DetectFn,
    embed_fn: EmbedFn,
    probe: Any,
    classes: Sequence[str],
    crop_fn: CropFn | None = None,
    padding: float = 0.15,
    min_score: float = 0.2,
    iou: float = 0.5,
    heads: Sequence[HeadFn] = (),
) -> list[BerryPrediction]:
    """One prediction per detected berry, highest detector score first.

    ``crop_fn`` defaults to gold's padded square (``padding`` must match the training crops).
    Thresholds match ``labeling.proposals.propose``.
    """
    crop_fn = crop_fn or partial(crop_box, padding=padding)
    boxes, scores = detect_fn(image)
    boxes, scores = np.asarray(boxes, dtype=float), np.asarray(scores, dtype=float)
    confident = scores >= min_score
    boxes, scores = boxes[confident], scores[confident]
    if not len(boxes):
        return []
    kept = non_max_suppression(boxes, scores, iou=iou)
    bboxes = [tuple(float(v) for v in boxes[i]) for i in kept]
    crops = [image.crop(crop_fn(bbox, image.width, image.height)) for bbox in bboxes]
    features = embed_fn(crops)
    probs = predict_all_classes(probe, features, len(classes))

    extras: list[dict[str, Any]] = [{} for _ in bboxes]
    for head in heads:
        results = list(head(features))
        if len(results) != len(bboxes):
            raise ValueError(f"head returned {len(results)} results for {len(bboxes)} berries")
        for extra, result in zip(extras, results, strict=True):
            extra.update(result)

    return [
        BerryPrediction(
            bbox=bbox,  # type: ignore[arg-type]
            score=float(scores[i]),
            maturity=classes[int(row.argmax())],
            probabilities={name: float(p) for name, p in zip(classes, row, strict=True)},
            extras=extra,
        )
        for bbox, i, row, extra in zip(bboxes, kept, probs, extras, strict=True)
    ]


_CARD_PADDING = re.compile(r"^- \*\*padding\*\*: `([0-9.]+)`", re.MULTILINE)


def resolve_padding(config: dict[str, Any]) -> float:
    """Crop padding the run was trained with: ``crop_padding`` in the run config, else the
    padding recorded in its gold release's dataset card."""
    if config.get("crop_padding") is not None:
        return float(config["crop_padding"])
    card = dataset_dir(config["dataset"]) / "DATASET_CARD.md" if "dataset" in config else None
    if card is not None and card.exists():
        match = _CARD_PADDING.search(card.read_text())
        if match:
            return float(match.group(1))
    raise ValueError(
        f"can't tell which crop padding gold {config.get('dataset')!r} used; pass --padding "
        "with the value from its DATASET_CARD.md"
    )


@dataclass
class Predictor:
    """A trained run plus the detector and embedder it needs, ready for photos."""

    probe: Any
    classes: tuple[str, ...]
    detect_fn: DetectFn
    embed_fn: EmbedFn
    padding: float
    min_score: float = 0.2
    iou: float = 0.5
    heads: Sequence[HeadFn] = ()

    def predict(self, image: Image.Image) -> list[BerryPrediction]:
        return predict_image(
            image,
            detect_fn=self.detect_fn,
            embed_fn=self.embed_fn,
            probe=self.probe,
            classes=self.classes,
            padding=self.padding,
            min_score=self.min_score,
            iou=self.iou,
            heads=self.heads,
        )


def _default_detector() -> DetectFn:  # pragma: no cover - needs the ml extra and weights
    from strawberry_picker.labeling.proposals import owlv2_detector

    return owlv2_detector()


def _default_embedder(backbone: str) -> EmbedFn:  # pragma: no cover - needs the ml extra
    from strawberry_picker.audit.embeddings import dinov2_embedder

    return dinov2_embedder(backbone)


def load_predictor(
    run_dir: Path,
    *,
    detect_fn: DetectFn | None = None,
    embed_factory: Callable[[str], EmbedFn] | None = None,
    padding: float | None = None,
    min_score: float = 0.2,
    iou: float = 0.5,
) -> Predictor:
    """Load ``models/maturity_cls/<run_id>/``; the run's config names backbone and classes."""
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    return Predictor(
        probe=load_probe(run_dir),
        classes=tuple(config["classes"]),
        detect_fn=detect_fn or _default_detector(),
        embed_fn=(embed_factory or _default_embedder)(config["model"]["backbone"]),
        padding=resolve_padding(config) if padding is None else padding,
        min_score=min_score,
        iou=iou,
    )


def annotate(image: Image.Image, predictions: Sequence[BerryPrediction]) -> Image.Image:
    """A copy of ``image`` with each berry's box, maturity and confidence drawn on it."""
    colours = {"unripe": (60, 200, 60), "ripe": (230, 30, 30), "overripe": (140, 60, 160)}
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(max(image.size) / 300))
    for p in predictions:
        colour = colours.get(p.maturity, (255, 200, 0))
        draw.rectangle(p.bbox, outline=colour, width=width)
        label = f"{p.maturity} {p.probabilities.get(p.maturity, 0.0):.0%}"
        x, y = p.bbox[0], max(0.0, p.bbox[1] - 12)
        draw.rectangle(draw.textbbox((x, y), label), fill=colour)
        draw.text((x, y), label, fill=(255, 255, 255))
    return out


def predict_paths(
    paths: Sequence[Path], predictor: Predictor, *, annotate_dir: Path | None = None
) -> list[dict[str, Any]]:
    """One JSON-ready record per image: ``{image, berries, error}``; bad files don't stop the run.

    Refuses the locked phone test set before reading anything.
    """
    ensure_not_locked(paths)
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            image = load_image(path)
        except UnsupportedImageError as error:
            records.append({"image": str(path), "berries": [], "error": str(error)})
            continue
        predictions = predictor.predict(image)
        records.append(
            {"image": str(path), "berries": [p.to_dict() for p in predictions], "error": None}
        )
        if annotate_dir is not None:
            annotate_dir.mkdir(parents=True, exist_ok=True)
            annotate(image, predictions).save(annotate_dir / f"{Path(path).stem}.jpg", quality=90)
    return records
