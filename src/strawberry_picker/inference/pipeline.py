"""Photo → one prediction per berry: detect, crop under the run's policy, embed, classify.

Every model is injected, as in ``labeling/proposals.py``: tests pass fakes, real use passes
``owlv2_detector`` (ADR 0009) and ``dinov2_embedder``. The classifier only ever saw one-berry
crops, so a whole multi-berry photo must never reach it directly; this module is the only
supported way to run a maturity run on a new photo.

Crops are cut by ``data.crops.crop_fruit`` under the crop policy the run was trained with
(ADR 0011), with the other detected berries of the same photo as neighbours.

Extension point: ``heads`` are callables ``(embeddings) -> [dict per berry]``. The defect
head (ADR 0010) plugs in through :func:`quality_head` and adds ``defects``/``grade`` to every
berry, reusing the same embeddings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from strawberry_picker.config import load_config
from strawberry_picker.data.crops import (
    CropPolicy,
    NeighbourMode,
    SegmentFn,
    crop_fruit,
    resolve_policy,
)
from strawberry_picker.data.readers import IMAGE_SUFFIXES
from strawberry_picker.images import open_rgb
from strawberry_picker.labeling.proposals import DetectFn, non_max_suppression
from strawberry_picker.models.defect_probe import DefectHead
from strawberry_picker.models.linear_probe import load_probe, predict_all_classes
from strawberry_picker.paths import Layer, layer_dir, models_dir
from strawberry_picker.schema import grade_from_defects

BBox = tuple[float, float, float, float]
EmbedFn = Callable[[list[Image.Image]], np.ndarray]
HeadFn = Callable[[np.ndarray], Sequence[dict[str, Any]]]

LOCKED_SOURCE = "phone_testset"  # AGENTS.md rule 6: read only by evaluate_phone_set.py


class UnsupportedImageError(ValueError):
    """The file can't be decoded (corrupt, truncated, not an image)."""


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
    """Decode like silver does (EXIF orientation, then RGB), HEIC included."""
    path = Path(path)
    try:
        return open_rgb(path)
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
    """Images named directly, plus every image under named folders (HEIC included)."""
    found: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            found += [
                p for p in item.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
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
    policy: CropPolicy | None = None,
    padding: float = 0.15,
    min_score: float = 0.2,
    iou: float = 0.5,
    heads: Sequence[HeadFn] = (),
    segment_fn: SegmentFn | None = None,
) -> list[BerryPrediction]:
    """One prediction per detected berry, highest detector score first.

    Each berry is cropped by ``crop_fruit`` under ``policy`` (default: padded square with
    ``padding``, which must match the training crops), and the other kept berries of the
    photo are its neighbours. Evaluation semantics apply: ``drop`` crops like ``keep``.
    Thresholds match ``labeling.proposals.propose``.
    """
    policy = (policy or CropPolicy(padding=padding)).for_evaluation()
    boxes, scores = detect_fn(image)
    boxes, scores = np.asarray(boxes, dtype=float), np.asarray(scores, dtype=float)
    confident = scores >= min_score
    boxes, scores = boxes[confident], scores[confident]
    if not len(boxes):
        return []
    kept = non_max_suppression(boxes, scores, iou=iou)
    bboxes = [tuple(float(v) for v in boxes[i]) for i in kept]
    crops = [
        crop_fruit(
            image,
            bbox,
            [other for j, other in enumerate(bboxes) if j != i],
            policy,
            segment_fn=segment_fn,
        )
        for i, bbox in enumerate(bboxes)
    ]
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


def run_policy(config: dict[str, Any], padding: float | None = None) -> CropPolicy:
    """The crop policy a run's crops were cut with (ADR 0011).

    ``resolve_policy`` reads the run's ``crop_policy``, else its gold release's
    ``crop_policy.json``, else the legacy policy of an older release. Runs that predate ADR
    0011 may also carry a ``crop_padding`` key, which is honoured when nothing else is set.
    ``padding`` (a command-line override) replaces the padding only.
    """
    policy = resolve_policy(config)
    if (
        padding is None
        and config.get("crop_policy") is None
        and config.get("crop_padding") is not None
    ):
        padding = float(config["crop_padding"])
    return policy if padding is None else replace(policy, padding=float(padding))


@dataclass
class Predictor:
    """A trained run plus the detector and embedder it needs, ready for photos."""

    probe: Any
    classes: tuple[str, ...]
    detect_fn: DetectFn
    embed_fn: EmbedFn
    policy: CropPolicy = field(default_factory=CropPolicy)
    min_score: float = 0.2
    iou: float = 0.5
    heads: Sequence[HeadFn] = ()
    segment_fn: SegmentFn | None = None

    def predict(self, image: Image.Image) -> list[BerryPrediction]:
        return predict_image(
            image,
            detect_fn=self.detect_fn,
            embed_fn=self.embed_fn,
            probe=self.probe,
            classes=self.classes,
            policy=self.policy,
            min_score=self.min_score,
            iou=self.iou,
            heads=self.heads,
            segment_fn=self.segment_fn,
        )


def _default_detector() -> DetectFn:  # pragma: no cover - needs the ml extra and weights
    from strawberry_picker.labeling.proposals import owlv2_detector

    return owlv2_detector()


def _default_embedder(backbone: str) -> EmbedFn:  # pragma: no cover - needs the ml extra
    from strawberry_picker.audit.embeddings import dinov2_embedder

    return dinov2_embedder(backbone)


def _default_segmenter() -> SegmentFn:  # pragma: no cover - needs the ml extra and weights
    from strawberry_picker.data.crops import sam_segmenter

    return sam_segmenter()


def quality_head(run_dir: Path, *, backbone: str, policy: CropPolicy) -> HeadFn:
    """A defect run (``models/quality_cls/<run_id>/``, ADR 0010) as an inference head.

    The head reuses the maturity run's embeddings, so its backbone and crop policy must be
    the maturity run's. Each berry gets ``defects`` (a list) and ``grade`` (by rule).
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    if config["model"]["backbone"] != backbone:
        raise ValueError(
            f"{run_dir} was trained on backbone {config['model']['backbone']!r}, but the "
            f"maturity run uses {backbone!r}; the defect head needs the same embeddings"
        )
    quality_policy = run_policy(config)
    if quality_policy.to_json() != policy.to_json():
        raise ValueError(
            f"{run_dir} was trained with a different crop policy than the maturity run; the "
            "heads share one set of crops, so pass a matching run or --padding"
        )
    head: DefectHead = load_probe(run_dir)

    def defect_head(features: np.ndarray) -> list[dict[str, Any]]:
        return [
            {"defects": list(found), "grade": grade_from_defects(found).value}
            for found in head.predict(features)
        ]

    return defect_head


def load_predictor(
    run_dir: Path,
    *,
    quality_run: Path | None = None,
    detect_fn: DetectFn | None = None,
    embed_factory: Callable[[str], EmbedFn] | None = None,
    segment_fn: SegmentFn | None = None,
    padding: float | None = None,
    min_score: float = 0.2,
    iou: float = 0.5,
) -> Predictor:
    """Load ``models/maturity_cls/<run_id>/``; the run's config names backbone and classes.

    ``quality_run`` adds the defect head from ``models/quality_cls/<run_id>/``: defects and
    grade for every berry, from the same embeddings.
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    backbone = config["model"]["backbone"]
    policy = run_policy(config, padding)
    heads: list[HeadFn] = []
    if quality_run is not None:
        heads.append(quality_head(quality_run, backbone=backbone, policy=policy))
    if policy.neighbours is NeighbourMode.SEGMENT and segment_fn is None:
        segment_fn = _default_segmenter()
    return Predictor(
        probe=load_probe(run_dir),
        classes=tuple(config["classes"]),
        detect_fn=detect_fn or _default_detector(),
        embed_fn=(embed_factory or _default_embedder)(backbone),
        policy=policy,
        min_score=min_score,
        iou=iou,
        heads=heads,
        segment_fn=segment_fn,
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


def newest_run(task: str) -> Path | None:
    """The newest run folder under ``models/<task>/`` (run ids are UTC timestamps)."""
    runs = sorted((models_dir() / task).glob("*/probe.joblib"))
    return runs[-1].parent if runs else None
