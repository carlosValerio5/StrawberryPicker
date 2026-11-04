"""Find whole-image rows (``bbox`` is null) whose image actually shows several berries.

Image-level sources label the whole photo, and the classifier treats it as one fruit. About
1 in 4 sampled ``fresh_rotten_fruits_2022`` images is a pile (docs/datasets.md, finding 14).
A zero-shot detector counts the berries in each such image. It never relabels anything: the
flags become a labeling task (``scripts/make_label_task.py multi-fruit``), and a person's
"several berries" label is what holds a row out of gold (``data/reviews.py``).

``flag_multi_fruit`` takes any ``detect_factory() -> detect_fn(image) -> (boxes, scores)``,
so tests use a fake and the audit uses ``labeling.proposals.owlv2_detector``. Raw detections
are cached per image and detector, so changing ``min_score`` or ``iou`` never re-detects.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from strawberry_picker.audit.embeddings import with_crop_ids
from strawberry_picker.labeling.proposals import DetectFn, non_max_suppression

FLAG_COLUMNS = (
    "crop_id",
    "image_id",
    "source_name",
    "source_label",
    "n_detected",
    "max_score",
    "boxes",
)
CACHE_COLUMNS = ("image_id", "detector", "boxes", "scores")
CACHE_FLOOR = 0.01  # raw boxes below this score are never kept in the cache


def whole_image_rows(silver: pd.DataFrame) -> pd.DataFrame:
    """Silver rows whose label covers the whole image, with gold's ``crop_id``."""
    rows = with_crop_ids(silver)
    return rows[rows.bbox.isna()].reset_index(drop=True)


def _detect(images_dir: Path, image_ids: list[str], detect_fn: DetectFn, detector: str):
    for image_id in image_ids:
        with Image.open(images_dir / f"{image_id}.jpg") as raw:
            image = raw.convert("RGB")  # silver images are already EXIF-oriented
        boxes, scores = detect_fn(image)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        keep = scores >= CACHE_FLOOR
        yield (
            image_id,
            detector,
            json.dumps(boxes[keep].round(2).tolist()),
            json.dumps(scores[keep].round(4).tolist()),
        )


def cached_detections(
    image_ids: list[str],
    images_dir: Path,
    cache_path: Path,
    detect_factory: Callable[[], DetectFn],
    *,
    detector: str,
) -> pd.DataFrame:
    """Raw ``boxes, scores`` (JSON) per image, detecting only images not cached yet.

    ``detect_factory`` is only called when something is missing, so weights load lazily.
    """
    cache = (
        pd.read_parquet(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=list(CACHE_COLUMNS))
    )
    known = set(cache.image_id[cache.detector == detector])
    missing = [i for i in dict.fromkeys(image_ids) if i not in known]
    if missing:
        new = pd.DataFrame(
            _detect(images_dir, missing, detect_factory(), detector), columns=list(CACHE_COLUMNS)
        )
        cache = new if cache.empty else pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
    return cache[cache.detector == detector].set_index("image_id").loc[list(image_ids)]


def count_fruit(
    boxes: np.ndarray, scores: np.ndarray, *, min_score: float, iou: float
) -> tuple[np.ndarray, np.ndarray]:
    """Confident boxes after non-maximum suppression, best first."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    confident = scores >= min_score
    boxes, scores = boxes[confident], scores[confident]
    kept = non_max_suppression(boxes, scores, iou=iou)
    return boxes[kept], scores[kept]


def flag_multi_fruit(
    silver: pd.DataFrame,
    images_dir: Path,
    cache_path: Path,
    detect_factory: Callable[[], DetectFn],
    *,
    detector: str,
    min_score: float = 0.2,
    iou: float = 0.5,
) -> pd.DataFrame:
    """One row per whole-image silver row: how many berries the detector sees (``FLAG_COLUMNS``)."""
    rows = whole_image_rows(silver)
    detections = cached_detections(
        rows.image_id.tolist(), images_dir, cache_path, detect_factory, detector=detector
    )
    counted = []
    for image_id in rows.image_id:
        raw = detections.loc[image_id]
        boxes, scores = count_fruit(
            np.array(json.loads(raw.boxes)),
            np.array(json.loads(raw.scores)),
            min_score=min_score,
            iou=iou,
        )
        counted.append(
            (
                len(boxes),
                float(scores.max()) if len(scores) else float("nan"),
                json.dumps([[round(v, 1) for v in box] for box in boxes.tolist()]),
            )
        )
    n, best, boxes = zip(*counted, strict=True) if counted else ((), (), ())
    return pd.DataFrame(
        {
            "crop_id": rows.crop_id,
            "image_id": rows.image_id,
            "source_name": rows.source_name,
            "source_label": rows.source_label,
            "n_detected": pd.Series(n, dtype=int),
            "max_score": pd.Series(best, dtype=float),
            "boxes": pd.Series(boxes, dtype=object),
        }
    )[list(FLAG_COLUMNS)]


def summarize(flags: pd.DataFrame, *, min_count: int = 2) -> pd.DataFrame:
    """Per source and source label: whole-image rows, and how many show ``min_count``+ berries."""
    summary = (
        flags.assign(flagged=flags.n_detected >= min_count)
        .groupby(["source_name", "source_label"], as_index=False)
        .agg(images=("crop_id", "size"), flagged=("flagged", "sum"))
    )
    summary["flagged"] = summary.flagged.astype(int)
    summary["flagged_share"] = (summary.flagged / summary.images).round(3)
    return summary


def write_outputs(flags: pd.DataFrame, out_dir: Path, *, min_count: int = 2) -> dict[str, Path]:
    """``flags.csv``, ``summary.csv`` and a short ``report.md`` in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(flags, min_count=min_count)
    paths = {
        "flags": out_dir / "flags.csv",
        "summary": out_dir / "summary.csv",
        "report": out_dir / "report.md",
    }
    flags.to_csv(paths["flags"], index=False)
    summary.to_csv(paths["summary"], index=False)
    header = "| source | label | whole-image rows | flagged | share |\n|---|---|---|---|---|\n"
    lines = "".join(
        f"| {r.source_name} | {r.source_label} | {r.images} | {r.flagged} | {r.flagged_share} |\n"
        for r in summary.itertuples()
    )
    paths["report"].write_text(
        "# Multi-fruit check of whole-image rows\n\n"
        f"Flagged = the detector sees {min_count} or more berries. Flags are candidates, not "
        "labels: `uv run python scripts/make_label_task.py multi-fruit` sends them to a "
        "person.\n\n" + header + lines
    )
    return paths
