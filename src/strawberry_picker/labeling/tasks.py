"""Build labeling tasks (``TASK_COLUMNS``) from silver rows, keyed by gold's crop ids."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strawberry_picker.audit.embeddings import _bbox, with_crop_ids
from strawberry_picker.labeling.tool import TASK_COLUMNS


def _given(maturity: object, defects: object) -> str:
    text = maturity if isinstance(maturity, str) else ""
    found = "+".join(defects) if defects is not None and len(defects) else ""
    return f"{text} +{found}" if text and found else text or found


def crop_task(silver: pd.DataFrame, bronze_dir: Path) -> pd.DataFrame:
    """One task row per silver fruit: its bronze image, box, and the label to check."""
    rows = with_crop_ids(silver)
    boxes = [_bbox(b) for b in rows.bbox]
    return pd.DataFrame(
        {
            "item_id": rows.crop_id,
            "image_path": [str(bronze_dir / p) for p in rows.bronze_path],
            "bbox": [json.dumps(b) if b else "" for b in boxes],
            "given": [_given(m, d) for m, d in zip(rows.maturity, rows.defects, strict=True)],
            "group": rows.image_id,
        }
    )[list(TASK_COLUMNS)]


def multi_fruit_task(
    silver: pd.DataFrame, flags: pd.DataFrame, bronze_dir: Path, *, min_count: int = 2
) -> pd.DataFrame:
    """Whole-image rows the detector counts ``min_count``+ berries in, not yet reviewed.

    ``flags`` comes from ``audit.multi_fruit.flag_multi_fruit``. The reviewer sees the whole
    image and presses ``s`` (several berries) or labels the one berry it shows.
    """
    rows = with_crop_ids(silver)
    counts = flags.set_index("crop_id").n_detected
    flagged = rows.crop_id.map(counts).fillna(0) >= min_count
    rows = rows[flagged & rows.reviewed_by.isna()]
    task = crop_task(rows.drop(columns="crop_id"), bronze_dir)
    n = task.item_id.map(counts).astype(int)
    task["given"] = [
        f"{given} (detector: {k} berries)".lstrip() for given, k in zip(task.given, n, strict=True)
    ]
    return task
