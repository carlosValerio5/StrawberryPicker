"""Gold layer: versioned, group-aware train/val/test datasets. Contract: ``data/gold/README.md``."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from strawberry_picker.paths import Layer, layer_dir
from strawberry_picker.schema import GOLD_COLUMNS, Defect, Grade, Maturity, grade_from_defects

_VERSION_NAME = re.compile(r"^[a-z]+_(cls|det|seg)_v\d+\.\d+$")

SPLITS = ("train", "val", "test")


class GoldTask(StrEnum):
    MATURITY_CLS = "maturity_cls"  # single label → ImageFolder: <split>/<maturity>/<crop>.jpg
    QUALITY_CLS = "quality_cls"  # multi-label → <split>/images/<crop>.jpg, labels in manifest


def dataset_dir(name: str) -> Path:
    """``data/gold/<task>_v<MAJOR>.<MINOR>/``, e.g. ``maturity_cls_v1.0``."""
    if not _VERSION_NAME.match(name):
        raise ValueError(f"gold dataset name must look like 'maturity_cls_v1.0', got {name!r}")
    return layer_dir(Layer.GOLD) / name


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


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    unknown = set(ratios) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split(s) {sorted(unknown)}; use {SPLITS}")
    if any(r < 0 for r in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"split ratios must be non-negative and sum to 1, got {dict(ratios)}")


def assign_splits(
    group_ids: Iterable[str], ratios: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Assign every group to a split by hashing ``(seed, group)``.

    Each group's split depends only on its own id, so adding data in a later version
    never moves an existing group to a different split.
    """
    _validate_ratios(ratios)
    bounds, total = [], 0.0
    for split in SPLITS:
        total += ratios.get(split, 0.0)
        bounds.append((total, split))

    def pick(group: str) -> str:
        u = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:15], 16) / 16**15
        return next(split for bound, split in bounds if u < bound)

    return {group: pick(group) for group in set(group_ids)}


def _select(manifest: pd.DataFrame, task: GoldTask, commercial_only: bool) -> pd.DataFrame:
    label = "maturity" if task is GoldTask.MATURITY_CLS else "defects"
    keep = manifest[label].notna() & ~manifest["needs_review"].astype(bool)
    if commercial_only:
        keep &= manifest["commercial_use"].astype(bool)
    return manifest[keep].copy()


def _stats(rows: pd.DataFrame, task: GoldTask) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        part = rows[rows["split"] == split]
        entry: dict[str, Any] = {"total": len(part)}
        if task is GoldTask.MATURITY_CLS:
            counts = part["maturity"].value_counts()
            entry["maturity"] = {m.value: int(counts.get(m.value, 0)) for m in Maturity}
        else:
            defect_lists = [list(d) for d in part["defects"]]
            grades = pd.Series([grade_from_defects(d).value for d in defect_lists]).value_counts()
            entry["defects"] = {d.value: sum(d.value in ds for ds in defect_lists) for d in Defect}
            entry["grade"] = {g.value: int(grades.get(g.value, 0)) for g in Grade}
        stats[split] = entry
    return stats


def _dataset_card(
    name: str, rows: pd.DataFrame, stats: dict[str, Any], settings: dict[str, Any], now: datetime
) -> str:
    sources = (
        rows.groupby(["source_name", "license", "commercial_use"]).size().reset_index(name="rows")
    )
    lines = [
        f"# {name}",
        "",
        f"Built {now.date().isoformat()} from `data/silver/manifest.parquet`.",
        "",
        "## Build settings",
        "",
        *(f"- **{key}**: `{value}`" for key, value in settings.items()),
        "",
        "## Sources",
        "",
        "| Source | License | Commercial use | Rows |",
        "|--------|---------|----------------|------|",
        *(
            f"| {s.source_name} | {s.license} | {'yes' if s.commercial_use else 'no'} | {s.rows} |"
            for s in sources.itertuples()
        ),
        "",
        "## Splits",
        "",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
        "## Known issues",
        "",
        "_Fill in before release: label noise, domain gaps, classes that are under-represented._",
    ]
    return "\n".join(lines) + "\n"


def build_gold(
    task: GoldTask | str,
    version: str,
    *,
    ratios: Mapping[str, float],
    seed: int,
    padding: float = 0.15,
    commercial_only: bool = False,
    jpeg_quality: int = 95,
    now: datetime | None = None,
) -> Path:
    """Release ``data/gold/<task>_v<version>`` from the current silver manifest.

    Rows that need human review, or lack the task's label, are left out. Raises
    ``FileExistsError`` if that version was already released — versions are immutable.
    """
    task = GoldTask(task)
    out = dataset_dir(f"{task.value}_v{version}")
    _validate_ratios(ratios)
    if out.exists():
        raise FileExistsError(f"{out} already released — cut a new version instead")

    silver = layer_dir(Layer.SILVER)
    manifest = pd.read_parquet(silver / "manifest.parquet")
    # Index fruits within each image before filtering, so crop ids match across tasks.
    manifest["crop_id"] = (
        manifest["image_id"] + "_" + manifest.groupby("image_id").cumcount().astype(str)
    )
    rows = _select(manifest, task, commercial_only)
    if rows.empty:
        raise ValueError(f"no rows in the silver manifest are usable for {task.value}")

    splits = assign_splits(rows["group_id"], ratios, seed)
    rows["split"] = rows["group_id"].map(splits)
    label_dir = rows["maturity"] if task is GoldTask.MATURITY_CLS else "images"
    rows["crop_path"] = rows["split"] + "/" + label_dir + "/" + rows["crop_id"] + ".jpg"

    partial = out.with_name(f".{out.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        for image_id, fruits in rows.groupby("image_id", sort=False):
            with Image.open(silver / "images" / f"{image_id}.jpg") as image:
                image = image.convert("RGB")
            for fruit in fruits.itertuples():
                bbox = None if fruit.bbox is None else tuple(fruit.bbox)
                target = partial / fruit.crop_path
                target.parent.mkdir(parents=True, exist_ok=True)
                image.crop(crop_box(bbox, image.width, image.height, padding)).save(
                    target, quality=jpeg_quality
                )

        rows = rows[list(GOLD_COLUMNS)].reset_index(drop=True)
        rows.to_parquet(partial / "manifest.parquet", index=False)
        stats = _stats(rows, task)
        (partial / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        settings = {
            "ratios": dict(ratios),
            "seed": seed,
            "padding": padding,
            "commercial_only": commercial_only,
        }
        card = _dataset_card(out.name, rows, stats, settings, now or datetime.now(UTC))
        (partial / "DATASET_CARD.md").write_text(card)
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return out
