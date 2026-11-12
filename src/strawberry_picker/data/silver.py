"""Silver layer: decode, dedupe and unify bronze sources. Contract: ``data/silver/README.md``."""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from strawberry_picker.data.bronze import all_pulls, latest_pull
from strawberry_picker.data.dedup import dihedral_hashes, find_near_duplicates
from strawberry_picker.data.readers import Annotation, read_annotations
from strawberry_picker.data.reviews import apply_reviews, load_reviews, resolve_reviews
from strawberry_picker.data.sources import GroupBy, SourceSpec
from strawberry_picker.paths import Layer, layer_dir
from strawberry_picker.schema import SILVER_COLUMNS

logger = logging.getLogger(__name__)

REJECTED_COLUMNS = ("source_name", "bronze_path", "reason", "detail")


@dataclass(frozen=True, slots=True)
class SilverBuild:
    manifest: pd.DataFrame
    rejected: pd.DataFrame


def compute_image_id(image: Image.Image) -> str:
    """Content-addressed id: sha256 over size + EXIF-corrected RGB pixels, first 16 hex chars.

    Identical photos from different sources (or re-encoded copies with the same
    pixels) collapse to one id, which is what deduplication keys on.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()[:16]


def _group_id(spec: SourceSpec, image_id: str, relative_parent: str) -> str:
    match spec.group_by:
        case GroupBy.SOURCE:
            return spec.name
        case GroupBy.PARENT_DIR:
            return f"{spec.name}/{relative_parent}"
    return f"{spec.name}/{image_id}"


def _apply_box_proposals(
    by_image: dict[Path, list[Annotation]], spec: SourceSpec, bronze: Path
) -> tuple[dict[Path, list[Annotation]], set[str]]:
    """Tight boxes for single-berry images; bronze paths of piles and empty images."""
    # Newest pull that holds this source's file: a later pull of the proposer (e.g. for a
    # new source) must not hide the proposals an older pull still has for this one.
    path = next(
        (
            p / f"{spec.name}.csv"
            for p in reversed(all_pulls(spec.box_proposals.source))
            if (p / f"{spec.name}.csv").is_file()
        ),
        None,
    )
    if path is None:
        logger.warning("%s: no box proposals found; whole images are used", spec.name)
        return by_image, set()
    proposals = pd.read_csv(path)
    confident = proposals[proposals.score >= spec.box_proposals.min_score]
    proposed = set(proposals.bronze_path)
    boxes = {
        name: [tuple(map(float, b)) for b in group[["x0", "y0", "x1", "y1"]].to_numpy()]
        for name, group in confident.groupby("bronze_path")
    }
    held: set[str] = set()
    out: dict[Path, list[Annotation]] = {}
    for image_path, found in by_image.items():
        name = image_path.relative_to(bronze).as_posix()
        if name not in proposed or any(a.bbox is not None for a in found):
            out[image_path] = found
        elif len(boxes.get(name, ())) == 1:
            out[image_path] = [replace(a, bbox=boxes[name][0]) for a in found]
        else:
            out[image_path] = found
            held.add(name)
    return out, held


def build_silver(
    sources: Mapping[str, SourceSpec],
    *,
    min_side_px: int = 32,
    jpeg_quality: int = 95,
    near_duplicate_max_distance: int = 24,
    now: datetime | None = None,
) -> SilverBuild:
    """Rebuild ``data/silver`` from the latest bronze pull of every enabled source.

    Writes ``images/<image_id>.jpg``, ``manifest.parquet`` and ``rejected.parquet``.
    Bronze is only read. A label without a mapping raises ``ValueError`` — fix
    ``configs/sources.yaml`` rather than guessing.
    """
    now = now or datetime.now(UTC)
    bronze = layer_dir(Layer.BRONZE)
    silver = layer_dir(Layer.SILVER)
    images_dir = silver / "images"
    shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    held: set[str] = set()  # bronze paths whose one label can't be pinned to one berry
    rejected: list[tuple[str, str, str, str]] = []  # REJECTED_COLUMNS
    hashes: dict[str, tuple[int, ...]] = {}  # bronze_path → dihedral hashes
    stored: dict[str, str] = {}  # bronze_path → image_id
    first_seen: dict[str, str] = {}  # image_id → bronze_path

    for spec in sorted(sources.values(), key=lambda s: s.name):
        if not spec.enabled:
            continue
        pull = latest_pull(spec.name)
        if pull is None:
            logger.warning("source %s has no bronze pull — skipped", spec.name)
            continue

        by_image: dict[Path, list[Annotation]] = {}
        for annotation in read_annotations(spec, pull):
            by_image.setdefault(annotation.image_path, []).append(annotation)
        if spec.box_proposals is not None:
            by_image, spec_held = _apply_box_proposals(by_image, spec, bronze)
            held |= spec_held

        for image_path in sorted(by_image):
            bronze_path = image_path.relative_to(bronze).as_posix()

            if any(fnmatch(image_path.relative_to(pull).as_posix(), p) for p in spec.exclude):
                rejected.append((spec.name, bronze_path, "excluded", ""))
                continue

            kept = []
            for annotation in by_image[image_path]:
                mapping = spec.map_label(annotation.source_label)
                if mapping is None:
                    rejected.append(
                        (spec.name, bronze_path, "label_dropped", annotation.source_label)
                    )
                else:
                    kept.append((annotation, mapping))
            if not kept:
                continue

            try:
                with Image.open(image_path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                rejected.append((spec.name, bronze_path, "corrupt", type(exc).__name__))
                continue
            if min(image.size) < min_side_px:
                size = f"{image.width}x{image.height}"
                rejected.append((spec.name, bronze_path, "too_small", size))
                continue

            image_id = compute_image_id(image)
            if image_id in first_seen:
                rejected.append((spec.name, bronze_path, "duplicate", first_seen[image_id]))
                continue
            first_seen[image_id] = bronze_path
            stored[bronze_path] = image_id
            hashes[bronze_path] = dihedral_hashes(image)
            image.save(images_dir / f"{image_id}.jpg", quality=jpeg_quality)

            parent = image_path.parent.relative_to(pull).as_posix()
            for annotation, mapping in kept:
                rows.append(
                    {
                        "image_id": image_id,
                        "source_name": spec.name,
                        "bronze_path": bronze_path,
                        "source_label": annotation.source_label,
                        "license": spec.license,
                        "commercial_use": spec.commercial_use,
                        "width": image.width,
                        "height": image.height,
                        "maturity": mapping.maturity and mapping.maturity.value,
                        "defects": None
                        if mapping.defects is None
                        else [d.value for d in mapping.defects],
                        "needs_review": mapping.needs_review,
                        "bbox": None if annotation.bbox is None else list(annotation.bbox),
                        "group_id": _group_id(spec, image_id, parent),
                        "ingested_at": now,
                        "reviewed_by": None,
                    }
                )

    near_duplicates = find_near_duplicates(hashes, max_distance=near_duplicate_max_distance)
    dropped_ids = {stored[path] for path in near_duplicates}
    for path, canonical in near_duplicates.items():
        (images_dir / f"{stored[path]}.jpg").unlink()
        rejected.append((path.split("/", 1)[0], path, "near_duplicate", canonical))

    manifest = pd.DataFrame(
        [r for r in rows if r["image_id"] not in dropped_ids], columns=list(SILVER_COLUMNS)
    )
    manifest = apply_reviews(manifest, resolve_reviews(load_reviews()))
    if held:  # after reviews: a review can't turn a pile into a single berry
        manifest.loc[manifest.bronze_path.isin(held), "needs_review"] = True
        logger.info("%d images held out of gold by box proposals", len(held))
    rejected_df = (
        pd.DataFrame(rejected, columns=list(REJECTED_COLUMNS))
        .sort_values(["bronze_path", "reason", "detail"])
        .reset_index(drop=True)
    )
    manifest.to_parquet(silver / "manifest.parquet", index=False)
    rejected_df.to_parquet(silver / "rejected.parquet", index=False)
    return SilverBuild(manifest=manifest, rejected=rejected_df)
