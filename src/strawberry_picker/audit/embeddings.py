"""Embed every silver fruit crop with a frozen backbone, for the statistical audit.

``embed_crops`` takes any ``embed_fn(list[Image]) -> (n, d) array``, so tests use a fake
and the audit uses ``dinov2_embedder``. Crops go through the same ``CropPolicy`` as gold
(ADR 0011): a row's ``neighbour_bboxes`` are masked when the policy says so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from strawberry_picker.data.crops import LEGACY_POLICY, CropPolicy, crop_fruit

EmbedFn = Callable[[list[Image.Image]], np.ndarray]


def audit_label(maturity: str | None, defects: Sequence[str] | None) -> str | None:
    """One label per crop: maturity if the source rates it, else its defects (or ``healthy``)."""
    if maturity is not None and maturity == maturity:  # NaN-safe
        return str(maturity)
    if defects is None:
        return None
    return "+".join(sorted(defects)) or "healthy"


def with_crop_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add ``crop_id`` = ``<image_id>_<n>``, numbering fruit within an image as gold does."""
    out = manifest.copy()
    out["crop_id"] = out.image_id + "_" + out.groupby("image_id").cumcount().astype(str)
    return out


def _bbox(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return [float(v) for v in value]


def embed_crops(
    manifest: pd.DataFrame,
    images_dir: Path,
    embed_fn: EmbedFn,
    *,
    batch_size: int = 64,
    padding: float = 0.15,
    policy: CropPolicy | None = None,
) -> np.ndarray:
    """One embedding row per manifest row, in manifest order.

    ``policy`` defaults to ``keep`` with ``padding``. ``drop`` crops like ``keep`` here:
    which rows to embed is the caller's choice.
    """
    policy = (policy or CropPolicy(padding=padding)).for_evaluation()
    neighbours = (
        manifest.neighbour_bboxes
        if "neighbour_bboxes" in manifest
        else pd.Series([None] * len(manifest), index=manifest.index)
    )
    vectors: list[np.ndarray] = []
    batch: list[Image.Image] = []
    for image_id, bbox, others in zip(manifest.image_id, manifest.bbox, neighbours, strict=True):
        with Image.open(images_dir / f"{image_id}.jpg") as raw:
            image = raw.convert("RGB")
        batch.append(crop_fruit(image, _bbox(bbox), others, policy))
        if len(batch) == batch_size:
            vectors.append(embed_fn(batch))
            batch = []
    if batch:
        vectors.append(embed_fn(batch))
    return np.concatenate(vectors)


def dinov2_embedder(model_name: str = "facebook/dinov2-small") -> EmbedFn:  # pragma: no cover
    """CLS embedding from a frozen DINOv2 (Apache-2.0). Needs the ``ml`` extra and weights."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(crops: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=crops, return_tensors="pt").to(device)
        return model(**inputs).pooler_output.float().cpu().numpy()

    return embed


def cached_embeddings(
    frame: pd.DataFrame,
    images_dir: Path,
    cache_path: Path,
    embed_factory: Callable[[], EmbedFn],
    *,
    model: str,
    batch_size: int = 64,
    policy: CropPolicy | None = None,
) -> np.ndarray:
    """Embeddings for ``frame`` rows (by ``crop_id``), embedding only crops not yet cached.

    The cache (``crop_id, model, crop, embedding``) keeps every model's and every crop
    policy's vectors side by side, so a masked release never reuses unmasked vectors.
    Without ``policy``, a gold manifest's ``crop_policy`` column decides; otherwise the
    legacy ``keep`` crop. Caches written before ADR 0011 hold legacy crops.
    ``embed_factory`` is only called when something is missing, so loading weights is lazy.
    """
    policy = (policy or _frame_policy(frame)).for_evaluation()
    key = policy.to_json()
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cache.empty and "crop" not in cache:
        cache["crop"] = LEGACY_POLICY.to_json()
    mine = (cache.model == model) & (cache.crop == key) if not cache.empty else None
    known = set(cache.crop_id[mine]) if mine is not None else set()
    missing = frame[~frame.crop_id.isin(known)]
    if len(missing):
        vectors = embed_crops(
            missing, images_dir, embed_factory(), batch_size=batch_size, policy=policy
        )
        new = pd.DataFrame(
            {"crop_id": missing.crop_id, "model": model, "crop": key, "embedding": list(vectors)}
        )
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
    by_crop = cache[(cache.model == model) & (cache.crop == key)].set_index("crop_id").embedding
    return np.stack(by_crop.loc[frame.crop_id].to_numpy())


def _frame_policy(frame: pd.DataFrame) -> CropPolicy:
    if "crop_policy" not in frame or frame.crop_policy.isna().all():
        return LEGACY_POLICY
    policies = set(frame.crop_policy.dropna())
    if len(policies) > 1:
        raise ValueError(f"rows were cut with {len(policies)} different crop policies")
    return CropPolicy.from_json(policies.pop())
