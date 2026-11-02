"""Embed every silver fruit crop with a frozen backbone, for the statistical audit.

``embed_crops`` takes any ``embed_fn(list[Image]) -> (n, d) array``, so tests use a fake
and the audit uses ``dinov2_embedder``. Crops use the same padded square as gold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from strawberry_picker.data.gold import crop_box

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
) -> np.ndarray:
    """One embedding row per manifest row, in manifest order."""
    vectors: list[np.ndarray] = []
    batch: list[Image.Image] = []
    for image_id, bbox in zip(manifest.image_id, manifest.bbox, strict=True):
        with Image.open(images_dir / f"{image_id}.jpg") as raw:
            image = raw.convert("RGB")
        batch.append(image.crop(crop_box(_bbox(bbox), *image.size, padding=padding)))
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
