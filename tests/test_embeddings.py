from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from strawberry_picker.audit.embeddings import audit_label, embed_crops, with_crop_ids


def test_audit_label_prefers_maturity_then_defects() -> None:
    assert audit_label("ripe", None) == "ripe"
    assert audit_label("unripe", ["rot"]) == "unripe"
    assert audit_label(None, []) == "healthy"
    assert audit_label(None, ["mold"]) == "mold"
    assert audit_label(None, ["mold", "rot"]) == "mold+rot"
    assert audit_label(None, None) is None


def test_with_crop_ids_numbers_fruit_within_each_image() -> None:
    manifest = pd.DataFrame({"image_id": ["b", "a", "b"], "bbox": [None, None, None]})

    assert with_crop_ids(manifest).crop_id.tolist() == ["b_0", "a_0", "b_1"]


def test_embed_crops_crops_batches_and_keeps_order(tmp_path) -> None:
    image = Image.new("RGB", (100, 50), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 50, 50))
    image.save(tmp_path / "img.jpg", quality=100)
    manifest = pd.DataFrame(
        {
            "image_id": ["img"] * 3,
            "bbox": [[5.0, 5.0, 45.0, 45.0], [55.0, 5.0, 95.0, 45.0], [5.0, 5.0, 45.0, 45.0]],
        }
    )
    batches: list[int] = []

    def mean_colour(crops: list[Image.Image]) -> np.ndarray:
        batches.append(len(crops))
        return np.stack([np.asarray(c, dtype=np.float32).mean(axis=(0, 1)) for c in crops])

    vectors = embed_crops(manifest, tmp_path, mean_colour, batch_size=2, padding=0.0)

    assert batches == [2, 1]
    assert vectors.shape == (3, 3)
    assert vectors[0, 0] > 200 and vectors[0, 2] < 50  # left crop is red
    assert vectors[1, 2] > 200 and vectors[1, 0] < 50  # right crop is blue
