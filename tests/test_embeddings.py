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


# --- crop policy (ADR 0008) -------------------------------------------------------------


def _red_and_blue(tmp_path) -> None:
    image = Image.new("RGB", (100, 50), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 50, 50))
    image.save(tmp_path / "img.jpg", quality=100)


def _mean_colour(crops: list[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(c, dtype=np.float32).mean(axis=(0, 1)) for c in crops])


def test_embed_crops_masks_neighbours_listed_on_the_row(tmp_path) -> None:
    from strawberry_picker.data.crops import CropPolicy

    _red_and_blue(tmp_path)
    manifest = pd.DataFrame(
        {
            "image_id": ["img"],
            "bbox": [[10.0, 10.0, 50.0, 40.0]],
            "neighbour_bboxes": [[[50.0, 0.0, 100.0, 50.0]]],
        }
    )
    kept = embed_crops(manifest, tmp_path, _mean_colour, padding=0.5)
    masked = embed_crops(
        manifest,
        tmp_path,
        _mean_colour,
        policy=CropPolicy(padding=0.5, neighbours="mask", fill=(0, 0, 0)),
    )
    assert kept[0, 2] > 50  # the blue neighbour leaks into the plain crop
    assert masked[0, 2] < 5  # and is gone once masked


def test_cached_embeddings_keeps_one_cache_per_crop_policy(tmp_path) -> None:
    from strawberry_picker.data.crops import CropPolicy

    from strawberry_picker.audit.embeddings import cached_embeddings

    _red_and_blue(tmp_path)
    calls: list[int] = []

    def factory():
        def embed(crops):
            calls.append(len(crops))
            return _mean_colour(crops)

        return embed

    frame = pd.DataFrame(
        {
            "crop_id": ["img_0"],
            "image_id": ["img"],
            "bbox": [[10.0, 10.0, 50.0, 40.0]],
            "neighbour_bboxes": [[[50.0, 0.0, 100.0, 50.0]]],
        }
    )
    cache = tmp_path / "cache.parquet"
    masked = CropPolicy(padding=0.5, neighbours="mask", fill=(0, 0, 0))

    plain = cached_embeddings(frame, tmp_path, cache, factory, model="toy")
    again = cached_embeddings(frame, tmp_path, cache, factory, model="toy", policy=masked)
    # A gold manifest carries its policy, so training honours it with no extra argument.
    gold_rows = frame.assign(crop_policy=masked.to_json())
    from_gold = cached_embeddings(gold_rows, tmp_path, cache, factory, model="toy")

    assert calls == [1, 1]
    assert plain[0, 2] > 5 and again[0, 2] < 5
    assert np.array_equal(again, from_gold)


def test_cached_embeddings_reads_caches_written_before_policies(tmp_path) -> None:
    from strawberry_picker.audit.embeddings import cached_embeddings

    _red_and_blue(tmp_path)
    cache = tmp_path / "cache.parquet"
    old = pd.DataFrame({"crop_id": ["img_0"], "model": ["toy"], "embedding": [np.ones(3)]})
    old.to_parquet(cache, index=False)
    frame = pd.DataFrame({"crop_id": ["img_0"], "image_id": ["img"], "bbox": [None]})

    def factory():
        raise AssertionError("legacy rows are the default policy; nothing to embed")

    vectors = cached_embeddings(frame, tmp_path, cache, factory, model="toy")
    assert np.array_equal(vectors, np.ones((1, 3)))
