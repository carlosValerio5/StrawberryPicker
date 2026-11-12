from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from strawberry_picker.inference.predict import (
    annotate_image,
    latest_maturity_run,
    load_maturity_run,
    predict_image,
    predictions_table,
)
from strawberry_picker.models.linear_probe import train_probe


def test_predict_image_whole_frame_without_detector() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    labels = np.tile([0, 1, 2], 20)
    features = rng.normal(size=(60, 4)) + labels[:, None] * 2
    probe = train_probe(features, labels, seed=0)
    classes = ("unripe", "ripe", "overripe")

    def embed(crops):
        return np.stack([np.full(4, float(np.asarray(c).mean()) / 255.0) for c in crops])

    image = Image.new("RGB", (40, 40), (220, 30, 30))
    preds = predict_image(image, probe, embed, classes)

    assert len(preds) == 1
    assert preds[0].bbox is None
    assert preds[0].label in classes
    assert 0 < preds[0].confidence <= 1
    assert set(preds[0].probs) == set(classes)
    annotated = annotate_image(image, preds)
    assert annotated.size == image.size
    assert predictions_table(preds)[0][1] == preds[0].label


def test_predict_image_uses_detector_boxes() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    labels = np.tile([0, 1], 30)
    features = rng.normal(size=(60, 3)) + labels[:, None] * 3
    probe = train_probe(features, labels, seed=1)
    classes = ("unripe", "ripe", "overripe")

    def embed(crops):
        return np.stack(
            [np.array([np.asarray(c).mean() / 255.0, 0.0, float(i)]) for i, c in enumerate(crops)]
        )

    def detect(_image):
        boxes = np.array([[5.0, 5.0, 35.0, 35.0], [50.0, 5.0, 90.0, 45.0]])
        scores = np.array([0.9, 0.8])
        return boxes, scores

    image = Image.new("RGB", (100, 60), (30, 160, 40))
    preds = predict_image(image, probe, embed, classes, detect_fn=detect, min_side_frac=0.1)

    assert len(preds) == 2
    assert preds[0].bbox == (5.0, 5.0, 35.0, 35.0)
    assert preds[0].detection_score == pytest.approx(0.9)
    assert preds[1].detection_score == pytest.approx(0.8)


def test_load_maturity_run_requires_a_probe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STRAWBERRY_MODELS_DIR", str(tmp_path))
    assert latest_maturity_run() is None
    with pytest.raises(FileNotFoundError, match="no model run"):
        load_maturity_run()
