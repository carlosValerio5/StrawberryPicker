from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from strawberry_picker.audit.embeddings import cached_embeddings
from strawberry_picker.evaluation.maturity import evaluate_maturity

CLASSES = ("unripe", "ripe", "overripe")


def test_cached_embeddings_only_embeds_new_crops(tmp_path) -> None:
    Image.new("RGB", (20, 20), (200, 0, 0)).save(tmp_path / "a.jpg")
    Image.new("RGB", (20, 20), (0, 0, 200)).save(tmp_path / "b.jpg")
    calls: list[int] = []

    def factory():
        def embed(crops):
            calls.append(len(crops))
            return np.stack([np.asarray(c, dtype=np.float32).mean(axis=(0, 1)) for c in crops])

        return embed

    first = pd.DataFrame({"crop_id": ["a_0"], "image_id": ["a"], "bbox": [None]})
    both = pd.DataFrame({"crop_id": ["b_0", "a_0"], "image_id": ["b", "a"], "bbox": [None, None]})
    cache = tmp_path / "cache.parquet"

    cached_embeddings(first, tmp_path, cache, factory, model="toy")
    vectors = cached_embeddings(both, tmp_path, cache, factory, model="toy")

    assert calls == [1, 1]  # a_0 was never embedded twice
    assert vectors.shape == (2, 3) and vectors[0, 2] > 150 and vectors[1, 0] > 150  # input order
    cached_embeddings(both, tmp_path, cache, factory, model="other")
    assert calls == [1, 1, 2]  # a different model is a different cache


def test_evaluate_maturity_reports_adr_0005_numbers() -> None:
    truth = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    probs = np.eye(3)[[0, 0, 1, 1, 0, 0, 1, 1]] * 0.9 + 0.1 / 3
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])

    report = evaluate_maturity(probs, truth, groups, CLASSES, n_resamples=200)

    assert report["n"] == 8 and report["n_groups"] == 4
    assert report["recall"]["unripe"]["point"] == pytest.approx(0.75)
    assert report["recall"]["ripe"]["point"] == pytest.approx(0.75)
    assert report["recall"]["overripe"]["point"] is None  # no overripe in the truth
    qwk = report["qwk"]
    assert qwk["low"] <= qwk["point"] <= qwk["high"]
    assert 0 <= report["ece"] <= 1
    assert report["confusion"] == [[3, 1, 0], [1, 3, 0], [0, 0, 0]]
    json.dumps(report)  # must be serialisable to metrics.json


def test_linear_probe_run_round_trip(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.linear_probe import load_probe, save_run, train_probe

    rng = np.random.default_rng(0)
    labels = np.tile([0, 1], 50)
    features = rng.normal(size=(100, 4)) + labels[:, None] * 3

    probe = train_probe(features, labels, seed=0)
    run = save_run(tmp_path / "run1", probe, {"task": "maturity_cls"}, {"ok": 1})

    assert yaml.safe_load((run / "config.yaml").read_text()) == {"task": "maturity_cls"}
    assert json.loads((run / "metrics.json").read_text()) == {"ok": 1}
    reloaded = load_probe(run)
    np.testing.assert_allclose(reloaded.predict_proba(features), probe.predict_proba(features))
    with pytest.raises(FileExistsError):
        save_run(tmp_path / "run1", probe, {}, {})
