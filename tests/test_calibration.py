"""Temperature scaling for the maturity probe (ADR 0005 §B3: ECE <= 0.10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strawberry_picker.evaluation.metrics import expected_calibration_error

CLASSES = ("unripe", "ripe", "overripe")


def _overconfident(n: int = 600, accuracy: float = 0.7, seed: int = 0):
    """Two-class probabilities that claim ~0.98 but are right only ``accuracy`` of the time."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    correct = rng.random(n) < accuracy
    predicted = np.where(correct, truth, 1 - truth)
    confidence = rng.uniform(0.96, 0.995, size=n)
    probs = np.zeros((n, 3))
    probs[np.arange(n), predicted] = confidence
    probs[np.arange(n), 1 - predicted] = 1 - confidence
    return probs, truth


def test_fit_temperature_softens_an_overconfident_probe() -> None:
    from strawberry_picker.models.calibration import apply_temperature, fit_temperature

    probs, truth = _overconfident()
    temperature = fit_temperature(probs, truth)
    calibrated = apply_temperature(probs, temperature)

    assert temperature > 1  # overconfident → soften
    before = expected_calibration_error(probs, truth)
    after = expected_calibration_error(calibrated, truth)
    assert before > 0.2 and after < 0.1 and after < before


def test_fit_temperature_sharpens_an_underconfident_probe() -> None:
    from strawberry_picker.models.calibration import fit_temperature

    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, size=400)
    probs = np.zeros((400, 3))
    probs[np.arange(400), truth] = 0.6  # always right, but only 60 % sure
    probs[np.arange(400), 1 - truth] = 0.4

    assert fit_temperature(probs, truth) < 1


def test_temperature_scaling_keeps_predictions_and_untrained_classes() -> None:
    from strawberry_picker.models.calibration import apply_temperature

    probs, _ = _overconfident(n=50)
    for temperature in (0.5, 1.0, 3.0):
        scaled = apply_temperature(probs, temperature)
        np.testing.assert_array_equal(scaled.argmax(axis=1), probs.argmax(axis=1))
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)
        assert (scaled[:, 2] == 0).all()  # never-trained overripe column stays exactly 0
    np.testing.assert_allclose(apply_temperature(probs, 1.0), probs)
    with pytest.raises(ValueError):
        apply_temperature(probs, 0.0)


def _probe(seed: int = 0):
    from strawberry_picker.models.linear_probe import train_probe

    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 50)
    features = rng.normal(size=(100, 4)) + labels[:, None] * 3
    return train_probe(features, labels, seed=0), features


def test_calibrated_run_round_trips_its_temperature(tmp_path) -> None:
    pytest.importorskip("sklearn")
    import joblib

    from strawberry_picker.models.calibration import TemperatureScaled, apply_temperature
    from strawberry_picker.models.linear_probe import load_probe, save_run

    base, features = _probe()
    run = save_run(tmp_path / "run", TemperatureScaled(base, 2.5), {}, {})

    assert json.loads((run / "calibration.json").read_text()) == {
        "method": "temperature",
        "temperature": 2.5,
    }
    assert not isinstance(joblib.load(run / "probe.joblib"), TemperatureScaled)  # plain sklearn
    reloaded = load_probe(run)
    assert isinstance(reloaded, TemperatureScaled) and reloaded.temperature == 2.5
    np.testing.assert_array_equal(reloaded.classes_, base.classes_)
    np.testing.assert_allclose(
        reloaded.predict_proba(features), apply_temperature(base.predict_proba(features), 2.5)
    )


def test_run_without_calibration_file_loads_as_temperature_one(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.linear_probe import load_probe, save_run

    base, features = _probe()
    run = save_run(tmp_path / "old", base, {}, {})  # what runs before calibration look like

    assert not (run / "calibration.json").exists()
    np.testing.assert_allclose(
        load_probe(run).predict_proba(features), base.predict_proba(features)
    )


def _rows_and_embeddings(seed: int = 0):
    """Source ``a`` (train/val/test) plus a held-out ``proxy`` that also carries val rows."""
    rng = np.random.default_rng(seed)
    n = 200
    rows = pd.DataFrame(
        {
            "source_name": ["a"] * 160 + ["proxy"] * 40,
            "split": ["train"] * 100 + ["val"] * 30 + ["test"] * 30 + ["val"] * 40,
            "maturity": np.tile(["unripe", "ripe"], n // 2),
            "group_id": [f"g{i // 2}" for i in range(n)],
        }
    )
    signal = np.where(rows.maturity == "ripe", 1.0, -1.0)
    embeddings = rng.normal(size=(n, 6))
    embeddings[:, 0] += signal
    embeddings[100:160, 1:] *= 6  # noisier off the training split: the probe is overconfident
    return rows, embeddings


def test_run_baseline_fits_the_temperature_on_val_only() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.calibration import TemperatureScaled, fit_temperature
    from strawberry_picker.models.linear_probe import predict_all_classes, run_baseline

    rows, embeddings = _rows_and_embeddings()
    probe, metrics = run_baseline(
        rows, embeddings, CLASSES, holdout_sources=["proxy"], seed=0, n_resamples=50
    )

    assert isinstance(probe, TemperatureScaled)
    calibration = metrics["calibration"]
    assert calibration["method"] == "temperature"
    assert calibration["n_val"] == 30  # the proxy's "val" rows are never used
    val = ((rows.split == "val") & (rows.source_name == "a")).to_numpy()
    codes = rows.maturity.map({"unripe": 0, "ripe": 1}).to_numpy()
    raw = predict_all_classes(probe.base, embeddings[val], len(CLASSES))
    assert calibration["temperature"] == pytest.approx(fit_temperature(raw, codes[val]))

    ece = calibration["ece"]
    assert set(ece) == {"in_distribution", "proxy"} and set(ece["proxy"]) == {"proxy"}
    for scores in (ece["in_distribution"], ece["proxy"]["proxy"]):
        assert set(scores) == {"before", "after"}
    assert metrics["in_distribution"]["ece"] == pytest.approx(ece["in_distribution"]["after"])
    json.dumps(metrics)


def test_run_baseline_calibration_none_keeps_the_raw_probe() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.calibration import TemperatureScaled
    from strawberry_picker.models.linear_probe import run_baseline

    rows, embeddings = _rows_and_embeddings()
    probe, metrics = run_baseline(
        rows,
        embeddings,
        CLASSES,
        holdout_sources=["proxy"],
        seed=0,
        n_resamples=50,
        calibration="none",
    )

    assert not isinstance(probe, TemperatureScaled)
    assert metrics["calibration"]["temperature"] == 1.0
    ece = metrics["calibration"]["ece"]["in_distribution"]
    assert ece["before"] == pytest.approx(ece["after"])
    with pytest.raises(ValueError, match="calibration"):
        run_baseline(rows, embeddings, CLASSES, holdout_sources=[], seed=0, calibration="isotonic")


def test_run_baseline_without_val_rows_falls_back_to_temperature_one() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.linear_probe import run_baseline

    rows, embeddings = _rows_and_embeddings()
    rows["split"] = rows.split.replace({"val": "train"})
    _, metrics = run_baseline(
        rows, embeddings, CLASSES, holdout_sources=["proxy"], seed=0, n_resamples=50
    )

    assert metrics["calibration"]["n_val"] == 0
    assert metrics["calibration"]["temperature"] == 1.0


class _SureProbe:
    """Always 95 % sure, and right on half of the crops."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        ripe = (features[:, 0] > 128).astype(float) * 0.9 + 0.05
        return np.column_stack([1 - ripe, ripe])


def test_phone_set_scores_the_calibrated_probabilities(tmp_path) -> None:
    from PIL import Image

    from strawberry_picker.evaluation.phone_set import evaluate_pull
    from strawberry_picker.labeling.tool import LabelStore, parse_label
    from strawberry_picker.models.calibration import TemperatureScaled

    pull = tmp_path / "pull"
    pull.mkdir()
    rows = []
    for p in range(6):
        Image.new("RGB", (40, 40), (230, 20, 20)).save(pull / f"P{p}.jpg")
        rows.append((f"P{p}_0", f"P{p}.jpg", "[2, 2, 38, 38]", "", f"P{p}"))
    pd.DataFrame(rows, columns=["item_id", "image_path", "bbox", "given", "group"]).to_csv(
        pull / "task.csv", index=False
    )
    store = LabelStore(pull / "labels_task.csv", "alice")
    for p, (item, *_) in enumerate(rows):
        store.record(parse_label({"item_id": item, "maturity": "ripe" if p % 2 else "unripe"}))

    def mean_colour(crops):
        return np.stack([np.asarray(c, dtype=float).mean(axis=(0, 1)) for c in crops])

    raw = evaluate_pull(pull, _SureProbe(), mean_colour, CLASSES, n_resamples=50)
    soft = evaluate_pull(
        pull, TemperatureScaled(_SureProbe(), 50.0), mean_colour, CLASSES, n_resamples=50
    )

    assert raw["metrics"]["ece"] == pytest.approx(0.45)  # 95 % sure, 50 % right
    assert soft["metrics"]["ece"] < 0.1  # same predictions, honest confidence
    assert soft["metrics"]["confusion"] == raw["metrics"]["confusion"]


def test_calibration_rejects_impossible_inputs() -> None:
    from strawberry_picker.models.calibration import TemperatureScaled, fit_temperature

    with pytest.raises(ValueError, match="zero rows"):
        fit_temperature(np.zeros((0, 3)), np.zeros(0))
    with pytest.raises(ValueError, match="> 0"):
        TemperatureScaled(_SureProbe(), -1.0)
    wrapped = TemperatureScaled(_SureProbe(), 2.0)
    assert wrapped.classes_.tolist() == [0, 1]
