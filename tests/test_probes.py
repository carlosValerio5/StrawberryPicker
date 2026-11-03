from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from strawberry_picker.audit.probes import (  # noqa: E402 (after importorskip)
    learning_curve,
    leave_one_source_out,
    source_probe,
)


def _two_sources(label_flip: bool = False, seed: int = 0):
    """200 rows, 2 sources, 2 labels; the label lives in feature 0, the source in feature 1."""
    rng = np.random.default_rng(seed)
    sources = np.repeat(["a", "b"], 100)
    labels = np.tile([0, 1], 100)
    signal = np.where(labels == 1, 3.0, -3.0)
    if label_flip:  # source b encodes the label the other way round
        signal = np.where(sources == "b", -signal, signal)
    features = rng.normal(size=(200, 4))
    features[:, 0] += signal
    features[:, 1] += np.where(sources == "a", 4.0, -4.0)
    groups = np.arange(200) // 2
    return features, labels, sources, groups


def test_source_probe_detects_separable_sources() -> None:
    features, _, sources, groups = _two_sources()

    probe = source_probe(features, sources, groups, n_splits=5, seed=0)

    assert probe["balanced_accuracy"] > 0.95
    assert probe["chance"] == 0.5
    assert probe["n_sources"] == 2


def test_source_probe_is_near_chance_when_sources_look_alike() -> None:
    features, _, sources, groups = _two_sources()
    features[:, 1] = np.random.default_rng(1).normal(size=200)

    assert source_probe(features, sources, groups, n_splits=5, seed=0)["balanced_accuracy"] < 0.65


def test_leave_one_source_out_exposes_source_specific_labels() -> None:
    consistent = leave_one_source_out(*_two_sources(), seed=0).set_index("held_out")
    flipped = leave_one_source_out(*_two_sources(label_flip=True), seed=0).set_index("held_out")

    assert list(consistent.index) == ["a", "b", "in_distribution"]
    assert consistent.loc["a", "macro_f1"] > 0.9
    assert flipped.loc["a", "macro_f1"] < 0.2  # trained on b's convention, tested on a's
    assert consistent.loc["b", "n_test"] == 100


def test_learning_curve_grows_training_set() -> None:
    features, labels, _, groups = _two_sources()

    curve = learning_curve(features, labels, groups, fractions=(0.1, 0.5, 1.0), seed=0)

    assert curve.fraction.tolist() == [0.1, 0.5, 1.0]
    assert curve.n_train.is_monotonic_increasing
    assert curve.n_test.nunique() == 1  # one fixed held-out set
    assert curve.macro_f1.iloc[-1] > 0.9
