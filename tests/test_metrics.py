from __future__ import annotations

import numpy as np
import pytest

from strawberry_picker.evaluation.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    krippendorff_alpha,
    per_class_precision,
    per_class_recall,
    quadratic_weighted_kappa,
)

NAN = np.nan
# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders x 12 units.
KRIPPENDORFF_2011 = np.array(
    [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, NAN, NAN, NAN],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NAN, 3],
        [NAN, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NAN],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NAN],
    ]
)


def test_krippendorff_alpha_matches_the_published_example() -> None:
    assert krippendorff_alpha(KRIPPENDORFF_2011, level="nominal") == pytest.approx(0.743, abs=1e-3)
    assert krippendorff_alpha(KRIPPENDORFF_2011, level="ordinal") == pytest.approx(0.815, abs=1e-3)


def test_krippendorff_alpha_is_one_for_perfect_agreement() -> None:
    ratings = np.array([[0, 1, 2, 2, 1], [0, 1, 2, 2, 1]], dtype=float)
    assert krippendorff_alpha(ratings, level="ordinal") == pytest.approx(1.0)


def test_ordinal_alpha_forgives_near_misses_more_than_nominal() -> None:
    ratings = np.array([[0, 0, 1, 1, 2, 2, 0, 2], [0, 1, 1, 2, 2, 1, 0, 2]], dtype=float)
    assert krippendorff_alpha(ratings, level="ordinal") > krippendorff_alpha(
        ratings, level="nominal"
    )


def test_krippendorff_alpha_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="level"):
        krippendorff_alpha(KRIPPENDORFF_2011, level="ratio")


def test_quadratic_weighted_kappa_matches_sklearn() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    truth = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0, 2])
    predicted = np.array([0, 1, 1, 2, 2, 2, 1, 1, 0, 0])
    expected = sklearn_metrics.cohen_kappa_score(truth, predicted, weights="quadratic")
    assert quadratic_weighted_kappa(truth, predicted, n_classes=3) == pytest.approx(expected)


def test_per_class_recall_reports_missing_classes_as_nan() -> None:
    recall = per_class_recall(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), n_classes=3)
    assert recall[0] == 0.5 and recall[1] == 1.0 and np.isnan(recall[2])


def test_per_class_precision_matches_sklearn_and_reports_unused_classes_as_nan() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    truth = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0, 2])
    predicted = np.array([0, 1, 1, 2, 2, 2, 1, 1, 0, 0])
    expected = sklearn_metrics.precision_score(truth, predicted, average=None, labels=[0, 1, 2])
    assert per_class_precision(truth, predicted, n_classes=3) == pytest.approx(expected)
    never_called = per_class_precision(np.array([0, 1, 2]), np.array([0, 1, 1]), n_classes=3)
    assert never_called[0] == 1.0 and never_called[1] == 0.5 and np.isnan(never_called[2])


def test_expected_calibration_error() -> None:
    confident_and_right = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(confident_and_right, np.array([0, 1])) == 0.0
    overconfident = np.array([[0.8, 0.2]] * 4)  # says 80 %, right half the time
    assert expected_calibration_error(overconfident, np.array([0, 1, 0, 1])) == pytest.approx(0.3)


def test_bootstrap_resamples_groups_and_is_reproducible() -> None:
    values = np.array([1.0, 1.0, 1.0, 0.0])
    groups = np.array(["photo_a", "photo_a", "photo_a", "photo_b"])

    def mean(index: np.ndarray) -> float:
        return float(values[index].mean())

    point, low, high = bootstrap_ci(mean, groups, n_resamples=500, seed=0)

    assert point == pytest.approx(0.75)
    assert low <= point <= high
    assert (low, high) == bootstrap_ci(mean, groups, n_resamples=500, seed=0)[1:]
    # Whole photos are drawn, so the only possible means are 0, 0.75 and 1.
    assert low in (0.0, 0.75) and high in (0.75, 1.0)
