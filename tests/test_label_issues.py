from __future__ import annotations

import numpy as np
import pytest

from strawberry_picker.audit.label_issues import find_label_issues, out_of_fold_probabilities


def test_flags_confident_disagreements_and_ranks_by_self_confidence() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],  # labelled 0, model is sure it's 1
            [0.2, 0.8],
            [0.3, 0.7],
            [0.95, 0.05],  # labelled 1, model is sure it's 0
        ]
    )

    issues = find_label_issues(labels, probs)

    assert issues.loc[issues.is_issue].index.tolist() == [5, 2]  # lowest self-confidence first
    assert issues.loc[5, "suggested"] == 0
    assert issues.loc[2, "suggested"] == 1
    assert issues.loc[2, "self_confidence"] == pytest.approx(0.1)


def test_uses_per_class_thresholds_not_argmax() -> None:
    # Class 1 is rare and the model is never very sure about it. Its threshold is the mean
    # self-confidence of class-1 rows (0.4), so 0.45 on a class-0 row is a confident
    # disagreement although argmax would still say 0.
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.55, 0.45], [0.6, 0.4], [0.6, 0.4]])

    issues = find_label_issues(labels, probs)

    assert issues.sort_index().is_issue.tolist() == [False, False, True, False, False]


def test_rows_with_no_confident_class_are_not_issues() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.75, 0.25]])
    # thresholds: class 0 = 0.8, class 1 = (0.8 + 0.25) / 2 = 0.525.
    # Row 3 (labelled 1): p0 = 0.75 < 0.8 and p1 = 0.25 < 0.525, so nothing is confident.

    assert not find_label_issues(labels, probs).loc[3, "is_issue"]


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="rows"):
        find_label_issues(np.array([0, 1]), np.array([[1.0, 0.0]]))


def test_out_of_fold_probabilities_keep_groups_apart() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 3)
    labels = groups % 2
    features = labels[:, None] * 4.0 + rng.normal(size=(60, 5))

    probs = out_of_fold_probabilities(features, labels, groups, n_splits=4, seed=0)

    assert probs.shape == (60, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs.argmax(axis=1) == labels).mean() > 0.9
