"""The ADR 0005 §B maturity report: one JSON-ready dict per evaluation set.

Every interval resamples whole groups (photos, or images for public sources), so berries
that share a photo never count as independent evidence (ADR 0007).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from strawberry_picker.evaluation.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    per_class_recall,
    quadratic_weighted_kappa,
)


def _interval(values: tuple[float, float, float]) -> dict[str, float | None]:
    names = ("point", "low", "high")
    return {k: (None if np.isnan(v) else float(v)) for k, v in zip(names, values, strict=True)}


def _macro_f1(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> float:
    scores = []
    for k in np.unique(truth):
        tp = np.sum((predicted == k) & (truth == k))
        precision = tp / max(np.sum(predicted == k), 1)
        recall = tp / np.sum(truth == k)
        scores.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def evaluate_maturity(
    probs: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """QWK, per-class recall (with intervals), macro-F1, ECE and the confusion matrix."""
    n_classes = len(classes)
    predicted = probs.argmax(axis=1)

    def qwk(index: np.ndarray) -> float:
        return quadratic_weighted_kappa(truth[index], predicted[index], n_classes=n_classes)

    def recall_of(k: int):
        def recall(index: np.ndarray) -> float:
            return float(per_class_recall(truth[index], predicted[index], n_classes=n_classes)[k])

        return recall

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    ci = {"n_resamples": n_resamples, "seed": seed}
    return {
        "n": int(len(truth)),
        "n_groups": int(len(np.unique(groups))),
        "classes": list(classes),
        "qwk": _interval(bootstrap_ci(qwk, groups, **ci)),
        "recall": {
            name: _interval(bootstrap_ci(recall_of(k), groups, **ci))
            for k, name in enumerate(classes)
        },
        "macro_f1": _macro_f1(truth, predicted, n_classes),
        "ece": expected_calibration_error(probs, truth),
        "confusion": confusion.tolist(),
    }
