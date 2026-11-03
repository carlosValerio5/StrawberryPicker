"""Linear probes on frozen embeddings that answer data questions, not modelling ones.

- ``source_probe``: can a model tell which dataset a crop came from? If it can, a
  classifier can also learn "which dataset" as a shortcut for the label.
- ``leave_one_source_out``: does a probe trained on the other sources work on this one?
  A large drop means the sources disagree on what the label looks like.
- ``learning_curve``: does more data still help? Decides whether to collect or to clean.

All splits are group-aware, so near-identical crops never sit on both sides.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from strawberry_picker.audit.label_issues import _linear_probe


def _group_folds(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return folds.split(np.zeros(len(labels)), labels, groups)


def _macro_f1(truth: np.ndarray, predicted: np.ndarray, classes: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0))


def source_probe(
    features: np.ndarray,
    sources: Sequence[str],
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validated balanced accuracy of predicting the source from the embedding."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    names, codes = np.unique(np.asarray(sources), return_inverse=True)
    predicted = np.empty_like(codes)
    for train, test in _group_folds(codes, groups, n_splits, seed):
        predicted[test] = _linear_probe().fit(features[train], codes[train]).predict(features[test])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(codes, predicted)),
        "chance": 1 / len(names),
        "n_sources": len(names),
        "sources": names.tolist(),
        "confusion": confusion_matrix(codes, predicted).tolist(),
    }


def leave_one_source_out(
    features: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Macro-F1 on each source when trained on all the others, plus pooled group-CV."""
    sources = np.asarray(sources)
    classes = np.unique(labels)
    rows = []
    for held_out in np.unique(sources):
        test = sources == held_out
        model = _linear_probe().fit(features[~test], labels[~test])
        score = _macro_f1(labels[test], model.predict(features[test]), classes)
        rows.append((str(held_out), int((~test).sum()), int(test.sum()), score))

    predicted = np.empty_like(labels)
    for train, test in _group_folds(labels, groups, n_splits, seed):
        predicted[test] = (
            _linear_probe().fit(features[train], labels[train]).predict(features[test])
        )
    rows.append(
        ("in_distribution", len(labels), len(labels), _macro_f1(labels, predicted, classes))
    )
    return pd.DataFrame(rows, columns=["held_out", "n_train", "n_test", "macro_f1"])


def learning_curve(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Macro-F1 on one fixed held-out set of groups, training on growing shares of the rest."""
    from sklearn.model_selection import GroupShuffleSplit

    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(split.split(features, labels, groups))
    train_groups = np.random.default_rng(seed).permutation(np.unique(groups[train]))
    classes = np.unique(labels)
    rows = []
    for fraction in fractions:
        keep = set(train_groups[: max(1, math.ceil(fraction * len(train_groups)))].tolist())
        subset = train[np.isin(groups[train], list(keep))]
        if len(np.unique(labels[subset])) < 2:
            score = float("nan")
        else:
            model = _linear_probe().fit(features[subset], labels[subset])
            score = _macro_f1(labels[test], model.predict(features[test]), classes)
        rows.append((fraction, len(subset), len(test), score))
    return pd.DataFrame(rows, columns=["fraction", "n_train", "n_test", "macro_f1"])
