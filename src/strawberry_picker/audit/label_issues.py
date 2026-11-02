"""Confident learning: rank the labels a model most confidently disagrees with.

Northcutt, Jiang & Chuang (2021), *Confident Learning: Estimating Uncertainty in Dataset
Labels*. Implemented here rather than depending on ``cleanlab``, which is AGPL-3.0
(AGENTS.md rule 7). The probabilities must be out-of-sample: a model never scores rows it
was trained on, otherwise it simply memorises the wrong labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )


def out_of_fold_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Class probabilities for every row from a linear probe that never saw that row's group.

    ``labels`` are integer codes ``0..k-1``; column ``j`` of the result is class ``j``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_classes = int(labels.max()) + 1
    probs = np.zeros((len(labels), n_classes))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(features, labels, groups):
        model = _linear_probe().fit(features[train], labels[train])
        probs[np.ix_(test, model.classes_)] = model.predict_proba(features[test])
    return probs


def find_label_issues(labels: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """One row per example: ``given``, ``suggested``, ``self_confidence``, ``is_issue``.

    Class ``k``'s threshold is the mean probability of ``k`` over rows labelled ``k``. A row
    is an issue when some *other* class clears its own threshold and is the most probable
    of the classes that do. Issues come first, least self-confident first.
    """
    labels = np.asarray(labels)
    if len(labels) != len(probs):
        raise ValueError(f"{len(labels)} labels but {len(probs)} probability rows")
    n_classes = probs.shape[1]
    thresholds = np.array(
        [probs[labels == k, k].mean() if (labels == k).any() else np.inf for k in range(n_classes)]
    )
    confident = probs >= thresholds
    suggested = np.where(confident, probs, -np.inf).argmax(axis=1)
    has_confident = confident.any(axis=1)
    suggested = np.where(has_confident, suggested, labels)
    frame = pd.DataFrame(
        {
            "given": labels,
            "suggested": suggested,
            "self_confidence": probs[np.arange(len(labels)), labels],
            "is_issue": has_confident & (suggested != labels),
        }
    )
    return frame.sort_values(
        ["is_issue", "self_confidence"], ascending=[False, True], kind="stable"
    )
