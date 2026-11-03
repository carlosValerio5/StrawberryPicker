"""The numbers ADR 0005 judges a model by. Pure numpy, so they also run without the ml extra.

- ``krippendorff_alpha``: do two (or more) people agree on the labels? (§A)
- ``quadratic_weighted_kappa``, ``per_class_recall``, ``expected_calibration_error``: §B.
- ``bootstrap_ci``: intervals that resample whole *photos*, because berries in one photo
  share lighting and background (ADR 0007).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_LEVELS = ("nominal", "ordinal")


def krippendorff_alpha(ratings: np.ndarray, *, level: str = "nominal") -> float:
    """Krippendorff's α for a ``coders × units`` array; ``NaN`` marks a missing rating.

    Ordinal distances follow Krippendorff (2011): the closer two categories are in the
    ranked distribution of all values, the smaller their disagreement.
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
    ratings = np.asarray(ratings, dtype=float)
    values = np.unique(ratings[~np.isnan(ratings)])
    index = {v: i for i, v in enumerate(values)}
    k = len(values)

    coincidences = np.zeros((k, k))
    for unit in ratings.T:
        present = unit[~np.isnan(unit)]
        if len(present) < 2:
            continue  # a value nobody else rated can't agree or disagree
        counts = np.bincount([index[v] for v in present], minlength=k).astype(float)
        coincidences += (np.outer(counts, counts) - np.diag(counts)) / (len(present) - 1)

    marginals = coincidences.sum(axis=1)
    n = marginals.sum()
    if level == "nominal":
        delta = 1.0 - np.eye(k)
    else:
        cumulative = np.cumsum(marginals)
        delta = np.zeros((k, k))
        for c in range(k):
            for d in range(c + 1, k):
                between = cumulative[d] - (cumulative[c - 1] if c else 0.0)
                delta[c, d] = delta[d, c] = (between - (marginals[c] + marginals[d]) / 2) ** 2
    expected = np.outer(marginals, marginals)
    return float(1 - (n - 1) * (coincidences * delta).sum() / (expected * delta).sum())


def quadratic_weighted_kappa(truth: np.ndarray, predicted: np.ndarray, *, n_classes: int) -> float:
    """Cohen's κ with quadratic weights: being off by two classes costs four times as much."""
    observed = np.zeros((n_classes, n_classes))
    np.add.at(observed, (truth, predicted), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    grid = np.arange(n_classes)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (n_classes - 1) ** 2
    disagreement_by_chance = (weights * expected).sum()
    if disagreement_by_chance == 0:  # one class only: κ is undefined
        return float("nan")
    return float(1 - (weights * observed).sum() / disagreement_by_chance)


def per_class_recall(truth: np.ndarray, predicted: np.ndarray, *, n_classes: int) -> np.ndarray:
    """Recall per class; ``NaN`` for a class with no true examples."""
    return np.array(
        [
            (predicted[truth == k] == k).mean() if (truth == k).any() else np.nan
            for k in range(n_classes)
        ]
    )


def expected_calibration_error(probs: np.ndarray, truth: np.ndarray, *, n_bins: int = 15) -> float:
    """Top-label ECE over ``n_bins`` equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in np.unique(bins):
        members = bins == b
        error += abs(correct[members].mean() - confidence[members].mean()) * members.mean()
    return float(error)


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """``(point, low, high)`` for ``metric(row_indices)``, resampling whole groups."""
    groups = np.asarray(groups)
    names, codes = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(codes == g) for g in range(len(names))]
    rng = np.random.default_rng(seed)
    scores = [
        metric(np.concatenate([members[g] for g in rng.integers(0, len(names), len(names))]))
        for _ in range(n_resamples)
    ]
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]  # e.g. recall of a class a resample didn't draw
    point = metric(np.arange(len(groups)))
    if not len(finite):
        return point, float("nan"), float("nan")
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(finite, [tail, 100 - tail], method="nearest")
    return point, float(low), float(high)
