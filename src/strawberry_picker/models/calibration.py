"""Temperature scaling: one scalar that makes probe confidences honest (ADR 0005 §B3).

The temperature ``T`` divides the log-probabilities before re-normalising, so
``softmax(log p / T)``. ``T > 1`` softens an overconfident model, ``T < 1`` sharpens an
underconfident one, and the argmax never changes (Guo et al. 2017). ``T`` is fitted on
the gold **val** split only: never on test, a held-out proxy source, or the phone set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CALIBRATION_METHODS = ("temperature", "none")
_EPS = 1e-12


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(log probs / T)`` row by row. Classes with probability 0 stay at 0."""
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    probs = np.asarray(probs, dtype=float)
    present = probs > 0
    logits = np.where(present, np.log(np.where(present, probs, 1.0)) / temperature, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.where(present, np.exp(logits), 0.0)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    picked = apply_temperature(probs, temperature)[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, _EPS, None)).mean())


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, *, bounds: tuple[float, float] = (0.05, 20.0)
) -> float:
    """The ``T`` within ``bounds`` that minimises negative log-likelihood on ``labels``."""
    from scipy.optimize import minimize_scalar

    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on zero rows")
    result = minimize_scalar(
        lambda log_t: _nll(probs, labels, float(np.exp(log_t))),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )
    return float(np.exp(result.x))


class TemperatureScaled:
    """A fitted probe whose ``predict_proba`` is temperature-scaled; otherwise the same probe."""

    def __init__(self, base: Any, temperature: float) -> None:
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.base = base
        self.temperature = float(temperature)

    @property
    def classes_(self) -> np.ndarray:
        return self.base.classes_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return apply_temperature(self.base.predict_proba(features), self.temperature)
