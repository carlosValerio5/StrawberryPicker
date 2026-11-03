"""Baseline maturity model: a linear probe on frozen DINOv2 embeddings (ADR 0003 baseline).

A run is ``models/<task>/<run_id>/`` holding ``probe.joblib``, the resolved
``config.yaml`` and ``metrics.json`` (AGENTS.md conventions). Runs are never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def train_probe(features: np.ndarray, labels: np.ndarray, *, seed: int = 0, c: float = 1.0):
    """Standardise, then class-balanced logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed),
    )
    return model.fit(features, labels)


def predict_all_classes(probe, features: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities for every class code ``0..n_classes-1``, zero for classes never trained."""
    probs = np.zeros((len(features), n_classes))
    probs[:, probe.classes_] = probe.predict_proba(features)
    return probs


def save_run(run_dir: Path, probe, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
    import joblib

    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; runs are never overwritten")
    run_dir.mkdir(parents=True)
    joblib.dump(probe, run_dir / "probe.joblib")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir


def load_probe(run_dir: Path):
    import joblib

    return joblib.load(run_dir / "probe.joblib")
