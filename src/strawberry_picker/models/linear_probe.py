"""Baseline maturity model: a linear probe on frozen DINOv2 embeddings (ADR 0003 baseline).

A run is ``models/<task>/<run_id>/`` holding ``probe.joblib``, the resolved
``config.yaml`` and ``metrics.json`` (AGENTS.md conventions). Runs are never overwritten.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from strawberry_picker.evaluation.maturity import evaluate_maturity


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


def run_baseline(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    classes: Sequence[str],
    *,
    holdout_sources: Sequence[str],
    seed: int,
    n_resamples: int = 2000,
    c: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Train on gold ``train`` rows outside ``holdout_sources``; score the rest separately.

    ``in_distribution`` is the gold ``test`` split of the training sources. Each held-out
    source is a development proxy (ADR 0007), scored on all of its rows. Neither is the
    ADR 0005 verdict; only the locked phone set is.
    """
    codes = rows.maturity.map({name: k for k, name in enumerate(classes)}).to_numpy()
    held_out = rows.source_name.isin(holdout_sources).to_numpy()
    train = (rows.split == "train").to_numpy() & ~held_out
    test = (rows.split == "test").to_numpy() & ~held_out
    probe = train_probe(embeddings[train], codes[train], seed=seed, c=c)

    def score(mask: np.ndarray) -> dict[str, Any]:
        probs = predict_all_classes(probe, embeddings[mask], len(classes))
        return evaluate_maturity(
            probs, codes[mask], rows.group_id.to_numpy()[mask], classes, n_resamples=n_resamples
        )

    return probe, {
        "n_train": int(train.sum()),
        "train_counts": rows[train].maturity.value_counts().to_dict(),
        "in_distribution": score(test),
        "proxy": {
            source: score((rows.source_name == source).to_numpy()) for source in holdout_sources
        },
    }
