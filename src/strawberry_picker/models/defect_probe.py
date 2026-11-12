"""Baseline defect head: one linear probe per defect on frozen crop embeddings (ADR 0010).

- Multi-label: each defect gets its own class-balanced logistic probe, trained only on rows
  whose defects were assessed. ``defects: null`` (not assessed) is never a negative.
- A defect with too few positive (or negative) training rows is left untrained and listed.
- Each probe's threshold is chosen on the gold ``val`` split only, by a rule fixed in the
  config before results are seen.
- The grade is derived from the thresholded defects by ``grade_from_defects``; it is never
  learned (data contract §2b).

The head is saved and loaded with ``models.linear_probe.save_run`` / ``load_probe``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from strawberry_picker.evaluation.metrics import bootstrap_ci
from strawberry_picker.models.linear_probe import train_probe
from strawberry_picker.schema import Defect, Grade, grade_from_defects

REJECT_RECALL_BAR = 0.80  # ADR 0005 §C, fixed before results
GRADES = tuple(g.value for g in (Grade.PREMIUM, Grade.STANDARD, Grade.REJECT))
DEFAULT_THRESHOLD = 0.5  # used when val has no positives to choose from

ThresholdRule = Mapping[str, Any]  # {"rule": "max_f1"} | {"rule": "recall_target", "target": x}


def _is_assessed(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def defect_targets(
    column: Iterable[object], defects: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """``(targets, assessed)``: an ``(n, len(defects))`` 0/1 matrix and an ``(n,)`` mask.

    Unassessed rows (null) get all-zero targets and ``assessed = False``; callers must mask
    them out rather than read the zeros as "no defect".
    """
    values = list(column)
    targets = np.zeros((len(values), len(defects)), dtype=int)
    assessed = np.zeros(len(values), dtype=bool)
    index = {name: k for k, name in enumerate(defects)}
    for row, value in enumerate(values):
        if not _is_assessed(value):
            continue
        assessed[row] = True
        for name in value:  # type: ignore[union-attr]
            Defect(name)  # unknown names fail loudly
            if name in index:
                targets[row, index[name]] = 1
    return targets, assessed


def _defect_lists(column: Iterable[object]) -> list[list[str] | None]:
    return [[str(d) for d in v] if _is_assessed(v) else None for v in column]  # type: ignore[union-attr]


def choose_threshold(probs: np.ndarray, truth: np.ndarray, rule: ThresholdRule) -> float:
    """Pick a decision threshold on validation scores; predictions are ``probs >= threshold``.

    - ``recall_target``: the highest threshold whose recall is at least ``target``.
    - ``max_f1``: the threshold with the best F1 (the highest one on ties).
    Without any validation positive, returns ``DEFAULT_THRESHOLD``.
    """
    kind = rule.get("rule")
    if kind not in {"max_f1", "recall_target"}:
        raise ValueError(f"unknown threshold rule {kind!r}; use max_f1 or recall_target")
    probs, truth = np.asarray(probs, dtype=float), np.asarray(truth, dtype=int)
    positives = np.sort(probs[truth == 1])[::-1]
    if not len(positives):
        return DEFAULT_THRESHOLD
    if kind == "recall_target":
        needed = max(1, math.ceil(float(rule["target"]) * len(positives)))
        return float(positives[needed - 1])
    best, best_f1 = DEFAULT_THRESHOLD, -1.0
    for threshold in np.unique(probs)[::-1]:
        predicted = probs >= threshold
        tp = int(np.sum(predicted & (truth == 1)))
        f1 = 2 * tp / (predicted.sum() + len(positives))
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best


@dataclass
class DefectHead:
    """Per-defect probes plus their validation thresholds; picklable for ``save_run``."""

    defects: tuple[str, ...]
    probes: dict[str, Any]
    thresholds: dict[str, float]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    untrained: dict[str, dict[str, int]] = field(default_factory=dict)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """``(n, len(self.defects))`` positive-class probabilities."""
        columns = [self.probes[d].predict_proba(features)[:, 1] for d in self.defects]
        return np.stack(columns, axis=1) if columns else np.zeros((len(features), 0))

    def predict(self, features: np.ndarray) -> list[tuple[str, ...]]:
        """Thresholded defects per row. Untrained defects are never predicted."""
        probs = self.predict_proba(features)
        cut = np.array([self.thresholds[d] for d in self.defects])
        return [
            tuple(d for d, hit in zip(self.defects, row, strict=True) if hit)
            for row in probs >= cut
        ]

    def grades(self, features: np.ndarray) -> list[str]:
        """Grade by rule (data contract §2b) from the thresholded defects."""
        return [grade_from_defects(found).value for found in self.predict(features)]


def train_defect_head(
    features: np.ndarray,
    targets: np.ndarray,
    assessed: np.ndarray,
    defects: Sequence[str],
    *,
    train: np.ndarray,
    val: np.ndarray,
    min_positive: int,
    seed: int,
    c: float = 1.0,
    thresholds: Mapping[str, ThresholdRule] | None = None,
) -> DefectHead:
    """Fit one probe per defect with at least ``min_positive`` positives and negatives.

    Only assessed rows in ``train`` are fitted; thresholds use assessed rows in ``val``.
    ``thresholds`` maps a defect (or ``default``) to its rule; the default is ``max_f1``.
    """
    rules = dict(thresholds or {})
    default_rule = rules.get("default", {"rule": "max_f1"})
    fit, tune = train & assessed, val & assessed
    probes, chosen, counts, untrained = {}, {}, {}, {}
    for k, name in enumerate(defects):
        y = targets[:, k]
        n_pos, n_neg = int(y[fit].sum()), int((1 - y[fit]).sum())
        count = {"n_positive": n_pos, "n_negative": n_neg}
        if n_pos < min_positive or n_neg < min_positive:
            untrained[name] = count
            continue
        probe = train_probe(features[fit], y[fit], seed=seed, c=c)
        val_probs = probe.predict_proba(features[tune])[:, 1]
        chosen[name] = choose_threshold(val_probs, y[tune], rules.get(name, default_rule))
        probes[name], counts[name] = probe, count
    return DefectHead(tuple(probes), probes, chosen, counts, untrained)


def _interval(values: tuple[float, float, float]) -> dict[str, float | None]:
    names = ("point", "low", "high")
    return {k: (None if np.isnan(v) else float(v)) for k, v in zip(names, values, strict=True)}


def _ratio(hits: np.ndarray, total: np.ndarray) -> float:
    return float(hits.sum() / total.sum()) if total.sum() else float("nan")


def evaluate_quality(
    predicted: Sequence[Sequence[str]],
    truth: Sequence[Sequence[str] | None],
    groups: np.ndarray,
    trained: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """ADR 0005 §C numbers: per-defect precision/recall and reject-grade recall, with intervals.

    Rows whose truth is ``None`` (not assessed) are left out. Intervals resample ``groups``
    (photos, or images for public sources). Reports never carry a verdict: only the locked
    phone set does, through ``reject_verdict``.
    """
    keep = np.array([t is not None for t in truth], dtype=bool)
    truth_kept = [list(t) for t, k in zip(truth, keep, strict=True) if k]
    pred_kept = [list(p) for p, k in zip(predicted, keep, strict=True) if k]
    groups = np.asarray(groups)[keep]
    if not len(groups):  # e.g. a proxy source with no assessed rows in this gold release
        return {"n": 0, "n_groups": 0, "trained_defects": list(trained), "defects": {}}
    ci = {"n_resamples": n_resamples, "seed": seed}

    def column(lists: list[list[str]], name: str) -> np.ndarray:
        return np.array([name in labels for labels in lists], dtype=bool)

    per_defect: dict[str, Any] = {}
    for name in trained:
        t, p = column(truth_kept, name), column(pred_kept, name)
        per_defect[name] = {
            "n_positive": int(t.sum()),
            "n_predicted": int(p.sum()),
            "precision": _interval(
                bootstrap_ci(lambda i, t=t, p=p: _ratio(t[i] & p[i], p[i]), groups, **ci)
            ),
            "recall": _interval(
                bootstrap_ci(lambda i, t=t, p=p: _ratio(t[i] & p[i], t[i]), groups, **ci)
            ),
        }

    true_grade = np.array([grade_from_defects(t).value for t in truth_kept])
    pred_grade = np.array([grade_from_defects(p).value for p in pred_kept])
    is_reject, called_reject = true_grade == Grade.REJECT, pred_grade == Grade.REJECT
    confusion = [
        [int(np.sum((true_grade == a) & (pred_grade == b))) for b in GRADES] for a in GRADES
    ]
    return {
        "n": int(keep.sum()),
        "n_groups": int(len(np.unique(groups))),
        "trained_defects": list(trained),
        "defects": per_defect,
        "reject_recall": _interval(
            bootstrap_ci(
                lambda i: _ratio(is_reject[i] & called_reject[i], is_reject[i]), groups, **ci
            )
        ),
        "reject_recall_bar": REJECT_RECALL_BAR,
        "grade_confusion": {"labels": list(GRADES), "rows_truth_cols_predicted": confusion},
    }


def reject_verdict(report: Mapping[str, Any], bar: float = REJECT_RECALL_BAR) -> bool:
    """ADR 0005 §C gate on a report from ``evaluate_quality``.

    Call this only on the locked phone set (``scripts/evaluate_phone_set.py``); public-source
    and proxy reports rank models and never get a verdict.
    """
    point = report["reject_recall"]["point"]
    return point is not None and point >= bar


def run_quality_baseline(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    defects: Sequence[str],
    *,
    holdout_sources: Sequence[str],
    seed: int,
    min_positive: int,
    thresholds: Mapping[str, ThresholdRule] | None = None,
    n_resamples: int = 2000,
    c: float = 1.0,
) -> tuple[DefectHead, dict[str, Any]]:
    """Train on gold ``train`` outside ``holdout_sources``, tune on ``val``, score the rest.

    ``in_distribution`` is the gold ``test`` split of the training sources. Each held-out
    source is a development proxy (ADR 0007), scored on all of its assessed rows.
    """
    targets, assessed = defect_targets(rows.defects, defects)
    held_out = rows.source_name.isin(holdout_sources).to_numpy()
    split = rows.split.to_numpy()
    head = train_defect_head(
        embeddings,
        targets,
        assessed,
        defects,
        train=(split == "train") & ~held_out,
        val=(split == "val") & ~held_out,
        min_positive=min_positive,
        seed=seed,
        c=c,
        thresholds=thresholds,
    )
    truth = _defect_lists(rows.defects)
    groups = rows.group_id.to_numpy()

    def score(mask: np.ndarray) -> dict[str, Any]:
        index = np.flatnonzero(mask)
        predicted = head.predict(embeddings[index]) if len(index) else []
        return evaluate_quality(
            predicted,
            [truth[i] for i in index],
            groups[index],
            head.defects,
            n_resamples=n_resamples,
            seed=seed,
        )

    fit = (split == "train") & ~held_out & assessed
    return head, {
        "n_train": int(fit.sum()),
        "trained_defects": list(head.defects),
        "train_counts": head.counts,
        "untrained_defects": head.untrained,
        "thresholds": head.thresholds,
        "in_distribution": score((split == "test") & ~held_out),
        "proxy": {
            source: score((rows.source_name == source).to_numpy()) for source in holdout_sources
        },
    }
