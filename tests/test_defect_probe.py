"""Defect head baseline (ADR 0010): per-defect linear probes, val-only thresholds, rule grade."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strawberry_picker.config import load_config
from strawberry_picker.schema import Defect

DEFECTS = tuple(d.value for d in Defect)


def test_defect_targets_keep_unassessed_apart_from_clean() -> None:
    from strawberry_picker.models.defect_probe import defect_targets

    column = pd.Series([["rot"], [], None, np.array(["mold", "rot"]), float("nan")])

    targets, assessed = defect_targets(column, ["mold", "rot"])

    assert assessed.tolist() == [True, True, False, True, False]
    assert targets.tolist() == [[0, 1], [0, 0], [0, 0], [1, 1], [0, 0]]


def test_defect_targets_reject_unknown_defects() -> None:
    from strawberry_picker.models.defect_probe import defect_targets

    with pytest.raises(ValueError, match="scratch"):
        defect_targets(pd.Series([["scratch"]]), DEFECTS)


def test_recall_target_threshold_is_the_highest_that_meets_the_target() -> None:
    from strawberry_picker.models.defect_probe import choose_threshold

    probs = np.array([0.95, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
    truth = np.array([1, 1, 0, 1, 1, 0, 0, 0])

    threshold = choose_threshold(probs, truth, {"rule": "recall_target", "target": 0.75})

    assert threshold == pytest.approx(0.6)  # 3 of 4 positives at >= 0.6; 0.8 only catches 2
    assert ((probs >= threshold) & (truth == 1)).sum() / truth.sum() >= 0.75


def test_max_f1_threshold() -> None:
    from strawberry_picker.models.defect_probe import choose_threshold

    probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    truth = np.array([1, 1, 0, 1, 0])

    # >= 0.8: P 1, R 2/3, F1 0.8. >= 0.3: P 3/4, R 1, F1 0.857 (best). >= 0.2: F1 0.75.
    assert choose_threshold(probs, truth, {"rule": "max_f1"}) == pytest.approx(0.3)


def test_threshold_without_validation_positives_falls_back_to_one_half() -> None:
    from strawberry_picker.models.defect_probe import choose_threshold

    assert choose_threshold(np.array([0.2, 0.7]), np.array([0, 0]), {"rule": "max_f1"}) == 0.5
    with pytest.raises(ValueError, match="rule"):
        choose_threshold(np.array([0.2]), np.array([1]), {"rule": "vibes"})


def _rows(n_per_source: int = 120, *, sources=("a", "proxy")) -> pd.DataFrame:
    """Rows where rot is common, mold is common, bruise is rare, and a few are unassessed."""
    frames = []
    for source in sources:
        k = np.arange(n_per_source)
        split = np.where(k % 10 < 6, "train", np.where(k % 10 < 8, "val", "test"))
        defects: list[object] = [
            ["rot"] if i % 4 == 0 else ["mold"] if i % 4 == 1 else [] for i in range(n_per_source)
        ]
        defects[2] = ["bruise"]
        defects[3] = None  # not assessed
        frames.append(
            pd.DataFrame(
                {
                    "source_name": source,
                    "split": split,
                    "defects": defects,
                    "group_id": [f"{source}/g{i // 2}" for i in range(n_per_source)],
                    "crop_id": [f"{source}_{i}" for i in range(n_per_source)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _embeddings(rows: pd.DataFrame, *, flip_source: str | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(len(rows), 6))
    rot = np.array([d is not None and "rot" in list(d) for d in rows.defects])
    mold = np.array([d is not None and "mold" in list(d) for d in rows.defects])
    features[:, 0] += np.where(rot, 4.0, -1.0)
    features[:, 1] += np.where(mold, 4.0, -1.0)
    if flip_source is not None:
        flip = (rows.source_name == flip_source).to_numpy()
        features[flip, :2] *= -1  # this source looks the other way round
    return features


def test_train_defect_head_skips_rare_defects_and_never_counts_unassessed() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.defect_probe import defect_targets, train_defect_head

    rows = _rows(sources=("a",))
    targets, assessed = defect_targets(rows.defects, DEFECTS)
    features = _embeddings(rows)
    train = (rows.split == "train").to_numpy()
    val = (rows.split == "val").to_numpy()

    head = train_defect_head(
        features, targets, assessed, DEFECTS, train=train, val=val, min_positive=10, seed=0
    )

    assert head.defects == ("mold", "rot")
    assert set(head.untrained) == {"bruise", "deformed", "pest_damage", "mechanical_damage"}
    assert head.untrained["bruise"]["n_positive"] == 1
    # Row 3 is unassessed and in train: it is neither a positive nor a negative.
    n_train_assessed = int((train & assessed).sum())
    assert head.counts["rot"]["n_positive"] + head.counts["rot"]["n_negative"] == n_train_assessed
    assert n_train_assessed == int(train.sum()) - 1
    assert set(head.thresholds) == {"mold", "rot"}


def test_defect_head_predicts_defects_and_derives_the_grade_by_rule() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.defect_probe import defect_targets, train_defect_head

    rows = _rows(sources=("a",))
    targets, assessed = defect_targets(rows.defects, DEFECTS)
    features = _embeddings(rows)
    head = train_defect_head(
        features,
        targets,
        assessed,
        DEFECTS,
        train=(rows.split == "train").to_numpy(),
        val=(rows.split == "val").to_numpy(),
        min_positive=10,
        seed=0,
    )

    probe_points = np.zeros((3, 6))
    probe_points[0, 0] = 8  # rot
    probe_points[1, 1] = 8  # mold
    probe_points[2, :2] = -4  # clean

    assert head.predict_proba(probe_points).shape == (3, 2)
    assert head.predict(probe_points) == [("rot",), ("mold",), ()]
    assert head.grades(probe_points) == ["reject", "reject", "premium"]


def test_thresholds_come_from_val_only() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.defect_probe import defect_targets, train_defect_head

    rows = _rows(sources=("a",))
    targets, assessed = defect_targets(rows.defects, DEFECTS)
    features = _embeddings(rows)
    train = (rows.split == "train").to_numpy()
    val = (rows.split == "val").to_numpy()
    kwargs = {"train": train, "val": val, "min_positive": 10, "seed": 0}

    first = train_defect_head(features, targets, assessed, DEFECTS, **kwargs)
    scrambled = features.copy()
    test = (rows.split == "test").to_numpy()
    scrambled[test] = np.random.default_rng(1).normal(size=(int(test.sum()), 6)) * 10
    second = train_defect_head(scrambled, targets, assessed, DEFECTS, **kwargs)

    assert first.thresholds == second.thresholds


def test_evaluate_quality_reports_adr_0005_c_numbers() -> None:
    from strawberry_picker.models.defect_probe import evaluate_quality

    trained = ("mold", "rot")
    truth = [["rot"], ["mold"], [], ["bruise"], ["rot", "mold"], [], None]
    predicted = [("rot",), (), ("mold",), (), ("rot",), (), ("rot",)]
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4"])

    report = evaluate_quality(predicted, truth, groups, trained, n_resamples=100)

    assert report["n"] == 6  # the unassessed row is left out
    assert report["n_groups"] == 3
    rot, mold = report["defects"]["rot"], report["defects"]["mold"]
    assert rot["precision"]["point"] == pytest.approx(1.0)
    assert rot["recall"]["point"] == pytest.approx(1.0)
    assert mold["precision"]["point"] == pytest.approx(0.0)
    assert mold["recall"]["point"] == pytest.approx(0.0)
    assert rot["n_positive"] == 2 and mold["n_positive"] == 2
    # Reject truth: rows 0, 1, 4. Predicted reject: rows 0, 2, 4 → recall 2/3.
    assert report["reject_recall"]["point"] == pytest.approx(2 / 3)
    low, high = report["reject_recall"]["low"], report["reject_recall"]["high"]
    assert low <= report["reject_recall"]["point"] <= high
    assert report["grade_confusion"]["labels"] == ["premium", "standard", "reject"]
    assert "verdict" not in report  # only the locked phone set gives one
    json.dumps(report)


def test_reject_verdict_uses_the_pre_registered_bar() -> None:
    from strawberry_picker.models.defect_probe import REJECT_RECALL_BAR, reject_verdict

    assert REJECT_RECALL_BAR == 0.80
    assert reject_verdict({"reject_recall": {"point": 0.81}}) is True
    assert reject_verdict({"reject_recall": {"point": 0.79}}) is False
    assert reject_verdict({"reject_recall": {"point": None}}) is False


def test_run_quality_baseline_never_trains_on_held_out_sources() -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.defect_probe import run_quality_baseline

    rows = _rows()
    features = _embeddings(rows, flip_source="proxy")

    head, metrics = run_quality_baseline(
        rows,
        features,
        DEFECTS,
        holdout_sources=["proxy"],
        seed=0,
        min_positive=10,
        n_resamples=50,
    )

    a_train = ((rows.source_name == "a") & (rows.split == "train")).to_numpy()
    assert metrics["n_train"] == int(a_train.sum()) - 1  # minus the unassessed row
    assert metrics["trained_defects"] == ["mold", "rot"]
    assert "bruise" in metrics["untrained_defects"]
    in_dist = metrics["in_distribution"]
    assert in_dist["reject_recall"]["point"] > 0.9
    proxy = metrics["proxy"]["proxy"]
    assert proxy["n"] == 119  # every assessed proxy row, any split
    assert proxy["reject_recall"]["point"] < 0.5  # held out, so its flip is exposed
    json.dumps(metrics)


def test_quality_run_round_trips_through_the_linear_probe_helpers(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from strawberry_picker.models.defect_probe import defect_targets, train_defect_head
    from strawberry_picker.models.linear_probe import load_probe, save_run

    rows = _rows(sources=("a",))
    targets, assessed = defect_targets(rows.defects, DEFECTS)
    features = _embeddings(rows)
    head = train_defect_head(
        features,
        targets,
        assessed,
        DEFECTS,
        train=(rows.split == "train").to_numpy(),
        val=(rows.split == "val").to_numpy(),
        min_positive=10,
        seed=0,
    )

    run = save_run(tmp_path / "run", head, {"task": "quality_cls"}, {"ok": 1})
    reloaded = load_probe(run)

    assert reloaded.predict(features) == head.predict(features)
    assert reloaded.thresholds == head.thresholds


def test_quality_config_is_valid() -> None:
    config = load_config("quality_cls")

    assert config["task"] == "quality_cls"
    assert set(config["defects"]) == set(DEFECTS)
    assert "kamat_2025" in config["holdout_sources"]
    assert config["eval"]["reject_recall_bar"] == 0.80
    rules = config["thresholds"]
    assert set(rules) - {"default"} <= set(DEFECTS)
    assert all(rule["rule"] in {"max_f1", "recall_target"} for rule in rules.values())
