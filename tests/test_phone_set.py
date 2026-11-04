from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strawberry_picker.evaluation.phone_set import ground_truth, labeler_agreement
from strawberry_picker.labeling.proposals import non_max_suppression, propose
from strawberry_picker.labeling.tool import LabelStore, parse_label


def test_non_max_suppression_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.6, 0.9, 0.5])
    assert non_max_suppression(boxes, scores, iou=0.5).tolist() == [1, 2]


def test_propose_builds_a_task_from_detector_boxes(tmp_path, make_image) -> None:
    photos = [make_image(tmp_path / "IMG_2.jpg", seed=2), make_image(tmp_path / "IMG_1.jpg")]

    def detector(image):
        return np.array([[5, 5, 30, 30], [6, 6, 31, 31], [40, 5, 60, 40]]), np.array(
            [0.8, 0.7, 0.1]
        )

    task = propose(photos, detector, min_score=0.2, iou=0.5)

    assert list(task.columns) == ["item_id", "image_path", "bbox", "given", "group"]
    assert task.item_id.tolist() == ["IMG_1_0", "IMG_2_0"]  # low score and duplicate dropped
    assert json.loads(task.bbox.iloc[0]) == [5.0, 5.0, 30.0, 30.0]
    assert task.group.tolist() == ["IMG_1", "IMG_2"] and set(task.given) == {""}


@pytest.fixture
def labels(tmp_path) -> pd.DataFrame:
    path = tmp_path / "labels.csv"

    def label(labeler: str, item: str, **payload) -> None:
        LabelStore(path, labeler).record(parse_label({"item_id": item, **payload}))

    label("alice", "p1_0", maturity="ripe")
    label("alice", "p1_1", maturity="unripe")
    label("alice", "p2_0", maturity="overripe", defects=["mold"])
    label("alice", "p2_1", not_fruit=True)
    label("alice", "p3_0", maturity="ripe")
    label("bob", "p1_0", maturity="ripe")
    label("bob", "p1_1", maturity="ripe")  # disagrees with alice
    label("bob", "p2_0", maturity="overripe", defects=["mold"])
    label("adjudicated", "p1_1", maturity="unripe")
    label("bob", "p3_0", unsure=True)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_ground_truth_uses_agreement_or_adjudication(labels) -> None:
    truth = ground_truth(labels).set_index("item_id")

    assert truth.loc["p1_0", "maturity"] == "ripe"  # both agree
    assert truth.loc["p1_1", "maturity"] == "unripe"  # adjudicated
    assert truth.loc["p2_0", "defects"] == ["mold"]
    assert "p2_1" not in truth.index  # not a strawberry
    assert "p3_0" not in truth.index  # bob unsure and never adjudicated
    assert truth.loc["p1_0", "group"] == "p1"  # the photo, for the bootstrap


def test_labeler_agreement_uses_double_labelled_items_only(labels) -> None:
    agreement = labeler_agreement(labels)

    assert agreement["labelers"] == ["alice", "bob"]
    assert agreement["n_items"] == 3  # p1_0, p1_1, p2_0 (unsure and not-fruit excluded)
    assert agreement["alpha_maturity"] < 1.0


class _FakeProbe:
    """Predicts 'ripe' when the crop's mean red channel is high."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        ripe = (features[:, 0] > 128).astype(float) * 0.9 + 0.05
        return np.column_stack([1 - ripe, ripe])


def test_evaluate_pull_gives_an_adr_0005_verdict(tmp_path) -> None:
    from PIL import Image

    from strawberry_picker.evaluation.phone_set import evaluate_pull

    pull = tmp_path / "pull"
    pull.mkdir()
    rows = []
    for p in range(8):
        image = Image.new("RGB", (80, 40), (230, 20, 20))  # left: red (ripe)
        image.paste((40, 200, 40), (40, 0, 80, 40))  # right: green (unripe)
        image.save(pull / f"P{p}.jpg")
        rows += [
            (f"P{p}_0", f"P{p}.jpg", "[2, 2, 38, 38]", "", f"P{p}"),
            (f"P{p}_1", f"P{p}.jpg", "[42, 2, 78, 38]", "", f"P{p}"),
        ]
    pd.DataFrame(rows, columns=["item_id", "image_path", "bbox", "given", "group"]).to_csv(
        pull / "task.csv", index=False
    )
    for labeler in ("alice", "bob"):
        store = LabelStore(pull / "labels_task.csv", labeler)
        for item, *_ in rows:
            store.record(
                parse_label(
                    {"item_id": item, "maturity": "ripe" if item.endswith("_0") else "unripe"}
                )
            )

    def mean_colour(crops):
        return np.stack([np.asarray(c, dtype=float).mean(axis=(0, 1)) for c in crops])

    report = evaluate_pull(
        pull, _FakeProbe(), mean_colour, ("unripe", "ripe", "overripe"), n_resamples=100
    )

    assert report["agreement"]["alpha_maturity"] == pytest.approx(1.0)
    assert report["metrics"]["n"] == 16 and report["metrics"]["n_groups"] == 8
    checks = {c["id"]: c for c in report["checks"]}
    assert checks["A"]["passed"]
    assert checks["B2"]["passed"] is None  # no overripe berries: recall can't be judged
    assert checks["B3"]["passed"]
    assert report["verdict"] == "incomplete"  # a class with no examples blocks a full pass


def test_unresolved_items_need_adjudication(labels) -> None:
    from strawberry_picker.evaluation.phone_set import unresolved_items

    assert unresolved_items(labels) == ["p3_0"]  # p1_1 was adjudicated; p2_1 is not a fruit


def test_evaluate_pull_crops_with_the_training_policy(tmp_path) -> None:
    from PIL import Image

    from strawberry_picker.data.crops import CropPolicy
    from strawberry_picker.evaluation.phone_set import evaluate_pull

    pull = tmp_path / "pull"
    pull.mkdir()
    image = Image.new("RGB", (80, 40), (230, 20, 20))  # left: red berry
    image.paste((40, 200, 40), (40, 0, 80, 40))  # right: green berry
    image.save(pull / "P0.jpg", quality=100)
    rows = [
        ("P0_0", "P0.jpg", "[10, 10, 30, 30]", "", "P0"),
        ("P0_1", "P0.jpg", "[40, 0, 80, 40]", "", "P0"),
        ("P0_2", "P0.jpg", "[0, 0, 80, 40]", "", "P0"),  # a bad proposal: rejected below
    ]
    pd.DataFrame(rows, columns=["item_id", "image_path", "bbox", "given", "group"]).to_csv(
        pull / "task.csv", index=False
    )
    store = LabelStore(pull / "labels_task.csv", "alice")
    store.record(parse_label({"item_id": "P0_0", "maturity": "ripe"}))
    store.record(parse_label({"item_id": "P0_1", "maturity": "unripe"}))
    store.record(parse_label({"item_id": "P0_2", "not_fruit": True}))
    seen = []

    def record(crops):
        seen.extend(np.asarray(c, dtype=float) for c in crops)
        return np.stack([c.mean(axis=(0, 1)) for c in seen[-len(crops) :]])

    policy = CropPolicy(padding=1.0, neighbours="drop", max_neighbour_overlap=0.0)
    report = evaluate_pull(
        pull,
        _FakeProbe(),
        record,
        ("unripe", "ripe", "overripe"),
        n_resamples=10,
        crop_policy=policy,
    )
    # Drop never skips a test berry; the crop is cut as `keep` and the green leaks in.
    assert report["metrics"]["n"] == 2
    assert report["crop_policy"] == policy.for_evaluation().to_dict()
    assert seen[0][..., 1].max() > 150

    seen.clear()
    masked = CropPolicy(padding=1.0, neighbours="mask", fill=(0, 0, 0))
    evaluate_pull(
        pull,
        _FakeProbe(),
        record,
        ("unripe", "ripe", "overripe"),
        n_resamples=10,
        crop_policy=masked,
    )
    # The green neighbour is masked; the rejected whole-photo box is not a neighbour.
    left = seen[0]  # crop_box → (0, 0, 60, 40); the target box is columns 10-30
    assert left[..., 1].max() < 60
    assert left[20, 2, 0] > 200  # context left of the target is kept
