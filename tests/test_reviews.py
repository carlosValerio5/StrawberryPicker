from __future__ import annotations

import pandas as pd
import pytest

from strawberry_picker.data.reviews import apply_reviews, load_reviews, resolve_reviews
from strawberry_picker.labeling.tool import LabelStore, parse_label


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["img1", "img1", "img2", "img3"],
            "source_name": ["kamat_2025"] * 4,
            "maturity": ["overripe", "unripe", "overripe", "ripe"],
            "defects": [["mold"], None, ["rot"], []],
            "needs_review": [True, True, True, False],
            "reviewed_by": [None] * 4,
        }
    )


def _review(tmp_path, pull: str, labeler: str, labels: list[dict]) -> None:
    store = LabelStore(tmp_path / "bronze" / "human_review" / pull / "labels.csv", labeler)
    for payload in labels:
        store.record(parse_label(payload))


@pytest.fixture
def reviews(tmp_data_dir, tmp_path) -> pd.DataFrame:
    _review(tmp_path, "2026-11-05", "alice", [
        {"item_id": "img1_0", "maturity": "overripe", "defects": ["mold"]},
        {"item_id": "img1_1", "not_fruit": True},  # a flower
        {"item_id": "img2_0", "maturity": "ripe"},
        {"item_id": "img3_0", "unsure": True},
    ])
    _review(tmp_path, "2026-11-05.2", "bob", [
        {"item_id": "img1_0", "maturity": "overripe", "defects": ["mold"]},
        {"item_id": "img2_0", "maturity": "overripe", "defects": ["rot"]},  # disagrees
    ])
    return load_reviews()


def test_load_reviews_reads_every_pull(reviews) -> None:
    assert set(reviews.labeler) == {"alice", "bob"}
    assert len(reviews) == 6


def test_resolve_reviews_needs_agreement(reviews) -> None:
    resolved = resolve_reviews(reviews).set_index("item_id")

    assert resolved.loc["img1_0", "agreed"] and resolved.loc["img1_0", "reviewed_by"] == "alice+bob"
    assert resolved.loc["img1_1", "agreed"] and resolved.loc["img1_1", "not_fruit"]
    assert not resolved.loc["img2_0", "agreed"]  # alice: ripe, bob: overripe + rot
    assert not resolved.loc["img3_0", "agreed"]  # unsure never counts as agreement


def test_resolve_reviews_respects_undo(tmp_data_dir, tmp_path) -> None:
    store = LabelStore(tmp_path / "bronze" / "human_review" / "2026-11-05" / "l.csv", "alice")
    store.record(parse_label({"item_id": "img1_0", "maturity": "ripe"}))
    store.undo()
    store.record(parse_label({"item_id": "img1_0", "maturity": "overripe"}))

    resolved = resolve_reviews(load_reviews()).set_index("item_id")

    assert resolved.loc["img1_0", "maturity"] == "overripe"


def test_apply_reviews_overrides_holds_and_never_drops_rows(reviews) -> None:
    result = apply_reviews(_manifest(), resolve_reviews(reviews))

    assert len(result) == 4  # crop ids stay stable: reviews never delete rows
    agreed, flower, disputed, unsure = (result.iloc[i] for i in range(4))
    assert (agreed.maturity, list(agreed.defects), agreed.needs_review) == ("overripe", ["mold"], False)
    assert agreed.reviewed_by == "alice+bob"
    assert flower.needs_review and flower.maturity is None and flower.reviewed_by == "alice"
    assert disputed.needs_review and disputed.maturity == "overripe"  # source label kept
    assert unsure.needs_review  # was gold-eligible; a reviewer was unsure, so it's held


def test_no_reviews_means_no_change(tmp_data_dir) -> None:
    manifest = _manifest()
    assert load_reviews().empty
    pd.testing.assert_frame_equal(apply_reviews(manifest, resolve_reviews(load_reviews())), manifest)
