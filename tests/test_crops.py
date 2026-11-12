from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from strawberry_picker.data import gold
from strawberry_picker.data.crops import (
    LEGACY_POLICY,
    CropPolicy,
    NeighbourMode,
    crop_box,
    crop_fruit,
    neighbour_overlap,
    resolve_policy,
    with_neighbours,
)

# --- geometry ---------------------------------------------------------------------------


def test_gold_still_exports_the_same_crop_box() -> None:
    assert gold.crop_box is crop_box


def test_no_neighbours_means_no_overlap() -> None:
    assert neighbour_overlap((0, 0, 10, 10), (2, 2, 8, 8), []) == 0.0


def test_overlap_is_the_share_of_the_crop_a_neighbour_covers() -> None:
    # Neighbour covers the right half of the crop; the target sits in the left half.
    assert neighbour_overlap((0, 0, 10, 10), (0, 0, 5, 10), [(5, 0, 20, 10)]) == 0.5


def test_overlapping_neighbours_count_once() -> None:
    others = [(5, 0, 10, 10), (5, 0, 10, 10), (7, 0, 12, 10)]
    assert neighbour_overlap((0, 0, 10, 10), (0, 0, 5, 10), others) == 0.5


def test_neighbour_pixels_inside_the_target_box_are_not_counted() -> None:
    # Neighbour spans columns 3-10; columns 3-5 are inside the target box.
    assert neighbour_overlap((0, 0, 10, 10), (0, 0, 5, 10), [(3, 0, 10, 10)]) == 0.5


def test_neighbours_outside_the_crop_do_not_count() -> None:
    assert neighbour_overlap((0, 0, 10, 10), (2, 2, 8, 8), [(20, 20, 30, 30)]) == 0.0


def test_overlap_without_a_target_box_uses_the_whole_crop() -> None:
    assert neighbour_overlap((0, 0, 10, 10), None, [(0, 0, 10, 5)]) == 0.5


# --- policy -----------------------------------------------------------------------------


def test_default_policy_is_how_v0_1_was_cut() -> None:
    assert CropPolicy() == LEGACY_POLICY
    assert LEGACY_POLICY.padding == 0.15
    assert LEGACY_POLICY.neighbours is NeighbourMode.KEEP


def test_policy_from_data_yaml_settings() -> None:
    settings = {
        "crop_padding": 0.05,
        "crop_neighbours": {"mode": "drop", "max_overlap": 0.2, "fill": [1, 2, 3]},
    }
    policy = CropPolicy.from_settings(settings)
    assert policy == CropPolicy(
        padding=0.05, neighbours=NeighbourMode.DROP, max_neighbour_overlap=0.2, fill=(1, 2, 3)
    )
    assert CropPolicy.from_settings({"crop_padding": 0.15}) == LEGACY_POLICY


@pytest.mark.parametrize(
    "kwargs",
    [
        {"padding": -0.1},
        {"neighbours": "blur"},
        {"max_neighbour_overlap": 1.5},
        {"fill": (300, 0, 0)},
        {"fill": (1, 2)},
    ],
)
def test_invalid_policies_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        CropPolicy(**kwargs)


def test_policy_round_trips_through_json() -> None:
    policy = CropPolicy(padding=0.1, neighbours="mask", fill=(10, 20, 30))
    assert policy.neighbours is NeighbourMode.MASK
    assert CropPolicy.from_json(policy.to_json()) == policy
    assert json.loads(policy.to_json())["neighbours"] == "mask"


def test_drop_only_filters_training_crops() -> None:
    policy = CropPolicy(neighbours="drop", max_neighbour_overlap=0.1)
    assert policy.keeps(0.1) and not policy.keeps(0.11)
    assert CropPolicy(neighbours="mask").keeps(0.9)
    # A real berry can't be skipped at evaluation or inference: drop crops like keep there.
    assert policy.for_evaluation().neighbours is NeighbourMode.KEEP
    assert CropPolicy(neighbours="mask").for_evaluation().neighbours is NeighbourMode.MASK


# --- cropping ---------------------------------------------------------------------------


def _two_berries() -> Image.Image:
    image = Image.new("RGB", (40, 20), (200, 0, 0))  # left half: red berry
    image.paste((0, 200, 0), (20, 0, 40, 20))  # right half: green berry
    return image


def test_keep_crops_exactly_like_crop_box() -> None:
    image = _two_berries()
    crop = crop_fruit(image, (5, 5, 15, 15), [(22, 2, 38, 18)], CropPolicy(padding=0.5))
    expected = image.crop(crop_box((5, 5, 15, 15), 40, 20, padding=0.5))
    assert np.array_equal(np.asarray(crop), np.asarray(expected))


def test_mask_fills_neighbour_pixels_outside_the_target() -> None:
    image = _two_berries()
    policy = CropPolicy(padding=0.5, neighbours="mask", fill=(7, 7, 7))
    crop = np.asarray(crop_fruit(image, (6, 2, 18, 14), [(17, 2, 38, 18)], policy))
    # crop_box → (0, 0, 24, 20); the neighbour starts at x=17 but x<18 is the target's own box.
    assert crop.shape == (20, 24, 3)
    assert (crop[5, 20] == 7).all()  # neighbour pixel → filled
    assert (crop[5, 17] == (200, 0, 0)).all()  # inside the target box → untouched
    assert (crop[5, 2] == (200, 0, 0)).all()  # plain context → untouched
    assert (crop[16, 20] == 7).all()  # below the target box, still the neighbour
    assert (crop[19, 20] == (0, 200, 0)).all()  # below the neighbour box → untouched


def test_segment_keeps_only_the_target_mask() -> None:
    image = _two_berries()
    seen = []

    def fake_segment(img: Image.Image, bbox) -> np.ndarray:
        seen.append(tuple(bbox))
        mask = np.zeros((img.height, img.width), dtype=bool)
        mask[5:15, 5:15] = True
        return mask

    policy = CropPolicy(padding=0.5, neighbours="segment", fill=(9, 9, 9))
    crop = np.asarray(crop_fruit(image, (5, 5, 15, 15), [], policy, segment_fn=fake_segment))
    assert seen == [(5, 5, 15, 15)]
    assert (crop[0, 0] == 9).all()  # outside the mask
    assert (crop[crop.shape[0] // 2, crop.shape[1] // 2] == (200, 0, 0)).all()


def test_segment_needs_a_segmenter() -> None:
    with pytest.raises(ValueError, match="segment_fn"):
        crop_fruit(_two_berries(), (5, 5, 15, 15), [], CropPolicy(neighbours="segment"))


def test_segment_mask_must_match_the_image() -> None:
    def wrong_size(img, bbox):
        return np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="segment mask"):
        crop_fruit(
            _two_berries(), None, None, CropPolicy(neighbours="segment"), segment_fn=wrong_size
        )


def test_neighbour_lists_skip_whole_image_rows() -> None:
    assert neighbour_overlap((0, 0, 10, 10), None, float("nan")) == 0.0
    assert neighbour_overlap((0, 0, 10, 10), None, [None, float("nan")]) == 0.0


# --- neighbours per row -----------------------------------------------------------------


def test_with_neighbours_lists_the_other_boxes_of_the_same_image() -> None:
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "a", "c"],
            "bbox": [[0.0, 0, 1, 1], [2.0, 2, 3, 3], [0.0, 0, 5, 5], [4.0, 4, 5, 5], None],
        }
    )
    out = with_neighbours(manifest)
    assert [list(map(list, n)) for n in out.neighbour_bboxes] == [
        [[2, 2, 3, 3], [4, 4, 5, 5]],
        [[0, 0, 1, 1], [4, 4, 5, 5]],
        [],
        [[0, 0, 1, 1], [2, 2, 3, 3]],
        [],
    ]
    assert "neighbour_bboxes" not in manifest  # input untouched


# --- which policy a model was trained with ----------------------------------------------


def test_resolve_policy_prefers_the_run_config(tmp_data_dir) -> None:
    config = {"dataset": "maturity_cls_v0.9", "crop_policy": CropPolicy(padding=0.3).to_dict()}
    assert resolve_policy(config) == CropPolicy(padding=0.3)


def test_resolve_policy_reads_the_gold_release(tmp_data_dir) -> None:
    release = tmp_data_dir / "gold" / "maturity_cls_v0.2"
    release.mkdir(parents=True)
    (release / "crop_policy.json").write_text(CropPolicy(neighbours="mask").to_json())
    assert resolve_policy({"dataset": "maturity_cls_v0.2"}) == CropPolicy(neighbours="mask")


def test_resolve_policy_for_a_release_older_than_policies(tmp_data_dir) -> None:
    (tmp_data_dir / "gold" / "maturity_cls_v0.1").mkdir(parents=True)
    assert resolve_policy({"dataset": "maturity_cls_v0.1"}) == LEGACY_POLICY


def test_resolve_policy_falls_back_to_data_yaml(tmp_data_dir) -> None:
    from strawberry_picker.config import load_config

    expected = CropPolicy.from_settings(load_config("data")["gold"])
    assert resolve_policy(None) == expected
    assert resolve_policy({"dataset": "maturity_cls_v9.9"}) == expected
