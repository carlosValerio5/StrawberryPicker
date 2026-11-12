"""Photo → per-berry predictions (ADR 0009), with fake detector, embedder and probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from strawberry_picker.data.crops import LEGACY_POLICY, CropPolicy, NeighbourMode
from strawberry_picker.inference import (
    BerryPrediction,
    LockedSetError,
    UnsupportedImageError,
    annotate,
    ensure_not_locked,
    image_paths,
    load_image,
    load_predictor,
    pipeline,
    predict_image,
    predict_paths,
    resolve_policy,
)
from strawberry_picker.models.calibration import TemperatureScaled, apply_temperature
from strawberry_picker.models.defect_probe import DefectHead
from strawberry_picker.models.linear_probe import predict_all_classes, save_run

CLASSES = ("unripe", "ripe", "overripe")


class FakeProbe:
    """Sklearn-like probe: trained on codes 0 and 1 only, like the real baseline."""

    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        ripe = features[:, 0:1]  # feature 0 = mean red, scaled to 0..1
        return np.hstack([1 - ripe, ripe])


def red_embed(crops: list[Image.Image]) -> np.ndarray:
    return np.stack([[np.asarray(c, dtype=float)[..., 0].mean() / 255.0] for c in crops])


def two_berry_photo() -> Image.Image:
    """Green berry on the left, red berry on the right, grey background."""
    image = Image.new("RGB", (200, 100), (128, 128, 128))
    image.paste((0, 200, 0), (10, 30, 50, 70))
    image.paste((255, 0, 0), (130, 30, 170, 70))
    return image


def fake_detect(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    boxes = [
        [130, 30, 170, 70],  # red berry
        [10, 30, 50, 70],  # green berry
        [131, 31, 171, 71],  # duplicate of the red one, lower score → NMS drops it
        [80, 0, 90, 10],  # background, below the score threshold
    ]
    return np.array(boxes, dtype=float), np.array([0.9, 0.8, 0.7, 0.05])


def test_predict_image_one_prediction_per_kept_box_best_first() -> None:
    predictions = predict_image(
        two_berry_photo(),
        detect_fn=fake_detect,
        embed_fn=red_embed,
        probe=FakeProbe(),
        classes=CLASSES,
        padding=0.0,
    )
    assert [p.bbox for p in predictions] == [(130, 30, 170, 70), (10, 30, 50, 70)]
    assert [p.score for p in predictions] == pytest.approx([0.9, 0.8])
    assert [p.maturity for p in predictions] == ["ripe", "unripe"]
    ripe = predictions[0].probabilities
    assert set(ripe) == set(CLASSES) and ripe["overripe"] == 0.0  # never-trained class
    assert sum(ripe.values()) == pytest.approx(1.0)


def test_predict_image_crops_with_the_given_padding() -> None:
    seen: list[tuple[int, int]] = []

    def embed(crops: list[Image.Image]) -> np.ndarray:
        seen.extend(c.size for c in crops)
        return red_embed(crops)

    predict_image(
        two_berry_photo(),
        detect_fn=fake_detect,
        embed_fn=embed,
        probe=FakeProbe(),
        classes=CLASSES,
        padding=0.25,
    )
    assert seen == [(60, 60), (60, 60)]  # 40 px box × (1 + 2 × 0.25)


def adjacent_photo_detector(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Two touching berries: the green box's padded crop reaches into the red one."""
    return np.array([[10, 30, 50, 70], [55, 30, 95, 70]], dtype=float), np.array([0.9, 0.8])


def adjacent_photo() -> Image.Image:
    image = Image.new("RGB", (200, 100), (128, 128, 128))
    image.paste((0, 200, 0), (10, 30, 50, 70))
    image.paste((255, 0, 0), (55, 30, 95, 70))
    return image


def test_predict_image_cuts_crops_with_the_policy_and_masks_neighbours() -> None:
    seen: list[Image.Image] = []

    def embed(crops: list[Image.Image]) -> np.ndarray:
        seen.extend(crops)
        return red_embed(crops)

    # padding 0.5: the green crop is image[10:90, 0:80], the red one image[10:90, 35:115]
    policy = CropPolicy(padding=0.5, neighbours=NeighbourMode.MASK, fill=(0, 0, 0))
    predict_image(
        adjacent_photo(),
        detect_fn=adjacent_photo_detector,
        embed_fn=embed,
        probe=FakeProbe(),
        classes=CLASSES,
        policy=policy,
    )
    green_crop, red_crop = seen
    assert green_crop.getpixel((30, 40)) == (0, 200, 0)  # own berry kept
    assert green_crop.getpixel((75, 40)) == (0, 0, 0)  # red neighbour painted over
    assert red_crop.getpixel((40, 40)) == (255, 0, 0)
    assert red_crop.getpixel((5, 40)) == (0, 0, 0)  # green neighbour painted over


def test_predict_image_keep_policy_leaves_neighbours_in_the_crop() -> None:
    seen: list[Image.Image] = []

    def embed(crops: list[Image.Image]) -> np.ndarray:
        seen.extend(crops)
        return red_embed(crops)

    predict_image(
        adjacent_photo(),
        detect_fn=adjacent_photo_detector,
        embed_fn=embed,
        probe=FakeProbe(),
        classes=CLASSES,
        policy=CropPolicy(padding=0.5),
    )
    assert seen[0].getpixel((75, 40)) == (255, 0, 0)


def test_predict_image_drop_policy_crops_every_berry_like_keep() -> None:
    crops: dict[str, list[bytes]] = {}
    for mode in (NeighbourMode.KEEP, NeighbourMode.DROP):
        out: list[Image.Image] = []

        def embed(images, out=out):
            out.extend(images)
            return red_embed(images)

        predict_image(
            adjacent_photo(),
            detect_fn=adjacent_photo_detector,
            embed_fn=embed,
            probe=FakeProbe(),
            classes=CLASSES,
            policy=CropPolicy(padding=0.5, neighbours=mode),
        )
        crops[mode.value] = [c.tobytes() for c in out]
    assert crops["drop"] == crops["keep"] and len(crops["keep"]) == 2


def test_predict_image_no_detections_skips_embedding() -> None:
    def embed(crops):  # pragma: no cover - must not be called
        raise AssertionError("embedded with no berries")

    def nothing(image):
        return np.zeros((0, 4)), np.zeros(0)

    out = predict_image(
        two_berry_photo(), detect_fn=nothing, embed_fn=embed, probe=FakeProbe(), classes=CLASSES
    )
    assert out == []


def test_predict_image_heads_add_fields_per_berry() -> None:
    """Extension point for the defect head: same embeddings, extra fields per berry."""

    def defect_head(features: np.ndarray) -> list[dict]:
        return [{"defects": [], "grade": "premium"} for _ in features]

    predictions = predict_image(
        two_berry_photo(),
        detect_fn=fake_detect,
        embed_fn=red_embed,
        probe=FakeProbe(),
        classes=CLASSES,
        heads=[defect_head],
    )
    assert all(p.extras == {"defects": [], "grade": "premium"} for p in predictions)
    record = predictions[0].to_dict()
    assert record["grade"] == "premium" and record["maturity"] == "ripe"
    json.dumps(record)  # JSON-serialisable


def test_predict_image_head_must_return_one_result_per_berry() -> None:
    with pytest.raises(ValueError, match="head"):
        predict_image(
            two_berry_photo(),
            detect_fn=fake_detect,
            embed_fn=red_embed,
            probe=FakeProbe(),
            classes=CLASSES,
            heads=[lambda features: [{}]],
        )


def test_load_image_applies_exif_orientation_and_rgb(tmp_path: Path) -> None:
    raw = Image.new("RGBA", (40, 20), (255, 0, 0, 128))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise to display
    path = tmp_path / "sideways.jpg"
    raw.convert("RGB").save(path, exif=exif)
    image = load_image(path)
    assert image.size == (20, 40) and image.mode == "RGB"

    png = tmp_path / "alpha.png"
    raw.save(png)
    assert load_image(png).mode == "RGB"


def test_load_image_decodes_heic_through_open_rgb(tmp_path: Path, monkeypatch) -> None:
    """HEIC goes through the shared decoder, like every other format (no rejection)."""
    decoded: list[Path] = []

    def fake_open_rgb(path: Path) -> Image.Image:
        decoded.append(Path(path))
        return Image.new("RGB", (4, 3), (1, 2, 3))

    monkeypatch.setattr(pipeline, "open_rgb", fake_open_rgb)
    path = tmp_path / "IMG_0001.HEIC"
    path.write_bytes(b"not decoded here")
    assert load_image(path).size == (4, 3)
    assert decoded == [path]


def test_load_image_corrupt_heic_gives_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "IMG_0002.HEIC"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic")
    with pytest.raises(UnsupportedImageError, match="IMG_0002.HEIC"):
        load_image(path)


def test_load_image_unreadable_file_gives_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(UnsupportedImageError, match="broken.jpg"):
        load_image(path)


def test_ensure_not_locked_refuses_the_phone_test_set(tmp_data_dir: Path) -> None:
    locked = tmp_data_dir / "bronze" / "phone_testset" / "2026-11-05" / "a.jpg"
    with pytest.raises(LockedSetError, match="evaluate_phone_set"):
        ensure_not_locked([locked])
    with pytest.raises(LockedSetError):
        ensure_not_locked([tmp_data_dir / "bronze"])  # a parent folder would include it
    ensure_not_locked([tmp_data_dir / "bronze" / "riseholme_2021", tmp_data_dir / "x.jpg"])


def test_ensure_not_locked_follows_symlinks(tmp_data_dir: Path, tmp_path: Path) -> None:
    locked = tmp_data_dir / "bronze" / "phone_testset"
    locked.mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(locked)
    with pytest.raises(LockedSetError):
        ensure_not_locked([link])


def test_image_paths_lists_folders_and_keeps_heic_for_reporting(tmp_path, make_image) -> None:
    a = make_image(tmp_path / "b" / "a.jpg")
    c = make_image(tmp_path / "c.png")
    heic = tmp_path / "d.heic"
    heic.write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    single = make_image(tmp_path / "single.jpg")
    assert image_paths([tmp_path]) == sorted([a, c, heic, single])
    assert image_paths([single]) == [single]
    with pytest.raises(FileNotFoundError):
        image_paths([tmp_path / "missing.jpg"])


def write_run(run_dir: Path, config: dict) -> Path:
    joblib = pytest.importorskip("joblib")
    run_dir.mkdir(parents=True)
    joblib.dump(FakeProbe(), run_dir / "probe.joblib")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return run_dir


def test_resolve_policy_prefers_run_config_then_gold_release(tmp_data_dir: Path) -> None:
    explicit = {"crop_policy": {"padding": 0.05, "neighbours": "mask"}}
    assert resolve_policy(explicit) == CropPolicy(padding=0.05, neighbours=NeighbourMode.MASK)
    gold = tmp_data_dir / "gold" / "maturity_cls_v0.2"
    gold.mkdir(parents=True)
    (gold / "crop_policy.json").write_text(CropPolicy(padding=0.2).to_json())
    assert resolve_policy({"dataset": "maturity_cls_v0.2"}).padding == 0.2
    # a release from before ADR 0011 has no policy file: it was cut with the legacy policy
    old = tmp_data_dir / "gold" / "maturity_cls_v0.1"
    old.mkdir(parents=True)
    assert resolve_policy({"dataset": "maturity_cls_v0.1"}) == LEGACY_POLICY
    assert isinstance(resolve_policy({}), CropPolicy)  # no run, no release: configs/data.yaml


def test_load_predictor_reads_the_run(tmp_path: Path, tmp_data_dir: Path) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_padding": 0.0},
    )
    built: list[str] = []

    def embed_factory(backbone: str):
        built.append(backbone)
        return red_embed

    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=embed_factory)
    assert built == ["toy"] and predictor.classes == CLASSES
    assert predictor.policy.padding == 0.0
    assert [p.maturity for p in predictor.predict(two_berry_photo())] == ["ripe", "unripe"]


def test_predict_paths_reports_errors_per_image(tmp_path: Path, tmp_data_dir: Path) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_padding": 0.0},
    )
    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=lambda _: red_embed)
    photo = tmp_path / "photos" / "punnet.jpg"
    photo.parent.mkdir()
    two_berry_photo().save(photo, quality=95)
    (tmp_path / "photos" / "iphone.heic").write_bytes(b"x")
    annotated = tmp_path / "annotated"

    records = predict_paths(image_paths([photo.parent]), predictor, annotate_dir=annotated)
    by_name = {Path(r["image"]).name: r for r in records}
    assert (
        by_name["iphone.heic"]["berries"] == [] and "iphone.heic" in by_name["iphone.heic"]["error"]
    )
    berries = by_name["punnet.jpg"]["berries"]
    assert [b["maturity"] for b in berries] == ["ripe", "unripe"]
    assert (annotated / "punnet.jpg").exists() and not (annotated / "iphone.jpg").exists()
    json.dumps(records)


def test_annotate_draws_boxes_without_touching_the_input() -> None:
    photo = two_berry_photo()
    before = photo.tobytes()
    prediction = BerryPrediction(
        bbox=(10, 30, 50, 70), score=0.8, maturity="unripe", probabilities={"unripe": 1.0}
    )
    out = annotate(photo, [prediction])
    assert photo.tobytes() == before and out.size == photo.size and out.tobytes() != before


def test_load_predictor_applies_the_run_calibration(tmp_path: Path, tmp_data_dir: Path) -> None:
    """calibration.json (ADR 0005 §B3) makes the loaded probe temperature-scaled."""
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_policy": {"padding": 0.0}},
    )
    (run / "calibration.json").write_text(json.dumps({"method": "temperature", "temperature": 4.0}))
    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=lambda _: red_embed)
    assert isinstance(predictor.probe, TemperatureScaled) and predictor.probe.temperature == 4.0

    features = np.array([[0.9]])
    raw = FakeProbe().predict_proba(features)
    expected = apply_temperature(raw, 4.0)
    got = predict_all_classes(predictor.probe, features, len(CLASSES))
    assert got[0, 1] == pytest.approx(expected[0, 1]) and 0.5 < got[0, 1] < 0.9  # softened


def test_load_predictor_without_calibration_file_keeps_raw_probabilities(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_policy": {"padding": 0.0}},
    )
    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=lambda _: red_embed)
    assert not isinstance(predictor.probe, TemperatureScaled)


def fake_defect_head(tmp_path: Path, *, backbone: str = "toy", padding: float = 0.0) -> Path:
    """A real ``DefectHead`` saved the way train_quality.py saves it, around the fake probe."""
    head = DefectHead(
        defects=("rot",),
        probes={"rot": FakeProbe()},
        thresholds={"rot": 0.5},
    )
    config = {
        "task": "quality_cls",
        "defects": ["rot"],
        "model": {"backbone": backbone},
        "crop_policy": {"padding": padding},
    }
    return save_run(tmp_path / "models" / "quality_cls" / "20261112T000000Z", head, config, {})


def test_load_predictor_adds_defects_and_grade_from_the_quality_run(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_policy": {"padding": 0.0}},
    )
    quality = fake_defect_head(tmp_path)
    predictor = load_predictor(
        run, quality_run=quality, detect_fn=fake_detect, embed_factory=lambda _: red_embed
    )
    berries = predictor.predict(two_berry_photo())
    # the red berry is ripe and rotten (reject); the green one is unripe and clean (premium)
    assert [(b.extras["defects"], b.extras["grade"]) for b in berries] == [
        (["rot"], "reject"),
        ([], "premium"),
    ]
    record = berries[0].to_dict()
    assert record["maturity"] == "ripe" and record["grade"] == "reject"
    json.dumps(record)


def test_load_predictor_without_quality_run_adds_no_defect_fields(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_policy": {"padding": 0.0}},
    )
    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=lambda _: red_embed)
    assert all(b.extras == {} for b in predictor.predict(two_berry_photo()))


def test_quality_run_must_share_the_maturity_backbone_and_crops(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_policy": {"padding": 0.0}},
    )
    other_backbone = fake_defect_head(tmp_path / "a", backbone="other")
    with pytest.raises(ValueError, match="backbone"):
        load_predictor(
            run,
            quality_run=other_backbone,
            detect_fn=fake_detect,
            embed_factory=lambda _: red_embed,
        )
    other_crops = fake_defect_head(tmp_path / "b", padding=0.2)
    with pytest.raises(ValueError, match="crop policy"):
        load_predictor(
            run, quality_run=other_crops, detect_fn=fake_detect, embed_factory=lambda _: red_embed
        )


def test_legacy_crop_padding_in_run_config_is_still_honoured(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    run = write_run(
        tmp_path / "run",
        {"classes": list(CLASSES), "model": {"backbone": "toy"}, "crop_padding": 0.3},
    )
    predictor = load_predictor(run, detect_fn=fake_detect, embed_factory=lambda _: red_embed)
    assert predictor.policy.padding == 0.3
