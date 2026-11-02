from __future__ import annotations

from PIL import Image

from strawberry_picker.data.silver import compute_image_id


def test_image_id_is_deterministic_and_content_based(red_image: Image.Image) -> None:
    assert compute_image_id(red_image) == compute_image_id(red_image.copy())
    assert len(compute_image_id(red_image)) == 16


def test_image_id_ignores_color_mode(red_image: Image.Image) -> None:
    assert compute_image_id(red_image.convert("RGBA")) == compute_image_id(red_image)


def test_image_id_distinguishes_content_and_shape(red_image: Image.Image) -> None:
    green = Image.new("RGB", (8, 8), (40, 160, 40))
    reshaped = Image.new("RGB", (4, 16), (200, 20, 30))
    ids = {compute_image_id(img) for img in (red_image, green, reshaped)}
    assert len(ids) == 3
