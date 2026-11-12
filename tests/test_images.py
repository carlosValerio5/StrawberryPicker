from PIL import Image

import strawberry_picker  # noqa: F401  (registers the HEIC opener)
from strawberry_picker.data.readers import IMAGE_SUFFIXES
from strawberry_picker.images import HEIC_SUFFIXES, open_rgb, register_heif


def test_heic_opener_is_registered_on_import():
    assert register_heif()
    assert ".heic" in Image.registered_extensions()


def test_readers_pick_up_heic_files():
    assert HEIC_SUFFIXES <= IMAGE_SUFFIXES


def test_open_rgb_converts_to_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (4, 3), 128).save(path)
    image = open_rgb(path)
    assert image.mode == "RGB" and image.size == (4, 3)
