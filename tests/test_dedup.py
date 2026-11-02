from __future__ import annotations

import io

from PIL import Image

from conftest import pattern_image
from strawberry_picker.data.dedup import dhash, dihedral_hashes, find_near_duplicates, hamming


def _jpeg_roundtrip(image: Image.Image, quality: int = 70) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue()))


def test_dhash_is_deterministic_64_bit() -> None:
    image = pattern_image(1)
    assert dhash(image) == dhash(image.copy())
    assert 0 <= dhash(image) < 2**64


def test_hamming() -> None:
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(5, 5) == 0


def test_recompressed_and_resized_copies_stay_close() -> None:
    image = pattern_image(1, size=(128, 96))
    assert hamming(dhash(image), dhash(_jpeg_roundtrip(image))) <= 6
    assert hamming(dhash(image), dhash(image.resize((64, 48)))) <= 6


def test_different_images_are_far_apart() -> None:
    assert hamming(dhash(pattern_image(1)), dhash(pattern_image(2))) > 12


def test_dihedral_hashes_start_with_identity() -> None:
    image = pattern_image(4)
    variants = dihedral_hashes(image)
    assert len(variants) == 8
    assert variants[0] == dhash(image)


def test_rotated_and_flipped_copies_are_near_duplicates() -> None:
    original = pattern_image(1)
    hashes = {
        "a_original": dihedral_hashes(original),
        "b_rot90": dihedral_hashes(original.transpose(Image.Transpose.ROTATE_90)),
        "c_flipped": dihedral_hashes(original.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        "d_other": dihedral_hashes(pattern_image(2)),
    }
    assert find_near_duplicates(hashes, max_distance=6) == {
        "b_rot90": "a_original",
        "c_flipped": "a_original",
    }


def test_canonical_is_first_key_in_sorted_order() -> None:
    image = pattern_image(3)
    hashes = {"z_copy": dihedral_hashes(image), "a_first": dihedral_hashes(image)}
    assert find_near_duplicates(hashes, max_distance=0) == {"z_copy": "a_first"}


def test_no_duplicates_in_empty_or_unique_sets() -> None:
    assert find_near_duplicates({}, max_distance=6) == {}
    unique = {f"k{i}": dihedral_hashes(pattern_image(i)) for i in range(5)}
    assert find_near_duplicates(unique, max_distance=6) == {}
