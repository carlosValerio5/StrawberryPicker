"""Near-duplicate detection with a difference hash (dHash).

Public datasets often ship their own augmented copies (flips, rotations, re-encodes)
next to the originals. If those land in different splits, test metrics are inflated,
so silver keeps one image per near-duplicate cluster. Hashing all eight flips and
rotations of each image makes the match invariant to those augmentations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

_DIHEDRAL = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit perceptual hash: sign of horizontal gradients on a tiny grayscale thumbnail."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dihedral_hashes(image: Image.Image) -> tuple[int, ...]:
    """dHash of all 8 flips/rotations; index 0 is the image as-is."""
    return tuple(dhash(image if op is None else image.transpose(op)) for op in _DIHEDRAL)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_near_duplicates(hashes: Mapping[str, Sequence[int]], max_distance: int) -> dict[str, str]:
    """Map each near-duplicate key to the canonical key it copies.

    Keys are visited in sorted order and the first member of a cluster is canonical,
    so results are deterministic. An image matches a canonical one when any of its
    dihedral variants is within ``max_distance`` bits of the canonical's identity hash.
    """
    keys = sorted(hashes)
    canonical_keys: list[str] = []
    canonical = np.empty(len(keys), dtype=np.uint64)
    duplicates: dict[str, str] = {}
    for key in keys:
        variants = np.array(hashes[key], dtype=np.uint64)
        n = len(canonical_keys)
        if n:
            distances = np.bitwise_count(variants[:, None] ^ canonical[None, :n]).min(axis=0)
            best = int(distances.argmin())
            if distances[best] <= max_distance:
                duplicates[key] = canonical_keys[best]
                continue
        canonical[n] = variants[0]
        canonical_keys.append(key)
    return duplicates
