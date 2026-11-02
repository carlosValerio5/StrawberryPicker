"""Near-duplicate detection with a difference hash (dHash).

Public datasets often ship their own augmented copies (flips, rotations, re-encodes)
next to the originals. If those land in different splits, test metrics are inflated,
so silver keeps one image per near-duplicate cluster. Hashing all eight flips and
rotations of each image makes the match invariant to those augmentations.

The hash is 256-bit (16×16), not the classic 64-bit (8×8). On small, blurry Riseholme
crops the 64-bit hash matched *different* berries, ripe with unripe included
(audit 2026-11-02, docs/datasets.md).
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


def dhash(image: Image.Image, hash_size: int = 16) -> int:
    """``hash_size²``-bit perceptual hash: sign of horizontal gradients on a grayscale thumbnail."""
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
    if not keys:
        return {}
    # Hashes wider than 64 bits are split into big-endian uint64 words.
    bits = max(h.bit_length() for key in keys for h in hashes[key])
    n_bytes = max(8, -(-bits // 64) * 8)

    def words(values: Sequence[int]) -> np.ndarray:
        raw = b"".join(v.to_bytes(n_bytes, "big") for v in values)
        return np.frombuffer(raw, dtype=">u8").astype(np.uint64).reshape(len(values), -1)

    canonical_keys: list[str] = []
    canonical = np.empty((len(keys), n_bytes // 8), dtype=np.uint64)
    duplicates: dict[str, str] = {}
    for key in keys:
        variants = words(hashes[key])
        n = len(canonical_keys)
        if n:
            xor = variants[:, None, :] ^ canonical[None, :n, :]
            distances = np.bitwise_count(xor).sum(axis=2).min(axis=0)
            best = int(distances.argmin())
            if distances[best] <= max_distance:
                duplicates[key] = canonical_keys[best]
                continue
        canonical[n] = variants[0]
        canonical_keys.append(key)
    return duplicates
