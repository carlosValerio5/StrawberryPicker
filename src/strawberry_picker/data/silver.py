"""Silver layer: decode, dedupe and unify bronze sources. Contract: ``data/silver/README.md``."""

from __future__ import annotations

import hashlib

from PIL import Image, ImageOps


def compute_image_id(image: Image.Image) -> str:
    """Content-addressed id: sha256 over size + EXIF-corrected RGB pixels, first 16 hex chars.

    Identical photos from different sources (or re-encoded copies with the same
    pixels) collapse to one id, which is what deduplication keys on.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()[:16]
