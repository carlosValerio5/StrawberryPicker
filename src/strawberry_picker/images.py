"""Image decoding shared by every stage: HEIC support and the one decode rule.

iPhones save HEIC by default. ``pi-heif`` (BSD-3-Clause, decode-only wheels) registers a
Pillow opener so ``Image.open`` reads ``.heic``/``.heif`` everywhere in the package; it is
registered when ``strawberry_picker`` is imported.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

HEIC_SUFFIXES = frozenset({".heic", ".heif"})


def register_heif() -> bool:
    """Teach Pillow to open HEIC/HEIF. Returns False if ``pi-heif`` isn't installed."""
    try:
        from pi_heif import register_heif_opener
    except ImportError:  # pragma: no cover - pi-heif is a core dependency
        return False
    register_heif_opener()
    return True


def open_rgb(path: Path) -> Image.Image:
    """Decode the way silver does: EXIF orientation applied, then RGB."""
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")
