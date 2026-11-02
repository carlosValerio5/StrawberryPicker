from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


def pattern_image(seed: int, size: tuple[int, int] = (64, 48)) -> Image.Image:
    """Smooth random pattern: distinct per seed, stable under JPEG and resizing."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize(size, Image.Resampling.BILINEAR)


@pytest.fixture
def make_image() -> Callable[..., Path]:
    """Write a synthetic image to ``path`` (format from the suffix) and return the path."""

    def _make(
        path: Path,
        seed: int = 0,
        size: tuple[int, int] = (64, 48),
        image: Image.Image | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        (image or pattern_image(seed, size)).save(path)
        return path

    return _make


@pytest.fixture
def make_spec() -> Callable[..., Any]:
    """Build a SourceSpec from a minimal raw registry entry plus overrides."""
    from strawberry_picker.data.sources import parse_source

    def _make(name: str = "toy_source", **overrides: Any):
        raw: dict[str, Any] = {
            "title": "Toy source",
            "url": "https://example.org/toy",
            "license": "CC-BY-4.0",
            "commercial_use": True,
            "format": "imagefolder",
            "labels": {"Ripe": {"maturity": "ripe", "defects": []}},
        }
        raw.update(overrides)
        return parse_source(name, raw)

    return _make


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the medallion root at an empty temp dir so tests never touch real data."""
    monkeypatch.setenv("STRAWBERRY_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def red_image() -> Image.Image:
    """Tiny synthetic 'ripe strawberry' — a solid red square."""
    return Image.new("RGB", (8, 8), (200, 20, 30))
