from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the medallion root at an empty temp dir so tests never touch real data."""
    monkeypatch.setenv("STRAWBERRY_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def red_image() -> Image.Image:
    """Tiny synthetic 'ripe strawberry' — a solid red square."""
    return Image.new("RGB", (8, 8), (200, 20, 30))
