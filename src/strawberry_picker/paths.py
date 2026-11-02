"""Single source of truth for filesystem locations.

Every module that reads or writes data, weights or reports resolves its path here.
Roots are read at call time so ``STRAWBERRY_DATA_DIR`` / ``STRAWBERRY_MODELS_DIR``
overrides (and test monkeypatching) take effect without re-importing.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

# src/strawberry_picker/paths.py -> repo root (valid for the editable install `uv sync` makes)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Layer(StrEnum):
    """Medallion data layers, in pipeline order."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


def data_dir() -> Path:
    """Root of the medallion data (``STRAWBERRY_DATA_DIR`` or ``<repo>/data``)."""
    return Path(os.environ.get("STRAWBERRY_DATA_DIR") or PROJECT_ROOT / "data")


def layer_dir(layer: Layer | str) -> Path:
    """Directory of one medallion layer. Raises ``ValueError`` for unknown layers."""
    return data_dir() / Layer(layer).value


def models_dir() -> Path:
    """Root for trained weights (``STRAWBERRY_MODELS_DIR`` or ``<repo>/models``)."""
    return Path(os.environ.get("STRAWBERRY_MODELS_DIR") or PROJECT_ROOT / "models")


def reports_dir() -> Path:
    return PROJECT_ROOT / "reports"


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"
