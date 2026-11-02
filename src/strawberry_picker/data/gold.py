"""Gold layer: versioned, group-aware train/val/test datasets. Contract: ``data/gold/README.md``."""

from __future__ import annotations

import re
from pathlib import Path

from strawberry_picker.paths import Layer, layer_dir

_VERSION_NAME = re.compile(r"^[a-z]+_(cls|det|seg)_v\d+\.\d+$")


def dataset_dir(name: str) -> Path:
    """``data/gold/<task>_v<MAJOR>.<MINOR>/``, e.g. ``maturity_cls_v1.0``."""
    if not _VERSION_NAME.match(name):
        raise ValueError(f"gold dataset name must look like 'maturity_cls_v1.0', got {name!r}")
    return layer_dir(Layer.GOLD) / name
