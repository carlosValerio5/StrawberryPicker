"""Bronze layer: land raw sources untouched. Contract: ``data/bronze/README.md``."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from strawberry_picker.paths import Layer, layer_dir

_SOURCE_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def source_dir(source_name: str, ingest_date: date) -> Path:
    """``data/bronze/<source_name>/<YYYY-MM-DD>/`` — where one pull of a source lands."""
    if not _SOURCE_NAME.match(source_name):
        raise ValueError(f"source_name must be snake_case, got {source_name!r}")
    return layer_dir(Layer.BRONZE) / source_name / ingest_date.isoformat()
