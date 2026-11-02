"""Load YAML configs from ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from strawberry_picker.paths import configs_dir


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a config by bare name (``"maturity_cls"`` → ``configs/maturity_cls.yaml``) or path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = configs_dir() / f"{path.name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f) or {}
