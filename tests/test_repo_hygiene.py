"""Guard rails: data, weights and secrets must stay out of git; configs must load."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from strawberry_picker.config import load_config
from strawberry_picker.paths import PROJECT_ROOT

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists(),
    reason="needs a git checkout",
)


def _ignored(rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", rel_path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "rel_path",
    [
        "data/bronze/kaggle_x/2026-11-02/img_001.jpg",
        "data/bronze/kaggle_x/2026-11-02/SOURCE.md",
        "data/silver/manifest.parquet",
        "data/silver/images/a1b2c3d4e5f60718.jpg",
        "data/gold/maturity_cls_v1.0/train/ripe/a1b2.jpg",
        "data/gold/maturity_cls_v1.0/DATASET_CARD.md",
        "data/scratch.csv",
        "models/maturity_cls/run1/weights.pt",
        "reports/figures/confusion.png",
        "src/strawberry_picker/backbone.safetensors",
        ".env",
    ],
)
def test_artifacts_are_ignored(rel_path: str) -> None:
    assert _ignored(rel_path)


@pytest.mark.parametrize(
    "rel_path",
    [
        "data/README.md",
        "data/bronze/README.md",
        "data/silver/README.md",
        "data/gold/README.md",
        "models/README.md",
        "reports/README.md",
        ".env.example",
    ],
)
def test_structure_docs_are_tracked(rel_path: str) -> None:
    assert not _ignored(rel_path)


@pytest.mark.parametrize("name", ["data", "maturity_cls"])
def test_configs_load(name: str) -> None:
    assert load_config(name)
