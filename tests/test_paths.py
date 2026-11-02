from __future__ import annotations

from datetime import date

import pytest

from strawberry_picker import paths
from strawberry_picker.data.bronze import source_dir
from strawberry_picker.data.gold import dataset_dir


def test_default_data_dir_is_repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAWBERRY_DATA_DIR", raising=False)
    assert paths.data_dir() == paths.PROJECT_ROOT / "data"


def test_every_layer_has_a_tracked_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRAWBERRY_DATA_DIR", raising=False)
    for layer in paths.Layer:
        assert (paths.layer_dir(layer) / "README.md").is_file(), layer


def test_env_override(tmp_data_dir) -> None:
    assert paths.layer_dir("silver") == tmp_data_dir / "silver"


def test_unknown_layer_rejected() -> None:
    with pytest.raises(ValueError):
        paths.layer_dir("platinum")


def test_bronze_source_dir(tmp_data_dir) -> None:
    assert source_dir("field_2026_11", date(2026, 11, 2)) == (
        tmp_data_dir / "bronze" / "field_2026_11" / "2026-11-02"
    )
    with pytest.raises(ValueError):
        source_dir("Field Photos", date(2026, 11, 2))


def test_gold_dataset_dir(tmp_data_dir) -> None:
    assert dataset_dir("maturity_cls_v1.0") == tmp_data_dir / "gold" / "maturity_cls_v1.0"
    with pytest.raises(ValueError):
        dataset_dir("maturity-latest")
