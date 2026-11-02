from __future__ import annotations

import pytest

from strawberry_picker.data.sources import (
    AnnotationFormat,
    GroupBy,
    LabelMapping,
    load_sources,
    parse_source,
)
from strawberry_picker.schema import Defect, Maturity


def test_label_mapping_is_typed(make_spec) -> None:
    spec = make_spec(
        labels={
            "Ripe": {"maturity": "ripe", "defects": []},
            "Rotten": {"maturity": "overripe", "defects": ["rot"], "needs_review": True},
        }
    )
    assert spec.map_label("Ripe") == LabelMapping(maturity=Maturity.RIPE, defects=())
    rotten = spec.map_label("Rotten")
    assert rotten.maturity is Maturity.OVERRIPE
    assert rotten.defects == (Defect.ROT,)
    assert rotten.needs_review


def test_missing_defects_means_not_assessed(make_spec) -> None:
    spec = make_spec(labels={"Ripe": {"maturity": "ripe"}})
    assert spec.map_label("Ripe").defects is None


def test_drop_label_maps_to_none(make_spec) -> None:
    spec = make_spec(labels={"Occluded": "drop"})
    assert spec.map_label("Occluded") is None


def test_unknown_label_is_a_config_error(make_spec) -> None:
    spec = make_spec()
    with pytest.raises(ValueError, match="sources.yaml"):
        spec.map_label("Mystery")


def test_unknown_label_dropped_when_configured(make_spec) -> None:
    spec = make_spec(unmapped="drop")
    assert spec.map_label("FreshApple") is None


@pytest.mark.parametrize(
    "labels",
    [
        {"Ripe": {"maturity": "turning"}},
        {"Ripe": {"defects": ["sunburn"]}},
        {"Ripe": "keep"},
    ],
)
def test_invalid_label_values_rejected(make_spec, labels) -> None:
    with pytest.raises(ValueError):
        make_spec(labels=labels)


def test_yolo_requires_class_names(make_spec) -> None:
    with pytest.raises(ValueError, match="class_names"):
        make_spec(format="yolo")
    spec = make_spec(format="yolo", class_names=["Ripe"])
    assert spec.format is AnnotationFormat.YOLO
    assert spec.class_names == ("Ripe",)


@pytest.mark.parametrize(
    "overrides",
    [{"format": "voc"}, {"group_by": "plant"}, {"unmapped": "guess"}, {"license": ""}],
)
def test_invalid_fields_rejected(make_spec, overrides) -> None:
    with pytest.raises(ValueError):
        make_spec(**overrides)


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValueError, match="url"):
        parse_source("toy", {"title": "t", "license": "x", "format": "imagefolder", "labels": {}})


def test_bad_source_name_rejected(make_spec) -> None:
    with pytest.raises(ValueError):
        make_spec(name="Toy Source")


def test_defaults(make_spec) -> None:
    spec = make_spec()
    assert spec.enabled
    assert spec.root == "."
    assert spec.group_by is GroupBy.IMAGE
    assert spec.exclude == ()


def test_repo_registry_is_valid() -> None:
    sources = load_sources()
    assert "riseholme_2021" in sources
    for name, spec in sources.items():
        assert spec.name == name
        assert spec.license
        assert spec.url.startswith("https://")
        assert spec.labels, name


def test_repo_registry_only_enables_verified_sources() -> None:
    # Each entry was checked against its downloaded files and contact sheets (docs/datasets.md).
    enabled = {name for name, spec in load_sources().items() if spec.enabled}
    assert enabled == {"riseholme_2021", "strawberry_ds_2023", "strawberry_disease_2025"}
