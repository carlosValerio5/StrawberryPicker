"""Source registry: what each bronze source is and how its labels map to our taxonomy.

The registry lives in ``configs/sources.yaml``; this module parses and validates it so
that a typo in a label mapping fails at load time, not halfway through a silver build.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from strawberry_picker.config import load_config
from strawberry_picker.data.bronze import validate_source_name
from strawberry_picker.schema import Defect, Maturity

DROP = "drop"


class AnnotationFormat(StrEnum):
    IMAGEFOLDER = "imagefolder"  # <root>/<label path>/<image>
    YOLO = "yolo"  # image + same-stem .txt (boxes or polygons, normalized)
    COCO = "coco"  # COCO json with bbox annotations


class GroupBy(StrEnum):
    """What counts as one leakage group when splitting gold datasets."""

    IMAGE = "image"  # each image is independent
    SOURCE = "source"  # the whole source stays in one split
    PARENT_DIR = "parent_dir"  # images sharing a folder (session, plant) stay together


@dataclass(frozen=True, slots=True)
class LabelMapping:
    maturity: Maturity | None = None
    defects: tuple[Defect, ...] | None = None  # () = assessed, none found; None = not assessed
    needs_review: bool = False


@dataclass(frozen=True, slots=True)
class BoxProposals:
    """Machine-proposed fruit boxes for a source whose labels cover the whole image.

    Proposals live in ``data/bronze/<source>/<date>/<source name>.csv``. An image with
    exactly one box scoring >= ``min_score`` is cropped to it; several (a pile) or none
    hold the image out of gold, because its one label can't be pinned to one berry.
    """

    source: str
    min_score: float


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    title: str
    url: str
    license: str
    commercial_use: bool
    format: AnnotationFormat
    labels: Mapping[str, LabelMapping | None]  # None = drop
    enabled: bool = True
    drop_unmapped: bool = False
    root: str = "."  # where annotations start, relative to the pull dir
    class_names: tuple[str, ...] = ()  # YOLO class index → source label
    group_by: GroupBy = GroupBy.IMAGE
    exclude: tuple[str, ...] = ()  # globs relative to the pull dir
    box_proposals: BoxProposals | None = None
    doi: str | None = None
    citation: str = ""
    notes: str = field(default="", compare=False)

    def map_label(self, source_label: str) -> LabelMapping | None:
        """Our labels for ``source_label``, or ``None`` if the row should be dropped."""
        if source_label in self.labels:
            return self.labels[source_label]
        if self.drop_unmapped:
            return None
        raise ValueError(
            f"{self.name}: label {source_label!r} has no mapping in configs/sources.yaml"
        )


def _parse_mapping(source: str, label: str, raw: Any) -> LabelMapping | None:
    if raw == DROP:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: mapping for {label!r} must be a table or 'drop', got {raw!r}")
    maturity = raw.get("maturity")
    defects = raw.get("defects")
    return LabelMapping(
        maturity=None if maturity is None else Maturity(maturity),
        defects=None if defects is None else tuple(Defect(d) for d in defects),
        needs_review=bool(raw.get("needs_review", False)),
    )


def _parse_box_proposals(source: str, raw: Any) -> BoxProposals | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or "source" not in raw or "min_score" not in raw:
        raise ValueError(f"{source}: box_proposals needs 'source' and 'min_score'")
    return BoxProposals(validate_source_name(str(raw["source"])), float(raw["min_score"]))


def parse_source(name: str, raw: Mapping[str, Any]) -> SourceSpec:
    """Validate one registry entry. Raises ``ValueError`` on anything malformed."""
    validate_source_name(name)
    missing = [key for key in ("title", "url", "license", "format", "labels") if key not in raw]
    if missing:
        raise ValueError(f"{name}: missing required field(s) {', '.join(missing)}")
    if not raw["license"]:
        raise ValueError(f"{name}: license must not be empty (write 'unknown' if unknown)")

    unmapped = raw.get("unmapped", "error")
    if unmapped not in ("error", DROP):
        raise ValueError(f"{name}: unmapped must be 'error' or 'drop', got {unmapped!r}")

    spec = SourceSpec(
        name=name,
        title=raw["title"],
        url=raw["url"],
        license=raw["license"],
        commercial_use=bool(raw.get("commercial_use", False)),
        format=AnnotationFormat(raw["format"]),
        labels={str(k): _parse_mapping(name, str(k), v) for k, v in raw["labels"].items()},
        enabled=bool(raw.get("enabled", True)),
        drop_unmapped=unmapped == DROP,
        root=raw.get("root", "."),
        class_names=tuple(raw.get("class_names", ())),
        group_by=GroupBy(raw.get("group_by", GroupBy.IMAGE)),
        exclude=tuple(raw.get("exclude", ())),
        box_proposals=_parse_box_proposals(name, raw.get("box_proposals")),
        doi=raw.get("doi"),
        citation=raw.get("citation", ""),
        notes=raw.get("notes", ""),
    )
    if spec.format is AnnotationFormat.YOLO and not spec.class_names:
        raise ValueError(f"{name}: YOLO sources need class_names (index → label)")
    return spec


def load_sources(path: str | Path = "sources") -> dict[str, SourceSpec]:
    """Load and validate the whole registry (default ``configs/sources.yaml``)."""
    raw = load_config(path).get("sources", {})
    return {name: parse_source(name, entry) for name, entry in raw.items()}
