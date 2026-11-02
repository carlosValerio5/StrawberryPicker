"""Label taxonomy and manifest records — code mirror of ``docs/data-contract.md``.

Change this file and the contract in the same commit; ``tests/test_schema.py``
fails if they drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum

TAXONOMY_VERSION = "v0"


class Maturity(StrEnum):
    """Ordinal: declaration order is unripe < ripe < overripe."""

    UNRIPE = "unripe"
    RIPE = "ripe"
    OVERRIPE = "overripe"

    @property
    def ordinal(self) -> int:
        return list(Maturity).index(self)


class Defect(StrEnum):
    BRUISE = "bruise"
    MOLD = "mold"
    ROT = "rot"
    DEFORMED = "deformed"
    PEST_DAMAGE = "pest_damage"
    MECHANICAL_DAMAGE = "mechanical_damage"


class Grade(StrEnum):
    PREMIUM = "premium"
    STANDARD = "standard"
    REJECT = "reject"


REJECT_DEFECTS = frozenset({Defect.MOLD, Defect.ROT})


def grade_from_defects(defects: Iterable[Defect | str]) -> Grade:
    """Derive the quality grade from defect labels (data contract §2b)."""
    found = {Defect(d) for d in defects}
    if found & REJECT_DEFECTS:
        return Grade.REJECT
    if found:
        return Grade.STANDARD
    return Grade.PREMIUM


@dataclass(frozen=True, slots=True)
class SilverRecord:
    """One row of ``data/silver/manifest.parquet`` (data contract §3)."""

    image_id: str
    source_name: str
    bronze_path: str
    license: str
    width: int
    height: int
    maturity: Maturity | None
    defects: tuple[Defect, ...] | None  # () = no defects, None = not assessed
    bbox: tuple[float, float, float, float] | None
    group_id: str
    ingested_at: datetime


SILVER_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(SilverRecord))
GOLD_COLUMNS: tuple[str, ...] = (*SILVER_COLUMNS, "split")
