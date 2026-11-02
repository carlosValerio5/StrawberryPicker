from __future__ import annotations

import pytest

from strawberry_picker.paths import PROJECT_ROOT
from strawberry_picker.schema import (
    SILVER_COLUMNS,
    Defect,
    Grade,
    Maturity,
    grade_from_defects,
)

CONTRACT = (PROJECT_ROOT / "docs" / "data-contract.md").read_text()


@pytest.mark.parametrize(
    ("defects", "grade"),
    [
        ([], Grade.PREMIUM),
        (["bruise"], Grade.STANDARD),
        ([Defect.DEFORMED, Defect.PEST_DAMAGE], Grade.STANDARD),
        (["bruise", "mold"], Grade.REJECT),
        ([Defect.ROT], Grade.REJECT),
    ],
)
def test_grade_from_defects(defects, grade) -> None:
    assert grade_from_defects(defects) is grade


def test_unknown_defect_rejected() -> None:
    with pytest.raises(ValueError):
        grade_from_defects(["sunburnt"])


def test_maturity_is_ordinal() -> None:
    assert [m.ordinal for m in Maturity] == [0, 1, 2]
    assert Maturity.UNRIPE.ordinal < Maturity.RIPE.ordinal < Maturity.OVERRIPE.ordinal


@pytest.mark.parametrize("value", [*Maturity, *Defect, *Grade, *SILVER_COLUMNS])
def test_contract_doc_mentions_every_code_value(value: str) -> None:
    """schema.py and docs/data-contract.md must not drift apart."""
    assert f"`{value}`" in CONTRACT
