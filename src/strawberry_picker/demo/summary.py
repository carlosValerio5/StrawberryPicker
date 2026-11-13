"""Batch verdict for one photo: counts, pack-out and plain-English reasons. No ML here.

Input is a list of ``BerryPrediction.to_dict()`` records (``bbox``, ``score``, ``maturity``,
``probabilities``, and ``defects``/``grade`` when the defect head ran).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from strawberry_picker.schema import REJECT_DEFECTS, Defect, Grade, Maturity, grade_from_defects

# How a defect reads in a sentence: "1 berry shows <phrase>".
DEFECT_PHRASES: dict[Defect, str] = {
    Defect.BRUISE: "bruising",
    Defect.MOLD: "mold",
    Defect.ROT: "rot",
    Defect.DEFORMED: "an irregular shape",
    Defect.PEST_DAMAGE: "pest damage",
    Defect.MECHANICAL_DAMAGE: "cuts or crush damage",
}


def berry_grade(berry: Mapping[str, Any]) -> Grade | None:
    """The berry's grade; derived from ``defects`` when only those are present."""
    if berry.get("grade"):
        return Grade(berry["grade"])
    if berry.get("defects") is not None:
        return grade_from_defects(berry["defects"])
    return None


def _berries(n: int) -> str:
    return f"{n} berry" if n == 1 else f"{n} berries"


def _verb(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def summarize(berries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per maturity, grade and defect, the verdict, and why it fired."""
    count = len(berries)
    maturity = Counter(Maturity(b["maturity"]) for b in berries)
    grades = [berry_grade(b) for b in berries]
    graded = any(g is not None for g in grades)
    grade_counts = Counter(g for g in grades if g is not None)
    defects = Counter(Defect(d) for b in berries for d in b.get("defects") or ())
    order = list(Defect)
    ranked = sorted(defects.items(), key=lambda item: (-item[1], order.index(item[0])))

    ready = sum(
        Maturity(b["maturity"]) is Maturity.RIPE and g is not Grade.REJECT
        for b, g in zip(berries, grades, strict=True)
    )
    packout = grade_counts[Grade.PREMIUM] / count if graded and count else None

    if count == 0:
        verdict = "empty"
    elif grade_counts[Grade.REJECT]:
        verdict = Grade.REJECT.value
    elif grade_counts[Grade.STANDARD] or maturity[Maturity.UNRIPE] or maturity[Maturity.OVERRIPE]:
        verdict = Grade.STANDARD.value
    else:
        verdict = Grade.PREMIUM.value

    return {
        "count": count,
        "maturity": {m.value: maturity[m] for m in Maturity},
        "grade": {g.value: grade_counts[g] for g in Grade} if graded else None,
        "defects": {d.value: n for d, n in ranked},
        "ready": ready,
        "packout": packout,
        "verdict": verdict,
        "reasons": _reasons(count, maturity, ranked, graded, verdict),
    }


def _reasons(
    count: int,
    maturity: Counter[Maturity],
    ranked: list[tuple[Defect, int]],
    graded: bool,
    verdict: str,
) -> list[str]:
    """Most severe first: reject defects, then ripeness, then the other defects."""
    if verdict == "empty":
        return ["No strawberries found in this photo"]
    if verdict == Grade.PREMIUM.value:
        quality = " with no visible defects" if graded else ""
        if count == 1:
            return [f"The berry is ripe{quality}"]
        return [f"All {count} berries are ripe{quality}"]

    def shows(defect: Defect, n: int) -> str:
        return f"{_berries(n)} {_verb(n, 'shows', 'show')} {DEFECT_PHRASES[defect]}"

    reasons = [shows(d, n) for d, n in ranked if d in REJECT_DEFECTS]
    if n := maturity[Maturity.UNRIPE]:
        reasons.append(f"{_berries(n)} {_verb(n, 'is', 'are')} not yet ripe")
    if n := maturity[Maturity.OVERRIPE]:
        reasons.append(f"{_berries(n)} {_verb(n, 'is', 'are')} past peak ripeness")
    reasons += [shows(d, n) for d, n in ranked if d not in REJECT_DEFECTS]
    return reasons
