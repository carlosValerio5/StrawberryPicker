"""Human review labels, applied to silver on every rebuild.

Reviewers label crops with ``scripts/label.py``. Their append-only label files are landed
in bronze as pulls of ``human_review`` (``data/bronze/human_review/<date>[.<n>]/*.csv``).
A crop is identified by ``crop_id`` = ``<image_id>_<n>``, the same id gold and the audit use.

Rules:
- every labeler's latest label counts (undos are replayed);
- a review only changes a row when all its reviewers agree and none said "unsure";
- "not a strawberry" keeps the row but holds it out of gold. Rows are never deleted, so
  crop ids of the other fruit in the image never shift.
- "several berries" (``multi_fruit``, for whole-image rows that show a pile) holds the row
  out of gold too, but keeps the source's label: it may be right for every berry, and a
  later boxing pass can reuse it.
"""

from __future__ import annotations

import pandas as pd

from strawberry_picker.data.bronze import all_pulls
from strawberry_picker.labeling.tool import LABEL_COLUMNS, current_labels

REVIEW_SOURCE = "human_review"
RESOLVED_COLUMNS = (
    "item_id",
    "maturity",
    "defects",
    "not_fruit",
    "agreed",
    "reviewed_by",
    "multi_fruit",
)


def load_reviews() -> pd.DataFrame:
    """Every label row from every ``human_review`` pull, in pull order.

    Files written before a column existed (``multi_fruit``) read it as empty.
    """
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False).reindex(
            columns=list(LABEL_COLUMNS), fill_value=""
        )
        for pull in all_pulls(REVIEW_SOURCE)
        for path in sorted(pull.glob("*.csv"))
    ]
    if not frames:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def resolve_reviews(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per reviewed crop: the agreed label, or ``agreed=False``."""
    by_item: dict[str, list] = {}
    for (labeler, item_id), label in current_labels(rows.to_dict("records")).items():
        by_item.setdefault(item_id, []).append((labeler, label))
    resolved = []
    for item_id, votes in by_item.items():
        labels = [label for _, label in votes]
        verdicts = {(lab.maturity, lab.defects, lab.not_fruit, lab.multi_fruit) for lab in labels}
        agreed = len(verdicts) == 1 and not any(lab.unsure for lab in labels)
        first = labels[0]
        resolved.append(
            (
                item_id,
                first.maturity,
                list(first.defects),
                first.not_fruit,
                agreed,
                "+".join(sorted(labeler for labeler, _ in votes)),
                first.multi_fruit,
            )
        )
    return pd.DataFrame(resolved, columns=list(RESOLVED_COLUMNS))


def apply_reviews(manifest: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """Silver rows with agreed reviews applied; disputed or unsure rows held for review."""
    if resolved.empty:
        return manifest
    out = manifest.copy()
    crop_ids = out.image_id + "_" + out.groupby("image_id").cumcount().astype(str)
    reviews = resolved.set_index("item_id")
    maturity, defects = out.maturity.tolist(), out.defects.tolist()
    needs_review, reviewed_by = out.needs_review.tolist(), out.reviewed_by.tolist()
    for i, crop_id in enumerate(crop_ids):
        if crop_id not in reviews.index:
            continue
        review = reviews.loc[crop_id]
        reviewed_by[i] = review.reviewed_by
        if not review.agreed or review.multi_fruit:
            needs_review[i] = True
        elif review.not_fruit:
            maturity[i], defects[i], needs_review[i] = None, None, True
        else:
            maturity[i], defects[i], needs_review[i] = review.maturity, review.defects, False
    out["maturity"], out["defects"] = maturity, defects
    out["needs_review"], out["reviewed_by"] = needs_review, reviewed_by
    return out
