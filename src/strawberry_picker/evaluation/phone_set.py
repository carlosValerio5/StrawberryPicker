"""Ground truth and labeler agreement for the locked phone test set (ADR 0005, 0007).

The phone set never enters silver or gold, so no training release can contain it. The
evaluation reads its bronze pull directly. Item ids are ``<photo stem>_<n>``; the photo is
the bootstrap group.

Ground truth per crop:
- a label by the special labeler ``adjudicated`` wins;
- otherwise every labeler must agree and none may be unsure;
- "not a strawberry" crops are left out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strawberry_picker.evaluation.metrics import krippendorff_alpha
from strawberry_picker.labeling.tool import Label, current_labels
from strawberry_picker.schema import Maturity

ADJUDICATED = "adjudicated"
MATURITY_CODES = {m.value: m.ordinal for m in Maturity}


def _votes(labels: pd.DataFrame) -> dict[str, dict[str, Label]]:
    votes: dict[str, dict[str, Label]] = {}
    for (labeler, item_id), label in current_labels(labels.to_dict("records")).items():
        votes.setdefault(item_id, {})[labeler] = label
    return votes


def ground_truth(labels: pd.DataFrame) -> pd.DataFrame:
    """``item_id, maturity, defects, group, decided_by`` for every usable crop."""
    rows = []
    for item_id, by_labeler in _votes(labels).items():
        if ADJUDICATED in by_labeler:
            final, decided_by = by_labeler[ADJUDICATED], ADJUDICATED
        else:
            labels_ = list(by_labeler.values())
            verdicts = {(lab.maturity, lab.defects, lab.not_fruit) for lab in labels_}
            if len(verdicts) != 1 or any(lab.unsure for lab in labels_):
                continue
            final, decided_by = labels_[0], "+".join(sorted(by_labeler))
        if final.not_fruit or final.unsure:
            continue
        rows.append(
            (item_id, final.maturity, list(final.defects), item_id.rsplit("_", 1)[0], decided_by)
        )
    return pd.DataFrame(rows, columns=["item_id", "maturity", "defects", "group", "decided_by"])


def labeler_agreement(labels: pd.DataFrame) -> dict[str, Any]:
    """Ordinal Krippendorff's α on maturity over crops labeled by two or more labelers."""
    votes = _votes(labels)
    labelers = sorted({who for v in votes.values() for who in v} - {ADJUDICATED})
    shared = [
        item
        for item, by in votes.items()
        if sum(who in by for who in labelers) >= 2
        and not any(lab.unsure or lab.not_fruit for lab in by.values())
    ]
    ratings = np.full((len(labelers), len(shared)), np.nan)
    for j, item in enumerate(shared):
        for i, who in enumerate(labelers):
            if who in votes[item]:
                ratings[i, j] = MATURITY_CODES[votes[item][who].maturity]
    alpha = krippendorff_alpha(ratings, level="ordinal") if shared else float("nan")
    return {"labelers": labelers, "n_items": len(shared), "alpha_maturity": alpha}
