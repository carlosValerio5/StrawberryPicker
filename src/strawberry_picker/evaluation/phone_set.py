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

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from strawberry_picker.data.gold import crop_box
from strawberry_picker.evaluation.maturity import evaluate_maturity
from strawberry_picker.evaluation.metrics import bootstrap_ci, krippendorff_alpha
from strawberry_picker.labeling.tool import LABEL_COLUMNS, Label, current_labels, load_task
from strawberry_picker.models.linear_probe import predict_all_classes
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


# ADR 0005 bars (amended by ADR 0007).
MIN_LABELER_ALPHA = 0.67
MIN_RECALL = 0.70
MAX_ECE = 0.10


def load_pull_labels(pull_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(p, dtype=str, keep_default_na=False)
        for p in sorted(pull_dir.glob("labels*.csv"))
    ]
    return (
        pd.concat(frames, ignore_index=True)[list(LABEL_COLUMNS)]
        if frames
        else pd.DataFrame(columns=list(LABEL_COLUMNS))
    )


def _alpha_ci(ratings: np.ndarray, groups: np.ndarray, n_resamples: int, seed: int):
    def alpha(index: np.ndarray) -> float:
        return krippendorff_alpha(ratings[:, index], level="ordinal")

    return bootstrap_ci(alpha, groups, n_resamples=n_resamples, seed=seed)


def evaluate_pull(
    pull_dir: Path,
    probe: Any,
    embed_fn: Callable[[list[Image.Image]], np.ndarray],
    classes: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score ``probe`` on a labeled phone pull and check every ADR 0005 bar."""
    task = load_task(pull_dir / "task.csv").set_index("item_id")
    labels = load_pull_labels(pull_dir)
    truth = ground_truth(labels)
    crops = []
    for item_id in truth.item_id:
        row = task.loc[item_id]
        with Image.open(row.image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        crops.append(image.crop(crop_box(json.loads(row.bbox) if row.bbox else None, *image.size)))
    probs = predict_all_classes(probe, embed_fn(crops), len(classes))
    codes = truth.maturity.map({name: k for k, name in enumerate(classes)}).to_numpy()
    groups = truth.group.to_numpy()
    metrics = evaluate_maturity(probs, codes, groups, classes, n_resamples=n_resamples, seed=seed)
    predicted = dict(zip(truth.item_id, probs.argmax(axis=1), strict=True))

    agreement = labeler_agreement(labels)
    votes = _votes(labels)
    labelers = agreement["labelers"]
    checks = []
    alpha = agreement["alpha_maturity"]
    checks.append(
        {
            "id": "A",
            "rule": f"labeler alpha >= {MIN_LABELER_ALPHA}",
            "value": alpha,
            "passed": None if agreement["n_items"] == 0 else bool(alpha >= MIN_LABELER_ALPHA),
        }
    )

    b1: dict[str, Any] = {
        "id": "B1",
        "rule": "alpha(model, each labeler) >= low CI of alpha(labelers)",
    }
    if len(labelers) >= 2 and agreement["n_items"]:
        shared = [
            i
            for i, by in votes.items()
            if all(w in by for w in labelers[:2])
            and i in predicted
            and not any(v.unsure or v.not_fruit for v in by.values())
        ]
        ratings = np.array(
            [[MATURITY_CODES[votes[i][w].maturity] for i in shared] for w in labelers[:2]],
            dtype=float,
        )
        _, low, _ = _alpha_ci(
            ratings, np.array([i.rsplit("_", 1)[0] for i in shared]), n_resamples, seed
        )
        model_alpha = {}
        for who in labelers:
            items = [i for i in predicted if who in votes.get(i, {}) and votes[i][who].maturity]
            pair = np.array(
                [
                    [MATURITY_CODES[votes[i][who].maturity] for i in items],
                    [predicted[i] for i in items],
                ],
                dtype=float,
            )
            model_alpha[who] = krippendorff_alpha(pair, level="ordinal")
        b1 |= {
            "value": {"labeler_alpha_low": low, "model_alpha": model_alpha},
            "passed": bool(all(a >= low for a in model_alpha.values())),
        }
    else:
        b1 |= {"value": None, "passed": None}
    checks.append(b1)

    recalls = [v["point"] for v in metrics["recall"].values()]
    present = [r for r in recalls if r is not None]
    failed = any(r < MIN_RECALL for r in present)
    checks.append(
        {
            "id": "B2",
            "rule": f"recall >= {MIN_RECALL} for every class",
            "value": metrics["recall"],
            "passed": False if failed else (None if None in recalls else True),
        }
    )
    checks.append(
        {
            "id": "B3",
            "rule": f"ECE <= {MAX_ECE}",
            "value": metrics["ece"],
            "passed": bool(metrics["ece"] <= MAX_ECE),
        }
    )
    checks.append(
        {
            "id": "C",
            "rule": "reject (mold/rot) recall >= 0.80",
            "value": None,
            "passed": None,
            "note": "no defect model yet",
        }
    )

    results = [c["passed"] for c in checks]
    verdict = (
        "fail"
        if False in results
        else ("pass" if all(r is True for r in results) else "incomplete")
    )
    return {"agreement": agreement, "metrics": metrics, "checks": checks, "verdict": verdict}
