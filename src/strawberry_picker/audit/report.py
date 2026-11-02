"""Statistical dataset audit: one pass over silver, with one embedding per fruit crop.

Writes to ``reports/dataset-audit/``:
- ``report.md`` and ``summary.json``: source probe, leave-one-source-out, learning curve,
  and suspected label errors per source.
- ``label_issues.csv``: every audited crop, ranked by how strongly the probe disagrees.
- ``review/<source>.csv`` + ``.jpg``: the top suspects for a human to mark ``verdict``.

ADR 0005 decides what the numbers mean: section D (keep, relabel or drop a source) needs
the human verdicts; section E lists the diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strawberry_picker.audit.contact_sheets import render_sheet
from strawberry_picker.audit.embeddings import _bbox, audit_label, with_crop_ids
from strawberry_picker.audit.label_issues import find_label_issues, out_of_fold_probabilities
from strawberry_picker.audit.probes import learning_curve, leave_one_source_out, source_probe
from strawberry_picker.data.readers import Annotation

TASK_MATURITY = "maturity"
TASK_DEFECTS = "defects"
REVIEW_COLUMNS = (
    "position",
    "crop_id",
    "bronze_path",
    "given",
    "suggested",
    "self_confidence",
    "verdict",
)


@dataclass(frozen=True)
class AuditResult:
    source_probe: dict[str, Any]
    loso: pd.DataFrame  # maturity task
    curve: pd.DataFrame  # maturity task
    issues: pd.DataFrame
    per_source: pd.DataFrame


def audit_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """Silver rows a human has not flagged, with one ``label`` and its ``task`` each."""
    frame = with_crop_ids(manifest)
    frame = frame[~frame.needs_review.astype(bool)].copy()
    frame["label"] = [audit_label(m, d) for m, d in zip(frame.maturity, frame.defects, strict=True)]
    frame = frame[frame.label.notna()].copy()
    frame["task"] = np.where(frame.maturity.notna(), TASK_MATURITY, TASK_DEFECTS)
    return frame.reset_index(drop=True)


def _label_issues(
    frame: pd.DataFrame, embeddings: np.ndarray, n_splits: int, seed: int
) -> pd.DataFrame:
    parts = []
    for task, rows in frame.groupby("task"):
        classes, codes = np.unique(rows.label.to_numpy(), return_inverse=True)
        if len(classes) < 2:
            continue
        index = rows.index.to_numpy()
        probs = out_of_fold_probabilities(
            embeddings[index], codes, rows.group_id.to_numpy(), n_splits=n_splits, seed=seed
        )
        found = find_label_issues(codes, probs)
        picked = rows.iloc[found.index]
        parts.append(
            pd.DataFrame(
                {
                    "crop_id": picked.crop_id.to_numpy(),
                    "source_name": picked.source_name.to_numpy(),
                    "bronze_path": picked.bronze_path.to_numpy(),
                    "task": task,
                    "given": classes[found.given.to_numpy()],
                    "suggested": classes[found.suggested.to_numpy()],
                    "self_confidence": found.self_confidence.to_numpy(),
                    "is_issue": found.is_issue.to_numpy(),
                }
            )
        )
    issues = pd.concat(parts, ignore_index=True)
    return issues.sort_values(
        ["is_issue", "self_confidence"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def run_audit(
    frame: pd.DataFrame, embeddings: np.ndarray, *, n_splits: int = 5, seed: int = 0
) -> AuditResult:
    """Probe, held-out sources, learning curve and label issues over ``audit_frame`` rows."""
    groups = frame.group_id.to_numpy()
    probe = source_probe(
        embeddings, frame.source_name.to_numpy(), groups, n_splits=n_splits, seed=seed
    )

    maturity = frame.index[frame.task == TASK_MATURITY].to_numpy()
    _, codes = np.unique(frame.label.to_numpy()[maturity], return_inverse=True)
    loso = leave_one_source_out(
        embeddings[maturity],
        codes,
        frame.source_name.to_numpy()[maturity],
        groups[maturity],
        n_splits=n_splits,
        seed=seed,
    )
    curve = learning_curve(embeddings[maturity], codes, groups[maturity], seed=seed)

    issues = _label_issues(frame, embeddings, n_splits, seed)
    per_source = (
        issues.groupby(["source_name", "task"])
        .agg(n=("crop_id", "size"), flagged=("is_issue", "sum"))
        .reset_index()
    )
    per_source["flagged_rate"] = per_source.flagged / per_source.n
    return AuditResult(probe, loso, curve, issues, per_source)


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def write_report(result: AuditResult, out_dir: Path) -> Path:
    """Write ``report.md``, ``summary.json`` and ``label_issues.csv``; return the report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_probe": result.source_probe,
        "leave_one_source_out": result.loso.to_dict(orient="records"),
        "learning_curve": result.curve.to_dict(orient="records"),
        "label_issues_per_source": result.per_source.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    result.issues.to_csv(out_dir / "label_issues.csv", index=False)

    probe = result.source_probe
    report = "\n\n".join(
        [
            "# Dataset audit",
            "Generated by `scripts/dataset_audit.py`. How to read it: ADR 0005 §D–E.",
            "## Can a probe tell the sources apart?",
            f"Balanced accuracy **{probe['balanced_accuracy']:.3f}** "
            f"(chance {probe['chance']:.3f}, {probe['n_sources']} sources).",
            "## Maturity: leave one source out (macro-F1)",
            _markdown_table(result.loso),
            "## Maturity: learning curve (macro-F1 on a fixed held-out set)",
            _markdown_table(result.curve),
            "## Suspected label errors (confident learning)",
            _markdown_table(result.per_source),
        ]
    )
    path = out_dir / "report.md"
    path.write_text(report + "\n")
    return path


def write_review_sheets(
    result: AuditResult,
    frame: pd.DataFrame,
    bronze_dir: Path,
    out_dir: Path,
    *,
    top_k: int = 100,
    tile_px: int = 128,
    columns: int = 10,
) -> dict[str, Path]:
    """Top ``top_k`` suspects per source: a CSV to fill in and a sheet in the same order."""
    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    bboxes = frame.set_index("crop_id").bbox
    written = {}
    for source, rows in result.issues[result.issues.is_issue].groupby("source_name"):
        top = rows.head(top_k).reset_index(drop=True)
        sheet = top.assign(position=np.arange(1, len(top) + 1), verdict="")
        sheet[list(REVIEW_COLUMNS)].to_csv(review_dir / f"{source}.csv", index=False)
        annotations = [
            Annotation(bronze_dir / path, given, tuple(b) if (b := _bbox(bboxes[crop])) else None)
            for crop, path, given in zip(top.crop_id, top.bronze_path, top.given, strict=True)
        ]
        title = f"{source} | suspected label errors 1-{len(top)}, row by row | see {source}.csv"
        render_sheet(annotations, title=title, tile_px=tile_px, columns=columns).save(
            review_dir / f"{source}.jpg", quality=90
        )
        written[str(source)] = review_dir / f"{source}.csv"
    return written
