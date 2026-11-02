from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from strawberry_picker.audit.report import (  # noqa: E402 (after importorskip)
    audit_frame,
    run_audit,
    write_report,
    write_review_sheets,
)


def _manifest() -> pd.DataFrame:
    rows = []
    for source, n in (("a", 60), ("b", 60)):
        for i in range(n):
            rows.append(
                {
                    "image_id": f"{source}{i:03d}",
                    "source_name": source,
                    "bronze_path": f"{source}/2026-11-02/{i}.png",
                    "maturity": "ripe" if i % 2 else "unripe",
                    "defects": [],
                    "needs_review": False,
                    "bbox": None,
                    "group_id": f"{source}/{i}",
                }
            )
    for i in range(40):
        rows.append(
            {
                "image_id": f"c{i:03d}",
                "source_name": "c",
                "bronze_path": f"c/2026-11-02/{i}.png",
                "maturity": None,
                "defects": ["rot"] if i % 2 else [],
                "needs_review": False,
                "bbox": None,
                "group_id": f"c/{i}",
            }
        )
    rows.append({**rows[0], "image_id": "skip", "needs_review": True})
    rows.append({**rows[0], "image_id": "nolabel", "maturity": None, "defects": None})
    return pd.DataFrame(rows)


def _embeddings(frame: pd.DataFrame, flip_row: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    positive = frame.label.isin(["ripe", "rot"]).to_numpy().copy()
    positive[flip_row] = not positive[flip_row]  # the image says the opposite of its label
    features = rng.normal(size=(len(frame), 6))
    features[:, 0] += np.where(positive, 3.0, -3.0)
    features[:, 1] += frame.source_name.map({"a": 3.0, "b": -3.0, "c": 0.0}).to_numpy()
    return features


def test_audit_frame_keeps_reviewable_labelled_rows() -> None:
    frame = audit_frame(_manifest())

    assert "skip" not in set(frame.image_id) and "nolabel" not in set(frame.image_id)
    assert set(frame.task) == {"maturity", "defects"}
    assert set(frame.loc[frame.source_name == "c", "label"]) == {"healthy", "rot"}
    assert frame.crop_id.iloc[0] == "a000_0"


def test_run_audit_flags_the_planted_label_error() -> None:
    frame = audit_frame(_manifest())
    planted = 3  # a003, labelled ripe

    result = run_audit(frame, _embeddings(frame, planted), n_splits=4, seed=0)

    assert result.issues.iloc[0].crop_id == frame.crop_id.iloc[planted]
    assert result.issues.iloc[0].suggested == "unripe"
    per_source = result.per_source.set_index("source_name")
    assert per_source.loc["a", "n"] == 60 and per_source.loc["a", "flagged"] >= 1
    assert per_source.loc["c", "task"] == "defects"
    assert result.source_probe["n_sources"] == 3
    assert list(result.loso.held_out) == ["a", "b", "in_distribution"]
    assert not result.curve.empty


def test_write_report_and_review_sheets(tmp_path, make_image) -> None:
    manifest = _manifest()
    frame = audit_frame(manifest)
    result = run_audit(frame, _embeddings(frame, 3), n_splits=4, seed=0)
    bronze = tmp_path / "bronze"
    for path in frame.bronze_path:
        make_image(bronze / path, size=(40, 40))

    write_report(result, tmp_path / "out")
    review = write_review_sheets(result, frame, bronze, tmp_path / "out", top_k=5, tile_px=16)

    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["source_probe"]["n_sources"] == 3
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Dataset audit")
    assert (tmp_path / "out" / "label_issues.csv").exists()
    a_sheet = pd.read_csv(tmp_path / "out" / "review" / "a.csv")
    assert list(a_sheet.columns) == [
        "position",
        "crop_id",
        "bronze_path",
        "given",
        "suggested",
        "self_confidence",
        "verdict",
    ]
    assert a_sheet.position.tolist() == list(range(1, len(a_sheet) + 1))
    assert (tmp_path / "out" / "review" / "a.jpg").exists()
    assert set(review) <= {"a", "b", "c"}


def test_random_sample_sheets_are_seeded_and_cover_every_source(tmp_path, make_image) -> None:
    from strawberry_picker.audit.report import write_random_sample_sheets

    frame = audit_frame(_manifest())
    bronze = tmp_path / "bronze"
    for path in frame.bronze_path:
        make_image(bronze / path, size=(40, 40))

    first = write_random_sample_sheets(frame, bronze, tmp_path / "out", n=10, seed=0, tile_px=8)
    second = write_random_sample_sheets(frame, bronze, tmp_path / "again", n=10, seed=0, tile_px=8)

    assert sorted(first) == ["a", "b", "c"]
    sample = pd.read_csv(first["a"])
    assert list(sample.columns) == ["position", "crop_id", "bronze_path", "given", "verdict"]
    assert len(sample) == 10 and sample.crop_id.is_unique
    assert sample.equals(pd.read_csv(second["a"]))
    assert (tmp_path / "out" / "random_sample" / "a.jpg").exists()
