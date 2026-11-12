"""The ADR 0005 verdict on the locked phone set: uv run python scripts/evaluate_phone_set.py.

Uses the newest model run (models/maturity_cls/*) and the newest bronze phone_testset pull
unless --run / --pull say otherwise. Writes reports/phone-eval/<run_id>/verdict.json.
If labelers still disagree somewhere, writes adjudication.csv in the pull's working copy
folder instead (label it with: scripts/label.py adjudication.csv --labeler adjudicated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strawberry_picker.audit.embeddings import dinov2_embedder
from strawberry_picker.config import load_config
from strawberry_picker.data.bronze import latest_pull
from strawberry_picker.data.crops import resolve_policy
from strawberry_picker.evaluation.phone_set import evaluate_pull, load_pull_labels, unresolved_items
from strawberry_picker.labeling.tool import load_task, write_task
from strawberry_picker.models.linear_probe import load_probe
from strawberry_picker.paths import models_dir, reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="model run folder (default: newest)")
    parser.add_argument("--pull", type=Path, help="phone pull or working folder (default: newest)")
    args = parser.parse_args()

    runs = sorted((models_dir() / "maturity_cls").glob("*/probe.joblib"))
    run = args.run or (runs[-1].parent if runs else None)
    pull = args.pull or latest_pull("phone_testset")
    if run is None:
        parser.error("no model run yet: uv run python scripts/train_maturity.py")
    if pull is None:
        parser.error("no phone set yet: see docs/labeling/test-set-protocol.md")

    labels = load_pull_labels(pull)
    open_items = unresolved_items(labels)
    if open_items:
        task = load_task(pull / "task.csv")
        path = write_task(task[task.item_id.isin(open_items)], pull / "adjudication.csv")
        print(f"{len(open_items)} crops still need adjudication -> {path}")
        print(f"label them together: uv run python scripts/label.py {path} --labeler adjudicated")
        return

    config = load_config(run / "config.yaml")
    report = evaluate_pull(
        pull,
        load_probe(run),
        dinov2_embedder(config["model"]["backbone"]),
        config["classes"],
        n_resamples=config["eval"]["bootstrap_resamples"],
        crop_policy=resolve_policy(config),  # crop the way the model was trained (ADR 0011)
    )
    out = reports_dir() / "phone-eval" / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(report, indent=2, default=float))
    for check in report["checks"]:
        status = {True: "PASS", False: "FAIL", None: "n/a "}[check["passed"]]
        print(f"[{status}] {check['id']:<3} {check['rule']}")
    print(f"\nverdict: {report['verdict'].upper()}  (details: {out / 'verdict.json'})")


if __name__ == "__main__":
    main()
