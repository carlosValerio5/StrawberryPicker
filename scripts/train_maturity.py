"""Train and score the baseline maturity probe: uv run python scripts/train_maturity.py.

Reads configs/maturity_cls.yaml and its gold release, embeds crops (cached), trains on the
gold train split outside `holdout_sources`, and writes models/maturity_cls/<run_id>/.
Development scores only; the ADR 0005 verdict comes from scripts/evaluate_phone_set.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from strawberry_picker.audit.embeddings import cached_embeddings, dinov2_embedder
from strawberry_picker.config import load_config
from strawberry_picker.data.gold import dataset_dir
from strawberry_picker.models.linear_probe import run_baseline, save_run
from strawberry_picker.paths import Layer, layer_dir, models_dir, reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="maturity_cls")
    args = parser.parse_args()

    config = load_config(args.config)
    gold = dataset_dir(config["dataset"])
    if not gold.exists():
        parser.error(f"{gold} missing: uv run python scripts/build_gold.py maturity_cls 0.1")
    rows = pd.read_parquet(gold / "manifest.parquet")
    backbone = config["model"]["backbone"]
    embeddings = cached_embeddings(
        rows,
        layer_dir(Layer.SILVER) / "images",
        reports_dir() / "dataset-audit" / "embeddings.parquet",
        lambda: dinov2_embedder(backbone),
        model=backbone,
    )
    probe, metrics = run_baseline(
        rows,
        embeddings,
        config["classes"],
        holdout_sources=config["holdout_sources"],
        seed=config["seed"],
        n_resamples=config["eval"]["bootstrap_resamples"],
        c=config["model"]["c"],
        calibration=config.get("calibration", "temperature"),
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = save_run(models_dir() / config["task"] / run_id, probe, config, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"run saved to {run}")


if __name__ == "__main__":
    main()
