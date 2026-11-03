"""Statistical dataset audit over silver: uv run python scripts/dataset_audit.py.

Needs the ml extra (`make setup-ml`). Embeddings are cached per crop in
reports/dataset-audit/embeddings.parquet, so new sources only embed their new crops.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from strawberry_picker.audit.embeddings import cached_embeddings, dinov2_embedder
from strawberry_picker.audit.report import (
    audit_frame,
    run_audit,
    write_random_sample_sheets,
    write_report,
    write_review_sheets,
)
from strawberry_picker.paths import Layer, layer_dir, reports_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--top-k", type=int, default=100, help="suspects per source to review")
    parser.add_argument("--sample", type=int, default=50, help="random crops per source")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    silver = layer_dir(Layer.SILVER)
    frame = audit_frame(pd.read_parquet(silver / "manifest.parquet"))
    out = reports_dir() / "dataset-audit"
    out.mkdir(parents=True, exist_ok=True)

    embeddings = cached_embeddings(
        frame,
        silver / "images",
        out / "embeddings.parquet",
        lambda: dinov2_embedder(args.model),
        model=args.model,
        batch_size=args.batch_size,
    )

    result = run_audit(frame, embeddings, seed=args.seed)
    report = write_report(result, out)
    bronze = layer_dir(Layer.BRONZE)
    sheets = write_review_sheets(result, frame, bronze, out, top_k=args.top_k)
    samples = write_random_sample_sheets(frame, bronze, out, n=args.sample, seed=args.seed)
    print(report.read_text())
    print("review sheets:", ", ".join(str(p) for p in sheets.values()))
    print("random samples:", ", ".join(str(p) for p in samples.values()))


if __name__ == "__main__":
    main()
