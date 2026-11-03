"""Contact sheets per source label: uv run python scripts/contact_sheets.py <source>.

Works on disabled sources too. That's the point: look at a source before enabling it.
"""

from __future__ import annotations

import argparse

from strawberry_picker.audit.contact_sheets import build_contact_sheets
from strawberry_picker.data.bronze import latest_pull
from strawberry_picker.data.sources import load_sources
from strawberry_picker.paths import reports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="source name from configs/sources.yaml")
    parser.add_argument("--per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = load_sources()
    if args.source not in sources:
        parser.error(f"unknown source {args.source!r}; known: {', '.join(sorted(sources))}")
    pull = latest_pull(args.source)
    if pull is None:
        parser.error(f"{args.source} has no bronze pull yet; run scripts/ingest_source.py first")

    out = reports_dir() / "dataset-audit" / args.source / "contact-sheets"
    summary = build_contact_sheets(
        sources[args.source], pull, out, per_label=args.per_label, seed=args.seed
    )
    print(summary.to_string(index=False))
    print(f"\nsheets in {out}")


if __name__ == "__main__":
    main()
