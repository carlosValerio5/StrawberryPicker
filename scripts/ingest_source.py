"""Land a downloaded dataset in bronze: uv run python scripts/ingest_source.py <source> <path>."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from strawberry_picker.data.bronze import ingest
from strawberry_picker.data.sources import load_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="source name from configs/sources.yaml")
    parser.add_argument("artifact", type=Path, help="downloaded .zip/.tar archive or folder")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    sources = load_sources()
    if args.source not in sources:
        parser.error(f"unknown source {args.source!r}; known: {', '.join(sorted(sources))}")
    pull = ingest(sources[args.source], args.artifact, args.date)
    print(f"ingested into {pull}")


if __name__ == "__main__":
    main()
