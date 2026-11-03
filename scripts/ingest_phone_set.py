"""Land the labeled phone test set in bronze: uv run python scripts/ingest_phone_set.py DIR.

DIR holds the photos, captures.csv, task.csv and labels*.csv. It lands in
data/bronze/phone_testset/<date>[.<n>]/; only scripts/evaluate_phone_set.py reads it.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from strawberry_picker.data.bronze import ingest_phone_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--revision", type=int, help="second+ landing on the same day")
    args = parser.parse_args()
    for required in ("task.csv", "captures.csv"):
        if not (args.folder / required).exists():
            parser.error(f"{args.folder / required} is missing")
    print(f"landed in {ingest_phone_set(args.folder, args.date, args.revision)}")


if __name__ == "__main__":
    main()
