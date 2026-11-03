"""Land finished label files in bronze: uv run python scripts/ingest_reviews.py labels_*.csv.

Then rebuild silver (`make silver`) to apply them.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from strawberry_picker.data.bronze import ingest_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--revision", type=int, help="second+ landing on the same day")
    args = parser.parse_args()
    print(f"landed in {ingest_reviews(args.files, args.date, args.revision)}")


if __name__ == "__main__":
    main()
