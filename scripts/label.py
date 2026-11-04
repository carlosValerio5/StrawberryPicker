"""Label crops with the keyboard: uv run python scripts/label.py <task.csv> --labeler alice.

Opens a page on http://127.0.0.1:<port>. Keys: 1/2/3 maturity, b m r d p c defects,
Enter save, n not a strawberry, u unsure, Backspace undo (with --multi-fruit also
s several berries). Stop with Ctrl-C; labels are saved as you go, and re-running resumes
where you left off.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from strawberry_picker.labeling.tool import LabelStore, load_task, make_server, work_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--labeler", required=True, help="your name; one per person")
    parser.add_argument("--blind", action="store_true", help="hide the source's label")
    parser.add_argument("--sample", type=int, help="label only a random N (e.g. 50)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, help="labels CSV (default: next to the task)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument(
        "--multi-fruit",
        action="store_true",
        help="offer the 's' several-berries key (the multi-fruit review task)",
    )
    args = parser.parse_args()

    task = load_task(args.task)
    out = args.out or args.task.with_name(f"labels_{args.task.stem}.csv")
    store = LabelStore(out, args.labeler)
    order = work_order(task, sample=args.sample, seed=args.seed)
    server = make_server(
        task, store, blind=args.blind, order=order, port=args.port, multi_fruit=args.multi_fruit
    )
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"{len(order)} crops, {len(store.done())} already labeled by {args.labeler}")
    print(f"labeling at {url}  (labels -> {out}; Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped; {len(store.done())} labeled. Land them with scripts/ingest_reviews.py")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
