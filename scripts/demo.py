"""Live inspection demo: uv run python scripts/demo.py

Serves a page on http://127.0.0.1:8766 where you drop a photo of strawberries and get each
berry's ripeness and quality, plus a verdict for the batch. The model warms up in the
background (minutes on CPU, the page says when it's ready); precomputed samples show at once.

Precompute samples before a presentation (any photos except the locked phone set):
    uv run python scripts/demo.py --precompute <image or folder>...
They land in reports/demo/ (<stem>.json + <stem>.jpg), or --samples.

Uses the newest models/maturity_cls/* and models/quality_cls/* runs unless --run,
--quality-run or --no-quality say otherwise.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from strawberry_picker.demo.server import make_server, precompute, resolve_inputs
from strawberry_picker.inference import LockedSetError, load_predictor, newest_run
from strawberry_picker.paths import demo_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run", type=Path, help="maturity run folder (default: newest)")
    parser.add_argument(
        "--quality-run", type=Path, help="defect run folder (default: newest quality_cls run)"
    )
    parser.add_argument("--no-quality", action="store_true", help="skip the defect head")
    parser.add_argument("--min-score", type=float, default=0.2, help="detector score threshold")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument(
        "--samples", type=Path, default=demo_dir(), help="sample folder (default: reports/demo)"
    )
    parser.add_argument(
        "--precompute", nargs="+", type=Path, metavar="PATH", help="write samples, then exit"
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.precompute:
        try:
            paths = resolve_inputs(args.precompute)  # refuses the locked set first
        except (LockedSetError, FileNotFoundError) as error:
            parser.error(str(error))
        if not paths:
            parser.error("no images found")

    run = args.run or newest_run("maturity_cls")
    if run is None:
        parser.error("no model run yet: uv run python scripts/train_maturity.py")
    quality_run = None if args.no_quality else args.quality_run or newest_run("quality_cls")

    def load():
        return load_predictor(run, quality_run=quality_run, min_score=args.min_score)

    if args.precompute:
        print(f"loading the model ({run.name}); this takes a few minutes on CPU…", flush=True)
        written = precompute(paths, load(), args.samples)
        print(f"{len(written)} of {len(paths)} photos -> {args.samples}")
        return

    server = make_server(
        load,
        samples_dir=args.samples,
        run=run.name,
        quality_run=quality_run.name if quality_run else None,
        port=args.port,
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"demo at {url} (model {run.name} warming up in the background; Ctrl-C to stop)")
    print(f"{len(server.demo.samples)} samples from {args.samples}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
