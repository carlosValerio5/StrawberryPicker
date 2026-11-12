"""Label crops: uv run python scripts/label.py <task.csv> --labeler alice.

Opens a page on http://127.0.0.1:<port>: tap the buttons, or use the keys 1/2/3 (maturity),
b m r d p c (defects), Enter (save), n (not a strawberry), u (unsure), Backspace (undo).
With --multi-fruit also s (several berries), on the multi-fruit review task.

On a phone: add --lan, open the printed link on a phone on the same Wi-Fi, and type your
name once. Each person labels under their own name. Stop with Ctrl-C; labels are saved as
you go, and re-running resumes where everyone left off.
"""

from __future__ import annotations

import argparse
import secrets
import socket
import subprocess
import webbrowser
from pathlib import Path

from strawberry_picker.labeling.tool import LabelStore, load_task, make_server, work_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--labeler", help="your name (required unless --lan)")
    parser.add_argument(
        "--lan", action="store_true", help="serve to phones on this Wi-Fi, behind a token"
    )
    parser.add_argument("--host", help="with --lan: address to listen on (default: Wi-Fi)")
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
    if not args.labeler and not args.lan:
        parser.error("--labeler is required (or use --lan and type your name on the phone)")
    store = LabelStore(out, args.labeler)
    order = work_order(task, sample=args.sample, seed=args.seed)
    host, token = ("127.0.0.1", None)
    if args.lan:
        host, token = args.host or lan_address(), secrets.token_urlsafe(12)
    server = make_server(
        task,
        store,
        blind=args.blind,
        order=order,
        host=host,
        port=args.port,
        token=token,
        multi_fruit=args.multi_fruit,
    )
    url = f"http://{host}:{server.server_address[1]}/" + (f"?token={token}" if token else "")
    print(f"{len(order)} crops; labels -> {out}")
    print(f"labeling at {url}", flush=True)
    if args.lan:
        print("open that link on a phone on the same Wi-Fi; Ctrl-C to stop", flush=True)
    elif not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. Land the labels with scripts/ingest_reviews.py")
    finally:
        server.server_close()


def lan_address() -> str:
    """The Wi-Fi address phones can reach: macOS en0/en1 first, since a VPN can own the
    default route; then whichever interface routes to the local network (no packet sent)."""
    for interface in ("en0", "en1"):
        found = subprocess.run(
            ["ipconfig", "getifaddr", interface], capture_output=True, text=True, check=False
        )
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]


if __name__ == "__main__":
    main()
