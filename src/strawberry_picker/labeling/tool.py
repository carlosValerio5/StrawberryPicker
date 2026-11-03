"""Keyboard labeling: one crop at a time, ~2 s per berry, labels appended to a CSV.

A *task* CSV lists what to label (``TASK_COLUMNS``); ``bbox`` is a JSON ``[x0, y0, x1, y1]``
or empty for a whole-image fruit. Labels are appended, never rewritten (``LABEL_COLUMNS``):
an undo is a new ``action=undo`` row. Several labelers can share one file.

The server binds to 127.0.0.1 and serves one page plus a tiny JSON API, so it needs no
dependency beyond the standard library and Pillow.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from strawberry_picker.data.gold import crop_box
from strawberry_picker.schema import Defect, Maturity

TASK_COLUMNS = ("item_id", "image_path", "bbox", "given", "group")
LABEL_COLUMNS = (
    "item_id",
    "labeler",
    "maturity",
    "defects",
    "not_fruit",
    "unsure",
    "action",
    "labeled_at",
)
CROP_PX = 448


@dataclass(frozen=True, slots=True)
class Label:
    item_id: str
    maturity: str | None
    defects: tuple[str, ...]
    not_fruit: bool = False
    unsure: bool = False


def parse_label(payload: dict[str, Any]) -> Label:
    """Validate a label against the taxonomy. A real label (not ``n``/``u``) needs maturity."""
    not_fruit = bool(payload.get("not_fruit", False))
    unsure = bool(payload.get("unsure", False))
    maturity = payload.get("maturity")
    if maturity is not None and maturity not in {m.value for m in Maturity}:
        raise ValueError(f"unknown maturity {maturity!r}")
    if maturity is None and not (not_fruit or unsure):
        raise ValueError("a label needs a maturity (or mark it not_fruit / unsure)")
    defects = tuple(sorted(set(payload.get("defects") or ())))
    unknown = set(defects) - {d.value for d in Defect}
    if unknown:
        raise ValueError(f"unknown defect(s) {sorted(unknown)}")
    return Label(str(payload["item_id"]), maturity, defects, not_fruit, unsure)


def current_labels(rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, str], Label]:
    """``(labeler, item_id) -> Label`` after replaying labels and undos in file order."""
    current: dict[tuple[str, str], Label] = {}
    for row in rows:
        key = (row["labeler"], row["item_id"])
        current.pop(key, None)
        if row["action"] == "label":
            current[key] = Label(
                row["item_id"],
                row["maturity"] or None,
                tuple(d for d in row["defects"].split("|") if d),
                row["not_fruit"] == "True",
                row["unsure"] == "True",
            )
    return current


class LabelStore:
    """One labeler's view of an append-only labels CSV."""

    def __init__(self, path: Path, labeler: str) -> None:
        self.path = Path(path)
        self.labeler = labeler

    def _append(self, row: dict[str, Any]) -> None:
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def _rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as handle:
            return [r for r in csv.DictReader(handle) if r["labeler"] == self.labeler]

    def record(self, label: Label, *, now: datetime | None = None) -> None:
        self._append(
            {
                "item_id": label.item_id,
                "labeler": self.labeler,
                "maturity": label.maturity or "",
                "defects": "|".join(label.defects),
                "not_fruit": label.not_fruit,
                "unsure": label.unsure,
                "action": "label",
                "labeled_at": (now or datetime.now(UTC)).isoformat(),
            }
        )

    def done(self) -> dict[str, Label]:
        """Current label per item for this labeler, after undos (insertion-ordered)."""
        return {item: label for (_, item), label in current_labels(self._rows()).items()}

    def undo(self, *, now: datetime | None = None) -> str | None:
        """Withdraw this labeler's most recent label; returns its item id."""
        current = self.done()
        if not current:
            return None
        item_id = next(reversed(current))
        self._append(
            {
                **dict.fromkeys(LABEL_COLUMNS, ""),
                "item_id": item_id,
                "labeler": self.labeler,
                "action": "undo",
                "labeled_at": (now or datetime.now(UTC)).isoformat(),
            }
        )
        return item_id


def load_task(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_task(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(TASK_COLUMNS)].to_csv(path, index=False)
    return path


def work_order(task: pd.DataFrame, *, sample: int | None = None, seed: int = 0) -> list[str]:
    """Items in task order, or a reproducible random ``sample`` of them."""
    items = task.item_id.tolist()
    if sample is not None and sample < len(items):
        items = random.Random(seed).sample(items, sample)
    return items


def render_crop(row: pd.Series, size: int = CROP_PX) -> bytes:
    with Image.open(row.image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    bbox = json.loads(row.bbox) if row.bbox else None
    crop = image.crop(crop_box(bbox, *image.size))
    buffer = io.BytesIO()
    ImageOps.pad(crop, (size, size), color=(255, 255, 255)).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def make_server(
    task: pd.DataFrame,
    store: LabelStore,
    *,
    blind: bool,
    order: list[str],
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """A local labeling server; call ``serve_forever()`` and open ``http://host:port``."""
    rows = task.set_index("item_id", drop=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/next":
                done = store.done()
                pending = [i for i in order if i not in done]
                item = None
                if pending:
                    item = {"item_id": pending[0]}
                    if not blind:
                        item["given"] = rows.loc[pending[0], "given"]
                total = len(order)
                self._json({"item": item, "done": total - len(pending), "total": total})
            elif self.path.startswith("/crop/") and self.path[6:] in rows.index:
                self._send(HTTPStatus.OK, render_crop(rows.loc[self.path[6:]]), "image/jpeg")
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/label":
                    label = parse_label(payload)
                    if label.item_id not in rows.index:
                        raise ValueError(f"unknown item {label.item_id!r}")
                    store.record(label)
                    self._json({"ok": True})
                elif self.path == "/api/undo":
                    self._json({"undone": store.undo()})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), Handler)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Strawberry labeling</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font:15px/1.4 system-ui,sans-serif;margin:0;background:#faf7f5;color:#222}
main{display:flex;gap:24px;padding:20px;flex-wrap:wrap}
img{width:448px;height:448px;max-width:90vw;max-height:90vw;border-radius:8px;background:#fff}
.k{display:inline-block;min-width:1.4em;padding:1px 6px;border:1px solid #bbb;
border-radius:4px;background:#fff;font-family:ui-monospace,monospace;text-align:center}
.on{background:#c62828;color:#fff;border-color:#c62828}
aside{max-width:420px}#msg{min-height:1.4em;color:#c62828}
</style></head><body><main>
<div><img id="crop" alt="strawberry crop"><div id="progress"></div><div id="given"></div></div>
<aside>
<h2>Strawberry labeling</h2>
<p><b>Maturity</b> &mdash; <span class="k" data-m="unripe">1</span> unripe (&lt;&frac34; pink/red)
<span class="k" data-m="ripe">2</span> ripe (&ge;&frac34; pink/red)
<span class="k" data-m="overripe">3</span> overripe (&ge;2 of: dark, dull, wrinkled, slumped,
leaking)</p>
<p><b>Defects</b> (toggle) &mdash; <span class="k" data-d="bruise">b</span> bruise
<span class="k" data-d="mold">m</span> mold <span class="k" data-d="rot">r</span> rot
<span class="k" data-d="deformed">d</span> deformed
<span class="k" data-d="pest_damage">p</span> pest
<span class="k" data-d="mechanical_damage">c</span> cut/crushed</p>
<p><span class="k">Enter</span> save &middot; <span class="k">n</span> not a strawberry &middot;
<span class="k">u</span> unsure &middot; <span class="k">&#9003;</span> undo</p>
<p>Rubric: docs/labeling/rubric.md. Judge only the centred berry.</p>
<p id="msg"></p>
</aside></main>
<script>
const KEYS={"1":"unripe","2":"ripe","3":"overripe"};
const DEF={b:"bruise",m:"mold",r:"rot",d:"deformed",p:"pest_damage",c:"mechanical_damage"};
let item=null, maturity=null, defects=new Set();
function paint(){document.querySelectorAll("[data-m]").forEach(e=>e.classList.toggle("on",
e.dataset.m===maturity));document.querySelectorAll("[data-d]").forEach(e=>e.classList.toggle(
"on",defects.has(e.dataset.d)));}
async function next(){const s=await (await fetch("/api/next")).json();item=s.item;
maturity=null;defects=new Set();paint();
document.getElementById("progress").textContent=s.done+" / "+s.total+" labeled";
if(!item){document.getElementById("crop").removeAttribute("src");
document.getElementById("msg").textContent="All done — you can close this tab.";return;}
document.getElementById("crop").src="/crop/"+encodeURIComponent(item.item_id);
document.getElementById("given").textContent=item.given?"source says: "+item.given:"";}
async function send(extra){const r=await fetch("/api/label",{method:"POST",body:JSON.stringify(
Object.assign({item_id:item.item_id,maturity:maturity,defects:[...defects]},extra))});
const j=await r.json();document.getElementById("msg").textContent=j.error||"";if(r.ok)next();}
document.addEventListener("keydown",async e=>{if(!item&&e.key!=="Backspace")return;
if(KEYS[e.key]){maturity=KEYS[e.key];paint();}
else if(DEF[e.key]){const d=DEF[e.key];defects.has(d)?defects.delete(d):defects.add(d);paint();}
else if(e.key==="Enter")send({});
else if(e.key==="n")send({maturity:null,defects:[],not_fruit:true});
else if(e.key==="u")send({maturity:null,defects:[],unsure:true});
else if(e.key==="Backspace"){e.preventDefault();await fetch("/api/undo",{method:"POST",
body:"{}"});next();}});
next();
</script></body></html>
"""
