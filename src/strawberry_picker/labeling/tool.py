"""Keyboard labeling: one crop at a time, ~2 s per berry, labels appended to a CSV.

A *task* CSV lists what to label (``TASK_COLUMNS``); ``bbox`` is a JSON ``[x0, y0, x1, y1]``
or empty for a whole-image fruit. Labels are appended, never rewritten (``LABEL_COLUMNS``):
an undo is a new ``action=undo`` row. Several labelers can share one file.

The server binds to 127.0.0.1 by default (``--lan`` in ``scripts/label.py`` opens it to the
Wi-Fi network behind a random token) and serves one responsive page, usable by keyboard on
a laptop or by touch on a phone, plus a tiny JSON API. Standard library and Pillow only.
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import random
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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

    def __init__(self, path: Path, labeler: str | None) -> None:
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
    """Read a task; relative ``image_path`` values are relative to the task file."""
    task = pd.read_csv(path, dtype=str, keep_default_na=False)
    folder = Path(path).parent
    task["image_path"] = [str(p if Path(p).is_absolute() else folder / p) for p in task.image_path]
    return task


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


_LABELER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def make_server(
    task: pd.DataFrame,
    store: LabelStore,
    *,
    blind: bool,
    order: list[str],
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """A labeling server; call ``serve_forever()`` and open ``http://host:port``.

    ``store.labeler`` is the default name; when it's ``None`` each client (e.g. a phone)
    sends its own ``labeler``. With ``token`` set, every request needs it: as ``?token=``
    on the first page load (then kept in an HttpOnly cookie) or an ``X-Token`` header.
    """
    rows = task.set_index("item_id", drop=False)
    stores: dict[str, LabelStore] = {}
    write_lock = threading.Lock()  # two phones can save at the same moment

    def store_for(name: str | None) -> LabelStore:
        name = name or store.labeler
        if not name:
            raise ValueError("a labeler name is required")
        if not _LABELER_NAME.match(name):
            raise ValueError("labeler names use letters, digits, '_' or '-' (max 32)")
        return stores.setdefault(name, LabelStore(store.path, name))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, kind: str, cookie: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _route(self) -> tuple[str, dict[str, str]] | None:
            """Path and query if the request carries the token (or none is needed)."""
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if token is None:
                return url.path, query
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            offered = [
                query.get("token", ""),
                self.headers.get("X-Token", ""),
                cookies["token"].value if "token" in cookies else "",
            ]
            if any(hmac.compare_digest(o, token) for o in offered if o):
                return url.path, query
            self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
            return None

        def do_GET(self) -> None:
            route = self._route()
            if route is None:
                return
            path, query = route
            try:
                if path == "/":
                    cookie = None
                    if token is not None and query.get("token") == token:
                        cookie = f"token={token}; HttpOnly; SameSite=Strict; Path=/"
                    self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8", cookie)
                elif path == "/api/config":
                    self._json({"labeler": store.labeler, "blind": blind})
                elif path == "/api/next":
                    done = store_for(query.get("labeler")).done()
                    pending = [i for i in order if i not in done]
                    item = None
                    if pending:
                        item = {"item_id": pending[0]}
                        if not blind:
                            item["given"] = rows.loc[pending[0], "given"]
                    total = len(order)
                    self._json({"item": item, "done": total - len(pending), "total": total})
                elif path.startswith("/crop/") and path[6:] in rows.index:
                    self._send(HTTPStatus.OK, render_crop(rows.loc[path[6:]]), "image/jpeg")
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            route = self._route()
            if route is None:
                return
            path, _ = route
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                labeler = store_for(payload.get("labeler"))
                if path == "/api/label":
                    label = parse_label(payload)
                    if label.item_id not in rows.index:
                        raise ValueError(f"unknown item {label.item_id!r}")
                    with write_lock:
                        labeler.record(label)
                    self._json({"ok": True})
                elif path == "/api/undo":
                    with write_lock:
                        self._json({"undone": labeler.undo()})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), Handler)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Strawberry labeling</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#faf7f5">
<style>
:root{--bg:#faf7f5;--fg:#222;--muted:#6b6b6b;--line:#d8d2cd;--card:#fff;--on:#c62828;
--on-fg:#fff;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161413;--fg:#eee;--muted:#a59f9a;
--line:#3a3532;--card:#211e1c;--on:#ef5350;--on-fg:#111}}
*{box-sizing:border-box}
body{font:16px/1.4 system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);
color:var(--fg);-webkit-tap-highlight-color:transparent}
main{max-width:980px;margin:0 auto;padding:12px 16px calc(16px + env(safe-area-inset-bottom));
display:grid;gap:16px;grid-template-columns:minmax(0,448px) minmax(0,1fr)}
@media (max-width:760px){main{grid-template-columns:1fr}}
#crop{width:100%;aspect-ratio:1;object-fit:contain;border-radius:12px;background:var(--card);
display:block}
#crop:not([src]){visibility:hidden}
.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:14px;
margin-top:6px;gap:8px}
h2{font-size:17px;margin:0 0 4px}
.row{display:grid;gap:8px;margin:8px 0}
.three{grid-template-columns:repeat(3,1fr)}.six{grid-template-columns:repeat(3,1fr)}
.four{grid-template-columns:2fr 1fr 1fr 1fr}
button{font:inherit;min-height:48px;border-radius:10px;border:1px solid var(--line);
background:var(--card);color:var(--fg);padding:6px 4px;touch-action:manipulation}
button small{display:block;color:var(--muted);font-size:12px}
button.on{background:var(--on);color:var(--on-fg);border-color:var(--on)}
button.on small{color:inherit}
#save{background:var(--ok);color:#fff;border-color:var(--ok);font-weight:600}
#save:disabled{opacity:.45}
.k{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}
#msg{min-height:1.4em;color:var(--on)}
#who{display:none;gap:8px;grid-template-columns:1fr auto}
#who input{font:inherit;min-height:48px;padding:0 12px;border-radius:10px;
border:1px solid var(--line);background:var(--card);color:var(--fg)}
details{color:var(--muted);font-size:14px}
</style></head><body><main>
<section>
<form id="who"><input id="name" placeholder="Your name (e.g. carlos)" autocomplete="off"
 autocapitalize="off" pattern="[A-Za-z0-9_-]{1,32}" required><button>Start</button></form>
<img id="crop" alt="strawberry crop">
<div class="meta"><span id="progress"></span><span id="given"></span></div>
</section>
<section>
<h2>Maturity</h2>
<div class="row three">
<button data-m="unripe">Unripe<small>&lt;&frac34; pink/red</small></button>
<button data-m="ripe">Ripe<small>&ge;&frac34; pink/red</small></button>
<button data-m="overripe">Overripe<small>&ge;2 past-peak signs</small></button>
</div>
<h2>Defects <span class="k">(tap to toggle)</span></h2>
<div class="row six">
<button data-d="bruise">Bruise</button><button data-d="mold">Mold</button>
<button data-d="rot">Rot</button><button data-d="deformed">Deformed</button>
<button data-d="pest_damage">Pest</button><button data-d="mechanical_damage">Cut</button>
</div>
<div class="row four">
<button id="save" disabled>Save</button><button id="notfruit">Not a<br>strawberry</button>
<button id="unsure">Unsure</button><button id="undo">Undo</button>
</div>
<p id="msg"></p>
<details><summary>Rules &amp; keys</summary>
<p>Overripe = pink/red plus two or more of: dark red-purple, dull skin, wrinkled or sunken,
slumped, leaking. Judge only the centred berry. Rubric: docs/labeling/rubric.md.</p>
<p class="k">Keyboard: 1/2/3 maturity &middot; b m r d p c defects &middot; Enter save &middot;
n not a strawberry &middot; u unsure &middot; Backspace undo</p>
<p id="me"></p></details>
</section></main>
<script>
const KEYS={"1":"unripe","2":"ripe","3":"overripe"};
const DEF={b:"bruise",m:"mold",r:"rot",d:"deformed",p:"pest_damage",c:"mechanical_damage"};
let item=null, maturity=null, defects=new Set(), labeler=null;
const $=id=>document.getElementById(id);
function remember(n){try{localStorage.setItem("labeler",n)}catch(e){}}
function recall(){try{return localStorage.getItem("labeler")}catch(e){return null}}
function paint(){document.querySelectorAll("[data-m]").forEach(e=>e.classList.toggle("on",
e.dataset.m===maturity));document.querySelectorAll("[data-d]").forEach(e=>e.classList.toggle(
"on",defects.has(e.dataset.d)));$("save").disabled=!maturity||!item;}
async function api(path,body){const r=await fetch(path,body?{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.assign({labeler},body))}
:{});const j=await r.json();$("msg").textContent=r.ok?"":(j.error||"error");return r.ok?j:null;}
async function next(){const s=await api("/api/next?labeler="+encodeURIComponent(labeler||""));
if(!s)return;item=s.item;maturity=null;defects=new Set();paint();
$("progress").textContent=s.done+" / "+s.total+" labeled";
if(!item){$("crop").removeAttribute("src");$("msg").textContent="All done — thank you!";
return;}$("crop").src="/crop/"+encodeURIComponent(item.item_id);
$("given").textContent=item.given?"source says: "+item.given:"";}
async function send(extra){if(!item)return;if(await api("/api/label",Object.assign(
{item_id:item.item_id,maturity,defects:[...defects]},extra)))next();}
async function undo(){await api("/api/undo",{});next();}
document.querySelectorAll("[data-m]").forEach(b=>b.onclick=()=>{maturity=b.dataset.m;paint();});
document.querySelectorAll("[data-d]").forEach(b=>b.onclick=()=>{const d=b.dataset.d;
defects.has(d)?defects.delete(d):defects.add(d);paint();});
$("save").onclick=()=>send({});
$("notfruit").onclick=()=>send({maturity:null,defects:[],not_fruit:true});
$("unsure").onclick=()=>send({maturity:null,defects:[],unsure:true});
$("undo").onclick=undo;
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
if(KEYS[e.key]){maturity=KEYS[e.key];paint();}
else if(DEF[e.key]){const d=DEF[e.key];defects.has(d)?defects.delete(d):defects.add(d);paint();}
else if(e.key==="Enter"&&maturity)send({});
else if(e.key==="n")send({maturity:null,defects:[],not_fruit:true});
else if(e.key==="u")send({maturity:null,defects:[],unsure:true});
else if(e.key==="Backspace"){e.preventDefault();undo();}});
function start(name){labeler=name;$("who").style.display="none";
$("me").textContent=name?"Labeling as "+name+".":"";next();}
$("who").onsubmit=e=>{e.preventDefault();const n=$("name").value.trim();remember(n);start(n);};
(async()=>{const c=await api("/api/config");if(!c)return;
if(c.labeler){start(null);return;}const n=recall();if(n){start(n);return;}
$("who").style.display="grid";$("name").focus();})();
</script></body></html>
"""
