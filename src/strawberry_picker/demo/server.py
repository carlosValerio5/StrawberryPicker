"""Inspection demo server: the page in ``page/`` plus a small JSON API. Standard library only.

The predictor loads in a background thread (minutes on CPU), and photos are inspected one
at a time by a single worker thread, so the page uploads, gets a job id at once, and polls.
Precomputed samples (``<stem>.json`` + ``<stem>.jpg`` in the samples folder, written by
``precompute``) are registered as finished jobs at startup, so they show instantly.

API:
    GET  /api/status        {"state": "warming"|"ready"|"error", "error", "run", "quality_run"}
    POST /api/inspect       raw image bytes -> 202 {"job", "image"}; 413 / 400 / 503
    GET  /api/jobs/<id>     {"state", "error", "image", "width", "height", "berries", "summary"}
    GET  /api/image/<id>    JPEG, longest side <= 1600 px (bboxes are in these coordinates)
    GET  /api/samples       {"samples": [{"id", "title", "image"}]}
"""

from __future__ import annotations

import io
import json
import mimetypes
import queue
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from strawberry_picker.demo.summary import summarize
from strawberry_picker.inference import (
    BerryPrediction,
    UnsupportedImageError,
    ensure_not_locked,
    image_paths,
    load_image,
)

PAGE_DIR = Path(__file__).with_name("page")
SERVED_MAX_SIDE = 1600
JPEG_QUALITY = 88
SAMPLE_PREFIX = "sample-"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".txt": "text/plain; charset=utf-8",
}


class SupportsPredict(Protocol):
    def predict(self, image: Image.Image) -> list[BerryPrediction]: ...


def content_type(path: Path) -> str:
    known = CONTENT_TYPES.get(path.suffix.lower())
    return known or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def decode_image(data: bytes) -> Image.Image:
    """Decode upload bytes like ``images.open_rgb``: EXIF orientation, then RGB (HEIC too)."""
    try:
        with Image.open(io.BytesIO(data)) as raw:
            return ImageOps.exif_transpose(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UnsupportedImageError(f"not a readable image ({error})") from error


def served_image(image: Image.Image) -> tuple[Image.Image, float]:
    """The copy the page shows (longest side <= ``SERVED_MAX_SIDE``) and its scale factor."""
    scale = min(1.0, SERVED_MAX_SIDE / max(image.size))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def job_payload(
    predictions: Sequence[BerryPrediction], *, image_id: str, size: tuple[int, int], scale: float
) -> dict[str, Any]:
    """A finished job: berries (bboxes scaled to the served image, plus ``id``) and summary."""
    berries = []
    for index, prediction in enumerate(predictions):
        record = prediction.to_dict()
        record["bbox"] = [v * scale for v in record["bbox"]]
        record["id"] = index
        berries.append(record)
    return {
        "state": "done",
        "error": None,
        "image": image_id,
        "width": size[0],
        "height": size[1],
        "berries": berries,
        "summary": summarize(berries),
    }


def inspect_photo(
    image: Image.Image, predictor: SupportsPredict, image_id: str
) -> tuple[dict[str, Any], bytes]:
    """Run the predictor on the full-resolution photo; return the payload and served JPEG."""
    served, scale = served_image(image)
    predictions = predictor.predict(image)
    payload = job_payload(predictions, image_id=image_id, size=served.size, scale=scale)
    return payload, jpeg_bytes(served)


def sample_title(stem: str) -> str:
    words = re.sub(r"[_\-]+", " ", stem).strip()
    return words[:1].upper() + words[1:] if words else stem


def resolve_inputs(inputs: Iterable[Path]) -> list[Path]:
    """Precompute inputs as image files; refuses the locked set before listing anything."""
    inputs = [Path(p) for p in inputs]
    ensure_not_locked(inputs)
    return image_paths(inputs)


def precompute(paths: Sequence[Path], predictor: SupportsPredict, samples_dir: Path) -> list[Path]:
    """Write ``<stem>.json`` + ``<stem>.jpg`` per photo into ``samples_dir``.

    Refuses the locked phone test set before reading anything (AGENTS.md rule 6).
    Unreadable files are skipped. Returns the JSON files written.
    """
    ensure_not_locked(paths)
    samples_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in paths:
        try:
            image = load_image(path)
        except UnsupportedImageError:
            continue
        stem = Path(path).stem
        payload, jpeg = inspect_photo(image, predictor, stem)
        payload["title"] = sample_title(stem)
        (samples_dir / f"{stem}.jpg").write_bytes(jpeg)
        target = samples_dir / f"{stem}.json"
        target.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(target)
    return written


def load_samples(samples_dir: Path | None) -> list[tuple[dict[str, Any], dict[str, Any], bytes]]:
    """``(sample entry, job payload, jpeg)`` for every readable sample pair, by file name."""
    if samples_dir is None or not Path(samples_dir).is_dir():
        return []
    samples = []
    for json_path in sorted(Path(samples_dir).glob("*.json")):
        jpg_path = json_path.with_suffix(".jpg")
        if not jpg_path.is_file():
            continue
        try:
            stored = json.loads(json_path.read_text())
            berries = list(stored["berries"])
            jpeg = jpg_path.read_bytes()
            with Image.open(io.BytesIO(jpeg)) as image:
                size = image.size
            summary = summarize(berries)  # recomputed: summary rules may have changed
        except (ValueError, KeyError, TypeError, OSError):
            continue  # a broken sample must not stop the demo
        sample_id = SAMPLE_PREFIX + re.sub(r"[^A-Za-z0-9_-]+", "-", json_path.stem)
        payload = {
            "state": "done",
            "error": None,
            "image": sample_id,
            "width": size[0],
            "height": size[1],
            "berries": berries,
            "summary": summary,
        }
        entry = {
            "id": sample_id,
            "title": stored.get("title") or sample_title(json_path.stem),
            "image": sample_id,
        }
        samples.append((entry, payload, jpeg))
    return samples


class DemoState:
    """Predictor lifecycle, the job table and served images, shared by handler threads."""

    def __init__(self, load_predictor: Callable[[], SupportsPredict]) -> None:
        self.lock = threading.Lock()
        self.state = "warming"
        self.error: str | None = None
        self.predictor: SupportsPredict | None = None
        self.loaded = threading.Event()  # set once the predictor is ready or has failed
        self.jobs: dict[str, dict[str, Any]] = {}
        self.images: dict[str, bytes] = {}
        self.samples: list[dict[str, Any]] = []
        self.queue: queue.Queue[tuple[str, Image.Image] | None] = queue.Queue()
        self._load_predictor = load_predictor

    def start(self) -> None:
        threading.Thread(target=self._warm, name="demo-warmup", daemon=True).start()
        threading.Thread(target=self._work, name="demo-worker", daemon=True).start()

    def stop(self) -> None:
        self.queue.put(None)

    def _warm(self) -> None:
        try:
            predictor = self._load_predictor()
        except Exception as error:  # surfaced through /api/status
            with self.lock:
                self.state, self.error = "error", f"{type(error).__name__}: {error}"
        else:
            with self.lock:
                self.predictor, self.state = predictor, "ready"
        finally:
            self.loaded.set()

    def _work(self) -> None:
        while (item := self.queue.get()) is not None:
            job_id, image = item
            self._update(job_id, state="running")
            try:
                if self.predictor is None:  # pragma: no cover - jobs are accepted only when ready
                    raise RuntimeError("the model is not loaded")
                served, scale = served_image(image)
                predictions = self.predictor.predict(image)
                payload = job_payload(predictions, image_id=job_id, size=served.size, scale=scale)
                with self.lock:
                    self.jobs[job_id] = payload
            except Exception as error:  # one bad photo must not stop the worker
                self._update(job_id, state="error", error=f"{type(error).__name__}: {error}")

    def _update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            self.jobs[job_id] = {**self.jobs[job_id], **fields}

    def add_samples(self, samples_dir: Path | None) -> None:
        for entry, payload, jpeg in load_samples(samples_dir):
            self.samples.append(entry)
            self.jobs[entry["id"]] = payload
            self.images[entry["image"]] = jpeg

    def submit(self, image: Image.Image) -> str:
        job_id = uuid.uuid4().hex[:12]
        served, _ = served_image(image)
        jpeg = jpeg_bytes(served)
        with self.lock:
            self.images[job_id] = jpeg
            self.jobs[job_id] = {
                "state": "queued",
                "error": None,
                "image": job_id,
                "width": served.width,
                "height": served.height,
                "berries": [],
                "summary": None,
            }
        self.queue.put((job_id, image))
        return job_id

    def status(self) -> tuple[str, str | None]:
        with self.lock:
            return self.state, self.error

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            found = self.jobs.get(job_id)
            return dict(found) if found is not None else None

    def image(self, image_id: str) -> bytes | None:
        with self.lock:
            return self.images.get(image_id)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    demo: DemoState

    def server_close(self) -> None:
        self.demo.stop()
        super().server_close()


def static_file(page_dir: Path, url_path: str) -> Path | None:
    """The file under ``page_dir`` this URL names, or None (missing, or outside the folder)."""
    relative = unquote(url_path).lstrip("/") or "index.html"
    if ".." in Path(relative).parts or "\\" in relative or "\x00" in relative:
        return None
    root = page_dir.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def make_server(
    load_predictor: Callable[[], SupportsPredict],
    *,
    samples_dir: Path | None,
    run: str,
    quality_run: str | None,
    host: str = "127.0.0.1",
    port: int = 0,
    max_bytes: int = 25_000_000,
    page_dir: Path = PAGE_DIR,
) -> DemoServer:
    """A demo server, already warming the predictor; call ``serve_forever()``."""
    demo = DemoState(load_predictor)
    demo.add_samples(samples_dir)
    demo.start()
    page_dir = Path(page_dir)

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

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/status":
                state, error = demo.status()
                self._json({"state": state, "error": error, "run": run, "quality_run": quality_run})
            elif path == "/api/samples":
                self._json({"samples": demo.samples})
            elif path.startswith("/api/jobs/"):
                job = demo.job(path.removeprefix("/api/jobs/"))
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown job")
                else:
                    self._json(job)
            elif path.startswith("/api/image/"):
                jpeg = demo.image(path.removeprefix("/api/image/"))
                if jpeg is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown image")
                else:
                    self._send(HTTPStatus.OK, jpeg, "image/jpeg")
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
            else:
                found = static_file(page_dir, path)
                if found is None:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                else:
                    self._send(HTTPStatus.OK, found.read_bytes(), content_type(found))

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/inspect":
                self.close_connection = True  # the body is left unread
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            state, error = demo.status()
            if state != "ready":
                self.close_connection = True
                message = "the model is still warming up" if state == "warming" else error
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, message or "the model is unavailable")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > max_bytes:
                self.close_connection = True
                limit = f"{max_bytes / 1_000_000:g} MB"
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"photo is larger than {limit}")
                return
            if length <= 0:
                self._error(HTTPStatus.BAD_REQUEST, "send the photo as the request body")
                return
            try:
                image = decode_image(self.rfile.read(length))
            except UnsupportedImageError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            job_id = demo.submit(image)
            self._json({"job": job_id, "image": job_id}, HTTPStatus.ACCEPTED)

    server = DemoServer((host, port), Handler)
    server.demo = demo
    return server
