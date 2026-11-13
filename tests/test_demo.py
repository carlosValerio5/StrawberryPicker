"""Inspection demo: batch summary and the HTTP API, with a fake predictor (no ML, no GPU)."""

from __future__ import annotations

import io
import json
import re
import runpy
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from strawberry_picker.demo import summarize
from strawberry_picker.demo.server import (
    PAGE_DIR,
    SERVED_MAX_SIDE,
    DemoServer,
    content_type,
    load_samples,
    make_server,
    precompute,
    resolve_inputs,
    sample_title,
)
from strawberry_picker.inference import BerryPrediction, LockedSetError
from strawberry_picker.paths import demo_dir, reports_dir
from strawberry_picker.schema import grade_from_defects

from conftest import pattern_image

ROOT = Path(__file__).resolve().parents[1]


def berry(maturity: str = "ripe", defects: list[str] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "score": 0.9,
        "maturity": maturity,
        "probabilities": {maturity: 1.0},
    }
    if defects is not None:
        record["defects"] = defects
        record["grade"] = grade_from_defects(defects).value
    return record


# --- summary ---------------------------------------------------------------------------


def test_summary_of_an_empty_photo() -> None:
    assert summarize([]) == {
        "count": 0,
        "maturity": {"unripe": 0, "ripe": 0, "overripe": 0},
        "grade": None,
        "defects": {},
        "ready": 0,
        "packout": None,
        "verdict": "empty",
        "reasons": ["No strawberries found in this photo"],
    }


def test_summary_all_premium() -> None:
    summary = summarize([berry(defects=[]) for _ in range(3)])
    assert summary["verdict"] == "premium"
    assert summary["grade"] == {"premium": 3, "standard": 0, "reject": 0}
    assert summary["ready"] == 3 and summary["packout"] == 1.0
    assert summary["reasons"] == ["All 3 berries are ripe with no visible defects"]
    assert summarize([berry(defects=[])])["reasons"] == [
        "The berry is ripe with no visible defects"
    ]


def test_summary_mixed_with_reject() -> None:
    berries = [
        berry(defects=[]),
        berry(defects=["rot"]),
        berry("unripe", defects=["bruise"]),
        berry("unripe", defects=[]),
        berry("overripe", defects=["bruise", "mold"]),
    ]
    summary = summarize(berries)
    assert summary["count"] == 5
    assert summary["maturity"] == {"unripe": 2, "ripe": 2, "overripe": 1}
    assert summary["grade"] == {"premium": 2, "standard": 1, "reject": 2}
    assert list(summary["defects"].items()) == [("bruise", 2), ("mold", 1), ("rot", 1)]
    assert summary["ready"] == 1  # the rotten ripe berry is not ready
    assert summary["packout"] == pytest.approx(0.4)
    assert summary["verdict"] == "reject"
    assert summary["reasons"] == [
        "1 berry shows mold",
        "1 berry shows rot",
        "2 berries are not yet ripe",
        "1 berry is past peak ripeness",
        "2 berries show bruising",
    ]


def test_summary_standard_from_ripeness_or_minor_defects() -> None:
    assert summarize([berry(defects=[]), berry("overripe", defects=[])])["verdict"] == "standard"
    minor = summarize([berry(defects=["deformed"])])
    assert minor["verdict"] == "standard"
    assert minor["reasons"] == ["1 berry shows an irregular shape"]


def test_summary_without_the_quality_head() -> None:
    ripe = summarize([berry(), berry()])
    assert ripe["grade"] is None and ripe["packout"] is None and ripe["defects"] == {}
    assert ripe["verdict"] == "premium" and ripe["ready"] == 2
    assert ripe["reasons"] == ["All 2 berries are ripe"]
    assert summarize([berry("unripe")])["reasons"] == ["1 berry is not yet ripe"]


def test_summary_derives_grade_from_defects_alone() -> None:
    summary = summarize([{**berry(), "defects": ["rot"]}])
    assert summary["grade"] == {"premium": 0, "standard": 0, "reject": 1}
    assert summary["verdict"] == "reject" and summary["ready"] == 0


# --- server helpers --------------------------------------------------------------------


class FakePredictor:
    """Two fixed berries in image coordinates: a premium ripe one and a rotten unripe one."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[tuple[int, int]] = []

    def predict(self, image: Image.Image) -> list[BerryPrediction]:
        self.seen.append(image.size)
        if self.fail:
            raise RuntimeError("detector exploded")
        w, h = image.size
        return [
            BerryPrediction(
                bbox=(0, 0, w / 2, h / 2),
                score=0.9,
                maturity="ripe",
                probabilities={"unripe": 0.1, "ripe": 0.85, "overripe": 0.05},
                extras={"defects": [], "grade": "premium"},
            ),
            BerryPrediction(
                bbox=(w / 2, h / 2, w, h),
                score=0.7,
                maturity="unripe",
                probabilities={"unripe": 0.8, "ripe": 0.15, "overripe": 0.05},
                extras={"defects": ["rot"], "grade": "reject"},
            ),
        ]


def jpeg(size: tuple[int, int] = (64, 48), seed: int = 0) -> bytes:
    buffer = io.BytesIO()
    pattern_image(seed, size).save(buffer, "JPEG")
    return buffer.getvalue()


def request(base: str, path: str, data: bytes | None = None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(base + path, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def get_json(base: str, path: str, data: bytes | None = None) -> tuple[int, Any]:
    status, headers, body = request(base, path, data)
    assert headers["Content-Type"] == "application/json"
    assert headers["Cache-Control"] == "no-store"
    return status, json.loads(body)


def wait_for_job(base: str, job: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, payload = get_json(base, f"/api/jobs/{job}")
        if payload["state"] in {"done", "error"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job} did not finish")


@pytest.fixture
def serve() -> Iterator[Callable[..., tuple[str, DemoServer]]]:
    servers: list[DemoServer] = []

    def _serve(loader: Callable[[], Any] | None = None, **kwargs: Any) -> tuple[str, DemoServer]:
        kwargs.setdefault("samples_dir", None)
        kwargs.setdefault("run", "run-1")
        kwargs.setdefault("quality_run", "quality-1")
        server = make_server(loader or FakePredictor, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", server

    yield _serve
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def page_dir(tmp_path: Path) -> Path:
    page = tmp_path / "page"
    (page / "fonts").mkdir(parents=True)
    (page / "index.html").write_text("<!doctype html><title>demo</title>")
    (page / "app.css").write_text("body{}")
    (page / "app.js").write_text("console.log(1)")
    (page / "mark.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (page / "fonts" / "body.woff2").write_bytes(b"wOF2")
    (tmp_path / "secret.txt").write_text("outside the page dir")
    return page


# --- routes ----------------------------------------------------------------------------


def test_page_routes_serve_files_with_their_content_type(serve, page_dir: Path) -> None:
    base, _ = serve(page_dir=page_dir)
    expected = {
        "/": ("text/html; charset=utf-8", b"<!doctype html>"),
        "/index.html": ("text/html; charset=utf-8", b"<!doctype html>"),
        "/app.css": ("text/css; charset=utf-8", b"body"),
        "/app.js": ("text/javascript; charset=utf-8", b"console"),
        "/mark.svg": ("image/svg+xml", b"<svg"),
        "/fonts/body.woff2": ("font/woff2", b"wOF2"),
    }
    for path, (kind, start) in expected.items():
        status, headers, body = request(base, path)
        assert (status, headers["Content-Type"]) == (200, kind), path
        assert headers["Cache-Control"] == "no-store" and body.startswith(start)


@pytest.mark.parametrize(
    "path",
    ["/../secret.txt", "/%2e%2e/secret.txt", "/fonts/../../secret.txt", "/missing.css", "/fonts"],
)
def test_traversal_and_missing_files_are_404(serve, page_dir: Path, path: str) -> None:
    base, _ = serve(page_dir=page_dir)
    status, payload = get_json(base, path)
    assert status == 404 and payload == {"error": "not found"}


def test_missing_page_dir_is_404(serve, tmp_path: Path) -> None:
    base, _ = serve(page_dir=tmp_path / "nope")
    assert get_json(base, "/")[0] == 404


def test_default_page_dir_and_content_types() -> None:
    assert PAGE_DIR == ROOT / "src" / "strawberry_picker" / "demo" / "page"
    assert content_type(Path("x.unknownext")) == "application/octet-stream"
    assert content_type(Path("x.gif")) == "image/gif"  # falls back to mimetypes


def test_the_shipped_page_and_its_assets_are_served(serve) -> None:
    base, _ = serve()  # default page dir: the real page
    status, headers, body = request(base, "/")
    page = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert 'name="viewport"' in page and "<title>StrawberryPicker" in page
    for asset in ("app.css", "app.js", "fonts/archivo-latin-wdth-normal.woff2"):
        assert asset in page
        assert request(base, f"/{asset}")[0] == 200
    # every id app.js looks up exists in the page
    script = request(base, "/app.js")[2].decode()
    for element_id in set(re.findall(r'\$\("([a-z-]+)"\)', script)):
        assert f'id="{element_id}"' in page, element_id
    # self-hosted fonts ship with their OFL license (AGENTS.md rule 8)
    for license_file in ("OFL-Archivo.txt", "OFL-MartianMono.txt"):
        text = (PAGE_DIR / "fonts" / license_file).read_text()
        assert "SIL Open Font License, Version 1.1" in text


def test_unknown_api_routes_are_404(serve) -> None:
    base, _ = serve()
    for path in ["/api/nope", "/api/jobs/missing", "/api/image/missing"]:
        status, payload = get_json(base, path)
        assert status == 404 and "error" in payload, path
    assert get_json(base, "/api/other", data=b"x")[0] == 404


def test_status_reports_ready_and_runs(serve) -> None:
    base, server = serve()
    assert server.demo.loaded.wait(5)
    status, payload = get_json(base, "/api/status")
    assert status == 200
    assert payload == {"state": "ready", "error": None, "run": "run-1", "quality_run": "quality-1"}


def test_upload_then_poll_until_done(serve) -> None:
    base, server = serve()
    server.demo.loaded.wait(5)
    status, accepted = get_json(base, "/api/inspect", data=jpeg((64, 48)))
    assert status == 202 and set(accepted) == {"job", "image"}
    payload = wait_for_job(base, accepted["job"])
    assert payload["state"] == "done" and payload["error"] is None
    assert (payload["width"], payload["height"]) == (64, 48)
    assert [b["id"] for b in payload["berries"]] == [0, 1]
    assert payload["berries"][0]["bbox"] == [0.0, 0.0, 32.0, 24.0]
    assert payload["berries"][1]["defects"] == ["rot"]
    assert payload["summary"]["verdict"] == "reject" and payload["summary"]["count"] == 2

    status, headers, body = request(base, f"/api/image/{accepted['image']}")
    assert (status, headers["Content-Type"]) == (200, "image/jpeg")
    assert Image.open(io.BytesIO(body)).size == (64, 48)


def test_large_photos_are_downscaled_and_boxes_follow(serve) -> None:
    predictor = FakePredictor()
    base, server = serve(lambda: predictor)
    server.demo.loaded.wait(5)
    _, accepted = get_json(base, "/api/inspect", data=jpeg((SERVED_MAX_SIDE * 2, 100)))
    payload = wait_for_job(base, accepted["job"])
    assert predictor.seen == [(SERVED_MAX_SIDE * 2, 100)]  # inference sees full resolution
    assert (payload["width"], payload["height"]) == (SERVED_MAX_SIDE, 50)
    assert payload["berries"][1]["bbox"] == [800.0, 25.0, 1600.0, 50.0]
    _, _, body = request(base, f"/api/image/{accepted['image']}")
    assert max(Image.open(io.BytesIO(body)).size) == SERVED_MAX_SIDE


def test_a_failing_prediction_marks_the_job_as_error(serve) -> None:
    base, server = serve(lambda: FakePredictor(fail=True))
    server.demo.loaded.wait(5)
    _, accepted = get_json(base, "/api/inspect", data=jpeg())
    payload = wait_for_job(base, accepted["job"])
    assert payload["state"] == "error" and "detector exploded" in payload["error"]
    assert payload["summary"] is None


def test_junk_and_empty_bodies_are_400(serve) -> None:
    base, server = serve()
    server.demo.loaded.wait(5)
    status, payload = get_json(base, "/api/inspect", data=b"definitely not a photo")
    assert status == 400 and "not a readable image" in payload["error"]
    assert get_json(base, "/api/inspect", data=b"")[0] == 400


def test_oversize_upload_is_413(serve) -> None:
    base, server = serve(max_bytes=1000)
    server.demo.loaded.wait(5)
    status, payload = get_json(base, "/api/inspect", data=b"x" * 2000)
    assert status == 413 and "larger than" in payload["error"]


def test_upload_while_warming_is_503(serve) -> None:
    release = threading.Event()

    def slow_loader() -> FakePredictor:
        release.wait(10)
        return FakePredictor()

    base, server = serve(slow_loader)
    assert get_json(base, "/api/status")[1]["state"] == "warming"
    status, payload = get_json(base, "/api/inspect", data=jpeg())
    assert status == 503 and "warming" in payload["error"]
    release.set()
    assert server.demo.loaded.wait(5)
    assert get_json(base, "/api/status")[1]["state"] == "ready"
    assert get_json(base, "/api/inspect", data=jpeg())[0] == 202


def test_a_failed_load_is_reported(serve) -> None:
    def broken_loader() -> FakePredictor:
        raise ValueError("no weights")

    base, server = serve(broken_loader)
    assert server.demo.loaded.wait(5)
    status = get_json(base, "/api/status")[1]
    assert status["state"] == "error" and "no weights" in status["error"]
    code, payload = get_json(base, "/api/inspect", data=jpeg())
    assert code == 503 and "no weights" in payload["error"]


# --- samples and precompute ------------------------------------------------------------


def test_precomputed_samples_are_served_as_done_jobs(serve, tmp_path: Path, make_image) -> None:
    photos = tmp_path / "photos"
    make_image(photos / "market_punnet.jpg", seed=1, size=(80, 60))
    make_image(photos / "field-tray.png", seed=2, size=(60, 40))
    (photos / "notes.jpg").write_bytes(b"not an image")
    samples = tmp_path / "samples"
    written = precompute(resolve_inputs([photos]), FakePredictor(), samples)
    assert sorted(p.name for p in written) == ["field-tray.json", "market_punnet.json"]
    (samples / "orphan.json").write_text("{}")  # no jpg: skipped
    (samples / "broken.json").write_text("{not json")
    (samples / "broken.jpg").write_bytes(jpeg())

    base, _ = serve(samples_dir=samples)
    _, listing = get_json(base, "/api/samples")
    assert listing["samples"] == [
        {"id": "sample-field-tray", "title": "Field tray", "image": "sample-field-tray"},
        {"id": "sample-market_punnet", "title": "Market punnet", "image": "sample-market_punnet"},
    ]
    status, payload = get_json(base, "/api/jobs/sample-market_punnet")
    assert status == 200 and payload["state"] == "done"
    assert (payload["width"], payload["height"]) == (80, 60)
    assert payload["image"] == "sample-market_punnet" and len(payload["berries"]) == 2
    assert payload["summary"]["verdict"] == "reject"
    status, headers, _ = request(base, "/api/image/sample-field-tray")
    assert (status, headers["Content-Type"]) == (200, "image/jpeg")


def test_no_samples_folder_means_no_samples(tmp_path: Path) -> None:
    assert load_samples(None) == [] and load_samples(tmp_path / "missing") == []
    assert sample_title("") == "" and sample_title("a__b") == "A b"


def test_precompute_refuses_the_locked_phone_set(tmp_data_dir: Path, make_image) -> None:
    locked = tmp_data_dir / "bronze" / "phone_testset"
    make_image(locked / "2026" / "a.jpg")
    predictor = FakePredictor()
    with pytest.raises(LockedSetError):
        resolve_inputs([locked])
    with pytest.raises(LockedSetError):
        precompute([locked / "2026" / "a.jpg"], predictor, tmp_data_dir / "out")
    assert predictor.seen == [] and not (tmp_data_dir / "out").exists()


def test_demo_cli_refuses_the_locked_set_before_loading(
    tmp_data_dir: Path, make_image, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    locked = make_image(tmp_data_dir / "bronze" / "phone_testset" / "a.jpg")
    argv = ["demo.py", "--precompute", str(locked), "--samples", str(tmp_data_dir / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / "scripts" / "demo.py"), run_name="__main__")
    assert exit_info.value.code == 2
    assert "locked phone test set" in capsys.readouterr().err


def test_demo_dir_lives_under_reports() -> None:
    assert demo_dir() == reports_dir() / "demo"
