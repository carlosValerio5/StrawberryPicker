from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pandas as pd
import pytest
from PIL import Image

from strawberry_picker.labeling.tool import (
    Label,
    LabelStore,
    load_task,
    make_server,
    parse_label,
    work_order,
    write_task,
)

NOW = datetime(2026, 11, 5, 10, 0, tzinfo=UTC)


@pytest.fixture
def task(tmp_path, make_image) -> pd.DataFrame:
    image = make_image(tmp_path / "IMG_1.jpg", size=(120, 80))
    frame = pd.DataFrame(
        {
            "item_id": [f"c{i}" for i in range(5)],
            "image_path": [str(image)] * 5,
            "bbox": [json.dumps([10, 10, 50, 50]), "", json.dumps([60, 10, 110, 70]), "", ""],
            "given": ["ripe", "unripe", "overripe", "ripe", "ripe"],
            "group": ["IMG_1"] * 5,
        }
    )
    write_task(frame, tmp_path / "task.csv")
    return load_task(tmp_path / "task.csv")


def test_parse_label_validates_against_the_taxonomy() -> None:
    label = parse_label({"item_id": "c0", "maturity": "ripe", "defects": ["rot", "bruise"]})
    assert label == Label("c0", "ripe", ("bruise", "rot"), not_fruit=False, unsure=False)
    assert parse_label({"item_id": "c1", "not_fruit": True}).maturity is None
    with pytest.raises(ValueError, match="maturity"):
        parse_label({"item_id": "c2", "maturity": "green"})
    with pytest.raises(ValueError, match="defect"):
        parse_label({"item_id": "c2", "maturity": "ripe", "defects": ["dirty"]})
    with pytest.raises(ValueError, match="maturity"):
        parse_label({"item_id": "c2"})  # a real label needs a maturity


def test_store_is_append_only_and_supports_undo(tmp_path) -> None:
    store = LabelStore(tmp_path / "labels.csv", "alice")
    store.record(parse_label({"item_id": "c0", "maturity": "ripe"}), now=NOW)
    store.record(parse_label({"item_id": "c1", "maturity": "unripe", "defects": ["mold"]}), now=NOW)

    assert store.undo(now=NOW) == "c1"
    assert set(store.done()) == {"c0"}
    rows = pd.read_csv(tmp_path / "labels.csv", keep_default_na=False)
    assert rows.action.tolist() == ["label", "label", "undo"]  # nothing is ever rewritten
    assert rows.loc[1, "defects"] == "mold"

    other = LabelStore(tmp_path / "labels.csv", "bob")  # same file, separate labeler
    assert other.done() == {}
    assert LabelStore(tmp_path / "labels.csv", "alice").done()["c0"].maturity == "ripe"


def test_work_order_samples_reproducibly(task) -> None:
    assert work_order(task) == ["c0", "c1", "c2", "c3", "c4"]
    sample = work_order(task, sample=3, seed=7)
    assert len(sample) == 3 and sample == work_order(task, sample=3, seed=7)
    assert set(sample) <= set(task.item_id)


def _call(base: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(request) as response:
        body = response.read()
        kind = response.headers.get_content_type()
    return (json.loads(body) if kind == "application/json" else body), kind


def test_server_round_trip(tmp_path, task) -> None:
    store = LabelStore(tmp_path / "labels.csv", "alice")
    server = make_server(task, store, blind=True, order=work_order(task))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page, kind = _call(base, "/")
        assert kind == "text/html" and b"strawberry" in page.lower()

        state, _ = _call(base, "/api/next")
        assert state["item"]["item_id"] == "c0" and state["done"] == 0 and state["total"] == 5
        assert "given" not in state["item"]  # blind mode hides the source's label

        crop, kind = _call(base, "/crop/c0")
        assert kind == "image/jpeg" and Image.open(io.BytesIO(crop)).size == (448, 448)

        _call(base, "/api/label", {"item_id": "c0", "maturity": "ripe", "defects": ["bruise"]})
        assert _call(base, "/api/next")[0]["item"]["item_id"] == "c1"
        assert _call(base, "/api/undo", {})[0]["undone"] == "c0"
        assert _call(base, "/api/next")[0]["item"]["item_id"] == "c0"

        with pytest.raises(urllib.error.HTTPError) as bad:
            _call(base, "/api/label", {"item_id": "c0", "maturity": "purple"})
        assert bad.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as unknown:
            _call(base, "/crop/nope")
        assert unknown.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_server_shows_the_given_label_when_not_blind(tmp_path, task) -> None:
    server = make_server(task, LabelStore(tmp_path / "l.csv", "a"), blind=False, order=["c2"])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        state, _ = _call(f"http://127.0.0.1:{server.server_address[1]}", "/api/next")
        assert state["item"]["given"] == "overripe"
    finally:
        server.shutdown()
        server.server_close()


def test_crop_task_from_silver_rows(tmp_path) -> None:
    from strawberry_picker.labeling.tasks import crop_task

    silver = pd.DataFrame(
        {
            "image_id": ["img1", "img1", "img2"],
            "bronze_path": ["src/2026-11-02/a.jpg", "src/2026-11-02/a.jpg", "src/2026-11-02/b.jpg"],
            "maturity": ["overripe", "ripe", None],
            "defects": [["mold"], [], ["rot"]],
            "bbox": [[1.0, 2.0, 30.0, 40.0], [5.0, 5.0, 9.0, 9.0], None],
        }
    )

    task = crop_task(silver, tmp_path / "bronze")

    assert list(task.columns) == ["item_id", "image_path", "bbox", "given", "group"]
    assert task.item_id.tolist() == ["img1_0", "img1_1", "img2_0"]
    assert task.image_path.iloc[0] == str(tmp_path / "bronze" / "src/2026-11-02/a.jpg")
    assert json.loads(task.bbox.iloc[0]) == [1.0, 2.0, 30.0, 40.0] and task.bbox.iloc[2] == ""
    assert task.given.tolist() == ["overripe +mold", "ripe", "rot"]
    assert task.group.tolist() == ["img1", "img1", "img2"]


def test_load_task_resolves_relative_image_paths(tmp_path, make_image) -> None:
    folder = tmp_path / "phone"
    make_image(folder / "IMG_1.jpg")
    pd.DataFrame(
        {
            "item_id": ["IMG_1_0"],
            "image_path": ["IMG_1.jpg"],
            "bbox": [""],
            "given": [""],
            "group": ["IMG_1"],
        }
    ).to_csv(folder / "task.csv", index=False)

    assert load_task(folder / "task.csv").image_path.iloc[0] == str(folder / "IMG_1.jpg")


def _serve(task, store, **kwargs):
    server = make_server(task, store, order=work_order(task), **kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _raw(url: str, *, cookie: str | None = None, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if cookie:
        request.add_header("Cookie", cookie)
    return urllib.request.urlopen(request)


def test_token_guards_every_route_and_is_kept_in_a_cookie(tmp_path, task) -> None:
    server, base = _serve(task, LabelStore(tmp_path / "l.csv", "alice"), blind=True, token="s3cret")
    try:
        for path in ("/", "/api/next", "/api/config", "/crop/c0"):
            with pytest.raises(urllib.error.HTTPError) as denied:
                _raw(base + path)
            assert denied.value.code == 403

        with _raw(base + "/?token=s3cret") as response:
            cookie = response.headers["Set-Cookie"].split(";")[0]
            assert "HttpOnly" in response.headers["Set-Cookie"]
        with _raw(base + "/api/next", cookie=cookie) as response:
            assert json.loads(response.read())["item"]["item_id"] == "c0"
        with pytest.raises(urllib.error.HTTPError) as wrong:
            _raw(base + "/api/next", cookie="token=nope")
        assert wrong.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_each_phone_labels_under_its_own_name(tmp_path, task) -> None:
    labels = tmp_path / "labels.csv"
    server, base = _serve(task, LabelStore(labels, None), blind=True)
    try:
        assert _call(base, "/api/config")[0] == {"labeler": None, "blind": True}
        _call(base, "/api/label", {"item_id": "c0", "maturity": "ripe", "labeler": "alice"})
        assert _call(base, "/api/next?labeler=alice")[0]["item"]["item_id"] == "c1"
        assert _call(base, "/api/next?labeler=bob")[0]["item"]["item_id"] == "c0"
        assert _call(base, "/api/undo", {"labeler": "bob"})[0]["undone"] is None
        assert _call(base, "/api/undo", {"labeler": "alice"})[0]["undone"] == "c0"

        for bad in ("", "a b", "x" * 40, "../etc"):
            with pytest.raises(urllib.error.HTTPError) as rejected:
                _call(base, "/api/label", {"item_id": "c0", "maturity": "ripe", "labeler": bad})
            assert rejected.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as nameless:
            _call(base, "/api/next")
        assert nameless.value.code == 400  # no default labeler and none given
    finally:
        server.shutdown()
        server.server_close()
    assert set(pd.read_csv(labels).labeler) == {"alice"}


def test_page_works_by_touch(tmp_path, task) -> None:
    server, base = _serve(task, LabelStore(tmp_path / "l.csv", "a"), blind=False)
    try:
        page = _call(base, "/")[0].decode()
    finally:
        server.shutdown()
        server.server_close()
    assert 'name="viewport"' in page
    for target in ('data-m="ripe"', 'data-d="mold"', 'id="save"', 'id="notfruit"', 'id="undo"'):
        assert target in page and "<button" in page
