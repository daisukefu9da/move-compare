"""スマホから小分けに送られた動画を受け取る口（dance_compare/ui/upload_api.py）."""

import pytest

starlette = pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dance_compare.ui.upload_api as api

    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path))
    app = Starlette(routes=[Route("/api/upload", api.upload_chunk, methods=["POST"])])
    return TestClient(app), api


def test_parallel_chunks_out_of_order_and_retry(client):
    c, api = client
    uid = "0123abcd-0000-4000-8000-000000000000"
    data = bytes(range(256)) * 100
    parts = [(0, data[:10000]), (10000, data[10000:20000]), (20000, data[20000:])]
    # 届く順番が前後しても、同じ部分の送り直しがあっても、決まった位置に書く
    for off, chunk in [parts[2], parts[0], parts[0], parts[1]]:
        assert c.post(f"/api/upload?id={uid}&offset={off}", content=chunk).status_code == 200
    assert api.resolve(uid, "IMG_1.MOV") is None  # まだ完了の合図が無い
    r = c.post(f"/api/upload?id={uid}&finalize=1&total={len(data)}&name=IMG_1.MOV")
    assert r.status_code == 200
    path = api.resolve(uid, "IMG_1.MOV")
    assert path and open(path, "rb").read() == data


def test_finalize_rejects_missing_parts(client):
    c, api = client
    uid = "0123abcd-0000-4000-8000-000000000002"
    c.post(f"/api/upload?id={uid}&offset=0", content=b"a" * 100)
    r = c.post(f"/api/upload?id={uid}&finalize=1&total=300&name=x.mp4")  # 途中の部分が届いていない
    assert r.status_code == 409 and r.json()["received"] == 100
    assert api.resolve(uid, "x.mp4") is None


def test_rejects_bad_ids_and_paths(client):
    c, api = client
    assert c.post("/api/upload?id=../../etc&offset=0", content=b"x").status_code == 400
    uid = "0123abcd-0000-4000-8000-000000000001"
    assert c.post(f"/api/upload?id={uid}&offset=0", content=b"x").status_code == 200
    r = c.post(f"/api/upload?id={uid}&finalize=1&total=1&name=../../evil.sh")
    assert r.status_code == 200
    path = api.resolve(uid, "../../evil.sh")
    assert path and path.startswith(api.UPLOAD_DIR) and path.endswith(".mp4")
