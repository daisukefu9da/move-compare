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


def test_chunked_upload_and_resume(client):
    c, api = client
    uid = "0123abcd-0000-4000-8000-000000000000"
    data = bytes(range(256)) * 100
    r = c.post(f"/api/upload?id={uid}&offset=0&final=0&name=IMG_1.MOV", content=data[:10000])
    assert r.status_code == 200 and r.json()["size"] == 10000
    # 同じ所をもう一度送った（通信が切れて送り直した）場合は、受け取った大きさを返して続きから送ってもらう
    r = c.post(f"/api/upload?id={uid}&offset=0&final=0&name=IMG_1.MOV", content=data[:10000])
    assert r.status_code == 409 and r.json()["size"] == 10000
    assert api.resolve(uid, "IMG_1.MOV") is None  # まだ途中
    r = c.post(f"/api/upload?id={uid}&offset=10000&final=1&name=IMG_1.MOV", content=data[10000:])
    assert r.status_code == 200
    path = api.resolve(uid, "IMG_1.MOV")
    assert path and open(path, "rb").read() == data


def test_rejects_bad_ids_and_paths(client):
    c, api = client
    assert c.post("/api/upload?id=../../etc&offset=0&final=1&name=a.mp4", content=b"x").status_code == 400
    uid = "0123abcd-0000-4000-8000-000000000001"
    r = c.post(f"/api/upload?id={uid}&offset=0&final=1&name=../../evil.sh", content=b"x")
    assert r.status_code == 200
    path = api.resolve(uid, "../../evil.sh")
    assert path and path.startswith(api.UPLOAD_DIR) and path.endswith(".mp4")
