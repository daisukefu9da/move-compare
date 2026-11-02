"""動画のアップロードを受け取る口（Streamlit の st.App に追加するルート. server.py から使う）.

スマホの画面（ui/frontend/uploader）は、動画を端末の中で軽くしてから、小分け（チャンク）にして同時に何本か送る。
1 回の通信を短くしてサーバー（Modal）の 1 リクエストあたりの時間制限に掛からないようにし、
同時に送ることで通信の往復の待ち時間（日本 ↔ 米国）を重ねる。

    POST /api/upload?id=<ランダムな ID>&offset=<この部分の先頭の位置>            （本体: この部分のバイト列）
    POST /api/upload?id=<ID>&finalize=1&total=<全体の大きさ>&name=<ファイル名>  （全部送り終わったら）
"""

from __future__ import annotations

import os
import re
import threading

UPLOAD_DIR = os.environ.get("DANCE_COMPARE_UPLOAD_DIR", ".cache/uploads")
MAX_CHUNK = 16 * 1024 * 1024        # 1 回に受け取る大きさの上限
MAX_TOTAL = 4 * 1024 * 1024 * 1024  # 1 本の動画の大きさの上限
_ID = re.compile(r"^[0-9a-f-]{8,64}$")
_lock = threading.Lock()
_received: dict[str, dict[int, int]] = {}  # id -> {offset: 受け取った大きさ}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]", "_", os.path.basename(name or "video.mp4"))[-80:]
    return name if os.path.splitext(name)[1].lower() in {".mp4", ".mov", ".m4v", ".webm", ".avi"} else name + ".mp4"


def final_path(upload_id: str, name: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{upload_id}_{_safe_name(name)}")


def resolve(upload_id: str, name: str) -> str | None:
    """送り終わった動画のパス（無ければ None）."""
    if not _ID.match(upload_id or ""):
        return None
    path = final_path(upload_id, name)
    return path if os.path.exists(path) else None


def _int(q, key: str) -> int | None:
    try:
        return int(q.get(key, ""))
    except ValueError:
        return None


async def upload_chunk(request):
    from starlette.responses import JSONResponse

    q = request.query_params
    upload_id = q.get("id", "")
    if not _ID.match(upload_id):
        return JSONResponse({"error": "bad id"}, status_code=400)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    part = os.path.join(UPLOAD_DIR, f"{upload_id}.part")

    if q.get("finalize") == "1":
        total = _int(q, "total")
        with _lock:
            got = sum(_received.get(upload_id, {}).values())
        if total is None or got != total or not os.path.exists(part):
            return JSONResponse({"error": "incomplete", "received": got}, status_code=409)
        os.replace(part, final_path(upload_id, q.get("name", "")))
        with _lock:
            _received.pop(upload_id, None)
        return JSONResponse({"size": total})

    offset = _int(q, "offset")
    if offset is None or offset < 0:
        return JSONResponse({"error": "bad offset"}, status_code=400)
    body = await request.body()
    if len(body) > MAX_CHUNK or offset + len(body) > MAX_TOTAL:
        return JSONResponse({"error": "too large"}, status_code=413)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.pwrite(fd, body, offset)  # 届いた順番に関係なく、決まった位置に書く
    finally:
        os.close(fd)
    with _lock:
        _received.setdefault(upload_id, {})[offset] = len(body)  # 同じ部分の送り直しは上書き
    return JSONResponse({"size": len(body)})
