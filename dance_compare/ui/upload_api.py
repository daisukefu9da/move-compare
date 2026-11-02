"""動画のアップロードを受け取る口（Streamlit の st.App に追加するルート. server.py から使う）.

スマホの画面（ui/frontend/uploader）は、動画を端末の中で軽くしてから、小分け（チャンク）にして送る。
1 回の通信を短くして、サーバー（Modal）の 1 リクエストあたりの時間制限に掛からないようにする。

    POST /api/upload?id=<ランダムな ID>&offset=<これまでに送ったバイト数>&final=<最後なら 1>&name=<ファイル名>
"""

from __future__ import annotations

import os
import re

UPLOAD_DIR = os.environ.get("DANCE_COMPARE_UPLOAD_DIR", ".cache/uploads")
MAX_CHUNK = 16 * 1024 * 1024      # 1 回に受け取る大きさの上限
MAX_TOTAL = 4 * 1024 * 1024 * 1024  # 1 本の動画の大きさの上限
_ID = re.compile(r"^[0-9a-f-]{8,64}$")


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


async def upload_chunk(request):
    from starlette.responses import JSONResponse

    q = request.query_params
    upload_id, name = q.get("id", ""), q.get("name", "")
    if not _ID.match(upload_id):
        return JSONResponse({"error": "bad id"}, status_code=400)
    try:
        offset = int(q.get("offset", "0"))
    except ValueError:
        return JSONResponse({"error": "bad offset"}, status_code=400)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    part = os.path.join(UPLOAD_DIR, f"{upload_id}.part")
    size = os.path.getsize(part) if os.path.exists(part) else 0
    if offset != size:  # 送り直し・順番違い: いまの大きさを返して、そこから続けてもらう
        return JSONResponse({"error": "offset", "size": size}, status_code=409)
    body = await request.body()
    if len(body) > MAX_CHUNK or size + len(body) > MAX_TOTAL:
        return JSONResponse({"error": "too large"}, status_code=413)
    with open(part, "ab") as f:
        f.write(body)
    size += len(body)
    if q.get("final") == "1":
        os.replace(part, final_path(upload_id, name))
    return JSONResponse({"size": size})
