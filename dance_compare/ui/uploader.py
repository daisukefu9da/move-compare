"""動画を端末の中で軽くしてから送るアップロード欄（Streamlit のカスタムコンポーネント）.

スマホで撮った 4K・60fps の動画をそのまま送ると時間がかかるので、ブラウザの動画変換の機能（WebCodecs）で
長辺 1280px・30fps 程度に縮めてから送る（分析には十分な大きさ）。送り先は server.py が追加する
/api/upload（ui/upload_api.py）。縮められない端末では元の動画を送る。
"""

from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components

from . import upload_api

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "uploader")
_component = components.declare_component("video_uploader", path=_FRONTEND)


def endpoint() -> str | None:
    """アップロードの送り先（server.py で起動したときだけある）."""
    return os.environ.get("DANCE_COMPARE_UPLOAD_API") or None


def video_uploader(label: str, key: str, max_side: int = 1280, fps: int = 30, bitrate: int = 4_000_000,
                   chunk_mb: int = 8) -> tuple[str | None, dict | None]:
    """(受け取った動画のパス, 送ったときの情報) を返す. まだ選ばれていなければ (None, None)."""
    # method: "auto"（ブラウザの動画変換 → だめなら再生しながら縮める）/ "video"（再生しながら縮める. 確認用）
    method = os.environ.get("DANCE_COMPARE_SHRINK_METHOD", "auto")
    v = _component(label=label, endpoint=endpoint(), max_side=max_side, fps=fps, bitrate=bitrate,
                   chunk_mb=chunk_mb, method=method, key=key, default=None)
    if not v:
        return None, None
    path = upload_api.resolve(v.get("id", ""), v.get("name", ""))
    if path is None:
        st.error("送った動画が見つかりません。もう一度選んでください（サーバーが再起動した可能性があります）。")
    return path, v
