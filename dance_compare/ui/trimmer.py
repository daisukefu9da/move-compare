"""動画を見ながら区間・時刻を選ぶ部品（Streamlit のカスタムコンポーネント）.

iPhone の動画のトリミングのように、コマの一覧の上で区間の端をドラッグすると、その位置の映像が表示される。
再生・コマ送り・「ここを開始に」などのボタンもある。ブラウザには軽いプレビュー動画（media/preview.py）を送る。
"""

from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components

from ..media.preview import make_preview

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "trimmer")
_component = components.declare_component("video_trimmer", path=_FRONTEND)


@st.cache_resource(show_spinner=False)
def _preview(video: str, mtime: float):
    return make_preview(video)


def _media_url(path: str | None, mimetype: str, coord: str) -> str | None:
    """ファイルを Streamlit の media の URL で配信する（st.video と同じ仕組み）."""
    if not path:
        return None
    from streamlit import runtime

    if not runtime.exists():
        return None
    return runtime.get_instance().media_file_mgr.add(path, mimetype, coord)


def _common(video: str, key: str) -> dict:
    with st.spinner("プレビュー動画を準備中…（初回のみ）"):
        p = _preview(video, os.path.getmtime(video))
    return dict(video=_media_url(p.video, "video/mp4", f"trimmer.{key}.video"),
                strip=_media_url(p.strip, "image/jpeg", f"trimmer.{key}.strip"),
                duration=p.duration, fps=p.fps)


def _changed(key: str, value) -> bool:
    """部品から新しい値が返ってきたか（同じ値の繰り返しは無視）."""
    if value is None or st.session_state.get(f"_{key}_seen") == value:
        return False
    st.session_state[f"_{key}_seen"] = value
    return True


def range_picker(video: str, key: str, empty_labels: tuple[str, str] = ("自動", "自動")
                 ) -> tuple[float | None, float | None]:
    """区間 (開始, 終了) [秒] を選ぶ. 端を動かしていない側は None.

    値は st.session_state[f"{key}_start"], [f"{key}_end"] に入る。ここに値を入れると部品の表示も変わる（自動推定など）。
    """
    sk, ek = f"{key}_start", f"{key}_end"
    v = _component(mode="range", start=st.session_state.get(sk), end=st.session_state.get(ek),
                   empty_labels=list(empty_labels), key=key, default=None, **_common(video, key))
    if _changed(key, v):
        st.session_state[sk], st.session_state[ek] = v.get("start"), v.get("end")
    return st.session_state.get(sk), st.session_state.get(ek)


def time_picker(video: str, key: str, default: float = 0.0) -> float:
    """1 つの時刻 [秒] を選ぶ. 値は st.session_state[key] に入る."""
    v = _component(mode="point", point=st.session_state.get(key, default), key=f"{key}_picker", default=None,
                   **_common(video, key))
    if _changed(key, v):
        st.session_state[key] = float(v.get("t", default))
    return float(st.session_state.get(key, default))
