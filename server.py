"""サーバーで公開するときの入口（スマホから軽くした動画を受け取る口 /api/upload を app.py に追加する）.

    streamlit run server.py      （手元では今までどおり streamlit run app.py でもよい）
"""

import os

os.environ.setdefault("DANCE_COMPARE_UPLOAD_API", "/api/upload")

import streamlit as st  # noqa: E402
from starlette.routing import Route  # noqa: E402

from dance_compare.ui.upload_api import upload_chunk  # noqa: E402

app = st.App("app.py", routes=[Route("/api/upload", upload_chunk, methods=["POST"])])
