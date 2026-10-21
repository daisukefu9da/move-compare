"""骨格推定結果のキャッシュ（精度調整で何度も解析し直すため）."""

from __future__ import annotations

import hashlib
import os

from ..media.proxy import make_proxy
from .base import ProgressFn, TrackSet

CACHE_DIR = os.environ.get("DANCE_COMPARE_CACHE", ".cache/poses")


def _key(video_path: str, backend_key: str) -> str:
    st = os.stat(video_path)
    raw = f"{os.path.abspath(video_path)}|{st.st_size}|{int(st.st_mtime)}|{backend_key}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def track_cached(backend, video_path: str, progress: ProgressFn | None = None, use_cache: bool = True,
                 max_side: int = 960, max_fps: float = 30.0) -> TrackSet:
    """骨格推定（結果はキャッシュ）. 推定は分析用に縮小・間引きした動画で行い、座標は元の動画の大きさに戻す."""
    key = f"{getattr(backend, 'cache_key', backend.name)}|proxy{max_side}@{max_fps}"
    path = os.path.join(CACHE_DIR, _key(video_path, key) + ".npz")
    if use_cache and os.path.exists(path):
        return TrackSet.load(path)
    px = make_proxy(video_path, max_side, max_fps, progress)
    ts = backend.track(px.path, progress)
    ts = ts.rescaled(px.scale_x, px.scale_y, px.width, px.height, px.source_fps)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        ts.save(path)
    return ts
