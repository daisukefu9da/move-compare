"""骨格推定結果のキャッシュ（精度調整で何度も解析し直すため）."""

from __future__ import annotations

import hashlib
import os

from .base import ProgressFn, TrackSet

CACHE_DIR = os.environ.get("DANCE_COMPARE_CACHE", ".cache/poses")


def _key(video_path: str, backend_key: str) -> str:
    st = os.stat(video_path)
    raw = f"{os.path.abspath(video_path)}|{st.st_size}|{int(st.st_mtime)}|{backend_key}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def track_cached(backend, video_path: str, progress: ProgressFn | None = None, use_cache: bool = True) -> TrackSet:
    path = os.path.join(CACHE_DIR, _key(video_path, getattr(backend, "cache_key", backend.name)) + ".npz")
    if use_cache and os.path.exists(path):
        return TrackSet.load(path)
    ts = backend.track(video_path, progress)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        ts.save(path)
    return ts
