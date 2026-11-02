"""骨格推定を別のサーバー（Modal の GPU）で行うバックエンド.

画面（Streamlit）は CPU のサーバーで動かし、骨格推定のときだけ GPU を借りる。GPU が混んでいて
なかなか空かない場合は、待つのをやめてこのサーバーの CPU で推定する（遅いが止まらない）。

環境変数 DANCE_COMPARE_REMOTE_POSE="アプリ名/関数名"（例: dance-compare/track_pose）で有効になる。
"""

from __future__ import annotations

import io
import os
import time

from .base import ProgressFn, TrackSet
from .yolo import YoloPoseBackend

LAST_NOTE: str | None = None  # 直近の推定で GPU を使えなかった場合の説明（画面に出す）


def remote_target() -> str | None:
    return os.environ.get("DANCE_COMPARE_REMOTE_POSE") or None


def _function(target: str):
    import modal

    app_name, fn_name = target.split("/", 1)
    return modal.Function.from_name(app_name, fn_name)


def warm(target: str | None = None) -> None:
    """GPU のサーバーを先に起こしておく（動画を選んだ時点で呼ぶと、推定を始める時の待ちが減る）."""
    target = target or remote_target()
    if not target:
        return
    try:
        _function(target).spawn(b"", "")
    except Exception:  # 起こせなくても推定の時にもう一度呼ぶので無視する
        pass


class RemotePoseBackend:
    """YOLO-pose を GPU のサーバーで動かす（結果は YoloPoseBackend と同じ）."""

    name = "yolo"

    def __init__(self, target: str, max_wait_s: float = 150.0, **kwargs):
        self.target = target
        self.max_wait_s = max_wait_s
        self.local = YoloPoseBackend(**kwargs)
        self.cache_key = self.local.cache_key  # 同じ結果なので、キャッシュも共通

    def track(self, video_path: str, progress: ProgressFn | None = None) -> TrackSet:
        global LAST_NOTE
        import modal

        LAST_NOTE = None
        with open(video_path, "rb") as f:
            data = f.read()  # 分析用に縮小した動画（数十 MB 程度）
        try:
            call = _function(self.target).spawn(data, os.path.splitext(video_path)[1])
        except Exception as e:
            LAST_NOTE = f"GPU のサーバーに頼めなかったため、CPU で推定しました（{e}）"
            return self.local.track(video_path, progress)
        t0 = time.time()
        while True:
            try:
                out = call.get(timeout=5)
                break
            except (TimeoutError, modal.exception.TimeoutError):
                if time.time() - t0 > self.max_wait_s:
                    call.cancel()
                    LAST_NOTE = (f"GPU が混んでいて {self.max_wait_s:.0f} 秒待っても空かなかったため、CPU で推定しました"
                                 "（いつもより時間がかかります）")
                    return self.local.track(video_path, progress)
        return TrackSet.load(io.BytesIO(out))


def track_bytes(data: bytes, suffix: str) -> bytes:
    """GPU のサーバー側: 動画のバイト列 → 骨格推定 → TrackSet（npz のバイト列）. 空なら起こすだけ."""
    import tempfile

    if not data:
        import numpy as np
        from ultralytics import YOLO

        from .yolo import auto_device

        # ライブラリ・モデル・GPU の準備（初回の推定にかかる数秒）を先に済ませておく
        device = auto_device()
        YOLO(YoloPoseBackend().model_name).predict(np.zeros((640, 640, 3), np.uint8), device=device,
                                                   half=device == "cuda", verbose=False)
        return b""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4") as f:
        f.write(data)
        f.flush()
        ts = YoloPoseBackend().track(f.name)
    buf = io.BytesIO()
    ts.save(buf)
    return buf.getvalue()
