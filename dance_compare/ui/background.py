"""動画を受け取ったらすぐ、裏で下ごしらえ（軽い動画の作成 → 骨格推定）を始める.

先生がお手本・区間を選んでいる間に進めておき、「人物を検出」「解析する」での待ち時間を減らす。
同じ動画の下ごしらえは 1 回だけ行い、画面側は wait() で終わるのを待つ（終わっていればすぐ返る）。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prepare")
_lock = threading.Lock()
_media: dict[str, Future] = {}
_pose: dict[tuple[str, str], Future] = {}


def log_time(stage: str, seconds: float, video: str = "") -> None:
    """かかった時間をログに出す（サーバー版の速さの確認用）."""
    print(f"[時間] {stage}: {seconds:.1f} 秒 {video}", flush=True)


def _prepare_media(video: str) -> None:
    from ..media.preview import prepare_media

    t = time.time()
    prepare_media(video)
    log_time("軽い動画の作成", time.time() - t, video)


def _pose_job(video: str, backend: str) -> None:
    from ..pipeline import extract_tracks

    wait_media(video)
    t = time.time()
    extract_tracks(video, backend)
    log_time("骨格推定", time.time() - t, video)


def start(video: str, backend: str = "yolo") -> None:
    """下ごしらえを始める（始めていれば何もしない）."""
    with _lock:
        if video not in _media:
            _media[video] = _pool.submit(_prepare_media, video)
        if (video, backend) not in _pose:
            _pose[(video, backend)] = _pool.submit(_pose_job, video, backend)


def _wait(f: Future | None) -> None:
    if f is None:
        return
    try:
        f.result()
    except Exception:  # 裏で失敗した場合は、画面側の処理でもう一度行う（エラーはそこで表示される）
        pass


def wait_media(video: str) -> None:
    _wait(_media.get(video))


def wait(video: str, backend: str = "yolo") -> None:
    """骨格推定まで終わるのを待つ."""
    _wait(_media.get(video))
    _wait(_pose.get((video, backend)))
