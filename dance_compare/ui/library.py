"""保存した手本の動画（レッスンで何度も使う手本を、毎回アップロードしなくて済むようにする）.

手本は名前を付けて MODEL_DIR に置く。骨格推定の結果はファイルの場所・大きさ・更新時刻で
キャッシュされるので、同じ手本を選び直せば骨格推定もやり直さない。
"""

from __future__ import annotations

import os
import re
import shutil

MODEL_DIR = os.environ.get("DANCE_COMPARE_MODEL_DIR", ".cache/model_videos")
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def _safe(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name.strip())
    return name.strip(". ")[:60]


def list_models() -> dict[str, str]:
    """{名前: 動画のパス}（新しい順）."""
    if not os.path.isdir(MODEL_DIR):
        return {}
    files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
             if not f.startswith(".") and os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    files = sorted((f for f in files if os.path.isfile(f)), key=os.path.getmtime, reverse=True)
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


def save_model(video: str, name: str) -> str:
    """動画を名前を付けて手本として保存し、保存先のパスを返す（同じ名前は上書き）."""
    name = _safe(name)
    if not name:
        raise ValueError("名前を入れてください")
    os.makedirs(MODEL_DIR, exist_ok=True)
    for old in list_models().items():
        if old[0] == name:
            os.remove(old[1])
    dst = os.path.join(MODEL_DIR, name + os.path.splitext(video)[1].lower())
    try:
        os.link(video, dst)  # 同じディスクならコピーしない
    except OSError:
        shutil.copy2(video, dst)
    return dst


def delete_model(name: str) -> None:
    path = list_models().get(name)
    if path:
        os.remove(path)
