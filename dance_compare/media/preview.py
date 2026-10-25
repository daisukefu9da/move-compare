"""画面で動画を見ながら区間を選ぶための、軽いプレビュー動画とコマの一覧画像.

元の動画（4K・60fps など）はブラウザに送るには重いので、長辺 640px・30fps 以下・音声つきの動画を作る。
キーフレームを細かく入れて、コマ送りや区間の端のドラッグでも素早く表示できるようにする。
プレビューの時刻は元の動画の時刻と同じ（元の動画の t 秒 = プレビューの t 秒）。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .video import probe, read_frame

PREVIEW_DIR = os.environ.get("DANCE_COMPARE_PREVIEW_CACHE", ".cache/preview")


@dataclass
class Preview:
    video: str          # ブラウザで再生する動画（作れなければ None）
    strip: str          # コマの一覧画像（横に並べた JPEG）
    duration: float     # 元の動画の長さ [秒]
    fps: float          # コマ送りの単位（プレビューのフレームレート）
    width: int          # 元の動画の大きさ（回転を反映）
    height: int


def _key(video: str, extra: str) -> str:
    st = os.stat(video)
    return hashlib.sha1(f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{extra}".encode()).hexdigest()[:16]


def make_preview(video: str, max_side: int = 640, max_fps: float = 30.0, thumbs: int = 12,
                 thumb_height: int = 72) -> Preview:
    """プレビュー動画とコマの一覧画像を作る（作成済みなら再利用）."""
    import cv2

    info = probe(video)
    h, w = read_frame(video, 0).shape[:2]
    fps = min(info.fps, max_fps) if info.fps > 0 else max_fps
    key = _key(video, f"{max_side}@{fps}|{thumbs}x{thumb_height}")
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    out = os.path.join(PREVIEW_DIR, f"{key}.mp4")
    if not os.path.exists(out) and shutil.which("ffmpeg"):
        scale = min(1.0, max_side / max(w, h))
        pw, ph = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
        tmp = out + ".part.mp4"
        vopts = ["-vf", f"fps={fps},scale={pw}:{ph}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                 "-g", str(max(1, int(round(fps / 2))))]  # 0.5 秒ごとにキーフレーム（シークを速く）
        audio = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"]
        attempts = [  # ハードウェアでの読み込み → 通常の読み込み → 音声なし（音声の形式が原因のことがある）
            ["-hwaccel", "auto", "-i", video] + vopts + audio,
            ["-i", video] + vopts + audio,
            ["-i", video, "-an"] + vopts,
        ]
        for args in attempts:
            r = subprocess.run(["ffmpeg", "-y", "-v", "error"] + args + ["-movflags", "+faststart", tmp],
                               capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(tmp, out)
                break

    strip = os.path.join(PREVIEW_DIR, f"{key}_strip.jpg")
    if not os.path.exists(strip):
        tw = max(2, int(round(thumb_height * w / h)))
        src = out if os.path.exists(out) else video
        sinfo = probe(src)
        imgs = []
        for i in range(thumbs):
            t = (i + 0.5) / thumbs * info.duration
            try:
                f = read_frame(src, min(int(t * sinfo.fps), max(0, sinfo.frames - 1)))
                imgs.append(cv2.resize(f, (tw, thumb_height), interpolation=cv2.INTER_AREA))
            except Exception:
                imgs.append(np.zeros((thumb_height, tw, 3), np.uint8))
        cv2.imwrite(strip, np.concatenate(imgs, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 80])

    return Preview(out if os.path.exists(out) else None, strip, info.duration, fps, w, h)
