"""分析用の軽い動画（プロキシ）を作る.

骨格推定は内部で 640px 程度に縮小して行うので、4K・60fps の動画をそのまま読むのは無駄が大きい
（1 コマずつの読み込み・縮小が CPU の大半を占める）。ffmpeg で一度だけ縮小・間引きした動画を作り、
骨格推定はそれで行う。座標は元の動画の解像度に戻すので、比較動画などは元の動画から作れる。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass

from .video import probe, read_frame

PROXY_DIR = os.environ.get("DANCE_COMPARE_PROXY_CACHE", ".cache/proxy")


@dataclass
class Proxy:
    path: str           # 分析に使う動画（縮小不要なら元の動画そのもの）
    scale_x: float      # 分析用の座標 × scale = 元の動画の座標
    scale_y: float
    width: int          # 元の動画の大きさ
    height: int
    source_fps: float   # 元の動画のフレームレート

    @property
    def is_original(self) -> bool:
        return self.scale_x == 1.0 and self.scale_y == 1.0


def make_proxy(video: str, max_side: int = 960, max_fps: float = 30.0, progress=None) -> Proxy:
    """長辺 max_side・max_fps 以下の分析用動画を作る（作成済みなら再利用）."""
    info = probe(video)
    # 回転情報（縦向きの iPhone 動画など）を反映した実際のコマの大きさを使う
    h, w = read_frame(video, 0).shape[:2]
    scale = min(1.0, max_side / max(w, h))
    fps = min(info.fps, max_fps) if info.fps > 0 else max_fps
    original = Proxy(video, 1.0, 1.0, w, h, info.fps)
    if (scale >= 1.0 and info.fps <= max_fps + 0.5) or shutil.which("ffmpeg") is None:
        return original
    pw, ph = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    st = os.stat(video)
    key = hashlib.sha1(f"{os.path.abspath(video)}|{st.st_size}|{int(st.st_mtime)}|{pw}x{ph}@{fps}".encode()).hexdigest()[:16]
    out = os.path.join(PROXY_DIR, f"{key}.mp4")
    if not os.path.exists(out):
        os.makedirs(PROXY_DIR, exist_ok=True)
        tmp = out + ".part.mp4"
        base = ["ffmpeg", "-y", "-v", "error"]
        rest = ["-i", video, "-an", "-vf", f"fps={fps},scale={pw}:{ph}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", tmp]
        if progress:
            progress(0, 1)
        # まずハードウェアでの読み込み（Mac の VideoToolbox など）を試し、だめなら通常の読み込み
        r = subprocess.run(base + ["-hwaccel", "auto"] + rest, capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run(base + rest, capture_output=True, text=True)
        if r.returncode != 0:
            return original  # 作れなければ元の動画で推定する（遅いが動く）
        os.replace(tmp, out)
        if progress:
            progress(1, 1)
    # 実際にできた大きさから倍率を求める
    ph2, pw2 = read_frame(out, 0).shape[:2]
    return Proxy(out, w / pw2, h / ph2, w, h, info.fps)
