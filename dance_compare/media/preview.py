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


@dataclass
class _Plan:
    out: str
    key: str
    vf: str
    vopts: list[str]   # 映像の出力設定（-vf 以外）
    w: int
    h: int


AUDIO_OUTPUT = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"]


def _plan(video: str, max_side: int = 640, max_fps: float = 30.0, thumbs: int = 12, thumb_height: int = 72) -> _Plan:
    info = probe(video)
    h, w = read_frame(video, 0).shape[:2]
    fps = min(info.fps, max_fps) if info.fps > 0 else max_fps
    key = _key(video, f"{max_side}@{fps}|{thumbs}x{thumb_height}")
    scale = min(1.0, max_side / max(w, h))
    pw, ph = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    vopts = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
             "-g", str(max(1, int(round(fps / 2))))]  # 0.5 秒ごとにキーフレーム（シークを速く）
    return _Plan(os.path.join(PREVIEW_DIR, f"{key}.mp4"), key, f"fps={fps},scale={pw}:{ph}", vopts, w, h)


def _ffmpeg(attempts: list[list[str]]) -> bool:
    for args in attempts:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, capture_output=True, text=True)
        if r.returncode == 0:
            return True
    return False


def prepare_media(video: str) -> None:
    """分析用の軽い動画（media/proxy）とプレビュー動画を、動画を 1 回読むだけでまとめて作る.

    iPhone の 4K 動画は読み込み（デコード）が重いので、別々に作ると 2 倍かかる。作成済みなら何もしない。
    """
    from .proxy import PROXY_OUTPUT, proxy_plan

    pp, pv = proxy_plan(video), _plan(video)
    if pp.out and not os.path.exists(pp.out) and not os.path.exists(pv.out) and shutil.which("ffmpeg"):
        os.makedirs(os.path.dirname(pp.out), exist_ok=True)
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        ptmp, vtmp = pp.out + ".part.mp4", pv.out + ".part.mp4"
        fc = f"[0:v]split=2[a][b];[a]{pp.vf}[p];[b]{pv.vf}[q]"
        outs = ["-filter_complex", fc, "-map", "[p]", *PROXY_OUTPUT, ptmp,
                "-map", "[q]", *pv.vopts, "-movflags", "+faststart"]
        ok = _ffmpeg([
            ["-hwaccel", "auto", "-i", video, *outs, "-map", "0:a?", *AUDIO_OUTPUT, vtmp],
            ["-i", video, *outs, "-map", "0:a?", *AUDIO_OUTPUT, vtmp],
            ["-i", video, *outs, vtmp],  # 音声の形式が原因で失敗する場合は音声なし
        ])
        if ok:
            os.replace(ptmp, pp.out)
            os.replace(vtmp, pv.out)
    # どちらかだけ足りない・まとめて作れなかった場合は、それぞれ作る（作成済みならすぐ返る）
    from .proxy import make_proxy

    make_proxy(video)
    make_preview(video)


def make_preview(video: str, max_side: int = 640, max_fps: float = 30.0, thumbs: int = 12,
                 thumb_height: int = 72) -> Preview:
    """プレビュー動画とコマの一覧画像を作る（作成済みなら再利用）."""
    import cv2

    info = probe(video)
    p = _plan(video, max_side, max_fps, thumbs, thumb_height)
    out, key, w, h = p.out, p.key, p.w, p.h
    fps = min(info.fps, max_fps) if info.fps > 0 else max_fps
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    if not os.path.exists(out) and shutil.which("ffmpeg"):
        tmp = out + ".part.mp4"
        tail = ["-movflags", "+faststart", tmp]
        if _ffmpeg([  # ハードウェアでの読み込み → 通常の読み込み → 音声なし（音声の形式が原因のことがある）
            ["-hwaccel", "auto", "-i", video, "-vf", p.vf, *p.vopts, *AUDIO_OUTPUT, *tail],
            ["-i", video, "-vf", p.vf, *p.vopts, *AUDIO_OUTPUT, *tail],
            ["-i", video, "-an", "-vf", p.vf, *p.vopts, *tail],
        ]):
            os.replace(tmp, out)

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
