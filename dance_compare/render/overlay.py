"""骨格を重ねた比較動画を作る.

[手本] [レビュー対象（ズレの大きい関節を赤く表示）] [正規化した骨格の重ね合わせ]
の 3 画面を横に並べ、下にカウントと現在の指摘を表示する。手本の時間軸で再生し、手本の音声を付ける。
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

from ..core.preprocess import clean
from ..core.preprocess import mirror as mirror_seq
from ..core.preprocess import normalize
from ..core.report import Report
from ..core.skeleton import ANGLE_DEFS, ANGLE_NAMES, BONES, PoseSequence
from ..feedback.phrases import describe
from ..media.video import SequentialReader

FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

REF_COLOR = (80, 200, 80)      # BGR
USER_COLOR = (0, 160, 255)
BAD_COLOR = (40, 40, 230)


def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    path = os.environ.get("DANCE_COMPARE_FONT")
    for p in ([path] if path else []) + FONT_CANDIDATES:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return None


def _draw_skeleton(img, kp, color, thickness=2, bad: set[int] = frozenset(), scale=1.0, offset=(0, 0)):
    import cv2

    pts = kp * scale + np.array(offset)
    for a, b in BONES:
        if np.all(np.isfinite(pts[[a, b]])):
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, thickness, cv2.LINE_AA)
    for j in range(pts.shape[0]):
        if np.all(np.isfinite(pts[j])):
            c = BAD_COLOR if j in bad else color
            r = 7 if j in bad else 3
            cv2.circle(img, tuple(int(v) for v in pts[j]), r, c, -1, cv2.LINE_AA)


def _fit(img, h):
    import cv2

    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)) // 2 * 2, h)), s


def render_comparison(
    report: Report,
    ref_video: str,
    user_video: str,
    ref_seq: PoseSequence,
    user_seq: PoseSequence,
    out_path: str,
    panel_height: int = 480,
    angle_threshold: float = 20.0,
    progress=None,
    notes: list[str] | None = None,
) -> str:
    """比較動画を out_path に書き出す. 注意事項（音声なしにした等）は notes に追加する.

    ref_seq / user_seq は元動画のフレームと 1 対 1 に対応する（切り出し前の）骨格系列。
    """
    c = report.meta["config"]
    ref_seq = clean(ref_seq, c["conf_threshold"], c["smooth_sigma_s"])
    user_seq = clean(user_seq, c["conf_threshold"], c["smooth_sigma_s"])
    mirrored = report.meta.get("mirror_ref", False)
    ref_body, _, _ = normalize((mirror_seq(ref_seq) if mirrored else ref_seq).keypoints)
    user_body, _, _ = normalize(user_seq.keypoints)

    fps = ref_seq.fps
    t0 = report.meta["ref_t0"]
    t1 = t0 + report.meta["ref_duration"]
    times = np.arange(t0, t1, 1.0 / fps)
    grid = report.count_grid()
    tl = report.timeline
    tl_t = np.asarray(tl["t"])
    angle_tl = {n: np.asarray(tl["angle_diff"][n]) for n in ANGLE_NAMES}

    H = panel_height
    bar_h = max(120, H // 4)
    font = _font(max(16, H // 24))
    font_big = _font(max(20, H // 16))
    rr = SequentialReader(ref_video)
    ur = SequentialReader(user_video)
    writer = _Writer(out_path, fps, ref_video, t0, t1 - t0)
    try:
        for k, t in enumerate(times):
            fi_r = min(int(round(t * fps)), len(ref_seq) - 1)
            tu = float(report.user_time(t))
            fi_u = min(max(int(round(tu * user_seq.fps)), 0), len(user_seq) - 1)
            fr = rr.get(fi_r)
            fu = ur.get(fi_u)
            if fr is None or fu is None:
                break
            bad = set()
            for n in ANGLE_NAMES:
                if abs(np.interp(t, tl_t, angle_tl[n])) >= angle_threshold:
                    bad.add(ANGLE_DEFS[n][1])
            pr, sr = _fit(fr.copy(), H)
            _draw_skeleton(pr, ref_seq.keypoints[fi_r], REF_COLOR, scale=sr)
            pu, su = _fit(fu.copy(), H)
            _draw_skeleton(pu, user_seq.keypoints[fi_u], USER_COLOR, bad=bad, scale=su)
            po = np.full((H, H, 3), 30, np.uint8)
            unit = H / 5.0
            to_px = np.array([unit, -unit])
            _draw_skeleton(po, ref_body[fi_r] * to_px, REF_COLOR, 3, offset=(H / 2, H * 0.55))
            _draw_skeleton(po, user_body[fi_u] * to_px, USER_COLOR, 2, bad=bad, offset=(H / 2, H * 0.55))
            frame = np.concatenate([pr, pu, po], axis=1)
            bar = np.full((bar_h, frame.shape[1], 3), 20, np.uint8)
            frame = np.concatenate([frame, bar], axis=0)
            label = grid.label(t) if grid is not None else ""
            active = [f for f in report.findings if f.t_start <= t < f.t_end][:2]
            texts = [(f"{label}  {_fmt(t)}", font_big, (255, 255, 255))]
            texts += [("・" + describe(f), font, (120, 170, 255)) for f in active]
            frame = _put_texts(frame, texts, H + 6, legend=(pr.shape[1], pu.shape[1]), font=font)
            writer.write(frame)
            if progress and k % 10 == 0:
                progress(k, len(times))
    finally:
        rr.close()
        ur.close()
        writer.close()
    if notes is not None:
        notes.extend(writer.notes)
    return out_path


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def _put_texts(frame, texts, y0, legend, font):
    """日本語を描画（PIL があれば）。無ければ ASCII 部分だけ OpenCV で描く."""
    import cv2

    if font is None:
        cv2.putText(frame, texts[0][0].encode("ascii", "ignore").decode(), (10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
    from PIL import Image, ImageDraw

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.text((10, 8), "お手本", font=font, fill=(80, 200, 80), stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((legend[0] + 10, 8), "レビュー対象", font=font, fill=(255, 160, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((legend[0] + legend[1] + 10, 8), "重ね合わせ", font=font, fill=(230, 230, 230))
    y = y0
    for text, f, color in texts:
        d.text((10, y), text, font=f, fill=color)
        y = d.textbbox((10, y), text, font=f)[3] + 6
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _ffmpeg_encoders() -> set[str]:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except OSError:
        return set()
    return {line.split()[1] for line in out.splitlines() if len(line.split()) > 1 and line.startswith(" V")}


def _tail(path: str, n: int = 15) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


class _Writer:
    """比較動画の書き出し.

    ffmpeg があれば、まず映像だけを H.264 で書き出し、そのあと手本の音声を付ける。
    音声を付けられない場合（iPhone の空間オーディオなど ffmpeg が読めない形式）は音声なしにする。
    ffmpeg が無ければ OpenCV の mp4v（音声なし）で書き出す。
    """

    def __init__(self, path, fps, audio_src, start, duration):
        self.path, self.fps = path, fps
        self.audio = (audio_src, start, duration)
        self.proc = None
        self.cv = None
        self.video_path = path + ".video.mp4"
        self.log_path = path + ".ffmpeg.log"
        self.notes: list[str] = []

    def _start(self, frame):
        import cv2

        h, w = frame.shape[:2]
        if shutil.which("ffmpeg") is None:
            self.cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.notes.append("ffmpeg が無いため、比較動画は音声なしで書き出しました")
            return
        enc = _ffmpeg_encoders()
        if "libx264" in enc:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        elif "h264_videotoolbox" in enc:  # Mac の標準ハードウェアエンコーダ
            codec = ["-c:v", "h264_videotoolbox", "-b:v", "6M"]
        else:
            codec = ["-c:v", "mpeg4", "-q:v", "4"]
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps}", "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 は縦横が偶数でないと書き出せない
               *codec, "-pix_fmt", "yuv420p", self.video_path]
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame):
        if self.proc is None and self.cv is None:
            self._start(frame)
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                self.proc.wait()
                self.proc = None  # close() で二重にエラーにしない
                self.log.close()
                raise RuntimeError("ffmpeg での動画の書き出しに失敗しました:\n" + _tail(self.log_path)) from None
        else:
            self.cv.write(frame)

    def close(self):
        if self.cv is not None:
            self.cv.release()
            return
        if self.proc is None:
            return
        self.proc.stdin.close()
        code = self.proc.wait()
        self.log.close()
        if code != 0:
            raise RuntimeError("ffmpeg での動画の書き出しに失敗しました:\n" + _tail(self.log_path))
        src, start, dur = self.audio
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", self.video_path, "-ss", f"{start}", "-t", f"{dur}", "-i", src,
               "-map", "0:v", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", self.path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            os.remove(self.video_path)
        else:
            os.replace(self.video_path, self.path)
            msg = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
            self.notes.append(f"手本の音声を比較動画に付けられなかったため、音声なしで書き出しました（{msg}）")
        os.remove(self.log_path)
