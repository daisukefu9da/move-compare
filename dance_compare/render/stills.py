"""「直すところ」の瞬間の静止画: お手本と自分を並べ、ズレている部位に印を付ける."""

from __future__ import annotations

import os

import numpy as np

from ..core.preprocess import clean
from ..core.report import Report
from ..core.skeleton import FLIP_INDEX, PoseSequence
from ..media.video import probe, read_frame
from ..media.video import SequentialReader
from .overlay import REF_COLOR, USER_COLOR, _draw_skeleton, _font, _frame_source, _Writer

MARK_COLOR = (40, 40, 230)  # BGR 赤


def _crop_box(kp: np.ndarray, w: int, h: int, aspect: float = 0.75) -> tuple[int, int, int, int]:
    """人物の周りを切り出す範囲（縦長 aspect = 幅/高さ）. 関節が取れていなければ画像全体.

    kp は (17, 2) か、短い動画の全コマ分 (N, 17, 2)（全コマで人物が収まる範囲にする）。
    """
    kp = kp.reshape(-1, 2)
    pts = kp[np.all(np.isfinite(kp), axis=1)]
    if len(pts) < 4:
        return 0, 0, w, h
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bh = (y1 - y0) * 1.35  # 頭の上（鼻より上）と足元に余白
    bw = (x1 - x0) * 1.3
    bh = max(bh, bw / aspect, 64)
    bw = bh * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * 0.05
    a, b = int(cx - bw / 2), int(cy - bh / 2)
    a, b = max(0, min(a, w - int(bw))), max(0, min(b, h - int(bh)))
    return a, b, min(w, a + int(bw)), min(h, b + int(bh))


def _panel(img: np.ndarray, kp: np.ndarray, color, marks: list[int], height: int, title: str, font,
           box: tuple[int, int, int, int] | None = None):
    import cv2

    h, w = img.shape[:2]
    x0, y0, x1, y1 = box or _crop_box(kp, w, h)
    crop = img[y0:y1, x0:x1]
    s = height / crop.shape[0]
    crop = cv2.resize(crop, (max(2, int(round(crop.shape[1] * s))), height), interpolation=cv2.INTER_AREA)
    k = (kp - np.array([x0, y0])) * s
    _draw_skeleton(crop, k, color, thickness=3)
    r = max(8, int(height * 0.035))  # 印の大きさは画像の高さに合わせる
    for j in marks:
        if np.all(np.isfinite(k[j])):
            c = tuple(int(v) for v in k[j])
            overlay = crop.copy()
            cv2.circle(overlay, c, r, MARK_COLOR, -1, cv2.LINE_AA)
            crop[:] = cv2.addWeighted(overlay, 0.35, crop, 0.65, 0)
            cv2.circle(crop, c, r, MARK_COLOR, 3, cv2.LINE_AA)
    return _text(crop, [(title, font, (255, 255, 255))], (10, 8), stroke=True)


def _text(img, lines, xy, stroke=False, max_width=None):
    """日本語の文字を描く（PIL）. フォントが無ければ何もしない."""
    import cv2

    if not lines or lines[0][1] is None:
        return img
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    x, y = xy
    for text, font, color in lines:
        for line in _wrap(text, font, max_width) if max_width else [text]:
            kw = {"stroke_width": 3, "stroke_fill": (0, 0, 0)} if stroke else {}
            d.text((x, y), line, font=font, fill=color, **kw)
            y = d.textbbox((x, y), line, font=font)[3] + 6
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _wrap(text: str, font, width: int) -> list[str]:
    out, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) > width and cur:
            out.append(cur)
            cur = "　" + ch if out else ch
        else:
            cur += ch
    return out + [cur] if cur else out


def render_moments(report: Report, ref_video: str, user_video: str, ref_seq: PoseSequence,
                   user_seq: PoseSequence, out_dir: str, height: int = 400) -> list[str]:
    """report.moments の各瞬間の静止画を out_dir に書き出し、各 moment に "image"（ファイル名）を入れる."""
    import cv2

    if not report.moments:
        return []
    c = report.meta["config"]
    ref_c = clean(ref_seq, c["conf_threshold"], c["smooth_sigma_s"])
    user_c = clean(user_seq, c["conf_threshold"], c["smooth_sigma_s"])
    mirrored = report.meta.get("mirror_ref", False)
    ref_vfps, user_vfps = probe(ref_video).fps, probe(user_video).fps
    font = _font(max(16, height // 22))
    paths = []
    for m in report.moments:
        t, tu = m["t"], m["user_t"]
        fr = read_frame(ref_video, int(round(t * ref_vfps)))
        fu = read_frame(user_video, int(round(tu * user_vfps)))
        kr = ref_c.keypoints[min(int(round(t * ref_c.fps)), len(ref_c) - 1)]
        ku = user_c.keypoints[min(int(round(tu * user_c.fps)), len(user_c) - 1)]
        marks = list(m["joints"])
        # 部位名はレビュー対象から見た左右. 手本を反転して比べた場合、元の手本の画像では左右が逆になる
        ref_marks = [FLIP_INDEX[j] for j in marks] if mirrored else marks
        pr = _panel(fr, kr, REF_COLOR, ref_marks, height, "お手本", font)
        pu = _panel(fu, ku, USER_COLOR, marks, height, "あなた", font)
        gap = np.full((height, 12, 3), 20, np.uint8)
        img = np.concatenate([pr, gap, pu], axis=1)  # 説明文は画面・report.md で画像の横に出す
        name = f"moment_{m['rank']}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        m["image"] = name
        paths.append(os.path.join(out_dir, name))
    return paths


def render_moment_clips(report: Report, ref_video: str, user_video: str, ref_seq: PoseSequence,
                        user_seq: PoseSequence, out_dir: str, height: int = 360) -> list[str]:
    """直すところの瞬間の前後を切り出した短い動画（スロー, 音声なし）. 各 moment に "clip"（ファイル名）を入れる.

    お手本と自分を人物の周りで切り出して左右に並べ、ズレている部位の印はズレが大きいコマの前後だけ出す。
    """
    if not report.moments:
        return []
    c = report.meta["config"]
    half = c.get("moment_clip_s", 1.5)
    speed = c.get("moment_clip_speed", 0.5)
    ref_c = clean(ref_seq, c["conf_threshold"], c["smooth_sigma_s"])
    user_c = clean(user_seq, c["conf_threshold"], c["smooth_sigma_s"])
    mirrored = report.meta.get("mirror_ref", False)
    # 映像は分析用の軽い動画から読む（パネルは小さいので画質は変わらず、速い）
    ref_src, ref_k = _frame_source(ref_video, height)
    user_src, user_k = _frame_source(user_video, height)
    ref_vfps, user_vfps = probe(ref_src).fps, probe(user_src).fps
    font = _font(max(16, height // 20))
    t_lo = report.meta["ref_t0"]
    t_hi = t_lo + report.meta["ref_duration"]
    fps = ref_c.fps
    paths = []
    for m in report.moments:
        times = np.arange(max(t_lo, m["t"] - half), min(t_hi, m["t"] + half), 1.0 / fps)
        if times.size < 2:
            continue
        utimes = np.asarray(report.user_time(times))
        ri = np.clip(np.round(times * ref_c.fps).astype(int), 0, len(ref_c) - 1)
        ui = np.clip(np.round(utimes * user_c.fps).astype(int), 0, len(user_c) - 1)
        kr_all, ku_all = ref_c.keypoints[ri] * ref_k, user_c.keypoints[ui] * user_k
        rr, ur = SequentialReader(ref_src), SequentialReader(user_src)
        f0r, f0u = rr.get(int(round(times[0] * ref_vfps))), ur.get(int(round(utimes[0] * user_vfps)))
        if f0r is None or f0u is None:
            continue
        box_r = _crop_box(kr_all, f0r.shape[1], f0r.shape[0])  # 全コマで人物が収まる範囲に固定（画面が揺れない）
        box_u = _crop_box(ku_all, f0u.shape[1], f0u.shape[0])
        marks = list(m["joints"])
        ref_marks = [FLIP_INDEX[j] for j in marks] if mirrored else marks
        name = f"moment_{m['rank']}.mp4"
        path = os.path.join(out_dir, name)
        writer = _Writer(path, fps * speed, None, 0, 0)
        try:
            for k, (t, tu) in enumerate(zip(times, utimes)):
                fr = rr.get(int(round(t * ref_vfps)))
                fu = ur.get(int(round(tu * user_vfps)))
                if fr is None or fu is None:
                    break
                near = abs(t - m["t"]) <= 0.35  # ズレが大きいコマの前後だけ印を出す
                pr = _panel(fr, kr_all[k], REF_COLOR, ref_marks if near else [], height, "お手本", font, box_r)
                pu = _panel(fu, ku_all[k], USER_COLOR, marks if near else [], height, "あなた", font, box_u)
                gap = np.full((height, 8, 3), 20, np.uint8)
                frame = np.concatenate([pr, gap, pu], axis=1)
                frame = frame[:, : frame.shape[1] // 2 * 2]
                writer.write(frame)
        finally:
            rr.close()
            ur.close()
            writer.close()
        m["clip"] = name
        paths.append(path)
    return paths
