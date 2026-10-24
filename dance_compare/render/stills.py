"""「直すところ」の瞬間の静止画: お手本と自分を並べ、ズレている部位に印を付ける."""

from __future__ import annotations

import os

import numpy as np

from ..core.preprocess import clean
from ..core.report import Report
from ..core.skeleton import FLIP_INDEX, PoseSequence
from ..media.video import probe, read_frame
from .overlay import REF_COLOR, USER_COLOR, _draw_skeleton, _font

MARK_COLOR = (40, 40, 230)  # BGR 赤


def _crop_box(kp: np.ndarray, w: int, h: int, aspect: float = 0.75) -> tuple[int, int, int, int]:
    """人物の周りを切り出す範囲（縦長 aspect = 幅/高さ）. 関節が取れていなければ画像全体."""
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


def _panel(img: np.ndarray, kp: np.ndarray, color, marks: list[int], height: int, title: str, font):
    import cv2

    h, w = img.shape[:2]
    x0, y0, x1, y1 = _crop_box(kp, w, h)
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
