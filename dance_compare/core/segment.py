"""比較する区間の自動推定（1動画モード用）.

踊る前も各自リズムを取っていて完全には止まらないことが多いので、「動いているか」ではなく
「二人の動きがそろっているか（同期度）」で振付の始まりと終わりを推定する。
各自で揺れている間は動きがバラバラで同期度が低く、振付が始まると同じ動きになって高くなる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AnalysisConfig
from .features import velocity
from .preprocess import clean, gaussian_smooth, mirror, normalize, resample
from .skeleton import BODY_JOINTS, PoseSequence


@dataclass
class SyncCurve:
    times: np.ndarray        # 各点の時刻（系列の先頭が 0）
    score: np.ndarray        # 同期度 0〜1（1 = 同じ動き）
    start: float | None      # 推定した開始・終了（同期した区間が無ければ None）
    end: float | None


def _motion(seq: PoseSequence, cfg: AnalysisConfig, fps: float) -> np.ndarray:
    """体格で正規化した関節の速度 (T, 24)."""
    s = resample(clean(seq, cfg.conf_threshold, cfg.smooth_sigma_s), fps)
    body, _, _ = normalize(s.keypoints)
    v = velocity(gaussian_smooth(body[:, BODY_JOINTS], 0.05 * fps), fps)
    return np.nan_to_num(v.reshape(v.shape[0], -1))


def sync_curve(ref: PoseSequence, user: PoseSequence, cfg: AnalysisConfig | None = None, *,
               mirror_ref: bool = False, fps: float = 15.0) -> SyncCurve:
    """二人の動きの同期度の時系列と、同期している区間の推定.

    各時刻の前後 sync_window_s 秒で、関節の速度ベクトルの向きがどれだけそろっているか（コサイン類似度）を、
    ±sync_max_lag_s のずれを許して最大値で求める。
    """
    cfg = cfg or AnalysisConfig()
    if mirror_ref:
        ref = mirror(ref)
    a, b = _motion(ref, cfg, fps), _motion(user, cfg, fps)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    half = max(1, int(cfg.sync_window_s * fps / 2))
    lags = range(-int(cfg.sync_max_lag_s * fps), int(cfg.sync_max_lag_s * fps) + 1)
    k = np.ones(2 * half + 1)
    best = np.zeros(n)
    for lag in lags:
        bb = np.roll(b, lag, axis=0)
        if lag > 0:
            bb[:lag] = 0
        elif lag < 0:
            bb[lag:] = 0
        dot = np.convolve(np.sum(a * bb, axis=1), k, mode="same")
        na = np.convolve(np.sum(a * a, axis=1), k, mode="same")
        nb = np.convolve(np.sum(bb * bb, axis=1), k, mode="same")
        best = np.maximum(best, dot / np.sqrt(na * nb + 1e-9))
    # 動きが小さい区間は同期度を下げる. 二人が同じようにリズムを取っているだけでも動きはそろうので、
    # 動画の中で大きく動いている部分（振付とみなす）と比べて小さい動きは、振付ではないとみなす
    moving = np.minimum(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    moving = np.convolve(moving, k / k.sum(), mode="same")
    ref_level = max(float(np.percentile(moving, 80)) * cfg.sync_relative_motion, cfg.sync_min_motion)
    score = best * np.clip(moving / ref_level, 0, 1)
    t = np.arange(n) / fps

    on = score >= cfg.sync_threshold
    min_len = int(cfg.sync_min_duration_s * fps)
    runs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return SyncCurve(t, score, None, None)
    # 窓の幅の分だけ同期度の立ち上がりは早く・終わりは遅くなるので、窓の半分だけ内側に寄せる
    start = max(0.0, (runs[0][0] + half) / fps)
    end = min(n / fps, (runs[-1][1] - half) / fps)
    if end <= start:
        start, end = runs[0][0] / fps, runs[-1][1] / fps
    return SyncCurve(t, score, start, end)
