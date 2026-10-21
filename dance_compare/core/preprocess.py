"""前処理: 欠損補間・平滑化・リサンプリング・左右反転・体格の正規化."""

from __future__ import annotations

import numpy as np

from .skeleton import FLIP_INDEX, KP, PoseSequence


def interpolate_missing(kp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """valid=False の点を時間方向に線形補間する。端は最も近い有効値で埋める.

    kp: (T, J, 2), valid: (T, J)。全フレーム欠損の関節は NaN のまま。
    """
    out = kp.copy()
    T, J, _ = kp.shape
    t = np.arange(T)
    for j in range(J):
        v = valid[:, j] & np.all(np.isfinite(kp[:, j]), axis=1)
        if v.sum() == 0:
            out[:, j] = np.nan
            continue
        for d in range(2):
            out[:, j, d] = np.interp(t, t[v], kp[v, j, d])
    return out


def gaussian_smooth(x: np.ndarray, sigma: float, axis: int = 0) -> np.ndarray:
    """NaN を含まない配列を時間方向にガウシアン平滑化（端は反射）."""
    if sigma <= 0 or x.shape[axis] < 3:
        return x.copy()
    radius = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.moveaxis(x, axis, 0)
    xp = np.pad(x, [(radius, radius)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    flat = xp.reshape(xp.shape[0], -1)
    out = np.stack([np.convolve(flat[:, i], k, mode="valid") for i in range(flat.shape[1])], axis=1)
    out = out.reshape((x.shape[0],) + x.shape[1:])
    return np.moveaxis(out, 0, axis)


def clean(seq: PoseSequence, conf_threshold: float, smooth_sigma_s: float) -> PoseSequence:
    """信頼度の低い関節を補間し、平滑化した PoseSequence を返す."""
    valid = seq.scores >= conf_threshold
    kp = interpolate_missing(seq.keypoints, valid)
    finite = np.all(np.isfinite(kp), axis=(0, 2))
    if finite.any():
        kp[:, finite] = gaussian_smooth(kp[:, finite], smooth_sigma_s * seq.fps)
    return PoseSequence(kp, seq.scores, seq.fps, seq.width, seq.height, seq.t0, dict(seq.meta))


def resample(seq: PoseSequence, fps: float) -> PoseSequence:
    """一定のフレームレートに線形補間でリサンプリングする."""
    if abs(seq.fps - fps) < 1e-6:
        return seq
    n = max(1, int(np.floor(seq.duration * fps)))
    t_new = np.arange(n) / fps
    t_old = seq.times
    kp = np.empty((n,) + seq.keypoints.shape[1:])
    for j in range(kp.shape[1]):
        for d in range(2):
            kp[:, j, d] = np.interp(t_new, t_old, seq.keypoints[:, j, d])
    sc = np.stack([np.interp(t_new, t_old, seq.scores[:, j]) for j in range(seq.scores.shape[1])], axis=1)
    return PoseSequence(kp, sc, fps, seq.width, seq.height, seq.t0, dict(seq.meta))


def mirror(seq: PoseSequence) -> PoseSequence:
    """左右反転（x を反転し、左右の関節を入れ替える）。向かい合わせの手本用."""
    kp = seq.keypoints.copy()
    kp[..., 0] = seq.width - kp[..., 0]
    kp = kp[:, FLIP_INDEX]
    sc = seq.scores[:, FLIP_INDEX]
    meta = dict(seq.meta, mirrored=not seq.meta.get("mirrored", False))
    return PoseSequence(kp, sc, seq.fps, seq.width, seq.height, seq.t0, meta)


def _mid(kp: np.ndarray, a: str, b: str) -> np.ndarray:
    return (kp[:, KP[a]] + kp[:, KP[b]]) / 2


def body_scale(kp: np.ndarray) -> float:
    """体のサイズ: 肩の中点〜腰の中点の距離（胴の長さ）の中央値."""
    torso = np.linalg.norm(_mid(kp, "left_shoulder", "right_shoulder") - _mid(kp, "left_hip", "right_hip"), axis=1)
    torso = torso[np.isfinite(torso) & (torso > 0)]
    if torso.size == 0:
        raise ValueError("胴体の関節が検出できませんでした")
    return float(np.median(torso))


def normalize(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """体格差を吸収した座標を返す. y は上向きを正にする.

    returns:
      body:  (T, J, 2) 毎フレーム腰の中点を原点にし、胴の長さで割った座標（姿勢の比較用）
      world: (T, J, 2) 先頭フレームの腰の中点を原点にし、胴の長さで割った座標（移動・速度用）
      scale: 胴の長さ [px]
    """
    scale = body_scale(kp)
    flip = np.array([1.0, -1.0])
    hip = _mid(kp, "left_hip", "right_hip")
    body = (kp - hip[:, None, :]) / scale * flip
    world = (kp - hip[:1, None, :]) / scale * flip
    return body, world, scale
