"""正規化済み座標から比較用の特徴量を計算する."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import gaussian_smooth
from .skeleton import ANGLE_DEFS, ANGLE_NAMES, BODY_JOINTS, END_EFFECTORS, KP


def joint_angles(body: np.ndarray) -> np.ndarray:
    """(T, J, 2) -> (T, len(ANGLE_NAMES)) 関節角度 [度]. 180 で真っ直ぐ."""
    out = np.empty((body.shape[0], len(ANGLE_NAMES)))
    for k, name in enumerate(ANGLE_NAMES):
        a, b, c = ANGLE_DEFS[name]
        v1 = body[:, a] - body[:, b]
        v2 = body[:, c] - body[:, b]
        cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
        out[:, k] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out


def facing_angle(body: np.ndarray) -> np.ndarray:
    """体の向き [度] の推定. 0=カメラ正面, 90=真横, 180=背中.

    2D なので左右どちらに回ったかは区別できない。肩幅の見かけの縮み具合と、
    左右の肩の並び順（背中向きだと逆になる）から推定する。
    """
    ls, rs = body[:, KP["left_shoulder"]], body[:, KP["right_shoulder"]]
    lh, rh = body[:, KP["left_hip"]], body[:, KP["right_hip"]]
    width = np.abs(ls[:, 0] - rs[:, 0]) + np.abs(lh[:, 0] - rh[:, 0])
    wmax = np.percentile(width, 95) + 1e-9
    theta = np.degrees(np.arccos(np.clip(width / wmax, 0, 1)))
    # 正面向きなら本人の左肩は画像の右側（x が大きい）にある
    back = (ls[:, 0] + lh[:, 0]) < (rs[:, 0] + rh[:, 0])
    return np.where(back, 180 - theta, theta)


def torso_lean(body: np.ndarray) -> np.ndarray:
    """上体の傾き [度]. 0=直立, 正=画面右へ傾く."""
    sh = (body[:, KP["left_shoulder"]] + body[:, KP["right_shoulder"]]) / 2
    return np.degrees(np.arctan2(sh[:, 0], sh[:, 1]))  # 腰が原点


def level(body: np.ndarray) -> np.ndarray:
    """体の高さ: 足首から腰までの高さ（胴の長さ比）. 小さいほど沈み込んでいる."""
    ank = np.minimum(body[:, KP["left_ankle"], 1], body[:, KP["right_ankle"], 1])
    return -ank


def stance_width(body: np.ndarray) -> np.ndarray:
    """足幅（胴の長さ比）."""
    return np.abs(body[:, KP["left_ankle"], 0] - body[:, KP["right_ankle"], 0])


def velocity(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(x, 1.0 / fps, axis=0)


@dataclass
class Features:
    """1 人分の特徴量（すべて analysis_fps のフレーム単位）."""

    fps: float
    body: np.ndarray          # (T, J, 2)
    world: np.ndarray         # (T, J, 2)
    angles: np.ndarray        # (T, A) [度]
    facing: np.ndarray        # (T,)
    lean: np.ndarray          # (T,)
    level: np.ndarray         # (T,)
    stance: np.ndarray        # (T,)
    hip: np.ndarray           # (T, 2) 腰の中点の位置（world 座標）
    end_speed: np.ndarray     # (T, 4) 手首・足首の速さ [胴長/秒]
    end_accel: np.ndarray     # (T, 4) 手首・足首の加速度の大きさ
    body_speed: np.ndarray    # (T,) 全身の動きの速さ（アクセント検出用）

    @property
    def n(self) -> int:
        return self.body.shape[0]

    def dtw_vectors(self) -> np.ndarray:
        """DTW で使う姿勢ベクトル: 主要関節の体中心座標 + 角度（スケールを合わせる）."""
        pose = self.body[:, BODY_JOINTS].reshape(self.n, -1)
        ang = np.radians(self.angles) * 0.5
        return np.concatenate([pose, ang], axis=1)


def compute_features(body: np.ndarray, world: np.ndarray, fps: float) -> Features:
    ends = list(END_EFFECTORS.values())
    hip = (world[:, KP["left_hip"]] + world[:, KP["right_hip"]]) / 2
    # 速度・加速度は微分でノイズが増えるので追加で平滑化する
    v = velocity(gaussian_smooth(world, 0.03 * fps), fps)
    a = velocity(gaussian_smooth(v, 0.03 * fps), fps)
    end_speed = np.linalg.norm(v[:, ends], axis=2)
    end_accel = np.linalg.norm(a[:, ends], axis=2)
    body_speed = np.mean(np.linalg.norm(v[:, BODY_JOINTS], axis=2), axis=1)
    return Features(
        fps=fps, body=body, world=world,
        angles=joint_angles(body), facing=facing_angle(body), lean=torso_lean(body),
        level=level(body), stance=stance_width(body), hip=hip,
        end_speed=end_speed, end_accel=end_accel, body_speed=body_speed,
    )
