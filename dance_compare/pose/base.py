"""骨格推定バックエンドの共通インターフェース.

どのバックエンドも「動画 -> TrackSet（人物 ID ごとの関節座標）」を返す。
将来、骨格推定を端末側（ブラウザの MediaPipe / TF.js など）に移す場合は、
端末から TrackSet 相当の JSON を受け取れば以降の処理はそのまま使える。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..core.skeleton import NUM_KEYPOINTS, PoseSequence

ProgressFn = Callable[[int, int], None]


@dataclass
class TrackSet:
    """動画内の全人物の追跡結果.

    keypoints: {id: (T, 17, 2)}  映っていないフレームは NaN
    scores:    {id: (T, 17)}     映っていないフレームは 0
    boxes:     {id: (T, 4)}      x1, y1, x2, y2。映っていないフレームは NaN
    """

    fps: float
    width: int
    height: int
    n_frames: int
    keypoints: dict[int, np.ndarray] = field(default_factory=dict)
    scores: dict[int, np.ndarray] = field(default_factory=dict)
    boxes: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def ids(self) -> list[int]:
        return sorted(self.keypoints)

    def add(self, tid: int, frame: int, kp: np.ndarray, sc: np.ndarray, box: np.ndarray) -> None:
        if tid not in self.keypoints:
            self.keypoints[tid] = np.full((self.n_frames, NUM_KEYPOINTS, 2), np.nan)
            self.scores[tid] = np.zeros((self.n_frames, NUM_KEYPOINTS))
            self.boxes[tid] = np.full((self.n_frames, 4), np.nan)
        if frame >= self.n_frames:
            return
        self.keypoints[tid][frame] = kp
        self.scores[tid][frame] = sc
        self.boxes[tid][frame] = box

    def present(self, tid: int) -> np.ndarray:
        return np.isfinite(self.boxes[tid][:, 0])

    def to_sequence(self, tid: int) -> PoseSequence:
        return PoseSequence(self.keypoints[tid], self.scores[tid], self.fps, self.width, self.height,
                            meta={"track_id": tid})

    @classmethod
    def from_sequence(cls, seq: PoseSequence, tid: int = 1) -> "TrackSet":
        """骨格系列 1 人分から TrackSet を作る（端末側などで推定済みの骨格を使う場合）."""
        ts = cls(seq.fps, seq.width, seq.height, len(seq))
        kp = seq.keypoints
        ts.keypoints[tid] = kp.copy()
        ts.scores[tid] = seq.scores.copy()
        with warnings.catch_warnings():  # 全関節が欠損したフレームの警告を抑える
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(kp, axis=1), np.nanmax(kp, axis=1)
        ts.boxes[tid] = np.concatenate([lo, hi], axis=1)
        return ts

    def trim_frames(self, n: int) -> "TrackSet":
        """実際に読めたフレーム数に合わせて切り詰める."""
        ts = TrackSet(self.fps, self.width, self.height, n)
        for tid in self.ids:
            ts.keypoints[tid] = self.keypoints[tid][:n]
            ts.scores[tid] = self.scores[tid][:n]
            ts.boxes[tid] = self.boxes[tid][:n]
        return ts

    # --- 保存（推定結果のキャッシュ用） ---
    def save(self, path: str) -> None:
        data = {"meta": np.array([self.fps, self.width, self.height, self.n_frames])}
        for tid in self.ids:
            data[f"kp_{tid}"] = self.keypoints[tid]
            data[f"sc_{tid}"] = self.scores[tid]
            data[f"bx_{tid}"] = self.boxes[tid]
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "TrackSet":
        z = np.load(path)
        fps, w, h, n = z["meta"]
        ts = cls(float(fps), int(w), int(h), int(n))
        for key in z.files:
            if key.startswith("kp_"):
                tid = int(key[3:])
                ts.keypoints[tid] = z[key]
                ts.scores[tid] = z[f"sc_{tid}"]
                ts.boxes[tid] = z[f"bx_{tid}"]
        return ts


class PoseBackend(Protocol):
    name: str

    def track(self, video_path: str, progress: ProgressFn | None = None) -> TrackSet:
        ...


def get_backend(name: str = "yolo", **kwargs) -> PoseBackend:
    if name == "yolo":
        from .yolo import YoloPoseBackend
        return YoloPoseBackend(**kwargs)
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend(**kwargs)
    raise ValueError(f"unknown backend: {name}")
