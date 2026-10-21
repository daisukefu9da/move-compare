"""骨格の定義と、関節座標の時系列を表すデータ構造.

関節の並びは COCO 17 点形式に統一する（YOLO-pose はそのまま、MediaPipe は変換して使う）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

KEYPOINTS = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
NUM_KEYPOINTS = len(KEYPOINTS)

# 左右反転時に入れ替える関節の組
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = list(range(NUM_KEYPOINTS))
for _a, _b in FLIP_PAIRS:
    FLIP_INDEX[_a], FLIP_INDEX[_b] = _b, _a

# 描画用の骨（線分）
BONES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

# 体幹より下の、比較に使う主要関節（顔は向きの推定にしか使わない）
BODY_JOINTS = list(range(5, 17))

# 関節角度: 名前 -> (a, b, c)  b を頂点とする角 a-b-c
ANGLE_DEFS = {
    "left_elbow": (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]),
    "right_elbow": (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]),
    "left_shoulder": (KP["left_hip"], KP["left_shoulder"], KP["left_elbow"]),
    "right_shoulder": (KP["right_hip"], KP["right_shoulder"], KP["right_elbow"]),
    "left_hip": (KP["left_shoulder"], KP["left_hip"], KP["left_knee"]),
    "right_hip": (KP["right_shoulder"], KP["right_hip"], KP["right_knee"]),
    "left_knee": (KP["left_hip"], KP["left_knee"], KP["left_ankle"]),
    "right_knee": (KP["right_hip"], KP["right_knee"], KP["right_ankle"]),
}
ANGLE_NAMES = list(ANGLE_DEFS)

# 動きの大きさ・キレを見る末端の関節
END_EFFECTORS = {
    "left_wrist": KP["left_wrist"],
    "right_wrist": KP["right_wrist"],
    "left_ankle": KP["left_ankle"],
    "right_ankle": KP["right_ankle"],
}

# 表示用の日本語名
JA_NAMES = {
    "left_elbow": "左肘", "right_elbow": "右肘",
    "left_shoulder": "左肩", "right_shoulder": "右肩",
    "left_hip": "左股関節", "right_hip": "右股関節",
    "left_knee": "左膝", "right_knee": "右膝",
    "left_wrist": "左手", "right_wrist": "右手",
    "left_ankle": "左足", "right_ankle": "右足",
}


@dataclass
class PoseSequence:
    """1 人分の関節座標の時系列.

    keypoints: (T, 17, 2) 画像のピクセル座標 (x 右向き, y 下向き)。欠損は NaN。
    scores:    (T, 17) 各関節の信頼度 (0-1)。
    fps:       フレームレート。
    width/height: 元動画の解像度（左右反転などに使う）。
    t0:        先頭フレームの元動画上の時刻 [秒]（切り出した場合のオフセット）。
    """

    keypoints: np.ndarray
    scores: np.ndarray
    fps: float
    width: int
    height: int
    t0: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.keypoints.ndim != 3 or self.keypoints.shape[1:] != (NUM_KEYPOINTS, 2):
            raise ValueError(f"keypoints must be (T, {NUM_KEYPOINTS}, 2), got {self.keypoints.shape}")
        if self.scores.shape != self.keypoints.shape[:2]:
            raise ValueError("scores must be (T, 17)")

    def __len__(self) -> int:
        return self.keypoints.shape[0]

    @property
    def duration(self) -> float:
        return len(self) / self.fps

    @property
    def times(self) -> np.ndarray:
        """先頭フレームを 0 とした各フレームの時刻."""
        return np.arange(len(self)) / self.fps

    def trim(self, start: float | None = None, end: float | None = None) -> "PoseSequence":
        """先頭を 0 とした時刻 [start, end) で切り出す."""
        s = 0 if start is None else max(0, int(round(start * self.fps)))
        e = len(self) if end is None else min(len(self), int(round(end * self.fps)))
        return PoseSequence(
            self.keypoints[s:e].copy(), self.scores[s:e].copy(), self.fps,
            self.width, self.height, self.t0 + s / self.fps, dict(self.meta),
        )

    # --- シリアライズ（端末側で骨格推定した結果を受け取る用途も想定） ---
    def to_dict(self) -> dict:
        kp = np.where(np.isnan(self.keypoints), None, np.round(self.keypoints, 2))
        return {
            "format": "coco17",
            "fps": self.fps, "width": self.width, "height": self.height, "t0": self.t0,
            "keypoints": kp.tolist(),
            "scores": np.round(np.nan_to_num(self.scores), 3).tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoseSequence":
        kp = np.array(d["keypoints"], dtype=object)
        kp = np.where(kp == None, np.nan, kp).astype(np.float64)  # noqa: E711
        return cls(kp, np.array(d["scores"], dtype=np.float64), float(d["fps"]),
                   int(d["width"]), int(d["height"]), float(d.get("t0", 0.0)), d.get("meta", {}))

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path: str) -> "PoseSequence":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
