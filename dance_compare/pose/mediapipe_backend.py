"""MediaPipe Pose Landmarker バックエンド（1 人用, Apache-2.0）.

複数人には弱いので 2 動画モード専用。33 点の出力を COCO 17 点に変換する。
モデルファイルは初回に自動でダウンロードする。
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from ..core.skeleton import NUM_KEYPOINTS
from ..media.video import iter_frames, probe
from .base import ProgressFn, TrackSet

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")

# COCO17 の各点に対応する MediaPipe のインデックス
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, variant: str = "full", model_dir: str = ".cache/models", min_confidence: float = 0.3):
        self.variant = variant
        self.model_dir = model_dir
        self.min_confidence = min_confidence

    @property
    def cache_key(self) -> str:
        return f"mediapipe-{self.variant}-{self.min_confidence}"

    def _model_path(self) -> str:
        path = os.path.join(self.model_dir, f"pose_landmarker_{self.variant}.task")
        if not os.path.exists(path):
            os.makedirs(self.model_dir, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL.format(variant=self.variant), path)
        return path

    def track(self, video_path: str, progress: ProgressFn | None = None) -> TrackSet:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        except ImportError as e:  # pragma: no cover
            raise ImportError("MediaPipe バックエンドには mediapipe が必要です: pip install mediapipe") from e

        info = probe(video_path)
        ts = TrackSet(info.fps, info.width, info.height, max(info.frames, 1))
        opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=self._model_path()),
                                     running_mode=RunningMode.VIDEO, num_poses=1,
                                     min_pose_detection_confidence=self.min_confidence,
                                     min_pose_presence_confidence=self.min_confidence,
                                     min_tracking_confidence=self.min_confidence)
        n = 0
        with PoseLandmarker.create_from_options(opts) as lm:
            for i, frame in iter_frames(video_path):
                if i >= ts.n_frames:
                    break
                n = i + 1
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = lm.detect_for_video(img, int(i * 1000 / info.fps))
                if res.pose_landmarks:
                    pts = res.pose_landmarks[0]
                    kp = np.array([[pts[k].x * info.width, pts[k].y * info.height] for k in MP_TO_COCO])
                    sc = np.array([pts[k].visibility if pts[k].visibility is not None else 1.0 for k in MP_TO_COCO])
                    box = np.array([*np.nanmin(kp, 0), *np.nanmax(kp, 0)])
                    ts.add(1, i, kp, sc, box)
                if progress:
                    progress(n, info.frames)
        assert ts.n_frames >= n and NUM_KEYPOINTS == len(MP_TO_COCO)
        return ts.trim_frames(n)
