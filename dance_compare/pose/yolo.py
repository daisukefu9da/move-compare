"""YOLO-pose (Ultralytics) バックエンド. 複数人の検出と追跡 (ByteTrack) ができる.

注意: Ultralytics は AGPL-3.0。自分用ツールでは問題ないが、Web サービスとして公開する場合は
ライセンス（ソース公開義務 or 商用ライセンス）を確認すること。
"""

from __future__ import annotations

import numpy as np

import os

from ..media.video import probe
from .base import ProgressFn, TrackSet

DEFAULT_TRACKER = os.path.join(os.path.dirname(__file__), "bytetrack_dance.yaml")


class YoloPoseBackend:
    name = "yolo"

    def __init__(self, model: str = "yolo11n-pose.pt", tracker: str = DEFAULT_TRACKER,
                 conf: float = 0.3, device: str | None = None, imgsz: int = 640):
        self.model_name = model
        self.tracker = tracker
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    @property
    def cache_key(self) -> str:
        return f"yolo-{self.model_name}-{os.path.basename(self.tracker)}-{self.conf}-{self.imgsz}"

    def track(self, video_path: str, progress: ProgressFn | None = None) -> TrackSet:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("YOLO バックエンドには ultralytics が必要です: pip install ultralytics") from e

        device = self.device or auto_device()
        try:
            return self._track(YOLO, video_path, device, progress)
        except (RuntimeError, NotImplementedError, AssertionError):
            if device == "cpu":
                raise
            # GPU で失敗した場合（mps の未対応演算など）は CPU でやり直す
            return self._track(YOLO, video_path, "cpu", progress)

    def _track(self, YOLO, video_path: str, device: str, progress: ProgressFn | None) -> TrackSet:
        info = probe(video_path)
        model = YOLO(self.model_name)
        ts = TrackSet(info.fps, info.width, info.height, max(info.frames, 1))
        results = model.track(source=video_path, stream=True, persist=True, tracker=self.tracker,
                              conf=self.conf, imgsz=self.imgsz, device=device, verbose=False)
        n = 0
        for i, r in enumerate(results):
            n = i + 1
            if i >= ts.n_frames:  # フレーム数の申告が実際より少ない動画への対策
                ts = _grow(ts, ts.n_frames * 2)
            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().numpy()
                kps = r.keypoints.xy.cpu().numpy()
                confs = r.keypoints.conf
                confs = confs.cpu().numpy() if confs is not None else np.ones(kps.shape[:2])
                boxes = r.boxes.xyxy.cpu().numpy()
                for tid, kp, sc, bx in zip(ids, kps, confs, boxes):
                    kp = kp.astype(np.float64)
                    kp[sc <= 0] = np.nan
                    ts.add(int(tid), i, kp, sc, bx)
            if progress:
                progress(n, info.frames)
        return ts.trim_frames(n)


def auto_device() -> str:
    """使える GPU があれば使う（Apple シリコンの Mac は mps, NVIDIA は cuda）.

    環境変数 DANCE_COMPARE_DEVICE（cpu / mps / cuda）で指定もできる。
    """
    env = os.environ.get("DANCE_COMPARE_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _grow(ts: TrackSet, n: int) -> TrackSet:
    out = TrackSet(ts.fps, ts.width, ts.height, n)
    pad = n - ts.n_frames
    for tid in ts.ids:
        out.keypoints[tid] = np.concatenate([ts.keypoints[tid], np.full((pad,) + ts.keypoints[tid].shape[1:], np.nan)])
        out.scores[tid] = np.concatenate([ts.scores[tid], np.zeros((pad,) + ts.scores[tid].shape[1:])])
        out.boxes[tid] = np.concatenate([ts.boxes[tid], np.full((pad, 4), np.nan)])
    return out
