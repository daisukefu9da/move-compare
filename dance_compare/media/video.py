"""動画の読み込み（OpenCV）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {path}")
    info = VideoInfo(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def iter_frames(path: str, start: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """(フレーム番号, BGR 画像) を順に返す."""
    import cv2

    cap = cv2.VideoCapture(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def read_frame(path: str, index: int = 0) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"フレーム {index} を読めません: {path}")
    return frame


class SequentialReader:
    """前方向にだけ進むランダムアクセス（比較動画の生成用. 毎回シークするより速い）."""

    def __init__(self, path: str):
        import cv2

        self.cap = cv2.VideoCapture(path)
        self.pos = -1
        self.frame = None

    def get(self, index: int) -> np.ndarray | None:
        import cv2

        if index < self.pos or index - self.pos > 90:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index - 1
        while self.pos < index:
            ok, f = self.cap.read()
            if not ok:
                return self.frame
            self.frame = f
            self.pos += 1
        return self.frame

    def close(self):
        self.cap.release()
