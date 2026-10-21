"""拍の時刻から「何エイト目の何カウント目か」を求める."""

from __future__ import annotations

import numpy as np


class CountGrid:
    """拍の列（手本の時間軸, 秒）. first_one で「1エイト目の1」を指定する.

    first_one より前は「イントロ」、最後の拍より後は平均間隔で外挿する。
    """

    def __init__(self, beats: np.ndarray, first_one: float | None = None):
        beats = np.asarray(beats, dtype=np.float64)
        if beats.size < 2:
            raise ValueError("拍が 2 つ以上必要です")
        self.beats = np.sort(beats)
        self.interval = float(np.median(np.diff(self.beats)))
        if first_one is None:
            self.origin = 0
        else:
            self.origin = int(np.argmin(np.abs(self.beats - first_one)))

    @classmethod
    def from_bpm(cls, bpm: float, first_beat: float, duration: float) -> "CountGrid":
        step = 60.0 / bpm
        start = first_beat - step * np.floor(first_beat / step)
        beats = np.arange(start, duration + step, step)
        return cls(beats, first_one=first_beat)

    @property
    def bpm(self) -> float:
        return 60.0 / self.interval

    def beat_position(self, t: float | np.ndarray) -> np.ndarray:
        """時刻 -> 拍番号（実数, origin の拍が 0）."""
        t = np.asarray(t, dtype=np.float64)
        b = self.beats
        idx = np.arange(b.size, dtype=np.float64)
        pos = np.interp(t, b, idx)
        pos = np.where(t < b[0], (t - b[0]) / self.interval, pos)
        pos = np.where(t > b[-1], b.size - 1 + (t - b[-1]) / self.interval, pos)
        return pos - self.origin

    def time_of(self, pos: float | np.ndarray) -> np.ndarray:
        """拍番号 -> 時刻."""
        pos = np.asarray(pos, dtype=np.float64) + self.origin
        b = self.beats
        idx = np.arange(b.size, dtype=np.float64)
        t = np.interp(pos, idx, b)
        t = np.where(pos < 0, b[0] + pos * self.interval, t)
        t = np.where(pos > b.size - 1, b[-1] + (pos - (b.size - 1)) * self.interval, t)
        return t

    def locate(self, t: float) -> tuple[int, int, bool]:
        """時刻 -> (エイト番号 1始まり, カウント 1-8, 裏拍か). イントロはエイト番号 0 以下."""
        p = float(self.beat_position(t))
        k = int(np.floor(p + 0.25))  # 拍の少し手前も同じカウントとみなす
        and_ = (p - np.floor(p)) >= 0.25 and (p - np.floor(p)) < 0.75
        if and_:
            k = int(np.floor(p))
        return k // 8 + 1, k % 8 + 1, and_

    def label(self, t0: float, t1: float | None = None) -> str:
        """「2エイト目の3〜4カウント」形式のラベル. t1 は区間の終わり（含まない）."""
        if t1 is None:
            e, c, a = self.locate(t0)
            if e <= 0:
                return "イントロ"
            return f"{e}エイト目の{c}{'&' if a else ''}カウント"
        p0, p1 = float(self.beat_position(t0)), float(self.beat_position(t1))
        k0 = int(np.floor(p0 + 0.25))
        k1 = max(k0, int(np.ceil(p1 - 0.25)) - 1)
        if k1 < 0:
            return "イントロ"
        k0 = max(k0, 0)
        e0, c0 = k0 // 8 + 1, k0 % 8 + 1
        e1, c1 = k1 // 8 + 1, k1 % 8 + 1
        if k0 == k1:
            return f"{e0}エイト目の{c0}カウント"
        if e0 == e1:
            return f"{e0}エイト目の{c0}〜{c1}カウント"
        return f"{e0}エイト目の{c0}〜{e1}エイト目の{c1}カウント"


def fmt_time(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"
