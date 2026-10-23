"""解析結果のデータ構造. 言語化（feedback）や UI はこれだけを見て動く."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ASPECTS = {
    "shape": "身体の形",
    "steps": "ステップ",
    "rhythm": "リズム",  # 廃止（スコア・指摘には使わない. 古い結果ファイルの読み込み用に残す）
    "timing": "手本とのタイミング",
    "amplitude": "動きの大きさ",
    "sharpness": "キレ",
    "level": "レベル・向き",
}


@dataclass
class Finding:
    """数値で表した 1 つの指摘. 文章化は feedback 側が行う.

    aspect:    ASPECTS のキー
    kind:      指摘の種類（例: "angle", "early_late", "amplitude", "level", "facing", ...）
    part:      関節・部位名（skeleton.JA_NAMES のキー）。全身なら None
    direction: ズレの向き（例: "more_bent", "late", "smaller", "lower" ...）
    value:     ズレの大きさ（unit の単位, 符号付き）
    t_start, t_end: 手本動画上の時刻 [秒]。全体傾向なら None
    label:     カウント表記（例: "2エイト目の3〜4カウント"）
    severity:  優先度（しきい値に対する倍率 × 継続時間の補正）
    """

    aspect: str
    kind: str
    direction: str
    value: float
    unit: str
    severity: float
    part: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    label: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_global(self) -> bool:
        return self.t_start is None


@dataclass
class Report:
    meta: dict
    scores: dict                  # 観点ごとのスコア (0-100) と overall
    findings: list[Finding]       # 区間ごとの指摘（優先度順）
    tendencies: list[Finding]     # 全体的な傾向
    windows: list[dict]           # 区間ごとの数値
    rhythm: dict                  # 拍に対するズレの統計など
    timeline: dict                # グラフ用の時系列（手本の時刻軸）
    warp: dict                    # 手本の時刻 -> レビュー対象の時刻 の対応表
    beats: list[float] = field(default_factory=list)  # 手本動画上の拍の時刻
    first_one: float | None = None                    # 1エイト目の 1 の時刻

    def user_time(self, ref_time: float | np.ndarray) -> np.ndarray:
        """手本動画の時刻 -> 対応するレビュー対象動画の時刻."""
        return np.interp(ref_time, self.warp["ref"], self.warp["user"])

    def count_grid(self):
        from .counts import CountGrid
        if len(self.beats) < 2:
            return None
        return CountGrid(np.array(self.beats), self.first_one)

    def to_dict(self) -> dict:
        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [conv(v) for v in x]
            if isinstance(x, np.ndarray):
                return conv(x.tolist())
            if isinstance(x, (np.floating, float)):
                return None if not np.isfinite(x) else round(float(x), 4)
            if isinstance(x, np.integer):
                return int(x)
            return x

        d = asdict(self)
        return conv(d)

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        d = dict(d)
        d["findings"] = [Finding(**f) for f in d["findings"]]
        d["tendencies"] = [Finding(**f) for f in d["tendencies"]]
        return cls(**d)
