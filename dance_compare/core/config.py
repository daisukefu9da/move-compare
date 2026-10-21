"""解析パラメータ.

精度調整（どの程度のズレを指摘するか等）はここの値を変えて行う。
TOML ファイルで上書きできる（`AnalysisConfig.load("tuning.toml")`）。
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields


@dataclass
class AnalysisConfig:
    # --- 前処理 ---
    conf_threshold: float = 0.3        # これ未満の関節は欠損扱いにして補間
    smooth_sigma_s: float = 0.04       # 座標の平滑化（ガウシアン, 秒）
    analysis_fps: float = 30.0         # 解析時に揃えるフレームレート

    # --- 時間合わせ (DTW) ---
    dtw_fps: float = 15.0              # DTW を計算するフレームレート（計算量削減）
    dtw_band_s: float | None = 3.0     # 対角線からのずれの上限 [秒]。None で無制限
    dtw_band_s_single: float = 1.0     # 1動画モード（同じ音楽で同時に踊っている）での上限
    dtw_step_penalty: float = 0.2      # 縦横ステップのペナルティ（コスト中央値に対する倍率）
    dtw_open_ends: bool = False        # レビュー対象の前後に余分な部分がある場合 True
    warp_smooth_s: float = 0.25        # 対応づけ結果の平滑化 [秒]

    # --- 区間（指摘の単位） ---
    window_counts: int = 2             # 何カウントごとに区切って評価するか
    window_s_fallback: float = 1.0     # 拍が取れないときの区間長 [秒]

    # --- 指摘のしきい値 ---
    angle_threshold_deg: float = 20.0      # 関節角度の差
    timing_threshold_ms: float = 120.0     # 手本との相対的な早い/遅い
    rhythm_threshold_ms: float = 60.0      # 拍に対する平均的なズレ
    amplitude_ratio_low: float = 0.75      # 振れ幅が手本の何倍未満なら「小さい」
    amplitude_ratio_high: float = 1.35     # 何倍超なら「大きい」
    sharpness_ratio_low: float = 0.7       # 加速度が手本の何倍未満なら「キレ不足」
    level_threshold: float = 0.12          # 腰の高さの差（胴の長さ比）
    facing_threshold_deg: float = 35.0     # 体の向きの差
    lean_threshold_deg: float = 15.0       # 上体の傾きの差
    stance_threshold: float = 0.25         # 足幅の差（胴の長さ比）
    travel_threshold: float = 0.4          # 区間内の移動量の差（胴の長さ比）
    max_findings: int = 12                 # 表示する指摘の最大数

    # --- リズム ---
    accent_mode: str = "peak"          # "peak": 速度のピーク / "hit": 急停止（減速のピーク）
    accent_min_interval_s: float = 0.18
    accent_prominence: float = 0.6     # ピークの目立ち具合（速度の標準偏差に対する倍率）
    beat_subdivision: int = 2          # 2 なら裏拍(&)も拍として扱う
    on_beat_tolerance_ms: float = 70.0
    landing_speed_on: float = 1.2      # 着地検出: 足首の下向き速度がこれを超えた後 [胴長/秒]
    landing_speed_off: float = 0.35    # これを下回った瞬間を着地とする

    # --- 音声による 2 動画の同期 ---
    audio_sync_min_confidence: float = 3.5  # 相関ピークの鋭さがこれ以上なら同期できたとみなす
    audio_sync_min_uniqueness: float = 0.15 # 2 番目の候補との差（小さいと拍単位でずれる恐れ）

    # --- スコア（差がこの値のとき約 37 点になる指数関数） ---
    score_scale_angle_deg: float = 30.0
    score_scale_timing_ms: float = 200.0
    score_scale_rhythm_ms: float = 120.0
    score_scale_amplitude: float = 0.35
    score_scale_sharpness: float = 0.5
    score_scale_level: float = 0.25

    @classmethod
    def load(cls, path: str) -> "AnalysisConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **kw) -> "AnalysisConfig":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if v is not None})
        return AnalysisConfig(**d)

    def to_dict(self) -> dict:
        return asdict(self)
