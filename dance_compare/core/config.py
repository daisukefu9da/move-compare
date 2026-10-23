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
    min_motion_for_timing: float = 0.3     # 手本の動きがこれより遅い区間はタイミングを評価しない [胴長/秒]
    edge_margin_s: float = 0.5             # 先頭・末尾のこの範囲にかかる区間はタイミングを評価しない
    rhythm_threshold_ms: float = 60.0      # 拍に対する平均的なズレ
    amplitude_tolerance: float = 0.25      # 振れ幅の差をどこまで許すか（0.25 なら手本の 75%〜133% は OK）
    sharpness_tolerance: float = 0.3       # キレ（加速・減速の鋭さ）が手本より何割弱くても許すか
    level_threshold: float = 0.15          # 腰の高さの差（胴の長さ比. 各自の普段の高さからの沈み込みで比べる）
    facing_threshold_deg: float = 35.0     # 体の向きの差
    lean_threshold_deg: float = 15.0       # 上体の傾きの差
    stance_threshold: float = 0.25         # 足幅の差（胴の長さ比）
    travel_threshold: float = 0.4          # 区間内の移動量の差（胴の長さ比）
    head_tilt_threshold_deg: float = 15.0  # 頭の傾き（左右）の差（両耳の線と両肩の線の角度）
    head_turn_threshold_deg: float = 30.0  # 頭の向き（左右）の差
    head_shift_threshold: float = 0.12     # 頭の横スライドの差（胴の長さ比. 各自の普段の位置からの差で比べる）
    head_nod_threshold: float = 0.08       # 頭の上下（うなずき・見上げ）の差（胴の長さ比. 同上）
    head_min_conf: float = 0.5             # 顔の点（鼻・耳）の信頼度がこれ未満の区間は頭を評価しない
    max_findings: int = 12                 # 表示する指摘の最大数

    # --- リズム ---
    accent_mode: str = "peak"          # "peak": 速度のピーク / "hit": 急停止（減速のピーク）
    accent_min_interval_s: float = 0.18
    accent_prominence: float = 0.6     # ピークの目立ち具合（速度の標準偏差に対する倍率）
    beat_subdivision: int = 2          # 2 なら裏拍(&)も拍として扱う
    on_beat_tolerance_ms: float = 70.0
    landing_speed_on: float = 1.2      # 着地検出: 足首の下向き速度がこれを超えた後 [胴長/秒]
    landing_speed_off: float = 0.35    # これを下回った瞬間を着地とする
    landing_match_window_s: float = 0.4  # 手本の着地に対応するレビュー対象の着地を探す範囲 [秒]
    landing_threshold_ms: float = 80.0   # 着地のタイミング差の指摘しきい値

    # --- 比較する区間の自動推定（1動画モード: 二人の動きがそろっている区間） ---
    sync_window_s: float = 2.0         # 同期度を見る窓の長さ [秒]
    sync_max_lag_s: float = 0.4        # 二人のタイミングのずれをどこまで許すか [秒]
    sync_threshold: float = 0.45       # 同期度（0〜1）がこれ以上なら「そろっている」
    sync_min_duration_s: float = 2.0   # そろった状態がこれだけ続いたら振付が始まったとみなす [秒]
    sync_min_motion: float = 0.3       # これより動きが小さい区間は同期度を下げる [胴長/秒]
    sync_relative_motion: float = 0.5  # 動画内の大きな動き（上位 20%）のこの割合より小さい動きも同期度を下げる

    # --- 音声による 2 動画の同期 ---
    audio_sync_min_confidence: float = 3.5  # 相関ピークの鋭さがこれ以上なら同期できたとみなす
    audio_sync_min_uniqueness: float = 0.15 # 2 番目の候補との差（小さいと拍単位でずれる恐れ）

    # --- 部位の重み（スコアと、指摘の優先度に掛ける. 0 にするとその部位は評価しない） ---
    weight_legs: float = 1.0           # 足（膝・股関節の角度、足先の振れ幅・キレ、足幅、着地）
    weight_hips: float = 1.0           # 腰・体幹（沈み込み、上体の傾き、体の向き、移動）
    weight_head: float = 1.0           # 頭（首の動きを含む: 傾き、向き、横スライド、上下）
    weight_arms: float = 1.0           # 腕（肘・肩の角度、手先の振れ幅・キレ）
    weight_timing: float = 1.0         # 手本とのタイミング（全身）
    top_moments: int = 3               # 「直すところ」として見せる瞬間の数

    # --- スコア ---
    # 各項目のズレを「指摘のしきい値の何倍か」で表し、次の対応で点数にする（しきい値と連動）:
    #   しきい値の半分以下 = 100 点, しきい値ちょうど = 80 点, 2 倍 = 50 点, 3 倍 = 25 点, 6 倍以上 = 0 点
    score_full_ratio: float = 0.5      # しきい値のこの割合以下のズレは減点しない（測定誤差の分）

    @classmethod
    def load(cls, path: str) -> "AnalysisConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        data = dict(data)
        # 古い項目名を読み替える
        if "amplitude_ratio_low" in data:
            data.setdefault("amplitude_tolerance", 1 - data.pop("amplitude_ratio_low"))
        if "sharpness_ratio_low" in data:
            data.setdefault("sharpness_tolerance", 1 - data.pop("sharpness_ratio_low"))
        for old in ("amplitude_ratio_high", "score_scale_angle_deg", "score_scale_timing_ms", "score_scale_rhythm_ms",
                    "score_scale_amplitude", "score_scale_sharpness", "score_scale_level"):
            data.pop(old, None)
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

    # 振れ幅・キレの比のしきい値
    @property
    def amplitude_low(self) -> float:
        return 1.0 - self.amplitude_tolerance

    @property
    def amplitude_high(self) -> float:
        return 1.0 / (1.0 - self.amplitude_tolerance)

    @property
    def sharpness_low(self) -> float:
        return 1.0 - self.sharpness_tolerance

    def weight(self, group: str) -> float:
        """部位（legs / hips / head / arms / timing）の重み."""
        return float(getattr(self, f"weight_{group}", 1.0))
