"""ブラウザ（Pyodide）から呼ぶ入口.

骨格推定と音声の取り出しはブラウザ側で行い、その結果（JSON）を受け取って今の解析をそのまま行う。
動画ファイル・ffmpeg・OpenCV は使わない。結果は画面に出しやすい JSON で返す。

骨格の JSON（tracks_json）の形:
    {"fps": 30, "width": 1920, "height": 1080, "n_frames": 900,
     "people": {"1": {"frames": [0, 1, ...],          # 映っているフレーム番号
                      "kp": [[x, y] * 17, ...],       # フレームごとの 17 点（元の動画のピクセル座標）
                      "sc": [[s] * 17, ...],          # 信頼度
                      "box": [[x1, y1, x2, y2], ...]}}}
"""

from __future__ import annotations

import json

import numpy as np

from .core.config import AnalysisConfig
from .feedback.rule_based import moment_lines, moment_title, tendency_line
from .pipeline import Pick, analyze_single_video, analyze_two_videos
from .pose.base import TrackSet


def tracks_from_json(text: str) -> TrackSet:
    d = json.loads(text)
    ts = TrackSet(float(d["fps"]), int(d["width"]), int(d["height"]), int(d["n_frames"]))
    for tid, p in d["people"].items():
        frames = np.asarray(p["frames"], dtype=int)
        kp = np.asarray(p["kp"], dtype=float).reshape(len(frames), 17, 2)
        sc = np.asarray(p["sc"], dtype=float).reshape(len(frames), 17)
        box = np.asarray(p["box"], dtype=float).reshape(len(frames), 4)
        for i, f in enumerate(frames):
            ts.add(int(tid), int(f), kp[i], sc[i], box[i])
    return ts


def _summary(result) -> str:
    rep = result.report
    return json.dumps({
        "scores": rep.scores,
        "moments": [{"title": moment_title(m), "lines": moment_lines(m), "t": m["t"]} for m in rep.moments],
        "tendencies": [tendency_line(f) for f in rep.tendencies],
        "notes": result.notes,
        "n_findings": len(rep.findings),
        "report": rep.to_dict(),
    }, ensure_ascii=False, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))


def run_single(tracks_json: str, ref_id: int | None, user_id: int | None, config_json: str = "{}") -> str:
    """1動画モード. ref_id / user_id は人物の ID（None なら大きく映っている順に自動）."""
    cfg = AnalysisConfig.from_dict(json.loads(config_json))
    ts = tracks_from_json(tracks_json)
    result = analyze_single_video("browser", Pick(ref_id), Pick(user_id), cfg, tracks=ts)
    return _summary(result)


def run_two(ref_json: str, user_json: str, ref_audio=None, user_audio=None, config_json: str = "{}") -> str:
    """2動画モード. 音声は audio.SR（22050Hz）のモノラル（無ければ None で、動きだけで時間を合わせる）."""
    cfg = AnalysisConfig.from_dict(json.loads(config_json))
    ya = None if ref_audio is None else np.asarray(ref_audio, dtype=np.float32)
    yb = None if user_audio is None else np.asarray(user_audio, dtype=np.float32)
    result = analyze_two_videos("ref", "user", cfg, ref_tracks=tracks_from_json(ref_json),
                                user_tracks=tracks_from_json(user_json), ref_audio=ya, user_audio=yb,
                                audio_sync=ya is not None and yb is not None)
    return _summary(result)
