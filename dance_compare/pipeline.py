"""動画から解析結果までをつなぐ高レベル API（CLI と Streamlit の両方から使う）."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .core import compare
from .core.config import AnalysisConfig
from .core.counts import CountGrid
from .core.report import Report
from .core.skeleton import PoseSequence
from .media import audio as audio_mod
from .media.video import probe
from .pose.base import TrackSet, get_backend
from .pose.cache import track_cached
from .pose.tracking import select_person


@dataclass
class Pick:
    """人物の指定方法. どれも無ければ「最も大きく映っている人」を自動で選ぶ."""

    track_id: int | None = None
    point: tuple[float, float] | None = None  # クリック位置（元動画のピクセル座標）
    frame: int = 0                            # point を指定したフレーム


@dataclass
class Rhythm:
    """拍の情報の指定. 何も指定しなければ音声から自動検出する."""

    bpm: float | None = None          # 手動指定の BPM
    first_one: float | None = None    # 手本動画上で「1エイト目の 1」の時刻 [秒]
    detect: bool = True               # 音声から拍を検出する


@dataclass
class AnalysisResult:
    report: Report
    ref_video: str
    user_video: str
    ref_seq: PoseSequence      # 元動画のフレームと 1 対 1 の骨格（切り出し前）
    user_seq: PoseSequence
    notes: list[str] = field(default_factory=list)  # 処理中の注意事項（同期できなかった等）


def extract_tracks(video: str, backend: str = "yolo", use_cache: bool = True, progress=None,
                   **backend_kwargs) -> TrackSet:
    return track_cached(get_backend(backend, **backend_kwargs), video, progress, use_cache)


def _beats_for(video: str, start: float, rhythm: Rhythm, duration: float, notes: list[str]):
    """(拍の時刻, BPM) を返す. 取れなければ (None, None)."""
    if rhythm.bpm:
        first = rhythm.first_one if rhythm.first_one is not None else start
        g = CountGrid.from_bpm(rhythm.bpm, first, duration)
        return g.beats, rhythm.bpm
    if not rhythm.detect:
        return None, None
    y = audio_mod.extract_audio(video)
    if y is None:
        notes.append(f"音声が取れなかったため拍を検出できませんでした: {os.path.basename(video)}")
        return None, None
    bpm, beats = audio_mod.detect_beats(y)
    if len(beats) < 4:
        notes.append(f"拍を検出できませんでした: {os.path.basename(video)}")
        return None, None
    return beats, bpm


def _first_one(beats, rhythm: Rhythm, start: float) -> float | None:
    if beats is None:
        return None
    if rhythm.first_one is not None:
        return rhythm.first_one
    after = beats[beats >= start - 0.05]
    return float(after[0]) if after.size else float(beats[0])


def analyze_two_videos(
    ref_video: str,
    user_video: str,
    cfg: AnalysisConfig | None = None,
    *,
    backend: str = "yolo",
    mirror: bool = False,
    ref_range: tuple[float | None, float | None] = (None, None),
    user_range: tuple[float | None, float | None] = (None, None),
    audio_sync: bool = True,
    rhythm: Rhythm | None = None,
    ref_pick: Pick | None = None,
    user_pick: Pick | None = None,
    use_cache: bool = True,
    progress=None,
    ref_tracks: TrackSet | None = None,
    user_tracks: TrackSet | None = None,
) -> AnalysisResult:
    """2動画モード: お手本の動画とレビュー対象の動画を比較する."""
    cfg = cfg or AnalysisConfig()
    rhythm = rhythm or Rhythm()
    notes: list[str] = []

    def stage(name):
        return (lambda i, n: progress(name, i, n)) if progress else None

    ref_tracks = ref_tracks or extract_tracks(ref_video, backend, use_cache, stage("手本の骨格推定"))
    user_tracks = user_tracks or extract_tracks(user_video, backend, use_cache, stage("レビュー対象の骨格推定"))
    rp, up = ref_pick or Pick(), user_pick or Pick()
    ref_seq = select_person(ref_tracks, rp.track_id, rp.point, rp.frame)
    user_seq = select_person(user_tracks, up.track_id, up.point, up.frame)

    r0, r1 = ref_range[0] or 0.0, ref_range[1] or ref_seq.duration
    u0, u1 = user_range[0] or 0.0, user_range[1] or user_seq.duration
    synced = False
    offset = None
    if audio_sync:
        ya, yb = audio_mod.extract_audio(ref_video), audio_mod.extract_audio(user_video)
        if ya is not None and yb is not None:
            offset, conf, uniq = audio_mod.estimate_offset(ya, yb)
            if conf >= cfg.audio_sync_min_confidence and uniq >= cfg.audio_sync_min_uniqueness:
                synced = True
                s = max(r0, u0 - offset)
                e = min(r1, u1 - offset)
                if e - s < 2.0:
                    notes.append("音声で同期すると重なる区間が短すぎるため、同期を使いませんでした")
                    synced = False
                else:
                    r0, r1, u0, u1 = s, e, s + offset, e + offset
                    notes.append(f"音声で同期しました（レビュー対象が {offset:+.2f} 秒ずれていました, 確信度 {conf:.1f}）")
            else:
                notes.append(f"音声での同期に自信がないため、動きだけで時間を合わせます"
                             f"（確信度 {conf:.1f}, 一意性 {uniq:.2f}）")
                offset = None
        else:
            notes.append("音声が無いため、動きだけで時間を合わせます")

    ref_beats, _ = _beats_for(ref_video, r0, rhythm, probe(ref_video).duration, notes)
    if synced and ref_beats is not None:
        user_beats = ref_beats + offset
    elif rhythm.bpm:
        user_beats = None  # 手動 BPM の場合、レビュー対象側の拍の位置は分からない
    else:
        user_beats, _ = _beats_for(user_video, u0, Rhythm(detect=rhythm.detect), probe(user_video).duration, notes)

    report = compare(
        ref_seq.trim(r0, r1), user_seq.trim(u0, u1), cfg,
        ref_beats=ref_beats, user_beats=user_beats, first_one=_first_one(ref_beats, rhythm, r0),
        mirror_ref=mirror, synced=synced,
    )
    report.meta.update(mode="two_videos", backend=backend, audio_offset=offset, notes=notes)
    return AnalysisResult(report, ref_video, user_video, ref_seq, user_seq, notes)


def analyze_single_video(
    video: str,
    ref_pick: Pick,
    user_pick: Pick,
    cfg: AnalysisConfig | None = None,
    *,
    backend: str = "yolo",
    mirror: bool = False,
    time_range: tuple[float | None, float | None] = (None, None),
    rhythm: Rhythm | None = None,
    use_cache: bool = True,
    progress=None,
    tracks: TrackSet | None = None,
) -> AnalysisResult:
    """1動画モード: 1 本の動画の中の「お手本」と「レビュー対象」を比較する."""
    cfg = cfg or AnalysisConfig()
    rhythm = rhythm or Rhythm()
    notes: list[str] = []
    if backend == "mediapipe":
        raise ValueError("1動画モードは複数人を追跡できる yolo バックエンドを使ってください")
    tracks = tracks or extract_tracks(video, backend, use_cache, (lambda i, n: progress("骨格推定", i, n)) if progress else None)
    ref_seq = select_person(tracks, ref_pick.track_id, ref_pick.point, ref_pick.frame)
    user_seq = select_person(tracks, user_pick.track_id, user_pick.point, user_pick.frame,
                             exclude=set(ref_seq.meta["track_ids"]))
    if set(ref_seq.meta["track_ids"]) & set(user_seq.meta["track_ids"]):
        raise ValueError("お手本とレビュー対象に同じ人物が選ばれています")

    # 二人とも映っている区間に絞る
    both = np.isfinite(ref_seq.keypoints[:, 5, 0]) & np.isfinite(user_seq.keypoints[:, 5, 0])
    present = np.where(both)[0]
    if present.size < ref_seq.fps * 2:
        raise ValueError("二人が同時に映っている区間が短すぎます")
    s = max(time_range[0] or 0.0, present[0] / ref_seq.fps)
    e = min(time_range[1] or ref_seq.duration, (present[-1] + 1) / ref_seq.fps)

    beats, _ = _beats_for(video, s, rhythm, ref_seq.duration, notes)
    report = compare(
        ref_seq.trim(s, e), user_seq.trim(s, e), cfg.replace(dtw_band_s=cfg.dtw_band_s_single),
        ref_beats=beats, user_beats=beats, first_one=_first_one(beats, rhythm, s),
        mirror_ref=mirror, synced=True,
    )
    report.meta.update(mode="single_video", backend=backend, notes=notes,
                       ref_track_ids=ref_seq.meta["track_ids"], user_track_ids=user_seq.meta["track_ids"])
    return AnalysisResult(report, video, video, ref_seq, user_seq, notes)


def save_outputs(result: AnalysisResult, out_dir: str, text: str, render: bool = True, progress=None) -> dict:
    """report.json / report.md / comparison.mp4 / 骨格の JSON を書き出す."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "json": os.path.join(out_dir, "report.json"),
        "markdown": os.path.join(out_dir, "report.md"),
        "ref_pose": os.path.join(out_dir, "ref_pose.json"),
        "user_pose": os.path.join(out_dir, "user_pose.json"),
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(result.report.to_dict(), f, ensure_ascii=False, indent=1)
    with open(paths["markdown"], "w", encoding="utf-8") as f:
        f.write(text)
        if result.notes:
            f.write("\n### 処理メモ\n\n" + "\n".join(f"- {n}" for n in result.notes) + "\n")
    result.ref_seq.save_json(paths["ref_pose"])
    result.user_seq.save_json(paths["user_pose"])
    if render:
        from .render.overlay import render_comparison

        paths["video"] = os.path.join(out_dir, "comparison.mp4")
        render_comparison(result.report, result.ref_video, result.user_video, result.ref_seq, result.user_seq,
                          paths["video"], angle_threshold=result.report.meta["config"]["angle_threshold_deg"],
                          progress=progress)
    return paths
