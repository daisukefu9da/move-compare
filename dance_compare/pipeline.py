"""動画から解析結果までをつなぐ高レベル API（CLI と Streamlit の両方から使う）."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .core import compare
from .core.config import AnalysisConfig
from .core.counts import CountGrid, fmt_time
from .core.report import Report
from .core.segment import SyncCurve, sync_curve
from .core.skeleton import PoseSequence
from .feedback.table import markdown_table, write_csv
from .media import audio as audio_mod
from .media.video import probe
from .pose.base import TrackSet, get_backend
from .pose.cache import track_cached
from .pose.tracking import select_pair, select_person


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


def _first_one(beats, rhythm: Rhythm, start: float, margin_beats: float = 0.1) -> float | None:
    """「1エイト目の1」の時刻. 指定が無ければ、start の margin_beats 拍手前以降で最初の拍."""
    if beats is None:
        return None
    if rhythm.first_one is not None:
        return rhythm.first_one
    interval = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    after = beats[beats >= start - margin_beats * interval]
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


def _select_pair(tracks: TrackSet, ref_pick: Pick, user_pick: Pick) -> tuple[PoseSequence, PoseSequence, float, float]:
    """1 本の動画から二人を選び、二人とも映っている区間 (開始, 終了) [秒] とともに返す."""
    ref_seq, user_seq = select_pair(tracks, (ref_pick.track_id, ref_pick.point, ref_pick.frame),
                                    (user_pick.track_id, user_pick.point, user_pick.frame))
    both = np.isfinite(ref_seq.keypoints[:, 5, 0]) & np.isfinite(user_seq.keypoints[:, 5, 0])
    present = np.where(both)[0]
    if present.size < ref_seq.fps * 2:
        raise ValueError("二人が同時に映っている区間が短すぎます")
    return ref_seq, user_seq, present[0] / ref_seq.fps, (present[-1] + 1) / ref_seq.fps


def _tracking_notes(ref_seq: PoseSequence, user_seq: PoseSequence, want: tuple[float, float],
                    got: tuple[float, float]) -> list[str]:
    """指定した区間より短くなった理由（どちらの追跡が途切れたか）."""
    out = []
    fps = ref_seq.fps
    for name, seq in (("お手本", ref_seq), ("レビュー対象", user_seq)):
        p = np.where(np.isfinite(seq.keypoints[:, 5, 0]))[0]
        first, last = p[0] / fps, (p[-1] + 1) / fps
        if want[0] is not None and first > want[0] + 0.2:
            out.append(f"{name}は {fmt_time(first)} より前を追跡できていません")
        if want[1] is not None and last < want[1] - 0.2:
            out.append(f"{name}は {fmt_time(last)} 以降を追跡できていません")
    if out:
        return [f"指定した区間（{fmt_time(want[0] or 0)}〜{fmt_time(want[1]) if want[1] else '最後'}）のうち、"
                f"{fmt_time(got[0])}〜{fmt_time(got[1])} しか比較できませんでした: " + "、".join(out)
                + "。人物の重なりなどで追跡が途切れた可能性があります。比較動画で骨格が乗っているか確認してください"]
    return []


def suggest_range(tracks: TrackSet, ref_pick: Pick, user_pick: Pick, cfg: AnalysisConfig | None = None,
                  mirror: bool = False) -> SyncCurve:
    """二人の動きがそろっている区間（振付を踊っている区間）を推定する. 時刻は動画の先頭が 0."""
    cfg = cfg or AnalysisConfig()
    ref_seq, user_seq, s0, e0 = _select_pair(tracks, ref_pick, user_pick)
    c = sync_curve(ref_seq.trim(s0, e0), user_seq.trim(s0, e0), cfg, mirror_ref=mirror)
    return SyncCurve(c.times + s0, c.score, None if c.start is None else c.start + s0,
                     None if c.end is None else c.end + s0)


def analyze_single_video(
    video: str,
    ref_pick: Pick,
    user_pick: Pick,
    cfg: AnalysisConfig | None = None,
    *,
    backend: str = "yolo",
    mirror: bool = False,
    time_range: tuple[float | None, float | None] = (None, None),
    auto_range: bool = True,
    rhythm: Rhythm | None = None,
    use_cache: bool = True,
    progress=None,
    tracks: TrackSet | None = None,
) -> AnalysisResult:
    """1動画モード: 1 本の動画の中の「お手本」と「レビュー対象」を比較する.

    time_range: 比較する区間 [秒]（指定した側が優先）。
    auto_range: 指定が無い側は、二人の動きがそろっている区間から自動で推定する（踊る前・後を除く）。
    """
    cfg = cfg or AnalysisConfig()
    rhythm = rhythm or Rhythm()
    notes: list[str] = []
    if backend == "mediapipe":
        raise ValueError("1動画モードは複数人を追跡できる yolo バックエンドを使ってください")
    tracks = tracks or extract_tracks(video, backend, use_cache, (lambda i, n: progress("骨格推定", i, n)) if progress else None)
    ref_seq, user_seq, s, e = _select_pair(tracks, ref_pick, user_pick)

    start, end = time_range
    sync = None
    if auto_range and (start is None or end is None):
        sync = suggest_range(tracks, ref_pick, user_pick, cfg, mirror)
        if sync.start is None:
            notes.append("二人の動きがそろっている区間が見つからなかったため、開始・終了の指定が無い側は"
                         "動画の最初・最後までを比較しました")
        else:
            start = sync.start if start is None else start
            end = sync.end if end is None else end
    want = (start, end)
    s = max(start if start is not None else s, s)
    e = min(end if end is not None else e, e)
    notes += _tracking_notes(ref_seq, user_seq, want, (s, e))
    if e - s < 2.0:
        raise ValueError(f"比較する区間が短すぎます（{fmt_time(s)}〜{fmt_time(e)}）")
    if sync is not None and sync.start is not None:
        notes.append(f"比較した区間: {fmt_time(s)}〜{fmt_time(e)}（二人の動きがそろっている所から自動で推定。"
                     "ずれていれば開始・終了を指定してください）")
    else:
        notes.append(f"比較した区間: {fmt_time(s)}〜{fmt_time(e)}")

    beats, _ = _beats_for(video, s, rhythm, ref_seq.duration, notes)
    report = compare(
        ref_seq.trim(s, e), user_seq.trim(s, e), cfg.replace(dtw_band_s=cfg.dtw_band_s_single),
        ref_beats=beats, user_beats=beats, first_one=_first_one(beats, rhythm, s, margin_beats=0.4),  # 区間の推定誤差を見込む
        mirror_ref=mirror, synced=True,
    )
    report.meta.update(mode="single_video", backend=backend, notes=notes,
                       ref_track_ids=ref_seq.meta["track_ids"], user_track_ids=user_seq.meta["track_ids"],
                       sync={"times": sync.times[::3], "score": sync.score[::3]} if sync is not None else None)
    return AnalysisResult(report, video, video, ref_seq, user_seq, notes)


def save_outputs(result: AnalysisResult, out_dir: str, text: str, render: bool = True, progress=None) -> dict:
    """report.json / report.md / comparison.mp4 / 骨格の JSON を書き出す."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "json": os.path.join(out_dir, "report.json"),
        "markdown": os.path.join(out_dir, "report.md"),
        "ref_pose": os.path.join(out_dir, "ref_pose.json"),
        "user_pose": os.path.join(out_dir, "user_pose.json"),
        "findings_csv": os.path.join(out_dir, "findings.csv"),
    }
    # 比較動画を先に作る（失敗しても解析結果は保存し、理由を処理メモに残す）
    if render:
        from .render.overlay import render_comparison

        video = os.path.join(out_dir, "comparison.mp4")
        try:
            render_comparison(result.report, result.ref_video, result.user_video, result.ref_seq, result.user_seq,
                              video, angle_threshold=result.report.meta["config"]["angle_threshold_deg"],
                              progress=progress, notes=result.notes)
            paths["video"] = video
        except Exception as e:  # 動画が作れなくても解析結果は返す
            result.notes.append(f"比較動画を作れませんでした: {e}")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(result.report.to_dict(), f, ensure_ascii=False, indent=1)
    with open(paths["markdown"], "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n### 検出一覧（カウント・部位・観点・ズレ量）\n\n" + markdown_table(result.report))
        if result.notes:
            f.write("\n### 処理メモ\n\n" + "\n".join(f"- {n}" for n in result.notes) + "\n")
    write_csv(result.report, paths["findings_csv"])
    result.ref_seq.save_json(paths["ref_pose"])
    result.user_seq.save_json(paths["user_pose"])
    return paths
