import shutil

import numpy as np
import pytest

from dance_compare.feedback import get_narrator
from dance_compare.media import audio
from dance_compare.pose.base import TrackSet
from dance_compare.pose.tracking import pick_by_point, select_person, stitch
from dance_compare.synth import synth_beats, synth_dancer


def test_audio_offset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=audio.SR * 10)
    y = np.concatenate([np.zeros(int(audio.SR * 1.5)), x])
    off, conf, uniq = audio.estimate_offset(x, y)
    assert abs(off - 1.5) < 0.03 and conf > 5 and uniq > 0.3
    off, _, _ = audio.estimate_offset(y, x)
    assert abs(off + 1.5) < 0.03


def test_numpy_beat_tracker_on_click_track():
    sr = audio.SR
    y = np.zeros(sr * 12)
    click = np.sin(2 * np.pi * 1000 * np.arange(600) / sr) * np.exp(-np.arange(600) / 100)
    for b in np.arange(1.0, 11.5, 0.5):
        y[int(b * sr):int(b * sr) + 600] += click
    bpm, beats = audio._beats_numpy(y, sr)
    assert abs(bpm - 120) < 4
    assert np.median(np.abs(beats - np.round(beats * 2) / 2)) < 0.03


def _two_people_tracks():
    a = synth_dancer(duration=4, center=(150, 260))
    b = synth_dancer(duration=4, center=(480, 260), amplitude=0.5)
    ts = TrackSet(30.0, 640, 480, len(a))
    for f in range(len(a)):
        for tid, s in ((1, a), (2, b)):
            if tid == 2 and f >= 60:
                tid = 7  # 途中で ID が変わる
            kp = s.keypoints[f]
            ts.add(tid, f, kp, s.scores[f], np.array([*kp.min(0), *kp.max(0)]))
    return ts


def test_pick_and_stitch():
    ts = _two_people_tracks()
    assert pick_by_point(ts, 0, 480, 300) == 2
    assert stitch(ts, 2, exclude={1}) == [2, 7]
    seq = select_person(ts, point=(480, 300), exclude={1})
    assert np.isfinite(seq.keypoints[:, 5, 0]).all()
    assert seq.meta["track_ids"] == [2, 7]


def test_trackset_save_load(tmp_path):
    ts = _two_people_tracks()
    p = str(tmp_path / "t.npz")
    ts.save(p)
    ts2 = TrackSet.load(p)
    assert ts2.ids == ts.ids
    assert np.allclose(ts2.keypoints[7], ts.keypoints[7], equal_nan=True)


def test_narrators(monkeypatch):
    from dance_compare.core import compare
    beats = synth_beats(16)
    r = compare(synth_dancer(), synth_dancer(delay_s=0.1, elbow_bend=(4.5, 6.5, 50)),
                ref_beats=beats, user_beats=beats, synced=True)
    text = get_narrator("rule").narrate(r)
    assert "右肘" in text and "2エイト目" in text and "遅れ" in text
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    g = get_narrator("gemini")
    assert g.narrate(r) == text  # キーが無ければルールベースにフォールバック
    assert g.last_error


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_end_to_end_with_synthetic_videos(tmp_path):
    from dance_compare.pipeline import analyze_two_videos, save_outputs
    from dance_compare.synth import write_synth_video

    beats = synth_beats(9)
    ref = synth_dancer(duration=8)
    user = synth_dancer(duration=9, delay_s=1.0 + 0.1)
    rv, uv = str(tmp_path / "r.mp4"), str(tmp_path / "u.mp4")
    write_synth_video(ref, rv, beats)
    write_synth_video(user, uv, beats + 1.0)
    res = analyze_two_videos(rv, uv, ref_tracks=TrackSet.from_sequence(ref),
                             user_tracks=TrackSet.from_sequence(user), use_cache=False)
    assert res.report.meta["synced"]
    assert abs(res.report.meta["audio_offset"] - 1.0) < 0.05
    paths = save_outputs(res, str(tmp_path / "out"), get_narrator("rule").narrate(res.report))
    import os
    assert os.path.getsize(paths["video"]) > 10000


def _tracks_with_switches(switches):
    """二人のダンサー. switches = [(人物, 切替フレーム, 新ID, 途切れるフレーム数, 重なるフレーム数)]"""
    a = synth_dancer(duration=12, center=(150, 260))
    b = synth_dancer(duration=12, center=(480, 260), amplitude=0.5)
    n = len(a)
    ts = TrackSet(30.0, 640, 480, n)
    ids = {0: [(0, 1)], 1: [(0, 2)]}  # 人物 -> [(開始フレーム, ID)]
    gaps = {0: [], 1: []}
    for who, f, new_id, gap, overlap in switches:
        ids[who].append((f, new_id))
        gaps[who].append((f, gap, overlap))
    for who, s in ((0, a), (1, b)):
        for f in range(n):
            cur = [tid for start, tid in ids[who] if start <= f][-1]
            hidden = any(g_f - g <= f < g_f for g_f, g, _ in gaps[who])
            if hidden:
                continue
            kp = s.keypoints[f]
            box = np.array([*kp.min(0), *kp.max(0)])
            ts.add(cur, f, kp, s.scores[f], box)
            # 新しい ID が古い ID の消える少し前から出ている（重なり）
            for g_f, g, ov in gaps[who]:
                if ov and g_f - ov <= f < g_f:
                    nxt = [tid for start, tid in ids[who] if start == g_f][0]
                    ts.add(nxt, f, kp, s.scores[f], box)
    return ts


def test_stitch_handles_long_gaps_and_overlaps():
    from dance_compare.pose.tracking import select_pair

    # レビュー対象: 4 秒で 2 秒見失って ID 7 に、8 秒で ID 9 が少し重なって出てくる
    ts = _tracks_with_switches([(1, 120, 7, 60, 0), (1, 240, 9, 0, 10)])
    ref, user = select_pair(ts, (1, None, 0), (2, None, 0))
    assert user.meta["track_ids"] == [2, 7, 9]
    assert ref.meta["track_ids"] == [1]
    assert np.isfinite(user.keypoints[-1, 5, 0])


def test_stitch_does_not_take_the_other_persons_id():
    from dance_compare.pose.tracking import select_pair

    # お手本の ID 1 が 6 秒で途切れ、レビュー対象には同時刻に新しい ID 5 が振られる
    ts = _tracks_with_switches([(1, 180, 5, 0, 0)])
    keep = ts.present(1).copy()
    keep[180:] = False
    for arr in (ts.keypoints[1], ts.boxes[1]):
        arr[~keep] = np.nan
    ref, user = select_pair(ts, (1, None, 0), (2, None, 0))
    assert 5 not in ref.meta["track_ids"]
    assert user.meta["track_ids"] == [2, 5]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_single_video_warns_when_range_is_cut_by_tracking(tmp_path):
    from dance_compare.pipeline import Pick, analyze_single_video
    from dance_compare.synth import write_synth_video

    ts = _tracks_with_switches([])
    for arr in (ts.keypoints[2], ts.boxes[2]):
        arr[240:] = np.nan  # レビュー対象が 8 秒以降見つからない
    video = str(tmp_path / "v.mp4")
    write_synth_video(synth_dancer(duration=12), video, synth_beats(12))
    res = analyze_single_video(video, Pick(1), Pick(2), tracks=ts, time_range=(1.0, 11.0))
    assert any("レビュー対象は 0:08.0 以降を追跡できていません" in n for n in res.notes)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_proxy_and_rescale(tmp_path, monkeypatch):
    import dance_compare.media.proxy as proxy
    from dance_compare.synth import write_synth_video

    monkeypatch.setattr(proxy, "PROXY_DIR", str(tmp_path / "proxy"))
    s = synth_dancer(duration=2, fps=60, width=1280, height=720, size=200, center=(640, 300))
    v = str(tmp_path / "v.mp4")
    write_synth_video(s, v)
    px = proxy.make_proxy(v, max_side=640, max_fps=30)
    assert px.path != v and abs(px.scale_x - 2) < 0.01 and px.source_fps == 60
    from dance_compare.media.video import probe
    info = probe(px.path)
    assert (info.width, info.height) == (640, 360) and abs(info.fps - 30) < 0.01

    ts = TrackSet(30.0, 640, 360, 60)
    ts.add(1, 0, s.keypoints[0] / 2, s.scores[0], np.array([10, 20, 30, 40.0]))
    big = ts.rescaled(px.scale_x, px.scale_y, px.width, px.height, px.source_fps)
    assert np.allclose(big.keypoints[1][0], s.keypoints[0])
    assert np.allclose(big.boxes[1][0], [20, 40, 60, 80])
    assert big.video_frame(15) == 30 and big.track_frame(30) == 15

    # 小さい動画は作り直さない
    assert proxy.make_proxy(v, max_side=2000, max_fps=60).path == v


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_preview_keeps_timeline(tmp_path, monkeypatch):
    import dance_compare.media.preview as preview
    from dance_compare.media.video import probe
    from dance_compare.synth import write_synth_video

    monkeypatch.setattr(preview, "PREVIEW_DIR", str(tmp_path / "preview"))
    s = synth_dancer(duration=3, fps=60, width=1280, height=720, size=200, center=(640, 300))
    v = str(tmp_path / "v.mp4")
    write_synth_video(s, v)
    p = preview.make_preview(v, thumbs=6, thumb_height=40)
    info = probe(p.video)
    # 長辺 640px・30fps に縮小し、時刻は元の動画と同じ（長さが変わらない）
    assert (info.width, info.height) == (640, 360) and abs(info.fps - 30) < 0.01 and p.fps == 30
    assert abs(info.duration - p.duration) < 0.1 and abs(p.duration - 3) < 0.1
    import cv2
    strip = cv2.imread(p.strip)
    assert strip.shape[0] == 40 and strip.shape[1] == 6 * 71
    assert preview.make_preview(v, thumbs=6, thumb_height=40).video == p.video  # 作成済みなら再利用
