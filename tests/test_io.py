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
