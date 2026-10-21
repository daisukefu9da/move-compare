import shutil

import numpy as np
import pytest

from dance_compare.core.segment import sync_curve
from dance_compare.synth import add_groove, synth_beats, synth_dancer

# 振付は 6.5 秒（拍 0.5 + 6.0）から. それまでは各自でリズムを取っている
DANCE_START = 6.5


def pair(ref_seed=1, user_seed=2, bpm=None, styles=(None, None), **user_kw):
    ref = add_groove(synth_dancer(duration=24, delay_s=6.0), 0, 6.3, ref_seed, bpm=bpm, style=styles[0])
    user = add_groove(synth_dancer(duration=24, delay_s=6.05, noise=1.0, seed=3, **user_kw), 0, 6.3, user_seed,
                      bpm=bpm, style=styles[1])
    return ref, user


def test_sync_curve_finds_choreography_start():
    c = sync_curve(*pair())
    assert c.start is not None and abs(c.start - DANCE_START) < 0.7
    assert c.end > 22
    assert c.score[c.times < 5].max() < 0.3


@pytest.mark.parametrize("styles", [(0, 1), (1, 2), (0, 2), (0, 0), (1, 1)])
def test_grooving_to_the_same_beat_is_not_choreography(styles):
    # 同じ曲の拍に合わせて、それぞれのやり方でリズムを取っている
    c = sync_curve(*pair(1, 2, bpm=120, styles=styles))
    assert c.start is not None and abs(c.start - DANCE_START) < 0.7
    # 注: 二人がまったく同じやり方（例: 首と肩で同じようにノる = style 2 どうし）だと振付と区別できない.
    #     その場合は画面で開始・終了を直してもらう前提（README に記載）


def _two_people_tracks(ref, user):
    from dance_compare.pose.base import TrackSet

    ts = TrackSet(ref.fps, 640, 480, len(ref))
    for tid, s in ((1, ref), (2, user)):
        for f in range(len(s)):
            kp = s.keypoints[f]
            ts.add(tid, f, kp, s.scores[f], np.array([*kp.min(0), *kp.max(0)]))
    return ts


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg が必要")
def test_single_video_compares_only_the_choreography(tmp_path):
    from dance_compare.pipeline import Pick, analyze_single_video
    from dance_compare.synth import write_synth_video

    ref, user = pair(elbow_bend=(10.5, 12.5, 50))
    beats = synth_beats(24)
    video = str(tmp_path / "v.mp4")
    write_synth_video(ref, video, beats)
    ts = _two_people_tracks(ref, user)

    res = analyze_single_video(video, Pick(1), Pick(2), tracks=ts)
    assert abs(res.report.meta["ref_t0"] - DANCE_START) < 0.7
    assert any("比較した区間" in n for n in res.notes)
    g = res.report.count_grid()
    assert abs(g.first_one - DANCE_START) < 0.05  # 振付の始まりが「1エイト目の1」
    elbow = [f for f in res.report.findings if f.part == "right_elbow"]
    assert elbow and "2エイト目" in elbow[0].label  # 10.5 秒 = 2エイト目の1

    # 開始を手動で指定すると、そちらが優先される
    res = analyze_single_video(video, Pick(1), Pick(2), tracks=ts, time_range=(8.0, None))
    assert abs(res.report.meta["ref_t0"] - 8.0) < 0.05
