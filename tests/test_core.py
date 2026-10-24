import json

import numpy as np
import pytest

from dance_compare.core import compare
from dance_compare.core.config import AnalysisConfig
from dance_compare.core.counts import CountGrid
from dance_compare.core.dtw import dtw, pairwise_cost, path_to_warp
from dance_compare.core.preprocess import mirror, normalize
from dance_compare.core.report import Report
from dance_compare.core.rhythm import beat_grid, movement_accents, offsets_to_grid
from dance_compare.core.skeleton import PoseSequence
from dance_compare.synth import synth_beats, synth_dancer

BEATS = synth_beats(16)


def run(user, **kw):
    kw.setdefault("synced", True)
    return compare(synth_dancer(), user, ref_beats=BEATS, user_beats=kw.pop("user_beats", BEATS), **kw)


def test_dtw_recovers_time_stretch():
    t = np.linspace(0, 4 * np.pi, 100)
    a = np.stack([np.sin(t), np.cos(t)], 1)
    b = np.stack([np.sin(t ** 1.1 / (4 * np.pi) ** 0.1), np.cos(t ** 1.1 / (4 * np.pi) ** 0.1)], 1)
    path = dtw(pairwise_cost(a, b))
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (99, 99)
    assert np.all(np.diff(path, axis=0) >= 0)
    warp = path_to_warp(path, 100)
    expected = np.interp(t, t ** 1.1 / (4 * np.pi) ** 0.1, np.arange(100))
    assert np.median(np.abs(warp - expected)) < 2


def test_count_labels():
    g = CountGrid(np.arange(0.5, 20, 0.5), first_one=0.5)
    assert g.label(0.5) == "1エイト目の1カウント"
    assert g.label(0.75) == "1エイト目の1&カウント"
    assert g.label(4.5) == "2エイト目の1カウント"
    assert g.label(1.5, 2.5) == "1エイト目の3〜4カウント"
    assert g.label(3.5, 5.5) == "1エイト目の7〜2エイト目の2カウント"
    assert g.label(0.1) == "イントロ"


def test_count_from_bpm():
    g = CountGrid.from_bpm(120, first_beat=1.0, duration=10)
    assert g.label(1.0) == "1エイト目の1カウント"
    assert abs(g.bpm - 120) < 1e-6


def test_mirror_twice_is_identity_and_normalize_is_scale_invariant():
    s = synth_dancer(duration=2)
    m2 = mirror(mirror(s))
    assert np.allclose(m2.keypoints, s.keypoints)
    big = synth_dancer(duration=2, size=200, center=(100, 300))
    b1, _, _ = normalize(s.keypoints)
    b2, _, _ = normalize(big.keypoints)
    assert np.allclose(b1, b2, atol=1e-6)


def test_identical_dance_has_no_findings():
    r = run(synth_dancer(noise=1.0))
    assert r.findings == []
    assert r.scores["overall"] > 90


def test_body_size_and_position_do_not_matter():
    r = run(synth_dancer(size=80, center=(200, 300), fps=24))
    assert r.findings == []


def test_detects_bent_elbow_in_the_right_place():
    r = run(synth_dancer(elbow_bend=(4.5, 6.5, 50)))
    elbows = [f for f in r.findings if f.kind == "angle" and f.part == "right_elbow"]
    assert elbows and elbows[0].direction == "smaller"
    assert abs(elbows[0].value + 50) < 15
    assert elbows[0].t_start < 5.0 and elbows[0].t_end > 6.0
    assert "2エイト目" in elbows[0].label


def test_detects_delay_and_rhythm():
    r = run(synth_dancer(delay_s=0.1))
    assert 70 < r.rhythm["user"]["mean_ms"] < 130
    assert abs(r.rhythm["ref"]["mean_ms"]) < 20
    kinds = {(f.aspect, f.direction) for f in r.tendencies}
    assert not [f for f in r.findings + r.tendencies if f.aspect == "rhythm"]  # リズムは評価しない
    assert "rhythm" not in r.scores
    assert ("timing", "late") in kinds


def test_detects_small_and_soft_movement():
    r = run(synth_dancer(amplitude=0.6))
    assert any(f.kind == "amplitude" and f.direction == "smaller" for f in r.tendencies)
    r = run(synth_dancer(sharpness=0.6))
    assert any(f.kind == "sharpness" for f in r.tendencies)
    assert not any(f.kind == "amplitude" and f.direction == "smaller" for f in r.tendencies)


def test_tempo_difference_is_absorbed_when_not_synced():
    user = synth_dancer(tempo=0.9, delay_s=0.3, duration=16 / 0.9 + 0.3)
    r = run(user, synced=False, user_beats=0.3 + BEATS / 0.9)
    assert r.scores["arms"] > 90 and r.scores["legs"] > 90
    assert not any(f.kind == "early_late" for f in r.findings)
    assert 1.05 < r.meta["tempo_ratio"] < 1.15


def test_mirror_option():
    ref = synth_dancer(elbow_bend=(0, 16, 40))  # 手本は右肘を曲げている
    user = mirror(synth_dancer(elbow_bend=(0, 16, 40)))  # 向かい合わせ = 左肘を曲げて見える
    r = compare(ref, user, ref_beats=BEATS, user_beats=BEATS, synced=True, mirror_ref=False)
    assert any(f.part in ("left_elbow", "right_elbow") for f in r.findings)
    # 手本を反転すると、レビュー対象自身の左肘として一致する
    user2 = synth_dancer(elbow_bend=None)
    user2.keypoints[:] = mirror(ref).keypoints
    r = compare(ref, user2, ref_beats=BEATS, user_beats=BEATS, synced=True, mirror_ref=True)
    assert not any(f.kind == "angle" for f in r.findings)


def test_missing_keypoints_are_interpolated():
    user = synth_dancer(noise=1.0)
    user.scores[100:110, 9] = 0.0
    user.keypoints[100:110, 9] = np.nan
    r = run(user)
    assert r.scores["overall"] > 85


def test_accent_offsets_against_beats():
    s = synth_dancer(delay_s=0.05)
    from dance_compare.core.features import compute_features
    body, world, _ = normalize(s.keypoints)
    f = compute_features(body, world, s.fps)
    acc = movement_accents(f.body_speed, s.fps)
    offs = offsets_to_grid(acc, beat_grid(BEATS, 2))
    assert 30 < np.median(offs) < 70


def test_report_json_roundtrip(tmp_path):
    r = run(synth_dancer(elbow_bend=(4.5, 6.5, 50)))
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r.to_dict(), ensure_ascii=False))
    r2 = Report.from_dict(json.loads(p.read_text()))
    assert len(r2.findings) == len(r.findings)
    assert r2.count_grid().label(4.5) == r.count_grid().label(4.5)


def test_pose_sequence_json_roundtrip(tmp_path):
    s = synth_dancer(duration=1)
    s.keypoints[3, 4] = np.nan
    p = tmp_path / "p.json"
    s.save_json(str(p))
    s2 = PoseSequence.load_json(str(p))
    assert np.allclose(s.keypoints, s2.keypoints, equal_nan=True, atol=0.01)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        AnalysisConfig.from_dict({"nope": 1})


def test_count_grid_first_one_outside_detected_beats():
    g = CountGrid(np.arange(1.0, 20, 0.5), first_one=0.5)  # 0.5 の拍が検出されていない
    assert g.label(0.5) == "1エイト目の1カウント"
    assert g.label(1.0) == "1エイト目の2カウント"
    assert abs(g.first_one - 0.5) < 1e-9


def test_body_proportions_do_not_count_as_level_difference():
    # 脚が長い（胴に対する脚の比率が違う）人でも「腰の位置が高い」とは判定しない
    user = synth_dancer(noise=1.0)
    kp = user.keypoints
    for j in (13, 14, 15, 16):
        kp[:, j, 1] += (kp[:, j, 1] - kp[:, 11, 1]) * 0.25
    r = run(user)
    assert not [f for f in r.findings + r.tendencies if f.kind == "level"]
    assert r.scores["hips"] > 95


def test_scores_follow_thresholds():
    from dance_compare.core.analyzer import grade

    assert grade(0.3) == 100 and grade(1.0) == 80 and grade(2.0) == 50 and grade(6.0) == 0
    user = synth_dancer(amplitude=0.6, noise=1.0)
    strict = run(user)
    loose = compare(synth_dancer(), user, AnalysisConfig(amplitude_tolerance=0.45), ref_beats=BEATS,
                    user_beats=BEATS, synced=True)
    assert loose.scores["arms"] > strict.scores["arms"] + 5  # 腕のスコアには肩の角度も入るので上がり幅は小さめ
    assert run(synth_dancer(noise=2.0)).scores["overall"] > 98


def test_old_config_keys_are_converted():
    c = AnalysisConfig.from_dict({"amplitude_ratio_low": 0.8, "sharpness_ratio_low": 0.6, "score_scale_angle_deg": 30})
    assert abs(c.amplitude_tolerance - 0.2) < 1e-9 and abs(c.sharpness_tolerance - 0.4) < 1e-9


def test_part_weights():
    user = synth_dancer(elbow_bend=(4.5, 6.5, 50), noise=1.0)
    r = run(user)
    assert r.scores["arms"] < 100 and any(f.group == "arms" for f in r.findings)
    r0 = compare(synth_dancer(), user, AnalysisConfig(weight_arms=0), ref_beats=BEATS, user_beats=BEATS, synced=True)
    assert r0.scores["arms"] is None and r0.scores["overall"] > 99  # 腕を評価しない
    assert not any(f.group == "arms" for f in r0.findings + r0.tendencies) and not r0.moments
    r2 = compare(synth_dancer(), user, AnalysisConfig(weight_arms=3), ref_beats=BEATS, user_beats=BEATS, synced=True)
    assert r2.scores["overall"] < r.scores["overall"]  # 腕を重く見ると総合点が下がる


def test_head_tilt_is_detected():
    r = run(synth_dancer(head_tilt=(4.5, 8.5, 25), noise=1.0))
    heads = [f for f in r.findings if f.kind == "head_tilt"]
    assert heads and heads[0].group == "head" and "2エイト目" in heads[0].label
    assert abs(abs(heads[0].value) - 25) < 8
    assert r.scores["head"] < 97  # 16 秒のうち 4 秒だけ倒している
    assert not [f for f in r.findings if f.group in ("arms", "legs")]


def test_moments_pick_the_biggest_distinct_places():
    user = synth_dancer(elbow_bend=(4.5, 5.5, 60), head_tilt=(10.5, 12.5, 30), noise=1.0)
    r = run(user)
    assert 1 <= len(r.moments) <= 3
    kinds = [m["items"][0]["kind"] for m in r.moments]
    assert "angle" in kinds and "head_tilt" in kinds
    m = next(m for m in r.moments if m["items"][0]["kind"] == "angle")
    assert 4.4 <= m["t"] <= 5.6 and 8 in m["joints"]  # 右肘
    assert m["rank"] == 1 or r.moments[0]["score"] >= m["score"]


def test_limit_rate_removes_freezes_and_jumps():
    from dance_compare.core.analyzer import limit_rate

    t = np.arange(100, dtype=float)
    warp = t + 10.0
    warp[40:52] = warp[40]           # 12 フレーム止まる
    warp[52:] -= 0.0                 # その後は元に戻る（急に追いつく）
    out = limit_rate(warp, 0.5, 2.0)
    d = np.diff(out)
    assert d.min() >= 0.5 - 1e-9 and d.max() <= 2.0 + 1e-9
    assert np.abs(out - warp).max() < 6  # 元の対応から大きく外れない
    assert np.allclose(limit_rate(t, 0.5, 2.0), t)  # 問題のない対応はそのまま


def test_warp_rate_is_bounded_in_compare():
    r = run(synth_dancer(delay_s=0.1, noise=1.0))
    w = np.asarray(r.warp["user"])
    rate = np.diff(w) / np.diff(np.asarray(r.warp["ref"]))
    assert rate.min() >= 0.5 - 0.02 and rate.max() <= 2.0 + 0.02
