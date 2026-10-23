"""手本とレビュー対象の骨格系列を比較し、Report を作る（解析の本体）."""

from __future__ import annotations

import numpy as np

import copy
from dataclasses import asdict

from .config import AnalysisConfig
from .counts import CountGrid
from .dtw import dtw, monotonic, pairwise_cost, path_to_warp
from .features import Features, compute_features
from .parts import ASPECT_OF_KIND, GROUPS, group_of, joints_of
from .preprocess import clean, gaussian_smooth, mirror, normalize, resample
from .report import Finding, Report
from .rhythm import (beat_grid, foot_landings, match_events, movement_accents,
                     offset_stats, offsets_to_grid)
from .skeleton import ANGLE_NAMES, END_EFFECTORS, PoseSequence

END_NAMES = list(END_EFFECTORS)


def _prepare(seq: PoseSequence, cfg: AnalysisConfig) -> tuple[PoseSequence, Features]:
    seq = resample(clean(seq, cfg.conf_threshold, cfg.smooth_sigma_s), cfg.analysis_fps)
    body, world, _ = normalize(seq.keypoints)
    return seq, compute_features(body, world, seq.fps, seq.scores)


def _sample(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """axis 0 を実数インデックス pos で線形補間して取り出す."""
    n = x.shape[0]
    pos = np.clip(pos, 0, n - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def align(ref: Features, user: Features, cfg: AnalysisConfig) -> np.ndarray:
    """DTW で、手本の各フレームに対応するレビュー対象のフレーム位置（実数）を求める."""
    step = max(1, int(round(ref.fps / cfg.dtw_fps)))
    a = ref.dtw_vectors()[::step]
    b = user.dtw_vectors()[::step]
    cost = pairwise_cost(a, b)
    penalty = cfg.dtw_step_penalty * float(np.median(cost))
    dtw_rate = ref.fps / step
    band = None if cfg.dtw_band_s is None else max(2, int(cfg.dtw_band_s * dtw_rate))
    path = dtw(cost, band=band, step_penalty=penalty, open_ends=cfg.dtw_open_ends)
    warp = path_to_warp(path, a.shape[0]) * step
    warp = np.interp(np.arange(ref.n) / step, np.arange(a.shape[0]), warp)
    warp = gaussian_smooth(warp, cfg.warp_smooth_s * ref.fps)
    warp = np.clip(monotonic(warp), 0, user.n - 1)
    warp = refine_warp(ref.dtw_vectors(), user.dtw_vectors(), warp, ref.fps, cfg)
    return np.clip(monotonic(warp), 0, user.n - 1)


def refine_warp(a: np.ndarray, b: np.ndarray, warp: np.ndarray, fps: float, cfg: AnalysisConfig,
                max_shift_s: float = 0.2, step: float = 0.25) -> np.ndarray:
    """DTW の結果を、前後 max_shift_s の範囲で 1/4 フレーム刻みにずらして局所的に合わせ直す.

    DTW は間引いたフレームで計算するので、時間差の分解能が粗い（15fps なら 67ms）。
    """
    shifts = np.arange(-max_shift_s * fps, max_shift_s * fps + 1e-9, step)
    cost = np.stack([np.sum((_sample(b, warp + d) - a) ** 2, axis=1) for d in shifts], axis=1)
    cost = gaussian_smooth(cost, 0.15 * fps)  # 前後 0.15 秒ほどの動きでまとめて判断する
    k = np.argmin(cost, axis=1)
    # 放物線補間で刻みより細かく
    kk = np.clip(k, 1, len(shifts) - 2)
    y0, y1, y2 = cost[np.arange(len(k)), kk - 1], cost[np.arange(len(k)), kk], cost[np.arange(len(k)), kk + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    best = shifts[kk] + np.clip(frac, -1, 1) * step
    return warp + gaussian_smooth(best, cfg.warp_smooth_s * fps)


def _windows(t_abs: np.ndarray, grid: CountGrid | None, cfg: AnalysisConfig) -> list[tuple[float, float]]:
    t0, t1 = float(t_abs[0]), float(t_abs[-1]) + 1e-9
    if grid is not None:
        wc = cfg.window_counts
        # 拍ちょうどに来る動きがその拍の区間に入るよう、区間を 1/4 拍前にずらす
        k = np.floor((grid.beat_position(t0) + 0.25) / wc) * wc - 0.25
        out = []
        while True:
            a, b = float(grid.time_of(k)), float(grid.time_of(k + wc))
            if a >= t1:
                break
            if b > t0:
                out.append((max(a, t0), min(b, t1)))
            k += wc
        return out
    edges = np.arange(t0, t1, cfg.window_s_fallback)
    return [(float(a), float(min(a + cfg.window_s_fallback, t1))) for a in edges]


def _abruptness(accel: np.ndarray, speed: np.ndarray) -> float:
    if len(accel) < 2:
        return float("nan")
    v = float(np.percentile(speed, 90))
    return float(np.percentile(accel, 90)) / v if v > 1e-6 else float("nan")


def grade(excess, full_ratio: float = 0.5) -> np.ndarray:
    """ズレ（しきい値の何倍か）-> 点数.

    しきい値の full_ratio 倍以下 = 100, しきい値 = 80, 2 倍 = 50, 3 倍 = 25, 6 倍以上 = 0。
    指摘が出ない程度のズレなら 80 点以上になり、しきい値を緩めると点数も上がる。
    """
    full = min(full_ratio, 0.95)
    return np.interp(np.abs(excess), [0, full, 1, 2, 3, 6], [100, 100, 80, 50, 25, 0])


def _worst_and_mean(points: np.ndarray) -> float:
    """区間内の複数の部位の点数をまとめる. 1 か所でも大きくずれていれば区間の点が下がるよう、最低点も半分入れる."""
    return float(0.5 * points.min() + 0.5 * points.mean())


def window_excess(w: dict, cfg: AnalysisConfig) -> dict[str, list[float]]:
    """区間の各項目のズレを「しきい値の何倍か」で、部位グループごとに集める（スコア計算用）."""
    out: dict[str, list[float]] = {g: [] for g in GROUPS}
    for name, d in w["angle_diff"].items():
        out[group_of("angle", name)].append(d / cfg.angle_threshold_deg)
    if w["ref_motion"] >= cfg.min_motion_for_timing and not w["edge"]:
        out["timing"].append(w["timing_ms"] / cfg.timing_threshold_ms)
    if not w["partial"]:
        amp_log, sh_log = -np.log(cfg.amplitude_low), -np.log(cfg.sharpness_low)
        for name, r in w["amplitude_ratio"].items():
            if np.isfinite(r) and r > 0:
                out[group_of("amplitude", name)].append(np.log(r) / amp_log)
        for name, r in w["sharpness_ratio"].items():
            if np.isfinite(r) and r > 0:  # 手本よりキレがある分は減点しない
                out[group_of("sharpness", name)].append(max(0.0, -np.log(r)) / sh_log)
    out["hips"] += [w["level_diff"] / cfg.level_threshold, w["facing_diff"] / cfg.facing_threshold_deg,
                    w["lean_diff"] / cfg.lean_threshold_deg, w["travel_diff"] / cfg.travel_threshold]
    out["legs"].append(w["stance_diff"] / cfg.stance_threshold)
    if w.get("landing_ms") is not None:
        out["legs"].append(w["landing_ms"] / cfg.landing_threshold_ms)
    for key, thr in (("tilt", cfg.head_tilt_threshold_deg), ("turn", cfg.head_turn_threshold_deg),
                     ("shift", cfg.head_shift_threshold), ("nod", cfg.head_nod_threshold)):
        v = w.get(f"head_{key}_diff")
        if v is not None and np.isfinite(v):
            out["head"].append(v / thr)
    return out


def _scores(windows: list[dict], cfg: AnalysisConfig) -> dict:
    """部位ごとのスコアと、重み付きの総合スコア.

    区間ごとの一貫したズレ（指摘と同じ単位）から計算する。1 コマごとの角度などは骨格推定のぶれを
    含むので使わない。リズム（拍とのズレ）は使わない（遅れ・走りは「手本とのタイミング」で評価）。
    """
    per_group: dict[str, list[float]] = {g: [] for g in GROUPS}
    for w in windows:
        for g, ex in window_excess(w, cfg).items():
            ex = [e for e in ex if e is not None and np.isfinite(e)]  # 画面外などで取れない項目は除く
            if ex:
                per_group[g].append(_worst_and_mean(grade(np.array(ex), cfg.score_full_ratio)))
    scores: dict = {g: (float(np.mean(v)) if v else None) for g, v in per_group.items()}
    total = weight = 0.0
    for g, v in scores.items():
        wg = cfg.weight(g)
        if v is not None and wg > 0:
            total += wg * v
            weight += wg
        elif wg <= 0:
            scores[g] = None  # 重み 0 の部位は評価しない
    scores["overall"] = total / weight if weight > 0 else None
    return scores


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def compare(
    ref: PoseSequence,
    user: PoseSequence,
    cfg: AnalysisConfig | None = None,
    *,
    ref_beats: np.ndarray | None = None,
    user_beats: np.ndarray | None = None,
    first_one: float | None = None,
    mirror_ref: bool = False,
    synced: bool = False,
) -> Report:
    """手本 ref とレビュー対象 user を比較する.

    ref_beats / user_beats: それぞれの元動画上の拍の時刻 [秒]（無ければ None）
    first_one: 手本動画上で「1エイト目の 1」に当たる時刻（None なら最初の拍）
    mirror_ref: 手本を左右反転して比較する（向かい合わせで撮影した手本用）
    synced: 2 つの系列の先頭が同じ音楽上の時刻に揃っている（同じ動画・音声同期済み）。
            True なら手本との時間差を絶対値で評価し、False なら全体のテンポ差は除いて評価する。
    """
    cfg = cfg or AnalysisConfig()
    if mirror_ref:
        ref = mirror(ref)
    ref_s, R = _prepare(ref, cfg)
    user_s, U = _prepare(user, cfg)
    fps = cfg.analysis_fps

    warp = align(R, U, cfg)
    t_ref = np.arange(R.n) / fps
    t_user = warp / fps
    if synced:
        expected = t_ref.copy()
    else:
        slope, intercept = np.polyfit(t_ref, t_user, 1)
        expected = slope * t_ref + intercept
    timing_dev = (t_user - expected) * 1000  # 正 = レビュー対象が遅い

    # 手本フレームごとの、対応するレビュー対象の値
    angle_diff = _sample(U.angles, warp) - R.angles
    # 腰の高さは各自の普段の高さ（中央値）からの沈み込みで比べる. そのまま比べると、脚と胴の長さの比率
    # （体型）や撮影位置の違いが「手本より低い/高い」と出てしまうため
    level_diff = (_sample(U.level, warp) - np.median(U.level)) - (R.level - np.median(R.level))
    facing_diff = _sample(U.facing, warp) - R.facing
    lean_diff = _sample(U.lean, warp) - R.lean
    stance_diff = _sample(U.stance, warp) - R.stance
    # 頭: 傾き・向きはそのまま、横スライド・上下は各自の普段の位置（中央値）からの差で比べる（顔の形・撮影角度の違いを除く）
    head_diff = {}
    for key in ("tilt", "turn", "shift", "nod"):
        u, r = _sample(U.head[key], warp), R.head[key]
        if key in ("shift", "nod"):
            u, r = u - np.nanmedian(U.head[key]), r - np.nanmedian(R.head[key])
        head_diff[key] = u - r
    head_ok = ((np.minimum(_sample(U.head_conf, warp), R.head_conf) >= cfg.head_min_conf)
               & (_sample(U.facing, warp) < 120) & (R.facing < 120))  # 顔が見えている（背中向きでない）コマだけ

    t_abs = t_ref + ref_s.t0
    grid = None
    if ref_beats is not None and len(ref_beats) >= 2:
        grid = CountGrid(np.asarray(ref_beats), first_one)

    # --- リズム ---
    rhythm: dict = {"accent_mode": cfg.accent_mode}
    ref_acc = movement_accents(R.body_speed, fps, cfg.accent_mode, cfg.accent_min_interval_s, cfg.accent_prominence)
    user_acc = movement_accents(U.body_speed, fps, cfg.accent_mode, cfg.accent_min_interval_s, cfg.accent_prominence)
    ref_acc_abs = ref_acc + ref_s.t0
    user_acc_abs = user_acc + user_s.t0
    user_acc_on_ref = np.interp(user_acc * fps, warp, t_ref) + ref_s.t0  # 手本の時間軸へ
    rhythm["ref_accents"] = ref_acc_abs.tolist()
    rhythm["user_accents"] = user_acc_abs.tolist()
    user_acc_off = np.full(user_acc.size, np.nan)
    land_r = foot_landings(R.world, fps, cfg.landing_speed_on, cfg.landing_speed_off)
    land_u = foot_landings(U.world, fps, cfg.landing_speed_on, cfg.landing_speed_off)
    if ref_beats is not None and len(ref_beats) >= 2:
        g = beat_grid(np.asarray(ref_beats), cfg.beat_subdivision)
        rhythm["ref"] = offset_stats(offsets_to_grid(ref_acc_abs, g), cfg.on_beat_tolerance_ms)
        rl = np.concatenate([v for v in land_r.values()]) + ref_s.t0
        rhythm["ref_landing"] = offset_stats(offsets_to_grid(rl, g), cfg.on_beat_tolerance_ms)
    if user_beats is not None and len(user_beats) >= 2:
        g = beat_grid(np.asarray(user_beats), cfg.beat_subdivision)
        offs = offsets_to_grid(user_acc_abs, g)
        rhythm["user"] = offset_stats(offs, cfg.on_beat_tolerance_ms)
        # 区間への割り当て用に、アクセントごとのズレを保持
        for k, t in enumerate(user_acc_abs):
            o = offsets_to_grid(np.array([t]), g)
            if o.size:
                user_acc_off[k] = o[0]
        ul = np.concatenate([v for v in land_u.values()]) + user_s.t0
        rhythm["user_landing"] = offset_stats(offsets_to_grid(ul, g), cfg.on_beat_tolerance_ms)
    # 着地を手本と比べる（予想時刻の付近で同じ足の着地を探す）
    land_vs_ref = {}
    for foot in land_r:
        exp_t = np.interp(land_r[foot] * fps, np.arange(R.n), expected)
        land_vs_ref[foot] = match_events(land_r[foot], land_u[foot], exp_t, cfg.landing_match_window_s)
    all_land = np.concatenate(list(land_vs_ref.values())) if land_vs_ref else np.array([])
    rhythm["landing_vs_ref"] = offset_stats(all_land[np.isfinite(all_land)], cfg.on_beat_tolerance_ms)
    rhythm["bpm"] = grid.bpm if grid is not None else None
    # 手本自体の拍に対するズレ. 拍検出の癖（数十 ms 早い/遅い）や曲のノリを含むので、レビュー対象の評価は
    # これを基準にする（手本が無い/少ない場合は 0）
    ref_stats = rhythm.get("ref", {})
    rhythm["bias_ms"] = float(ref_stats["median_ms"]) if ref_stats.get("count", 0) >= 4 else 0.0

    # --- 区間ごとの評価 ---
    windows = []
    win = _windows(t_abs, grid, cfg)
    nominal = max((b - a) for a, b in win) if win else 0
    for (a, b) in win:
        m = (t_abs >= a) & (t_abs < b)
        if m.sum() < 2 or (b - a) < nominal * 0.5:  # 端の短すぎる区間は評価しない
            continue
        idx = np.where(m)[0]
        w = {
            "t_start": a, "t_end": b,
            "label": grid.label(a, b) if grid is not None else None,
            "angle_diff": {n: float(np.mean(angle_diff[m, k])) for k, n in enumerate(ANGLE_NAMES)},
            "timing_ms": float(np.mean(timing_dev[m])),
            "ref_motion": float(np.mean(R.body_speed[m])),  # 手本の動きの速さ（止まっている区間は時間差が測れない）
            # 系列の両端は DTW の端点の影響で時間差が不正確になりやすい
            # 動画の端で途中までしか無い区間（振れ幅・キレは動きの一部しか見えず比較できない）
            "partial": bool((b - a) < nominal * 0.9),
            "edge": bool(a - t_abs[0] < cfg.edge_margin_s or t_abs[-1] - b < cfg.edge_margin_s),
            "level_diff": float(np.mean(level_diff[m])),
            "facing_diff": float(np.mean(facing_diff[m])),
            "lean_diff": float(np.mean(lean_diff[m])),
            "stance_diff": float(np.mean(stance_diff[m])),
        }
        hm = m & head_ok
        for key in head_diff:  # 顔が見えているコマが区間の半分未満なら評価しない
            w[f"head_{key}_diff"] = float(np.mean(head_diff[key][hm])) if hm.sum() >= 0.5 * m.sum() else None
        # 移動量（重心 = 腰の中点の移動）
        r_move = R.hip[idx[-1]] - R.hip[idx[0]]
        u_move = _sample(U.hip, np.array([warp[idx[-1]]]))[0] - _sample(U.hip, np.array([warp[idx[0]]]))[0]
        w["travel_ref"] = r_move.tolist()
        w["travel_user"] = u_move.tolist()
        w["travel_diff"] = float(np.linalg.norm(u_move - r_move))
        # 動きの大きさ（末端の振れ幅）とキレ（加速度）. 速度・加速度は各自の時間軸で微分済みの値を対応づけて使う
        wm = warp[m]
        amp, sharp = {}, {}
        for k, name in enumerate(END_NAMES):
            j = END_EFFECTORS[name]
            r_ext = float(np.linalg.norm(np.ptp(R.body[m, j], axis=0)))
            u_ext = float(np.linalg.norm(np.ptp(_sample(U.body[:, j], wm), axis=0)))
            amp[name] = u_ext / r_ext if r_ext >= 0.3 else np.nan
            # キレ = 加速度 / 速さ（動きの大きさに依存しない「急さ」）
            r_sharp = _abruptness(R.end_accel[m, k], R.end_speed[m, k])
            u_sharp = _abruptness(_sample(U.end_accel[:, k], wm), _sample(U.end_speed[:, k], wm))
            sharp[name] = u_sharp / r_sharp if r_ext >= 0.5 and r_sharp > 0 else np.nan
        w["amplitude_ratio"] = amp
        w["sharpness_ratio"] = sharp
        # 着地のタイミング（この区間にある手本の着地と、対応するレビュー対象の着地の差）
        lt = np.concatenate([land_r[ft] + ref_s.t0 for ft in land_r])
        lv = np.concatenate([land_vs_ref[ft] for ft in land_r])
        sel = (lt >= a) & (lt < b) & np.isfinite(lv)
        w["landing_ms"] = float(np.mean(lv[sel])) if sel.any() else None
        w["landing_feet"] = sorted({ft for ft in land_r for x, y in zip(land_r[ft] + ref_s.t0, land_vs_ref[ft])
                                    if a <= x < b and np.isfinite(y) and abs(y) >= cfg.landing_threshold_ms})
        in_w = (user_acc_on_ref >= a) & (user_acc_on_ref < b) & np.isfinite(user_acc_off)
        w["rhythm_ms"] = float(np.mean(user_acc_off[in_w])) if in_w.sum() >= 2 else None
        windows.append(w)

    raw = _window_findings(windows, cfg)
    findings = _merge(copy.deepcopy(raw), grid)
    tendencies = _tendencies(windows, rhythm, timing_dev, synced, cfg)
    scores = _scores(windows, cfg)
    series = {
        "angle": angle_diff, "level": level_diff, "facing": facing_diff, "lean": lean_diff,
        "stance": stance_diff, "early_late": timing_dev,
        **{f"head_{k}": np.where(head_ok, v, 0.0) for k, v in head_diff.items()},
    }
    moments = _moments(raw, windows, series, R, t_abs, t_user + user_s.t0, land_r, ref_s.t0, grid, cfg)

    step = max(1, int(round(fps / 10)))  # グラフ用に 10fps 程度へ間引く
    timeline = {
        "t": t_abs[::step],
        "angle_diff": {n: angle_diff[::step, k] for k, n in enumerate(ANGLE_NAMES)},
        "timing_ms": timing_dev[::step],
        "level_diff": level_diff[::step],
        "facing_diff": facing_diff[::step],
        "ref_speed": R.body_speed[::step],
        "user_speed": _sample(U.body_speed, warp)[::step],
    }
    meta = {
        "ref_t0": ref_s.t0, "ref_duration": ref_s.duration,
        "user_t0": user_s.t0, "user_duration": user_s.duration,
        "mirror_ref": mirror_ref, "synced": synced,
        "tempo_ratio": float((t_user[-1] - t_user[0]) / max(t_ref[-1] - t_ref[0], 1e-9)),
        "config": cfg.to_dict(),
    }
    return Report(
        meta=meta, scores=scores, findings=findings, tendencies=tendencies, windows=windows,
        rhythm=rhythm, timeline=timeline,
        warp={"ref": t_abs[::3], "user": (t_user + user_s.t0)[::3]},
        beats=[] if grid is None else grid.beats.tolist(),
        first_one=None if grid is None else grid.first_one,
        moments=moments,
    )


# ---------------------------------------------------------------------------
# 指摘の抽出
# ---------------------------------------------------------------------------

def _window_findings(windows: list[dict], cfg: AnalysisConfig) -> list[Finding]:
    """区間ごとの指摘（隣り合う区間をまとめる前）. 優先度には部位の重みを掛け、重み 0 の部位は出さない."""
    raw: list[Finding] = []

    def add(w, kind, direction, value, unit, sev, part=None, **extra):
        group = group_of(kind, part)
        weight = cfg.weight(group)
        if weight <= 0:
            return
        raw.append(Finding(aspect=ASPECT_OF_KIND[kind], kind=kind, direction=direction, value=float(value),
                           unit=unit, severity=float(sev) * weight, part=part, t_start=w["t_start"],
                           t_end=w["t_end"], label=w["label"], extra=extra, group=group))

    for w in windows:
        for name, d in w["angle_diff"].items():
            if abs(d) >= cfg.angle_threshold_deg:
                # 角度が小さい = より曲がっている（肩・股関節は腕・脚が体に近い）
                add(w, "angle", "smaller" if d < 0 else "larger", d, "deg", abs(d) / cfg.angle_threshold_deg,
                    part=name)
        if (abs(w["timing_ms"]) >= cfg.timing_threshold_ms and w["ref_motion"] >= cfg.min_motion_for_timing
                and not w["edge"]):
            add(w, "early_late", "late" if w["timing_ms"] > 0 else "early", w["timing_ms"], "ms",
                abs(w["timing_ms"]) / cfg.timing_threshold_ms)
        for name, r in w["amplitude_ratio"].items():
            if not np.isfinite(r) or w["partial"]:
                continue
            if r < cfg.amplitude_low:
                add(w, "amplitude", "smaller", r, "ratio", np.log(r) / np.log(cfg.amplitude_low), part=name)
            elif r > cfg.amplitude_high:
                add(w, "amplitude", "larger", r, "ratio", np.log(r) / np.log(cfg.amplitude_high), part=name)
        for name, r in w["sharpness_ratio"].items():
            if np.isfinite(r) and r < cfg.sharpness_low and not w["partial"]:
                add(w, "sharpness", "weaker", r, "ratio", np.log(r) / np.log(cfg.sharpness_low), part=name)
        lm = w.get("landing_ms")
        if lm is not None and abs(lm) >= cfg.landing_threshold_ms:
            feet = w.get("landing_feet") or []
            add(w, "landing", "late" if lm > 0 else "early", lm, "ms", abs(lm) / cfg.landing_threshold_ms,
                part=feet[0] if len(feet) == 1 else None)
        if abs(w["level_diff"]) >= cfg.level_threshold:
            add(w, "level", "higher" if w["level_diff"] > 0 else "lower", w["level_diff"], "torso",
                abs(w["level_diff"]) / cfg.level_threshold)
        if abs(w["facing_diff"]) >= cfg.facing_threshold_deg:
            add(w, "facing", "more_turned" if w["facing_diff"] > 0 else "less_turned", w["facing_diff"], "deg",
                abs(w["facing_diff"]) / cfg.facing_threshold_deg)
        if abs(w["lean_diff"]) >= cfg.lean_threshold_deg:
            add(w, "lean", "right" if w["lean_diff"] > 0 else "left", w["lean_diff"], "deg",
                abs(w["lean_diff"]) / cfg.lean_threshold_deg)
        if abs(w["stance_diff"]) >= cfg.stance_threshold:
            add(w, "stance", "wider" if w["stance_diff"] > 0 else "narrower", w["stance_diff"], "torso",
                abs(w["stance_diff"]) / cfg.stance_threshold)
        if w["travel_diff"] >= cfg.travel_threshold:
            r, u = np.array(w["travel_ref"]), np.array(w["travel_user"])
            add(w, "travel", "less" if np.linalg.norm(u) < np.linalg.norm(r) else "more", w["travel_diff"], "torso",
                w["travel_diff"] / cfg.travel_threshold, ref=r.tolist(), user=u.tolist())
        for key, thr, unit, pos, neg in (
                ("tilt", cfg.head_tilt_threshold_deg, "deg", "right", "left"),
                ("turn", cfg.head_turn_threshold_deg, "deg", "left", "right"),
                ("shift", cfg.head_shift_threshold, "torso", "left", "right"),
                ("nod", cfg.head_nod_threshold, "torso", "up", "down")):
            v = w.get(f"head_{key}_diff")
            if v is not None and abs(v) >= thr:
                add(w, f"head_{key}", pos if v > 0 else neg, v, unit, abs(v) / thr)
    return raw


def _merge(raw: list[Finding], grid: CountGrid | None) -> list[Finding]:
    """隣り合う区間の同じ指摘をまとめる."""
    raw.sort(key=lambda f: (f.aspect, f.kind, f.part or "", f.direction, f.t_start))
    merged: list[Finding] = []
    for f in raw:
        last = merged[-1] if merged else None
        if (last is not None and (last.aspect, last.kind, last.part, last.direction)
                == (f.aspect, f.kind, f.part, f.direction) and abs(last.t_end - f.t_start) < 1e-6):
            n = last.extra["n_windows"]
            last.value = (last.value * n + f.value) / (n + 1)
            last.extra["n_windows"] = n + 1
            last.extra["max_severity"] = max(last.extra["max_severity"], f.severity)
            last.t_end = f.t_end
            continue
        f.extra.update(n_windows=1, max_severity=f.severity)
        merged.append(f)
    for f in merged:
        n = f.extra["n_windows"]
        f.severity = float(f.extra.pop("max_severity") * (1 + 0.3 * (n - 1)))
        if grid is not None:
            f.label = grid.label(f.t_start, f.t_end)
    merged.sort(key=lambda f: -f.severity)
    return merged


def _tendencies(windows, rhythm, timing_dev, synced, cfg) -> list[Finding]:
    out: list[Finding] = []
    lv = rhythm.get("landing_vs_ref", {})
    if lv.get("count", 0) >= 3 and abs(lv["mean_ms"]) >= cfg.landing_threshold_ms:
        out.append(Finding("timing", "landing", "late" if lv["mean_ms"] > 0 else "early", lv["mean_ms"], "ms",
                           abs(lv["mean_ms"]) / cfg.landing_threshold_ms, group="legs"))
    if synced:
        mean_t = float(np.median(timing_dev))
        if abs(mean_t) >= cfg.timing_threshold_ms * 0.6:
            out.append(Finding("timing", "early_late", "late" if mean_t > 0 else "early", mean_t, "ms",
                               abs(mean_t) / cfg.timing_threshold_ms * 1.5, group="timing"))
    for name in END_NAMES:
        vals = [w["amplitude_ratio"][name] for w in windows if np.isfinite(w["amplitude_ratio"][name])]
        if len(vals) >= 3:
            r = float(np.exp(np.mean(np.log(vals))))
            if r < cfg.amplitude_low:
                out.append(Finding("amplitude", "amplitude", "smaller", r, "ratio",
                                   np.log(r) / np.log(cfg.amplitude_low) * 1.5, part=name,
                                   group=group_of("amplitude", name)))
            elif r > cfg.amplitude_high:
                out.append(Finding("amplitude", "amplitude", "larger", r, "ratio",
                                   np.log(r) / np.log(cfg.amplitude_high) * 1.5, part=name,
                                   group=group_of("amplitude", name)))
        vals = [w["sharpness_ratio"][name] for w in windows if np.isfinite(w["sharpness_ratio"][name])]
        if len(vals) >= 3:
            r = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-6)))))
            if r < cfg.sharpness_low:
                out.append(Finding("sharpness", "sharpness", "weaker", r, "ratio",
                                   np.log(r) / np.log(cfg.sharpness_low) * 1.5, part=name,
                                   group=group_of("sharpness", name)))
    # 部位の重みを掛け、重み 0 の部位は出さない
    out = [f for f in out if cfg.weight(f.group) > 0]
    for f in out:
        f.severity *= cfg.weight(f.group)
    out.sort(key=lambda f: -f.severity)
    return out


# ---------------------------------------------------------------------------
# 直すところ（ズレの大きい瞬間の上位）
# ---------------------------------------------------------------------------

def _peak_frame(f: Finding, m: np.ndarray, series: dict, R: Features, land_r: dict, t0: float,
                fps: float) -> int:
    """指摘の区間の中で、ズレが一番はっきり見えるコマ（手本の時間軸のフレーム番号）."""
    idx = np.where(m)[0]
    if f.kind == "angle":
        x = series["angle"][:, ANGLE_NAMES.index(f.part)]
    elif f.kind in series:
        x = series[f.kind]
    elif f.kind in ("amplitude", "sharpness") and f.part in END_EFFECTORS:
        # 手本の手先・足先が区間内でいちばん大きく出ているコマ（振れ幅の差が見えやすい）
        p = R.body[:, END_EFFECTORS[f.part]]
        x = np.linalg.norm(p - p[m].mean(axis=0), axis=1)
    elif f.kind == "landing":
        feet = [f.part] if f.part in land_r else list(land_r)
        times = np.concatenate([land_r[ft] for ft in feet]) if feet else np.array([])
        times = times[(times + t0 >= f.t_start) & (times + t0 < f.t_end)]
        if times.size:
            return int(np.clip(round(times[0] * fps), idx[0], idx[-1]))
        return int(idx[len(idx) // 2])
    else:
        return int(idx[len(idx) // 2])
    xs = gaussian_smooth(np.abs(np.nan_to_num(x)), 0.1 * fps)
    return int(idx[np.argmax(xs[idx])])


def _moments(raw: list[Finding], windows: list[dict], series: dict, R: Features, t_abs: np.ndarray,
             t_user_abs: np.ndarray, land_r: dict, t0: float, grid: CountGrid | None,
             cfg: AnalysisConfig) -> list[dict]:
    """ズレの大きい瞬間を上位 cfg.top_moments 個選ぶ.

    区間ごとに指摘の優先度（部位の重み込み）から点数を出し、隣り合わない区間から大きい順に選ぶ。
    同じ内容の指摘が繰り返し出る場合（毎エイト同じ癖など）は、別の課題が入るよう 2 つ目以降で軽く見る。
    各瞬間は、その区間で一番大きい指摘のズレが最も大きいコマ。
    """
    by_win: dict[int, list[Finding]] = {}
    starts = [w["t_start"] for w in windows]
    for f in raw:
        if f.t_start in starts:
            by_win.setdefault(starts.index(f.t_start), []).append(f)
    def key(f: Finding) -> tuple:
        return f.kind, f.part, f.direction

    def score(i: int, covered: set) -> float:
        """一番大きい指摘で決める（残りは 1/4 だけ足す. 中くらいのズレが複数あるより、大きな 1 つを優先）.
        すでに選んだ瞬間と同じ内容の指摘は軽く見る（繰り返しを避ける）."""
        sev = sorted((f.severity * (0.3 if key(f) in covered else 1.0) for f in by_win[i]), reverse=True)[:3]
        return sev[0] + 0.25 * sum(sev[1:]) if sev else 0.0

    chosen: list[int] = []
    covered: set = set()
    while len(chosen) < cfg.top_moments:
        # 同じ場所ばかりにならないよう選んだ区間の隣は避け、一番大きい指摘がすでに見せた内容と同じ区間も除く
        # （毎エイト同じ癖が出る場合は 1 か所だけ見せ、「他に何か所」を添える）
        cands = [i for i in by_win if all(abs(i - j) >= 2 for j in chosen)
                 and key(max(by_win[i], key=lambda f: f.severity)) not in covered]
        if not cands:
            break
        best = max(cands, key=lambda i: score(i, covered))
        chosen.append(best)
        covered |= {key(f) for f in sorted(by_win[best], key=lambda f: -f.severity)[:3]}
    fps = R.fps
    out = []
    for rank, i in enumerate(chosen, 1):
        items = sorted(by_win[i], key=lambda f: -f.severity)[:3]
        w = windows[i]
        m = (t_abs >= w["t_start"]) & (t_abs < w["t_end"])
        k = _peak_frame(items[0], m, series, R, land_r, t0, fps)
        joints = sorted({j for f in items for j in joints_of(f.kind, f.part)})
        # 一番大きい指摘と同じ内容のズレが出ている他の場所（前後に続いている区間は同じ場所とみなす）
        main = key(items[0])
        has = {j for j in by_win if any(key(f) == main for f in by_win[j])}
        same_place = {i}
        for step in (-1, 1):
            j = i + step
            while j in has:
                same_place.add(j)
                j += step
        repeats = [windows[j]["label"] or f"{windows[j]['t_start']:.1f}秒" for j in sorted(has - same_place)]
        out.append({
            "rank": rank,
            "t": float(t_abs[k]), "user_t": float(t_user_abs[k]),
            "label": grid.label(float(t_abs[k])) if grid is not None else None,
            "window_label": w["label"], "t_start": w["t_start"], "t_end": w["t_end"],
            "score": float(score(i, set())),
            "items": [asdict(f) for f in items],
            "joints": joints,
            "repeats": repeats,
        })
    return out
