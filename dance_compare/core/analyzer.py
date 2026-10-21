"""手本とレビュー対象の骨格系列を比較し、Report を作る（解析の本体）."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .counts import CountGrid
from .dtw import dtw, monotonic, pairwise_cost, path_to_warp
from .features import Features, compute_features
from .preprocess import clean, gaussian_smooth, mirror, normalize, resample
from .report import Finding, Report
from .rhythm import (beat_grid, foot_landings, match_events, movement_accents,
                     offset_stats, offsets_to_grid)
from .skeleton import ANGLE_NAMES, END_EFFECTORS, PoseSequence

END_NAMES = list(END_EFFECTORS)


def _prepare(seq: PoseSequence, cfg: AnalysisConfig) -> tuple[PoseSequence, Features]:
    seq = resample(clean(seq, cfg.conf_threshold, cfg.smooth_sigma_s), cfg.analysis_fps)
    body, world, _ = normalize(seq.keypoints)
    return seq, compute_features(body, world, seq.fps)


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


def _exp_score(x: float, scale: float) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return float(100 * np.exp(-abs(x) / scale))


def _mean_or_none(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


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
    level_diff = _sample(U.level, warp) - R.level
    facing_diff = _sample(U.facing, warp) - R.facing
    lean_diff = _sample(U.lean, warp) - R.lean
    stance_diff = _sample(U.stance, warp) - R.stance

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

    findings = _window_findings(windows, cfg, grid, rhythm.get("bias_ms", 0.0))
    tendencies = _tendencies(windows, rhythm, timing_dev, synced, cfg)

    # --- スコア ---
    def logs(key):
        vals = [np.log(v) for w in windows for v in w[key].values() if np.isfinite(v) and v > 0]
        return float(np.mean(np.abs(vals))) if vals else float("nan")

    scores = {
        "shape": _exp_score(_nanmean(np.abs(angle_diff)), cfg.score_scale_angle_deg),
        "timing": _exp_score(_nanmean(np.abs(timing_dev)), cfg.score_scale_timing_ms),
        "amplitude": _exp_score(logs("amplitude_ratio"), cfg.score_scale_amplitude),
        "sharpness": _exp_score(logs("sharpness_ratio"), cfg.score_scale_sharpness),
    }
    lv = [_exp_score(_nanmean(np.abs(level_diff)), cfg.score_scale_level),
          _exp_score(_nanmean(np.abs(facing_diff)), cfg.facing_threshold_deg * 2),
          _exp_score(_nanmean(np.abs(lean_diff)), cfg.lean_threshold_deg * 2)]
    scores["level"] = _mean_or_none(lv)
    st = [_exp_score(_nanmean(np.abs(stance_diff)), cfg.stance_threshold * 2),
          _exp_score(_nanmean([w["travel_diff"] for w in windows]), cfg.travel_threshold * 2)]
    lvr = rhythm["landing_vs_ref"]
    if lvr.get("count", 0) >= 3:
        st.append(_exp_score(abs(lvr["mean_ms"]) + lvr["std_ms"] * 0.5, cfg.score_scale_timing_ms))
    scores["steps"] = _mean_or_none(st)
    if "user" in rhythm and rhythm["user"].get("count", 0) >= 3:
        u = rhythm["user"]
        scores["rhythm"] = _exp_score(abs(u["mean_ms"] - rhythm["bias_ms"]) + u["std_ms"] * 0.5,
                                      cfg.score_scale_rhythm_ms)
    else:
        scores["rhythm"] = None
    valid = [v for v in scores.values() if v is not None]
    scores["overall"] = float(np.mean(valid)) if valid else None

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
    )


# ---------------------------------------------------------------------------
# 指摘の抽出
# ---------------------------------------------------------------------------

def _window_findings(windows: list[dict], cfg: AnalysisConfig, grid: CountGrid | None,
                     bias: float = 0.0) -> list[Finding]:
    """区間ごとの指摘. bias は拍に対する手本のズレ [ms]（拍検出の癖や曲のノリの分を差し引く基準）."""
    raw: list[Finding] = []

    def add(w, aspect, kind, direction, value, unit, sev, part=None, **extra):
        raw.append(Finding(aspect=aspect, kind=kind, direction=direction, value=float(value), unit=unit,
                           severity=float(sev), part=part, t_start=w["t_start"], t_end=w["t_end"],
                           label=w["label"], extra=extra))

    for w in windows:
        for name, d in w["angle_diff"].items():
            if abs(d) >= cfg.angle_threshold_deg:
                # 角度が小さい = より曲がっている（肩・股関節は腕・脚が体に近い）
                add(w, "shape", "angle", "smaller" if d < 0 else "larger", d, "deg",
                    abs(d) / cfg.angle_threshold_deg, part=name)
        if (abs(w["timing_ms"]) >= cfg.timing_threshold_ms and w["ref_motion"] >= cfg.min_motion_for_timing
                and not w["edge"]):
            add(w, "timing", "early_late", "late" if w["timing_ms"] > 0 else "early", w["timing_ms"], "ms",
                abs(w["timing_ms"]) / cfg.timing_threshold_ms)
        r = w["rhythm_ms"]
        if r is not None and abs(r - bias) >= cfg.rhythm_threshold_ms * 1.5:
            add(w, "rhythm", "beat_offset", "late" if r > bias else "early", r, "ms",
                abs(r - bias) / (cfg.rhythm_threshold_ms * 1.5))
        for name, r in w["amplitude_ratio"].items():
            if not np.isfinite(r) or w["partial"]:
                continue
            if r < cfg.amplitude_ratio_low:
                add(w, "amplitude", "amplitude", "smaller", r, "ratio",
                    np.log(r) / np.log(cfg.amplitude_ratio_low), part=name)
            elif r > cfg.amplitude_ratio_high:
                add(w, "amplitude", "amplitude", "larger", r, "ratio",
                    np.log(r) / np.log(cfg.amplitude_ratio_high), part=name)
        for name, r in w["sharpness_ratio"].items():
            if np.isfinite(r) and r < cfg.sharpness_ratio_low and not w["partial"]:
                add(w, "sharpness", "sharpness", "weaker", r, "ratio",
                    np.log(r) / np.log(cfg.sharpness_ratio_low), part=name)
        lm = w.get("landing_ms")
        if lm is not None and abs(lm) >= cfg.landing_threshold_ms:
            feet = w.get("landing_feet") or []
            part = feet[0] if len(feet) == 1 else None
            add(w, "steps", "landing", "late" if lm > 0 else "early", lm, "ms",
                abs(lm) / cfg.landing_threshold_ms, part=part)
        if abs(w["level_diff"]) >= cfg.level_threshold:
            add(w, "level", "level", "higher" if w["level_diff"] > 0 else "lower", w["level_diff"], "torso",
                abs(w["level_diff"]) / cfg.level_threshold)
        if abs(w["facing_diff"]) >= cfg.facing_threshold_deg:
            add(w, "level", "facing", "more_turned" if w["facing_diff"] > 0 else "less_turned",
                w["facing_diff"], "deg", abs(w["facing_diff"]) / cfg.facing_threshold_deg)
        if abs(w["lean_diff"]) >= cfg.lean_threshold_deg:
            add(w, "level", "lean", "right" if w["lean_diff"] > 0 else "left", w["lean_diff"], "deg",
                abs(w["lean_diff"]) / cfg.lean_threshold_deg)
        if abs(w["stance_diff"]) >= cfg.stance_threshold:
            add(w, "steps", "stance", "wider" if w["stance_diff"] > 0 else "narrower", w["stance_diff"], "torso",
                abs(w["stance_diff"]) / cfg.stance_threshold)
        if w["travel_diff"] >= cfg.travel_threshold:
            r, u = np.array(w["travel_ref"]), np.array(w["travel_user"])
            direction = "less" if np.linalg.norm(u) < np.linalg.norm(r) else "more"
            add(w, "steps", "travel", direction, w["travel_diff"], "torso",
                w["travel_diff"] / cfg.travel_threshold, ref=r.tolist(), user=u.tolist())

    return _merge(raw, grid)


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
    u = rhythm.get("user")
    bias = rhythm.get("bias_ms", 0.0)
    if u and u.get("count", 0) >= 3 and abs(u["mean_ms"] - bias) >= cfg.rhythm_threshold_ms:
        ref = rhythm.get("ref", {})
        ul = rhythm.get("user_landing", {})
        out.append(Finding("rhythm", "beat_offset", "late" if u["mean_ms"] > bias else "early", u["mean_ms"], "ms",
                           abs(u["mean_ms"] - bias) / cfg.rhythm_threshold_ms * 1.5,
                           extra={"ref_mean_ms": ref.get("mean_ms"),
                                  "landing_mean_ms": ul.get("mean_ms") if ul.get("count", 0) >= 3 else None,
                                  "on_beat_ratio": u["on_beat_ratio"]}))
    lv = rhythm.get("landing_vs_ref", {})
    if lv.get("count", 0) >= 3 and abs(lv["mean_ms"]) >= cfg.landing_threshold_ms:
        out.append(Finding("steps", "landing", "late" if lv["mean_ms"] > 0 else "early", lv["mean_ms"], "ms",
                           abs(lv["mean_ms"]) / cfg.landing_threshold_ms))
    if synced:
        mean_t = float(np.median(timing_dev))
        if abs(mean_t) >= cfg.timing_threshold_ms * 0.6:
            out.append(Finding("timing", "early_late", "late" if mean_t > 0 else "early", mean_t, "ms",
                               abs(mean_t) / cfg.timing_threshold_ms * 1.5))
    for name in END_NAMES:
        vals = [w["amplitude_ratio"][name] for w in windows if np.isfinite(w["amplitude_ratio"][name])]
        if len(vals) >= 3:
            r = float(np.exp(np.mean(np.log(vals))))
            if r < cfg.amplitude_ratio_low * 1.1:
                out.append(Finding("amplitude", "amplitude", "smaller", r, "ratio",
                                   np.log(r) / np.log(cfg.amplitude_ratio_low) * 1.5, part=name))
            elif r > cfg.amplitude_ratio_high / 1.1:
                out.append(Finding("amplitude", "amplitude", "larger", r, "ratio",
                                   np.log(r) / np.log(cfg.amplitude_ratio_high) * 1.5, part=name))
        vals = [w["sharpness_ratio"][name] for w in windows if np.isfinite(w["sharpness_ratio"][name])]
        if len(vals) >= 3:
            r = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-6)))))
            if r < cfg.sharpness_ratio_low * 1.1:
                out.append(Finding("sharpness", "sharpness", "weaker", r, "ratio",
                                   np.log(r) / np.log(cfg.sharpness_ratio_low) * 1.5, part=name))
    lvl = float(np.mean([w["level_diff"] for w in windows])) if windows else 0.0
    if abs(lvl) >= cfg.level_threshold * 0.8:
        out.append(Finding("level", "level", "higher" if lvl > 0 else "lower", lvl, "torso",
                           abs(lvl) / cfg.level_threshold * 1.5))
    out.sort(key=lambda f: -f.severity)
    return out
