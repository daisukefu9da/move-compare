"""リズム: 動きのアクセント検出と、拍に対するズレの計算."""

from __future__ import annotations

import numpy as np

from .skeleton import KP


def find_peaks(x: np.ndarray, min_distance: int, prominence: float) -> np.ndarray:
    """単純なピーク検出. 周囲 min_distance 内で最大かつ、両側の谷から prominence 以上高い点."""
    n = x.size
    if n < 3:
        return np.array([], dtype=int)
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    keep = []
    for i in cand:
        lo, hi = max(0, i - min_distance), min(n, i + min_distance + 1)
        if x[i] < np.max(x[lo:hi]):
            continue
        # 両側の、より高い点に当たるまで（なければ端まで）の最小値を谷とする
        left = x[:i][::-1]
        higher = np.where(left > x[i])[0]
        lmin = np.min(left[: higher[0]]) if higher.size else (np.min(left) if left.size else x[i])
        right = x[i + 1:]
        higher = np.where(right > x[i])[0]
        rmin = np.min(right[: higher[0]]) if higher.size else (np.min(right) if right.size else x[i])
        if x[i] - max(lmin, rmin) >= prominence:
            if keep and i - keep[-1] < min_distance:
                continue
            keep.append(i)
    return np.array(keep, dtype=int)


def movement_accents(body_speed: np.ndarray, fps: float, mode: str = "peak",
                     min_interval_s: float = 0.18, prominence: float = 0.6) -> np.ndarray:
    """動きのアクセントの時刻 [秒].

    peak: 全身の速さのピーク（大きく動いた瞬間）
    hit:  減速のピーク（ヒット・ストップで急に止まる瞬間）
    """
    if mode == "hit":
        signal = np.maximum(-np.gradient(body_speed) * fps, 0)
    elif mode == "peak":
        signal = body_speed
    else:
        raise ValueError(f"unknown accent mode: {mode}")
    std = np.std(signal)
    if std <= 1e-9:
        return np.array([])
    idx = find_peaks(signal, max(1, int(min_interval_s * fps)), prominence * std)
    # 放物線補間でフレーム間の位置を推定
    t = idx.astype(np.float64)
    ok = (idx > 0) & (idx < signal.size - 1)
    i = idx[ok]
    y0, y1, y2 = signal[i - 1], signal[i], signal[i + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    t[ok] += np.clip(delta, -0.5, 0.5)
    return t / fps


def beat_grid(beats: np.ndarray, subdivision: int) -> np.ndarray:
    """拍を subdivision 分割したグリッド（2 なら表拍と裏拍）."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if subdivision <= 1 or beats.size < 2:
        return beats
    pts = [beats[:-1] + (beats[1:] - beats[:-1]) * k / subdivision for k in range(subdivision)]
    return np.sort(np.concatenate(pts + [beats[-1:]]))


def offsets_to_grid(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """各時刻から最も近いグリッド点までの符号付きズレ [ms]（正=遅れ）. グリッド外は除く."""
    times = np.asarray(times, dtype=np.float64)
    if grid.size < 2 or times.size == 0:
        return np.array([])
    half = np.median(np.diff(grid)) / 2
    times = times[(times >= grid[0] - half) & (times <= grid[-1] + half)]
    idx = np.clip(np.searchsorted(grid, times), 1, grid.size - 1)
    left, right = grid[idx - 1], grid[idx]
    nearest = np.where(times - left < right - times, left, right)
    return (times - nearest) * 1000


def offset_stats(offsets_ms: np.ndarray, tolerance_ms: float) -> dict:
    if offsets_ms.size == 0:
        return {"count": 0}
    return {
        "count": int(offsets_ms.size),
        "mean_ms": float(np.mean(offsets_ms)),
        "median_ms": float(np.median(offsets_ms)),
        "std_ms": float(np.std(offsets_ms)),
        "on_beat_ratio": float(np.mean(np.abs(offsets_ms) <= tolerance_ms)),
    }


def foot_landings(world: np.ndarray, fps: float, speed_on: float, speed_off: float) -> dict[str, np.ndarray]:
    """足の着地時刻 [秒]. 足首が下向きに speed_on 以上で動いた後、速さが speed_off 未満に落ちた瞬間."""
    out = {}
    for name in ("left_ankle", "right_ankle"):
        y = world[:, KP[name], 1]
        vy = np.gradient(y) * fps
        speed = np.abs(vy)
        times = []
        falling = False
        for i in range(y.size):
            if vy[i] < -speed_on:
                falling = True
            elif falling and speed[i] < speed_off:
                # 速さが speed_off を下回った瞬間をフレーム間で線形補間
                prev = speed[i - 1] if i > 0 else speed[i]
                frac = (prev - speed_off) / (prev - speed[i]) if prev > speed[i] else 1.0
                times.append((i - 1 + float(np.clip(frac, 0, 1))) / fps)
                falling = False
        out[name] = np.array(times)
    return out


def match_events(ref_t: np.ndarray, user_t: np.ndarray, expected: np.ndarray, window_s: float) -> np.ndarray:
    """手本のイベントごとに、予想時刻 expected 付近のレビュー対象のイベントとの差 [ms]. 見つからなければ NaN."""
    out = np.full(ref_t.size, np.nan)
    if user_t.size == 0:
        return out
    for k, e in enumerate(expected):
        d = user_t - e
        i = int(np.argmin(np.abs(d)))
        if abs(d[i]) <= window_s:
            out[k] = d[i] * 1000
    return out
