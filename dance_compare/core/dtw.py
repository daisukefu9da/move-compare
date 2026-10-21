"""DTW (Dynamic Time Warping) による時間合わせ."""

from __future__ import annotations

import numpy as np


def pairwise_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D), (M, D) -> (N, M) ユークリッド距離."""
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2 * a @ b.T, 0))


def dtw(cost: np.ndarray, band: int | None = None, step_penalty: float = 0.0,
        open_ends: bool = False) -> np.ndarray:
    """コスト行列から最小コストの対応経路を求める.

    cost: (N, M)。行が手本、列がレビュー対象。
    band: 対角線（傾き M/N）から列方向に何フレームまでずれてよいか。
    step_penalty: 縦・横ステップ（片方だけ進む）に加えるコスト。止まりすぎを防ぐ。
    open_ends: True なら、レビュー対象側の始点・終点を自由にする（前後の余分を許容）。
    returns: (L, 2) の経路 [(i, j), ...]
    """
    N, M = cost.shape
    C = cost.astype(np.float64).copy()
    if band is not None and not open_ends:
        i = np.arange(N)[:, None]
        j = np.arange(M)[None, :]
        center = i * (M - 1) / max(N - 1, 1)
        C[np.abs(j - center) > band] = np.inf

    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    if open_ends:
        D[0, :] = 0.0
    # 反対角線ごとにまとめて計算する（Python ループを N+M 回に抑える）
    for d in range(2, N + M + 1):
        i_lo, i_hi = max(1, d - M), min(N, d - 1)
        if i_lo > i_hi:
            continue
        ii = np.arange(i_lo, i_hi + 1)
        jj = d - ii
        diag = D[ii - 1, jj - 1]
        up = D[ii - 1, jj] + step_penalty
        left = D[ii, jj - 1] + step_penalty
        D[ii, jj] = C[ii - 1, jj - 1] + np.minimum(diag, np.minimum(up, left))

    i, j = N, (int(np.argmin(D[N, 1:])) + 1 if open_ends else M)
    if not np.isfinite(D[i, j]):
        raise ValueError("DTW の経路が見つかりません（band が狭すぎる可能性があります）")
    path = [(i - 1, j - 1)]
    while i > 1 or (j > 1 and not open_ends):
        if i == 1:
            if open_ends:
                break
            j -= 1
        elif j == 1:
            i -= 1
        else:
            cands = (D[i - 1, j - 1], D[i - 1, j] + step_penalty, D[i, j - 1] + step_penalty)
            k = int(np.argmin(cands))
            if k == 0:
                i, j = i - 1, j - 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))
    return np.array(path[::-1])


def path_to_warp(path: np.ndarray, n: int) -> np.ndarray:
    """経路から、手本の各フレーム i に対応するレビュー対象のフレーム位置（実数）を返す."""
    warp = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(warp, path[:, 0], path[:, 1])
    np.add.at(counts, path[:, 0], 1)
    filled = counts > 0
    warp[filled] /= counts[filled]
    idx = np.arange(n)
    return np.interp(idx, idx[filled], warp[filled])


def monotonic(x: np.ndarray) -> np.ndarray:
    """単調非減少にする（平滑化で生じる微小な逆行を除く）."""
    return np.maximum.accumulate(x)
