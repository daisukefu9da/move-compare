"""人物の選択と、途切れた追跡 ID のつなぎ合わせ."""

from __future__ import annotations

import numpy as np

from ..core.skeleton import PoseSequence
from .base import TrackSet


def people_at(ts: TrackSet, frame: int, search: int = 30) -> tuple[int, dict[int, np.ndarray]]:
    """frame 付近で人が映っている最初のフレームと、その時の {id: box} を返す（選択 UI 用）."""
    for f in range(frame, min(ts.n_frames, frame + search)):
        boxes = {tid: ts.boxes[tid][f] for tid in ts.ids if np.isfinite(ts.boxes[tid][f, 0])}
        if boxes:
            return f, boxes
    return frame, {}


def pick_by_point(ts: TrackSet, frame: int, x: float, y: float) -> int:
    """クリック位置 (x, y) にいる人物の ID. 箱の中に無ければ箱の中心が最も近い人."""
    f, boxes = people_at(ts, frame)
    if not boxes:
        raise ValueError("指定したフレーム付近に人物が検出されていません")
    inside = {tid: (b[2] - b[0]) * (b[3] - b[1]) for tid, b in boxes.items()
              if b[0] <= x <= b[2] and b[1] <= y <= b[3]}
    if inside:
        return min(inside, key=inside.get)  # 重なっていたら小さい（手前の）箱
    return min(boxes, key=lambda t: np.hypot((boxes[t][0] + boxes[t][2]) / 2 - x, (boxes[t][1] + boxes[t][3]) / 2 - y))


def main_person(ts: TrackSet) -> int:
    """最も大きく長く映っている人物の ID（2 動画モードで踊っている本人を自動で選ぶ）."""
    if not ts.ids:
        raise ValueError("人物が検出されませんでした")

    def weight(tid):
        b = ts.boxes[tid]
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return float(np.nansum(area))
    return max(ts.ids, key=weight)


def stitch(ts: TrackSet, seed: int, exclude: set[int] = frozenset(), max_gap_s: float = 1.5,
           max_dist: float = 1.0) -> list[int]:
    """追跡 ID が途中で入れ替わった場合に、同じ人物とみられる後続の ID をつなぐ.

    seed の追跡が途切れたら、その後 max_gap_s 以内に現れ、最後の位置から箱の高さ×max_dist 以内に
    出てきた ID を同一人物とみなす。exclude（もう一方の人物など）は使わない。
    同様に、seed より前に途切れた ID も遡ってつなぐ。
    """
    chain = [seed]
    used = set(exclude) | {seed}

    def span(tid):
        p = np.where(ts.present(tid))[0]
        return (int(p[0]), int(p[-1])) if p.size else (None, None)

    def center(tid, f):
        b = ts.boxes[tid][f]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b[3] - b[1]

    gap = int(max_gap_s * ts.fps)
    # 後ろ向きにつなぐ
    while True:
        s, e = span(chain[-1])
        if e is None or e >= ts.n_frames - 1:
            break
        c, h = center(chain[-1], e)
        best, best_d = None, None
        for tid in ts.ids:
            if tid in used:
                continue
            s2, _ = span(tid)
            if s2 is None or not (e < s2 <= e + gap):
                continue
            c2, _ = center(tid, s2)
            d = np.linalg.norm(c2 - c) / max(h, 1)
            if d <= max_dist and (best_d is None or d < best_d):
                best, best_d = tid, d
        if best is None:
            break
        chain.append(best)
        used.add(best)
    # 前向き（seed より前）につなぐ
    while True:
        s, _ = span(chain[0])
        if s is None or s == 0:
            break
        c, h = center(chain[0], s)
        best, best_d = None, None
        for tid in ts.ids:
            if tid in used:
                continue
            _, e2 = span(tid)
            if e2 is None or not (s - gap <= e2 < s):
                continue
            c2, _ = center(tid, e2)
            d = np.linalg.norm(c2 - c) / max(h, 1)
            if d <= max_dist and (best_d is None or d < best_d):
                best, best_d = tid, d
        if best is None:
            break
        chain.insert(0, best)
        used.add(best)
    return chain


def merge_tracks(ts: TrackSet, ids: list[int]) -> PoseSequence:
    """複数の ID を 1 人分の系列にまとめる（同じフレームに複数あれば信頼度の高い方）."""
    kp = np.full_like(ts.keypoints[ids[0]], np.nan)
    sc = np.zeros_like(ts.scores[ids[0]])
    best = np.full(ts.n_frames, -1.0)
    for tid in ids:
        m = ts.present(tid)
        q = ts.scores[tid].mean(axis=1)
        take = m & (q > best)
        kp[take] = ts.keypoints[tid][take]
        sc[take] = ts.scores[tid][take]
        best[take] = q[take]
    return PoseSequence(kp, sc, ts.fps, ts.width, ts.height, meta={"track_ids": ids})


def select_person(ts: TrackSet, track_id: int | None = None, point: tuple[float, float] | None = None,
                  frame: int = 0, exclude: set[int] = frozenset()) -> PoseSequence:
    """ID / クリック位置 / 自動（最も大きく映っている人）のいずれかで 1 人を選び、追跡をつなげて返す."""
    if track_id is None:
        track_id = pick_by_point(ts, frame, *point) if point is not None else main_person(ts)
    if track_id not in ts.keypoints:
        raise ValueError(f"ID {track_id} の人物はいません（候補: {ts.ids}）")
    return merge_tracks(ts, stitch(ts, track_id, exclude=set(exclude)))
