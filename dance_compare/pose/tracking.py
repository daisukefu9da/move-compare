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


def stitch(ts: TrackSet, seed: int, exclude: set[int] = frozenset(), max_gap_s: float = 3.0,
           max_overlap_s: float = 0.5, max_dist: float = 1.0, dist_per_s: float = 0.5,
           rival: list[int] | None = None) -> list[int]:
    """追跡 ID が途中で入れ替わった場合に、同じ人物とみられる前後の ID をつなぐ.

    seed の追跡が途切れたら、その前後（max_overlap_s 秒前〜max_gap_s 秒後）に現れた ID のうち、
    最後の位置から「箱の高さ ×（max_dist + dist_per_s × 経過秒）」以内に出てきた最も近いものを同一人物とみなす。
    （追跡器は、見失った人物に新しい ID を振ることがあり、古い ID が消える少し前に新しい ID が出ることも多い）
    exclude（もう一方の人物など）は使わない。seed より前に途切れた ID も同様に遡ってつなぐ。
    rival（もう一方の人物の ID の列）を渡すと、候補がその人物の直前の位置により近い場合は、
    もう一方の人物の ID とみなしてつながない（二人の追跡が同時に途切れたときの取り違え防止）。
    """
    chain = [seed]
    used = set(exclude) | {seed}
    gap, overlap = int(max_gap_s * ts.fps), int(max_overlap_s * ts.fps)
    rival_centers = None
    if rival:
        rb = np.full((ts.n_frames, 4), np.nan)
        for tid in rival:
            m = ts.present(tid)
            rb[m] = ts.boxes[tid][m]
        rival_centers = np.stack([(rb[:, 0] + rb[:, 2]) / 2, (rb[:, 1] + rb[:, 3]) / 2], axis=1)

    def rival_near(f):
        """もう一方の人物の、フレーム f に最も近い時点の位置（gap 以内）."""
        if rival_centers is None:
            return None
        lo, hi = max(0, f - gap), min(ts.n_frames, f + gap + 1)
        idx = np.where(np.isfinite(rival_centers[lo:hi, 0]))[0]
        if idx.size == 0:
            return None
        return rival_centers[lo + idx[np.argmin(np.abs(lo + idx - f))]]

    def span(tid):
        p = np.where(ts.present(tid))[0]
        return (int(p[0]), int(p[-1])) if p.size else (None, None)

    def center(tid, f):
        b = ts.boxes[tid][f]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b[3] - b[1]

    def best_link(f_end, c, h, forward):
        best, best_d = None, None
        for tid in ts.ids:
            if tid in used:
                continue
            s2, e2 = span(tid)
            if s2 is None:
                continue
            if forward:  # 途切れた後に始まる（少し前から始まっていても可）ID
                f2, dt = s2, s2 - f_end
                if not (-overlap <= dt <= gap) or e2 <= f_end:
                    continue
            else:  # 始まる前に終わる（少し後まで続いていても可）ID
                f2, dt = e2, f_end - e2
                if not (-overlap <= dt <= gap) or s2 >= f_end:
                    continue
            c2, _ = center(tid, f2)
            d = np.linalg.norm(c2 - c) / max(h, 1)
            rc = rival_near(f2)
            if rc is not None and np.linalg.norm(c2 - rc) / max(h, 1) < d:
                continue  # もう一方の人物の方が近い
            if d <= max_dist + dist_per_s * max(dt, 0) / ts.fps and (best_d is None or d < best_d):
                best, best_d = tid, d
        return best

    while True:  # 後ろへつなぐ
        _, e = span(chain[-1])
        if e is None or e >= ts.n_frames - 1:
            break
        nxt = best_link(e, *center(chain[-1], e), forward=True)
        if nxt is None:
            break
        chain.append(nxt)
        used.add(nxt)
    while True:  # 前へつなぐ
        s, _ = span(chain[0])
        if s is None or s == 0:
            break
        prv = best_link(s, *center(chain[0], s), forward=False)
        if prv is None:
            break
        chain.insert(0, prv)
        used.add(prv)
    return chain


def resolve_id(ts: TrackSet, track_id: int | None = None, point: tuple[float, float] | None = None,
               frame: int = 0) -> int:
    """ID / クリック位置 / 自動（最も大きく映っている人）から追跡 ID を決める.

    frame はクリック位置を指定した、元の動画のフレーム番号。
    """
    if track_id is None:
        track_id = pick_by_point(ts, ts.track_frame(frame), *point) if point is not None else main_person(ts)
    if track_id not in ts.keypoints:
        raise ValueError(f"ID {track_id} の人物はいません（候補: {ts.ids}）")
    return track_id


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
    return merge_tracks(ts, stitch(ts, resolve_id(ts, track_id, point, frame), exclude=set(exclude)))


def select_pair(ts: TrackSet, ref: tuple, user: tuple) -> tuple[PoseSequence, PoseSequence]:
    """1 本の動画から二人を選ぶ. ref / user は (track_id, point, frame).

    お互いの ID を取り込まないよう、先に両方の ID を決めてから追跡をつなぐ。
    """
    rid, uid = resolve_id(ts, *ref), resolve_id(ts, *user)
    if rid == uid:
        raise ValueError("お手本とレビュー対象に同じ人物が選ばれています")
    # まず互いの最初の ID だけを除いて仮につなぎ、それを相手の位置として使って本番のつなぎ直しをする
    u_tmp = stitch(ts, uid, exclude={rid}, rival=[rid])
    r_chain = stitch(ts, rid, exclude=set(u_tmp), rival=u_tmp)
    u_chain = stitch(ts, uid, exclude=set(r_chain), rival=r_chain)
    return merge_tracks(ts, r_chain), merge_tracks(ts, u_chain)
