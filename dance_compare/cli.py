"""コマンドライン.

  python -m dance_compare compare ref.mp4 me.mp4 --out out/
  python -m dance_compare people group.mp4 --out people.jpg      # 人物 ID の確認
  python -m dance_compare single group.mp4 --ref-id 1 --user-id 3
  python -m dance_compare poses ref_pose.json user_pose.json     # 推定済みの骨格どうしを比較
  python -m dance_compare demo --out demo/                         # 合成データで動作確認
  python -m dance_compare validate validation/cases.toml           # 期待される検出結果との照合
"""

from __future__ import annotations

import argparse
import sys

from .core.config import AnalysisConfig
from .feedback import get_narrator
from .pipeline import (Pick, Rhythm, analyze_single_video, analyze_two_videos,
                       extract_tracks, save_outputs)


def _point(s: str | None):
    if not s:
        return None
    x, y = s.split(",")
    return float(x), float(y)


def _progress(stage: str, i: int, n: int):
    if n and (i % 30 == 0 or i == n):
        sys.stderr.write(f"\r{stage}: {i}/{n}  ")
        sys.stderr.flush()
        if i == n:
            sys.stderr.write("\n")


def _common(p: argparse.ArgumentParser):
    p.add_argument("--out", default="out", help="出力先フォルダ")
    p.add_argument("--config", help="解析パラメータの TOML ファイル")
    p.add_argument("--backend", default="yolo", choices=["yolo", "mediapipe"])
    p.add_argument("--mirror", action="store_true", help="手本を左右反転して比較（向かい合わせの手本）")
    p.add_argument("--bpm", type=float, help="BPM を手動で指定（音声から検出しない）")
    p.add_argument("--first-one", type=float, help="手本動画で「1エイト目の1」に当たる時刻 [秒]")
    p.add_argument("--no-beats", action="store_true", help="拍を使わない（秒で表示）")
    p.add_argument("--narrator", default="rule", choices=["rule", "gemini"], help="指摘文の作り方")
    p.add_argument("--no-video", action="store_true", help="比較動画を作らない")
    p.add_argument("--no-cache", action="store_true", help="骨格推定のキャッシュを使わない")


def _config(args) -> AnalysisConfig:
    cfg = AnalysisConfig.load(args.config) if args.config else AnalysisConfig()
    if getattr(args, "open_ends", False):
        cfg = cfg.replace(dtw_open_ends=True)
    return cfg


def _rhythm(args) -> Rhythm:
    return Rhythm(bpm=args.bpm, first_one=args.first_one, detect=not args.no_beats)


def _finish(result, args):
    narrator = get_narrator(args.narrator)
    paths = save_outputs(result, args.out, narrator, render=not args.no_video,
                         progress=lambda i, n: _progress("比較動画", i, n))
    text = paths["text"]
    if getattr(narrator, "last_error", None):
        print(f"[注意] {narrator.last_error}", file=sys.stderr)
    for n in result.notes:
        print(f"[メモ] {n}", file=sys.stderr)
    print(text)
    print("\n出力:", file=sys.stderr)
    for k, v in paths.items():
        print(f"  {k}: {v}", file=sys.stderr)


def cmd_compare(args):
    result = analyze_two_videos(
        args.ref, args.user, _config(args), backend=args.backend, mirror=args.mirror,
        ref_range=(args.ref_start, args.ref_end), user_range=(args.user_start, args.user_end),
        audio_sync=not args.no_audio_sync, rhythm=_rhythm(args),
        ref_pick=Pick(args.ref_id, _point(args.ref_point), args.frame),
        user_pick=Pick(args.user_id, _point(args.user_point), args.frame),
        use_cache=not args.no_cache, progress=_progress,
    )
    _finish(result, args)


def cmd_single(args):
    if args.ref_id is None and not args.ref_point or args.user_id is None and not args.user_point:
        sys.exit("お手本とレビュー対象を --ref-id/--ref-point と --user-id/--user-point で指定してください"
                 "（ID は `people` コマンドで確認できます）")
    result = analyze_single_video(
        args.video, Pick(args.ref_id, _point(args.ref_point), args.frame),
        Pick(args.user_id, _point(args.user_point), args.frame), _config(args),
        backend=args.backend, mirror=args.mirror, time_range=(args.start, args.end),
        auto_range=not args.no_auto_range,
        rhythm=_rhythm(args), use_cache=not args.no_cache, progress=_progress,
    )
    _finish(result, args)


def cmd_people(args):
    import cv2

    from .media.video import read_frame
    from .pose.tracking import people_at

    ts = extract_tracks(args.video, args.backend, not args.no_cache, lambda i, n: _progress("骨格推定", i, n))
    f, boxes = people_at(ts, ts.track_frame(args.frame))
    img = read_frame(args.video, ts.video_frame(f))
    for tid, b in boxes.items():
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f"ID {tid}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.imwrite(args.out, img)
    print(f"{f / ts.fps:.1f} 秒（元の動画のフレーム {ts.video_frame(f)}）の人物: " + ", ".join(f"ID {t} (x={int((b[0]+b[2])/2)}, y={int((b[1]+b[3])/2)})"
                                             for t, b in boxes.items()))
    print(f"画像: {args.out}")


def cmd_poses(args):
    from .core import compare
    from .core.skeleton import PoseSequence

    ref = PoseSequence.load_json(args.ref)
    user = PoseSequence.load_json(args.user)
    report = compare(ref, user, _config(args), mirror_ref=args.mirror)
    print(get_narrator(args.narrator).narrate(report))


def cmd_demo(args):
    import os

    from .pose.base import TrackSet
    from .synth import synth_beats, synth_dancer, write_synth_video

    os.makedirs(args.out, exist_ok=True)
    dur = 16.0
    beats = synth_beats(dur + 1.5)
    ref = synth_dancer(duration=dur)
    # レビュー対象: 1.5 秒遅れて撮影開始、拍より 90ms 遅れ、振れ幅 75%、2エイト目の前半で右肘が曲がる、体格・位置も違う
    user = synth_dancer(duration=dur + 1.5, delay_s=1.5 + 0.09, amplitude=0.75, elbow_bend=(6.0, 8.0, 45),
                        size=100, center=(300, 250), noise=1.0, seed=1)
    ref_v, user_v = os.path.join(args.out, "demo_ref.mp4"), os.path.join(args.out, "demo_user.mp4")
    write_synth_video(ref, ref_v, beats)
    write_synth_video(user, user_v, beats + 1.5)
    args.out = os.path.join(args.out, "result")
    result = analyze_two_videos(ref_v, user_v, AnalysisConfig(), ref_tracks=TrackSet.from_sequence(ref),
                                user_tracks=TrackSet.from_sequence(user), use_cache=False)
    args.narrator = "rule"
    _finish(result, args)


def cmd_validate(args):
    from .validation import validate

    results = validate(args.cases, args.out, render=args.video, use_cache=not args.no_cache, only=args.case)
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} ケースが合格. 結果: {args.out}/validation.md")
    sys.exit(0 if passed == len(results) else 1)


def cmd_make_validation_set(args):
    from .validation import make_synthetic_set

    path = make_synthetic_set(args.out)
    print(f"作成しました: {path}\n実行: python -m dance_compare validate {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dance_compare", description="お手本とのダンス比較")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compare", help="2動画モード")
    p.add_argument("ref", help="お手本の動画")
    p.add_argument("user", help="レビュー対象の動画")
    _common(p)
    for who in ("ref", "user"):
        p.add_argument(f"--{who}-start", type=float)
        p.add_argument(f"--{who}-end", type=float)
        p.add_argument(f"--{who}-id", type=int, help="人物 ID（複数人が映っている場合）")
        p.add_argument(f"--{who}-point", help="人物の位置 x,y（複数人が映っている場合）")
    p.add_argument("--frame", type=int, default=0, help="--*-point を指定するフレーム")
    p.add_argument("--no-audio-sync", action="store_true", help="音声で 2 本の動画を同期しない")
    p.add_argument("--open-ends", action="store_true", help="レビュー対象の前後に余分な部分がある")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("single", help="1動画モード（1 本の動画の中の二人を比較）")
    p.add_argument("video")
    _common(p)
    p.add_argument("--ref-id", type=int)
    p.add_argument("--user-id", type=int)
    p.add_argument("--ref-point")
    p.add_argument("--user-point")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--start", type=float, help="比較を始める時刻 [秒]（省略すると自動で推定）")
    p.add_argument("--end", type=float, help="比較を終える時刻 [秒]（省略すると自動で推定）")
    p.add_argument("--no-auto-range", action="store_true",
                   help="二人の動きがそろっている区間の自動推定をしない（動画全体を比較）")
    p.set_defaults(func=cmd_single)

    p = sub.add_parser("people", help="動画に映っている人物と ID を画像で確認")
    p.add_argument("video")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="people.jpg")
    p.add_argument("--backend", default="yolo", choices=["yolo"])
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_people)

    p = sub.add_parser("poses", help="骨格 JSON どうしを比較（動画なし）")
    p.add_argument("ref")
    p.add_argument("user")
    p.add_argument("--config")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--narrator", default="rule", choices=["rule", "gemini"])
    p.set_defaults(func=cmd_poses)

    p = sub.add_parser("demo", help="合成した棒人間の動画で一連の処理を試す")
    p.add_argument("--out", default="demo")
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("validate", help="テスト動画と期待される検出結果を照合する（精度検証）")
    p.add_argument("cases", help="ケースファイル (TOML)")
    p.add_argument("--out", default="validation_out")
    p.add_argument("--case", action="append", help="このケースだけ実行（複数指定可）")
    p.add_argument("--video", action="store_true", help="各ケースの比較動画も作る")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("make-validation-set", help="合成データで検証セットの見本を作る")
    p.add_argument("--out", default="validation/synth")
    p.set_defaults(func=cmd_make_validation_set)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
