"""ローカル用の簡易 Web UI.

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import streamlit as st

from dance_compare.core.config import AnalysisConfig
from dance_compare.core.counts import fmt_time
from dance_compare.core.parts import GROUPS
from dance_compare.core.skeleton import JA_NAMES
from dance_compare.feedback import get_narrator
from dance_compare.feedback.rule_based import moment_lines, moment_title, tendency_line
from dance_compare.feedback.table import findings_rows
from dance_compare.pipeline import (Pick, Rhythm, analyze_single_video, analyze_two_videos,
                                    extract_tracks, save_outputs, suggest_range)
from dance_compare.ui.trimmer import range_picker, time_picker

UPLOAD_DIR = ".cache/uploads"
OUT_DIR = "out"

st.set_page_config(page_title="ダンス比較", layout="wide")
st.title("ダンス比較")


def video_input(label: str, key: str) -> str | None:
    """アップロード or ローカルのパス指定で動画を受け取り、ファイルパスを返す."""
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader(label, type=["mp4", "mov", "m4v", "avi", "webm"], key=f"{key}_up")
    path = c2.text_input("またはファイルのパス", key=f"{key}_path",
                         help="大きい動画はアップロードせず、ここにパスを入れる方が速い（ファイルをこの欄にドラッグ＆ドロップでも可）")
    if up is not None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        buf = up.getbuffer()  # コピーせずに参照（大きい動画でメモリを二重に使わない）
        head = hashlib.sha1(buf[: 1 << 20]).hexdigest()[:10]  # 先頭 1MB + サイズで同じ動画かを判定
        p = os.path.join(UPLOAD_DIR, f"{head}_{len(buf)}_{up.name}")
        if not os.path.exists(p):
            with open(p + ".part", "wb") as f:
                f.write(buf)
            os.replace(p + ".part", p)
        return p
    if path:
        path = path.strip().strip('"').strip("'")  # ドラッグ＆ドロップや「パスをコピー」で付く引用符を除く
        if os.path.exists(path):
            return path
        st.error(f"ファイルが見つかりません: {path}")
    return None


# ---------------------------------------------------------------------------
# サイドバー: 設定
# ---------------------------------------------------------------------------
with st.sidebar:
    mode = st.radio("入力", ["2動画モード", "1動画モード"], help="1動画モードは 1 本の動画に映る二人を比較します")
    backend = st.selectbox("骨格推定", ["yolo", "mediapipe"] if mode == "2動画モード" else ["yolo"],
                           help="mediapipe は 1 人用。yolo は複数人の追跡ができます")
    mirror = st.checkbox("手本を左右反転して比較", help="手本が向かい合わせ（鏡写しでない）で撮影されている場合")

    st.subheader("時間合わせ")
    audio_sync = st.checkbox("音声で 2 本の動画を同期する", value=True, disabled=mode != "2動画モード")
    open_ends = st.checkbox("レビュー対象の前後に余分な部分がある", value=False, disabled=mode != "2動画モード",
                            help="2動画モード用. レビュー対象の動画だけ撮影開始が早い・終了が遅い場合")

    st.subheader("部位の重み")
    st.caption("スコアと「直すところ」の選び方に効きます。0 にするとその部位は評価しません"
               "（例: 腕の形を自由にしているジャンルなら腕を下げる）。")
    w_legs = st.slider("足", 0.0, 3.0, 1.0, 0.25)
    w_hips = st.slider("腰（体幹）", 0.0, 3.0, 1.0, 0.25)
    w_head = st.slider("頭（首の動きを含む）", 0.0, 3.0, 1.0, 0.25)
    w_arms = st.slider("腕", 0.0, 3.0, 1.0, 0.25)
    w_timing = st.slider("タイミング（全身）", 0.0, 3.0, 1.0, 0.25)

    st.subheader("指摘のしきい値")
    base = AnalysisConfig()
    cfg_file = st.file_uploader("設定ファイル (TOML)", type=["toml"])
    if cfg_file is not None:
        import tomllib
        base = AnalysisConfig.from_dict(tomllib.loads(cfg_file.getvalue().decode()))
    st.caption("どの項目も、**大きくするほど判定が緩く**なり（指摘が減り）、スコアも上がります。"
               "しきい値の半分以下のズレは減点しません。")
    angle_thr = st.slider("関節角度の差 [°]", 5.0, 60.0, base.angle_threshold_deg, 1.0,
                          help="手本との関節の角度の差がこれ以上なら指摘")
    timing_thr = st.slider("手本とのタイミング差 [ms]", 30.0, 400.0, base.timing_threshold_ms, 10.0,
                           help="同じ動きを手本より早く/遅くやっている時間がこれ以上なら指摘")
    amp_tol = st.slider("動きの大きさの差 [%]", 5, 60, int(round(base.amplitude_tolerance * 100)), 5,
                        help="手先・足先の振れ幅が手本とこれ以上違えば指摘（25% なら手本の 75% 未満で「小さい」）")
    with st.expander("高度な設定"):
        sharp_tol = st.slider("キレの差 [%]", 10, 70, int(round(base.sharpness_tolerance * 100)), 5,
                              help="加速・減速の鋭さが手本よりこれ以上弱ければ指摘")
        level_thr = st.slider("腰の高さ（沈み込み）の差 [胴の長さの%]", 5, 50, int(round(base.level_threshold * 100)), 1,
                              help="各自の普段の高さからの沈み込み方を比べる")
        facing_thr = st.slider("体の向きの差 [°]", 10.0, 90.0, base.facing_threshold_deg, 5.0)
        lean_thr = st.slider("上体の傾きの差 [°]", 5.0, 45.0, base.lean_threshold_deg, 1.0)
        stance_thr = st.slider("足幅の差 [胴の長さの%]", 10, 80, int(round(base.stance_threshold * 100)), 5)
        travel_thr = st.slider("移動量の差 [胴の長さの%]", 10, 150, int(round(base.travel_threshold * 100)), 5)
        landing_thr = st.slider("着地タイミングの差 [ms]", 30.0, 300.0, base.landing_threshold_ms, 10.0)
        head_tilt_thr = st.slider("頭の傾きの差 [°]", 5.0, 45.0, base.head_tilt_threshold_deg, 1.0)
        head_turn_thr = st.slider("顔の向きの差 [°]", 10.0, 90.0, base.head_turn_threshold_deg, 5.0)
        head_shift_thr = st.slider("頭の横スライドの差 [胴の長さの%]", 3, 40, int(round(base.head_shift_threshold * 100)), 1)
        head_nod_thr = st.slider("頭の上下の差 [胴の長さの%]", 3, 30, int(round(base.head_nod_threshold * 100)), 1)
        full_ratio = st.slider("減点しないズレの範囲 [しきい値に対する割合]", 0.0, 0.9, base.score_full_ratio, 0.05,
                               help="測定誤差を見込んで、しきい値のこの割合以下のズレは減点しない")
        window_s = st.select_slider("評価の区切り [秒]", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], base.window_s_fallback)

    st.subheader("言語化")
    narrator_name = st.radio("指摘文", ["rule", "gemini"],
                             format_func=lambda n: "ルールベース" if n == "rule" else "Gemini で整理（要 GEMINI_API_KEY）")
    if narrator_name == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            st.caption("無料枠では送信内容（数値の要約のみ）が Google の製品改善に使われる可能性があります。")
        else:
            st.warning("環境変数 GEMINI_API_KEY が未設定のため、ルールベースの文章になります。")
    make_video = st.checkbox("比較動画を作る", value=True)

cfg = base.replace(angle_threshold_deg=angle_thr, timing_threshold_ms=timing_thr,
                   amplitude_tolerance=amp_tol / 100, sharpness_tolerance=sharp_tol / 100,
                   level_threshold=level_thr / 100, facing_threshold_deg=facing_thr, lean_threshold_deg=lean_thr,
                   stance_threshold=stance_thr / 100, travel_threshold=travel_thr / 100,
                   landing_threshold_ms=landing_thr, score_full_ratio=full_ratio,
                   head_tilt_threshold_deg=head_tilt_thr, head_turn_threshold_deg=head_turn_thr,
                   head_shift_threshold=head_shift_thr / 100, head_nod_threshold=head_nod_thr / 100,
                   weight_legs=w_legs, weight_hips=w_hips, weight_head=w_head, weight_arms=w_arms,
                   weight_timing=w_timing,
                   window_s_fallback=window_s, dtw_open_ends=open_ends)
rhythm = Rhythm()  # 拍・カウントは使わない（自動検出が不安定なため. 場所は時刻で示す）


class Progress:
    def __init__(self):
        self.bar = st.progress(0.0, text="準備中…")

    def __call__(self, stage, i, n):
        if n:
            self.bar.progress(min(i / n, 1.0), text=f"{stage}: {i}/{n}")


@st.cache_data(show_spinner=False)
def people_image(video: str, backend: str, frame: int):
    """人物 ID を描いた画像と {id: box}."""
    import cv2

    from dance_compare.media.video import read_frame
    from dance_compare.pose.tracking import people_at

    ts = extract_tracks(video, backend)
    f, boxes = people_at(ts, frame)
    img = read_frame(video, ts.video_frame(f))  # f は分析用に間引いたフレーム番号
    for tid, b in boxes.items():
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(img, f"ID {tid}", (x1 + 4, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), {int(k): v.tolist() for k, v in boxes.items()}, f


@st.cache_data(show_spinner=False)
def id_spans(video: str, backend: str) -> dict[int, tuple[float, float]]:
    """各 ID が映っている時間帯（最初と最後）[秒]."""
    ts = extract_tracks(video, backend)
    out = {}
    for tid in ts.ids:
        p = np.where(ts.present(tid))[0]
        if p.size:
            out[int(tid)] = (p[0] / ts.fps, (p[-1] + 1) / ts.fps)
    return out


def _auto_fill(video, backend, ref_id, user_id, f, cfg, mirror):
    """「自動で推定」ボタン: 二人の動きがそろっている区間を推定して、開始・終了の欄に入れる."""
    try:
        curve = suggest_range(extract_tracks(video, backend), Pick(ref_id), Pick(user_id),
                              cfg, mirror)
    except Exception as e:  # 画面上にメッセージを出して続ける
        st.session_state["sync_curve"] = (video, ref_id, user_id, None, str(e))
        return
    st.session_state["sync_curve"] = (video, ref_id, user_id, curve, None)
    if curve.start is not None:
        st.session_state["range_start"] = round(curve.start, 2)
        st.session_state["range_end"] = round(curve.end, 2)


def range_inputs(video, backend, ref_id, user_id, f, cfg, mirror):
    """1動画モードの「比較する区間」の欄. (開始, 終了, 空欄を自動推定するか) を返す."""
    st.markdown("#### 比較する区間")
    st.caption("踊り始める前・踊り終えた後を比較に入れないための指定です。動画を見ながら黄色の端を動かすか、"
               "再生して「ここを開始に」「ここを終了に」を押してください。動かしていない側（灰色）は、"
               "二人の動きがそろっている所から自動で推定します。")
    start, end = range_picker(video, "range")
    c1, c2 = st.columns([2, 3])
    c1.button("自動で推定して区間に入れる", on_click=_auto_fill, args=(video, backend, ref_id, user_id, f, cfg, mirror),
              help="推定結果が区間に入るので、映像で確認して必要なら端を動かしてください")
    auto_range = c2.checkbox("動かしていない側は自動で推定する（オフなら動画の最初・最後まで）", value=True)

    sc = st.session_state.get("sync_curve")
    if sc and sc[:3] == (video, ref_id, user_id):
        curve, err = sc[3], sc[4]
        if err:
            st.error(f"推定できませんでした: {err}")
        else:
            if curve.start is None:
                st.warning("二人の動きがそろっている区間が見つかりませんでした。区間の端を手で動かしてください。")
            else:
                st.info(f"推定: {curve.start:.1f}秒 〜 {curve.end:.1f}秒")
            df = pd.DataFrame({"同期度": curve.score, "しきい値": cfg.sync_threshold},
                              index=np.round(curve.times, 2))
            st.line_chart(df, height=160)
            st.caption("同期度: 二人の動きがどれだけそろっているか（1 = 同じ動き）。しきい値を超えた状態が続く区間を振付とみなします。")
    return start, end, auto_range


# ---------------------------------------------------------------------------
# 入力
# ---------------------------------------------------------------------------
result = None
if mode == "2動画モード":
    ref_video = video_input("お手本の動画", "ref")
    user_video = video_input("レビュー対象の動画", "user")
    r0 = r1 = u0 = u1 = None
    if ref_video and user_video:
        if st.checkbox("比較する区間を指定する（任意）",
                       help="比較に使う部分だけを、動画を見ながら選べます。動かしていない端は動画の最初・最後のままです。"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**お手本**")
                r0, r1 = range_picker(ref_video, "ref_range", ("最初から", "最後まで"))
            with c2:
                st.markdown("**レビュー対象**")
                u0, u1 = range_picker(user_video, "user_range", ("最初から", "最後まで"))
    if st.button("解析する", type="primary", disabled=not (ref_video and user_video)):
        prog = Progress()
        with st.spinner("解析中…（初回は骨格推定に時間がかかります）"):
            result = analyze_two_videos(
                ref_video, user_video, cfg, backend=backend, mirror=mirror,
                ref_range=(r0 or None, r1), user_range=(u0 or None, u1),
                audio_sync=audio_sync, rhythm=rhythm, progress=prog)
else:
    video = video_input("動画（二人以上が映っているもの）", "single")
    if video:
        if st.button("人物を検出"):
            prog = Progress()
            with st.spinner("骨格推定中…"):
                extract_tracks(video, backend, progress=lambda i, n: prog("骨格推定", i, n))
            st.session_state["detected"] = video
        if st.session_state.get("detected") == video:
            st.markdown("#### 人物を選ぶ時刻")
            st.caption("比較したい二人がはっきり映っている所で止めてください。その時刻の人物が下に表示されます。")
            pick_t = time_picker(video, "pick_t")
            frame = int(round(pick_t * extract_tracks(video, backend).fps))
            img, boxes, f = people_image(video, backend, frame)
            ids = sorted(boxes)
            spans = id_spans(video, backend)

            def id_label(i):
                a, b = spans.get(i, (0.0, 0.0))
                return f"ID {i}（{fmt_time(a)}〜{fmt_time(b)} に映っている）"
            clicked = None
            try:  # クリックで選べるコンポーネントがあれば使う（pip install streamlit-image-coordinates）
                from streamlit_image_coordinates import streamlit_image_coordinates
                w = min(900, img.shape[1])
                st.caption("画像をクリックすると、その人物を下の欄で選びやすくなります")
                pt = streamlit_image_coordinates(img, width=w, key="pick")
                if pt:
                    s = img.shape[1] / w
                    from dance_compare.pose.tracking import pick_by_point
                    clicked = pick_by_point(extract_tracks(video, backend), f, pt["x"] * s, pt["y"] * s)
                    st.info(f"クリックした人物: ID {clicked}")
            except ImportError:
                st.image(img, caption=f"{fmt_time(f / extract_tracks(video, backend).fps)} の人物", width="stretch")
            if len(ids) < 2:
                st.error("この時刻には二人以上が検出されていません。別の時刻を試してください。")
            else:
                c1, c2 = st.columns(2)
                ref_id = c1.selectbox("お手本", ids, index=0, format_func=id_label)
                user_id = c2.selectbox("レビュー対象", ids, index=ids.index(clicked) if clicked in ids else 1,
                                       format_func=id_label)
                st.caption("「映っている」時間帯は、追跡が途切れる前のその ID だけの範囲です。"
                           "解析では、途切れた後に別の ID になった同じ人物もつなげて比較します。")
                start, end, auto_range = range_inputs(video, backend, ref_id, user_id, f, cfg, mirror)
                if st.button("解析する", type="primary", disabled=ref_id == user_id):
                    prog = Progress()
                    with st.spinner("解析中…"):
                        result = analyze_single_video(video, Pick(ref_id), Pick(user_id), cfg,
                                                      backend=backend, mirror=mirror, rhythm=rhythm, progress=prog,
                                                      time_range=(start, end), auto_range=auto_range)

if result is not None:
    narrator = get_narrator(narrator_name)
    out_dir = os.path.join(OUT_DIR, "latest")
    with st.spinner("比較動画を作成中…" if make_video else "保存中…"):
        prog = Progress() if make_video else None
        paths = save_outputs(result, out_dir, narrator, render=make_video,
                             progress=(lambda i, n: prog("比較動画", i, n)) if prog else None)
    st.session_state["last"] = (result, paths["text"], paths, getattr(narrator, "last_error", None))

# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------
if "last" in st.session_state:
    result, text, paths, err = st.session_state["last"]
    rep = result.report
    out_dir = os.path.dirname(paths["json"])
    for n in result.notes:
        if "できません" in n or "付けられなかった" in n:
            st.warning(n)
        else:
            st.info(n)
    if err:
        st.warning(err)

    # 部位別スコア（重み 0 の部位は「-」）
    sc = rep.scores
    keys = [k for k in GROUPS if k in sc]
    cols = st.columns(len(keys) + 1)
    cols[0].metric("総合", "-" if sc.get("overall") is None else f"{sc['overall']:.0f}")
    for c, k in zip(cols[1:], keys):
        c.metric(GROUPS[k], "-" if sc[k] is None else f"{sc[k]:.0f}")

    if "video" in paths and os.path.exists(paths["video"]):
        st.subheader("比較動画")
        st.video(paths["video"])

    # 直すところ（ズレの大きい瞬間の上位）: 左に前後の短い動画（スロー）、右に修正点
    st.subheader("直すところ")
    if not rep.moments:
        st.success("大きなズレは見つかりませんでした。")
    else:
        speed = rep.meta["config"].get("moment_clip_speed", 0.5)
        st.caption(f"ズレの大きい瞬間の前後を {speed:g} 倍速のスローで繰り返し再生します。赤い丸がズレている部位です。")
    for m in rep.moments:
        clip = os.path.join(out_dir, m.get("clip") or "")
        img = os.path.join(out_dir, m.get("image") or "")
        left, right = st.columns([3, 2], vertical_alignment="center")
        if m.get("clip") and os.path.exists(clip):
            left.video(clip, loop=True, autoplay=True, muted=True)
        elif m.get("image") and os.path.exists(img):
            left.image(img, width="stretch")
        right.markdown(f"**{moment_title(m)}**\n\n" + "\n".join(f"- {line}" for line in moment_lines(m)))

    if rep.tendencies:
        st.markdown("**全体の傾向**\n\n" + "\n".join(f"- {tendency_line(f)}" for f in rep.tendencies))

    with st.expander("詳細（すべての指摘・時系列）"):
        rows = [{"範囲": r["scope"], "カウント": r["label"], "開始[秒]": r["t_start"], "終了[秒]": r["t_end"],
                 "部位グループ": r["group_ja"], "観点": r["aspect_ja"], "項目": r["kind_ja"], "部位": r["part_ja"],
                 "ズレ量": r["value_text"], "優先度": r["severity"], "説明": r["description"]}
                for r in findings_rows(rep)]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        tl = rep.timeline
        t = np.round(np.asarray(tl["t"]), 2)
        tab1, tab2, tab3 = st.tabs(["関節角度の差 [°]", "タイミング差 [ms]", "動きの速さ"])
        with tab1:
            st.line_chart(pd.DataFrame({JA_NAMES[k]: v for k, v in tl["angle_diff"].items()}, index=t))
            st.caption("正 = 手本より角度が大きい（伸びている・腕/脚が上がっている）")
        with tab2:
            st.line_chart(pd.DataFrame({"レビュー対象の遅れ": tl["timing_ms"]}, index=t))
        with tab3:
            st.line_chart(pd.DataFrame({"手本": tl["ref_speed"], "レビュー対象": tl["user_speed"]}, index=t))
        if narrator_name == "gemini":
            st.markdown(text)
    c1, c2, c3 = st.columns(3)
    with open(paths["findings_csv"], "rb") as f:
        c3.download_button("findings.csv", f.read(), "findings.csv")
    with open(paths["json"], "rb") as f:
        c1.download_button("report.json", f.read(), "report.json")
    with open(paths["markdown"], "rb") as f:
        c2.download_button("report.md", f.read(), "report.md")
