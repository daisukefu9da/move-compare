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
from dance_compare.core.report import ASPECTS
from dance_compare.core.skeleton import JA_NAMES
from dance_compare.feedback import get_narrator
from dance_compare.feedback.table import findings_rows
from dance_compare.pipeline import (Pick, Rhythm, analyze_single_video, analyze_two_videos,
                                    extract_tracks, save_outputs, suggest_range)

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

    st.subheader("拍・カウント")
    beat_mode = st.radio("拍の取り方", ["音声から自動検出", "BPM を指定", "使わない"], horizontal=False)
    bpm = None
    if beat_mode == "BPM を指定":
        bpm = st.number_input("BPM", 40.0, 240.0, 120.0, 0.5)
    first_one = st.number_input("「1エイト目の1」の時刻 [秒]（手本動画, 0 なら自動）", 0.0, 3600.0, 0.0, 0.01)

    st.subheader("時間合わせ")
    audio_sync = st.checkbox("音声で 2 本の動画を同期する", value=True, disabled=mode != "2動画モード")
    open_ends = st.checkbox("レビュー対象の前後に余分な部分がある", value=False, disabled=mode != "2動画モード",
                            help="2動画モード用. レビュー対象の動画だけ撮影開始が早い・終了が遅い場合")

    st.subheader("指摘のしきい値")
    base = AnalysisConfig()
    cfg_file = st.file_uploader("設定ファイル (TOML)", type=["toml"])
    if cfg_file is not None:
        import tomllib
        base = AnalysisConfig.from_dict(tomllib.loads(cfg_file.getvalue().decode()))
    angle_thr = st.slider("関節角度の差 [°]", 5.0, 60.0, base.angle_threshold_deg, 1.0)
    timing_thr = st.slider("手本とのタイミング差 [ms]", 30.0, 400.0, base.timing_threshold_ms, 10.0)
    rhythm_thr = st.slider("拍とのズレ [ms]", 20.0, 200.0, base.rhythm_threshold_ms, 5.0)
    amp_low = st.slider("動きが小さいとみなす比率", 0.3, 0.95, base.amplitude_ratio_low, 0.05)
    window_counts = st.select_slider("評価の区切り [カウント]", [1, 2, 4, 8], base.window_counts)
    accent_mode = st.radio("動きのアクセント", ["peak", "hit"], index=0 if base.accent_mode == "peak" else 1,
                           format_func=lambda m: "速度のピーク" if m == "peak" else "急停止（ヒット）")

    st.subheader("言語化")
    narrator_name = st.radio("指摘文", ["rule", "gemini"],
                             format_func=lambda n: "ルールベース" if n == "rule" else "Gemini で整理（要 GEMINI_API_KEY）")
    if narrator_name == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            st.caption("無料枠では送信内容（数値の要約のみ）が Google の製品改善に使われる可能性があります。")
        else:
            st.warning("環境変数 GEMINI_API_KEY が未設定のため、ルールベースの文章になります。")
    make_video = st.checkbox("比較動画を作る", value=True)

cfg = base.replace(angle_threshold_deg=angle_thr, timing_threshold_ms=timing_thr, rhythm_threshold_ms=rhythm_thr,
                   amplitude_ratio_low=amp_low, window_counts=window_counts, accent_mode=accent_mode,
                   dtw_open_ends=open_ends)
rhythm = Rhythm(bpm=bpm, first_one=first_one or None, detect=beat_mode != "使わない")


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
    img = read_frame(video, f)
    for tid, b in boxes.items():
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(img, f"ID {tid}", (x1 + 4, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), {int(k): v.tolist() for k, v in boxes.items()}, f


@st.cache_data(show_spinner=False)
def frame_at(video: str, t: float):
    """動画の t 秒のコマ（RGB）."""
    import cv2

    from dance_compare.media.video import probe, read_frame

    info = probe(video)
    img = read_frame(video, max(0, min(int(round(t * info.fps)), info.frames - 1)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _auto_fill(video, backend, ref_id, user_id, f, cfg, mirror):
    """「自動で推定」ボタン: 二人の動きがそろっている区間を推定して、開始・終了の欄に入れる."""
    try:
        curve = suggest_range(extract_tracks(video, backend), Pick(ref_id, frame=f), Pick(user_id, frame=f),
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
    st.caption("踊り始める前・踊り終えた後を比較に入れないための指定です。空欄の側は、二人の動きがそろっている所から自動で推定します。")
    c1, c2, c3 = st.columns([2, 2, 2])
    start = c1.number_input("開始 [秒]（空欄 = 自動）", min_value=0.0, value=None, step=0.1, key="range_start")
    end = c2.number_input("終了 [秒]（空欄 = 自動）", min_value=0.0, value=None, step=0.1, key="range_end")
    c3.write("")
    c3.button("自動で推定して欄に入れる", on_click=_auto_fill, args=(video, backend, ref_id, user_id, f, cfg, mirror),
              help="推定結果は欄に入るので、確認して必要なら書き換えてください")
    auto_range = st.checkbox("空欄の側は自動で推定する（オフなら動画の最初・最後まで）", value=True)

    sc = st.session_state.get("sync_curve")
    if sc and sc[:3] == (video, ref_id, user_id):
        curve, err = sc[3], sc[4]
        if err:
            st.error(f"推定できませんでした: {err}")
        else:
            if curve.start is None:
                st.warning("二人の動きがそろっている区間が見つかりませんでした。開始・終了を手動で入れてください。")
            else:
                st.info(f"推定: {curve.start:.1f}秒 〜 {curve.end:.1f}秒")
            df = pd.DataFrame({"同期度": curve.score, "しきい値": cfg.sync_threshold},
                              index=np.round(curve.times, 2))
            st.line_chart(df, height=160)
            st.caption("同期度: 二人の動きがどれだけそろっているか（1 = 同じ動き）。しきい値を超えた状態が続く区間を振付とみなします。")
    if start is not None or end is not None:
        cols = st.columns(2)
        if start is not None:
            cols[0].image(frame_at(video, start), caption=f"開始 {start:.1f}秒", width="stretch")
        if end is not None:
            cols[1].image(frame_at(video, end), caption=f"終了 {end:.1f}秒", width="stretch")
    return start, end, auto_range


# ---------------------------------------------------------------------------
# 入力
# ---------------------------------------------------------------------------
result = None
if mode == "2動画モード":
    ref_video = video_input("お手本の動画", "ref")
    user_video = video_input("レビュー対象の動画", "user")
    with st.expander("区間の指定（任意）"):
        c = st.columns(4)
        r0 = c[0].number_input("手本 開始 [秒]", 0.0, value=0.0)
        r1 = c[1].number_input("手本 終了 [秒]（0 = 最後まで）", 0.0, value=0.0)
        u0 = c[2].number_input("対象 開始 [秒]", 0.0, value=0.0)
        u1 = c[3].number_input("対象 終了 [秒]（0 = 最後まで）", 0.0, value=0.0)
    if st.button("解析する", type="primary", disabled=not (ref_video and user_video)):
        prog = Progress()
        with st.spinner("解析中…（初回は骨格推定に時間がかかります）"):
            result = analyze_two_videos(
                ref_video, user_video, cfg, backend=backend, mirror=mirror,
                ref_range=(r0 or None, r1 or None), user_range=(u0 or None, u1 or None),
                audio_sync=audio_sync, rhythm=rhythm, progress=prog)
else:
    video = video_input("動画（二人以上が映っているもの）", "single")
    if video:
        frame = st.number_input("人物を選ぶフレーム番号", 0, value=0, step=10)
        if st.button("人物を検出"):
            prog = Progress()
            with st.spinner("骨格推定中…"):
                extract_tracks(video, backend, progress=lambda i, n: prog("骨格推定", i, n))
            st.session_state["detected"] = (video, frame)
        if st.session_state.get("detected") == (video, frame):
            img, boxes, f = people_image(video, backend, frame)
            ids = sorted(boxes)
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
                st.image(img, caption=f"フレーム {f} の人物", width="stretch")
            if len(ids) < 2:
                st.error("二人以上が検出されていません。別のフレームを試してください。")
            else:
                c1, c2 = st.columns(2)
                ref_id = c1.selectbox("お手本", ids, index=0, format_func=lambda i: f"ID {i}")
                user_id = c2.selectbox("レビュー対象", ids, index=ids.index(clicked) if clicked in ids else 1,
                                       format_func=lambda i: f"ID {i}")
                start, end, auto_range = range_inputs(video, backend, ref_id, user_id, f, cfg, mirror)
                if st.button("解析する", type="primary", disabled=ref_id == user_id):
                    prog = Progress()
                    with st.spinner("解析中…"):
                        result = analyze_single_video(video, Pick(ref_id, frame=f), Pick(user_id, frame=f), cfg,
                                                      backend=backend, mirror=mirror, rhythm=rhythm, progress=prog,
                                                      time_range=(start, end), auto_range=auto_range)

if result is not None:
    narrator = get_narrator(narrator_name)
    text = narrator.narrate(result.report)
    out_dir = os.path.join(OUT_DIR, "latest")
    with st.spinner("比較動画を作成中…" if make_video else "保存中…"):
        prog = Progress() if make_video else None
        paths = save_outputs(result, out_dir, text, render=make_video,
                             progress=(lambda i, n: prog("比較動画", i, n)) if prog else None)
    st.session_state["last"] = (result, text, paths, getattr(narrator, "last_error", None))

# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------
if "last" in st.session_state:
    result, text, paths, err = st.session_state["last"]
    rep = result.report
    for n in result.notes:
        if "できません" in n or "付けられなかった" in n:
            st.warning(n)
        else:
            st.info(n)
    if err:
        st.warning(err)

    sc = rep.scores
    cols = st.columns(len(sc))
    cols[0].metric("総合", "-" if sc["overall"] is None else f"{sc['overall']:.0f}")
    for c, (k, v) in zip(cols[1:], [(k, v) for k, v in sc.items() if k != "overall"]):
        c.metric(ASPECTS[k], "-" if v is None else f"{v:.0f}")

    left, right = st.columns([3, 2])
    with left:
        if "video" in paths and os.path.exists(paths["video"]):
            st.video(paths["video"])
        st.subheader("時系列")
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
    with right:
        st.markdown(text)

    st.subheader("検出一覧（カウント・部位・観点・ズレ量）")
    rows = [{"範囲": r["scope"], "カウント": r["label"], "開始[秒]": r["t_start"], "終了[秒]": r["t_end"],
             "観点": r["aspect_ja"], "項目": r["kind_ja"], "部位": r["part_ja"], "ズレ量": r["value_text"],
             "優先度": r["severity"], "説明": r["description"]} for r in findings_rows(rep)]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    c1, c2, c3 = st.columns(3)
    with open(paths["findings_csv"], "rb") as f:
        c3.download_button("findings.csv", f.read(), "findings.csv")
    with open(paths["json"], "rb") as f:
        c1.download_button("report.json", f.read(), "report.json")
    with open(paths["markdown"], "rb") as f:
        c2.download_button("report.md", f.read(), "report.md")
