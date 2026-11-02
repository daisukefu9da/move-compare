"""Modal（GPU のクラウド）で Streamlit のアプリを動かす.

    modal deploy deploy/modal_app.py     （GitHub Actions の .github/workflows/modal.yml が push のたびに実行）

- 画面（今の app.py）は CPU のコンテナで動かし、骨格推定だけ GPU のコンテナ（track_pose）で行う。
  GPU が混んでいても画面は開ける。GPU がなかなか空かなければ CPU で推定する（pose/remote.py）。
  使っていない間はコンテナが止まり、課金も止まる
- パスワードはかけていない（URL を知っていれば誰でも開ける）。かける場合は ui 関数に
  secrets=[modal.Secret.from_name("dance-compare")]（DANCE_COMPARE_PASSWORD）を付ける
- 保存した手本の動画だけを Volume（/data）に残す。アップロードした動画や解析結果はコンテナの一時領域に置き、
  コンテナが止まると消える（生徒の動画をサーバーに残さない）
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = "/app"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "fonts-noto-cjk", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(ROOT / "deploy" / "requirements-server.txt"))
    # 骨格推定のモデルを先に取っておく（起動のたびにダウンロードしない）
    .run_commands(f"mkdir -p {APP_DIR} && cd {APP_DIR} && python -c \"from ultralytics import YOLO; YOLO('yolo11n-pose.pt')\"")
    .env({
        "DANCE_COMPARE_MODEL_DIR": "/data/model_videos",
        "DANCE_COMPARE_FONT": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "DANCE_COMPARE_HIDE_PATH_INPUT": "1",   # サーバーのファイルをパスで読ませない
        "YOLO_CONFIG_DIR": "/tmp/ultralytics",
    })
    .add_local_dir(str(ROOT / "dance_compare"), f"{APP_DIR}/dance_compare", ignore=["**/__pycache__"])
    .add_local_dir(str(ROOT / ".streamlit"), f"{APP_DIR}/.streamlit")
    .add_local_file(str(ROOT / "app.py"), f"{APP_DIR}/app.py")
    .add_local_file(str(ROOT / "server.py"), f"{APP_DIR}/server.py")
)

app = modal.App("dance-compare", image=image)
data = modal.Volume.from_name("dance-compare-data", create_if_missing=True)


@app.function(
    gpu=["T4", "L4", "A10G"],  # 安い順に、空いている GPU を使う（T4 だけだと混んでいて起動できないことがあった）
    timeout=15 * 60,
    scaledown_window=5 * 60,   # 最後に使ってから 5 分で止める（画面を開いた時に起こすので、動画を送り終わるまで待てるように）
    max_containers=2,
)
def track_pose(video: bytes, suffix: str) -> bytes:
    """骨格推定だけを GPU で行う（画面のサーバーから呼ばれる）. 空のバイト列なら起動しておくだけ."""
    import os
    import sys

    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)  # yolo11n-pose.pt はここにある
    from dance_compare.pose.remote import track_bytes

    return track_bytes(video, suffix)


@app.function(
    cpu=4.0,                   # 動画の変換（ffmpeg）と比較動画の作成を速くする
    memory=6144,
    volumes={"/data": data},
    env={"DANCE_COMPARE_REMOTE_POSE": "dance-compare/track_pose"},  # 骨格推定は track_pose（GPU）に頼む
    timeout=60 * 60 * 4,       # 1 回の起動で使える最長時間
    scaledown_window=10 * 60,  # 画面のサーバーは CPU なので安い。最後に使ってから 10 分で止める
    max_containers=1,          # 先生 1 人で使う前提。同時に何台も起動しない
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=180)
def ui():
    import subprocess

    subprocess.Popen(
        "streamlit run server.py --server.port 8000 --server.address 0.0.0.0 --server.headless true "
        "--server.enableCORS false --server.enableXsrfProtection false --server.websocketPingInterval 20",
        shell=True, cwd=APP_DIR,
    )
