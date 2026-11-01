"""Modal（GPU のクラウド）で Streamlit のアプリを動かす.

    modal deploy deploy/modal_app.py     （GitHub Actions の .github/workflows/modal.yml が push のたびに実行）

- GPU（T4）付きのコンテナで今の app.py をそのまま動かす。使っていない間はコンテナが止まり、課金も止まる
- アクセス用のパスワードは Modal の Secret「dance-compare」の DANCE_COMPARE_PASSWORD
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
)

app = modal.App("dance-compare", image=image)
data = modal.Volume.from_name("dance-compare-data", create_if_missing=True)


@app.function(
    gpu="T4",
    volumes={"/data": data},
    secrets=[modal.Secret.from_name("dance-compare")],
    timeout=60 * 60 * 4,       # 1 回の起動で使える最長時間
    scaledown_window=5 * 60,   # 最後に使ってから 5 分で止める（GPU の課金を止める）
    max_containers=1,          # 先生 1 人で使う前提。同時に何台も起動しない
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=180)
def ui():
    import subprocess

    subprocess.Popen(
        "streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true "
        "--server.enableCORS false --server.enableXsrfProtection false",
        shell=True, cwd=APP_DIR,
    )
