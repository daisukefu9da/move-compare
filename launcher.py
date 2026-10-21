"""起動用スクリプト: 最新版の取り込み → 必要なら準備 → 画面の起動 を自動で行う.

start.command（Mac）/ start.bat（Windows）をダブルクリックすると、これが実行される。
標準ライブラリだけで動く（仮想環境を作る前に実行するため）。

1. git pull で GitHub の最新版を取り込む（失敗しても今の版で起動を続ける）
2. 仮想環境 .venv が無ければ作る
3. requirements.txt が前回から変わっていれば pip install する
4. streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQ = os.path.join(ROOT, "requirements.txt")
STAMP = os.path.join(VENV, ".requirements.sha1")


def say(msg: str) -> None:
    print(f"[起動] {msg}", flush=True)


def venv_python() -> str:
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def update() -> None:
    if shutil.which("git") is None or not os.path.isdir(os.path.join(ROOT, ".git")):
        say("git が無い（または git clone したフォルダでない）ため、更新の確認をとばします")
        return
    say("最新版を確認しています…")
    r = subprocess.run(["git", "pull", "--ff-only"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        out = r.stdout.strip()
        say("すでに最新です" if "Already up to date" in out or "最新" in out else "最新版を取り込みました")
    else:
        say("更新を取り込めませんでした（今の版で起動します）")
        say("ネットにつながっていない、またはこのフォルダのファイルを編集している可能性があります")
        print(r.stderr.strip())


def ensure_venv() -> None:
    if not os.path.exists(venv_python()):
        if sys.version_info < (3, 11):
            sys.exit(f"Python 3.11 以上が必要です（今は {sys.version.split()[0]}）")
        say("初回の準備をしています（数分〜十数分かかります）…")
        subprocess.check_call([sys.executable, "-m", "venv", VENV])
    with open(REQ, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()
    old = open(STAMP).read().strip() if os.path.exists(STAMP) else ""
    if digest != old:
        say("必要なパッケージをインストールしています…")
        subprocess.check_call([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([venv_python(), "-m", "pip", "install", "-r", REQ])
        with open(STAMP, "w") as f:
            f.write(digest)


def main() -> None:
    os.chdir(ROOT)
    update()
    ensure_venv()
    if shutil.which("ffmpeg") is None:
        say("注意: ffmpeg が見つかりません。音声（拍・同期）と比較動画の書き出しが使えません")
    say("画面を起動します（終了するにはこのウィンドウを閉じるか Ctrl+C）")
    subprocess.call([venv_python(), "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
