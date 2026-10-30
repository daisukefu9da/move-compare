"""ブラウザ版（PoC）の公開用ファイル一式を web/_site に作る.

    python web/build.py

- web/poc/ の画面
- web/models/ の姿勢推定モデル（ONNX）
- ONNX Runtime Web（npm から）と Pyodide + numpy（GitHub のリリースから）を vendor/ に置く
  （CDN に頼らず同じサイトから配る。ダウンロードしたものは .cache/web/ に取っておく）
- 解析コード（dance_compare のうちブラウザで使う部分）を dance_compare.zip に固める
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
SITE = os.path.join(WEB, "_site")
CACHE = os.path.join(ROOT, ".cache", "web")

ORT_VERSION = "1.22.0"
ORT_FILES = ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]
PYODIDE_VERSION = "0.27.7"
PYODIDE_FILES = ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json",
                 "numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl"]
# ブラウザでは使わない（動画ファイル・ffmpeg・OpenCV・学習済みモデルの読み込み）部分
PY_EXCLUDE = ("render/", "ui/", "pose/yolo.py", "pose/mediapipe_backend.py", "cli.py", "synth.py", "validation.py",
              "__main__.py")


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("download", url)
        with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
    return path


def vendor_ort(dst: str) -> None:
    d = os.path.join(CACHE, f"ort-{ORT_VERSION}")
    if not all(os.path.exists(os.path.join(d, f)) for f in ORT_FILES):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["npm", "pack", f"onnxruntime-web@{ORT_VERSION}", "--pack-destination", d],
                       check=True, capture_output=True)
        with tarfile.open(os.path.join(d, f"onnxruntime-web-{ORT_VERSION}.tgz")) as t:
            for f in ORT_FILES:
                m = t.getmember(f"package/dist/{f}")
                with t.extractfile(m) as src, open(os.path.join(d, f), "wb") as out:
                    shutil.copyfileobj(src, out)
    os.makedirs(dst, exist_ok=True)
    for f in ORT_FILES:
        shutil.copy2(os.path.join(d, f), dst)


def vendor_pyodide(dst: str) -> None:
    d = os.path.join(CACHE, f"pyodide-{PYODIDE_VERSION}")
    if not all(os.path.exists(os.path.join(d, f)) for f in PYODIDE_FILES):
        tar = _download(f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/"
                        f"pyodide-{PYODIDE_VERSION}.tar.bz2", os.path.join(CACHE, f"pyodide-{PYODIDE_VERSION}.tar.bz2"))
        os.makedirs(d, exist_ok=True)
        with tarfile.open(tar) as t:
            for f in PYODIDE_FILES:
                with t.extractfile(f"pyodide/{f}") as src, open(os.path.join(d, f), "wb") as out:
                    shutil.copyfileobj(src, out)
        os.remove(tar)  # 必要なファイルだけ残す（約 390MB → 17MB）
    os.makedirs(dst, exist_ok=True)
    for f in PYODIDE_FILES:
        shutil.copy2(os.path.join(d, f), dst)


def pack_python(path: str) -> None:
    pkg = os.path.join(ROOT, "dance_compare")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(pkg):
            for f in files:
                if not f.endswith(".py"):
                    continue
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, pkg).replace(os.sep, "/")
                if not rel.startswith(PY_EXCLUDE):
                    z.write(full, "dance_compare/" + rel)


def main() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    shutil.copytree(os.path.join(WEB, "poc"), SITE)
    shutil.copytree(os.path.join(WEB, "models"), os.path.join(SITE, "models"))
    vendor_ort(os.path.join(SITE, "vendor", "ort"))
    vendor_pyodide(os.path.join(SITE, "vendor", "pyodide"))
    pack_python(os.path.join(SITE, "dance_compare.zip"))
    open(os.path.join(SITE, ".nojekyll"), "w").close()  # GitHub Pages で _ から始まるファイルも配る
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(SITE) for f in fs)
    print(f"built {SITE} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
