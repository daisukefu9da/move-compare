"""E2E テスト用の動画を作る（実写の二人が映った写真を、ゆっくり動かして 12 秒の動画にする）.

合成の棒人間は YOLO が人として検出しないため、Ultralytics に入っているサンプル写真（zidane.jpg）を使う。
Playwright の Chromium は H.264 を再生できないので VP9（webm）で書き出す。

    python web/e2e/make_sample.py  → .cache/e2e/two_people.webm
"""

from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

OUT = os.path.join(".cache", "e2e")


def main() -> str:
    import ultralytics

    src = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg")
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "two_people.webm")
    if os.path.exists(dst):
        return dst
    img = cv2.imread(src)
    h, w = img.shape[:2]
    tmp = os.path.join(OUT, "two_people_raw.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    for i in range(30 * 12):
        t = i / 30
        s = 1.0 + 0.08 * np.sin(t * 2)
        m = np.array([[s, 0, (1 - s) * w / 2 + 40 * np.sin(t * 1.3)], [0, s, (1 - s) * h / 2 + 15 * np.sin(t * 2.1)]])
        vw.write(cv2.resize(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT), (960, 540)))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-shortest", "-c:v", "libvpx-vp9", "-b:v", "1M", "-c:a", "libopus", dst], check=True)
    os.remove(tmp)
    return dst


if __name__ == "__main__":
    print(main())
