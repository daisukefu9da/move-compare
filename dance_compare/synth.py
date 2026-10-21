"""テスト・動作確認用の合成ダンサー（棒人間）.

拍ごとにポーズが切り替わる簡単な振付を生成する。遅れ・振れ幅・特定区間の肘の角度などを
わざとずらしたレビュー対象を作り、解析結果が期待どおりになるかを確かめるのに使う。
"""

from __future__ import annotations

import numpy as np

from .core.skeleton import KP, NUM_KEYPOINTS, PoseSequence

# 1 エイト分の振付: (左腕の上げ角, 右腕の上げ角, 膝の曲げ, 横移動の目標)  角度は度, 下向き 0
CHOREO = [
    (20, 20, 0, 0.0), (90, 20, 25, 0.0), (90, 90, 0, 0.0), (160, 90, 25, 0.0),
    (160, 160, 0, 0.5), (90, 160, 25, 0.5), (20, 90, 0, 0.0), (20, 20, 25, 0.0),
]


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _choreo_state(beat_pos: np.ndarray, sharp: float = 0.2) -> np.ndarray:
    """拍位置（実数）-> (T, 4) 各パラメータ. 拍 k で CHOREO[k] のポーズへ、拍を中心に sharp 拍かけて移る."""
    states = np.array(CHOREO, dtype=np.float64)
    k = np.floor(beat_pos + 0.5).astype(int)
    x = beat_pos - k  # -0.5..0.5, 0 が拍の位置
    prev = states[(k - 1) % len(states)]
    cur = states[k % len(states)]
    s = _smoothstep((x + sharp / 2) / sharp)[:, None]
    return prev + (cur - prev) * s


def synth_dancer(
    duration: float = 16.0,
    fps: float = 30.0,
    bpm: float = 120.0,
    first_beat: float = 0.5,
    delay_s: float = 0.0,
    tempo: float = 1.0,
    amplitude: float = 1.0,
    sharpness: float = 0.2,
    elbow_bend: tuple[float, float, float] | None = None,
    size: float = 120.0,
    center: tuple[float, float] = (320.0, 260.0),
    width: int = 640,
    height: int = 480,
    noise: float = 0.0,
    seed: int = 0,
) -> PoseSequence:
    """合成ダンサーの骨格系列.

    delay_s:    拍に対する遅れ [秒]
    tempo:      1 より小さいとゆっくり踊る
    amplitude:  腕の振れ幅の倍率
    sharpness:  ポーズ遷移にかける拍数（大きいほどぬるっと動く = キレがない）
    elbow_bend: (開始秒, 終了秒, 曲げる角度) その区間だけ右肘を曲げる
    size:       胴の長さ [px]（体格差のテスト用）
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    t = np.arange(n) / fps
    period = 60.0 / bpm
    beat_pos = ((t - delay_s) * tempo - first_beat) / period
    st = _choreo_state(beat_pos, sharpness)
    larm = 20 + (st[:, 0] - 20) * amplitude
    rarm = 20 + (st[:, 1] - 20) * amplitude
    knee = st[:, 2]
    shift = st[:, 3]

    L = size
    kp = np.zeros((n, NUM_KEYPOINTS, 2))
    hip_x = center[0] + shift * L
    sink = L * 0.25 * (1 - np.cos(np.radians(knee)))
    hip_y = center[1] + sink
    kp[:, KP["left_hip"]] = np.stack([hip_x + 0.2 * L, hip_y], 1)   # 正面向き: 本人の左は画像の右
    kp[:, KP["right_hip"]] = np.stack([hip_x - 0.2 * L, hip_y], 1)
    kp[:, KP["left_shoulder"]] = np.stack([hip_x + 0.3 * L, hip_y - L], 1)
    kp[:, KP["right_shoulder"]] = np.stack([hip_x - 0.3 * L, hip_y - L], 1)
    kp[:, KP["nose"]] = np.stack([hip_x, hip_y - 1.35 * L], 1)
    for i, (nm, sign) in enumerate([("left", 1), ("right", -1)]):
        kp[:, KP[f"{nm}_eye"]] = kp[:, KP["nose"]] + np.array([sign * 0.07 * L, -0.05 * L])
        kp[:, KP[f"{nm}_ear"]] = kp[:, KP["nose"]] + np.array([sign * 0.15 * L, 0.0])

    def arm(side, sign, raise_deg, elbow_deg):
        sh = kp[:, KP[f"{side}_shoulder"]]
        a = np.radians(raise_deg)
        d1 = np.stack([sign * np.sin(a), np.cos(a)], 1)  # 下向き 0, 外側へ上がる
        el = sh + d1 * 0.6 * L
        b = a + np.radians(elbow_deg)
        d2 = np.stack([sign * np.sin(b), np.cos(b)], 1)
        kp[:, KP[f"{side}_elbow"]] = el
        kp[:, KP[f"{side}_wrist"]] = el + d2 * 0.55 * L

    r_elbow = np.zeros(n)
    if elbow_bend is not None:
        s, e, deg = elbow_bend
        r_elbow = np.where((t >= s) & (t < e), deg, 0.0)
    arm("left", 1, larm, np.zeros(n))
    arm("right", -1, rarm, r_elbow)

    for side, sign in [("left", 1), ("right", -1)]:
        hp = kp[:, KP[f"{side}_hip"]]
        kb = np.radians(knee)
        thigh = np.stack([sign * 0.1 * np.ones(n) + np.sin(kb) * 0.5 * sign, np.cos(kb)], 1)
        thigh /= np.linalg.norm(thigh, axis=1, keepdims=True)
        kn = hp + thigh * 0.9 * L
        shin = np.stack([sign * 0.05 * np.ones(n), np.ones(n)], 1)
        shin /= np.linalg.norm(shin, axis=1, keepdims=True)
        kp[:, KP[f"{side}_knee"]] = kn
        kp[:, KP[f"{side}_ankle"]] = kn + shin * 0.9 * L

    if noise > 0:
        kp += rng.normal(0, noise, kp.shape)
    return PoseSequence(kp, np.ones((n, NUM_KEYPOINTS)) * 0.9, fps, width, height)


def synth_beats(duration: float, bpm: float = 120.0, first_beat: float = 0.5) -> np.ndarray:
    return np.arange(first_beat, duration, 60.0 / bpm)


def write_synth_video(seq: PoseSequence, path: str, beats: np.ndarray | None = None) -> None:
    """合成ダンサーを棒人間の動画にする（拍の位置にクリック音を入れる）. ffmpeg が必要."""
    import os
    import subprocess
    import tempfile

    import cv2

    from .core.skeleton import BONES

    sr = 22050
    n = len(seq)
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, "click.wav")
    y = np.zeros(int(seq.duration * sr) + sr)
    if beats is not None:
        # 拍の位置にクリック音、その上にランダムなメロディを重ねる
        # （完全に周期的な音だと、音声同期が拍単位で曖昧になるため）
        rng = np.random.default_rng(123)
        m = int(0.03 * sr)
        click = np.sin(2 * np.pi * 1800 * np.arange(m) / sr) * np.exp(-np.arange(m) / (0.005 * sr))
        for b in beats:
            i = int(b * sr)
            if i + m <= y.size:
                y[i:i + m] += click
        step = (beats[1] - beats[0]) / 2
        n_notes = int((beats[-1] - beats[0]) / step) + 1
        notes = 220 * 2 ** (rng.integers(0, 24, n_notes) / 12)
        L = int(step * sr)
        env = np.exp(-np.arange(L) / (0.08 * sr))
        for k, f in enumerate(notes):
            i = int((beats[0] + k * step) * sr)
            if i + L > y.size:
                break
            y[i:i + L] += 0.4 * np.sin(2 * np.pi * f * np.arange(L) / sr) * env
    import wave
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32000).astype(np.int16).tobytes())
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{seq.width}x{seq.height}", "-r", f"{seq.fps}", "-i", "-", "-i", wav,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        img = np.full((seq.height, seq.width, 3), 235, np.uint8)
        kp = seq.keypoints[i]
        for a, b in BONES:
            cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (60, 60, 60), 6, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in kp[0]), int(0.18 * np.linalg.norm(kp[5] - kp[11])), (60, 60, 60), -1)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
