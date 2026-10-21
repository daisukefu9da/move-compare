"""音声: 動画からの抽出、拍の検出、2 本の動画の音声による同期.

librosa が入っていればそれで拍を検出し、無ければ numpy だけの簡易実装を使う。
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

SR = 22050
HOP = 512


def extract_audio(path: str, sr: int = SR) -> np.ndarray | None:
    """ffmpeg で動画からモノラル音声を取り出す. 音声が無ければ None."""
    if shutil.which("ffmpeg") is None:
        return None
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    y = np.frombuffer(out, dtype=np.float32)
    if y.size < sr // 2 or np.max(np.abs(y)) < 1e-4:
        return None
    return y.astype(np.float64)


def _band_spectrogram(y: np.ndarray, sr: int, hop: int, n_bands: int, n_fft: int = 2048) -> np.ndarray:
    """対数周波数で n_bands に分けた対数エネルギー (T, n_bands). フレームの中心が時刻 i*hop."""
    y = np.pad(y, (n_fft // 2, n_fft // 2))
    n = 1 + (y.size - n_fft) // hop
    win = np.hanning(n_fft)
    edges = np.unique(np.geomspace(2, n_fft // 2 + 1, n_bands + 1).astype(int))
    out = np.zeros((n, edges.size - 1))
    for s in range(0, n, 4096):  # メモリを抑えるため分割して計算
        idx = np.arange(n_fft)[None, :] + hop * np.arange(s, min(n, s + 4096))[:, None]
        mag = np.abs(np.fft.rfft(y[idx] * win, axis=1))
        out[s:s + idx.shape[0]] = np.stack(
            [np.log1p(100 * mag[:, edges[i]:edges[i + 1]].mean(axis=1)) for i in range(edges.size - 1)], axis=1)
    return out


def _local_detrend(x: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width) / width
    return np.stack([x[:, i] - np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], axis=1)


def onset_bands(y: np.ndarray, sr: int = SR, hop: int = HOP, n_bands: int = 8) -> np.ndarray:
    """周波数帯ごとの立ち上がりの強さ (T, n_bands)（スペクトルフラックス）."""
    # 窓が長いと立ち上がりが早めに検出されるので短めの窓を使う
    spec = _band_spectrogram(y, sr, hop, n_bands, n_fft=1024)
    flux = np.maximum(np.diff(spec, axis=0, prepend=spec[:1]), 0)
    flux = np.maximum(_local_detrend(flux, 16), 0)
    return flux / (np.max(flux, axis=0, keepdims=True) + 1e-9)


def sync_features(y: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """音声同期用の特徴: 各時刻のスペクトルの形（音程・音色）の変化.

    音量やリズムの成分は一定テンポの曲だと拍ごとに似た形になり、拍単位の取り違えが起きやすいので除く。
    """
    spec = _band_spectrogram(y, sr, hop, 48)
    shape = spec - spec.mean(axis=1, keepdims=True)  # 各時刻のスペクトルの形（音量・リズムの成分を除く）
    shape = _local_detrend(shape, int(sr / hop))
    return (shape - shape.mean(axis=0)) / (shape.std(axis=0) + 1e-9)


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """立ち上がりの強さ（全帯域の合計, 0-1）."""
    env = onset_bands(y, sr, hop).sum(axis=1)
    return env / (np.max(env) + 1e-9)


def _beats_numpy(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """自己相関でテンポを求め、動的計画法で拍の位置を決める（librosa が無いとき用）."""
    env = onset_envelope(y, sr)
    fr = sr / HOP
    # 立ち上がりは 1 フレームの鋭いピークなので、少しぼかしてから周期を調べる
    k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    env_s = np.convolve(env, k / k.sum(), mode="same")
    ac = np.correlate(env_s, env_s, mode="full")[env.size - 1:]
    ac /= ac[0] + 1e-9
    lags = np.arange(ac.size)
    bpms = 60 * fr / np.maximum(lags, 1e-9)
    ok = (bpms >= 60) & (bpms <= 200)
    # 周期の 2 倍・4 倍でも一致するほど拍らしい（くし形）。120 BPM 付近を優先する重み（librosa と同様）
    comb = ac[ok] + 0.5 * ac[np.minimum(lags[ok] * 2, ac.size - 1)] + 0.25 * ac[np.minimum(lags[ok] * 4, ac.size - 1)]
    weight = np.exp(-0.5 * (np.log2(bpms[ok] / 120)) ** 2)
    period = int(lags[ok][np.argmax(comb * weight)])
    # DP: 拍間隔が period に近く、立ち上がりの強い点をつなぐ
    score = env.copy()
    back = np.full(env.size, -1)
    for i in range(env.size):
        lo, hi = i - 2 * period, i - period // 2
        if hi <= 0:
            continue
        j = np.arange(max(0, lo), hi)
        pen = -100 * (np.log((i - j) / period)) ** 2
        k = int(np.argmax(score[j] + pen))
        score[i] = env[i] + score[j[k]] + pen[k]
        back[i] = j[k]
    i = int(np.argmax(score[-2 * period:]) + max(0, env.size - 2 * period))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    # 前後の無音部分に付いた拍を除く
    strong = env[beats] >= 0.1 * np.median(env[beats])
    if strong.any():
        first, last = np.argmax(strong), len(strong) - np.argmax(strong[::-1])
        beats = beats[first:last]
    beats = beats * HOP / sr
    return 60 * fr / period, beats


def detect_beats(y: np.ndarray, sr: int = SR) -> tuple[float, np.ndarray]:
    """(BPM, 拍の時刻 [秒]) を返す."""
    try:
        import librosa
    except ImportError:
        return _beats_numpy(y, sr)
    tempo, frames = librosa.beat.beat_track(y=y.astype(np.float32), sr=sr, hop_length=HOP, units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return float(np.atleast_1d(tempo)[0]), np.asarray(times)


def estimate_offset(y_ref: np.ndarray, y_user: np.ndarray, sr: int = SR,
                    max_offset_s: float | None = None) -> tuple[float, float, float]:
    """音声の相互相関から 2 本の動画のずれを求める.

    returns: (offset, confidence, uniqueness)
      offset:     「user の時刻 = ref の時刻 + offset」となる秒数
      confidence: 相関ピークの鋭さ（目安: 4 以上なら信頼できる）
      uniqueness: 2 番目に高いピーク（±0.3 秒より外）との差の比率. 曲が単調で拍ずれの候補が
                  同じくらい高い場合は小さくなる（目安: 0.1 未満は拍単位で誤る恐れ）
    """
    A = sync_features(y_ref, sr)
    B = sync_features(y_user, sr)
    a, b = A[:, 0], B[:, 0]
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    # 帯域ごとの相関を足し合わせる
    corr = np.zeros(n)
    for i in range(A.shape[1]):
        corr += np.fft.irfft(np.fft.rfft(B[:, i], n) * np.conj(np.fft.rfft(A[:, i], n)), n)
    lags = np.concatenate([np.arange(0, b.size), np.arange(-(a.size - 1), 0)])
    corr = np.concatenate([corr[: b.size], corr[n - (a.size - 1):]])
    # 重なりが短いずれほど相関が小さくなるので、重なりの長さで正規化する
    overlap = np.minimum(a.size, b.size + np.minimum(lags, 0)) - np.maximum(lags, 0)
    overlap = np.minimum(overlap, np.minimum(a.size, b.size))
    fr = sr / HOP
    keep = overlap >= max(2 * fr, 0.5 * min(a.size, b.size))  # 短い方の半分以上が重なる候補だけ
    if max_offset_s is not None:
        keep &= np.abs(lags) <= max_offset_s * fr
    if not keep.any():
        return 0.0, 0.0, 0.0
    lags, corr = lags[keep], corr[keep] / overlap[keep]
    k = int(np.argmax(corr))
    conf = float((corr[k] - np.median(corr)) / (np.std(corr) + 1e-9))
    far = np.abs(lags - lags[k]) > 0.3 * fr
    second = float(np.max(corr[far])) if far.any() else 0.0
    base = float(np.median(corr))
    uniq = float((corr[k] - second) / (corr[k] - base + 1e-12)) if corr[k] > base else 0.0
    return float(lags[k] / fr), conf, uniq
