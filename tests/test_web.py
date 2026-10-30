"""ブラウザ（Pyodide）用の入口 dance_compare/web.py のテスト."""

import json
import subprocess
import sys

import numpy as np

from dance_compare.synth import synth_dancer


def _tracks_json(*dancers, n=None):
    n = n or len(dancers[0])
    people = {}
    for tid, s in enumerate(dancers, start=1):
        kp = s.keypoints[:n]
        people[str(tid)] = {"frames": list(range(n)), "kp": kp.reshape(-1).round(1).tolist(),
                            "sc": s.scores[:n].reshape(-1).tolist(),
                            "box": np.concatenate([kp.min(1), kp.max(1)], axis=1).reshape(-1).tolist()}
    return json.dumps({"fps": 30, "width": 640, "height": 480, "n_frames": n, "people": people})


def test_run_single_from_browser_json():
    from dance_compare.web import run_single

    a = synth_dancer(duration=12, center=(150, 260))
    b = synth_dancer(duration=12, center=(480, 260), elbow_bend=(4.0, 8.0, 60.0))
    out = json.loads(run_single(_tracks_json(a, b), 1, 2))
    assert out["scores"]["arms"] < out["scores"]["legs"]
    assert out["moments"] and out["moments"][0]["lines"]
    assert out["report"]["meta"]["mode"] == "single_video"


def test_run_two_without_audio():
    from dance_compare.web import run_two

    a = synth_dancer(duration=10)
    b = synth_dancer(duration=10, delay_s=0.2)
    out = json.loads(run_two(_tracks_json(a), _tracks_json(b)))
    assert out["report"]["meta"]["mode"] == "two_videos" and "timing" in out["scores"]


def test_web_entry_does_not_need_opencv():
    # ブラウザには OpenCV が無いので、cv2 を読めない状態でも動くこと
    code = ("import sys; sys.modules['cv2'] = None; import json; from dance_compare.synth import synth_dancer;"
            "from tests.test_web import _tracks_json; from dance_compare.web import run_single;"
            "a = synth_dancer(duration=8, center=(150, 260)); b = synth_dancer(duration=8, center=(480, 260));"
            "print(json.loads(run_single(_tracks_json(a, b), 1, 2))['scores']['overall'])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
