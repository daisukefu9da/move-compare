"""精度検証: テスト動画と「期待される検出結果」をセットで管理し、一括で照合する.

ケースファイル（TOML）の書き方は validation/README.md を参照。

    python -m dance_compare validate validation/cases.toml --out validation_out/
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field

import numpy as np

from .core.config import AnalysisConfig
from .core.counts import fmt_time
from .core.parts import GROUPS
from .core.report import ASPECTS, Finding, Report
from .core.skeleton import JA_NAMES, PoseSequence
from .feedback.table import KIND_JA, DIRECTION_JA, finding_row, format_value

FILTER_KEYS = ("aspect", "group", "kind", "part", "direction")
LEGACY_ASPECTS = ("steps", "level", "rhythm")  # 廃止した観点. 指定されていても観点では絞らない（kind などで照合）
CASE_KEYS = {
    "name", "mode", "ref", "user", "video", "ref_pose", "user_pose", "mirror", "bpm", "first_one", "no_beats",
    "audio_sync", "ref_id", "user_id", "ref_point", "user_point", "frame", "ref_start", "ref_end",
    "user_start", "user_end", "start", "end", "backend", "config", "count_tolerance", "fail_on_unexpected",
    "expect", "forbid", "note", "auto_range",
}


# ---------------------------------------------------------------------------
# ケースの読み込み
# ---------------------------------------------------------------------------

@dataclass
class Expectation:
    aspect: str | None = None
    group: str | None = None      # 部位グループ（legs / hips / head / arms / timing）
    kind: str | None = None
    part: str | list[str] | None = None
    direction: str | None = None
    start: str | None = None      # "2-3" = 2エイト目の3カウント
    end: str | None = None        # "2-4"（このカウントを含む）
    start_s: float | None = None  # 秒で指定する場合（手本動画の時刻）
    end_s: float | None = None
    min: float | None = None      # ズレ量の範囲（一覧の「単位」で, 符号付き）
    max: float | None = None
    note: str = ""

    @property
    def located(self) -> bool:
        return any(v is not None for v in (self.start, self.end, self.start_s, self.end_s))

    def matches_kind(self, f: Finding) -> bool:
        for k in FILTER_KEYS:
            want = getattr(self, k)
            if want is None or (k == "aspect" and want in LEGACY_ASPECTS):
                continue
            have = getattr(f, k)
            if isinstance(want, list):
                if have not in want:
                    return False
            elif have != want:
                return False
        return True

    def describe(self) -> str:
        parts = [GROUPS.get(self.group, self.group)] if self.group else []
        if self.aspect:
            parts.append(ASPECTS.get(self.aspect, self.aspect))
        if self.kind:
            parts.append(KIND_JA.get(self.kind, self.kind))
        if self.part:
            ps = self.part if isinstance(self.part, list) else [self.part]
            parts.append("/".join(JA_NAMES.get(p, p) for p in ps))
        if self.direction:
            parts.append(DIRECTION_JA.get(self.direction, self.direction))
        where = "全体"
        if self.start or self.end:
            where = f"{self.start or '?'}〜{self.end or self.start}カウント"
        elif self.start_s is not None or self.end_s is not None:
            where = f"{self.start_s or 0:.1f}〜{self.end_s or 0:.1f}秒"
        rng = ""
        if self.min is not None or self.max is not None:
            rng = f" [{'' if self.min is None else self.min}〜{'' if self.max is None else self.max}]"
        return f"{where}: " + " / ".join(parts) + rng


@dataclass
class Case:
    name: str
    params: dict
    expect: list[Expectation] = field(default_factory=list)
    forbid: list[Expectation] = field(default_factory=list)
    note: str = ""


def _expectation(d: dict) -> Expectation:
    d = dict(d)
    unknown = set(d) - set(Expectation.__dataclass_fields__)
    if unknown:
        raise ValueError(f"expect/forbid に不明な項目があります: {sorted(unknown)}")
    if not any(k in d for k in ("aspect", "group", "kind")):
        raise ValueError("expect/forbid には aspect（観点）・group（部位）・kind（項目）のどれかが必要です")
    if "aspect" in d and d["aspect"] not in ASPECTS:
        raise ValueError(f"aspect は {list(ASPECTS)} のいずれか: {d['aspect']}")
    if "group" in d and d["group"] not in GROUPS:
        raise ValueError(f"group は {list(GROUPS)} のいずれか: {d['group']}")
    return Expectation(**d)


def load_cases(path: str) -> tuple[dict, list[Case]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    settings = dict(data.get("settings", {}))
    settings.setdefault("base_dir", os.path.dirname(os.path.abspath(path)))
    if not os.path.isabs(settings["base_dir"]):
        settings["base_dir"] = os.path.join(os.path.dirname(os.path.abspath(path)), settings["base_dir"])
    cases = []
    for c in data.get("case", []):
        unknown = set(c) - CASE_KEYS
        if unknown:
            raise ValueError(f"ケース {c.get('name')} に不明な項目があります: {sorted(unknown)}")
        params = {k: v for k, v in c.items() if k not in ("expect", "forbid", "name", "note")}
        cases.append(Case(c.get("name", f"case{len(cases) + 1}"), params,
                          [_expectation(e) for e in c.get("expect", [])],
                          [_expectation(e) for e in c.get("forbid", [])], c.get("note", "")))
    return settings, cases


# ---------------------------------------------------------------------------
# 解析の実行
# ---------------------------------------------------------------------------

def run_case(case: Case, settings: dict, use_cache: bool = True):
    from .pipeline import Pick, Rhythm, analyze_single_video, analyze_two_videos
    from .pose.base import TrackSet

    p = {**{k: v for k, v in settings.items() if k in CASE_KEYS}, **case.params}
    base = settings["base_dir"]

    def path(key):
        v = p.get(key)
        return None if v is None else (v if os.path.isabs(v) else os.path.join(base, v))

    cfg = AnalysisConfig.load(path("config")) if p.get("config") else AnalysisConfig()
    rhythm = Rhythm(bpm=p.get("bpm"), first_one=p.get("first_one"), detect=not p.get("no_beats", False))
    backend = p.get("backend", "yolo")

    def pick(who):
        pt = p.get(f"{who}_point")
        return Pick(p.get(f"{who}_id"), tuple(pt) if pt else None, p.get("frame", 0))

    def tracks(key):
        f = path(key)
        return TrackSet.from_sequence(PoseSequence.load_json(f)) if f else None

    if p.get("mode", "two_videos") == "single":
        return analyze_single_video(path("video"), pick("ref"), pick("user"), cfg, backend=backend,
                                    mirror=p.get("mirror", False), time_range=(p.get("start"), p.get("end")),
                                    auto_range=p.get("auto_range", True), rhythm=rhythm, use_cache=use_cache)
    return analyze_two_videos(
        path("ref"), path("user"), cfg, backend=backend, mirror=p.get("mirror", False),
        ref_range=(p.get("ref_start"), p.get("ref_end")), user_range=(p.get("user_start"), p.get("user_end")),
        audio_sync=p.get("audio_sync", True), rhythm=rhythm, ref_pick=pick("ref"), user_pick=pick("user"),
        use_cache=use_cache, ref_tracks=tracks("ref_pose"), user_tracks=tracks("user_pose"))


# ---------------------------------------------------------------------------
# 照合
# ---------------------------------------------------------------------------

def _count_pos(s: str) -> float:
    """"2-3" -> 拍番号（1エイト目の1 が 0）. "2-3&" は裏拍."""
    s = s.strip()
    half = s.endswith("&")
    e, c = s.rstrip("&").split("-")
    return (int(e) - 1) * 8 + (int(c) - 1) + (0.5 if half else 0.0)


def expected_range(e: Expectation, report: Report, tolerance_counts: float) -> tuple[float, float]:
    """期待する区間 [t0, t1)（手本動画の時刻）. 前後に tolerance_counts カウントの余裕を持たせる."""
    if e.start_s is not None or e.end_s is not None:
        g = report.count_grid()
        pad = tolerance_counts * (g.interval if g else 0.5)
        t0 = e.start_s if e.start_s is not None else -np.inf
        t1 = e.end_s if e.end_s is not None else np.inf
        return t0 - pad, t1 + pad
    g = report.count_grid()
    if g is None:
        raise ValueError("拍が検出されていないためカウントで照合できません（start_s / end_s を使ってください）")
    p0 = _count_pos(e.start or e.end)
    p1 = _count_pos(e.end or e.start) + 1  # 終わりのカウントを含む
    return float(g.time_of(p0 - tolerance_counts)), float(g.time_of(p1 + tolerance_counts))


@dataclass
class Check:
    expectation: Expectation
    status: str                 # "OK" / "値が範囲外" / "場所違い" / "未検出" / "NG（検出された）"
    matched: list[Finding] = field(default_factory=list)
    passed: bool = False


def _in_value_range(e: Expectation, f: Finding) -> bool:
    return (e.min is None or f.value >= e.min) and (e.max is None or f.value <= e.max)


def check_expectation(e: Expectation, report: Report, tol: float) -> Check:
    same = [f for f in report.tendencies + report.findings if e.matches_kind(f)]
    if not same:
        return Check(e, "未検出")
    if e.located:
        t0, t1 = expected_range(e, report, tol)
        here = [f for f in same if not f.is_global and f.t_start < t1 and f.t_end > t0]
        if not here:
            return Check(e, "場所違い", same)
    else:
        # 全体の傾向として出ているのが本来の形. 区間の指摘としてしか出ていなくても検出とみなす
        here = [f for f in same if f.is_global] or same
    ok = [f for f in here if _in_value_range(e, f)]
    if ok:
        return Check(e, "OK", ok, True)
    return Check(e, "値が範囲外", here)


def check_forbid(e: Expectation, report: Report, tol: float) -> Check:
    hits = [f for f in report.tendencies + report.findings if e.matches_kind(f)]
    if e.located and hits:
        t0, t1 = expected_range(e, report, tol)
        hits = [f for f in hits if f.is_global or (f.t_start < t1 and f.t_end > t0)]
    return Check(e, "NG（検出された）" if hits else "OK", hits, not hits)


@dataclass
class CaseResult:
    case: Case
    checks: list[Check]
    forbids: list[Check]
    unexpected: list[Finding]
    report: Report | None = None
    error: str | None = None
    out_dir: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        ok = all(c.passed for c in self.checks) and all(c.passed for c in self.forbids)
        if self.case.params.get("fail_on_unexpected"):
            ok = ok and not self.unexpected
        return ok


def evaluate(case: Case, report: Report, tol: float) -> CaseResult:
    checks = [check_expectation(e, report, tol) for e in case.expect]
    forbids = [check_forbid(e, report, tol) for e in case.forbid]
    # どの期待にも当てはまらない検出（誤検出の候補）
    unexpected = [f for f in report.tendencies + report.findings
                  if not any(e.matches_kind(f) for e in case.expect)]
    return CaseResult(case, checks, forbids, unexpected, report)


def validate(path: str, out_dir: str, render: bool = False, use_cache: bool = True,
             only: list[str] | None = None, log=print) -> list[CaseResult]:
    from .feedback import get_narrator
    from .pipeline import save_outputs

    settings, cases = load_cases(path)
    results = []
    for case in cases:
        if only and case.name not in only:
            continue
        log(f"▶ {case.name}")
        tol = float(case.params.get("count_tolerance", settings.get("count_tolerance", 1)))
        case_dir = os.path.join(out_dir, _safe(case.name))
        try:
            res = run_case(case, settings, use_cache)
            cr = evaluate(case, res.report, tol)
            save_outputs(res, case_dir, get_narrator("rule"), render=render)
            cr.out_dir = case_dir
        except Exception as ex:  # 1 ケースの失敗で全体を止めない
            cr = CaseResult(case, [], [], [], error=f"{type(ex).__name__}: {ex}")
        results.append(cr)
        log(f"  {'PASS' if cr.passed else 'FAIL'}" + (f"  {cr.error}" if cr.error else ""))
    write_summary(results, out_dir)
    return results


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# ---------------------------------------------------------------------------
# 結果の出力
# ---------------------------------------------------------------------------

def _finding_brief(f: Finding) -> str:
    where = "全体" if f.is_global else f"{f.label or ''}（{fmt_time(f.t_start)}〜{fmt_time(f.t_end)}）"
    part = JA_NAMES.get(f.part or "") or GROUPS.get(f.group or "", "全身")
    return f"{where} {ASPECTS.get(f.aspect, f.aspect)}/{KIND_JA.get(f.kind, f.kind)}/{part} {format_value(f)}"


def summary_markdown(results: list[CaseResult]) -> str:
    out = ["# 精度検証の結果", "",
           "| ケース | 判定 | 期待どおり検出 | 検出されるべきでない | 想定外の検出 |", "|---|---|---|---|---|"]
    for r in results:
        if r.error:
            out.append(f"| {r.case.name} | ❌ エラー | - | - | - |")
            continue
        exp = f"{sum(c.passed for c in r.checks)}/{len(r.checks)}"
        fb = f"{sum(c.passed for c in r.forbids)}/{len(r.forbids)}" if r.forbids else "-"
        out.append(f"| {r.case.name} | {'✅' if r.passed else '❌'} | {exp} | {fb} | {len(r.unexpected)} |")
    total = len(results)
    out += ["", f"**{sum(r.passed for r in results)}/{total} ケースが合格**", ""]
    for r in results:
        out += [f"## {r.case.name} {'✅' if r.passed else '❌'}", ""]
        if r.case.note:
            out += [r.case.note, ""]
        if r.error:
            out += [f"エラー: `{r.error}`", ""]
            continue
        if r.checks:
            out += ["| 期待 | 結果 | 実際の検出 |", "|---|---|---|"]
            for c in r.checks:
                got = "<br>".join(_finding_brief(f) for f in c.matched[:3]) or "-"
                out.append(f"| {c.expectation.describe()} | {'✅' if c.passed else '❌'} {c.status} | {got} |")
            out.append("")
        if r.forbids:
            out += ["| 検出されるべきでない | 結果 | 実際の検出 |", "|---|---|---|"]
            for c in r.forbids:
                got = "<br>".join(_finding_brief(f) for f in c.matched[:3]) or "-"
                out.append(f"| {c.expectation.describe()} | {'✅' if c.passed else '❌'} {c.status} | {got} |")
            out.append("")
        if r.unexpected:
            out += ["想定外の検出（誤検出の候補）:", ""] + [f"- {_finding_brief(f)}" for f in r.unexpected[:15]]
            if len(r.unexpected) > 15:
                out.append(f"- ほか {len(r.unexpected) - 15} 件")
            out.append("")
        if r.out_dir:
            out += [f"詳細: `{r.out_dir}/report.md`, `{r.out_dir}/findings.csv`", ""]
    return "\n".join(out)


def write_summary(results: list[CaseResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation.md"), "w", encoding="utf-8") as f:
        f.write(summary_markdown(results))
    data = []
    for r in results:
        data.append({
            "name": r.case.name, "passed": r.passed, "error": r.error,
            "expect": [{"expect": c.expectation.describe(), "status": c.status, "passed": c.passed,
                        "matched": [finding_row(f) for f in c.matched]} for c in r.checks],
            "forbid": [{"forbid": c.expectation.describe(), "status": c.status, "passed": c.passed,
                        "matched": [finding_row(f) for f in c.matched]} for c in r.forbids],
            "unexpected": [finding_row(f) for f in r.unexpected],
        })
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 合成データによるサンプルの検証セット（仕組みの動作確認・回帰テスト用）
# ---------------------------------------------------------------------------

SYNTH_CASES = '''# 合成した棒人間による検証セット（make-validation-set で生成）
# 手本: 120 BPM, 拍は 0.5 秒から（1エイト目の1 = 0.5 秒）。足は拍ごとに左右交互に着地する。
# レビュー対象の動画は手本より 1.0 秒遅れて撮影を始めた想定（音声で同期される）。
# 骨格は *_pose.json を使う（骨格推定はしない）。

[settings]
count_tolerance = 1   # カウント範囲の前後 1 カウントまでは「正しい場所」とみなす
first_one = 0.5       # 手本動画で「1エイト目の1」の時刻（自動検出だと 1 拍ずれることがあるので明示）

[[case]]
name = "elbow_bent"
note = "2エイト目の1〜4カウントだけ、右肘を 45° 曲げたままにする"
ref = "ref.mp4"
ref_pose = "ref_pose.json"
user = "elbow_bent.mp4"
user_pose = "elbow_bent_pose.json"
  [[case.expect]]
  aspect = "shape"
  kind = "angle"
  part = "right_elbow"
  direction = "smaller"
  start = "2-1"
  end = "2-4"
  min = -70
  max = -25
  [[case.forbid]]
  kind = "early_late"
  [[case.forbid]]
  aspect = "shape"
  part = "left_elbow"

[[case]]
name = "half_beat_late"
note = "全体を半拍（250ms）遅らせて踊る。裏拍(&)に乗っているので「拍とのズレ」では見えず、手本との時間差で検出されるべき"
ref = "ref.mp4"
ref_pose = "ref_pose.json"
user = "half_beat_late.mp4"
user_pose = "half_beat_late_pose.json"
  [[case.expect]]
  aspect = "timing"
  kind = "early_late"
  direction = "late"
  min = 180
  max = 320
  [[case.expect]]
  group = "legs"
  kind = "landing"
  direction = "late"
  min = 180
  max = 320
  [[case.forbid]]
  aspect = "shape"

[[case]]
name = "landing_late"
note = "3エイト目の1〜8カウントだけ、足の着地を 150ms 遅らせる（上半身は手本どおり）"
ref = "ref.mp4"
ref_pose = "ref_pose.json"
user = "landing_late.mp4"
user_pose = "landing_late_pose.json"
  [[case.expect]]
  group = "legs"
  kind = "landing"
  direction = "late"
  start = "3-1"
  end = "3-8"
  min = 100
  max = 220
  [[case.forbid]]
  aspect = "shape"
  [[case.forbid]]
  kind = "landing"
  start = "1-1"
  end = "2-6"

[[case]]
name = "small_amplitude"
note = "腕の上げ角の振れ幅を手本の 60% に抑える（手先の移動距離では 35〜75% 程度になる）。腕が上がりきらないので肩の角度の差も出る"
ref = "ref.mp4"
ref_pose = "ref_pose.json"
user = "small_amplitude.mp4"
user_pose = "small_amplitude_pose.json"
  [[case.expect]]
  aspect = "amplitude"
  part = "right_wrist"
  direction = "smaller"
  min = 0.3
  max = 0.8
  [[case.expect]]
  aspect = "amplitude"
  part = "left_wrist"
  direction = "smaller"
  min = 0.3
  max = 0.8
  [[case.expect]]
  aspect = "shape"
  part = ["left_shoulder", "right_shoulder"]
  direction = "smaller"
  [[case.forbid]]
  kind = "early_late"
  [[case.forbid]]
  aspect = "sharpness"
'''


def make_synthetic_set(out_dir: str) -> str:
    """合成の検証セット（動画・骨格 JSON・cases.toml）を作り、cases.toml のパスを返す."""
    from .synth import synth_beats, synth_dancer, write_synth_video

    os.makedirs(out_dir, exist_ok=True)
    dur, offset = 16.0, 1.0
    beats = synth_beats(dur + offset + 1)
    common = dict(steps=True, noise=1.0, size=100, center=(300, 240))
    ref = synth_dancer(duration=dur, steps=True, noise=1.0, seed=0)
    ref.save_json(os.path.join(out_dir, "ref_pose.json"))
    write_synth_video(ref, os.path.join(out_dir, "ref.mp4"), beats)
    users = {
        "elbow_bent": dict(elbow_bend=(4.5 + offset, 6.5 + offset, 45)),
        "half_beat_late": dict(delay_s=offset + 0.25),
        "landing_late": dict(step_delay=(8.3 + offset, 12.3 + offset, 0.15)),
        "small_amplitude": dict(amplitude=0.6),
    }
    for name, kw in users.items():
        kw.setdefault("delay_s", offset)
        u = synth_dancer(duration=dur + offset, seed=1, **common, **kw)
        u.save_json(os.path.join(out_dir, f"{name}_pose.json"))
        write_synth_video(u, os.path.join(out_dir, f"{name}.mp4"), beats + offset)
    path = os.path.join(out_dir, "cases.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SYNTH_CASES)
    return path
