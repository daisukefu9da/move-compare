"""検出結果の一覧（どのカウント・どの部位・どの観点で・どれくらいズレたか）.

精度検証で「意図したズレが正しい箇所・観点で出ているか」を確認しやすいよう、
区間ごとの指摘と全体の傾向を 1 つの表にまとめる。
"""

from __future__ import annotations

import csv

from ..core.counts import fmt_time
from ..core.parts import GROUPS
from ..core.report import ASPECTS, Finding, Report
from ..core.skeleton import JA_NAMES
from .phrases import describe

DIRECTION_JA = {
    "smaller": "小さい", "larger": "大きい", "late": "遅い", "early": "早い", "weaker": "弱い",
    "higher": "高い", "lower": "低い", "more_turned": "回りすぎ", "less_turned": "回り不足",
    "right": "右へ", "left": "左へ", "wider": "広い", "narrower": "狭い", "less": "少ない", "more": "多い",
    "up": "上", "down": "下",
}
KIND_JA = {
    "angle": "関節角度", "early_late": "手本との時間差", "beat_offset": "拍とのズレ", "amplitude": "振れ幅",
    "sharpness": "キレ", "level": "腰の高さ", "facing": "体の向き", "lean": "上体の傾き", "stance": "足幅",
    "travel": "移動量", "landing": "着地タイミング",
    "head_tilt": "頭の傾き", "head_turn": "顔の向き", "head_shift": "頭の横スライド", "head_nod": "頭の上下",
}
COLUMNS = [  # (キー, CSV の見出し)
    ("scope", "範囲"), ("label", "カウント"), ("t_start", "開始[秒]"), ("t_end", "終了[秒]"),
    ("group_ja", "部位グループ"), ("aspect_ja", "観点"), ("kind_ja", "項目"), ("part_ja", "部位"), ("direction_ja", "向き"),
    ("value", "ズレ量"), ("unit", "単位"), ("severity", "優先度"), ("description", "説明"),
    ("group", "group"), ("aspect", "aspect"), ("kind", "kind"), ("part", "part"), ("direction", "direction"),
]


def format_value(f: Finding) -> str:
    v = f.value
    if f.unit == "deg":
        return f"{v:+.0f}°"
    if f.unit == "ms":
        return f"{v:+.0f}ms"
    if f.unit == "ratio":
        return f"{v * 100:.0f}%"
    if f.unit == "torso":
        return f"{v * 100:+.0f}%（胴長比）"
    return f"{v:.2f}"


def finding_row(f: Finding) -> dict:
    return {
        "scope": "全体" if f.is_global else "区間",
        "label": f.label or "",
        "t_start": None if f.t_start is None else round(f.t_start, 2),
        "t_end": None if f.t_end is None else round(f.t_end, 2),
        "group": f.group or "", "group_ja": GROUPS.get(f.group or "", ""),
        "aspect": f.aspect, "aspect_ja": ASPECTS.get(f.aspect, f.aspect),
        "kind": f.kind, "kind_ja": KIND_JA.get(f.kind, f.kind),
        "part": f.part or "", "part_ja": JA_NAMES.get(f.part or "") or ("頭" if f.group == "head" else "全身"),
        "direction": f.direction, "direction_ja": DIRECTION_JA.get(f.direction, f.direction),
        "value": round(f.value, 3), "value_text": format_value(f), "unit": f.unit,
        "severity": round(f.severity, 2), "description": describe(f),
    }


def findings_rows(report: Report) -> list[dict]:
    """全体の傾向 → 区間ごとの指摘（時刻順）の順で並べる."""
    rows = [finding_row(f) for f in report.tendencies]
    rows += [finding_row(f) for f in sorted(report.findings, key=lambda f: (f.t_start, -f.severity))]
    return rows


def write_csv(report: Report, path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # Excel で文字化けしないよう BOM 付き
        w = csv.writer(f)
        w.writerow([h for _, h in COLUMNS])
        for r in findings_rows(report):
            w.writerow(["" if r[k] is None else r[k] for k, _ in COLUMNS])


def markdown_table(report: Report) -> str:
    rows = findings_rows(report)
    if not rows:
        return "検出されたズレはありません。\n"
    out = ["| カウント | 時刻 | 部位グループ | 観点 | 部位 | 項目 | ズレ量 | 内容 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        when = "全体" if r["scope"] == "全体" else f"{fmt_time(r['t_start'])}〜{fmt_time(r['t_end'])}"
        out.append(f"| {r['label'] or '-'} | {when} | {r['group_ja']} | {r['aspect_ja']} | {r['part_ja']} | {r['kind_ja']} | "
                   f"{r['value_text']} | {r['description']} |")
    return "\n".join(out) + "\n"
