"""ルールベースの言語化（AI なし）."""

from __future__ import annotations

from collections import defaultdict

from ..core.counts import fmt_time
from ..core.parts import GROUPS
from ..core.report import ASPECTS, Finding, Report
from ..core.skeleton import JA_NAMES
from .phrases import describe, where

CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def group_findings(findings: list[Finding]) -> list[list[Finding]]:
    """同じ部位・同じ向きの指摘をまとめる（振付の繰り返しで同じ癖が何度も出る場合など）."""
    groups: dict[tuple, list[Finding]] = defaultdict(list)
    for f in findings:
        groups[(f.aspect, f.kind, f.part, f.direction)].append(f)
    out = [sorted(g, key=lambda f: f.t_start) for g in groups.values()]
    out.sort(key=lambda g: -(max(f.severity for f in g) * (1 + 0.15 * (len(g) - 1))))
    return out


def tendency_line(f: Finding) -> str:
    if f.kind == "early_late":
        return f"全体的に手本より平均{abs(f.value):.0f}ms{'遅れている' if f.direction == 'late' else '早い'}"
    if f.kind == "amplitude":
        return f"全体的に{JA_NAMES[f.part]}の動きが手本の約{f.value * 100:.0f}%と{'小さい' if f.direction == 'smaller' else '大きい'}"
    if f.kind == "sharpness":
        return f"全体的に{JA_NAMES[f.part]}のキレが手本の約{f.value * 100:.0f}%"
    if f.kind == "landing":
        return f"全体的に足の着地が手本より平均{abs(f.value):.0f}ms{'遅い' if f.direction == 'late' else '早い'}"
    return "全体的に" + describe(f)


def moment_title(m: dict) -> str:
    """「① 0:32.4 付近」（カウント表示を使う場合は「① 2エイト目の3カウント（0:32.4）」）."""
    mark = CIRCLED[m["rank"] - 1] if m["rank"] <= len(CIRCLED) else f"{m['rank']}."
    if m.get("label"):
        return f"{mark} {m['label']}（{fmt_time(m['t'])}）"
    return f"{mark} {fmt_time(m['t'])} 付近"


def moment_lines(m: dict) -> list[str]:
    lines = [describe(Finding(**f)) for f in m["items"]]
    reps = m.get("repeats") or []
    if reps:
        shown = "、".join(reps[:3]) + (" ほか" if len(reps) > 3 else "")
        lines.append(f"同じズレが他に{len(reps)}か所あります（{shown}）")
    return lines


class RuleBasedNarrator:
    name = "rule"

    def __init__(self, max_items: int | None = None):
        self.max_items = max_items

    def lines(self, report: Report) -> dict:
        """文章の部品（Gemini など他の Narrator への入力にも使う）."""
        n = self.max_items or report.meta.get("config", {}).get("max_findings", 12)
        others = []
        for g in group_findings(report.findings)[:n]:
            places = [where(f) for f in g]
            place = "、".join(places[:3]) + (f" ほか（計{len(places)}か所）" if len(places) > 3 else "")
            others.append({"aspect": ASPECTS.get(g[0].aspect, g[0].aspect), "part": GROUPS.get(g[0].group or "", ""),
                           "where": place, "what": describe(g[0]), "severity": round(max(f.severity for f in g), 2),
                           "count": len(g)})
        return {
            "scores": {("総合" if k == "overall" else GROUPS.get(k, k)): (None if v is None else round(v))
                       for k, v in report.scores.items()},
            "moments": [{"when": moment_title(m), "what": moment_lines(m)} for m in report.moments],
            "tendencies": [tendency_line(f) for f in report.tendencies],
            "all_findings": others,
        }

    def narrate(self, report: Report) -> str:
        p = self.lines(report)
        sc = report.scores
        out = ["## 比較結果", ""]
        if sc.get("overall") is not None:
            out.append(f"**総合スコア: {sc['overall']:.0f} / 100**")
        parts = [f"{GROUPS[k]} {sc[k]:.0f}" for k in GROUPS if sc.get(k) is not None]
        out += ["", "部位別: " + " ／ ".join(parts), ""]
        if report.moments:
            out += ["### 直すところ（ズレの大きい瞬間）", ""]
            for m in report.moments:
                out.append(f"**{moment_title(m)}**")
                if m.get("image"):
                    out += ["", f'<img src="{m["image"]}" width="480">', ""]
                out += [f"- {line}" for line in moment_lines(m)]
                out.append("")
        else:
            out += ["大きなズレは見つかりませんでした。", ""]
        if p["tendencies"]:
            out += ["### 全体の傾向", ""] + [f"- {t}" for t in p["tendencies"]] + [""]
        return "\n".join(out)
