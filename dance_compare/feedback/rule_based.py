"""ルールベースの言語化（AI なし）."""

from __future__ import annotations

from collections import defaultdict

from ..core.report import ASPECTS, Finding, Report
from ..core.skeleton import JA_NAMES
from .phrases import describe, where


def group_findings(findings: list[Finding]) -> list[list[Finding]]:
    """同じ部位・同じ向きの指摘をまとめる（振付の繰り返しで同じ癖が何度も出る場合など）."""
    groups: dict[tuple, list[Finding]] = defaultdict(list)
    for f in findings:
        groups[(f.aspect, f.kind, f.part, f.direction)].append(f)
    out = [sorted(g, key=lambda f: f.t_start) for g in groups.values()]
    out.sort(key=lambda g: -(max(f.severity for f in g) * (1 + 0.15 * (len(g) - 1))))
    return out


def tendency_line(f: Finding) -> str:
    if f.kind == "beat_offset":
        ref = f.extra.get("ref_mean_ms")
        if ref is not None:
            s = (f"全体的に手本より拍に対して平均{abs(f.value - ref):.0f}ms"
                 f"{'遅れ気味' if f.direction == 'late' else '早め（走り気味）'}（拍とのズレ: 本人{f.value:+.0f}ms／手本{ref:+.0f}ms）")
        else:
            s = f"全体的に拍より平均{abs(f.value):.0f}ms{'遅れ気味' if f.direction == 'late' else '早め（走り気味）'}"
        land = f.extra.get("landing_mean_ms")
        if land is not None and abs(land) >= abs(f.value) * 0.8:
            s += f"。特にステップの着地（平均{land:+.0f}ms）"
        return s
    if f.kind == "early_late":
        return f"全体的に手本より平均{abs(f.value):.0f}ms{'遅れている' if f.direction == 'late' else '早い'}"
    if f.kind == "amplitude":
        return f"全体的に{JA_NAMES[f.part]}の動きが手本の約{f.value * 100:.0f}%と{'小さい' if f.direction == 'smaller' else '大きい'}"
    if f.kind == "sharpness":
        return f"全体的に{JA_NAMES[f.part]}のキレが手本の約{f.value * 100:.0f}%"
    return "全体的に" + describe(f)


class RuleBasedNarrator:
    name = "rule"

    def __init__(self, max_items: int | None = None):
        self.max_items = max_items

    def lines(self, report: Report) -> dict:
        """文章の部品（Gemini など他の Narrator への入力にも使う）."""
        n = self.max_items or report.meta.get("config", {}).get("max_findings", 12)
        items = []
        for g in group_findings(report.findings)[:n]:
            places = [where(f) for f in g]
            if len(places) > 3:
                place = "、".join(places[:3]) + f" ほか（計{len(places)}か所）"
            else:
                place = "、".join(places)
            items.append({"aspect": ASPECTS[g[0].aspect], "where": place, "what": describe(g[0]),
                          "severity": round(max(f.severity for f in g), 2), "count": len(g)})
        return {
            "scores": {ASPECTS.get(k, "総合"): (None if v is None else round(v)) for k, v in report.scores.items()},
            "tendencies": [tendency_line(f) for f in report.tendencies],
            "items": items,
        }

    def narrate(self, report: Report) -> str:
        p = self.lines(report)
        sc = report.scores
        out = ["## 比較結果", ""]
        if sc.get("overall") is not None:
            out.append(f"**総合スコア: {sc['overall']:.0f} / 100**")
        parts = [f"{ASPECTS[k]} {v:.0f}" for k, v in sc.items() if k != "overall" and v is not None]
        out += ["", "観点別: " + " ／ ".join(parts), ""]
        if p["tendencies"]:
            out += ["### 全体の傾向", ""] + [f"- {t}" for t in p["tendencies"]] + [""]
        if p["items"]:
            out += ["### 直すところ（優先度順）", ""]
            for i, it in enumerate(p["items"], 1):
                out.append(f"{i}. **{it['where']}**：{it['what']}")
            out.append("")
        if not p["tendencies"] and not p["items"]:
            out += ["大きなズレは見つかりませんでした。", ""]
        rh = report.rhythm
        if "user" in rh and rh["user"].get("count"):
            u = rh["user"]
            line = f"拍に合っている割合: {u['on_beat_ratio'] * 100:.0f}%（平均 {u['mean_ms']:+.0f}ms, ばらつき {u['std_ms']:.0f}ms）"
            if "ref" in rh and rh["ref"].get("count"):
                line += f"／手本 {rh['ref']['on_beat_ratio'] * 100:.0f}%（平均 {rh['ref']['mean_ms']:+.0f}ms）"
            out += ["### リズム", "", line, ""]
        return "\n".join(out)
