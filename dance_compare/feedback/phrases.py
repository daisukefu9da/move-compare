"""Finding 1 件を日本語の短い文にする（ルールベースの言語化の部品）."""

from __future__ import annotations

from ..core.counts import fmt_time
from ..core.report import ASPECTS, Finding
from ..core.skeleton import JA_NAMES


def _side(part: str) -> str:
    return "右" if part.startswith("right") else "左"


def describe(f: Finding) -> str:
    """「何がどうずれているか」の部分."""
    v = abs(f.value)
    p = JA_NAMES.get(f.part or "", "")
    k, d = f.kind, f.direction
    if k == "angle":
        joint = f.part.split("_", 1)[1]
        if joint in ("elbow", "knee"):
            return (f"{p}が手本より約{v:.0f}°深く曲がっている" if d == "smaller"
                    else f"{p}の曲げが手本より約{v:.0f}°浅い（伸びている）")
        if joint == "shoulder":
            arm = _side(f.part) + "腕"
            return (f"{arm}の上がり方が手本より約{v:.0f}°低い" if d == "smaller"
                    else f"{arm}が手本より約{v:.0f}°高く上がっている")
        if joint == "hip":
            leg = _side(f.part) + "脚"
            return (f"{leg}の付け根の曲がり（脚の上げ・上体の前傾）が手本より約{v:.0f}°大きい" if d == "smaller"
                    else f"{leg}の付け根の曲がり（脚の上げ・上体の前傾）が手本より約{v:.0f}°小さい")
    if k == "early_late":
        return f"手本より約{v:.0f}ms{'遅れている' if d == 'late' else '早い'}"
    if k == "beat_offset":
        return f"拍に対して約{v:.0f}ms{'遅れ' if d == 'late' else '早い（走り気味）'}"
    if k == "amplitude":
        return f"{p}の動きが手本の約{v * 100:.0f}%と{'小さい' if d == 'smaller' else '大きい'}"
    if k == "sharpness":
        return f"{p}のキレ（加速・止めの鋭さ）が手本の約{v * 100:.0f}%しかない"
    if k == "level":
        return (f"腰の位置が手本より高い（沈み込みが足りない。胴の長さの約{v * 100:.0f}%）" if d == "higher"
                else f"腰の位置が手本より低い（沈み込みすぎ。胴の長さの約{v * 100:.0f}%）")
    if k == "facing":
        return (f"体の向きが手本より約{v:.0f}°多く回っている" if d == "more_turned"
                else f"体の向きが手本ほど回っていない（約{v:.0f}°不足）")
    if k == "lean":
        return f"上体が手本より画面の{'右' if d == 'right' else '左'}へ約{v:.0f}°傾いている"
    if k == "stance":
        return f"足幅が手本より{'広い' if d == 'wider' else '狭い'}（胴の長さの約{v * 100:.0f}%）"
    if k == "travel":
        return f"移動の量・方向が手本と違う（{'移動が足りない' if d == 'less' else '動きすぎ'}）"
    if k == "landing":
        foot = p or "足"
        return f"{foot}の着地が手本より約{v:.0f}ms{'遅い' if d == 'late' else '早い'}"
    return f"{ASPECTS.get(f.aspect, f.aspect)}: {f.kind} {f.direction} {f.value:.2f}"


def where(f: Finding) -> str:
    if f.t_start is None:
        return "全体"
    t = f"{fmt_time(f.t_start)}〜{fmt_time(f.t_end)}"
    return f"{f.label}（{t}）" if f.label else t
