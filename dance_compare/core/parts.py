"""部位（足・腰・頭・腕）と観点の定義. 指摘の種類をどの部位・観点に入れるかを決める."""

from __future__ import annotations

from .skeleton import ANGLE_DEFS, END_EFFECTORS, KP

# スコア・重み付けの単位
GROUPS = {
    "legs": "足",
    "hips": "腰（体幹）",
    "head": "頭",
    "arms": "腕",
    "timing": "タイミング",
}

# 観点（何がずれているか）
ASPECT_OF_KIND = {
    "angle": "shape", "level": "shape", "lean": "shape", "facing": "shape", "stance": "shape",
    "head_tilt": "shape", "head_turn": "shape", "head_shift": "shape", "head_nod": "shape",
    "amplitude": "amplitude", "travel": "amplitude",
    "sharpness": "sharpness",
    "early_late": "timing", "landing": "timing",
}

HEAD_KINDS = ("head_tilt", "head_turn", "head_shift", "head_nod")


def group_of(kind: str, part: str | None) -> str:
    """指摘の種類と部位名から、部位グループを決める."""
    if kind in HEAD_KINDS:
        return "head"
    if kind in ("level", "lean", "facing", "travel"):
        return "hips"
    if kind in ("landing", "stance"):
        return "legs"
    if kind == "early_late":
        return "timing"
    if part:
        if part.endswith(("elbow", "shoulder", "wrist")):
            return "arms"
        if part.endswith(("hip", "knee", "ankle")):
            return "legs"
    return "hips"


def joints_of(kind: str, part: str | None) -> list[int]:
    """静止画で印を付ける関節（レビュー対象から見た左右）."""
    if kind == "angle" and part in ANGLE_DEFS:
        return [ANGLE_DEFS[part][1]]
    if kind in ("amplitude", "sharpness") and part in END_EFFECTORS:
        return [END_EFFECTORS[part]]
    if kind == "landing":
        return [KP[part]] if part in KP else [KP["left_ankle"], KP["right_ankle"]]
    if kind == "stance":
        return [KP["left_ankle"], KP["right_ankle"]]
    if kind in ("level", "travel"):
        return [KP["left_hip"], KP["right_hip"]]
    if kind == "lean":
        return [KP["left_shoulder"], KP["right_shoulder"]]
    if kind == "facing":
        return [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    if kind in HEAD_KINDS:
        return [KP["nose"], KP["left_ear"], KP["right_ear"]]
    return []
