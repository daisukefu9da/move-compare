from __future__ import annotations

from typing import Protocol

from ..core.report import Report


class Narrator(Protocol):
    """Report（数値の指摘）を文章にするもの."""

    name: str

    def narrate(self, report: Report) -> str:
        """Markdown の文章を返す."""
        ...


def get_narrator(name: str = "rule", **kwargs) -> Narrator:
    if name == "rule":
        from .rule_based import RuleBasedNarrator
        return RuleBasedNarrator(**kwargs)
    if name == "gemini":
        from .gemini import GeminiNarrator
        return GeminiNarrator(**kwargs)
    raise ValueError(f"unknown narrator: {name}")
