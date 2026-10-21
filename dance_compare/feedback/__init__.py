"""指摘の言語化. Narrator を差し替えることで、ルールベース / Gemini / 将来の別AI を切り替える."""

from .base import Narrator, get_narrator

__all__ = ["Narrator", "get_narrator"]
