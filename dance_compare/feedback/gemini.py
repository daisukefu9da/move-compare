"""Gemini API による指摘の整理・優先順位づけ（任意）.

- 送るのは動画ではなく、ルールベースで作った数値の要約テキストだけ。
- API キーは環境変数 GEMINI_API_KEY で渡す（コードに書かない）。
- 無料枠では送信内容が Google の製品改善に使われる可能性がある点に注意。
- 失敗した場合はルールベースの文章を返す。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..core.report import Report
from .rule_based import RuleBasedNarrator

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """あなたはダンスの先生です。以下は、お手本とレビュー対象者のダンスを骨格推定で比較した結果です。
数値の要約をもとに、レビュー対象者へのアドバイスを日本語で書いてください。

条件:
- 最初に全体の評価を 2〜3 文で。
- 次に「まず直すべきこと」を最大 3 つ、優先度の高い順に。同じ原因から来ていそうな指摘はまとめる。
- 各項目には、場所（カウント表記）と、具体的な直し方（体の使い方・意識するポイント）を書く。
- 数値は参考程度に残してよいが、測定誤差があるので断定しすぎない。
- 要約に無いことは推測で付け足さない。
- Markdown で、全体で 400 字程度。

比較結果:
{summary}
"""


class GeminiNarrator:
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        self.fallback = RuleBasedNarrator()
        self.last_error: str | None = None

    def summary_text(self, report: Report) -> str:
        p = self.fallback.lines(report)
        return json.dumps(p, ensure_ascii=False, indent=1)

    def narrate(self, report: Report) -> str:
        base = self.fallback.narrate(report)
        if not self.api_key:
            self.last_error = "GEMINI_API_KEY が設定されていません"
            return base
        body = {"contents": [{"parts": [{"text": PROMPT.format(summary=self.summary_text(report))}]}]}
        req = urllib.request.Request(
            API_URL.format(model=self.model), data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (urllib.error.URLError, KeyError, IndexError, TimeoutError, json.JSONDecodeError) as e:
            self.last_error = f"Gemini の呼び出しに失敗しました: {e}"
            return base
        return f"## AI によるアドバイス（{self.model}）\n\n{text.strip()}\n\n---\n\n{base}"
