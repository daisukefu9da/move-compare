"""ダンス比較ツール.

- core/      : 解析ロジック（numpy のみに依存。UI・動画I/O・推論エンジンから独立）
- pose/      : 骨格推定バックエンド（YOLO-pose / MediaPipe）と人物追跡
- media/     : 動画・音声の入出力、拍検出、音声による同期
- feedback/  : 指摘の言語化（ルールベース / Gemini、差し替え可能）
- render/    : 骨格を重ねた比較動画の生成
- pipeline   : 上記をつなぐ高レベル API
"""

__version__ = "0.1.0"
