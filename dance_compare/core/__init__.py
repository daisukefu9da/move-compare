"""解析コア. numpy のみに依存し、ファイルI/Oや推論エンジンを含まない.

将来ブラウザ/スマホ側で骨格推定を行い、関節座標だけをこのモジュールに渡す
（あるいはこのロジックを別言語へ移植する）ことを想定している。
"""

from .skeleton import PoseSequence, KEYPOINTS
from .analyzer import compare
from .report import Report, Finding

__all__ = ["PoseSequence", "KEYPOINTS", "compare", "Report", "Finding"]
