"""크로스플랫폼 한글 폰트 설정."""
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


_FONT_CANDIDATES = {
    "Darwin": [
        ("AppleGothic", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        ("Apple SD Gothic Neo", "/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ],
    "Windows": [
        ("Malgun Gothic", "C:/Windows/Fonts/malgun.ttf"),
    ],
    "Linux": [
        ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        ("Noto Sans CJK KR", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ],
}


def setup_korean_font() -> str:
    """matplotlib 한글 폰트 적용. 실패 시 기본값."""
    system = platform.system()
    for name, _ in _FONT_CANDIDATES.get(system, []):
        try:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return matplotlib.rcParams["font.family"]


def get_font_path() -> str | None:
    """WordCloud용 폰트 파일 경로."""
    system = platform.system()
    for _, path in _FONT_CANDIDATES.get(system, []):
        if Path(path).exists():
            return path
    return None
