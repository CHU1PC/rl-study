from typing import Final

import matplotlib as mpl
from matplotlib import font_manager

# スライドと同じ 3 色. 濃紺=学習した価値, アクセント=罰や失敗, 本文=文字.
MAIN: Final = "#1F3864"
ACCENT: Final = "#B45309"
INK: Final = "#333333"
SUB: Final = "#595959"
WALL: Final = "#BFBFBF"

JAPANESE_FONTS: Final = ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]


def use_japanese_font() -> None:
    """matplotlib の既定フォントを, 入っている日本語フォントの最初の 1 つにする."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    mpl.rcParams["font.family"] = next((name for name in JAPANESE_FONTS if name in installed), "DejaVu Sans")
    mpl.rcParams["axes.unicode_minus"] = False
