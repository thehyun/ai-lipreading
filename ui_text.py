"""OpenCV 창에 한글/일본어를 그리기 위한 도우미 (cv2.putText는 영어만 지원)."""
from __future__ import annotations

import os
import platform

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = {
    "Windows": {
        "ja": ["C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc"],
        "default": ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"],
    },
    "Darwin": {
        "ja": ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"],
        "default": ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"],
    },
    "Linux": {
        "ja": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
        "default": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
    },
}
_cache: dict = {}


def get_font(size: int, lang: str = "ko"):
    key = (size, lang)
    if key in _cache:
        return _cache[key]
    table = _FONT_CANDIDATES.get(platform.system(), _FONT_CANDIDATES["Linux"])
    paths = (table.get("ja", []) if lang == "ja" else []) + table["default"] + table.get("ja", [])
    font = None
    for p in paths:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
        print("[경고] 한글/일본어 글꼴을 찾지 못했어요. 글자가 네모로 보일 수 있어요. ui_text.py에 글꼴 경로를 추가하세요.")
    _cache[key] = font
    return font


def draw_lines(bgr: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], lang: str = "ko",
               origin=(12, 10), size: int = 24, box: bool = True) -> np.ndarray:
    """lines: [(문자열, (B,G,R)), ...] 를 왼쪽 위부터 차례로 그린다."""
    img = Image.fromarray(bgr[:, :, ::-1])
    draw = ImageDraw.Draw(img, "RGBA")
    font = get_font(size, lang)
    x, y = origin
    gap = int(size * 1.35)
    if box and lines:
        width = max(draw.textlength(t, font=font) for t, _ in lines)
        draw.rectangle([x - 8, y - 6, x + width + 8, y + gap * len(lines)], fill=(0, 0, 0, 150))
    for text, (b, g, r) in lines:
        draw.text((x, y), text, font=font, fill=(r, g, b))
        y += gap
    return np.array(img)[:, :, ::-1].copy()
