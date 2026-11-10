"""Наложения поверх кадра: облачко-мысль с эмодзи над сэром Латом и «игровое окно».

Эмоции сэра Лата показываются только эмодзи в облачке (цветной шрифт
Noto Color Emoji): облачко плавно появляется за 0.2 с, висит ~1.8 с и плавно
исчезает. Сам сэр Лат при этом не меняет позу и мимику.
"""
from __future__ import annotations

import math
import os
import textwrap
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import font

EMOJI = {
    "surprised": "😲",
    "laughing": "😂",
    "scared": "😱",
    "angry": "😠",
    "thoughtful": "🤔",
    "intrigue": "👀",
}
BUBBLE_IN, BUBBLE_HOLD, BUBBLE_OUT = 0.2, 1.8, 0.3
EMOJI_FONTS = [
    os.getenv("SIRLAT_EMOJI_FONT", ""),
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]


@lru_cache(maxsize=1)
def emoji_font() -> ImageFont.FreeTypeFont:
    for p in EMOJI_FONTS:
        if p and Path(p).exists():
            return ImageFont.truetype(p, 109)          # цветной bitmap-шрифт: только размер 109
    raise RuntimeError("Не найден цветной шрифт эмодзи Noto Color Emoji "
                       "(apt install fonts-noto-color-emoji или SIRLAT_EMOJI_FONT=путь)")


@lru_cache(maxsize=16)
def emoji_image(ch: str, size: int) -> Image.Image:
    f = emoji_font()
    img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((80, 80), ch, font=f, embedded_color=True, anchor="mm")
    img = img.crop(img.getbbox() or (0, 0, 160, 160))
    k = size / max(img.size)
    return img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)


@lru_cache(maxsize=16)
def thought_bubble(ch: str, size: int, tail_left: bool = True) -> Image.Image:
    """Облачко-мысль (RGBA) размером ~size×size с эмодзи и «хвостиком» из кружков."""
    W = H = int(size * 1.5)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy, r = W * 0.56, H * 0.38, size * 0.38
    fill = (246, 240, 230, 236)
    # облако из нескольких кругов
    for ang in range(0, 360, 45):
        a = math.radians(ang)
        rr = r * (0.55 if ang % 90 else 0.62)
        x, y = cx + math.cos(a) * r * 0.62, cy + math.sin(a) * r * 0.48
        d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=fill)
    d.ellipse((cx - r * 0.85, cy - r * 0.7, cx + r * 0.85, cy + r * 0.7), fill=fill)
    # хвостик к голове
    sign = -1 if tail_left else 1
    for k, (dx, dy, rr) in enumerate(((0.62, 0.95, 0.14), (0.9, 1.22, 0.09))):
        x, y = cx + sign * r * dx, cy + r * dy
        d.ellipse((x - r * rr, y - r * rr, x + r * rr, y + r * rr), fill=fill)
    # мягкая тень и тёплый отсвет снизу, чтобы облачко не было «наклейкой»
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(img.split()[3].filter(ImageFilter.GaussianBlur(size * 0.04)).point(lambda v: v * 0.35))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(shadow, (int(size * 0.02), int(size * 0.03)))
    out.alpha_composite(img)
    em = emoji_image(ch, int(size * 0.5))
    out.alpha_composite(em, (int(cx - em.width / 2), int(cy - em.height / 2)))
    return out


def bubble_events(items, min_gap: float = 4.0) -> list[tuple[float, str]]:
    """(время, эмодзи) — облачко в начале фразы с эмоцией; без повторов подряд."""
    out, last_t, last_e = [], -99.0, None
    for it in items:
        e = EMOJI.get(it.phrase.emotion)
        if not e:
            continue
        t = it.start
        if out and t < out[-1][0] + BUBBLE_IN + BUBBLE_HOLD + BUBBLE_OUT:
            continue                                  # предыдущее облачко ещё видно
        if e == last_e and t - last_t < min_gap:
            continue
        out.append((t, e))
        last_t, last_e = t, e
    return out


def bubble_state(events, t: float):
    """(эмодзи, прозрачность 0..1, масштаб) для момента t или None."""
    for t0, e in events:
        u = t - t0
        if 0 <= u < BUBBLE_IN + BUBBLE_HOLD + BUBBLE_OUT:
            if u < BUBBLE_IN:
                k = u / BUBBLE_IN
                k = k * k * (3 - 2 * k)
                return e, k, 0.85 + 0.15 * k
            if u < BUBBLE_IN + BUBBLE_HOLD:
                return e, 1.0, 1.0
            k = 1 - (u - BUBBLE_IN - BUBBLE_HOLD) / BUBBLE_OUT
            return e, k * k * (3 - 2 * k), 1.0
    return None


def game_window(text: str, width: int, title: str = "СИСТЕМА") -> Image.Image:
    """Полупрозрачное «системное окно» с текстом (RGBA)."""
    scale = width / 900
    f_title = font(int(34 * scale), "mono")
    f_body = font(int(40 * scale), "sans")
    lines = textwrap.wrap(text, 30) or [""]
    lh = int(56 * scale)
    pad = int(40 * scale)
    head = int(70 * scale)
    h = head + pad + lh * len(lines) + pad
    img = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, h - 1), int(18 * scale), fill=(10, 30, 70, 215),
                        outline=(110, 200, 255, 255), width=max(2, int(4 * scale)))
    d.rectangle((int(4 * scale), int(4 * scale), width - int(5 * scale), head), fill=(30, 80, 150, 230))
    d.text((width / 2, head / 2 + 2), f"[ {title} ]", font=f_title, fill=(200, 240, 255), anchor="mm")
    y = head + pad
    for ln in lines:
        d.text((width / 2, y + lh / 2), ln, font=f_body, fill="white", anchor="mm")
        y += lh
    glow = img.filter(ImageFilter.GaussianBlur(int(10 * scale)))
    out = Image.new("RGBA", (width + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    out.alpha_composite(glow, (pad, pad))
    out.alpha_composite(img, (pad, pad))
    return out

