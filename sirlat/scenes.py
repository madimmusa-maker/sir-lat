"""Обобщённые сцены, нарисованные кодом: подземелье, таверна, город,
силуэт героя, тень монстра и «игровое окно» с текстом.

Никаких персонажей и страниц из манхвы — только абстрактные декорации.
Каждая сцена рисуется чуть больше кадра, чтобы по ней можно было плавно
«панорамировать» камерой.
"""
from __future__ import annotations

import math
import random
import textwrap
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import font

SCENE_KEYS = ["dungeon", "tavern", "city", "hero", "monster", "game"]
SCENE_ALIASES = {
    "подземелье": "dungeon", "данж": "dungeon", "пещера": "dungeon",
    "таверна": "tavern", "трактир": "tavern",
    "город": "city", "улица": "city",
    "герой": "hero", "силуэт": "hero",
    "монстр": "monster", "тень": "monster", "босс": "monster",
    "окно": "game", "игровое окно": "game", "система": "game",
}
OVERSCAN = 1.12


def _gradient(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, dtype=np.float32)[None, None, :]
    b = np.array(bottom, dtype=np.float32)[None, None, :]
    arr = (a * (1 - t) + b * t).repeat(w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _glow(img, xy, radius, color, strength=1.0):
    lay = Image.new("RGB", img.size, (0, 0, 0))
    d = ImageDraw.Draw(lay)
    x, y = xy
    for i in range(8, 0, -1):
        r = radius * i / 8
        k = (1 - i / 9) * strength
        d.ellipse((x - r, y - r, x + r, y + r),
                  fill=tuple(int(c * k) for c in color))
    lay = lay.filter(ImageFilter.GaussianBlur(radius / 4))
    return Image.fromarray(np.clip(np.asarray(img, np.int16) + np.asarray(lay, np.int16), 0, 255)
                           .astype(np.uint8))


def _vignette(img, strength=0.55):
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - strength * (d ** 2) / 2, 0.15, 1)[..., None]
    return Image.fromarray((np.asarray(img, np.float32) * m).astype(np.uint8))


def _torch(d, x, y, s):
    d.rectangle((x - 6 * s, y, x + 6 * s, y + 60 * s), fill=(70, 45, 25))
    d.polygon([(x - 18 * s, y), (x + 18 * s, y), (x + 10 * s, y + 14 * s), (x - 10 * s, y + 14 * s)],
              fill=(60, 60, 70))
    d.ellipse((x - 16 * s, y - 40 * s, x + 16 * s, y + 4 * s), fill=(255, 150, 40))
    d.ellipse((x - 8 * s, y - 26 * s, x + 8 * s, y + 2 * s), fill=(255, 230, 140))


def dungeon(w, h, seed=0):
    rnd = random.Random(seed)
    img = _gradient(w, h, (18, 20, 30), (8, 8, 14))
    d = ImageDraw.Draw(img)
    s = h / 1080
    # каменная кладка
    bh = int(70 * s)
    for row, y in enumerate(range(0, int(h * 0.78), bh)):
        off = (row % 2) * int(80 * s)
        for x in range(-off, w, int(160 * s)):
            c = rnd.randint(34, 50)
            d.rectangle((x + 3, y + 3, x + 160 * s - 3, y + bh - 3), fill=(c, c + 2, c + 10))
    # арка-проход в центре
    aw, ah = w * 0.22, h * 0.55
    ax, ay = w / 2 - aw / 2, h * 0.78 - ah
    d.rectangle((ax, ay + aw / 2, ax + aw, h * 0.78), fill=(4, 4, 8))
    d.ellipse((ax, ay, ax + aw, ay + aw), fill=(4, 4, 8))
    # пол
    d.rectangle((0, h * 0.78, w, h), fill=(26, 24, 30))
    for i in range(14):
        x = w / 2 + (i - 7) * w * 0.12
        d.line((w / 2 + (i - 7) * w * 0.02, h * 0.78, x, h), fill=(36, 34, 42), width=3)
    # факелы
    for fx in (0.22, 0.78):
        _torch(d, w * fx, h * 0.36, s * 1.4)
    for fx in (0.22, 0.78):
        img = _glow(img, (w * fx, h * 0.33), 380 * s, (150, 80, 25))
    # глаза во тьме прохода
    d = ImageDraw.Draw(img)
    ex, ey = w / 2, h * 0.58
    for dx in (-30 * s, 30 * s):
        d.ellipse((ex + dx - 9 * s, ey - 5 * s, ex + dx + 9 * s, ey + 5 * s), fill=(255, 60, 50))
    return _vignette(img)


def tavern(w, h, seed=0):
    rnd = random.Random(seed)
    img = _gradient(w, h, (60, 34, 20), (28, 16, 10))
    d = ImageDraw.Draw(img)
    s = h / 1080
    # доски стен
    for x in range(0, w, int(110 * s)):
        c = rnd.randint(70, 90)
        d.rectangle((x + 2, 0, x + 110 * s - 2, h * 0.7), fill=(c, int(c * 0.6), int(c * 0.35)))
    # балки
    for y in (h * 0.08, h * 0.3):
        d.rectangle((0, y, w, y + 36 * s), fill=(48, 28, 14))
    # полки с бочками и кружками
    d.rectangle((w * 0.05, h * 0.42, w * 0.45, h * 0.44), fill=(50, 30, 15))
    for i in range(5):
        x = w * 0.08 + i * w * 0.07
        d.rounded_rectangle((x, h * 0.34, x + 50 * s, h * 0.42), 8, fill=(160, 120, 60))
        d.rectangle((x + 4, h * 0.35, x + 46 * s, h * 0.36), fill=(220, 220, 200))
    for i in range(3):
        x = w * 0.6 + i * w * 0.11
        d.ellipse((x, h * 0.45, x + 150 * s, h * 0.7), fill=(110, 66, 30))
        for k in (0.3, 0.7):
            d.line((x + 5, h * (0.45 + 0.25 * k), x + 145 * s, h * (0.45 + 0.25 * k)),
                   fill=(70, 70, 70), width=int(8 * s))
    # стойка
    d.rectangle((0, h * 0.7, w, h), fill=(90, 50, 24))
    d.rectangle((0, h * 0.7, w, h * 0.73), fill=(130, 80, 40))
    # очаг
    fx, fy = w * 0.5, h * 0.58
    d.rectangle((fx - 140 * s, fy - 120 * s, fx + 140 * s, fy + 70 * s), fill=(50, 46, 48))
    d.rectangle((fx - 100 * s, fy - 70 * s, fx + 100 * s, fy + 70 * s), fill=(15, 8, 6))
    d.polygon([(fx - 70 * s, fy + 70 * s), (fx, fy - 50 * s), (fx + 70 * s, fy + 70 * s)], fill=(255, 140, 30))
    d.polygon([(fx - 35 * s, fy + 70 * s), (fx, fy), (fx + 35 * s, fy + 70 * s)], fill=(255, 220, 120))
    img = _glow(img, (fx, fy), 520 * s, (170, 90, 30))
    # лампы
    for lx in (0.2, 0.82):
        img = _glow(img, (w * lx, h * 0.18), 220 * s, (140, 100, 40))
    return _vignette(img, 0.5)


def city(w, h, seed=0):
    rnd = random.Random(seed)
    img = _gradient(w, h, (40, 60, 120), (240, 150, 110))
    d = ImageDraw.Draw(img)
    s = h / 1080
    # луна/солнце
    img = _glow(img, (w * 0.7, h * 0.3), 260 * s, (120, 90, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.7 - 70 * s, h * 0.3 - 70 * s, w * 0.7 + 70 * s, h * 0.3 + 70 * s), fill=(255, 230, 180))
    # дальний ряд — замок
    base = h * 0.62
    col = (70, 60, 100)
    cx = w * 0.3
    d.rectangle((cx - 160 * s, base - 260 * s, cx + 160 * s, base), fill=col)
    for tx in (-160, -40, 80):
        d.rectangle((cx + tx * s, base - 380 * s, cx + (tx + 80) * s, base), fill=col)
        d.polygon([(cx + (tx - 10) * s, base - 380 * s), (cx + (tx + 40) * s, base - 470 * s),
                   (cx + (tx + 90) * s, base - 380 * s)], fill=(60, 50, 90))
    # ближний ряд домов
    x = -20
    while x < w:
        bw = rnd.randint(140, 240) * s
        bh = rnd.randint(200, 360) * s
        c = rnd.randint(30, 50)
        d.rectangle((x, h * 0.98 - bh, x + bw, h), fill=(c, c - 5, c + 20))
        d.polygon([(x - 15 * s, h * 0.98 - bh), (x + bw / 2, h * 0.98 - bh - 110 * s),
                   (x + bw + 15 * s, h * 0.98 - bh)], fill=(c + 30, 30, 40))
        for wy in range(int(h * 0.98 - bh + 40 * s), int(h - 60 * s), int(80 * s)):
            for wx in (x + bw * 0.25, x + bw * 0.62):
                lit = rnd.random() < 0.55
                d.rectangle((wx, wy, wx + 34 * s, wy + 44 * s),
                            fill=(255, 200, 100) if lit else (20, 20, 35))
        x += bw + 10 * s
    return _vignette(img, 0.4)


def _hero_shape(d, x, y, s, fill):
    """Безликий силуэт героя в плаще с мечом (вид со спины)."""
    d.ellipse((x - 45 * s, y - 330 * s, x + 45 * s, y - 230 * s), fill=fill)          # голова
    d.polygon([(x - 70 * s, y - 240 * s), (x + 70 * s, y - 240 * s),
               (x + 150 * s, y), (x - 150 * s, y)], fill=fill)                         # плащ
    d.polygon([(x + 110 * s, y - 120 * s), (x + 130 * s, y - 120 * s),
               (x + 260 * s, y - 520 * s), (x + 245 * s, y - 525 * s)], fill=fill)     # меч
    d.rectangle((x + 95 * s, y - 140 * s, x + 150 * s, y - 125 * s), fill=fill)


def hero(w, h, seed=0):
    img = _gradient(w, h, (20, 30, 70), (150, 70, 90))
    s = h / 1080
    img = _glow(img, (w * 0.5, h * 0.55), 600 * s, (200, 120, 60))
    d = ImageDraw.Draw(img)
    # горы
    d.polygon([(0, h * 0.75), (w * 0.2, h * 0.5), (w * 0.4, h * 0.72), (w * 0.62, h * 0.45),
               (w * 0.85, h * 0.7), (w, h * 0.55), (w, h), (0, h)], fill=(40, 30, 60))
    d.rectangle((0, h * 0.86, w, h), fill=(15, 12, 22))
    _hero_shape(d, w * 0.45, h * 0.88, s * 1.3, (8, 6, 12))
    # лучи
    rays = Image.new("L", img.size, 0)
    rd = ImageDraw.Draw(rays)
    for a in range(0, 360, 18):
        r = math.radians(a)
        rd.polygon([(w * 0.5, h * 0.55), (w * 0.5 + 2000 * math.cos(r), h * 0.55 + 2000 * math.sin(r)),
                    (w * 0.5 + 2000 * math.cos(r + 0.08), h * 0.55 + 2000 * math.sin(r + 0.08))], fill=24)
    rays = rays.filter(ImageFilter.GaussianBlur(10))
    arr = np.asarray(img, np.int16) + np.asarray(rays, np.int16)[..., None]
    return _vignette(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), 0.45)


def monster(w, h, seed=0):
    rnd = random.Random(seed)
    img = _gradient(w, h, (30, 6, 12), (6, 2, 6))
    s = h / 1080
    img = _glow(img, (w * 0.5, h * 0.95), 700 * s, (130, 20, 30))
    d = ImageDraw.Draw(img)
    cx, cy = w * 0.5, h * 0.62
    sh = (4, 0, 4)
    # огромная тень: рога, плечи, когти
    d.ellipse((cx - 420 * s, cy - 200 * s, cx + 420 * s, cy + 600 * s), fill=sh)
    d.ellipse((cx - 170 * s, cy - 380 * s, cx + 170 * s, cy - 60 * s), fill=sh)
    for side in (-1, 1):
        d.polygon([(cx + side * 120 * s, cy - 330 * s), (cx + side * 330 * s, cy - 560 * s),
                   (cx + side * 200 * s, cy - 260 * s)], fill=sh)
        for k in range(3):
            bx = cx + side * (380 + k * 40) * s
            d.polygon([(bx, cy + 260 * s), (bx + side * 70 * s, cy + 120 * s), (bx + side * 20 * s, cy + 290 * s)],
                      fill=sh)
    # светящиеся глаза
    for side in (-1, 1):
        ex = cx + side * 70 * s
        d.polygon([(ex - 45 * s, cy - 230 * s), (ex + 45 * s, cy - 230 * s + side * 20 * s),
                   (ex, cy - 200 * s)], fill=(255, 70, 40))
    img = _glow(img, (cx, cy - 220 * s), 200 * s, (160, 40, 20))
    # искры
    d = ImageDraw.Draw(img)
    for _ in range(80):
        x, y = rnd.uniform(0, w), rnd.uniform(0, h)
        r = rnd.uniform(1, 4) * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, rnd.randint(80, 160), 40))
    return _vignette(img, 0.5)


def game_backdrop(w, h, seed=0):
    """Фон для игрового окна: тёмная «цифровая» сетка."""
    img = _gradient(w, h, (10, 16, 40), (4, 6, 18))
    d = ImageDraw.Draw(img)
    step = int(60 * h / 1080)
    for x in range(0, w, step):
        d.line((x, 0, x, h), fill=(20, 40, 80), width=1)
    for y in range(0, h, step):
        d.line((0, y, w, y), fill=(20, 40, 80), width=1)
    img = _glow(img, (w / 2, h / 2), 600 * h / 1080, (20, 60, 120))
    return _vignette(img, 0.5)


GENERATORS = {
    "dungeon": dungeon, "tavern": tavern, "city": city,
    "hero": hero, "monster": monster, "game": game_backdrop,
}


@lru_cache(maxsize=24)
def scene_image(key: str, w: int, h: int, seed: int = 0) -> np.ndarray:
    """Сцена с запасом под панораму, в виде numpy RGB."""
    gen = GENERATORS.get(key, dungeon)
    big = gen(int(w * OVERSCAN), int(h * OVERSCAN), seed)
    return np.ascontiguousarray(np.asarray(big.convert("RGB")))


# ------------------------------------------------------------ игровое окно ---

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


def normalize_scene(name: str) -> str | None:
    n = name.strip().lower()
    if n in SCENE_KEYS:
        return n
    return SCENE_ALIASES.get(n)


if __name__ == "__main__":
    from .config import ROOT
    out = ROOT / ".cache" / "scenes_preview"
    out.mkdir(parents=True, exist_ok=True)
    for k in SCENE_KEYS:
        Image.fromarray(scene_image(k, 960, 540)).save(out / f"{k}.jpg")
    game_window("Получен навык: «Ирония рыцаря» (ур. 1)", 700).save(out / "window.png")
    print("Превью сцен в", out)
