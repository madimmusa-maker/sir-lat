"""Нарисованные кодом сцены в едином мультяшном стиле — единственный фон видео.

Сцены: подземелье, пещера, таверна, город, лес, игровое окно, безликий силуэт
героя и тень монстра. Никаких фото, реальных мест, людей и стоковых съёмок.

Каждая сцена состоит из слоёв:
  * задник (рисуется один раз, чуть больше кадра — для зума и панорамы);
  * передний план (колонны, стволы, камни) — двигается сильнее: параллакс;
  * источники света — мерцают (факелы, очаг, фонари) или «дышат» (кристаллы);
  * языки пламени у факелов и очага;
  * частицы: искры, пыль, светлячки, блёстки, «цифровые» квадратики.

Камера в каждом плане медленно зумит и смещается. Чтобы это было дёшево,
уровни зума считаются заранее (шаг 0.25%), а в каждом кадре только вырезается
окно нужного размера.
"""
from __future__ import annotations

import math
import random
import re
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import font

SCENE_KEYS = ["dungeon", "cave", "tavern", "city", "forest", "game", "hero", "monster"]
SCENE_NAMES = {
    "dungeon": "подземелье", "cave": "пещера", "tavern": "таверна", "city": "город",
    "forest": "лес", "game": "окно", "hero": "герой", "monster": "монстр",
}
SCENE_ALIASES = {
    "подземелье": "dungeon", "данж": "dungeon", "темница": "dungeon", "склеп": "dungeon",
    "пещера": "cave", "грот": "cave", "шахта": "cave",
    "таверна": "tavern", "трактир": "tavern",
    "город": "city", "улица": "city", "замок": "city", "деревня": "city",
    "лес": "forest", "чаща": "forest",
    "окно": "game", "игровое окно": "game", "система": "game", "интерфейс": "game",
    "герой": "hero", "силуэт": "hero",
    "монстр": "monster", "тень": "monster", "босс": "monster",
}
OV = 1.15            # запас задника под зум и панораму
INK = (20, 16, 32)   # единый цвет контуров


def normalize_scene(name: str) -> str | None:
    n = name.strip().lower().replace("ё", "е")
    if n in SCENE_KEYS:
        return n
    return SCENE_ALIASES.get(n)


# ------------------------------------------------- выбор сцены по смыслу ---

# корни слов -> сцена; порядок важен при равенстве (раньше — приоритетнее)
SCENE_WORDS = [
    ("game", r"систем|окн[оаеу]|уровн|навык|квест|задани|статус|опыт|класс|характеристик|"
             r"очк[оиа]|наград|скилл|левел|перк|интерфейс|прокач"),
    ("monster", r"монстр|тен[ьи]|чудовищ|чудищ|босс|гоблин|дракон|демон|твар[ьи]|орк|нежит|"
                r"призрак|скелет|зомби|страшн"),
    ("cave", r"пещер|грот|шахт|рудник|кристалл|сталакт"),
    ("dungeon", r"подземел|данж|темниц|склеп|катакомб|лабиринт|руин|подвал|решетк"),
    ("forest", r"\bлес(а|у|ом|е|ной|ная|ные|ных|ам|ах)?\b|деревь|чащ|рощ|тропинк|болот|охот|"
               r"\bзвер|волк|поляна|кустах"),
    ("tavern", r"таверн|трактир|кабак|харчевн|\bэл[ья]\b|кружк|торговк|торговц|лавк|хозяйк|"
               r"ужин|\bеда\b|\bпив|цены|гильди"),
    ("city", r"город|улиц|площад|столиц|рын[оке]|толп|стражник|королевств|дворец|замок|замке|"
             r"\bворот|деревн|жител"),
    ("hero", r"геро[йяюеи]|\bмеч|битв|сраж|\bбо[йяю]\b|побед|клинок|рыцар|\bсил[аыу]\b|"
             r"тренир|подвиг|судьб"),
]
_SCENE_RE = [(k, re.compile(p, re.I)) for k, p in SCENE_WORDS]


def classify(text: str) -> str | None:
    """Сцена по словам фразы или None, если ничего не подсказывает."""
    t = text.lower().replace("ё", "е")
    best, score = None, 0
    for key, rx in _SCENE_RE:
        # «герой» — запасной вариант: конкретное место во фразе важнее меча и героя
        n = len(rx.findall(t)) * (0.45 if key == "hero" else 1.0)
        if n > score:
            best, score = key, n
    return best


# ------------------------------------------------------------- помощники ---

def _gradient(w, h, stops):
    """Вертикальный градиент по списку (доля, цвет)."""
    t = np.linspace(0, 1, h)[:, None]
    out = np.zeros((h, 3), np.float32)
    for i in range(len(stops) - 1):
        (a, ca), (b, cb) = stops[i], stops[i + 1]
        m = ((t >= a) & (t <= b))[:, 0]
        k = ((t[m] - a) / max(b - a, 1e-6))
        out[m] = np.array(ca, np.float32) * (1 - k) + np.array(cb, np.float32) * k
    arr = np.repeat(out[:, None, :], w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _add_glow(img: Image.Image, xy, radius, color, strength=1.0):
    """Мягкое свечение (аддитивно), запекается в задник."""
    g = _glow_array(int(radius), color, strength)
    arr = np.asarray(img, np.float32).copy()
    _add_region(arr, g, int(xy[0] - radius), int(xy[1] - radius))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@lru_cache(maxsize=64)
def _glow_array(r: int, color, strength=1.0) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / max(r, 1)
    k = np.clip(1 - d, 0, 1) ** 2.2 * strength
    return k[..., None] * np.array(color, np.float32)[None, None, :]


def _add_region(arr: np.ndarray, add: np.ndarray, x: int, y: int, k: float = 1.0):
    H, W = arr.shape[:2]
    h, w = add.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    arr[y0:y1, x0:x1] += add[y0 - y:y1 - y, x0 - x:x1 - x] * k


def _vignette(img: Image.Image, strength=0.5):
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - strength * (d ** 2) / 2, 0.2, 1)[..., None]
    return Image.fromarray((np.asarray(img, np.float32) * m).astype(np.uint8))


def _poly(d, pts, fill, s, outline=True):
    d.polygon(pts, fill=fill)
    if outline:
        d.line(list(pts) + [pts[0]], fill=INK, width=max(2, int(4 * s)), joint="curve")


def _shade(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


# ----------------------------------------------------------------- сцены ---

@dataclass
class SceneArt:
    back: Image.Image                              # RGB, размер кадра × OV
    near: list = field(default_factory=list)       # [(RGBA, x, y, параллакс)]
    lights: list = field(default_factory=list)     # [dict(x, y, r, color, kind)]
    flames: list = field(default_factory=list)     # [(x, y, масштаб)]
    particles: list = field(default_factory=list)  # виды частиц
    eyes: list = field(default_factory=list)       # мигающие глаза во тьме [(x, y, r)]


def _torch(d, x, y, s):
    d.rectangle((x - 7 * s, y, x + 7 * s, y + 70 * s), fill=(84, 54, 30), outline=INK, width=max(2, int(3 * s)))
    d.polygon([(x - 22 * s, y), (x + 22 * s, y), (x + 12 * s, y + 16 * s), (x - 12 * s, y + 16 * s)],
              fill=(70, 70, 84), outline=INK)


def _bricks(d, x0, y0, x1, y1, bw, bh, base, rnd, s):
    row = 0
    y = y0
    while y < y1:
        off = (row % 2) * bw / 2
        x = x0 - off
        while x < x1:
            c = _shade(base, rnd.uniform(0.82, 1.12))
            d.rounded_rectangle((x + 3 * s, y + 3 * s, x + bw - 3 * s, y + bh - 3 * s), radius=int(6 * s), fill=c)
            if rnd.random() < 0.08:
                cx = x + rnd.uniform(0.3, 0.7) * bw
                d.line([(cx, y + 6 * s), (cx + 10 * s, y + bh * 0.5), (cx - 4 * s, y + bh - 8 * s)],
                       fill=_shade(base, 0.55), width=max(1, int(2 * s)))
            x += bw
        y += bh
        row += 1


def _pillar(w, h, s, color, rnd):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, h), fill=color + (255,))
    _bricks(d, 0, 0, w, h, w / 2, 90 * s, _shade(color, 1.1), rnd, s)
    d.rectangle((0, 0, w * 0.12, h), fill=_shade(color, 0.7) + (255,))
    d.rectangle((w * 0.88, 0, w, h), fill=_shade(color, 0.6) + (255,))
    d.line([(1, 0), (1, h)], fill=INK, width=max(2, int(4 * s)))
    d.line([(w - 2, 0), (w - 2, h)], fill=INK, width=max(2, int(4 * s)))
    return img


def dungeon(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (40, 42, 62)), (0.7, (24, 24, 38)), (1, (14, 12, 22))])
    d = ImageDraw.Draw(img)
    floor = h * 0.78
    _bricks(d, 0, 0, w, floor, 150 * s, 66 * s, (52, 54, 74), rnd, s)
    # пол с перспективой
    d.rectangle((0, floor, w, h), fill=(30, 28, 40))
    for i in range(-9, 10):
        d.line([(w / 2 + i * w * 0.02, floor), (w / 2 + i * w * 0.14, h)], fill=(40, 38, 52), width=max(1, int(3 * s)))
    for k in range(1, 5):
        y = floor + (h - floor) * (k / 5) ** 1.6
        d.line([(0, y), (w, y)], fill=(40, 38, 52), width=max(1, int(2 * s)))
    # арка
    aw = min(w * 0.26, 420 * s * 1.3)
    ay = floor - aw * 1.25
    for k, c in ((1.18, (70, 72, 94)), (1.0, (6, 6, 12))):
        ww = aw * k
        xx = w / 2 - ww / 2
        yy = floor - aw * 1.25 - (ww - aw) / 2
        d.rectangle((xx, yy + ww / 2, xx + ww, floor), fill=c, outline=INK, width=max(2, int(4 * s)))
        d.pieslice((xx, yy, xx + ww, yy + ww), 180, 360, fill=c, outline=INK, width=max(2, int(4 * s)))
        d.rectangle((xx + 3 * s, yy + ww / 2 - 2 * s, xx + ww - 3 * s, floor - 2 * s), fill=c)
    # цепи
    for cx in (w * 0.36, w * 0.64):
        for k in range(9):
            y = k * 26 * s
            d.ellipse((cx - 8 * s, y, cx + 8 * s, y + 30 * s), outline=(24, 22, 30), width=max(2, int(5 * s)))
    # факелы
    lights, flames = [], []
    for fx in (0.2, 0.8):
        x, y = w * fx, h * 0.4
        _torch(d, x, y, s * 1.3)
        img = _add_glow(img, (x, y - 20 * s), 300 * s, (120, 64, 24))
        d = ImageDraw.Draw(img)
        lights.append(dict(x=x, y=y - 20 * s, r=340 * s, color=(150, 80, 30), kind="torch"))
        flames.append((x, y + 2 * s, 1.3 * s))
    img = _vignette(img, 0.55)
    near = [
        (_pillar(int(w * 0.09), h, s, (46, 46, 64), rnd), 0, 0, 1.6),
        (_pillar(int(w * 0.09), h, s, (46, 46, 64), rnd), int(w * 0.91), 0, 1.6),
    ]
    eyes = [(w / 2 - 30 * s, ay + aw * 1.0, 8 * s), (w / 2 + 30 * s, ay + aw * 1.0, 8 * s)]
    return SceneArt(img, near, lights, flames, ["dust", "embers"], eyes)


def cave(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (14, 28, 42)), (0.6, (10, 22, 34)), (1, (6, 12, 20))])
    # дальний выход: столб света
    img = _add_glow(img, (w * 0.72, h * 0.12), 420 * s, (40, 70, 90))
    d = ImageDraw.Draw(img)
    shaft = Image.new("L", img.size, 0)
    ImageDraw.Draw(shaft).polygon([(w * 0.66, 0), (w * 0.78, 0), (w * 0.62, h * 0.85), (w * 0.42, h * 0.85)], fill=40)
    shaft = shaft.filter(ImageFilter.GaussianBlur(30 * s))
    arr = np.asarray(img, np.float32) + np.asarray(shaft, np.float32)[..., None] * np.array([0.6, 0.9, 1.0])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # слои скал: дальние светлее, ближние темнее
    for layer, (col, top) in enumerate((((30, 52, 66), 0.45), ((22, 40, 52), 0.58), ((16, 30, 40), 0.72))):
        pts = [(0, h)]
        x = 0
        while x <= w:
            pts.append((x, h * top + rnd.uniform(-1, 1) * h * 0.07))
            x += rnd.uniform(60, 140) * s
        pts += [(w, h)]
        _poly(d, pts, col, s)
    # сталактиты в глубине
    x = 0
    while x < w:
        ln = rnd.uniform(60, 220) * s
        ww = rnd.uniform(30, 70) * s
        _poly(d, [(x, 0), (x + ww, 0), (x + ww / 2, ln)], (26, 46, 60), s)
        x += ww + rnd.uniform(10, 60) * s
    # светящиеся кристаллы
    lights = []
    for (fx, fy, col) in ((0.3, 0.8, (90, 220, 255)), (0.62, 0.68, (190, 120, 255)), (0.85, 0.86, (90, 220, 255))):
        cx, cy = w * fx, h * fy
        for k in range(5):
            a = math.radians(-90 + rnd.uniform(-40, 40))
            ln = rnd.uniform(50, 120) * s
            bw = rnd.uniform(12, 22) * s
            tip = (cx + math.cos(a) * ln, cy + math.sin(a) * ln)
            nx, ny = -math.sin(a) * bw, math.cos(a) * bw
            _poly(d, [(cx + nx, cy + ny), (cx - nx, cy - ny), tip], col, s)
            d.line([(cx, cy), tip], fill=_shade(col, 1.25), width=max(1, int(3 * s)))
        img = _add_glow(img, (cx, cy - 40 * s), 200 * s, _shade(col, 0.35))
        d = ImageDraw.Draw(img)
        lights.append(dict(x=cx, y=cy - 40 * s, r=230 * s, color=_shade(col, 0.45), kind="pulse"))
    img = _vignette(img, 0.6)
    # передний план: сталактиты сверху и валуны внизу
    top = Image.new("RGBA", (w, int(h * 0.22)), (0, 0, 0, 0))
    td = ImageDraw.Draw(top)
    x = -20 * s
    while x < w:
        ww = rnd.uniform(60, 140) * s
        ln = rnd.uniform(0.3, 1.0) * top.height
        _poly(td, [(x, 0), (x + ww, 0), (x + ww * 0.55, ln)], (10, 20, 28), s)
        x += ww * rnd.uniform(0.7, 1.3)
    rocks = Image.new("RGBA", (int(w * 0.34), int(h * 0.26)), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rocks)
    rw, rh = rocks.size
    _poly(rd, [(0, rh), (rw * 0.1, rh * 0.35), (rw * 0.35, rh * 0.1), (rw * 0.6, rh * 0.3), (rw * 0.85, rh * 0.25),
               (rw, rh * 0.6), (rw, rh)], (12, 22, 30), s)
    near = [(top, 0, 0, 1.5), (rocks, int(w * 0.66), int(h - rh), 1.7)]
    return SceneArt(img, near, lights, [], ["sparkles"])


def tavern(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (70, 40, 24)), (1, (30, 18, 12))])
    d = ImageDraw.Draw(img)
    for x in np.arange(0, w, 110 * s):
        c = rnd.randint(78, 98)
        d.rectangle((x + 3 * s, 0, x + 110 * s - 3 * s, h * 0.72), fill=(c, int(c * 0.6), int(c * 0.36)))
        d.line([(x + 55 * s, h * 0.1), (x + 55 * s, h * 0.6)], fill=(c - 18, int(c * 0.5), int(c * 0.3)),
               width=max(1, int(2 * s)))
    for y in (h * 0.08, h * 0.3):
        d.rectangle((0, y, w, y + 40 * s), fill=(52, 30, 16), outline=INK, width=max(2, int(3 * s)))
    # полки с кружками
    d.rectangle((w * 0.05, h * 0.44, w * 0.42, h * 0.46), fill=(54, 32, 16), outline=INK)
    for i in range(6):
        x = w * 0.07 + i * w * 0.058
        d.rounded_rectangle((x, h * 0.37, x + 46 * s, h * 0.44), 8 * s, fill=(170, 126, 64), outline=INK,
                            width=max(2, int(3 * s)))
        d.rectangle((x + 5 * s, h * 0.375, x + 41 * s, h * 0.385), fill=(236, 232, 210))
    # бочки
    for i in range(3):
        x = w * 0.62 + i * w * 0.11
        d.ellipse((x, h * 0.46, x + 160 * s, h * 0.72), fill=(116, 70, 32), outline=INK, width=max(2, int(4 * s)))
        for k in (0.3, 0.7):
            yy = h * (0.46 + 0.26 * k)
            d.line([(x + 8 * s, yy), (x + 152 * s, yy)], fill=(64, 64, 70), width=max(2, int(8 * s)))
    # очаг
    fx, fy = w * 0.47, h * 0.6
    d.rectangle((fx - 150 * s, fy - 130 * s, fx + 150 * s, fy + 80 * s), fill=(58, 52, 56), outline=INK,
                width=max(2, int(4 * s)))
    d.rectangle((fx - 105 * s, fy - 80 * s, fx + 105 * s, fy + 80 * s), fill=(18, 8, 6))
    for k in range(3):
        d.rounded_rectangle((fx - 80 * s + k * 50 * s, fy + 55 * s, fx - 20 * s + k * 50 * s, fy + 75 * s),
                            6 * s, fill=(70, 40, 20), outline=INK)
    img = _add_glow(img, (fx, fy), 480 * s, (150, 80, 28))
    lights = [dict(x=fx, y=fy, r=520 * s, color=(170, 90, 30), kind="fire")]
    flames = [(fx - 40 * s, fy + 60 * s, 1.6 * s), (fx + 10 * s, fy + 62 * s, 2.1 * s), (fx + 55 * s, fy + 60 * s, 1.4 * s)]
    # фонари
    d = ImageDraw.Draw(img)
    for lx in (0.2, 0.82):
        x, y = w * lx, h * 0.2
        d.line([(x, h * 0.12), (x, y - 20 * s)], fill=INK, width=max(2, int(3 * s)))
        d.rounded_rectangle((x - 18 * s, y - 20 * s, x + 18 * s, y + 26 * s), 6 * s, fill=(255, 210, 120), outline=INK,
                            width=max(2, int(3 * s)))
        img = _add_glow(img, (x, y), 200 * s, (110, 80, 30))
        d = ImageDraw.Draw(img)
        lights.append(dict(x=x, y=y, r=230 * s, color=(120, 90, 36), kind="lamp"))
    img = _vignette(img, 0.5)
    # передний план: стойка с кружками
    ch = int(h * 0.16)
    counter = Image.new("RGBA", (w, ch), (0, 0, 0, 0))
    cd = ImageDraw.Draw(counter)
    cd.rectangle((0, ch * 0.25, w, ch), fill=(92, 52, 24), outline=INK, width=max(2, int(4 * s)))
    cd.rectangle((0, ch * 0.25, w, ch * 0.38), fill=(136, 84, 40), outline=INK, width=max(2, int(3 * s)))
    for fx2 in (0.55, 0.7, 0.9):
        x = w * fx2
        cd.rounded_rectangle((x, ch * 0.0, x + 60 * s, ch * 0.3), 8 * s, fill=(190, 140, 70), outline=INK,
                             width=max(2, int(3 * s)))
        cd.ellipse((x + 4 * s, -6 * s, x + 56 * s, 14 * s), fill=(250, 245, 225))
    near = [(counter, 0, h - ch, 1.5)]
    return SceneArt(img, near, lights, flames, ["dust"])


def city(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (34, 44, 98)), (0.55, (150, 90, 130)), (0.8, (246, 156, 112)), (1, (246, 156, 112))])
    img = _add_glow(img, (w * 0.72, h * 0.28), 300 * s, (110, 80, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.72 - 70 * s, h * 0.28 - 70 * s, w * 0.72 + 70 * s, h * 0.28 + 70 * s), fill=(255, 236, 190),
              outline=INK, width=max(2, int(3 * s)))
    for _ in range(40):
        x, y = rnd.uniform(0, w), rnd.uniform(0, h * 0.35)
        r = rnd.uniform(1, 2.5) * s
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 240, 220))
    # замок вдали
    base = h * 0.64
    col = (78, 66, 112)
    cx = w * 0.3
    _poly(d, [(cx - 170 * s, base), (cx - 170 * s, base - 260 * s), (cx + 170 * s, base - 260 * s), (cx + 170 * s, base)], col, s)
    for tx in (-180, -50, 80):
        x0 = cx + tx * s
        _poly(d, [(x0, base), (x0, base - 390 * s), (x0 + 90 * s, base - 390 * s), (x0 + 90 * s, base)], col, s)
        _poly(d, [(x0 - 12 * s, base - 390 * s), (x0 + 45 * s, base - 480 * s), (x0 + 102 * s, base - 390 * s)], (64, 52, 98), s)
        d.rectangle((x0 + 35 * s, base - 330 * s, x0 + 55 * s, base - 300 * s), fill=(255, 210, 120))
    # дома
    x = -20 * s
    while x < w:
        bw = rnd.uniform(150, 250) * s
        bh = rnd.uniform(220, 380) * s
        c = rnd.randint(36, 56)
        bot = h * 0.97
        _poly(d, [(x, h), (x, bot - bh), (x + bw, bot - bh), (x + bw, h)], (c, c - 6, c + 24), s)
        _poly(d, [(x - 16 * s, bot - bh), (x + bw / 2, bot - bh - 110 * s), (x + bw + 16 * s, bot - bh)], (c + 50, 34, 44), s)
        for wy in np.arange(bot - bh + 40 * s, h - 70 * s, 84 * s):
            for wx in (x + bw * 0.22, x + bw * 0.6):
                lit = rnd.random() < 0.55
                d.rounded_rectangle((wx, wy, wx + 36 * s, wy + 46 * s), 5 * s,
                                    fill=(255, 206, 110) if lit else (24, 22, 40), outline=INK, width=max(1, int(3 * s)))
        x += bw + 12 * s
    img = _vignette(img, 0.4)
    # передний план: фонарные столбы
    lights, near = [], []
    pw, ph = int(90 * s), int(h * 0.62)
    for fx in (0.1, 0.9):
        post = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        pd = ImageDraw.Draw(post)
        pd.rectangle((pw / 2 - 9 * s, 50 * s, pw / 2 + 9 * s, ph), fill=(28, 26, 36), outline=INK)
        pd.rounded_rectangle((pw / 2 - 30 * s, 0, pw / 2 + 30 * s, 60 * s), 8 * s, fill=(255, 214, 120), outline=INK,
                             width=max(2, int(4 * s)))
        x = int(w * fx - pw / 2)
        y = int(h - ph)
        near.append((post, x, y, 1.6))
        lights.append(dict(x=w * fx, y=y + 30 * s, r=220 * s, color=(150, 110, 50), kind="lamp", near=True))
    return SceneArt(img, near, lights, [], ["fireflies"])


def _tree(d, x, base, height, width, col, s):
    _poly(d, [(x - width * 0.08, base), (x - width * 0.05, base - height * 0.4),
              (x + width * 0.05, base - height * 0.4), (x + width * 0.08, base)], _shade(col, 0.7), s, outline=False)
    for k in range(3):
        y = base - height * (0.3 + k * 0.22)
        ww = width * (1 - k * 0.25)
        _poly(d, [(x - ww / 2, y), (x, y - height * 0.38), (x + ww / 2, y)], col, s)


def forest(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (20, 40, 60)), (0.6, (34, 70, 74)), (1, (12, 26, 24))])
    img = _add_glow(img, (w * 0.7, h * 0.2), 260 * s, (60, 80, 70))
    d = ImageDraw.Draw(img)
    d.ellipse((w * 0.7 - 55 * s, h * 0.2 - 55 * s, w * 0.7 + 55 * s, h * 0.2 + 55 * s), fill=(236, 246, 220),
              outline=INK, width=max(2, int(3 * s)))
    for layer, (col, base, hh) in enumerate((((44, 86, 84), 0.62, 360), ((30, 64, 58), 0.78, 460), ((20, 46, 40), 0.95, 560))):
        x = -60 * s
        while x < w + 60 * s:
            height = rnd.uniform(0.8, 1.2) * hh * s
            _tree(d, x, h * base, height, height * 0.5, col, s)
            x += rnd.uniform(70, 150) * s * (1 + layer * 0.3)
    # тропинка
    _poly(d, [(w * 0.46, h * 0.7), (w * 0.54, h * 0.7), (w * 0.75, h), (w * 0.3, h)], (70, 84, 60), s)
    # лучи
    rays = Image.new("L", img.size, 0)
    rd = ImageDraw.Draw(rays)
    for fx in (0.35, 0.5, 0.62):
        rd.polygon([(w * fx, 0), (w * (fx + 0.05), 0), (w * (fx - 0.05), h * 0.9), (w * (fx - 0.12), h * 0.9)], fill=26)
    rays = rays.filter(ImageFilter.GaussianBlur(24 * s))
    arr = np.asarray(img, np.float32) + np.asarray(rays, np.float32)[..., None] * np.array([0.8, 1.0, 0.8])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # светящиеся грибы
    lights = []
    for fx, fy in ((0.38, 0.9), (0.72, 0.86)):
        cx, cy = w * fx, h * fy
        for k in range(3):
            x = cx + (k - 1) * 22 * s
            d.rectangle((x - 4 * s, cy - 18 * s, x + 4 * s, cy), fill=(220, 230, 200))
            d.pieslice((x - 16 * s, cy - 34 * s, x + 16 * s, cy - 6 * s), 180, 360, fill=(120, 240, 200), outline=INK)
        lights.append(dict(x=cx, y=cy - 20 * s, r=120 * s, color=(50, 120, 100), kind="pulse"))
    img = _vignette(img, 0.55)
    # передний план: стволы и кусты
    near = []
    tw, th = int(w * 0.1), h
    for fx in (0.0, 0.92):
        trunk = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        td = ImageDraw.Draw(trunk)
        _poly(td, [(tw * 0.1, th), (tw * 0.2, 0), (tw * 0.8, 0), (tw * 0.95, th)], (30, 24, 26), s)
        for k in range(6):
            y = rnd.uniform(0.1, 0.9) * th
            td.line([(tw * 0.35, y), (tw * 0.45, y + 60 * s)], fill=(46, 36, 38), width=max(2, int(4 * s)))
        near.append((trunk, int(w * fx), 0, 1.6))
    bush = Image.new("RGBA", (int(w * 0.4), int(h * 0.16)), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bush)
    bw_, bh_ = bush.size
    for k in range(7):
        x = k * bw_ / 6
        r = rnd.uniform(0.5, 0.9) * bh_
        bd.ellipse((x - r * 0.8, bh_ - r, x + r * 0.8, bh_ + r), fill=(16, 38, 30), outline=INK, width=max(2, int(3 * s)))
    near.append((bush, int(w * 0.6), h - bh_, 1.7))
    return SceneArt(img, near, lights, [], ["fireflies"])


def game(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (10, 18, 48)), (1, (4, 6, 20))])
    d = ImageDraw.Draw(img)
    step = int(64 * s)
    for x in range(0, w, step):
        d.line((x, 0, x, h), fill=(20, 42, 86), width=1)
    for y in range(0, h, step):
        d.line((0, y, w, y), fill=(20, 42, 86), width=1)
    img = _add_glow(img, (w / 2, h / 2), 560 * s, (20, 60, 130))
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    # плавающие панели интерфейса
    for _ in range(7):
        pw, ph = rnd.uniform(160, 320) * s, rnd.uniform(70, 150) * s
        x, y = rnd.uniform(0, w - pw), rnd.uniform(0, h - ph)
        ld.rounded_rectangle((x, y, x + pw, y + ph), 10 * s, fill=(30, 80, 160, 70), outline=(110, 200, 255, 140),
                             width=max(1, int(2 * s)))
        for k in range(3):
            ld.rectangle((x + 14 * s, y + 18 * s + k * 26 * s, x + pw * rnd.uniform(0.4, 0.85), y + 26 * s + k * 26 * s),
                         fill=(120, 210, 255, 120))
    # кольцо в центре
    cx, cy, r = w / 2, h / 2, 230 * s
    ld.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(110, 220, 255, 160), width=max(2, int(6 * s)))
    ld.ellipse((cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7), outline=(110, 220, 255, 90), width=max(2, int(3 * s)))
    img = img.convert("RGBA")
    img.alpha_composite(lay)
    img = _vignette(img.convert("RGB"), 0.5)
    lights = [dict(x=cx, y=cy, r=420 * s, color=(30, 90, 160), kind="pulse")]
    return SceneArt(img, [], lights, [], ["digital"])


def _hero_shape(d, x, y, s, fill):
    """Безликий силуэт героя в плаще с мечом (вид со спины)."""
    d.ellipse((x - 45 * s, y - 330 * s, x + 45 * s, y - 230 * s), fill=fill)
    d.polygon([(x - 70 * s, y - 240 * s), (x + 70 * s, y - 240 * s), (x + 150 * s, y), (x - 150 * s, y)], fill=fill)
    d.polygon([(x + 110 * s, y - 120 * s), (x + 130 * s, y - 120 * s), (x + 260 * s, y - 520 * s),
               (x + 245 * s, y - 525 * s)], fill=fill)
    d.rectangle((x + 95 * s, y - 140 * s, x + 150 * s, y - 125 * s), fill=fill)


def hero(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (24, 32, 76)), (0.6, (170, 80, 100)), (1, (240, 150, 90))])
    img = _add_glow(img, (w * 0.55, h * 0.58), 560 * s, (190, 110, 50))
    rays = Image.new("L", img.size, 0)
    rd = ImageDraw.Draw(rays)
    for a in range(0, 360, 15):
        r = math.radians(a)
        rd.polygon([(w * 0.55, h * 0.58), (w * 0.55 + 3000 * math.cos(r), h * 0.58 + 3000 * math.sin(r)),
                    (w * 0.55 + 3000 * math.cos(r + 0.07), h * 0.58 + 3000 * math.sin(r + 0.07))], fill=22)
    rays = rays.filter(ImageFilter.GaussianBlur(10 * s))
    arr = np.asarray(img, np.float32) + np.asarray(rays, np.float32)[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    _poly(d, [(0, h * 0.76), (w * 0.18, h * 0.52), (w * 0.38, h * 0.72), (w * 0.62, h * 0.47), (w * 0.84, h * 0.7),
              (w, h * 0.56), (w, h), (0, h)], (52, 36, 70), s)
    _poly(d, [(w * 0.3, h), (w * 0.42, h * 0.84), (w * 0.72, h * 0.82), (w * 0.86, h)], (26, 18, 34), s)
    _hero_shape(d, w * 0.56, h * 0.85, s * 1.25, (10, 8, 16))
    img = _vignette(img, 0.45)
    grass = Image.new("RGBA", (w, int(h * 0.1)), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grass)
    gw, gh = grass.size
    for x in np.arange(0, w, 14 * s):
        hh = rnd.uniform(0.4, 1.0) * gh
        gd.polygon([(x, gh), (x + 6 * s, gh - hh), (x + 12 * s, gh)], fill=(14, 10, 20))
    lights = [dict(x=w * 0.55, y=h * 0.58, r=500 * s, color=(120, 70, 30), kind="pulse")]
    return SceneArt(img, [(grass, 0, h - gh, 1.6)], lights, [], ["dust", "embers"])


def monster(w, h, rnd):
    s = min(w, h) / (1080 * OV)
    img = _gradient(w, h, [(0, (36, 6, 14)), (1, (8, 2, 6))])
    img = _add_glow(img, (w * 0.5, h * 0.95), 640 * s, (130, 20, 30))
    d = ImageDraw.Draw(img)
    cx, cy = w * 0.5, h * 0.62
    sh = (6, 0, 6)
    d.ellipse((cx - 430 * s, cy - 200 * s, cx + 430 * s, cy + 620 * s), fill=sh)
    d.ellipse((cx - 170 * s, cy - 380 * s, cx + 170 * s, cy - 60 * s), fill=sh)
    for side in (-1, 1):
        d.polygon([(cx + side * 120 * s, cy - 330 * s), (cx + side * 340 * s, cy - 570 * s),
                   (cx + side * 200 * s, cy - 260 * s)], fill=sh)
        for k in range(3):
            bx = cx + side * (390 + k * 40) * s
            d.polygon([(bx, cy + 260 * s), (bx + side * 70 * s, cy + 120 * s), (bx + side * 20 * s, cy + 290 * s)], fill=sh)
    img = _vignette(img, 0.5)
    rocks = Image.new("RGBA", (w, int(h * 0.14)), (0, 0, 0, 0))
    rk = ImageDraw.Draw(rocks)
    rw, rh = rocks.size
    x = 0
    while x < w:
        ww = rnd.uniform(80, 200) * s
        _poly(rk, [(x, rh), (x + ww * 0.3, rh * rnd.uniform(0.1, 0.5)), (x + ww * 0.6, rh * 0.4), (x + ww, rh)],
              (14, 4, 10), s)
        x += ww * 0.8
    lights = [dict(x=cx, y=cy - 215 * s, r=240 * s, color=(170, 40, 20), kind="pulse")]
    eyes = [(cx - 70 * s, cy - 215 * s, 22 * s), (cx + 70 * s, cy - 215 * s, 22 * s)]
    return SceneArt(img, [(rocks, 0, h - rh, 1.6)], lights, [], ["ash"], eyes)


BUILDERS = {"dungeon": dungeon, "cave": cave, "tavern": tavern, "city": city, "forest": forest,
            "game": game, "hero": hero, "monster": monster}


@lru_cache(maxsize=10)
def scene_art(key: str, W: int, H: int, seed: int = 0) -> SceneArt:
    w, h = int(W * OV), int(H * OV)
    return BUILDERS.get(key, dungeon)(w, h, random.Random(f"{key}-{seed}"))


# -------------------------------------------------------------- анимация ---

PARTICLES = {
    # цвет, количество, радиус, скорость по x/y (пикс/с на 1080p), колебание, мерцание
    "dust":      ((255, 230, 190), 40, (1.5, 3.0), (-8, 8), (-10, 6), 14, 0.5),
    "embers":    ((255, 150, 60), 26, (1.8, 3.5), (-12, 12), (-60, -25), 18, 0.8),
    "sparkles":  ((140, 230, 255), 40, (1.5, 3.2), (-6, 6), (-10, 10), 10, 1.0),
    "fireflies": ((220, 255, 120), 26, (2.5, 4.0), (-14, 14), (-10, 10), 40, 1.0),
    "digital":   ((120, 220, 255), 40, (2.0, 4.5), (-4, 4), (-60, -25), 6, 0.6),
    "ash":       ((255, 90, 50), 55, (1.5, 3.5), (-14, 14), (-80, -30), 20, 0.7),
}


@lru_cache(maxsize=64)
def _dot(r: int, color, square=False) -> np.ndarray:
    size = r * 4 + 1
    y, x = np.mgrid[:size, :size].astype(np.float32) - r * 2
    if square:
        k = ((np.abs(x) <= r) & (np.abs(y) <= r)).astype(np.float32) + \
            np.clip(1 - np.sqrt(x * x + y * y) / (r * 2), 0, 1) ** 2 * 0.5
    else:
        d = np.sqrt(x * x + y * y) / (r * 2)
        k = np.clip(1 - d, 0, 1) ** 2.5 * 1.6
    return k[..., None] * np.array(color, np.float32)[None, None, :]


@lru_cache(maxsize=16)
def _flame_frames(scale: float) -> list:
    """Несколько вариантов языка пламени (RGBA numpy), чтобы огонь «жил»."""
    out = []
    rnd = random.Random(int(scale * 100))
    for i in range(6):
        w, h = int(46 * scale) + 4, int(84 * scale) + 4
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lean = rnd.uniform(-0.18, 0.18) * w
        hh = rnd.uniform(0.75, 1.0)
        for col, k in (((255, 120, 30, 235), 1.0), ((255, 190, 60, 245), 0.7), ((255, 245, 190, 255), 0.4)):
            bw = w * 0.45 * k
            top = h - h * hh * (0.25 + 0.75 * k)
            d.polygon([(w / 2 - bw, h - 2), (w / 2 + lean * k, top), (w / 2 + bw, h - 2)], fill=col)
            d.ellipse((w / 2 - bw, h - bw * 1.6 - 2, w / 2 + bw, h - 2), fill=col)
        img = img.filter(ImageFilter.GaussianBlur(max(0.6, scale * 0.8)))
        a = np.asarray(img, np.float32)
        out.append((a[..., :3], a[..., 3:4] / 255))
    return out


def _noise(t, phase):
    return (0.5 * math.sin(2 * math.pi * 7.3 * t + phase) + 0.3 * math.sin(2 * math.pi * 11.1 * t + phase * 1.7)
            + 0.2 * math.sin(2 * math.pi * 17.9 * t + phase * 2.3))


def _blend(frame, rgb, al, x, y):
    H, W = frame.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    a = al[y0 - y:y1 - y, x0 - x:x1 - x]
    src = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (reg + (src - reg) * a).astype(np.uint8)


def _add(frame, add, x, y, k=1.0):
    H, W = frame.shape[:2]
    h, w = add.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1 or k <= 0.01:
        return
    reg = frame[y0:y1, x0:x1].astype(np.float32) + add[y0 - y:y1 - y, x0 - x:x1 - x] * k
    frame[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


class AnimatedScene:
    """Один «план»: сцена + движение камеры на n_frames кадров."""

    ZSTEP = 0.0025

    def __init__(self, key: str, W: int, H: int, fps: int, n_frames: int, seed: int, shot_seed: int):
        self.key = key if key in BUILDERS else "dungeon"
        self.W, self.H, self.fps, self.n = W, H, fps, max(1, n_frames)
        self.art = scene_art(self.key, W, H, seed)
        self.back = self.art.back
        self.bw, self.bh = self.back.size
        rnd = random.Random(shot_seed)
        zin = rnd.random() < 0.6
        zr = rnd.uniform(0.035, 0.06)
        self.z0, self.z1 = (1.0, 1.0 + zr) if zin else (1.0 + zr, 1.0)
        mx = (self.bw - W) / 2 * 0.55
        my = (self.bh - H) / 2 * 0.45
        self.p0 = (rnd.uniform(-mx, mx), rnd.uniform(-my, my))
        self.p1 = (rnd.uniform(-mx, mx), rnd.uniform(-my, my))
        self.levels = {}
        s = min(W, H) / 1080
        self.s = s
        self.parts = []
        for kind in self.art.particles:
            col, cnt, (r0, r1), vx, vy, wob, tw = PARTICLES[kind]
            for _ in range(cnt):
                self.parts.append(dict(
                    x=rnd.uniform(0, W), y=rnd.uniform(0, H), vx=rnd.uniform(*vx) * s, vy=rnd.uniform(*vy) * s,
                    wob=wob * s * rnd.uniform(0.5, 1.0), ph=rnd.uniform(0, 6.28), f=rnd.uniform(0.3, 1.2),
                    tw=tw, dot=_dot(max(1, int(round(rnd.uniform(r0, r1) * s))), col, kind == "digital")))
        self.light_ph = [rnd.uniform(0, 6.28) for _ in self.art.lights]
        self.flames = [(x, y, _flame_frames(round(sc, 2))) for x, y, sc in self.art.flames]
        self.near = []
        # передний план нарисован у краёв задника — сдвигаем к краям видимой части
        mx0, my0 = (self.bw - W) / 2 * 0.85, (self.bh - H) / 2 * 0.85
        for img, x, y, par in self.art.near:
            iw, ih = img.size
            if x <= 1:
                x += mx0
            elif x + iw >= self.bw - 1:
                x -= mx0
            if y <= 1:
                y += my0
            elif y + ih >= self.bh - 1:
                y -= my0
            a = np.asarray(img, np.float32)
            self.near.append((a[..., :3], a[..., 3:4] / 255, x, y, par))

    def _level(self, z):
        i = int(round((z - 1.0) / self.ZSTEP))
        if i not in self.levels:
            zz = 1.0 + i * self.ZSTEP
            size = (int(round(self.bw * zz)), int(round(self.bh * zz)))
            img = self.back if i == 0 else self.back.resize(size, Image.BILINEAR)
            self.levels[i] = (np.asarray(img), zz)
        return self.levels[i]

    def frame(self, i: int) -> np.ndarray:
        W, H = self.W, self.H
        t = i / self.fps
        u = min(1.0, i / self.n)
        u = u * u * (3 - 2 * u)
        z = self.z0 + (self.z1 - self.z0) * u
        px = self.p0[0] + (self.p1[0] - self.p0[0]) * u
        py = self.p0[1] + (self.p1[1] - self.p0[1]) * u
        lvl, zz = self._level(z)
        lh, lw = lvl.shape[:2]
        ox = int(round((self.bw / 2 + px) * zz - W / 2))
        oy = int(round((self.bh / 2 + py) * zz - H / 2))
        ox = max(0, min(lw - W, ox))
        oy = max(0, min(lh - H, oy))
        frame = lvl[oy:oy + H, ox:ox + W].copy()

        def to_screen(x, y, par=1.0):
            sx = x * zz - ox - (px * zz) * (par - 1)
            sy = y * zz - oy - (py * zz) * (par - 1)
            return sx, sy

        # мерцающий/«дышащий» свет
        for L, ph in zip(self.art.lights, self.light_ph):
            kind = L["kind"]
            if kind == "torch":
                k = 0.45 + 0.25 * _noise(t, ph)
            elif kind == "fire":
                k = 0.5 + 0.3 * _noise(t * 0.8, ph)
            elif kind == "lamp":
                k = 0.45 + 0.08 * _noise(t * 0.5, ph)
            else:  # pulse
                k = 0.45 + 0.3 * math.sin(2 * math.pi * t / 3.2 + ph)
            r = int(L["r"])
            sx, sy = to_screen(L["x"], L["y"], 1.6 if L.get("near") else 1.0)
            _add(frame, _glow_array(r, tuple(L["color"]), 1.0), int(sx - r), int(sy - r), k)
        # глаза во тьме: медленно «дышат» и иногда моргают
        for j, (ex, ey, er) in enumerate(self.art.eyes):
            sx, sy = to_screen(ex, ey)
            blink = (t * 0.37 + j * 0.0) % 4.0 < 0.15
            k = 0 if blink else 0.7 + 0.3 * math.sin(2 * math.pi * t / 2.5)
            col = (255, 60, 40)
            _add(frame, _glow_array(int(er * 3), col, 1.0), int(sx - er * 3), int(sy - er * 3), k)
            _add(frame, _glow_array(int(er), (255, 200, 160), 1.0), int(sx - er), int(sy - er), k)
        # языки пламени
        for fx, fy, frames in self.flames:
            v = frames[int((t * 12 + fx) % len(frames))]
            sx, sy = to_screen(fx, fy)
            rgb, al = v
            _blend(frame, rgb, al, int(sx - rgb.shape[1] / 2), int(sy - rgb.shape[0]))
        # передний план (параллакс)
        for rgb, al, x, y, par in self.near:
            sx, sy = to_screen(x, y, par)
            _blend(frame, rgb, al, int(sx), int(sy))
        # частицы
        for p in self.parts:
            x = (p["x"] + p["vx"] * t + p["wob"] * math.sin(t * p["f"] + p["ph"])) % (W + 40) - 20
            y = (p["y"] + p["vy"] * t + p["wob"] * 0.6 * math.cos(t * p["f"] * 1.3 + p["ph"])) % (H + 40) - 20
            k = 1 - p["tw"] * 0.5 * (1 + math.sin(t * (2 + p["f"] * 3) + p["ph"]))
            d = p["dot"]
            _add(frame, d, int(x - d.shape[1] / 2), int(y - d.shape[0] / 2), max(0.0, k))
        return frame

    def close(self):
        self.levels.clear()


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


if __name__ == "__main__":
    from .config import ROOT
    out = ROOT / ".cache" / "scenes_preview"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k in SCENE_KEYS:
        sc = AnimatedScene(k, 960, 540, 25, 100, 1, 2)
        im = Image.fromarray(sc.frame(30))
        im.save(out / f"{k}.jpg")
        tiles.append((im, SCENE_NAMES[k]))
    sheet = Image.new("RGB", (960 * 4, 580 * 2), (30, 30, 40))
    d = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(tiles):
        x, y = (i % 4) * 960, (i // 4) * 580
        sheet.paste(im, (x, y))
        d.text((x + 480, y + 560), name, font=font(28), fill="white", anchor="mm")
    sheet.resize((1920, 580)).save(out / "sheet.jpg", quality=88)
    print("Превью сцен в", out)
