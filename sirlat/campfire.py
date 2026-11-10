"""Единственная сцена: ночная поляна с костром (backgrounds/forest_campfire.*).

Живость: очень медленный зум к костру, мерцание света от огня (на фоне и на
сэре Лате), летящие искры. Сэр Лат сидит слева у бревна; чтобы он не выделялся
на низкополигональной картинке: матовые цвета (character.campfire_style),
тёплый свет костра с одной стороны и холодные синие тени с другой, мягкая тень
на земле, общая цветокоррекция кадра.

Точки на картинке (доли ширины/высоты) можно уточнить файлом
backgrounds/forest_campfire.json, например {"fire": [0.555, 0.63], "ground": 0.715}.
"""
from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .config import ROOT

BG_NAME = "forest_campfire"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_MARKS = {
    "fire": [0.555, 0.63],       # центр пламени
    "sparks": [0.555, 0.585],    # откуда летят искры (верх пламени)
    "ground": 0.715,             # уровень земли слева от костра
    "lat_x": 0.2,                # центр сэра Лата по горизонтали (доля кадра)
}
OV = 1.06                        # запас под медленный зум
ZOOM = 0.035                     # насколько приблизиться за полцикла
ZOOM_PERIOD = 160.0              # секунд на цикл «приблизить — отдалить»


class BackgroundMissing(RuntimeError):
    pass


def find_background() -> Path:
    d = ROOT / "backgrounds"
    for p in sorted(d.glob("*")) if d.is_dir() else []:
        if p.stem.lower() == BG_NAME and p.suffix.lower() in IMG_EXT and p.stat().st_size > 1000:
            return p
    raise BackgroundMissing(
        f"Нет фона backgrounds/{BG_NAME}.png (или .jpg). Загрузите картинку ночной поляны с костром.")


def landmarks(path: Path) -> dict:
    marks = dict(DEFAULT_MARKS)
    js = path.with_suffix(".json")
    if js.exists():
        marks.update(json.loads(js.read_text(encoding="utf-8")))
    return marks


def flicker(t: float) -> float:
    """Яркость огня 0.6…1.0: плавное «дыхание» пламени без резких скачков."""
    n = (0.45 * math.sin(2 * math.pi * 1.3 * t) + 0.3 * math.sin(2 * math.pi * 2.7 * t + 1.1)
         + 0.17 * math.sin(2 * math.pi * 4.1 * t + 2.3) + 0.08 * math.sin(2 * math.pi * 7.3 * t + 0.4))
    return 0.8 + 0.2 * n


@lru_cache(maxsize=8)
def _glow(r: int, color) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    k = np.clip(1 - np.sqrt(x * x + y * y) / max(r, 1), 0, 1) ** 2
    return k[..., None] * np.array(color, np.float32)


def _add(frame, add, x, y, k=1.0):
    H, W = frame.shape[:2]
    h, w = add.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    reg = frame[y0:y1, x0:x1].astype(np.float32) + add[y0 - y:y1 - y, x0 - x:x1 - x] * k
    frame[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


class CampfireScene:
    ZSTEP = 0.001

    def __init__(self, W: int, H: int, fps: int, seed: int = 0):
        self.W, self.H, self.fps = W, H, fps
        self.path = find_background()
        self.m = landmarks(self.path)
        img = ImageOps.exif_transpose(Image.open(self.path)).convert("RGB")
        iw, ih = img.size
        BW, BH = round(W * OV), round(H * OV)
        sc = max(BW / iw, BH / ih)
        big = img.resize((round(iw * sc), round(ih * sc)), Image.LANCZOS)
        fx, fy = self.m["fire"]
        offx = int(min(max(fx * big.width - BW / 2, 0), big.width - BW))
        offy = int(min(max(fy * big.height - BH * 0.62, 0), big.height - BH))
        self.back = big.crop((offx, offy, offx + BW, offy + BH))
        if sc > 1.3:  # картинка увеличена — чуть подтачиваем
            self.back = self.back.filter(ImageFilter.UnsharpMask(radius=2, percent=50, threshold=2))
        self.BW, self.BH = BW, BH
        to_back = lambda u, v: (u * big.width - offx, v * big.height - offy)  # noqa: E731
        self.FX, self.FY = to_back(*self.m["fire"])
        self.SX, self.SY = to_back(*self.m["sparks"])
        self.GY = to_back(0, self.m["ground"])[1]
        # при зуме 1 кадр по центру задника; позиция костра в кадре — опорная точка зума
        self.ox1, self.oy1 = (BW - W) / 2, (BH - H) / 2
        self.levels = {}
        self.s = min(W, H) / 1080
        rnd = random.Random(seed)
        s = self.s
        self.sparks = [dict(P=rnd.uniform(1.4, 3.0), ph=rnd.uniform(0, 3), x0=rnd.uniform(-1, 1) * 40 * s,
                            vx=rnd.uniform(-18, 18) * s, vy=rnd.uniform(-170, -90) * s, wob=rnd.uniform(4, 14) * s,
                            f=rnd.uniform(2, 5), r=max(1, int(round(rnd.uniform(1.4, 3.0) * s))))
                       for _ in range(46)]
        self._lut, self._vig = _grade_tables(W, H)

    # ---- геометрия в кадре (при зуме 1)
    def to_frame(self, X, Y):
        return X - self.ox1, Y - self.oy1

    @property
    def fire_xy(self):
        return self.to_frame(self.FX, self.FY)

    @property
    def ground_y(self):
        return self.to_frame(0, self.GY)[1]

    # ---- кадр
    def _level(self, z):
        i = int(round((z - 1.0) / self.ZSTEP))
        if i not in self.levels:
            zz = 1.0 + i * self.ZSTEP
            img = self.back if i == 0 else self.back.resize(
                (round(self.BW * zz), round(self.BH * zz)), Image.BILINEAR)
            self.levels[i] = (np.asarray(img), zz)
        return self.levels[i]

    def frame(self, t: float) -> tuple[np.ndarray, float]:
        """Кадр фона в момент t (секунды от начала видео) и яркость огня."""
        W, H = self.W, self.H
        z = 1 + ZOOM * (0.5 - 0.5 * math.cos(2 * math.pi * t / ZOOM_PERIOD))
        lvl, zz = self._level(z)
        fxf, fyf = self.fire_xy
        ox = int(round(min(max(self.FX * zz - fxf, 0), lvl.shape[1] - W)))
        oy = int(round(min(max(self.FY * zz - fyf, 0), lvl.shape[0] - H)))
        frame = lvl[oy:oy + H, ox:ox + W].copy()
        k = flicker(t)
        s = self.s
        # мерцание света костра на поляне (светлее/темнее вокруг огня)
        r = int(520 * s)
        g = _glow(r, (255, 140, 60))
        fx, fy = self.FX * zz - ox, self.FY * zz - oy
        _add(frame, g, int(fx - r), int(fy - r), (k - 0.8) * 0.55)
        # искры
        sx, sy = self.SX * zz - ox, self.SY * zz - oy
        for p in self.sparks:
            age = (t + p["ph"]) % p["P"]
            u = age / p["P"]
            x = sx + p["x0"] + p["vx"] * age + p["wob"] * math.sin(p["f"] * age + p["ph"] * 3)
            y = sy + p["vy"] * age + 30 * s * age * age
            a = (1 - u) ** 1.4 * (0.6 + 0.5 * (k - 0.6))
            d = _glow(p["r"] * 2, (255, 175, 80))
            _add(frame, d, int(x - p["r"] * 2), int(y - p["r"] * 2), a * 1.6)
        return frame, k

    def grade(self, frame: np.ndarray) -> np.ndarray:
        """Общая цветокоррекция: тёплые света, холодные тени, мягкая виньетка."""
        out = np.empty_like(frame)
        for c in range(3):
            out[..., c] = self._lut[c][frame[..., c]]
        return ((out.astype(np.uint16) * self._vig[..., None]) >> 8).astype(np.uint8)

    def close(self):
        self.levels.clear()


@lru_cache(maxsize=4)
def _grade_tables(W: int, H: int):
    x = np.arange(256, dtype=np.float32) / 255
    s_curve = x + 0.08 * np.sin(2 * np.pi * x) * (-1) * 0.5        # мягкий S-контраст
    lum_w = np.clip((x - 0.5) * 2, 0, 1)                            # доля «светов»
    sh_w = np.clip((0.5 - x) * 2, 0, 1)                             # доля «теней»
    lut = []
    for warm, cool in ((0.035, -0.02), (0.012, -0.004), (-0.03, 0.035)):   # R, G, B
        v = s_curve + warm * lum_w + cool * sh_w
        lut.append(np.clip(v * 255, 0, 255).astype(np.uint8))
    y, xg = np.ogrid[:H, :W]
    d = np.sqrt(((xg - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    vig = np.clip(1 - 0.22 * d ** 2, 0.6, 1) * 256
    return lut, vig.astype(np.uint16)


# ----------------------------------------------------- сэр Лат у костра ---

def light_sprite(rgb: np.ndarray, al: np.ndarray, k: float, fire_right: bool = True) -> np.ndarray:
    """Тёплый мерцающий свет костра с одной стороны, холодные синие тени с другой."""
    h, w = rgb.shape[:2]
    # градиент — по реальной ширине фигуры, а не по всему холсту спрайта
    cols = np.nonzero(al[..., 0].max(0) > 0.3)[0]
    x0, x1 = (cols[0], cols[-1]) if len(cols) else (0, w - 1)
    gx = np.clip((np.arange(w, dtype=np.float32) - x0) / max(1, x1 - x0), 0, 1)
    if not fire_right:
        gx = 1 - gx
    g = np.clip((gx - 0.05) / 0.8, 0, 1) ** 1.3                  # к костру — теплее и светлее
    gy = np.linspace(0.75, 1.0, h, dtype=np.float32)[:, None]      # ниже — ближе к огню
    warm = (g[None, :] * gy)[..., None] * min(1.0, k * 1.3)
    cool = (1 - g)[None, :, None]
    out = rgb.astype(np.float32)
    # сторона костра: тёплый оранжевый оттенок (меньше синего и зелёного) + отсвет
    out *= 1 + warm * (np.array([1.32, 0.9, 0.55], np.float32) - 1)
    out += warm * np.array([46, 14, 0], np.float32)
    # другая сторона: холодные синие тени
    out *= 1 - cool * 0.55 * (1 - np.array([0.74, 0.84, 1.1], np.float32))
    # контровой свет: оранжевая кромка силуэта со стороны костра
    a = al[..., 0]
    d = max(3, w // 70)
    nb = np.zeros_like(a)
    if fire_right:
        nb[:, :-d] = a[:, d:]
    else:
        nb[:, d:] = a[:, :-d]
    rim = np.clip(a - nb, 0, 1) * gy
    out += (rim * k * 1.3)[..., None] * np.array([255, 140, 60], np.float32)
    return np.clip(out, 0, 255)


@lru_cache(maxsize=4)
def ground_shadow(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    img = Image.new("L", (int(w * 1.4), int(h * 3)), 0)
    ImageDraw.Draw(img).ellipse((w * 0.2, h, w * 1.2, h * 2), fill=200)
    img = img.filter(ImageFilter.GaussianBlur(h * 0.45))
    a = np.asarray(img, np.float32)[..., None] / 255
    rgb = np.zeros(a.shape[:2] + (3,), np.float32) + np.array([8, 10, 22], np.float32)
    return rgb, a


@lru_cache(maxsize=4)
def log_sprite(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Низкополигональное бревно (цвета — как у бревна на картинке) перед сэром Латом:
    шестигранный ствол с гранями от освещённой к теневой, торец с кольцами, тень."""
    W_, H_ = w, int(h * 1.35)
    img = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # контактная тень на земле
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).ellipse((w * 0.04, h * 0.78, w * 1.0, h * 1.3), fill=170)
    sh = sh.filter(ImageFilter.GaussianBlur(h * 0.12))
    img.paste((10, 8, 16, 255), (0, 0), sh)
    r = h / 2
    cx, cy = r * 0.78, r
    x1 = w - r * 0.5
    # грани ствола сверху вниз: свет -> тень
    bands = [(-90, -50, (214, 112, 62)), (-50, -10, (190, 92, 50)), (-10, 30, (158, 70, 40)),
             (30, 70, (122, 52, 33)), (70, 90, (88, 38, 28))]
    for a0, a1, col in bands:
        y0 = cy + r * math.sin(math.radians(a0))
        y1 = cy + r * math.sin(math.radians(a1))
        x0a = cx + r * 0.5 * math.cos(math.radians(a0))
        x0b = cx + r * 0.5 * math.cos(math.radians(a1))
        d.polygon([(x0a, y0), (x1, y0 + h * 0.015), (x1, y1 - h * 0.015), (x0b, y1)], fill=col + (255,))
    # правый (дальний) край — скруглённый торец в тени
    d.ellipse((x1 - r * 0.35, cy - r * 0.97, x1 + r * 0.35, cy + r * 0.97), fill=(98, 42, 30, 255))
    # ближний торец: шестиугольник со спилом и кольцами
    hexa = lambda rr, rot=0: [(cx + rr * 0.55 * math.cos(math.radians(a + rot)),  # noqa: E731
                               cy + rr * math.sin(math.radians(a + rot))) for a in range(0, 360, 60)]
    d.polygon(hexa(r), fill=(150, 76, 46, 255))
    d.polygon(hexa(r * 0.86), fill=(206, 136, 86, 255))
    d.polygon(hexa(r * 0.6, 30), fill=(184, 112, 70, 255))
    d.polygon(hexa(r * 0.36), fill=(208, 140, 90, 255))
    d.polygon(hexa(r * 0.14, 30), fill=(170, 98, 60, 255))
    img = img.filter(ImageFilter.GaussianBlur(max(0.8, h * 0.012)))
    arr = np.asarray(img, np.float32)
    return arr[..., :3] * np.array([0.82, 0.84, 0.96], np.float32), arr[..., 3:4] / 255
