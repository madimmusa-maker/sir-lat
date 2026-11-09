"""Сэр Пингвин Лат — живая мимика поверх готовой картинки персонажа.

Основа — assets/sir_lat/sir_lat.png (без фона, по пояс), её делает
`python -m sirlat.prepare_character`. Где клюв и глаза — в landmarks.json.

Каждый кадр собирается из непрерывных параметров (ничего не «щёлкает»):
  * клюв   — открытие плавно следует за громкостью; форма зависит от гласной
             («а» — широко, «о/у» — узко и округло, «и/е» — плоско и широко);
  * глаза  — моргание в случайные моменты (иногда двойное), взгляд в сторону,
             прищур, широко раскрытые глаза, наклон век (хмурится/волнуется);
  * шлем   — забрало поднимается от удивления и опускается, когда злится;
  * тело   — дыхание, кивки в такт речи, наклон вперёд на важных фразах,
             подпрыгивание на смешных, дрожь от страха;
  * крылья — короткие взмахи на ударных словах, поднятое крыло в позе «указывает».
Эмоции смешиваются весами и сменяются за ~0.25 с.

Запуск `python -m sirlat.character` сохраняет кадры эмоций в assets/sir_lat/.
"""
from __future__ import annotations

import json
import math
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import ASSETS

BASE_FILE = ASSETS / "sir_lat.png"
LANDMARKS_FILE = ASSETS / "landmarks.json"

EMOTION_KEYS = ["important", "surprised", "laughing", "angry", "thoughtful", "scared"]
POSE_KEYS = ["none", "point", "sword", "shield"]

PAD_X, PAD_TOP = 0.14, 0.24   # запас холста под наклоны, прыжки и взмахи


def _load_landmarks():
    if not LANDMARKS_FILE.exists():
        raise SystemExit("Нет assets/sir_lat/landmarks.json — запустите: python -m sirlat.prepare_character")
    return json.loads(LANDMARKS_FILE.read_text(encoding="utf-8"))


LM = _load_landmarks()
BASE_W, BASE_H = LM["size"]
W = int(BASE_W * (1 + 2 * PAD_X))
H = int(BASE_H * (1 + PAD_TOP))
OFF_X, OFF_Y = (W - BASE_W) // 2, H - BASE_H


@lru_cache(maxsize=1)
def base_image() -> Image.Image:
    return Image.open(BASE_FILE).convert("RGBA")


# --------------------------------------------------------------- параметры ---
# Все размеры в пикселях исходной картинки (масштабируются под кадр).
# angle: + — наклон к экрану (влево); visor: − вверх, + вниз;
# Глаза по умолчанию полностью открыты; прищур (lid_top/lid_bot/curve) — только у
# смеющегося и не дольше 1 с. lid_top/lid_bot: доля закрытия века; tilt: + внутренние уголки
# век ниже (хмурится), − выше (волнуется); curve: «улыбчивые» нижние веки.
EMO = {
    "important":  dict(angle=-2.5, scale=1.04, sx=1.00, sy=1.02, dx=0.00, dy=0.000, visor=-2,
                       lid_top=0.00, lid_bot=0.00, tilt=0.00, curve=0.0, eye=1.00, gx=0.0, gy=0.0,
                       sway=0.8, shake=0.0, hop=0.0, tint=(1.0, 1.0, 1.0)),
    "surprised":  dict(angle=0.0, scale=1.08, sx=0.97, sy=1.05, dx=0.00, dy=-0.020, visor=-14,
                       lid_top=0.00, lid_bot=0.00, tilt=0.00, curve=0.0, eye=1.22, gx=0.0, gy=-0.1,
                       sway=0.5, shake=0.0, hop=0.0, tint=(1.0, 1.0, 1.0)),
    "laughing":   dict(angle=2.0, scale=1.03, sx=1.00, sy=1.00, dx=0.00, dy=0.000, visor=-5,
                       lid_top=0.14, lid_bot=0.22, tilt=0.00, curve=0.30, eye=1.00, gx=0.0, gy=0.0,
                       sway=1.4, shake=0.0, hop=1.0, tint=(1.04, 1.0, 0.98)),
    "angry":      dict(angle=6.0, scale=1.07, sx=1.04, sy=0.97, dx=-0.02, dy=0.000, visor=9,
                       lid_top=0.00, lid_bot=0.00, tilt=0.00, curve=0.0, eye=0.95, gx=-0.2, gy=0.0,
                       sway=0.4, shake=0.6, hop=0.0, tint=(1.06, 0.95, 0.94)),
    "thoughtful": dict(angle=7.0, scale=0.98, sx=1.00, sy=1.00, dx=0.00, dy=0.000, visor=-4,
                       lid_top=0.00, lid_bot=0.00, tilt=0.00, curve=0.0, eye=1.00, gx=-0.55, gy=-0.6,
                       sway=1.2, shake=0.0, hop=0.0, tint=(1.0, 1.0, 1.0)),
    "scared":     dict(angle=-7.0, scale=0.91, sx=0.96, sy=0.98, dx=0.03, dy=0.000, visor=-7,
                       lid_top=0.00, lid_bot=0.00, tilt=0.00, curve=0.0, eye=1.14, gx=0.0, gy=0.0,
                       sway=0.3, shake=2.5, hop=0.0, tint=(0.94, 0.98, 1.08)),
}
# позы: добавки к телу и положение крыльев (градусы, + — крыло поднимается наружу)
POSE = {
    "none":   dict(angle=0, scale=1.00, sy=1.00, dx=0.00, wing_l=0, wing_r=0),
    "point":  dict(angle=6, scale=1.00, sy=1.00, dx=-0.03, wing_l=26, wing_r=0),
    "sword":  dict(angle=-2, scale=1.05, sy=1.03, dx=0.00, wing_l=0, wing_r=20),
    "shield": dict(angle=-6, scale=0.93, sy=1.00, dx=0.02, wing_l=-8, wing_r=-8),
}

VOWELS = {"а": "a", "я": "a", "о": "o", "ё": "o", "у": "u", "ю": "u",
          "ы": "i", "и": "i", "е": "e", "э": "e"}
VOWEL_OPEN = {"a": 1.0, "o": 0.82, "u": 0.62, "e": 0.62, "i": 0.5, None: 0.7}
VOWEL_WIDTH = {"a": 1.0, "o": 0.72, "u": 0.6, "e": 1.08, "i": 1.14, None: 0.95}


def vowel_at(word: str, progress: float) -> str | None:
    """Какая гласная звучит в момент progress (0..1) внутри слова."""
    letters = [c for c in word.lower() if c.isalpha()]
    if not letters:
        return None
    i = min(len(letters) - 1, max(0, int(progress * len(letters))))
    for d in (0, 1, -1, 2, -2):
        j = i + d
        if 0 <= j < len(letters) and letters[j] in VOWELS:
            return VOWELS[letters[j]]
    return None


def _approach(cur, target, dt, tau):
    return cur + (target - cur) * (1 - math.exp(-dt / max(tau, 1e-4)))


# ---------------------------------------------------------- помощники ---

def _sample(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Билинейная выборка RGBA (float32) по дробным координатам; за краем — прозрачно."""
    h, w = src.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    out = np.zeros(xs.shape + (4,), np.float32)
    for oy, wy in ((0, 1 - fy), (1, fy)):
        for ox, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = x0 + ox, y0 + oy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            v = src[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)].astype(np.float32)
            v[~ok] = 0
            out += v * wx * wy
    return out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def transform(img: Image.Image, angle, scale, sx, sy, dx, dy) -> Image.Image:
    """Наклон/масштаб вокруг нижней середины (точки «пояса»)."""
    w, h = img.size
    px, py = w / 2, h * 0.97
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    kx, ky = scale * sx, scale * sy
    a, b = c / kx, -s / kx
    d_, e = s / ky, c / ky
    cx_, cy_ = px + dx * w, py + dy * h
    return img.transform((w, h), Image.AFFINE,
                         (a, b, px - a * cx_ - b * cy_, d_, e, py - d_ * cx_ - e * cy_),
                         resample=Image.BICUBIC)


def _detect_wings(base: np.ndarray):
    """Тёмные ласты по бокам ниже наплечников: маска и точка «плеча»."""
    h, w = base.shape[:2]
    lum = base[..., :3].astype(np.int32) @ np.array([299, 587, 114]) // 1000
    dark = (lum < 80) & (base[..., 3] > 128)
    wings = {}
    for side, (x0, x1, y0) in {"l": (0, 0.2, 0.76), "r": (0.88, 1.0, 0.70)}.items():
        m = np.zeros_like(dark)
        xa, xb, ya = int(w * x0), int(w * x1), int(h * y0)
        m[ya:, xa:xb] = dark[ya:, xa:xb]
        ys, xs = np.nonzero(m)
        if len(xs) < 80:
            continue
        top = ys.min()
        near_top = ys < top + max(6, (ys.max() - top) * 0.12)
        pivot = (float(xs[near_top].mean()), float(top))
        wings[side] = (m, pivot)
    return wings


# ------------------------------------------------------------------ аниматор ---

class Animator:
    """Собирает кадры сэра Лата нужной высоты; состояние плавно меняется во времени."""

    def __init__(self, fps: int, height: int, seed: int = 0, gestures=None, screen_side: str = "left"):
        self.fps, self.dt = fps, 1.0 / fps
        self.rnd = random.Random(seed)
        self.s = s = height / H
        self.cw, self.ch = round(W * s), height
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.alpha_composite(base_image(), (OFF_X, OFF_Y))
        canvas = canvas.resize((self.cw, self.ch), Image.LANCZOS)
        self.canvas = np.array(canvas)
        self.P = lambda x, y: ((x + OFF_X) * s, (y + OFF_Y) * s)
        self._prep_eyes()
        self._prep_beak()
        self._prep_wings()
        self._prep_accents()
        eye_top = min(e["cy"] - e["ry"] for e in self.eyes)
        self.visor_full = eye_top - 26 * s
        self.visor_zero = eye_top + 3 * s
        self.gestures = sorted(gestures or [])
        # С какой стороны от сэра Лата «экран» (субтитры, события). Параметры EMO/POSE
        # заданы для экрана слева; если экран справа — наклоны, взгляд и крылья зеркалятся.
        self.mirror = screen_side == "right"
        # состояние
        self.w_emo = {k: 1.0 if k == "important" else 0.0 for k in EMO}
        self.w_pose = {k: 1.0 if k == "none" else 0.0 for k in POSE}
        self.open = 0.0
        self.width = 1.0
        self.lean = 0.0
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]
        self.next_saccade = self.rnd.uniform(0.8, 2.5)
        # моргание: 0.12–0.18 с, раз в 3–6 с, иногда двойное
        self.next_blink = self.rnd.uniform(1.5, 4.0)
        self.blink_t0, self.blink_dur = -10.0, 0.15
        self.double_pending = False
        self.last_blink = 0.0
        self.prev_emotion = None
        self.emotion_since = 0.0
        self.t = 0.0
        self._mouth_cache = {}

    # ---------- подготовка
    def _prep_eyes(self):
        s = self.s
        self.eyes = []
        src = self.canvas.astype(np.float32)
        for i, e in enumerate(LM["eyes"]):
            cx, cy = self.P(e["x"], e["y"])
            rx, ry = e["w"] * s * 0.56 + 1.5, e["h"] * s * 0.6 + 1.5
            bx0, bx1 = int(cx - rx * 2.0), int(cx + rx * 2.0) + 1
            by0, by1 = int(cy - ry * 2.0), int(cy + ry * 2.0) + 1
            patch = src[by0:by1, bx0:bx1].copy()
            ph, pw = patch.shape[:2]
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            lx, ly = cx - bx0, cy - by0
            # «глаз без глаза»: заливка плавным переходом от пикселей над и под глазом
            ya = max(0, int(ly - ry * 1.35) - 2)
            yb = min(ph - 1, int(ly + ry * 1.35) + 2)
            top = patch[max(0, ya - 2):ya + 1].mean(0)
            bot = patch[yb:min(ph, yb + 3)].mean(0)
            t = np.clip((yy - ya) / max(1, yb - ya), 0, 1)[..., None]
            fill_v = top[None] * (1 - t) + bot[None] * t
            xa = max(0, int(lx - rx * 1.35) - 2)
            xb = min(pw - 1, int(lx + rx * 1.35) + 2)
            left = patch[:, max(0, xa - 2):xa + 1].mean(1)
            right = patch[:, xb:min(pw, xb + 3)].mean(1)
            u = np.clip((xx - xa) / max(1, xb - xa), 0, 1)[..., None]
            fill_h = left[:, None] * (1 - u) + right[:, None] * u
            fill = 0.5 * fill_v + 0.5 * fill_h
            d = ((xx - lx) / (rx * 1.3)) ** 2 + ((yy - ly) / (ry * 1.3)) ** 2
            # гармоническая заливка: цвет плавно «растекается» от краёв области
            # внутрь, поэтому на месте глаза нет заметного пятна или кольца
            hole = d <= 1.0
            eyeless = patch.copy()
            eyeless[hole] = fill[hole]
            for _ in range(400):
                avg = (np.roll(eyeless, 1, 0) + np.roll(eyeless, -1, 0)
                       + np.roll(eyeless, 1, 1) + np.roll(eyeless, -1, 1)) / 4
                eyeless[hole] = avg[hole]
            hole_soft = np.clip((1.0 - d) * 4, 0, 1)
            # сам глаз — RGBA с мягкой эллиптической маской
            de = ((xx - lx) / (rx * 1.12)) ** 2 + ((yy - ly) / (ry * 1.12)) ** 2
            ea = np.clip((1.0 - de) * 5, 0, 1)
            eye_rgba = patch.copy()
            eye_rgba[..., 3] = ea * 255
            inner = 1 if cx < self.P(*self._beak_center())[0] else -1   # где внутренний уголок
            self.eyes.append(dict(cx=cx, cy=cy, rx=rx, ry=ry, box=(bx0, by0, bx1, by1),
                                  eyeless=eyeless, eye=Image.fromarray(eye_rgba.astype(np.uint8), "RGBA"),
                                  lx=lx, ly=ly, inner=inner, yy=yy, xx=xx, hole=hole_soft))

    def _beak_center(self):
        x0, y0, x1, y1 = LM["beak"]
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _prep_beak(self):
        s = self.s
        x0, y0, x1, y1 = LM["beak"]
        (bx0, by0), (bx1, by1) = self.P(x0, y0), self.P(x1, y1)
        self.beak = (bx0, by0, bx1, by1)
        self.split = self.P(0, LM["beak_split"])[1]
        bh = by1 - by0
        self.max_open = bh * 0.42
        pad = int(8 * s) + 4
        self.mbox = (int(bx0) - pad, int(by0) - pad, int(bx1) + pad + 1, int(by1 + self.max_open * 1.3) + pad)
        rx0, ry0, rx1, ry1 = self.mbox
        reg = Image.fromarray(self.canvas[ry0:ry1, rx0:rx1])
        arr = np.array(reg).astype(np.int32)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        orange = (r > 120) & (r - b > 50) & (g > b + 15) & (arr[..., 3] > 150)
        sp = int(round(self.split - ry0))
        lower = orange.copy()
        lower[:max(0, sp - 1)] = False
        upper = orange.copy()
        upper[sp + 1:] = False
        self.m_region = reg
        self.m_lower = Image.fromarray((lower * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)) \
            .filter(ImageFilter.GaussianBlur(0.8))
        self.m_upper = Image.fromarray((upper * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
        # цвет мордочки под клювом
        cxl = int((bx0 + bx1) / 2 - rx0)
        below = arr[min(arr.shape[0] - 1, int(by1 - ry0 + bh * 0.4)), max(0, cxl - 4):cxl + 4, :3]
        self.face_rgb = tuple(int(v) for v in np.median(below, axis=0))
        lo_px = arr[lower][:, :3]
        self.beak_shade = tuple(int(v * 0.62) for v in np.median(lo_px, axis=0)) if len(lo_px) else (120, 60, 20)

    def _prep_wings(self):
        base = np.array(base_image())
        self.wings = {}
        for side, (mask, pivot) in _detect_wings(base).items():
            m = Image.fromarray((mask * 255).astype(np.uint8))
            m = m.filter(ImageFilter.MaxFilter(max(3, int(BASE_W * 0.05) | 1))) \
                .filter(ImageFilter.GaussianBlur(BASE_W * 0.025))
            full = Image.new("L", (W, H), 0)
            full.paste(m, (OFF_X, OFF_Y))
            full = np.array(full.resize((self.cw, self.ch), Image.BILINEAR), np.float32) / 255
            ys, xs = np.nonzero(full > 0.02)
            px, py = self.P(*pivot)
            reach = float(ys.max() - py) * 0.9
            x0 = int(max(0, min(xs.min(), px - reach) - 4))
            x1 = int(min(self.cw, max(xs.max(), px + reach) + 4))
            y0 = int(max(0, py - reach * 0.6))
            y1 = int(min(self.ch, ys.max() + 4))
            wgt = np.zeros((y1 - y0, x1 - x0), np.float32)
            # вес «растекается» наружу, чтобы поднятое крыло было куда рисовать
            src_w = full[y0:y1, x0:x1]
            spread = Image.fromarray((src_w * 255).astype(np.uint8)) \
                .filter(ImageFilter.MaxFilter(max(3, int(reach * 0.35) | 1))) \
                .filter(ImageFilter.GaussianBlur(reach * 0.12))
            wgt = np.maximum(src_w, np.array(spread, np.float32) / 255)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            # у плеча вес плавно падает — наплечник почти не двигается
            dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            wgt *= _smoothstep(reach * 0.05, reach * 0.35, dist)
            sign = -1 if side == "l" else 1   # + угол: крыло уходит наружу и вверх
            self.wings[side] = dict(box=(x0, y0, x1, y1), w=wgt, px=px, py=py, xx=xx, yy=yy, sign=sign)

    def _prep_accents(self):
        s = self.s
        self.accents = {}
        hx, hy = self.P(BASE_W * 0.86, BASE_H * 0.22)
        k = s * BASE_W / 568
        for emo in ("scared", "angry", "thoughtful"):
            lay = Image.new("RGBA", (self.cw, self.ch), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            if emo == "scared":
                for (x, y, r) in ((hx, hy + 30 * k, 16 * k), (hx + 26 * k, hy + 80 * k, 10 * k)):
                    d.polygon([(x, y - r * 2.3), (x + r, y), (x - r, y)], fill=(150, 205, 255, 235))
                    d.ellipse((x - r, y - r, x + r, y + r), fill=(150, 205, 255, 235))
                    d.ellipse((x - r * .45, y - r * .5, x - r * .05, y - r * .05), fill=(255, 255, 255, 220))
            elif emo == "angry":
                cx, cy = hx + 10 * k, hy - 20 * k
                for a in (45, 135, 225, 315):
                    rad = math.radians(a)
                    d.line([(cx + 9 * k * math.cos(rad), cy + 9 * k * math.sin(rad)),
                            (cx + 30 * k * math.cos(rad + .4), cy + 30 * k * math.sin(rad + .4))],
                           fill=(225, 40, 50, 255), width=max(2, int(8 * k)))
            else:
                for (x, y, r) in ((hx, hy - 10 * k, 8 * k), (hx + 24 * k, hy - 50 * k, 12 * k),
                                  (hx + 52 * k, hy - 100 * k, 17 * k)):
                    d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 235),
                              outline=(60, 70, 90, 200), width=max(1, int(3 * k)))
            arr = np.array(lay, np.float32)
            ys, xs = np.nonzero(arr[..., 3])
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            self.accents[emo] = (box, arr[box[1]:box[3], box[0]:box[2]])

    # ---------- части лица
    def _mouth_patch(self, open_px: float, width: float) -> np.ndarray | None:
        key = (round(open_px * 2) / 2, round(width * 20) / 20)
        if key[0] < 0.5:
            return None
        if key in self._mouth_cache:
            return self._mouth_cache[key]
        op, wd = key
        reg = self.m_region
        rx0, ry0 = self.mbox[:2]
        bx0, by0, bx1, by1 = (v - o for v, o in zip(self.beak, (rx0, ry0, rx0, ry0)))
        sp = self.split - ry0
        bw, bh = bx1 - bx0, by1 - by0
        frac = op / self.max_open
        out = Image.composite(Image.new("RGBA", reg.size, self.beak_shade + (255,)), reg, self.m_lower)
        # рот: ширина зависит от гласной
        cx = (bx0 + bx1) / 2
        half = bw * min(0.47, 0.42 * wd)
        my0, my1 = sp - bh * 0.06, sp + op + (by1 - sp) * (0.55 + 0.2 * (1 - wd))
        mouth = Image.new("RGBA", reg.size, (0, 0, 0, 0))
        md = ImageDraw.Draw(mouth)
        md.ellipse((cx - half, my0, cx + half, my1), fill=(66, 16, 24, 255))
        if frac > 0.4:
            tw = half * 0.9
            md.ellipse((cx - tw, my1 - (my1 - my0) * 0.42, cx + tw, my1 + 1), fill=(210, 88, 100, 255))
        out.alpha_composite(mouth.filter(ImageFilter.GaussianBlur(0.6)))
        # нижняя половина клюва опускается; «о/у» — уже, «а» — вытягивается
        lower = reg.copy()
        lower.putalpha(Image.fromarray(np.minimum(np.array(self.m_lower), np.array(reg)[..., 3])))
        lb = (int(bx0) - 3, int(sp) - 2, int(bx1) + 4, int(by1) + 4)
        piece = lower.crop(lb)
        nw = max(2, int(piece.width * (0.86 + 0.14 * min(wd, 1.1))))
        nh = max(2, int(piece.height * (1 + 0.12 * frac)))
        piece = piece.resize((nw, nh), Image.BICUBIC)
        out.alpha_composite(piece, (int(lb[0] + (lb[2] - lb[0] - nw) / 2), int(lb[1] + op)))
        # верхняя половина чуть приподнимается
        upper = reg.copy()
        upper.putalpha(Image.fromarray(np.minimum(np.array(self.m_upper), np.array(reg)[..., 3])))
        out.alpha_composite(upper, (0, -int(round(op * 0.12))))
        res = np.array(out)
        if len(self._mouth_cache) > 400:
            self._mouth_cache.clear()
        self._mouth_cache[key] = res
        return res

    def _draw_eye(self, arr, e, gx, gy, scale, lid_top, lid_bot, tilt, curve):
        bx0, by0, bx1, by1 = e["box"]
        rx, ry, lx, ly = e["rx"], e["ry"], e["lx"], e["ly"]
        eyeless = e["eyeless"]
        ph, pw = eyeless.shape[:2]
        # глаз: сдвиг (взгляд) и масштаб (широко раскрыт)
        a = 1 / scale
        ox, oy = lx + gx * rx * 0.42, ly + gy * ry * 0.3
        eye = e["eye"].transform((pw, ph), Image.AFFINE, (a, 0, lx - a * ox, 0, a, ly - a * oy),
                                 resample=Image.BILINEAR)
        ev = np.array(eye, np.float32)
        al = ev[..., 3:4] / 255
        img = eyeless.copy()
        img[..., :3] = eyeless[..., :3] * (1 - al) + ev[..., :3] * al
        # веки цвета мордочки: край века повторяет изгиб глаза
        xx, yy = e["xx"], e["yy"]
        nx = np.clip((xx - ox) / (rx * scale), -1, 1)
        arch = np.sqrt(np.clip(1 - nx ** 2, 0, 1))
        rys = ry * scale
        y_top = oy - rys * (1 - 2 * lid_top) * (0.35 + 0.65 * arch) + tilt * rys * 0.6 * nx * e["inner"]
        y_bot = oy + rys * (1 - 2 * lid_bot) * (0.35 + 0.65 * arch) - curve * rys * arch
        closed = y_top >= y_bot - 0.6
        lid = np.clip(y_top - yy + 0.5, 0, 1) + np.clip(yy - y_bot + 0.5, 0, 1)
        lid = np.maximum(np.clip(lid, 0, 1), closed.astype(np.float32))[..., None]
        # веко — только в пределах глаза и его «впадины»; у открытого глаза ничего не меняем
        lid = lid * e["hole"][..., None] * min(1.0, (lid_top + lid_bot) * 8)
        img[..., :3] = img[..., :3] * (1 - lid) + eyeless[..., :3] * lid
        # тёмная линия по краю верхнего века; у закрытого глаза — дуга
        inside = (np.abs(nx) < 0.97) & (yy > oy - rys * 1.3) & (yy < oy + rys * 1.3)
        lw = max(1.0, rys * 0.12)
        col = np.array([30, 30, 36], np.float32)
        k = np.zeros_like(xx)
        if lid_top > 0.06:
            open_line = np.clip(1 - np.abs(yy - y_top) / lw, 0, 1) * inside * ~closed
            k = np.maximum(k, open_line * min(1.0, lid_top * 3) * 0.75)
        if closed.any():
            y_c = np.minimum(y_top, y_bot) + rys * 0.12 * arch
            wl = lw * (0.6 + 0.8 * arch)
            k = np.maximum(k, np.clip(1 - np.abs(yy - y_c) / wl, 0, 1) * closed * (np.abs(nx) < 0.9) * 0.9)
        img[..., :3] = img[..., :3] * (1 - k[..., None]) + col * k[..., None]
        arr[by0:by1, bx0:bx1] = img

    def _visor(self, arr, dy):
        if abs(dy) < 0.25:
            return
        y_end = int(self.visor_zero) + 1
        ys = np.arange(y_end, dtype=np.float32)
        f = 1 - _smoothstep(self.visor_full, self.visor_zero, ys)
        src_y = ys - dy * f
        y0 = np.floor(src_y).astype(np.int32)
        a = (src_y - y0)[:, None, None]
        top = arr[np.clip(y0, 0, arr.shape[0] - 1)].astype(np.float32)
        nxt = arr[np.clip(y0 + 1, 0, arr.shape[0] - 1)].astype(np.float32)
        out = top * (1 - a) + nxt * a
        out[y0 < 0] = 0
        arr[:y_end] = out

    def _wing(self, arr, side, angle):
        wg = self.wings.get(side)
        if not wg or abs(angle) < 0.3:
            return
        x0, y0, x1, y1 = wg["box"]
        th = np.radians(angle * wg["sign"]) * wg["w"]
        c, s_ = np.cos(th), np.sin(th)
        dx, dy = wg["xx"] - wg["px"], wg["yy"] - wg["py"]
        # обратное отображение: из какого места исходника взять пиксель
        sx = wg["px"] + c * dx - s_ * dy
        sy = wg["py"] + s_ * dx + c * dy
        arr[y0:y1, x0:x1] = _sample(arr, sx, sy)

    # ---------- кадр
    def frame(self, t: float, emotion: str, pose: str = "none", env: float = 0.0,
              vowel: str | None = None, emphatic: bool = False, speaking: bool = False) -> Image.Image:
        dt = self.dt
        self.t = t
        emotion = emotion if emotion in EMO else "important"
        pose = pose if pose in POSE else "none"
        for k in self.w_emo:
            self.w_emo[k] = _approach(self.w_emo[k], 1.0 if k == emotion else 0.0, dt, 0.08)
        for k in self.w_pose:
            self.w_pose[k] = _approach(self.w_pose[k], 1.0 if k == pose else 0.0, dt, 0.1)
        if emotion != self.prev_emotion:
            self.prev_emotion = emotion
            self.emotion_since = t
        E = {k: sum(self.w_emo[e] * EMO[e][k] for e in EMO) for k in EMO["important"] if k != "tint"}
        tint = [sum(self.w_emo[e] * EMO[e]["tint"][i] for e in EMO) for i in range(3)]
        Pz = {k: sum(self.w_pose[p] * POSE[p][k] for p in POSE) for k in POSE["none"]}

        # клюв: быстрое открытие, чуть более медленное закрытие
        tgt_open = min(1.0, env * VOWEL_OPEN.get(vowel, 0.7) * 1.25) if speaking else 0.0
        self.open = _approach(self.open, tgt_open, dt, 0.025 if tgt_open > self.open else 0.06)
        self.width = _approach(self.width, VOWEL_WIDTH.get(vowel, 0.95) if speaking else 1.0, dt, 0.05)
        open_ = self.open + (0.18 if emotion == "laughing" else 0) * self.w_emo["laughing"]

        # взгляд: случайные «саккады», чаще всего — к экрану
        if t >= self.next_saccade:
            r = self.rnd.random()
            if r < 0.5:
                self.gaze_target = [0.0, 0.0]
            elif r < 0.82:
                self.gaze_target = [-0.9, self.rnd.uniform(-0.25, 0.15)]
            else:
                self.gaze_target = [self.rnd.uniform(-1, 1), self.rnd.uniform(-0.6, 0.4)]
            fast = self.w_emo["scared"] > 0.5
            self.next_saccade = t + (self.rnd.uniform(0.35, 0.9) if fast else self.rnd.uniform(1.2, 3.8))
        for i in range(2):
            self.gaze[i] = _approach(self.gaze[i], self.gaze_target[i], dt, 0.035)
        gx = max(-1.2, min(1.2, self.gaze[0] * (1 - self.w_emo["thoughtful"] * 0.6) + E["gx"]))
        gy = max(-1.0, min(1.0, self.gaze[1] * (1 - self.w_emo["thoughtful"] * 0.6) + E["gy"]))

        # моргание: быстро закрыть (0.04 с), один кадр закрыто (0.04 с), открыть;
        # всего 0.12–0.18 с. Раз в 3–6 с, иногда сразу второе (двойное).
        if t >= self.next_blink:
            self.blink_t0 = self.next_blink
            self.blink_dur = self.rnd.uniform(0.12, 0.18)
            if not self.double_pending and self.rnd.random() < 0.18:
                self.double_pending = True
                self.next_blink = self.blink_t0 + self.blink_dur + self.rnd.uniform(0.08, 0.14)
            else:
                self.double_pending = False
                self.next_blink = self.blink_t0 + self.rnd.uniform(3.0, 6.0)
        bt = t - self.blink_t0
        blink = 0.0
        if 0 <= bt < 0.04:
            blink = bt / 0.04
        elif 0.04 <= bt < 0.08:
            blink = 1.0
        elif 0.08 <= bt < self.blink_dur:
            blink = 1 - (bt - 0.08) / (self.blink_dur - 0.08)
        self.last_blink = blink
        # прищур — только у смеющегося и не дольше 1 с (последние 0.2 с — плавно открывает)
        since = t - self.emotion_since
        squint = self.w_emo["laughing"] * (1.0 if since < 0.8 else max(0.0, 1 - (since - 0.8) / 0.2))
        L = EMO["laughing"]
        base_top, base_bot = L["lid_top"] * squint, L["lid_bot"] * squint
        lid_top = base_top + (1 - base_top) * blink
        lid_bot = base_bot * (1 - blink)
        E["curve"] = L["curve"] * squint
        E["tilt"] = 0.0

        # жесты крыльями на ударных словах
        wing = {"l": Pz["wing_l"], "r": Pz["wing_r"]}
        hop_g = 0.0
        for (gt, side, power) in self.gestures:
            if gt > t + 0.05:
                break
            u = t - gt
            if 0 <= u < 0.65:
                env_g = math.sin(math.pi * min(1, u / 0.16) / 2) * (1 - _smoothstep(0.2, 0.65, u))
                wing[side] += power * env_g * (1 + 0.25 * math.sin(u * 30))
                hop_g = max(hop_g, env_g * 0.25)

        # тело: дыхание, кивки, наклон на важных фразах, прыжки, дрожь
        self.lean = _approach(self.lean, 1.0 if emphatic else 0.0, dt, 0.15)
        sway = E["sway"]
        angle = (E["angle"] + Pz["angle"] + math.sin(2 * math.pi * t / 3.4) * 1.5 * sway
                 + self.lean * 4.0 + self.open * 1.4)
        breath = math.sin(2 * math.pi * t / 2.6)
        sy = E["sy"] * Pz["sy"] * (1 + 0.012 * breath)
        sx = E["sx"] * (1 - 0.006 * breath)
        dx = E["dx"] + Pz["dx"] - self.lean * 0.012
        dy = E["dy"] - 0.004 * max(0.0, breath) - self.open * 0.004
        hop = E["hop"] * abs(math.sin(2 * math.pi * t * 2.1)) * 0.03 + hop_g * 0.02
        dy -= hop
        angle += E["hop"] * math.sin(2 * math.pi * t * 2.1) * 2.5
        if E["shake"] > 0.05:
            k = E["shake"] * self.s / self.cw
            dx += k * (math.sin(2 * math.pi * 11.3 * t) + 0.6 * math.sin(2 * math.pi * 17.9 * t + 1))
            dy += k * 0.6 * math.sin(2 * math.pi * 13.1 * t + 2)
        if self.mirror:
            angle, dx, gx = -angle, -dx, -gx
            wing = {"l": wing["r"], "r": wing["l"]}
        state = dict(open=open_, width=self.width, gx=gx, gy=gy, eye=E["eye"], lid_top=lid_top,
                     lid_bot=lid_bot, tilt=E["tilt"], curve=E["curve"] * (1 - blink), visor=E["visor"],
                     wing=wing, tint=tint, accents={k: self.w_emo[k] for k in self.accents},
                     angle=angle, scale=E["scale"] * Pz["scale"], sx=sx, sy=sy, dx=dx, dy=dy)
        return self.render(state)

    def render(self, st) -> Image.Image:
        arr = self.canvas.copy()
        for e in self.eyes:
            self._draw_eye(arr, e, st["gx"], st["gy"], st["eye"], st["lid_top"], st["lid_bot"],
                           st["tilt"], st["curve"])
        mp = self._mouth_patch(st["open"] * self.max_open, st["width"])
        if mp is not None:
            x0, y0, x1, y1 = self.mbox
            arr[y0:y1, x0:x1] = mp
        self._visor(arr, st["visor"] * self.s)
        for side, ang in st["wing"].items():
            self._wing(arr, side, ang)
        f = None
        tint = st["tint"]
        if max(abs(v - 1) for v in tint) > 0.004:
            f = arr.astype(np.float32)
            f[..., :3] *= np.array(tint, np.float32)
        for emo, wgt in st["accents"].items():
            if wgt < 0.02:
                continue
            if f is None:
                f = arr.astype(np.float32)
            (x0, y0, x1, y1), acc = self.accents[emo]
            a = acc[..., 3:4] / 255 * wgt
            reg = f[y0:y1, x0:x1]
            reg[..., :3] = reg[..., :3] * (1 - a) + acc[..., :3] * a
            reg[..., 3:4] = np.maximum(reg[..., 3:4], acc[..., 3:4] * wgt)
        if f is not None:
            arr = np.clip(f, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGBA")
        return transform(img, st["angle"], st["scale"], st["sx"], st["sy"], st["dx"], st["dy"])

    def still(self, emotion: str, pose: str = "none", open_: float = 0.0, width: float = 1.0,
              blink: float = 0.0, gaze=(0.0, 0.0)) -> Image.Image:
        """Статичный кадр в «устоявшемся» положении (для PNG)."""
        E, Pz = EMO[emotion], POSE[pose]
        return self.render(dict(
            open=open_, width=width, gx=E["gx"] + gaze[0], gy=E["gy"] + gaze[1], eye=E["eye"],
            lid_top=E["lid_top"] + (1 - E["lid_top"]) * blink, lid_bot=E["lid_bot"] * (1 - blink),
            tilt=E["tilt"], curve=E["curve"], visor=E["visor"],
            wing={"l": Pz["wing_l"], "r": Pz["wing_r"]}, tint=E["tint"],
            accents={k: 1.0 if k == emotion else 0.0 for k in self.accents},
            angle=E["angle"] + Pz["angle"], scale=E["scale"] * Pz["scale"], sx=E["sx"],
            sy=E["sy"] * Pz["sy"], dx=E["dx"] + Pz["dx"], dy=E["dy"]))


# ---------------------------------------------------------- экспорт ассетов ---

def export_assets(out_dir=ASSETS):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "important": "важный", "surprised": "удивлённый", "laughing": "смеющийся",
        "angry": "злой", "thoughtful": "задумчивый", "scared": "испуганный",
    }
    an = Animator(25, H)
    files = []
    for e in EMOTION_KEYS:
        f = out_dir / f"emotion_{e}.png"
        an.still(e, open_=0.7 if e in ("laughing", "surprised") else 0.0).save(f)
        files.append((f, names[e]))
    pose_emotion = {"point": "important", "sword": "angry", "shield": "scared"}
    pose_names = {"point": "указывает", "sword": "с мечом", "shield": "за щитом"}
    for pz, e in pose_emotion.items():
        f = out_dir / f"pose_{pz}.png"
        an.still(e, pz).save(f)
        files.append((f, pose_names[pz]))
    anim = out_dir / "anim"
    anim.mkdir(exist_ok=True)
    for old in anim.glob("*.png"):
        old.unlink()
    for i, (o, w, label) in enumerate(((0.0, 1.0, "closed"), (0.35, 0.95, "small"), (1.0, 1.0, "a"),
                                       (0.8, 0.72, "o"), (0.6, 0.6, "u"), (0.5, 1.14, "i"))):
        an.still("important", open_=o, width=w).save(anim / f"mouth_{i}_{label}.png")
    an.still("important", blink=1.0).save(anim / "blink.png")
    an.still("important", gaze=(-1.0, 0.0)).save(anim / "look_left.png")

    from .config import font
    cols = 5
    tw = 300
    th = round(tw * H / W)
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGBA", (cols * tw, rows * (th + 50)), (60, 70, 92, 255))
    d = ImageDraw.Draw(sheet)
    for i, (f, label) in enumerate(files):
        im = Image.open(f).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * (th + 50)
        sheet.alpha_composite(im, (x, y))
        d.text((x + tw / 2, y + th + 22), label, font=font(26), fill="white", anchor="mm")
    sheet.convert("RGB").save(out_dir / "preview_sheet.jpg", quality=90)
    print(f"Сохранено {len(files)} PNG + кадры клюва, моргания и взгляда в {out_dir}")


if __name__ == "__main__":
    export_assets()
