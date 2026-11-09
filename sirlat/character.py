"""Сэр Пингвин Лат — оригинальный пингвин-рыцарь, нарисованный кодом (Pillow).

Пингвин рисуется по пояс, в мультяшном стиле: стальной шлем с поднятым
забралом и красным плюмажем, жёлтые «брови-хохолки», оранжевый клюв,
красная перевязь с пряжкой на белом брюшке и стальные наплечники.

Любой кадр задаётся параметрами:
    emotion — important / surprised / laughing / angry / thoughtful / scared
    pose    — none / point / sword / shield
    mouth   — 0..3 (насколько открыт клюв, для липсинка)
    blink   — True, если глаза закрыты (моргание)

Запуск `python -m sirlat.character` сохраняет все PNG в assets/sir_lat/.
"""
from __future__ import annotations

import math
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

from .config import ASSETS

W, H = 800, 900          # размер итогового спрайта
S = 2                    # суперсэмплинг для сглаживания

EMOTION_KEYS = ["important", "surprised", "laughing", "angry", "thoughtful", "scared"]
POSE_KEYS = ["none", "point", "sword", "shield"]
MOUTH_LEVELS = 4

# Палитра
INK = (22, 24, 34, 255)            # контур
BODY = (32, 37, 52, 255)           # чёрное оперение
BODY_HI = (58, 66, 88, 255)
BELLY = (246, 243, 233, 255)
BELLY_SH = (214, 210, 200, 255)
STEEL = (178, 188, 202, 255)
STEEL_HI = (226, 233, 242, 255)
STEEL_SH = (118, 128, 146, 255)
STEEL_DK = (84, 92, 110, 255)
PLUME = (206, 44, 52, 255)
PLUME_HI = (240, 96, 92, 255)
BEAK = (255, 150, 40, 255)
BEAK_SH = (222, 108, 22, 255)
MOUTH_IN = (110, 20, 34, 255)
TONGUE = (236, 96, 112, 255)
BROW = (255, 210, 60, 255)
SASH = (170, 32, 44, 255)
GOLD = (236, 186, 64, 255)
GOLD_SH = (180, 128, 30, 255)
EYE_W = (255, 255, 255, 255)
PUPIL = (20, 20, 28, 255)
BLUSH = (255, 120, 140, 150)
SWEAT = (140, 200, 255, 230)
WOOD = (120, 76, 40, 255)
SHIELD_BLUE = (40, 76, 150, 255)
SHIELD_BLUE_HI = (70, 112, 190, 255)

OUT = 6  # толщина контура (в итоговых пикселях)


def _s(v):
    """Масштабирует координаты/размеры в суперсэмплинг."""
    if isinstance(v, (list, tuple)):
        return type(v)(_s(x) for x in v)
    return v * S


class Pen:
    """Обёртка над ImageDraw, работающая в координатах итогового спрайта."""

    def __init__(self, img: Image.Image):
        self.img = img
        self.d = ImageDraw.Draw(img)

    def ellipse(self, box, fill, outline=INK, width=OUT):
        self.d.ellipse(_s(box), fill=fill, outline=outline, width=int(width * S) if outline else 0)

    def polygon(self, pts, fill, outline=INK, width=OUT):
        pts = [_s(p) for p in pts]
        self.d.polygon(pts, fill=fill)
        if outline:
            self.d.line(pts + [pts[0]], fill=outline, width=int(width * S), joint="curve")

    def line(self, pts, fill=INK, width=OUT):
        self.d.line([_s(p) for p in pts], fill=fill, width=int(width * S), joint="curve")

    def rrect(self, box, r, fill, outline=INK, width=OUT):
        self.d.rounded_rectangle(_s(box), radius=r * S, fill=fill,
                                 outline=outline, width=int(width * S) if outline else 0)

    def arc(self, box, start, end, fill=INK, width=OUT):
        self.d.arc(_s(box), start, end, fill=fill, width=int(width * S))

    def chord(self, box, start, end, fill, outline=INK, width=OUT):
        self.d.chord(_s(box), start, end, fill=fill, outline=outline,
                     width=int(width * S) if outline else 0)

    def circle(self, c, r, fill, outline=INK, width=OUT):
        self.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill, outline, width)


def _layer():
    return Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))


def _bezier(p0, p1, p2, n=24):
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------- крылья ---

@lru_cache(maxsize=32)
def _flipper(pivot, length, width, angle_deg, with_pauldron=True):
    """Крыло-ласт: вытянутая капля, повёрнутая вокруг плеча.

    angle_deg = 0 — крыло висит вниз; положительный угол — поворот против
    часовой стрелки (кончик уходит вправо).
    """
    lay = _layer()
    p = Pen(lay)
    cx, cy = pivot
    # капля, направленная вниз из точки pivot
    pts = []
    n = 40
    for i in range(n + 1):
        t = i / n * math.pi
        # профиль: широкое у плеча, острое к кончику
        r = width / 2 * math.sin(t) * (1.0 - 0.35 * (i / n))
        y = cy + (1 - math.cos(t)) / 2 * length
        pts.append((cx + r, y))
    for i in range(n, -1, -1):
        t = i / n * math.pi
        r = width / 2 * math.sin(t) * (1.0 - 0.35 * (i / n))
        y = cy + (1 - math.cos(t)) / 2 * length
        pts.append((cx - r * 0.9, y))
    p.polygon(pts, BODY)
    # блик
    hl = [(cx - width * 0.12, cy + length * 0.15), (cx - width * 0.05, cy + length * 0.7)]
    p.line(hl, fill=BODY_HI, width=8)
    if with_pauldron:
        p.chord((cx - width * 0.75, cy - width * 0.55, cx + width * 0.75, cy + width * 0.75),
                180, 360, STEEL)
        p.chord((cx - width * 0.75, cy - width * 0.55, cx + width * 0.75, cy + width * 0.75),
                0, 25, STEEL, outline=None)
        p.arc((cx - width * 0.55, cy - width * 0.35, cx + width * 0.55, cy + width * 0.55),
              200, 330, fill=STEEL_HI, width=6)
        p.circle((cx, cy - width * 0.25), 6, STEEL_DK, outline=None)
    return lay.rotate(angle_deg, resample=Image.BICUBIC, center=_s(pivot))


# ----------------------------------------------------------------- предметы ---

def _sword(img):
    p = Pen(img)
    x = 640
    # клинок
    p.polygon([(x - 18, 600), (x - 18, 120), (x, 70), (x + 18, 120), (x + 18, 600)], STEEL)
    p.line([(x, 110), (x, 590)], fill=STEEL_HI, width=6)
    # гарда
    p.rrect((x - 80, 596, x + 80, 626), 12, GOLD)
    p.circle((x - 80, 611), 16, GOLD)
    p.circle((x + 80, 611), 16, GOLD)
    # рукоять и навершие
    p.rrect((x - 14, 624, x + 14, 730), 6, WOOD)
    p.circle((x, 742), 22, GOLD)


def _shield(img):
    p = Pen(img, )
    top, bot = 395, 905
    l, r = 175, 625
    cx = (l + r) / 2
    pts = [(l, top + 20)] + _bezier((l, top + 260), (l + 10, bot - 120), (cx, bot)) + \
          _bezier((cx, bot), (r - 10, bot - 120), (r, top + 260)) + [(r, top + 20)] + \
          _bezier((r, top + 20), (cx, top - 30), (l, top + 20))
    p.polygon(pts, SHIELD_BLUE, width=8)
    # кант
    inner = [(l + 28, top + 40)] + _bezier((l + 28, top + 260), (l + 38, bot - 140), (cx, bot - 40)) + \
            _bezier((cx, bot - 40), (r - 38, bot - 140), (r - 28, top + 260)) + [(r - 28, top + 40)] + \
            _bezier((r - 28, top + 40), (cx, top), (l + 28, top + 40))
    p.line(inner + [inner[0]], fill=GOLD, width=10)
    # герб: рыбка
    fx, fy = cx, top + 230
    p.ellipse((fx - 95, fy - 45, fx + 55, fy + 45), GOLD)
    p.polygon([(fx + 45, fy), (fx + 115, fy - 50), (fx + 115, fy + 50)], GOLD)
    p.circle((fx - 55, fy - 8), 9, INK, outline=None)
    p.arc((fx - 20, fy - 35, fx + 30, fy + 35), 300, 60, fill=GOLD_SH, width=6)
    # блик
    p.line(_bezier((l + 55, top + 80), (l + 50, top + 260), (l + 110, top + 400)),
           fill=SHIELD_BLUE_HI, width=14)


# ------------------------------------------------------------------- голова ---

FACE_BOX = (262, 170, 538, 420)   # проём шлема, где видно мордочку
EYE_L = (345, 262)
EYE_R = (455, 262)


def _eye_params(emotion):
    """Возвращает параметры глаз: (rx, ry, pupil_r, pupil_dx, pupil_dy, lid_l, lid_r)

    lid — (высота века сверху в долях 0..1, наклон в долях): наклон > 0 —
    внутренний угол века ниже (злость), < 0 — внешний ниже (грусть/страх).
    """
    return {
        "important":  (40, 44, 17, 0, 4, (0.42, -0.05), (0.42, 0.05)),
        "surprised":  (46, 56, 13, 0, 0, (0.0, 0.0), (0.0, 0.0)),
        "laughing":   (40, 44, 17, 0, 0, (0.0, 0.0), (0.0, 0.0)),
        "angry":      (40, 42, 15, 0, 4, (0.30, 0.35), (0.30, 0.35)),
        "thoughtful": (40, 44, 16, -14, -16, (0.18, 0.0), (0.18, 0.0)),
        "scared":     (46, 54, 10, 0, 2, (0.05, -0.25), (0.05, -0.25)),
    }[emotion]


def _draw_eye(img, center, side, emotion, blink):
    p = Pen(img)
    cx, cy = center
    rx, ry, pr, pdx, pdy, lid_l, lid_r = _eye_params(emotion)
    lid = lid_l if side < 0 else lid_r
    if emotion == "laughing":
        # закрытые от смеха глаза «^ ^»
        p.line([(cx - 34, cy + 10), (cx, cy - 22), (cx + 34, cy + 10)], fill=EYE_W, width=11)
        return
    if blink:
        p.line(_bezier((cx - rx, cy + 4), (cx, cy + 22), (cx + rx, cy + 4)), fill=EYE_W, width=9)
        return
    eye = _layer()
    ep = Pen(eye)
    ep.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), EYE_W, outline=None)
    # зрачок + блики
    px, py = cx + pdx, cy + pdy
    ep.circle((px, py), pr, PUPIL, outline=None)
    ep.circle((px - pr * 0.35, py - pr * 0.4), max(4, pr * 0.32), EYE_W, outline=None)
    ep.circle((px + pr * 0.4, py + pr * 0.35), max(2, pr * 0.14), EYE_W, outline=None)
    # веко цвета мордочки
    h, tilt = lid
    if h > 0 or tilt:
        inner = -side  # внутренний угол глаза смотрит к центру морды
        y_inner = cy - ry + 2 * ry * (h + max(tilt, 0))
        y_outer = cy - ry + 2 * ry * (h + max(-tilt, 0))
        x_inner = cx + inner * rx * 1.1
        x_outer = cx - inner * rx * 1.1
        poly = [(x_inner, cy - ry - 30), (x_outer, cy - ry - 30), (x_outer, y_outer), (x_inner, y_inner)]
        ep.d.polygon([_s(q) for q in poly], fill=(0, 0, 0, 0))
        mask = eye.split()[3]
        eye2 = _layer()
        eye2.paste(eye, (0, 0), mask)
        eye = eye2
        ep = Pen(eye)
        ep.line([(x_inner, y_inner), (x_outer, y_outer)], fill=BODY_HI, width=6)
    img.alpha_composite(eye)


def _draw_brows(p, emotion):
    """Жёлтые брови-хохолки (как у хохлатых пингвинов)."""
    L, R = EYE_L, EYE_R
    shapes = {
        # (внутр. точка dy, внешн. точка dy, изгиб) относительно верха глаза
        "important":  (-18, -30, -6),
        "surprised":  (-48, -46, -16),
        "laughing":   (-40, -36, -10),
        "angry":      (-2, -40, 4),
        "thoughtful": None,
        "scared":     (-50, -30, 6),
    }
    if emotion == "thoughtful":
        _brow(p, L, -46, -40, -12, -1)     # левая приподнята
        _brow(p, R, -16, -22, 0, 1)        # правая опущена
        return
    a, b, c = shapes[emotion]
    _brow(p, L, a, b, c, -1)
    _brow(p, R, a, b, c, 1)


def _brow(p, eye, dy_in, dy_out, bend, side):
    cx, cy = eye
    top = cy - 44
    x_in = cx - side * 38
    x_out = cx + side * 46
    p0 = (x_in, top + dy_in)
    p2 = (x_out, top + dy_out)
    p1 = ((x_in + x_out) / 2, (p0[1] + p2[1]) / 2 + bend)
    pts = _bezier(p0, p1, p2, 16)
    p.line(pts, fill=INK, width=20)
    p.line(pts, fill=BROW, width=12)
    # пёрышко-хохолок на внешнем конце
    tip = (x_out + side * 26, top + dy_out - 16)
    p.line([p2, tip], fill=INK, width=14)
    p.line([p2, tip], fill=BROW, width=7)


def _draw_beak(img, emotion, mouth):
    """Клюв. mouth 0..3 — степень открытия нижней части."""
    p = Pen(img)
    cx = 400
    top = 318
    base = {"surprised": 1, "laughing": 2, "scared": 1}.get(emotion, 0)
    m = max(mouth, base) if mouth > 0 or base else 0
    open_px = [0, 16, 30, 44][min(m, 3)]
    if emotion == "laughing":
        open_px += 8
    # рот (внутренность) между надклювьем и подклювьем
    if open_px > 0:
        p.polygon([(cx - 46, top + 14), (cx + 46, top + 14),
                   (cx + 24, top + 30 + open_px), (cx, top + 40 + open_px),
                   (cx - 24, top + 30 + open_px)], MOUTH_IN, width=5)
        if open_px >= 28:
            p.ellipse((cx - 22, top + 18 + open_px, cx + 22, top + 42 + open_px), TONGUE, outline=None)
    # подклювье
    lo = top + 14 + open_px
    p.polygon([(cx - 40, lo), (cx + 40, lo), (cx + 16, lo + 22), (cx, lo + 30), (cx - 16, lo + 22)],
              BEAK_SH, width=5)
    # надклювье
    up = [(cx - 52, top + 2), (cx - 30, top - 14), (cx + 30, top - 14), (cx + 52, top + 2),
          (cx + 26, top + 22), (cx, top + 34), (cx - 26, top + 22)]
    p.polygon(up, BEAK, width=5)
    p.line([(cx - 28, top - 4), (cx + 6, top - 8)], fill=(255, 205, 130, 255), width=6)
    p.circle((cx - 14, top + 2), 3, BEAK_SH, outline=None)
    p.circle((cx + 14, top + 2), 3, BEAK_SH, outline=None)
    if emotion == "angry" and open_px == 0:
        # сжатый клюв: «зубастая» черта
        p.line([(cx - 34, top + 18), (cx - 12, top + 24), (cx + 12, top + 18), (cx + 34, top + 24)],
               fill=BEAK_SH, width=4)


def _draw_helmet(img, emotion, visor_shift):
    p = Pen(img)
    # плюмаж
    plume = Pen(img)
    pts = _bezier((420, 70), (300, -10), (170, 150), 30) + _bezier((170, 150), (230, 40), (400, 100), 30)
    plume.polygon(pts, PLUME, width=6)
    plume.line(_bezier((410, 76), (300, 20), (200, 128), 20), fill=PLUME_HI, width=8)
    plume.line(_bezier((395, 90), (300, 50), (230, 120), 20), fill=(160, 28, 38, 255), width=4)
    # купол шлема
    p.ellipse((200, 62, 600, 470), STEEL, width=7)
    # блик купола
    p.arc((232, 92, 568, 440), 200, 260, fill=STEEL_HI, width=14)
    # гребень
    p.line([(400, 66), (400, 150)], fill=STEEL_SH, width=10)
    # проём для мордочки (тёмное оперение головы)
    fx0, fy0, fx1, fy1 = FACE_BOX
    p.rrect((fx0, fy0, fx1, fy1), 90, BODY, width=7)
    # нижний обод шлема / бармица
    p.chord((206, 300, 594, 500), 0, 180, STEEL, width=7)
    p.rrect((fx0 + 20, fy1 - 40, fx1 - 20, fy1 + 6), 30, BODY, outline=None)
    p.arc((206, 300, 594, 500), 20, 160, fill=STEEL_SH, width=6)
    for i in range(7):
        a = math.radians(20 + i * 23.3)
        x = 400 + 194 * math.cos(a) * 0.98
        y = 400 + 100 * math.sin(a) * 0.98
        p.circle((x, y - 4), 6, STEEL_DK, outline=None)
    # поднятое забрало (полоса с прорезями над проёмом)
    vy = 110 + visor_shift
    p.rrect((250, vy, 550, vy + 78), 38, STEEL, width=7)
    p.arc((262, vy + 6, 538, vy + 70), 200, 340, fill=STEEL_HI, width=6)
    for i in range(5):
        x = 300 + i * 50
        p.rrect((x - 14, vy + 30, x + 14, vy + 46), 6, STEEL_DK, outline=None)
    # шарниры забрала
    p.circle((236, vy + 70), 16, STEEL_SH)
    p.circle((564, vy + 70), 16, STEEL_SH)


def _draw_head(img, emotion, mouth, blink):
    visor_shift = {"surprised": -26, "scared": 52, "angry": 8, "important": -6}.get(emotion, 0)
    _draw_helmet(img, emotion, 0 if emotion == "scared" else visor_shift)
    p = Pen(img)
    _draw_eye(img, EYE_L, -1, emotion, blink)
    _draw_eye(img, EYE_R, 1, emotion, blink)
    _draw_brows(p, emotion)
    if emotion == "laughing":
        blush = _layer()
        bp = Pen(blush)
        bp.ellipse((282, 300, 340, 336), BLUSH, outline=None)
        bp.ellipse((460, 300, 518, 336), BLUSH, outline=None)
        img.alpha_composite(blush.filter(ImageFilter.GaussianBlur(4 * S)))
    _draw_beak(img, emotion, mouth)
    if emotion == "scared":
        # забрало «съехало» от страха и прикрыло лоб
        vy = 110 + visor_shift
        p.rrect((250, vy, 550, vy + 78), 38, STEEL, width=7)
        p.arc((262, vy + 6, 538, vy + 70), 200, 340, fill=STEEL_HI, width=6)
        for i in range(5):
            x = 300 + i * 50
            p.rrect((x - 14, vy + 30, x + 14, vy + 46), 6, STEEL_DK, outline=None)
        # капли пота
        for (x, y, r) in ((585, 200, 20), (612, 268, 13)):
            p.polygon([(x, y - r * 2.2), (x + r, y), (x - r, y)], SWEAT, width=4)
            p.circle((x, y + 2), r, SWEAT, width=4)
    if emotion == "angry":
        # «венка» злости на шлеме
        cx, cy = 548, 150
        for a in (45, 135, 225, 315):
            r = math.radians(a)
            p.line([(cx + 10 * math.cos(r), cy + 10 * math.sin(r)),
                    (cx + 30 * math.cos(r + 0.35), cy + 30 * math.sin(r + 0.35))],
                   fill=(230, 40, 50, 255), width=9)
    if emotion == "thoughtful":
        for i, (x, y, r) in enumerate(((600, 140, 9), (630, 100, 13), (668, 54, 18))):
            p.circle((x, y), r, (255, 255, 255, 235), width=4)


# -------------------------------------------------------------------- тело ---

def _draw_body(img):
    p = Pen(img)
    p.ellipse((130, 330, 670, 1180), BODY, width=7)
    p.arc((170, 360, 630, 1120), 200, 250, fill=BODY_HI, width=12)
    p.ellipse((236, 430, 564, 1150), BELLY, width=6)
    p.arc((250, 450, 550, 1130), 20, 80, fill=BELLY_SH, width=14)
    # перевязь
    p.polygon([(226, 560), (262, 520), (580, 820), (560, 870)], SASH, width=6)
    p.rrect((370, 650, 430, 712), 10, GOLD, width=5)
    p.rrect((386, 666, 414, 696), 4, GOLD_SH, outline=None)
    # горжет (стальной воротник)
    p.chord((214, 360, 586, 520), 0, 180, STEEL, width=7)
    p.arc((236, 372, 564, 502), 25, 155, fill=STEEL_HI, width=6)
    for x in (270, 335, 400, 465, 530):
        y = 440 + 70 * math.sqrt(max(0, 1 - ((x - 400) / 186) ** 2)) - 6
        p.circle((x, y), 6, STEEL_DK, outline=None)


# ------------------------------------------------------------------- сборка ---

LEFT_PIVOT = (175, 500)
RIGHT_PIVOT = (625, 500)


@lru_cache(maxsize=64)
def render(emotion: str = "important", pose: str = "none", mouth: int = 0,
           blink: bool = False) -> Image.Image:
    """Рисует сэра Лата и возвращает RGBA-изображение W×H."""
    if emotion not in EMOTION_KEYS:
        emotion = "important"
    if pose not in POSE_KEYS:
        pose = "none"
    img = _layer()

    # крылья позади тела
    if pose == "point":
        img.alpha_composite(_flipper(RIGHT_PIVOT, 300, 120, 18))
    elif pose == "sword":
        img.alpha_composite(_flipper(LEFT_PIVOT, 300, 120, -14))
    elif pose == "shield":
        pass
    else:
        img.alpha_composite(_flipper(LEFT_PIVOT, 300, 120, -16))
        img.alpha_composite(_flipper(RIGHT_PIVOT, 300, 120, 16))

    _draw_body(img)
    _draw_head(img, emotion, mouth, blink)

    if pose == "point":
        # крыло указывает вверх-влево (на экран, т.к. сэр Лат стоит справа)
        img.alpha_composite(_flipper((200, 480), 240, 120, -148))
    elif pose == "sword":
        _sword(img)
        img.alpha_composite(_flipper(RIGHT_PIVOT, 260, 120, -12))
    elif pose == "shield":
        img.alpha_composite(_flipper(LEFT_PIVOT, 260, 120, -40))
        img.alpha_composite(_flipper(RIGHT_PIVOT, 260, 120, 40))
        _shield(img)

    out = img.resize((W, H), Image.LANCZOS)
    return out


@lru_cache(maxsize=192)
def frame_sprite(emotion: str, pose: str, mouth: int, blink: bool, angle_step: int,
                 height: int) -> Image.Image:
    """Спрайт для видео: масштабированный и слегка повёрнутый (покачивание)."""
    base = render(emotion, pose, mouth, blink)
    w = round(W * height / H)
    spr = base.resize((w, height), Image.LANCZOS)
    if angle_step:
        spr = spr.rotate(angle_step * 0.6, resample=Image.BICUBIC,
                         center=(w / 2, height), expand=False)
    return spr


# ---------------------------------------------------------- экспорт ассетов ---

def export_assets(out_dir=ASSETS):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "important": "важный", "surprised": "удивлённый", "laughing": "смеющийся",
        "angry": "злой", "thoughtful": "задумчивый", "scared": "испуганный",
    }
    files = []
    for e in EMOTION_KEYS:
        f = out_dir / f"emotion_{e}.png"
        render(e, "none", 0, False).save(f)
        files.append((f, names[e]))
    pose_emotion = {"point": "important", "sword": "angry", "shield": "scared"}
    pose_names = {"point": "указывает крылом", "sword": "держит меч", "shield": "прячется за щитом"}
    for pz, e in pose_emotion.items():
        f = out_dir / f"pose_{pz}.png"
        render(e, pz, 0, False).save(f)
        files.append((f, pose_names[pz]))
    # кадры липсинка и моргания
    anim = out_dir / "anim"
    anim.mkdir(exist_ok=True)
    for m in range(MOUTH_LEVELS):
        render("important", "none", m, False).save(anim / f"mouth_{m}.png")
    render("important", "none", 0, True).save(anim / "blink.png")

    # контактный лист для быстрого просмотра
    from .config import font
    cols = 5
    tw, th = 320, 360
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGBA", (cols * tw, rows * (th + 50)), (60, 70, 92, 255))
    d = ImageDraw.Draw(sheet)
    for i, (f, label) in enumerate(files):
        im = Image.open(f).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * (th + 50)
        sheet.alpha_composite(im, (x, y))
        d.text((x + tw / 2, y + th + 22), label, font=font(26), fill="white", anchor="mm")
    sheet.convert("RGB").save(out_dir / "preview_sheet.jpg", quality=90)
    print(f"Сохранено {len(files)} PNG + анимационные кадры в {out_dir}")


if __name__ == "__main__":
    export_assets()
