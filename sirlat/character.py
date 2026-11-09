"""Сэр Пингвин Лат — анимация поверх готовой картинки персонажа.

Основа — assets/sir_lat/sir_lat.png (персонаж без фона, по пояс), её делает
`python -m sirlat.prepare_character` из исходника в assets/sir_lat/source/.
Положение клюва и глаз берётся из assets/sir_lat/landmarks.json.

Что анимируется:
  * клюв — нижняя половина клюва опускается в такт громкости голоса
    (деформация области клюва, за ней появляется рот);
  * моргание — глаза закрываются «веками» цвета мордочки;
  * покачивание — лёгкий наклон и дыхание;
  * эмоции и позы — наклон, масштаб, сжатие/растяжение, оттенок и небольшие
    значки (капли пота, «венка» злости, пузырьки мыслей).

Запуск `python -m sirlat.character` сохраняет кадры эмоций и поз в assets/sir_lat/.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import ASSETS

BASE_FILE = ASSETS / "sir_lat.png"
LANDMARKS_FILE = ASSETS / "landmarks.json"

EMOTION_KEYS = ["important", "surprised", "laughing", "angry", "thoughtful", "scared"]
POSE_KEYS = ["none", "point", "sword", "shield"]
MOUTH_LEVELS = 4

PAD_X, PAD_TOP = 0.14, 0.24   # запас холста под наклоны и увеличение


def _load_landmarks():
    if not LANDMARKS_FILE.exists():
        raise SystemExit("Нет assets/sir_lat/landmarks.json — запустите: python -m sirlat.prepare_character")
    return json.loads(LANDMARKS_FILE.read_text(encoding="utf-8"))


LM = _load_landmarks()
BASE_W, BASE_H = LM["size"]
# размер холста спрайта (персонаж стоит внизу по центру)
W = int(BASE_W * (1 + 2 * PAD_X))
H = int(BASE_H * (1 + PAD_TOP))
OFF_X, OFF_Y = (W - BASE_W) // 2, H - BASE_H


@lru_cache(maxsize=1)
def base_image() -> Image.Image:
    return Image.open(BASE_FILE).convert("RGBA")


# --------------------------------------------------------------- параметры ---

@dataclass(frozen=True)
class Look:
    angle: float = 0.0    # наклон, градусы (+ — к экрану, т.е. влево)
    scale: float = 1.0
    sx: float = 1.0       # сжатие/растяжение
    sy: float = 1.0
    dx: float = 0.0       # сдвиг в долях ширины
    dy: float = 0.0       # сдвиг в долях высоты (− — вверх)
    sway: float = 1.0     # сила покачивания
    bob: float = 1.0      # сила «дыхания»
    speed: float = 1.0    # темп покачивания
    shake: float = 0.0    # дрожь, пиксели (в масштабе исходника)


EMOTION_LOOK = {
    # важный: грудь колесом, чуть откинулся назад
    "important": Look(angle=-3, scale=1.04, sy=1.02, sway=0.8),
    # удивлённый: подпрыгнул, вытянулся
    "surprised": Look(angle=0, scale=1.10, sx=0.96, sy=1.06, dy=-0.03, sway=0.5),
    # смеющийся: качается быстро и сильно
    "laughing": Look(angle=2, scale=1.03, sway=1.6, bob=2.2, speed=2.2),
    # злой: подался к зрителю, «насупился»
    "angry": Look(angle=6, scale=1.08, sx=1.04, sy=0.97, dx=-0.02, sway=0.4, shake=1.2),
    # задумчивый: склонил голову, медленно покачивается
    "thoughtful": Look(angle=8, scale=0.98, sway=1.3, speed=0.6),
    # испуганный: съёжился, отпрянул и дрожит
    "scared": Look(angle=-7, scale=0.90, sx=0.96, sy=0.98, dx=0.03, sway=0.3, shake=2.5),
}
POSE_LOOK = {
    "none": Look(),
    "point": Look(angle=7, dx=-0.03),                       # тянется к экрану
    "sword": Look(angle=-2, scale=1.05, sy=1.03, dy=-0.01),  # гордая стойка
    "shield": Look(angle=-6, scale=0.93, dx=0.02),           # прячется
}
TINT = {
    "angry": (1.06, 0.95, 0.94),
    "scared": (0.94, 0.98, 1.08),
    "laughing": (1.04, 1.0, 0.98),
}


# ---------------------------------------------------------- клюв и глаза ---

def _face_color(arr: np.ndarray, cx, cy, r_in, r_out):
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (d >= r_in) & (d <= r_out) & (arr[..., 3] > 200)
    px = arr[ring][:, :3].astype(np.int32)
    if not len(px):
        return (230, 230, 230)
    lum = px.sum(1)
    bright = px[lum > np.percentile(lum, 50)]
    return tuple(int(v) for v in np.median(bright, axis=0))


def _soft_mask(size, shape_fn, blur):
    m = Image.new("L", size, 0)
    shape_fn(ImageDraw.Draw(m))
    return m.filter(ImageFilter.GaussianBlur(blur)) if blur else m


def apply_mouth(img: Image.Image, level: int) -> Image.Image:
    """Открывает клюв: нижняя часть клюва опускается, за ней — рот."""
    if level <= 0:
        return img
    x0, y0, x1, y1 = LM["beak"]
    split = LM["beak_split"]
    bw, bh = x1 - x0, y1 - y0
    d = round(bh * [0, 0.11, 0.22, 0.33][min(level, 3)])
    arr = np.array(img)
    r, g, b = (arr[..., i].astype(np.int32) for i in range(3))
    orange = (r > 120) & (r - b > 50) & (g > b + 15) & (arr[..., 3] > 150)
    m = np.zeros(orange.shape, bool)
    m[split - 1:y1 + 3, x0 - 3:x1 + 3] = orange[split - 1:y1 + 3, x0 - 3:x1 + 3]
    lower = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)) \
        .filter(ImageFilter.GaussianBlur(1))
    patch = img.copy()
    patch.putalpha(Image.fromarray(np.minimum(np.array(lower), arr[..., 3])))

    out = img.copy()
    # 1) закрашиваем старое место нижней половины клюва цветом мордочки
    face = _face_color(arr, (x0 + x1) / 2, y1 + bh * 0.3, bh * 0.35, bh * 0.75)
    fill = Image.new("RGBA", img.size, face + (255,))
    out = Image.composite(fill, out, lower)
    # 2) рот: тёмный овал между половинками клюва
    mx0, mx1 = x0 + bw * 0.12, x1 - bw * 0.12
    my0 = split - bh * 0.08
    my1 = split + d + (y1 - split) * 0.7
    mouth = Image.new("RGBA", img.size, (0, 0, 0, 0))
    md = ImageDraw.Draw(mouth)
    md.ellipse((mx0, my0, mx1, my1), fill=(70, 18, 26, 255))
    if level >= 2:
        tw = (mx1 - mx0) * 0.5
        cx = (mx0 + mx1) / 2
        md.ellipse((cx - tw / 2, my1 - (my1 - my0) * 0.45, cx + tw / 2, my1 + 2), fill=(214, 92, 104, 255))
    out.alpha_composite(mouth.filter(ImageFilter.GaussianBlur(0.8)))
    # 3) нижняя половина клюва на новом месте (слегка вытянута — «деформация»)
    lower_box = (x0 - 4, split - 2, x1 + 4, y1 + 4)
    piece = patch.crop(lower_box)
    stretch = 1 + 0.05 * level
    piece = piece.resize((piece.width, int(piece.height * stretch)), Image.BICUBIC)
    out.alpha_composite(piece, (lower_box[0], lower_box[1] + d))
    # 4) верхняя половина клюва поверх (чтобы её край был чистым)
    upper = np.zeros(orange.shape, bool)
    upper[y0 - 3:split + 1, x0 - 3:x1 + 3] = orange[y0 - 3:split + 1, x0 - 3:x1 + 3]
    um = Image.fromarray((upper * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    up = img.copy()
    up.putalpha(Image.fromarray(np.minimum(np.array(um), arr[..., 3])))
    out.alpha_composite(up)
    return out


def apply_blink(img: Image.Image) -> Image.Image:
    """Закрывает глаза: область глаза заливается цветом мордочки (плавный
    переход от пикселей над глазом к пикселям под ним) и рисуется дуга века."""
    arr = np.array(img).astype(np.float32)
    out = arr.copy()
    for e in LM["eyes"]:
        cx, cy = e["x"], e["y"]
        rx, ry = e["w"] * 0.62 + 3, e["h"] * 0.66 + 3
        x0, x1 = int(cx - rx - 3), int(cx + rx + 4)
        ya, yb = int(cy - ry - 4), int(cy + ry + 4)
        top = arr[ya - 3:ya, x0:x1].mean(0)        # строка над глазом
        bot = arr[yb:yb + 3, x0:x1].mean(0)        # строка под глазом
        t = np.linspace(0, 1, yb - ya)[:, None, None]
        fill = top[None] * (1 - t) + bot[None] * t
        mask = np.array(_soft_mask(img.size, lambda dr: dr.ellipse(
            (cx - rx, cy - ry, cx + rx, cy + ry), fill=255), 3.0), np.float32)[ya:yb, x0:x1, None] / 255
        out[ya:yb, x0:x1] = out[ya:yb, x0:x1] * (1 - mask) + fill * mask
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(res)
    for e in LM["eyes"]:
        cx, cy = e["x"], e["y"]
        rx, ry = e["w"] * 0.5, e["h"] * 0.4
        lw = max(3, int(e["h"] * 0.11))
        dr.arc((cx - rx, cy - ry, cx + rx, cy + ry), 20, 160, fill=(28, 28, 34, 255), width=lw)
    return res


# ------------------------------------------------------------- значки ---

def _accent(emotion: str) -> Image.Image | None:
    """Небольшие значки эмоций в координатах холста."""
    if emotion not in ("scared", "angry", "thoughtful"):
        return None
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    hx, hy = OFF_X + BASE_W * 0.86, OFF_Y + BASE_H * 0.22   # справа у шлема
    s = BASE_W / 568
    if emotion == "scared":
        for (x, y, r) in ((hx, hy + 30 * s, 16 * s), (hx + 26 * s, hy + 80 * s, 10 * s)):
            d.polygon([(x, y - r * 2.3), (x + r, y), (x - r, y)], fill=(150, 205, 255, 235))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(150, 205, 255, 235))
            d.ellipse((x - r * 0.45, y - r * 0.5, x - r * 0.05, y - r * 0.05), fill=(255, 255, 255, 220))
    elif emotion == "angry":
        cx, cy = hx + 10 * s, hy - 20 * s
        for a in (45, 135, 225, 315):
            rad = math.radians(a)
            d.line([(cx + 9 * s * math.cos(rad), cy + 9 * s * math.sin(rad)),
                    (cx + 30 * s * math.cos(rad + 0.4), cy + 30 * s * math.sin(rad + 0.4))],
                   fill=(225, 40, 50, 255), width=max(3, int(8 * s)))
    else:
        for (x, y, r) in ((hx, hy - 10 * s, 8 * s), (hx + 24 * s, hy - 50 * s, 12 * s),
                          (hx + 52 * s, hy - 100 * s, 17 * s)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 235),
                      outline=(60, 70, 90, 200), width=max(2, int(3 * s)))
    return lay


# ---------------------------------------------------------------- сборка ---

def _tinted(img: Image.Image, emotion: str) -> Image.Image:
    t = TINT.get(emotion)
    if not t:
        return img
    arr = np.array(img).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * np.array(t, np.float32), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


@lru_cache(maxsize=64)
def variant(emotion: str, mouth: int, blink: bool, height: int) -> Image.Image:
    """Неподвижный кадр на холсте W×H, масштабированный к нужной высоте."""
    img = base_image()
    if blink:
        img = apply_blink(img)
    img = apply_mouth(img, mouth)
    img = _tinted(img, emotion)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.alpha_composite(img, (OFF_X, OFF_Y))
    acc = _accent(emotion)
    if acc is not None:
        canvas.alpha_composite(acc)
    if height != H:
        canvas = canvas.resize((round(W * height / H), height), Image.LANCZOS)
    return canvas


def transform(img: Image.Image, angle, scale, sx, sy, dx, dy) -> Image.Image:
    """Наклон/масштаб вокруг нижней середины (точки «пояса»)."""
    w, h = img.size
    px, py = w / 2, h * 0.97
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    kx, ky = scale * sx, scale * sy
    ox, oy = dx * w, dy * h
    # обратное преобразование: выход -> исходник
    a = c / kx
    b = -s / kx
    d_ = s / ky
    e = c / ky
    cx_ = px + ox
    cy_ = py + oy
    cc = px - a * cx_ - b * cy_
    f = py - d_ * cx_ - e * cy_
    return img.transform((w, h), Image.AFFINE, (a, b, cc, d_, e, f), resample=Image.BICUBIC)


class Animator:
    """Плавно анимирует сэра Лата кадр за кадром (пружина между эмоциями)."""

    FIELDS = ("angle", "scale", "sx", "sy", "dx", "dy")

    def __init__(self, fps: int, seed: int = 0):
        self.fps = fps
        self.rnd = random.Random(seed)
        self.cur = None
        self.vel = {k: 0.0 for k in self.FIELDS}

    def _target(self, emotion, pose):
        e = EMOTION_LOOK.get(emotion, EMOTION_LOOK["important"])
        p = POSE_LOOK.get(pose, POSE_LOOK["none"])
        return {
            "angle": e.angle + p.angle, "scale": e.scale * p.scale, "sx": e.sx * p.sx,
            "sy": e.sy * p.sy, "dx": e.dx + p.dx, "dy": e.dy + p.dy,
        }, e

    def step(self, emotion, pose):
        tgt, look = self._target(emotion, pose)
        if self.cur is None:
            self.cur = dict(tgt)
        for k in self.FIELDS:
            # пружина с лёгким «перелётом» — эмоции сменяются живо
            self.vel[k] += (tgt[k] - self.cur[k]) * 0.22
            self.vel[k] *= 0.62
            self.cur[k] += self.vel[k]
        return look

    def frame(self, t: float, emotion: str, pose: str, mouth: int, blink: bool, height: int) -> Image.Image:
        look = self.step(emotion, pose)
        c = self.cur
        sp = look.speed
        angle = c["angle"] + math.sin(2 * math.pi * t * sp / 3.4) * 1.6 * look.sway
        breath = math.sin(2 * math.pi * t * sp / 1.8)
        sy = c["sy"] * (1 + 0.012 * breath * look.bob)
        sx = c["sx"] * (1 - 0.006 * breath * look.bob)
        dy = c["dy"] - 0.006 * max(0.0, breath) * look.bob
        dx = c["dx"]
        if look.shake:
            k = look.shake / BASE_W
            dx += self.rnd.uniform(-k, k)
            dy += self.rnd.uniform(-k, k) * 0.6
        if emotion == "laughing":
            angle += math.sin(2 * math.pi * t * 2.6) * 2.2
        img = variant(emotion, mouth, blink, height)
        return transform(img, angle, c["scale"], sx, sy, dx, dy)

    def still(self, emotion: str, pose: str = "none", mouth: int = 0, blink: bool = False,
              height: int | None = None) -> Image.Image:
        """Статичный кадр в «устоявшемся» положении (для PNG)."""
        tgt, _ = self._target(emotion, pose)
        img = variant(emotion, mouth, blink, height or H)
        return transform(img, tgt["angle"], tgt["scale"], tgt["sx"], tgt["sy"], tgt["dx"], tgt["dy"])


# ---------------------------------------------------------- экспорт ассетов ---

def export_assets(out_dir=ASSETS):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "important": "важный", "surprised": "удивлённый", "laughing": "смеющийся",
        "angry": "злой", "thoughtful": "задумчивый", "scared": "испуганный",
    }
    an = Animator(25)
    files = []
    for e in EMOTION_KEYS:
        f = out_dir / f"emotion_{e}.png"
        an.still(e, mouth=2 if e in ("laughing", "surprised") else 0).save(f)
        files.append((f, names[e]))
    pose_emotion = {"point": "important", "sword": "angry", "shield": "scared"}
    pose_names = {"point": "указывает", "sword": "с мечом", "shield": "за щитом"}
    for pz, e in pose_emotion.items():
        f = out_dir / f"pose_{pz}.png"
        an.still(e, pz).save(f)
        files.append((f, pose_names[pz]))
    anim = out_dir / "anim"
    anim.mkdir(exist_ok=True)
    for m in range(MOUTH_LEVELS):
        variant("important", m, False, H).save(anim / f"mouth_{m}.png")
    variant("important", 0, True, H).save(anim / "blink.png")

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
    print(f"Сохранено {len(files)} PNG + кадры клюва и моргания в {out_dir}")


if __name__ == "__main__":
    export_assets()
