"""Подготовка картинки сэра Лата: удаление фона (rembg), обрезка по пояс,
поиск клюва и глаз для анимации.

    python -m sirlat.prepare_character                  # исходник из assets/sir_lat/source/
    python -m sirlat.prepare_character путь/к/картинке.png --waist 0.74

Результат:
    assets/sir_lat/sir_lat.png       — персонаж без фона, по пояс
    assets/sir_lat/landmarks.json    — где клюв и глаза (для липсинка и моргания)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import ASSETS

SOURCE_DIR = ASSETS / "source"


def find_source() -> Path:
    files = sorted(p for p in SOURCE_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not files:
        raise SystemExit(f"Положите картинку персонажа в {SOURCE_DIR}")
    return files[0]


def remove_background(img: Image.Image, model: str) -> Image.Image:
    from rembg import new_session, remove
    out = remove(img.convert("RGB"), session=new_session(model))
    # убираем полупрозрачный «мусор» по краям (тени, ореол фона)
    a = np.array(out.split()[3]).astype(np.float32)
    a[a < 24] = 0
    a = np.clip((a - 24) * 255 / (255 - 24), 0, 255)
    out.putalpha(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)))
    return out


def _blob_center(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    return float(xs.mean()), float(ys.mean()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def detect_landmarks(img: Image.Image) -> dict:
    """Ищет клюв (оранжевая область) и глаза (два тёмных пятна над клювом)."""
    arr = np.array(img).astype(np.int32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    h, w = a.shape
    orange = (r > 140) & (r - b > 70) & (g > b + 25) & (g < r - 15) & (a > 200)
    # клюв — самое крупное оранжевое пятно в верхней половине (лапы обрезаны)
    orange[int(h * 0.75):] = False
    ys, xs = np.nonzero(orange)
    if not len(xs):
        raise SystemExit("Не нашёл клюв (оранжевую область). Укажите координаты в landmarks.json вручную.")
    cy0, cx0 = np.median(ys), np.median(xs)
    near = (np.abs(ys - cy0) < h * 0.08) & (np.abs(xs - cx0) < w * 0.12)
    ys, xs = ys[near], xs[near]
    beak = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    bw, bh = beak[2] - beak[0], beak[3] - beak[1]

    # глаза: тёмные пятна в полосе над клювом, окружённые светлой «мордочкой»
    lum = (r * 299 + g * 587 + b * 114) // 1000
    dark = (lum < 45) & (a > 200)
    bright = (lum > 160) & (a > 200)
    k = max(4, int(bw * 0.3))
    lit_l = np.zeros_like(bright)
    lit_r = np.zeros_like(bright)
    for d in range(2, k):
        lit_l[:, d:] |= bright[:, :-d]
        lit_r[:, :-d] |= bright[:, d:]
    lit_u = np.zeros_like(bright)
    lit_d = np.zeros_like(bright)
    for d in range(2, k * 2):
        lit_u[d:] |= bright[:-d]
        lit_d[:-d] |= bright[d:]
    eye_px = dark & lit_l & lit_r & lit_u & lit_d
    top, bot = max(0, beak[1] - int(bh * 1.6)), beak[1] + int(bh * 0.1)
    cx = (beak[0] + beak[2]) / 2
    eyes = []
    for lo, hi in ((cx - bw * 2.4, cx - bw * 0.05), (cx + bw * 0.05, cx + bw * 2.4)):
        x0, x1 = max(0, int(lo)), min(w, int(hi))
        c = _blob_center(eye_px[top:bot, x0:x1])
        if c:
            eyes.append({"x": round(c[0] + x0, 1), "y": round(c[1] + top, 1),
                         "w": round(c[2], 1), "h": round(c[3], 1)})
    return {"beak": beak, "beak_split": round(beak[1] + bh * 0.55), "eyes": eyes}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="исходная картинка (по умолчанию из assets/sir_lat/source/)")
    ap.add_argument("--waist", type=float, default=0.84,
                    help="где резать «по пояс»: доля высоты фигуры от макушки (0.6–0.95)")
    ap.add_argument("--model", default="isnet-general-use", help="модель rembg")
    args = ap.parse_args(argv)

    src = Path(args.image) if args.image else find_source()
    print(f"Исходник: {src}")
    cut = remove_background(Image.open(src), args.model)
    bbox = cut.getbbox()
    cut = cut.crop(bbox)
    # по пояс
    cut = cut.crop((0, 0, cut.width, int(cut.height * args.waist)))
    cut = cut.crop(cut.getbbox())
    lm = detect_landmarks(cut)
    lm["size"] = [cut.width, cut.height]
    ASSETS.mkdir(parents=True, exist_ok=True)
    cut.save(ASSETS / "sir_lat.png")
    (ASSETS / "landmarks.json").write_text(json.dumps(lm, indent=2), encoding="utf-8")
    print(f"Сохранено: {ASSETS / 'sir_lat.png'} {cut.size}; клюв {lm['beak']}, глаз найдено: {len(lm['eyes'])}")


if __name__ == "__main__":
    main()
