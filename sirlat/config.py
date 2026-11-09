"""Общие настройки проекта."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "sir_lat"
NOTES = ROOT / "notes"
OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache"

# Голос
VOICE = os.getenv("SIRLAT_VOICE", "ru-RU-DmitryNeural")
VOICE_RATE = os.getenv("SIRLAT_RATE", "-8%")    # чуть медленнее
VOICE_PITCH = os.getenv("SIRLAT_PITCH", "-6Hz")  # чуть ниже

# Gemini
# Модели перебираются по очереди, пока одна не ответит. GEMINI_MODEL (если задана)
# пробуется первой; GEMINI_MODELS — полный список через запятую.
GEMINI_FALLBACK = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash"]
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()] \
    or GEMINI_FALLBACK
_first = os.getenv("GEMINI_MODEL", "").strip()
if _first:
    GEMINI_MODELS = [_first] + [m for m in GEMINI_MODELS if m != _first]

FPS = int(os.getenv("SIRLAT_FPS", "25"))
SAMPLE_RATE = 24000

# Эмоции: русское название из разметки -> внутренний ключ
EMOTIONS = {
    "важный": "important",
    "удивлённый": "surprised",
    "смеющийся": "laughing",
    "злой": "angry",
    "задумчивый": "thoughtful",
    "испуганный": "scared",
}
# Позы: русское название -> ключ
POSES = {
    "указывает": "point",
    "меч": "sword",
    "щит": "shield",
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
SERIF_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
] + FONT_CANDIDATES
MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
] + FONT_CANDIDATES


@lru_cache(maxsize=None)
def font(size: int, kind: str = "sans") -> ImageFont.FreeTypeFont:
    cands = {"sans": FONT_CANDIDATES, "serif": SERIF_CANDIDATES, "mono": MONO_CANDIDATES}[kind]
    env = os.getenv("SIRLAT_FONT")
    if env:
        cands = [env] + cands
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError(
        "Не найден шрифт с кириллицей. Установите fonts-dejavu-core "
        "или укажите путь в переменной SIRLAT_FONT."
    )
