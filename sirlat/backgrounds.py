"""Фоны из своих картинок: папка backgrounds/ (вертикальные 9:16 и горизонтальные 16:9).

Фон для каждой фразы выбирается по смыслу через имя файла: слова имени
(forest_campfire.png -> «forest», «campfire») сравниваются со словами фразы
(«лес», «костёр») и с типом сцены, который подсказывает фраза (лес, подземелье…).
Можно называть файлы и по-русски: «лес_костёр.png».
Если подходящей картинки нет — остаётся прежний фон, а в начале берётся любая.

Движение: медленный зум и панорама + параллакс: низ картинки (ближний план)
смещается сильнее верха (дальний план).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import ROOT

BG_DIR = ROOT / "backgrounds"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OV = 1.18   # запас под зум, панораму и параллакс

# английские (и русские) слова в именах файлов -> корни русских слов во фразах
SYN = {
    "forest": "лес|дерев|чащ|рощ", "wood": "лес|дерев", "woods": "лес", "tree": "дерев",
    "campfire": "костер|костр|огон|привал|лагер", "fire": "огон|пламен|костер|пожар", "bonfire": "костер",
    "camp": "лагер|привал|палатк", "dungeon": "подземел|данж|темниц|склеп|лабиринт",
    "cave": "пещер|грот", "mine": "шахт|рудник|шахтер", "miner": "шахтер|шахт",
    "crystal": "кристалл", "tavern": "таверн|трактир|кабак|харчевн", "inn": "таверн|трактир|постоял",
    "bar": "таверн|бар", "city": "город|улиц|площад|столиц", "town": "город|деревн",
    "street": "улиц", "castle": "замок|замк|крепост|дворц|дворец", "palace": "дворец|дворц",
    "village": "деревн|сел", "market": "рынок|рынк|торгов|лавк", "shop": "лавк|магазин|торгов",
    "game": "систем|окн|уровн|навык|квест|задани|статус|опыт", "system": "систем|окн|интерфейс",
    "window": "окн|систем", "ui": "систем|окн|интерфейс", "status": "статус|характеристик",
    "level": "уровн|левел", "quest": "квест|задани", "hero": "геро", "knight": "рыцар|сэр",
    "sword": "меч|клинок", "battle": "битв|сраж|бой|бою|бит", "fight": "бой|драк|сраж",
    "monster": "монстр|чудовищ|твар|чудищ", "shadow": "тен[ьи]|мрак", "boss": "босс",
    "goblin": "гоблин", "dragon": "дракон", "demon": "демон", "orc": "орк", "skeleton": "скелет",
    "ghost": "призрак|дух", "night": "ноч|ночь", "moon": "лун", "sunset": "закат", "dawn": "рассвет",
    "mountain": "гор[аыу]|горн|скал", "sea": "мор[ея]|океан", "river": "рек|ручей",
    "prison": "тюрьм|темниц|застенк|камер", "throne": "трон", "room": "комнат", "hall": "зал",
    "library": "библиотек|книг", "temple": "храм|алтар", "church": "храм|церк",
    "school": "академи|школ|учени", "academy": "академи", "snow": "снег|зим", "rain": "дожд",
    "desert": "пустын", "ship": "корабл|судн", "road": "дорог|путь|тропа",
}
# тип сцены -> слова имени файла, которые ему соответствуют
SCENE_TOKENS = {
    "forest": {"forest", "wood", "woods", "tree", "лес"},
    "dungeon": {"dungeon", "prison", "подземелье", "темница"},
    "cave": {"cave", "mine", "crystal", "пещера", "шахта"},
    "tavern": {"tavern", "inn", "bar", "таверна"},
    "city": {"city", "town", "street", "castle", "village", "market", "город"},
    "game": {"game", "system", "window", "ui", "status", "level", "quest", "окно", "система"},
    "hero": {"hero", "knight", "sword", "battle", "fight", "герой"},
    "monster": {"monster", "shadow", "boss", "goblin", "dragon", "demon", "монстр", "тень"},
}


def _tokens(name: str) -> list[str]:
    base = name.rsplit(".", 1)[0].lower().replace("ё", "е")
    return [t for t in re.split(r"[^a-zа-я]+", base) if len(t) >= 2]


@dataclass
class Background:
    path: Path
    tokens: list
    w: int
    h: int
    used: int = 0
    rx: list = field(default_factory=list)

    @property
    def vertical(self) -> bool:
        return self.h > self.w


def bg_dirs() -> list[Path]:
    dirs = [BG_DIR] if BG_DIR.is_dir() else []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() and p.name.lower() == "backgrounds" and p != BG_DIR and not (
                {".git", "output", ".cache"} & set(p.parts)):
            dirs.append(p)
    return dirs


def load_all() -> list[Background]:
    out = []
    for d in bg_dirs():
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or not p.is_file() or p.stat().st_size < 1000:
                continue
            try:
                with Image.open(p) as im:
                    w, h = ImageOps.exif_transpose(im).size
            except Exception as e:  # noqa: BLE001
                print(f"Фоны: не читается {p.relative_to(ROOT)} ({e})")
                continue
            toks = _tokens(p.name)
            rx = []
            for t in toks:
                if t in SYN:
                    rx.append(re.compile(SYN[t]))
                elif re.fullmatch(r"[а-я]+", t) and len(t) >= 3:
                    rx.append(re.compile(t[:max(3, len(t) - 2)]))   # русское слово: по корню
            out.append(Background(p, toks, w, h, rx=rx))
    return out


def _is_generated(b: Background) -> bool:
    return "generated" in b.path.relative_to(ROOT).parts


class Picker:
    """Выбирает картинку для фразы по смыслу; для формата — нужной ориентации.

    Два уровня: сначала ваши картинки из backgrounds/, и только если ни одна
    не подходит по смыслу — сгенерированные из backgrounds/generated/.
    """

    def __init__(self, vertical: bool, seed: int = 0, include_generated: bool = True, quiet: bool = False):
        allb = load_all()
        if not include_generated:
            allb = [b for b in allb if not _is_generated(b)]
        self.rnd = random.Random(seed)
        self.tiers = []
        for tier in ([b for b in allb if not _is_generated(b)], [b for b in allb if _is_generated(b)]):
            same = [b for b in tier if b.vertical == vertical]
            self.tiers.append(same or tier)
        self.items = self.tiers[0] + self.tiers[1]
        if self.items and not quiet:
            print(f"Фоны: ваших картинок {len(self.tiers[0])}, сгенерированных {len(self.tiers[1])}")

    def __bool__(self):
        return bool(self.items)

    def score(self, b: Background, text: str, scene: str | None) -> float:
        low = text.lower().replace("ё", "е")
        s = sum(1.0 for rx in b.rx if rx.search(low))
        if scene and SCENE_TOKENS.get(scene, set()) & set(b.tokens):
            s += 1.5
        return s

    def best_score(self, text: str, scene: str | None) -> float:
        return max((self.score(b, text, scene) for b in self.items), default=0.0)

    def choose(self, text: str, scene: str | None, current: Path | None) -> Path | None:
        """Лучшая по смыслу картинка или None (тогда фон не меняется)."""
        for tier in self.tiers:                       # свои картинки — в приоритете
            if not tier:
                continue
            scored = [(self.score(b, text, scene), b) for b in tier]
            best = max(s for s, _ in scored)
            if best <= 0:
                continue
            top = [b for s, b in scored if s == best]
            if current is not None and any(b.path == current for b in top):
                return current        # подходящая уже на экране — не дёргаем
            b = min(top, key=lambda b: (b.used, self.rnd.random()))
            b.used += 1
            return b.path
        if current is None:           # самое начало — любая (сначала своя)
            pool = self.tiers[0] or self.tiers[1]
            b = min(pool, key=lambda b: (b.used, self.rnd.random()))
            b.used += 1
            return b.path
        return None


@lru_cache(maxsize=6)
def _prepared(path: str, W: int, H: int) -> Image.Image:
    """Картинка, обрезанная «по заполнению» под кадр с запасом OV."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    tw, th = int(W * OV), int(H * OV)
    out = ImageOps.fit(im, (tw, th), Image.LANCZOS, centering=(0.5, 0.45))
    k = tw / max(1, min(im.width, im.height * tw / th))
    if k > 1.4:   # маленькую картинку (генерация до 1024 пикс.) после увеличения чуть подтачиваем
        from PIL import ImageFilter
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    return out


class ImageScene:
    """План с картинкой: медленный зум, панорама и параллакс по высоте кадра."""

    ZSTEP = 0.0025

    def __init__(self, path: Path, W: int, H: int, fps: int, n_frames: int, shot_seed: int):
        self.W, self.H, self.fps, self.n = W, H, fps, max(1, n_frames)
        self.back = _prepared(str(path), W, H)
        self.bw, self.bh = self.back.size
        rnd = random.Random(shot_seed)
        zr = rnd.uniform(0.035, 0.06)
        self.z0, self.z1 = (1.0, 1.0 + zr) if rnd.random() < 0.6 else (1.0 + zr, 1.0)
        mx = (self.bw - W) / 2
        self.par = mx * 0.35                                 # сила параллакса (пикс.)
        side = rnd.choice((-1, 1))
        self.p0 = (-side * mx * 0.3, rnd.uniform(-1, 1) * (self.bh - H) / 2 * 0.3)
        self.p1 = (side * mx * 0.3, rnd.uniform(-1, 1) * (self.bh - H) / 2 * 0.3)
        self.levels = {}
        self.band = 6
        ys = np.arange(0, H, self.band)
        self.row_k = (ys + self.band / 2) / H - 0.5           # −0.5 вверху … +0.5 внизу

    def _level(self, z):
        i = int(round((z - 1.0) / self.ZSTEP))
        if i not in self.levels:
            zz = 1.0 + i * self.ZSTEP
            img = self.back if i == 0 else self.back.resize(
                (int(round(self.bw * zz)), int(round(self.bh * zz))), Image.BILINEAR)
            self.levels[i] = (np.asarray(img), zz)
        return self.levels[i]

    def frame(self, i: int) -> np.ndarray:
        W, H = self.W, self.H
        u = min(1.0, i / self.n)
        u = u * u * (3 - 2 * u)
        z = self.z0 + (self.z1 - self.z0) * u
        px = self.p0[0] + (self.p1[0] - self.p0[0]) * u
        py = self.p0[1] + (self.p1[1] - self.p0[1]) * u
        lvl, zz = self._level(z)
        lh, lw = lvl.shape[:2]
        ox = (self.bw / 2 + px) * zz - W / 2
        oy = int(round((self.bh / 2 + py) * zz - H / 2))
        oy = max(0, min(lh - H, oy))
        # параллакс: ближний план (низ) уезжает дальше, дальний (верх) — меньше
        shift = (u - 0.5) * 2 * self.par * zz * (1 if self.p1[0] > self.p0[0] else -1)
        out = np.empty((H, W, 3), np.uint8)
        for k, y in zip(self.row_k, range(0, H, self.band)):
            x = int(round(ox + shift * k * 1.2))
            x = max(0, min(lw - W, x))
            y1 = min(H, y + self.band)
            out[y:y1] = lvl[oy + y:oy + y1, x:x + W]
        return out

    def close(self):
        self.levels.clear()
