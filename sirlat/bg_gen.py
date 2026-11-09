"""Автоматическая генерация фонов для сцен сценария.

Для сцен, к которым не подошла ни одна своя картинка из backgrounds/:
  1. Gemini описывает место на английском (без людей и персонажей) и даёт
     2–4 английских слова-тега (без ключа — готовые описания по типу сцены);
  2. картинка генерируется через Pollinations.ai (бесплатно, без ключа),
     а если он не ответил — через Hugging Face FLUX.1-schnell (секрет HF_TOKEN);
  3. фон сохраняется в backgrounds/generated/ с именем из тегов
     (например dungeon_torch_corridor_3fa2c1_9x16.jpg) и описанием в index.json;
  4. для похожей сцены (тот же тип и общие теги, та же ориентация) берётся
     уже готовый фон — повторно не генерируем.

Свои картинки из backgrounds/ всегда в приоритете (см. backgrounds.Picker).
У бесплатного Pollinations лимит: 1 картинка в ~45 с с одного IP, до 1024 пикс.
по длинной стороне; поэтому новых фонов за запуск не больше BG_GEN_MAX.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from urllib.parse import quote

import requests
from PIL import Image

from .backgrounds import BG_DIR, Picker
from .scenes import classify

GEN_DIR = BG_DIR / "generated"
INDEX = GEN_DIR / "index.json"
STYLE = ("Empty location, NO people, NO characters. Low-poly 3D animation style, matte materials, "
         "visible flat facets, soft cinematic lighting. No text, logos or watermarks.")
SIZES = {True: (576, 1024), False: (1024, 576)}      # вертикальный 9:16 / горизонтальный 16:9
POLL_GAP = 46.0                                       # пауза между запросами к бесплатному Pollinations

FALLBACK = {
    "dungeon": ("a dark stone dungeon corridor with iron doors, hanging chains and burning wall torches",
                ["dungeon", "torch", "corridor"]),
    "cave": ("a vast underground cave with glowing blue crystals, stalactites and a misty pool",
             ["cave", "crystal"]),
    "tavern": ("a cozy medieval fantasy tavern interior with wooden tables, barrels and a warm fireplace",
               ["tavern", "fireplace"]),
    "city": ("a medieval fantasy town street at dusk with timber houses, lanterns and a distant castle",
             ["city", "street", "castle"]),
    "forest": ("a mysterious fantasy forest clearing at night with a small campfire and tall pine trees",
               ["forest", "campfire", "night"]),
    "game": ("an abstract glowing blue holographic game interface panel floating in a dark digital void",
             ["system", "window"]),
    "hero": ("a dramatic cliff above a misty valley at sunset with an old sword stuck in the ground",
             ["hero", "sword", "sunset"]),
    "monster": ("a ruined dark arena with deep claw marks, red glowing mist and a huge ominous shadow on the wall",
                ["monster", "shadow"]),
    None: ("a fantasy landscape with ancient ruins under a dramatic sky", ["ruins", "landscape"]),
}
KNOWN_TAGS = ("forest, campfire, night, dungeon, torch, corridor, cave, crystal, mine, tavern, inn, "
              "fireplace, city, town, street, castle, village, market, shop, system, window, level, quest, "
              "hero, sword, battle, monster, shadow, boss, goblin, dragon, ruins, mountain, river, sea, "
              "prison, throne, hall, library, temple, academy, snow, desert, road, camp, moon, sunset")

_last_poll = 0.0


def _log(msg):
    print(f"Фоны-генерация: {msg}")


# --------------------------------------------------------------- индекс ---

def _load_index() -> list[dict]:
    if INDEX.exists():
        try:
            data = json.loads(INDEX.read_text(encoding="utf-8"))
            return [d for d in data if (GEN_DIR / d["file"]).exists()]
        except Exception:  # noqa: BLE001
            pass
    return []


def _save_index(items: list[dict]):
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def _similar(a: dict, scene: str | None, tags: list[str], vertical: bool) -> bool:
    if a.get("vertical") != vertical:
        return False
    ta, tb = set(a.get("tags", [])), set(tags)
    if scene and a.get("scene") == scene and len(ta & tb) >= 1:
        return True
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


# ---------------------------------------------------------------- сцены ---

def scene_runs(phrases, min_words: int = 12) -> list[dict]:
    """Подряд идущие фразы с одним типом сцены — одна «сцена» для фона."""
    runs, cur = [], None
    for ph in phrases:
        sc = (ph.scene if getattr(ph, "scene_set", False) else None) or classify(ph.text) or \
            getattr(ph, "scene", None) or (cur["scene"] if cur else None)
        if cur and cur["scene"] == sc:
            cur["text"] += " " + ph.text
        else:
            cur = {"scene": sc, "text": ph.text}
            runs.append(cur)
    # слишком короткие сцены приклеиваем к предыдущей
    out = []
    for r in runs:
        if out and len(r["text"].split()) < min_words and r["scene"] in (None, out[-1]["scene"]):
            out[-1]["text"] += " " + r["text"]
        else:
            out.append(r)
    return out


def describe(runs: list[dict]) -> list[tuple[str, list[str]]]:
    """Английское описание места и теги для каждой сцены (Gemini или готовые)."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key and runs:
        from .script import _gemini
        numbered = "\n".join(f"{i + 1}. [{r['scene'] or 'unknown'}] {r['text'][:500]}" for i, r in enumerate(runs))
        prompt = f"""Ниже — сцены из видеопересказа манхвы (по-русски). Для каждой сцены придумай фон:
опиши МЕСТО действия на английском, 12–25 слов, конкретно и наглядно (что видно, свет, время суток).
Строго без людей, персонажей, лиц, фигур и текста — только пустое место. Если в сцене монстр или
тень, передай это через обстановку (следы когтей, зловещая тень на стене, туман).
Теги — 2–4 английских существительных о месте, по возможности из списка: {KNOWN_TAGS}.
Сцены:
{numbered}
Верни JSON-массив из {len(runs)} объектов {{"desc": "...", "tags": ["...", "..."]}} в том же порядке."""
        try:
            raw = _gemini(prompt, key, json_mode=True, temperature=0.6)
            data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
            if len(data) != len(runs):
                raise ValueError(f"ожидалось {len(runs)} описаний, пришло {len(data)}")
            out = []
            for d, r in zip(data, runs):
                desc = re.sub(r"\s+", " ", str(d.get("desc", ""))).strip()[:300]
                tags = [re.sub(r"[^a-z]", "", str(t).lower()) for t in d.get("tags", [])]
                tags = [t for t in tags if len(t) >= 2][:4]
                if not desc:
                    desc, tags = FALLBACK.get(r["scene"], FALLBACK[None])
                out.append((desc, tags or list(FALLBACK.get(r["scene"], FALLBACK[None])[1])))
            _log(f"описания сцен от Gemini: {len(out)}")
            return out
        except Exception as e:  # noqa: BLE001
            _log(f"Gemini не описал сцены ({e}) — беру готовые описания")
    return [FALLBACK.get(r["scene"], FALLBACK[None]) for r in runs]


# ------------------------------------------------------------- генерация ---

def _check_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    if min(im.size) < 256:
        raise RuntimeError(f"слишком маленькая картинка {im.size}")
    return im.convert("RGB")


def pollinations(prompt: str, w: int, h: int, seed: int) -> Image.Image:
    global _last_poll
    token = os.getenv("POLLINATIONS_TOKEN", "").strip()
    url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
    params = {"width": w, "height": h, "seed": seed, "nologo": "true",
              "model": os.getenv("POLLINATIONS_MODEL", "flux")}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = None
    for attempt in range(4):
        wait = POLL_GAP - (time.time() - _last_poll)
        if not token and wait > 0:
            time.sleep(wait)
        _last_poll = time.time()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=240)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                im = _check_image(r.content)
                # бесплатный Pollinations ставит знак «pollinations.ai» в правом нижнем углу — срезаем
                return im.crop((0, 0, im.width, int(im.height * 0.93)))
            m = re.search(r"[Rr]etry after (\d+)", r.text)
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
            if r.status_code in (402, 429) and m:
                time.sleep(min(90, int(m.group(1)) + 2))
            elif r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Pollinations не ответил: {last}")


def hf_flux(prompt: str, w: int, h: int, seed: int) -> Image.Image:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("нет секрета HF_TOKEN")
    body = {"inputs": prompt, "parameters": {"width": w, "height": h, "num_inference_steps": 4, "seed": seed}}
    last = None
    for url in ("https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
                "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"):
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=240,
                                  headers={"Authorization": f"Bearer {token}", "Accept": "image/png"})
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    return _check_image(r.content)
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                if r.status_code == 503:          # модель загружается
                    try:
                        time.sleep(min(60, float(r.json().get("estimated_time", 20))))
                    except Exception:  # noqa: BLE001
                        time.sleep(20)
                elif r.status_code in (429, 500, 502, 504):
                    time.sleep(10 * (attempt + 1))
                else:
                    break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(5)
    raise RuntimeError(f"FLUX.1-schnell не ответил: {last}")


def generate(desc: str, vertical: bool, seed: int) -> tuple[Image.Image, str]:
    # стиль — в начале промпта: так модель держит его лучше
    prompt = f"{STYLE} Scene: {desc.rstrip('.')}."
    w, h = SIZES[vertical]
    errors = []
    for name, fn in (("pollinations", pollinations), ("hf-flux-schnell", hf_flux)):
        try:
            return fn(prompt, w, h, seed), name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            _log(f"{name} не сработал ({str(e)[:160]})")
    raise RuntimeError("; ".join(errors))


# ------------------------------------------------------------------ итог ---

def ensure_backgrounds(phrases, vertical: bool, max_new: int | None = None) -> None:
    """Догенерирует фоны для сцен, к которым не подходит ни одна своя картинка."""
    if os.getenv("SIRLAT_BG_GEN", "1") == "0":
        return
    if max_new is None:
        max_new = int(os.getenv("BG_GEN_MAX", "6" if vertical else "12"))
    runs = scene_runs(phrases)
    user = Picker(vertical=vertical, include_generated=False, quiet=True)
    need = [r for r in runs if not (user and user.best_score(r["text"], r["scene"]) > 0)]
    if not need:
        _log("все сцены покрыты вашими картинками из backgrounds/")
        return
    index = _load_index()
    specs = describe(need)
    made = reused = 0
    orient = "9x16" if vertical else "16x9"
    for r, (desc, tags) in zip(need, specs):
        old = next((a for a in index if _similar(a, r["scene"], tags, vertical)), None)
        if old:
            reused += 1
            continue
        if made >= max_new:
            continue
        seed = int(hashlib.sha1(desc.encode()).hexdigest()[:6], 16)
        try:
            img, engine = generate(desc, vertical, seed)
        except Exception as e:  # noqa: BLE001
            _log(f"не получилось сгенерировать «{desc[:60]}…»: {e}")
            continue
        words = list(dict.fromkeys(([r["scene"]] if r["scene"] else []) + tags))[:5]
        name = f"{'_'.join(words) or 'scene'}_{seed:06x}_{orient}.jpg"
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        img.save(GEN_DIR / name, quality=92)
        index.append({"file": name, "scene": r["scene"], "tags": tags, "desc": desc,
                      "vertical": vertical, "engine": engine, "size": list(img.size)})
        _save_index(index)
        made += 1
        _log(f"{engine}: {name}  ← {desc[:80]}")
    _log(f"сцен без своих картинок: {len(need)}; новых фонов: {made}, взято готовых: {reused}")
