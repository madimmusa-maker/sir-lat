"""Сценарий: заметки -> Gemini -> размеченный текст от лица сэра Лата.

Формат сценария (его же можно править руками и подавать через --script):

    # Название выпуска
    ## Часть 1: Заголовок части
    {сцена: подземелье}          <- фон для следующих фраз (подземелье, таверна,
                                    город, герой, монстр, окно)
    {сток: dark dungeon torch}   <- запрос (англ.) для стоковых клипов Pixabay
    {окно: Получен навык: ...}   <- «игровое окно» поверх фона на следующей фразе
    [важный] Короткая фраза.
    [удивлённый, меч] Эмоция и (необязательно) поза через запятую.

Эмоции: важный, удивлённый, смеющийся, злой, задумчивый, испуганный.
Позы: указывает, меч, щит.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

from .config import EMOTIONS, GEMINI_MODELS, POSES
from .scenes import normalize_scene


@dataclass
class Phrase:
    text: str
    emotion: str = "important"
    pose: str = "none"
    scene: str | None = None
    stock: str | None = None
    window: str | None = None


@dataclass
class Part:
    title: str
    phrases: list[Phrase] = field(default_factory=list)


@dataclass
class Script:
    title: str
    parts: list[Part] = field(default_factory=list)

    def word_count(self) -> int:
        return sum(len(p.text.split()) for part in self.parts for p in part.phrases)


# ------------------------------------------------------------------ парсер ---

TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.+)$")
META_RE = re.compile(r"^\{\s*([^:}]+)\s*:\s*([^}]*)\}\s*$")


def _split_long(text: str, limit: int = 18) -> list[str]:
    if len(text.split()) <= limit:
        return [text]
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len((cur + " " + p).split()) > limit:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out


def parse_script(text: str) -> Script:
    title = "Хроники сэра Лата"
    script = Script(title)
    part: Part | None = None
    scene = stock = window = None
    for raw in text.splitlines():
        line = raw.strip().strip("*").strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("## "):
            part = Part(line[3:].strip())
            script.parts.append(part)
            continue
        if line.startswith("# "):
            script.title = line[2:].strip()
            continue
        m = META_RE.match(line)
        if m:
            key, val = m.group(1).strip().lower(), m.group(2).strip()
            if key in ("сцена", "фон", "scene"):
                scene = normalize_scene(val) or scene
            elif key in ("сток", "stock"):
                stock = val or None
            elif key in ("окно", "window"):
                window = val or None
            continue
        m = TAG_RE.match(line)
        if m:
            tags = [t.strip().lower().replace("ё", "е") for t in m.group(1).split(",")]
            body = m.group(2).strip()
        else:
            tags, body = [], line
        emotion, pose = "important", "none"
        for t in tags:
            for ru, key in EMOTIONS.items():
                if t == ru.replace("ё", "е") or t == key:
                    emotion = key
            for ru, key in POSES.items():
                if t == ru or t == key or t.startswith(ru[:4]):
                    pose = key
        if part is None:
            part = Part("Вступление")
            script.parts.append(part)
        for chunk in _split_long(body):
            part.phrases.append(Phrase(chunk, emotion, pose, scene, stock, window))
            window = None
    script.parts = [p for p in script.parts if p.phrases]
    return script


# ------------------------------------------------------------------ Gemini ---

PERSONA = """Ты — сэр Пингвин Лат, пингвин-рыцарь в железном шлеме с забралом и ведущий \
YouTube-канала «Хроники сэра Лата», где он пересказывает и обозревает манхву.
Характер: важный, но добродушный; самоироничный; обожает рыбу, турниры и кодекс чести; \
любит сравнивать происходящее со своими «рыцарскими подвигами»; иногда пугается, но \
храбрится. Говорит короткими живыми фразами, по-русски, с юмором.

СТРОГИЕ ПРАВИЛА:
- Пересказывай события СВОИМИ словами по заметкам автора. Ничего не цитируй из оригинала \
дословно: никаких реплик персонажей в кавычках, никаких фраз из перевода.
- Добавляй личные комментарии, шутки, оценки (например «это я одобряю», «семь рыбок из десяти»).
- Каждая фраза — отдельная строка, 4–16 слов, в начале строки метка эмоции в квадратных скобках.
- Эмоции (только эти): [важный], [удивлённый], [смеющийся], [злой], [задумчивый], [испуганный].
- Иногда добавляй позу через запятую: [злой, меч], [важный, указывает], [испуганный, щит].
- Метки фона ставь отдельной строкой: {сцена: X}, где X — подземелье, таверна, город, \
герой, монстр или окно. Строкой {сток: ...} задай 2–4 английских слова для поиска \
стокового видео (атмосфера, без названий манхвы и имён). Иногда добавляй \
{окно: короткий шуточный текст «системного окна»} — придумай его сам.
- Не используй эмодзи, markdown-выделения, ремарки в скобках и списки.
- Числа пиши словами, если это удобно для озвучки."""


class ModelUnavailable(Exception):
    """Модель не существует или недоступна для этого ключа (404 и т.п.)."""


_active_model: str | None = None   # модель, которая уже ответила в этом запуске


def _gemini_once(model: str, body: dict, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, json=body, timeout=180, headers={"x-goog-api-key": api_key})
    text = r.text[:300].replace("\n", " ")
    if r.status_code == 404 or (r.status_code == 400 and re.search(
            r"not found|not supported|is not available|unknown model", r.text, re.I)):
        raise ModelUnavailable(f"HTTP {r.status_code}: {text}")
    if r.status_code in (401, 403):
        raise PermissionError(f"Gemini отклонил ключ (HTTP {r.status_code}): {text}")
    if r.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError(f"HTTP {r.status_code}: {text}")
    r.raise_for_status()
    data = r.json()
    cands = data.get("candidates") or []
    if not cands or not cands[0].get("content", {}).get("parts"):
        reason = (cands[0].get("finishReason") if cands else None) or data.get("promptFeedback")
        raise RuntimeError(f"пустой ответ ({reason})")
    return "".join(p.get("text", "") for p in cands[0]["content"]["parts"]).strip()


def _gemini(prompt: str, api_key: str, json_mode: bool = False, temperature: float = 0.95) -> str:
    """Запрос к Gemini с перебором моделей: первая ответившая запоминается."""
    global _active_model
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": PERSONA}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    models = list(GEMINI_MODELS)
    if _active_model:
        models.remove(_active_model)
        models.insert(0, _active_model)
    errors = []
    for model in models:
        for attempt in range(3):
            try:
                text = _gemini_once(model, body, api_key)
                if model != _active_model:
                    print(f"Gemini: сработала модель {model}")
                    _active_model = model
                return text
            except ModelUnavailable as e:
                print(f"Gemini: модель {model} недоступна ({e}) — пробую следующую")
                errors.append(f"{model}: {e}")
                if model == _active_model:
                    _active_model = None
                break
            except PermissionError:
                raise
            except Exception as e:  # noqa: BLE001  сеть, 429, 5xx, пустой ответ
                wait = 5 * 2 ** attempt
                print(f"  Gemini ({model}): ошибка ({e}); повтор через {wait} с")
                errors.append(f"{model}: {e}")
                time.sleep(wait)
        else:
            print(f"Gemini: модель {model} так и не ответила — пробую следующую")
    raise RuntimeError("Ни одна модель Gemini не ответила:\n  " + "\n  ".join(errors[-6:]))


def _outline(notes: str, n_parts: int, api_key: str) -> dict:
    prompt = f"""Вот мои заметки по главам манхвы (своими словами):
---
{notes}
---
Составь план видео-пересказа из {n_parts} частей. Первая часть — вступление сэра Лата \
(приветствие, о чём выпуск, интрига), последняя — итоги, оценка и прощание. Остальные — \
пересказ событий по порядку с комментариями.
Верни JSON: {{"title": "название выпуска (без названия манхвы, если его нет в заметках)",
"parts": [{{"title": "заголовок части", "points": "что именно рассказать (2–4 предложения \
по заметкам)", "scene": "подземелье|таверна|город|герой|монстр|окно", "stock": "2-4 english words"}}]}}"""
    raw = _gemini(prompt, api_key, json_mode=True, temperature=0.7)
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


def _part_text(notes: str, outline: dict, idx: int, words: int, api_key: str) -> str:
    part = outline["parts"][idx]
    plan = "\n".join(f"{i + 1}. {p['title']}" for i, p in enumerate(outline["parts"]))
    prompt = f"""Заметки автора по главам:
---
{notes}
---
План выпуска «{outline['title']}»:
{plan}

Напиши ТОЛЬКО часть {idx + 1} «{part['title']}». Что в ней рассказать: {part['points']}
Объём: примерно {words} слов (это важно для длины видео). Не повторяй другие части, \
не здоровайся заново, если это не первая часть.
Начни со строк {{сцена: {part.get('scene', 'подземелье')}}} и {{сток: {part.get('stock', 'fantasy')}}}, \
меняй сцену 2–4 раза по ходу части. Дальше — строки вида [эмоция] фраза."""
    return _gemini(prompt, api_key)


def generate_long(notes: str, api_key: str, minutes: float = 12) -> str:
    words_total = int(minutes * 105)
    n_parts = max(5, min(12, round(minutes / 2)))
    print(f"Gemini: план из {n_parts} частей (~{words_total} слов)…")
    outline = _outline(notes, n_parts, api_key)
    out = [f"# {outline.get('title', 'Хроники сэра Лата')}"]
    per_part = words_total // len(outline["parts"])
    for i, p in enumerate(outline["parts"]):
        print(f"Gemini: часть {i + 1}/{len(outline['parts'])} — {p['title']}")
        out.append(f"\n## {p['title']}")
        out.append(_part_text(notes, outline, i, per_part, api_key))
    return "\n".join(out)


def generate_short(notes: str, api_key: str) -> str:
    prompt = f"""Заметки автора по главам:
---
{notes}
---
Напиши сценарий вертикального шортс-тизера (до 50 секунд озвучки, 85–105 слов).
Первая фраза — мощный крючок. Затем 2–3 самых интригующих момента без спойлера развязки \
и шуточная оценка. Последняя фраза — зови смотреть полный выпуск на канале.
Формат: первая строка «# название», затем «## Тизер», строка {{сцена: ...}}, {{сток: ...}}, \
затем строки [эмоция] фраза. Сцену меняй 2–3 раза."""
    return _gemini(prompt, api_key)


# -------------------------------------------------------- офлайн-заглушка ---

def offline_script(notes: str, short: bool = False) -> str:
    """Простейший сценарий без Gemini — для проверки сборки видео."""
    lines = [ln.strip("-• \t") for ln in notes.splitlines() if ln.strip()]
    scenes = ["подземелье", "таверна", "город", "герой", "монстр"]
    emos = ["удивлённый", "задумчивый", "смеющийся", "злой", "испуганный", "важный"]
    out = ["# Хроники сэра Лата: тестовый выпуск"]
    if short:
        out += ["## Тизер", "{сцена: монстр}", "{сток: dark castle}",
                "[испуганный, щит] Друзья, тут такое случилось, что я спрятался за щит!"]
        for i, ln in enumerate(lines[:3]):
            out += [f"{{сцена: {scenes[i % 5]}}}", f"[{emos[i % 6]}] {ln}"]
        out += ["[важный, указывает] Полный разбор уже на канале. Рыцарское слово!"]
        return "\n".join(out)
    out += ["## Вступление", "{сцена: таверна}", "{сток: medieval tavern}",
            "[важный] Приветствую, благородные зрители! С вами сэр Пингвин Лат.",
            "[задумчивый] Сегодня разбираем свежие главы. Садитесь поудобнее.",
            "{окно: Начато задание: досмотреть до конца}",
            "[смеющийся] Награда за задание — моё глубокое уважение!"]
    for i, ln in enumerate(lines):
        out += [f"## Эпизод {i + 1}", f"{{сцена: {scenes[i % 5]}}}", "{сток: fantasy landscape}",
                f"[{emos[i % 6]}] Итак, что у нас дальше.", f"[{emos[(i + 1) % 6]}] {ln}",
                f"[{emos[(i + 2) % 6]}, меч] Вот это я понимаю, поворот!"]
    out += ["## Итоги", "{сцена: герой}",
            "[важный, указывает] Моя оценка: восемь рыбок из десяти.",
            "[смеющийся] Подписывайтесь, и да хранит вас забрало!"]
    return "\n".join(out)


def make_script(notes: str, fmt: str, minutes: float = 12, offline: bool = False) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if offline or not key:
        if not offline:
            print("ВНИМАНИЕ: GEMINI_API_KEY не задан — использую тестовый офлайн-сценарий.")
        return offline_script(notes, short=(fmt == "short"))
    if fmt == "short":
        return generate_short(notes, key)
    return generate_long(notes, key, minutes)
