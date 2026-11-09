"""Озвучка своим голосом: запись с телефона -> чистый голос сэра Лата + сценарий.

Если в папке voice/ лежит запись с тем же именем, что и заметки
(например notes/goblin_ch1.txt -> voice/goblin_ch1.m4a), видео делается по ней:

  1. шумоподавление (ffmpeg afftdn) и срез гула/шипения;
  2. распознавание faster-whisper с метками времени каждого слова;
  3. чистка: слова-паразиты («э», «эм», «ну», «короче», «типа»…), повторы,
     оборванные слова и всё нераспознанное между словами вырезаются,
     паузы длиннее 0.4 с становятся границами фраз;
  4. обработка под персонажа: сдвиг высоты VOICE_PITCH (в полутонах, по
     умолчанию 0), тёплый эквалайзер, компрессор, громкость −14 LUFS;
  5. эмоции сэра Лата по смыслу фраз (Gemini, если есть ключ, иначе правила).

Субтитры и сценарий берутся из распознанного текста. Почищенный голос
сохраняется отдельным mp3 для прослушивания.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT, SAMPLE_RATE

VOICE_DIR = ROOT / "voice"
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".opus", ".aac", ".flac", ".amr", ".3gp", ".webm", ".mp4", ".mov"}
WORK_SR = 48000
MAX_PAUSE = 0.4

FILLERS = {"э", "ээ", "эээ", "эээээ", "эм", "эмм", "ээм", "ам", "мм", "ммм", "хм", "хмм", "ну", "короче",
           "типа", "кхм", "эхм", "а-а", "ааа"}
FILLER_BIGRAMS = {("как", "бы"), ("это", "самое")}


def _norm(w: str) -> str:
    return re.sub(r"[^0-9a-zа-я]", "", w.lower().replace("ё", "е"))


# ------------------------------------------------------------ поиск записи ---

class RecordingError(RuntimeError):
    """Запись с именем заметок есть, но использовать её нельзя (пустая, не аудио)."""


def voice_dirs() -> list[Path]:
    """Все папки voice/ в репозитории (корневая — первой)."""
    dirs = [VOICE_DIR] if VOICE_DIR.is_dir() else []
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir() and p.name.lower() == "voice" and p != VOICE_DIR and not (
                {".git", "output", ".cache", "node_modules"} & set(p.parts)):
            dirs.append(p)
    return dirs


def _split_name(name: str) -> tuple[str, str]:
    name = name.rstrip(".")
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return base, "." + ext.lower()
    return name, ""


def _matches(base: str, stem: str) -> bool:
    b, s_ = base.lower().strip(), stem.lower()
    # «goblin_ch1», «Goblin_ch1», «goblin_ch1 (1)» — да; «goblin_ch10» — нет
    return b == s_ or b.startswith(s_ + " ") or b.startswith(s_ + "(")


def find_recording(stem: str) -> Path | None:
    """Запись голоса: voice/<имя заметок>.<любое аудио-расширение> во всех папках voice/.

    Пишет в лог все файлы папок voice/ и какой взят. Если файл с этим именем
    есть, но пустой или не аудио — RecordingError (сборка не уйдёт в TTS молча).
    """
    if not stem:
        return None
    dirs = voice_dirs()
    if not dirs:
        print("Голос: папки voice/ нет — озвучка синтезом речи")
        return None
    candidates = []
    print("Голос: файлы в папках voice/:")
    for d in dirs:
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            size = p.stat().st_size
            print(f"  {p.relative_to(ROOT)}  ({size / 1024:.0f} КБ)" if size >= 1024
                  else f"  {p.relative_to(ROOT)}  ({size} байт)")
            base, ext = _split_name(p.name)
            if _matches(base, stem) and ext not in (".txt", ".md", ".json"):
                candidates.append(p)
    if not candidates:
        print(f"Голос: записи «{stem}» нет — озвучка синтезом речи")
        return None
    good = [p for p in candidates if p.stat().st_size > 4096 and _probe_seconds(p) > 0.5]
    if not good:
        bad = ", ".join(f"{p.relative_to(ROOT)} ({p.stat().st_size} байт)" for p in candidates)
        raise RecordingError(
            f"Запись голоса для «{stem}» есть, но её нельзя прочитать: {bad}. Скорее всего, загрузка "
            "с телефона не прошла (файл пустой). Загрузите запись заново как voice/"
            f"{stem}.m4a или .mp3 (через сайт GitHub — до 25 МБ) и запустите сборку снова. "
            "Чтобы озвучить синтезом речи, удалите этот файл.")
    # предпочитаем файл с понятным аудио-расширением и самый большой
    good.sort(key=lambda p: (_split_name(p.name)[1] not in AUDIO_EXT, -p.stat().st_size))
    rec = good[0]
    print(f"Голос: взят файл {rec.relative_to(ROOT)} ({rec.stat().st_size / 1e6:.1f} МБ, "
          f"{_probe_seconds(rec):.0f} с) — озвучка TTS не используется")
    return rec


def _probe_seconds(p: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


# -------------------------------------------------------------------- звук ---

def _ffmpeg(args_in, af: str | None, sr: int, data: bytes | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"] + args_in
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    res = subprocess.run(cmd, input=data, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg: {res.stderr.decode(errors='ignore')[-400:]}")
    return np.frombuffer(res.stdout, np.float32).copy()


def _from_array(x: np.ndarray, sr: int, af: str, out_sr: int) -> np.ndarray:
    return _ffmpeg(["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0"], af, out_sr, x.astype(np.float32).tobytes())


def denoise(path: Path) -> np.ndarray:
    """Шумоподавление и срез гула/шипения. Результат: float32 mono WORK_SR."""
    af = "highpass=f=75,lowpass=f=13000,afftdn=nr=14:nf=-30:tn=1,afftdn=nr=6:nf=-40"
    x = _ffmpeg(["-i", str(path)], af, WORK_SR)
    if len(x) < WORK_SR * 0.5:
        raise RuntimeError("запись слишком короткая или не читается")
    return x


def character_chain(x: np.ndarray, pitch: float) -> np.ndarray:
    """Высота (полутоны), тёплый эквалайзер, компрессор, −14 LUFS."""
    flt = []
    if abs(pitch) > 0.01:
        f = 2 ** (pitch / 12)
        flt += [f"asetrate={int(WORK_SR * f)}", f"aresample={WORK_SR}", f"atempo={1 / f:.5f}"]
    flt += [
        "equalizer=f=180:t=q:w=0.9:g=2.5",     # тепло и «грудь»
        "equalizer=f=3200:t=q:w=1.4:g=-1.5",   # меньше резкости
        "equalizer=f=6500:t=q:w=2:g=-2",       # мягче шипящие
        "highshelf=f=11000:g=-2",
        "acompressor=threshold=-21dB:ratio=3:attack=8:release=160:makeup=3",
        "alimiter=limit=0.95",
    ]
    y = _from_array(x, WORK_SR, ",".join(flt), WORK_SR)
    # громкость: двухпроходный loudnorm до −14 LUFS
    meas = subprocess.run(
        ["ffmpeg", "-v", "info", "-f", "f32le", "-ar", str(WORK_SR), "-ac", "1", "-i", "pipe:0",
         "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
        input=y.astype(np.float32).tobytes(), capture_output=True).stderr.decode(errors="ignore")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas, re.S)
    if m:
        j = json.loads(m.group(0))
        ln = (f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
              f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:"
              f"offset={j['target_offset']}:linear=true")
    else:
        ln = "loudnorm=I=-14:TP=-1:LRA=11"
    z = _from_array(y, WORK_SR, ln, WORK_SR)
    # контроль: если до −14 LUFS не дотянули (мешал потолок пиков) — доводим
    # усилением с лимитером, пока не уложимся в ±0.5 LU
    for _ in range(3):
        cur = measure_lufs(z, WORK_SR)
        if cur is None or abs(cur + 14) <= 0.5:
            break
        z = _from_array(z, WORK_SR, f"volume={-14 - cur:.2f}dB,alimiter=limit=0.89:level=false", WORK_SR)
    # фильтры могут чуть сдвинуть длину — приводим к исходной
    n = len(y)
    return z[:n] if len(z) >= n else np.pad(z, (0, n - len(z)))


def measure_lufs(x: np.ndarray, sr: int) -> float | None:
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                          "-i", "pipe:0", "-af", "ebur128", "-f", "null", "-"],
                         input=x.astype(np.float32).tobytes(), capture_output=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", res.stderr.decode(errors="ignore"))
    return float(m[-1]) if m else None


def save_mp3(x: np.ndarray, sr: int, path: Path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                    "-c:a", "libmp3lame", "-b:a", "192k", str(path)],
                   input=x.astype(np.float32).tobytes(), check=True)


# ------------------------------------------------------------ распознавание ---

def transcribe(x: np.ndarray) -> list[dict]:
    from faster_whisper import WhisperModel
    name = os.getenv("WHISPER_MODEL", "medium")
    model = WhisperModel(name, device="cpu", compute_type="int8")
    x16 = _from_array(x, WORK_SR, None, 16000)
    # подсказка «как говорят», чтобы whisper записывал и паразиты (их потом вырежем)
    prompt = "Э-э, ну, короче, эм... Приветствую, путники! Я сэр Лат."
    segs, info = model.transcribe(x16, language="ru", word_timestamps=True, initial_prompt=prompt,
                                  vad_filter=False, beam_size=5, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words or []:
            t = w.word.strip()
            if t and words and (t.startswith("-") or w.word[:1] not in (" ", "") and t[:1] in "-—"):
                words[-1]["text"] += t                   # «по» + «-настоящему»
                words[-1]["end"] = float(w.end)
                continue
            if t:
                words.append(dict(text=t, start=float(w.start), end=float(w.end),
                                  p=float(getattr(w, "probability", 1.0))))
    print(f"Свой голос: faster-whisper ({name}) распознал {len(words)} слов, {x16.size / 16000:.0f} с записи")
    return words


# -------------------------------------------------------------------- чистка ---

def clean_words(words: list[dict]) -> tuple[list[dict], dict]:
    """Убирает паразиты, повторы и оборванные слова. Возвращает (слова, статистика)."""
    stats = {"паразиты": 0, "повторы": 0, "оборванные": 0}
    ws = [dict(w, n=_norm(w["text"])) for w in words]
    keep = [True] * len(ws)
    for i, w in enumerate(ws):
        if not w["n"]:
            keep[i] = False
        elif w["n"] in FILLERS or re.fullmatch(r"э+м*|м+|а+|хм+", w["n"]):
            keep[i] = False
            stats["паразиты"] += 1
        elif i + 1 < len(ws) and (w["n"], ws[i + 1]["n"]) in FILLER_BIGRAMS:
            keep[i] = keep[i + 1] = False
            stats["паразиты"] += 1
    # оборванные: «пом- помнит», «прос просыпается», слово с дефисом/многоточием в конце
    for i, w in enumerate(ws):
        if not keep[i]:
            continue
        nxt = next((j for j in range(i + 1, len(ws)) if keep[j]), None)
        cut = w["text"].endswith(("-", "—", "…", "...")) and len(w["n"]) <= 6
        if nxt is not None:
            a, b = w["n"], ws[nxt]["n"]
            if len(a) >= 2 and a != b and b.startswith(a) and len(b) - len(a) >= 2:
                cut = True
        if cut and nxt is not None:
            keep[i] = False
            stats["оборванные"] += 1
    # повторы: «помнит помнит», «я сэр я сэр», «ничего не помнит ничего не помнит»
    idx = [i for i in range(len(ws)) if keep[i]]
    k = 0
    while k < len(idx):
        removed = False
        for n in (4, 3, 2, 1):
            if k + 2 * n <= len(idx):
                a = [ws[i]["n"] for i in idx[k:k + n]]
                b = [ws[i]["n"] for i in idx[k + n:k + 2 * n]]
                gap = ws[idx[k + n]]["start"] - ws[idx[k + n - 1]]["end"]
                if a == b and gap < 1.5 and not (n == 1 and len(a[0]) <= 2):
                    for i in idx[k:k + n]:
                        keep[i] = False
                    stats["повторы"] += 1
                    idx = idx[:k] + idx[k + n:]
                    removed = True
                    break
        if not removed:
            k += 1
    # пауза перед словом — только настоящая тишина, без вырезанных паразитов/повторов
    out, removed_dur = [], 0.0
    for w, kp in zip(ws, keep):
        if kp:
            if out:
                w["pause"] = max(0.0, w["start"] - out[-1]["end"] - removed_dur)
            else:
                w["pause"] = 0.0
            out.append(w)
            removed_dur = 0.0
        elif out:
            removed_dur += w["end"] - w["start"] + 0.1
    return out, stats


def split_phrases(words: list[dict], max_words: int = 16) -> list[list[dict]]:
    """Границы фраз: пауза > 0.4 с, конец предложения или слишком длинная фраза."""
    out, cur = [], []
    for i, w in enumerate(words):
        if cur:
            gap = w.get("pause", w["start"] - cur[-1]["end"])
            end_sent = cur[-1]["text"].rstrip("»\"')").endswith((".", "!", "?", "…")) and len(cur) >= 4
            long_ = len(cur) >= max_words and cur[-1]["text"].endswith((",", ";", ":", "—"))
            if (gap > MAX_PAUSE and len(cur) >= 2) or end_sent or long_ or len(cur) >= max_words + 6:
                out.append(cur)
                cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _phrase_audio(x: np.ndarray, words: list[dict], sr: int) -> tuple[np.ndarray, list[dict]]:
    """Склеивает звук слов фразы: вырезает всё между словами длиннее 0.12 с
    (там были паразиты, повторы или тишина), края — с плавными переходами."""
    pad_a, pad_b, fade = 0.05, 0.07, int(0.012 * sr)
    spans = []
    for w in words:
        a, b = max(0.0, w["start"] - pad_a), min(len(x) / sr, w["end"] + pad_b)
        if spans and a - spans[-1][1] < 0.12:
            spans[-1][1] = max(spans[-1][1], b)     # соседние слова — одним куском
        else:
            spans.append([a, b])
    pieces, out_words, pos = [], [], 0.0
    gap = np.zeros(int(0.08 * sr), np.float32)      # короткая «склейка» между кусками
    span_of = []
    for si, (a, b) in enumerate(spans):
        seg = x[int(a * sr):int(b * sr)].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        if si:
            pieces.append(gap)
            pos += len(gap) / sr
        span_of.append((a, pos))
        pieces.append(seg)
        pos += len(seg) / sr
    for w in words:
        a0, p0 = next((a, p) for (a, p), (sa, sb) in zip(span_of, spans) if sa <= w["start"] + 1e-6 <= sb + 0.2)
        out_words.append(dict(text=w["text"], start=p0 + (w["start"] - a0), end=p0 + (w["end"] - a0)))
    return np.concatenate(pieces) if pieces else np.zeros(1, np.float32), out_words


# -------------------------------------------------------------------- эмоции ---

EMO_RULES = [
    ("laughing", r"ха-?ха|хах|смешн|шутк|забавн|ржу|умора|комед|улыб|весел"),
    ("scared", r"страш|жутк|ужас|боюсь|опасн|кошмар|дрож|бежать|спасайся|паник"),
    ("angry", r"бесит|злит|подл|предател|несправедл|ненавиж|возмут|наглост|грабеж|грабит|обман"),
    ("surprised", r"\bвау\b|\bого\b|неожидан|внезапно|поворот|не может быть|оказывается|удив|шок"),
    ("thoughtful", r"почему|интересно|загадк|тайн|кажется|возможно|думаю|наверн|вопрос|\?"),
]
POSE_RULES = [
    ("point", r"смотрите|обратите|вот\b|видите|запомните|подписыв|глав[аеу]"),
    ("sword", r"битв|сраж|\bбой\b|меч|атак|вперед|в бой|победа"),
    ("shield", r"страш|прячусь|спрят|защит|бежать|осторожн"),
]
RU_EMO = {"important": "важный", "surprised": "удивлённый", "laughing": "смеющийся",
          "angry": "злой", "thoughtful": "задумчивый", "scared": "испуганный"}
RU_POSE = {"point": "указывает", "sword": "меч", "shield": "щит"}


def emotions_by_rules(texts: list[str]) -> list[tuple[str, str]]:
    out = []
    for t in texts:
        low = t.lower().replace("ё", "е")
        emo = next((k for k, rx in EMO_RULES if re.search(rx, low)), None)
        if not emo:
            emo = "surprised" if t.count("!") >= 2 else "important"
        pose = next((k for k, rx in POSE_RULES if re.search(rx, low)), "none")
        out.append((emo, pose))
    return out


def emotions_by_gemini(texts: list[str]) -> list[tuple[str, str]] | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or not texts:
        return None
    from .script import _gemini
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = f"""Это фразы сэра Лата (пингвина-рыцаря, ведущего обзоров манхвы), уже записанные голосом.
Текст НЕ меняй. Для каждой фразы выбери эмоцию по смыслу и, если уместно, позу.
Эмоции: important (важный), surprised, laughing, angry, thoughtful, scared.
Позы: none, point (указывает крылом), sword (держит меч), shield (прячется за щитом).
Не делай все фразы одинаковыми, но и не меняй эмоцию на каждой фразе без причины.
Фразы:
{numbered}
Верни JSON-массив из {len(texts)} объектов вида {{"e": "...", "p": "..."}} в том же порядке."""
    try:
        raw = _gemini(prompt, key, json_mode=True, temperature=0.3)
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        if len(data) != len(texts):
            raise ValueError(f"ожидалось {len(texts)} меток, пришло {len(data)}")
        out = []
        for d in data:
            e = d.get("e") if d.get("e") in RU_EMO else "important"
            p = d.get("p") if d.get("p") in ("none", "point", "sword", "shield") else "none"
            out.append((e, p))
        print("Свой голос: эмоции расставил Gemini")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"Свой голос: Gemini не расставил эмоции ({e}) — беру правила")
        return None


# -------------------------------------------------------------------- итог ---

@dataclass
class OwnVoice:
    phrases: list            # [(text, emotion, pose, audio float32 SAMPLE_RATE, words[{text,start,end}])]
    clean_mp3: Path
    stats: dict

    def script_text(self, title: str) -> str:
        lines = [f"# {title}", "## Своим голосом"]
        for text, emo, pose, _, _ in self.phrases:
            tag = RU_EMO[emo] + (f", {RU_POSE[pose]}" if pose != "none" else "")
            lines.append(f"[{tag}] {text}")
        return "\n".join(lines) + "\n"


def prepare(path: Path, out_dir: Path, stem: str) -> OwnVoice:
    print(f"Свой голос: запись {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    x = denoise(path)
    words = transcribe(x)
    if not words:
        raise RuntimeError("в записи не удалось распознать речь")
    kept, stats = clean_words(words)
    groups = split_phrases(kept)
    # звук каждой фразы
    raw_phr = [_phrase_audio(x, g, WORK_SR) for g in groups]
    # обработка всего голоса одним куском (фразы через 0.38 с), потом режем обратно
    gap = np.zeros(int(0.38 * WORK_SR), np.float32)
    bounds, chunks, pos = [], [], 0
    for i, (a, _) in enumerate(raw_phr):
        if i:
            chunks.append(gap)
            pos += len(gap)
        bounds.append((pos, pos + len(a)))
        chunks.append(a)
        pos += len(a)
    full = np.concatenate(chunks)
    pitch = float(os.getenv("VOICE_PITCH", "0") or 0)
    proc = character_chain(full, pitch)
    clean_mp3 = out_dir / f"{stem}_golos_chistyj.mp3"
    save_mp3(proc, WORK_SR, clean_mp3)
    proc24 = _from_array(proc, WORK_SR, None, SAMPLE_RATE)
    k = len(proc24) / len(proc)
    texts = []
    for g in groups:
        t = " ".join(w["text"] for w in g).strip()
        texts.append(t[:1].upper() + t[1:])
    emos = emotions_by_gemini(texts) or emotions_by_rules(texts)
    phrases = []
    for (a, b), (_, pw), text, (emo, pose) in zip(bounds, raw_phr, texts, emos):
        audio = proc24[int(a * k):int(b * k)]
        if pw:
            pw[0]["text"] = pw[0]["text"][:1].upper() + pw[0]["text"][1:]
        phrases.append((text, emo, pose, audio, pw))
    before = len(x) / WORK_SR
    after = len(full) / WORK_SR
    stats.update({"было, с": round(before, 1), "стало, с": round(after, 1), "фраз": len(phrases),
                  "высота, полутонов": pitch})
    stats["громкость, LUFS"] = measure_lufs(proc, WORK_SR)
    print("Свой голос: чистка — " + ", ".join(f"{k_}: {v}" for k_, v in stats.items()))
    print(f"Свой голос: почищенный звук → {clean_mp3}")
    return OwnVoice(phrases, clean_mp3, stats)
