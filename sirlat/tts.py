"""Озвучка по фразам с точными таймингами слов.

Цепочка движков (первый ответивший используется для всего видео):
  1. Silero TTS v4_ru (голос eugene, затем aidar) — основной;
  2. edge-tts ru-RU-DmitryNeural — тайминги слов из WordBoundary;
     несколько вариантов запроса и повторы с паузой;
  3. gTTS.
Любая ошибка движка не роняет сборку: берётся следующий голос.
У запасных движков тайминги слов даёт faster-whisper (см. tts_fallback.py).
Озвученные фразы кэшируются в .cache/tts/.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from .config import CACHE, SAMPLE_RATE, VOICE, VOICE_PITCH, VOICE_RATE


@dataclass
class Word:
    text: str      # слово как в сценарии (с пунктуацией)
    start: float   # секунды от начала фразы
    end: float


@dataclass
class Speech:
    audio: np.ndarray   # float32 mono SAMPLE_RATE
    words: list[Word]
    engine: str = "edge"

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]", "", s.lower().replace("ё", "е"))


def fmt_rate(value) -> str:
    """'-8%', '-8', -8, '8%' -> '-8%' / '+8%' (edge-tts требует знак и %)."""
    m = re.fullmatch(r"\s*([+-]?)(\d+)\s*%?\s*", str(value))
    if not m:
        raise ValueError(f"Неверная скорость речи: {value!r} (пример: -8%)")
    return f"{m.group(1) or '+'}{int(m.group(2))}%"


def fmt_pitch(value) -> str:
    """'-6Hz', '-6', -6, '6hz' -> '-6Hz' / '+6Hz'."""
    m = re.fullmatch(r"\s*([+-]?)(\d+)\s*(?:hz)?\s*", str(value), re.I)
    if not m:
        raise ValueError(f"Неверная высота голоса: {value!r} (пример: -6Hz)")
    return f"{m.group(1) or '+'}{int(m.group(2))}Hz"


RATE, PITCH = fmt_rate(VOICE_RATE), fmt_pitch(VOICE_PITCH)


def decode_audio(data: bytes) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=data, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


async def _edge(text: str, rate: str, pitch: str) -> tuple[bytes, list[dict]]:
    import edge_tts
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    com = edge_tts.Communicate(text, VOICE, rate=rate, pitch=pitch,
                               boundary="WordBoundary", proxy=proxy)
    audio = bytearray()
    bounds = []
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            bounds.append({
                "text": chunk["text"],
                "start": chunk["offset"] / 1e7,
                "end": (chunk["offset"] + chunk["duration"]) / 1e7,
            })
    return bytes(audio), bounds


def _edge_phrase(text: str) -> tuple[bytes, list[dict]]:
    """Несколько вариантов запроса, у каждого — повторы с растущей паузой."""
    variants = [(RATE, PITCH, 3), (RATE, "+0Hz", 2), ("+0%", "+0Hz", 2)]
    last = None
    for rate, pitch, attempts in variants:
        for i in range(attempts):
            try:
                data, bounds = asyncio.run(_edge(text, rate, pitch))
                if len(data) < 1000:
                    raise RuntimeError("пустой звук")
                if (rate, pitch) != (RATE, PITCH):
                    print(f"  edge-tts ответил с параметрами {rate}, {pitch}")
                return data, bounds
            except Exception as e:  # noqa: BLE001
                last = e
                wait = 3 * 2 ** i
                print(f"  edge-tts ({rate}, {pitch}): {type(e).__name__}: {str(e)[:120]}; "
                      f"повтор через {wait} с")
                time.sleep(wait)
    raise RuntimeError(f"edge-tts не отвечает: {type(last).__name__}: {last}")


def align(text: str, bounds: list[dict], total: float) -> list[Word]:
    """Сопоставляет границы слов edge-tts с токенами исходной фразы."""
    tokens = text.split()
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    cur, consumed = 0, ""
    for b in bounds:
        nb = _norm(b["text"])
        if not nb:
            continue
        for j in range(cur, min(cur + 4, len(tokens))):
            nt = _norm(tokens[j])
            if not nt:
                continue
            done = consumed if j == cur else ""
            rest = nt[len(done):]
            if rest.startswith(nb) or nb.startswith(rest):
                start = times[j][0] if times[j] else b["start"]
                times[j] = (start, b["end"])
                done += nb
                if len(done) >= len(nt):
                    cur, consumed = j + 1, ""
                else:  # слово озвучено несколькими кусками (через дефис и т.п.)
                    cur, consumed = j, done
                break
        else:
            # числа и символы edge-tts произносит словами — привязываем к текущему токену
            if cur < len(tokens) and re.search(r"\d", tokens[cur]):
                start = times[cur][0] if times[cur] else b["start"]
                times[cur] = (start, b["end"])
    # интерполяция пропусков (числа, символы, несовпадения)
    words = []
    for i, tok in enumerate(tokens):
        if times[i] is None:
            prev_end = next((times[k][1] for k in range(i - 1, -1, -1) if times[k]), 0.0)
            nxt = next(((k, times[k][0]) for k in range(i + 1, len(tokens)) if times[k]), (len(tokens), total))
            gap_n = nxt[0] - i + 1
            step = max(0.05, (nxt[1] - prev_end) / gap_n)
            times[i] = (prev_end, prev_end + step)
        words.append(Word(tok, times[i][0], max(times[i][1], times[i][0] + 0.05)))
    return words


def _fake_tts(text: str) -> tuple[np.ndarray, list[dict]]:
    """Офлайн-«бормотание» для проверки липсинка и субтитров без сети."""
    sr = SAMPLE_RATE
    out, bounds, t = [], [], 0.08
    out.append(np.zeros(int(0.08 * sr), np.float32))
    rng = np.random.default_rng(len(text))
    for w in text.split():
        n = max(1, len(_norm(w)))
        dur = 0.12 + 0.055 * n
        ts = np.arange(int(dur * sr)) / sr
        syl = max(1, sum(ch in "аеёиоуыэюя" for ch in w.lower()))
        env = np.abs(np.sin(np.pi * syl * ts / dur)) ** 0.7
        f0 = 110 + 15 * rng.standard_normal()
        sig = (np.sin(2 * np.pi * f0 * ts) + 0.4 * np.sin(2 * np.pi * 2.7 * f0 * ts)) * env * 0.3
        out.append(sig.astype(np.float32))
        bounds.append({"text": w, "start": t, "end": t + dur})
        t += dur
        gap = 0.06 + (0.2 if w[-1] in ",.!?…" else 0)
        out.append(np.zeros(int(gap * sr), np.float32))
        t += gap
    return np.concatenate(out), bounds


SILERO_FIRST = os.getenv("SIRLAT_SILERO_SPEAKER", "eugene")
# Порядок голосов: Silero основной (edge-tts на GitHub Actions часто не отвечает),
# edge-tts — запасной, gTTS — последний. Можно переопределить: SIRLAT_TTS=edge,silero,gtts
_SILERO = [f"silero:{sp}" for sp in dict.fromkeys([SILERO_FIRST, "eugene", "aidar"])]
_ORDER = [x.strip() for x in os.getenv("SIRLAT_TTS", "silero,edge,gtts").split(",") if x.strip()]
ENGINES = [e for name in _ORDER for e in (_SILERO if name == "silero" else [name])]
_broken: set[str] = set()
_announced: set[str] = set()


def _run_engine(engine: str, text: str) -> tuple[np.ndarray, list[dict]]:
    from . import tts_fallback as fb
    if engine == "edge":
        data, bounds = _edge_phrase(text)
        return decode_audio(data), bounds
    if engine.startswith("silero:"):
        audio = fb.silero(text, engine.split(":", 1)[1], RATE, PITCH)
    elif engine == "gtts":
        audio = fb.gtts(text, RATE, PITCH)
    else:
        raise ValueError(engine)
    bounds = fb.whisper_words(audio, prompt=text) or fb.estimate_words(text, audio)
    return audio, bounds


def synthesize(text: str, offline: bool = False, engine: str | None = None) -> Speech:
    """Озвучивает фразу первым работающим движком (или заданным engine)."""
    if offline:
        audio, bounds = _fake_tts(text)
        return Speech(audio, align(text, bounds, len(audio) / SAMPLE_RATE), "offline")
    d = CACHE / "tts"
    d.mkdir(parents=True, exist_ok=True)
    chain = [engine] if engine else [e for e in ENGINES if e not in _broken]
    errors = []
    for eng in chain:
        if not engine and eng in _broken:
            continue
        key = hashlib.sha1(f"{eng}|{VOICE}|{RATE}|{PITCH}|{text}".encode()).hexdigest()[:20]
        f_audio, f_meta = d / f"{key}.npy", d / f"{key}.json"
        if f_audio.exists() and f_meta.exists():
            audio = np.load(f_audio)
            bounds = json.loads(f_meta.read_text())
        else:
            try:
                audio, bounds = _run_engine(eng, text)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{eng}: {type(e).__name__}: {str(e)[:200]}")
                if not engine:
                    # torch/модель Silero не загрузились — второй голос Silero тоже не поможет
                    _broken.update(_SILERO if eng.startswith("silero:") else [eng])
                    print(f"TTS: {eng} не сработал ({type(e).__name__}) — переключаюсь на запасной голос")
                continue
            np.save(f_audio, audio.astype(np.float32))
            f_meta.write_text(json.dumps(bounds, ensure_ascii=False))
        if eng not in _announced:
            print(f"TTS: озвучиваю движком {eng}")
            _announced.add(eng)
        return Speech(audio, align(text, bounds, len(audio) / SAMPLE_RATE), eng)
    raise RuntimeError("Ни один голос не сработал:\n  " + "\n  ".join(errors))


def mouth_levels(audio: np.ndarray, fps: int, n_frames: int) -> np.ndarray:
    """Уровень открытия клюва 0..3 для каждого кадра по громкости голоса."""
    hop = SAMPLE_RATE / fps
    rms = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        a, b = int(i * hop), int((i + 1) * hop)
        seg = audio[a:b]
        if len(seg):
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    voiced = rms[rms > 1e-3]
    ref = np.percentile(voiced, 90) if len(voiced) else 1.0
    x = np.clip(rms / max(ref, 1e-6), 0, 1.5)
    # лёгкое сглаживание, чтобы клюв не «дребезжал»
    x = np.convolve(x, [0.25, 0.5, 0.25], mode="same")
    levels = np.zeros(n_frames, np.int8)
    levels[x > 0.12] = 1
    levels[x > 0.38] = 2
    levels[x > 0.68] = 3
    return levels


def mouth_envelope(audio: np.ndarray, fps: int, n_frames: int) -> np.ndarray:
    """Громкость голоса для клюва: 0..1 на кадр, плавно (а не «открыт/закрыт»)."""
    sub = 4                                  # считаем чаще кадров, потом сглаживаем
    hop = SAMPLE_RATE / (fps * sub)
    n = n_frames * sub
    rms = np.zeros(n, np.float32)
    for i in range(n):
        seg = audio[int(i * hop):int((i + 1) * hop) + int(hop)]
        if len(seg):
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    voiced = rms[rms > 1e-3]
    ref = np.percentile(voiced, 92) if len(voiced) else 1.0
    x = np.clip(rms / max(ref, 1e-6), 0, 1.3)
    x = np.clip((x - 0.06) / 0.94, 0, None) ** 0.75       # тише — почти закрыт, громко — шире
    x = np.convolve(x, np.ones(3) / 3, mode="same")
    return np.clip(x.reshape(n_frames, sub).mean(1), 0, 1)
