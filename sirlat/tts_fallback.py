"""Запасная озвучка, если edge-tts не отвечает: Silero TTS (v4_ru), затем gTTS.

У запасных движков нет таймингов слов, поэтому слова находит faster-whisper
(распознавание речи с метками времени); в субтитры всё равно идут слова
из сценария — whisper даёт только время.

torch для Silero ставится автоматически при первом обращении (CPU-сборка,
~200 МБ), чтобы не тянуть его в каждую сборку, где edge-tts работает.
Модели кэшируются: Silero — ~/.cache/silero, whisper — ~/.cache/huggingface.

Лицензия моделей Silero — CC BY-NC (некоммерческое использование).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

SILERO_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
SILERO_CACHE = Path(os.path.expanduser("~/.cache/silero/v4_ru.pt"))
SILERO_SR = 48000
SILERO_SPEAKERS = ("eugene", "aidar")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_silero = None
_whisper = None
_whisper_broken = False


def _retry(fn, what, attempts=3, delay=4):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < attempts:
                print(f"  {what}: ошибка ({type(e).__name__}: {str(e)[:160]}); повтор через {delay * 2 ** i} с")
                time.sleep(delay * 2 ** i)
    raise last


def _ffmpeg_decode(path: str, tempo: float = 1.0, semitones: float = 0.0) -> np.ndarray:
    """Аудиофайл -> float32 mono SAMPLE_RATE, с изменением темпа и высоты."""
    flt = [f"aresample={SAMPLE_RATE}"]   # сначала к общей частоте, иначе asetrate исказит темп
    if abs(semitones) > 0.01:
        f = 2 ** (semitones / 12)
        flt += [f"asetrate={int(SAMPLE_RATE * f)}", f"aresample={SAMPLE_RATE}", f"atempo={1 / f:.4f}"]
    if abs(tempo - 1) > 0.001:
        flt.append(f"atempo={max(0.5, min(2.0, tempo)):.4f}")
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    cmd += ["-af", ",".join(flt)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def prosody(rate: str, pitch: str) -> tuple[float, float]:
    """'-8%', '-6Hz' -> (темп 0.92, полутона ≈ -0.86) для мужского голоса (~7 Гц на полутон)."""
    r = float(rate.rstrip("%")) / 100
    p = float(pitch.lower().rstrip("hz"))
    return 1 + r, p / 7


# ------------------------------------------------------------------ Silero ---

def _ensure_torch():
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        return
    print("Silero: ставлю torch (CPU)…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)


def _load_silero():
    global _silero
    if _silero is not None:
        return _silero
    _ensure_torch()
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if not SILERO_CACHE.exists() or SILERO_CACHE.stat().st_size < 1_000_000:
        SILERO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(SILERO_CACHE) + ".part"
        _retry(lambda: torch.hub.download_url_to_file(SILERO_URL, tmp, progress=False),
               "Silero: загрузка модели")
        os.replace(tmp, SILERO_CACHE)
    model = torch.package.PackageImporter(str(SILERO_CACHE)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    _silero = model
    print("Silero: модель v4_ru загружена")
    return model


def silero(text: str, speaker: str, rate: str, pitch: str) -> np.ndarray:
    from .ru_numbers import expand
    model = _load_silero()
    spoken = expand(text)
    kw = dict(text=spoken, speaker=speaker, sample_rate=SILERO_SR, put_accent=True, put_yo=True)
    try:
        audio = model.apply_tts(**kw, put_stress_homo=True, put_yo_homo=True)
    except TypeError:
        audio = model.apply_tts(**kw)
    x = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    if len(x) < SILERO_SR * 0.2:
        raise RuntimeError("Silero вернул слишком короткий звук")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SILERO_SR)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        tempo, semi = prosody(rate, pitch)
        return _ffmpeg_decode(path, tempo, semi)
    finally:
        os.unlink(path)


# -------------------------------------------------------------------- gTTS ---

def gtts(text: str, rate: str, pitch: str) -> np.ndarray:
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
        path = fh.name
    try:
        def go():
            gTTS(text=text, lang="ru", tld="com", slow=False).save(path)
            if os.path.getsize(path) < 500:
                raise RuntimeError("gTTS вернул пустой звук")
        _retry(go, "gTTS", attempts=4, delay=5)
        tempo, semi = prosody(rate, pitch)
        # голос gTTS женский и быстрый: чуть медленнее и ниже, чтобы ближе к сэру Лату
        return _ffmpeg_decode(path, tempo * 0.97, semi - 2.0)
    finally:
        os.unlink(path)


# ---------------------------------------------------------- faster-whisper ---

def whisper_words(audio: np.ndarray, prompt: str = "") -> list[dict] | None:
    """Слова с временем [{'text','start','end'}] или None, если whisper не сработал."""
    global _whisper, _whisper_broken
    if _whisper_broken:
        return None
    try:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            print(f"faster-whisper: модель {WHISPER_MODEL} загружена")
        # whisper ждёт 16 кГц
        x = np.interp(np.arange(0, len(audio), SAMPLE_RATE / 16000), np.arange(len(audio)), audio)
        segs, _ = _whisper.transcribe(x.astype(np.float32), language="ru", word_timestamps=True,
                                      initial_prompt=prompt[:400] or None, vad_filter=False,
                                      beam_size=5)
        words = [{"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                 for s in segs for w in (s.words or []) if w.word.strip()]
        return words or None
    except Exception as e:  # noqa: BLE001
        print(f"faster-whisper: ошибка ({type(e).__name__}: {e}) — тайминги будут примерными")
        _whisper_broken = True
        return None


def estimate_words(text: str, audio: np.ndarray) -> list[dict]:
    """Примерные тайминги: слова делят звучащую часть фразы пропорционально длине."""
    win = SAMPLE_RATE // 100
    n = len(audio) // win
    rms = np.sqrt((audio[:n * win].reshape(n, win) ** 2).mean(1) + 1e-12) if n else np.zeros(1)
    loud = np.nonzero(rms > max(rms.max() * 0.06, 1e-3))[0]
    on, off = (loud[0] / 100, (loud[-1] + 1) / 100) if len(loud) else (0, len(audio) / SAMPLE_RATE)
    toks = text.split()
    wts = [len(t) + 1.5 for t in toks]
    k = (off - on) / max(1e-6, sum(wts))
    out, t = [], on
    for tok, w in zip(toks, wts):
        out.append({"text": tok, "start": t, "end": t + w * k * 0.95})
        t += w * k
    return out
