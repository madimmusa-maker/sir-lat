"""Голос сэра Лата — только Piper, модель IGM (rraaww/ru_piper, igm3804-russian-ru-medium).

Автор модели разрешил использование с монетизацией (см. assets/CREDITS.md).
Других голосов нет: если модель не скачалась или не загрузилась, сборка
останавливается с понятной ошибкой.

Эмоции в голосе — только через настройки Piper: по разметке фразы немного
меняются length_scale (темп, ±5–10%) и noise_scale/noise_w (живость интонации).
Тайминги слов для субтитров и клюва даёт faster-whisper (у модели их нет).
Озвученные фразы кэшируются в .cache/tts/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CACHE, SAMPLE_RATE

MODEL_NAME = "ru_RU-igm3804-medium"
MODEL_URL = "https://huggingface.co/rraaww/ru_piper/resolve/main/" + MODEL_NAME
MODEL_DIR = Path(os.path.expanduser(os.getenv("PIPER_MODEL_DIR", "~/.cache/piper")))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
BASE_LENGTH = float(os.getenv("SIRLAT_LENGTH", "1.0"))     # общий темп (1.0 — как у модели)

# эмоция -> (length_scale, noise_scale, noise_w): чуть быстрее/живее или медленнее/ровнее
EMO_VOICE = {
    "important":  (1.00, 0.667, 0.80),
    "surprised":  (0.93, 0.75, 0.90),
    "laughing":   (0.92, 0.76, 0.90),
    "angry":      (0.94, 0.72, 0.85),
    "scared":     (0.93, 0.74, 0.90),
    "thoughtful": (1.08, 0.60, 0.70),
    "intrigue":   (1.09, 0.60, 0.70),
}


class VoiceUnavailable(RuntimeError):
    """Голос IGM не загрузился — без него видео не собираем."""


@dataclass
class Word:
    text: str      # слово как в сценарии (с пунктуацией)
    start: float   # секунды от начала фразы
    end: float


@dataclass
class Speech:
    audio: np.ndarray   # float32 mono SAMPLE_RATE
    words: list[Word]
    engine: str = "piper-igm"

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]", "", s.lower().replace("ё", "е"))


# ------------------------------------------------------------------ модель ---

_voice = None


def _download(name: str, min_size: int):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_DIR / name
    if dst.exists() and dst.stat().st_size >= min_size:
        return dst
    import requests
    last = None
    for attempt in range(4):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(f"{MODEL_URL[:MODEL_URL.rfind('/') + 1]}{name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if tmp.stat().st_size < min_size:
                raise RuntimeError(f"файл слишком маленький ({tmp.stat().st_size} байт)")
            tmp.rename(dst)
            return dst
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise VoiceUnavailable(f"не удалось скачать {name} с huggingface.co/rraaww/ru_piper: {last}")


def load_voice():
    """Загружает Piper IGM (скачивает при первом запуске). Ошибка — VoiceUnavailable."""
    global _voice
    if _voice is not None:
        return _voice
    try:
        from piper import PiperVoice
    except ImportError as e:
        raise VoiceUnavailable(f"не установлен piper-tts ({e}); pip install -r requirements.txt") from e
    onnx = _download(MODEL_NAME + ".onnx", 10_000_000)
    cfg = _download(MODEL_NAME + ".onnx.json", 1000)
    try:
        json.loads(cfg.read_text(encoding="utf-8"))
        _voice = PiperVoice.load(str(onnx), config_path=str(cfg))
    except Exception as e:  # noqa: BLE001
        raise VoiceUnavailable(f"модель {MODEL_NAME} не загрузилась: {e}") from e
    print(f"Голос: Piper {MODEL_NAME} (rraaww/ru_piper) загружен")
    return _voice


# ----------------------------------------------------------------- звук ---

def _resample(x: np.ndarray, sr: int, out_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == out_sr:
        return x.astype(np.float32)
    res = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                          "-f", "f32le", "-ar", str(out_sr), "-ac", "1", "pipe:1"],
                         input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(res.stdout, np.float32).copy()


def piper_say(text: str, emotion: str) -> np.ndarray:
    from piper import SynthesisConfig
    from .ru_numbers import expand
    voice = load_voice()
    ls, ns, nw = EMO_VOICE.get(emotion, EMO_VOICE["important"])
    cfg = SynthesisConfig(length_scale=ls * BASE_LENGTH, noise_scale=ns, noise_w_scale=nw)
    chunks = list(voice.synthesize(expand(text), syn_config=cfg))
    if not chunks:
        raise VoiceUnavailable(f"Piper не вернул звук для фразы: {text!r}")
    sr = chunks[0].sample_rate
    x = np.concatenate([c.audio_float_array for c in chunks])
    return _resample(x, sr)


def postprocess(audio: np.ndarray) -> np.ndarray:
    """Единственная обработка: лёгкое шумоподавление и громкость −14 LUFS."""
    if len(audio) < SAMPLE_RATE // 2:
        return audio
    raw = audio.astype(np.float32).tobytes()
    base = ["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0"]
    dn = "afftdn=nr=6:nf=-45"
    meas = subprocess.run(base + ["-af", f"{dn},loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                          input=raw, capture_output=True).stderr.decode(errors="ignore")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas, re.S)
    ln = "loudnorm=I=-14:TP=-1.5:LRA=11"
    if m:
        j = json.loads(m.group(0))
        if j["input_i"] not in ("-inf", "inf"):
            ln += (f":measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
                   f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    res = subprocess.run(base + ["-af", f"{dn},{ln}", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
                         input=raw, capture_output=True)
    y = np.frombuffer(res.stdout, np.float32).copy() if res.returncode == 0 else audio
    n = len(audio)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


# ---------------------------------------------------- тайминги слов ---

_whisper = None
_whisper_broken = False


def whisper_words(audio: np.ndarray, prompt: str = "") -> list[dict] | None:
    global _whisper, _whisper_broken
    if _whisper_broken:
        return None
    try:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            print(f"faster-whisper: модель {WHISPER_MODEL} загружена (тайминги слов)")
        x16 = _resample(audio, SAMPLE_RATE, 16000)
        segs, _ = _whisper.transcribe(x16, language="ru", word_timestamps=True,
                                      initial_prompt=prompt[:400] or None, vad_filter=False, beam_size=5)
        words = [{"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                 for s in segs for w in (s.words or []) if w.word.strip()]
        return words or None
    except Exception as e:  # noqa: BLE001
        print(f"faster-whisper: ошибка ({type(e).__name__}: {e}) — тайминги будут примерными")
        _whisper_broken = True
        return None


def estimate_words(text: str, audio: np.ndarray) -> list[dict]:
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


def align(text: str, bounds: list[dict], total: float) -> list[Word]:
    """Сопоставляет слова, найденные faster-whisper, с токенами исходной фразы."""
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
            # числа голос произносит словами — привязываем к текущему токену
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


def synthesize(text: str, emotion: str = "important", offline: bool = False) -> Speech:
    """Озвучивает фразу голосом IGM с настройками по эмоции (кэш в .cache/tts/)."""
    ls, ns, nw = EMO_VOICE.get(emotion, EMO_VOICE["important"])
    d = CACHE / "tts"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{MODEL_NAME}|{ls * BASE_LENGTH}|{ns}|{nw}|{WHISPER_MODEL}|{text}".encode()).hexdigest()[:20]
    f_audio, f_meta = d / f"{key}.npy", d / f"{key}.json"
    if f_audio.exists() and f_meta.exists():
        audio = np.load(f_audio)
        bounds = json.loads(f_meta.read_text())
    else:
        audio = piper_say(text, emotion)
        bounds = whisper_words(audio, prompt=text) or estimate_words(text, audio)
        np.save(f_audio, audio.astype(np.float32))
        f_meta.write_text(json.dumps(bounds, ensure_ascii=False))
    return Speech(audio, align(text, bounds, len(audio) / SAMPLE_RATE))


def mouth_envelope(audio: np.ndarray, fps: int, n_frames: int) -> np.ndarray:
    """Громкость голоса для клюва, 0..1 на кадр: быстрая атака (≈20 мс) и мягкий
    спад (≈75 мс), чтобы клюв открывался на ударных слогах и плавно закрывался."""
    sub = 8                                   # 200 отсчётов в секунду при 25 кадрах
    rate = fps * sub
    hop = SAMPLE_RATE / rate
    n = n_frames * sub
    rms = np.zeros(n, np.float32)
    win = int(hop * 2)
    for i in range(n):
        a = int(i * hop)
        seg = audio[a:a + win]
        if len(seg):
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    voiced = rms[rms > 1e-3]
    ref = np.percentile(voiced, 95) if len(voiced) else 1.0
    x = np.clip(rms / max(ref, 1e-6), 0, 1.2)
    x = np.clip((x - 0.08) / 0.92, 0, None) ** 0.7      # тихо — закрыт, громко — шире
    att = 1 - np.exp(-1 / (rate * 0.02))
    rel = 1 - np.exp(-1 / (rate * 0.075))
    y = np.zeros_like(x)
    v = 0.0
    for i, xi in enumerate(x):
        v += (xi - v) * (att if xi > v else rel)
        y[i] = v
    return np.clip(y.reshape(n_frames, sub).mean(1), 0, 1)
