"""Пробы голоса для сэра Лата: одна фраза разными голосами и в разной подаче.

    python -m sirlat.voice_samples                      # всё в voice_samples/
    python -m sirlat.voice_samples --text "Своя фраза"  --out папка

Голоса: Silero aidar и eugene, edge-tts ru-RU-DmitryNeural.
Подача (для каждого голоса):
    obychny     — как есть;
    vyshe       — чуть выше и быстрее (≈ +2,5 полутона, темп +10%);
    energichno  — заметно выше и энергичнее (≈ +4,5 полутона, темп +18%).
Громкость всех проб выровнена, чтобы сравнивать только голос.
Итог — mp3 файлы вида aidar_vyshe.mp3 и отчёт otchet.txt.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from .config import ROOT, SAMPLE_RATE

DEFAULT_TEXT = "Приветствую, путники! Я сэр Лат, и сегодня у нас самый безумный скрытый класс!"

# имя -> (описание, темп %, полутона, сдвиг edge-tts в Гц)
STYLES = {
    "obychny":    ("обычный", 0, 0.0, 0),
    "vyshe":      ("чуть выше и быстрее: +2,5 полутона, темп +10%", 10, 2.5, 17),
    "energichno": ("заметно выше и энергичнее: +4,5 полутона, темп +18%", 18, 4.5, 32),
}
VOICES = {
    "aidar": "Silero v4_ru, голос aidar",
    "eugene": "Silero v4_ru, голос eugene",
    "dmitry": "edge-tts ru-RU-DmitryNeural",
}


def _save_mp3(audio: np.ndarray, path: Path):
    """float32 mono SAMPLE_RATE -> mp3 с выравниванием громкости."""
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as fh:
        fh.write(audio.astype(np.float32).tobytes())
        raw = fh.name
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-i", raw, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100",
                        "-c:a", "libmp3lame", "-q:a", "2", str(path)], check=True)
    finally:
        Path(raw).unlink(missing_ok=True)


def _silero(text: str, speaker: str, tempo_pct: int, semis: float) -> np.ndarray:
    from . import tts_fallback as fb
    # prosody() в tts_fallback переводит Гц в полутона как 7 Гц = 1 полутон
    return fb.silero(text, speaker, f"{tempo_pct:+d}%", f"{round(semis * 7):+d}Hz")


def _shift(audio: np.ndarray, tempo_pct: int, semis: float) -> np.ndarray:
    from .tts_fallback import _ffmpeg_decode
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return _ffmpeg_decode(path, 1 + tempo_pct / 100, semis)
    finally:
        Path(path).unlink(missing_ok=True)


def _edge(text: str, tempo_pct: int, hz: int) -> np.ndarray:
    from .tts import _edge as edge_call, decode_audio
    last = None
    for i in range(4):
        try:
            data, _ = asyncio.run(edge_call(text, f"{tempo_pct:+d}%", f"{hz:+d}Hz"))
            if len(data) < 1000:
                raise RuntimeError("пустой звук")
            return decode_audio(data)
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 3 * 2 ** i
            print(f"  edge-tts: {type(e).__name__}: {str(e)[:120]}; повтор через {wait} с")
            time.sleep(wait)
    raise RuntimeError(f"edge-tts не ответил: {type(last).__name__}: {last}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Пробы голоса сэра Лата")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default=str(ROOT / "voice_samples"))
    args = ap.parse_args(argv)
    text = (args.text or DEFAULT_TEXT).strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = [f"Фраза: {text}", "", "Подача:"]
    report += [f"  {k:<11} — {v[0]}" for k, v in STYLES.items()]
    report += ["", "Файлы:"]
    ok = 0
    for voice, desc in VOICES.items():
        for style, (sdesc, tempo, semis, hz) in STYLES.items():
            name = f"{voice}_{style}.mp3"
            print(f"Проба: {name} ({desc}; {sdesc})")
            try:
                note = ""
                if voice == "dmitry":
                    try:
                        audio = _edge(text, tempo, hz)
                    except Exception:  # noqa: BLE001
                        if style == "obychny":
                            raise
                        # Microsoft иногда не отдаёт звук со сдвигом высоты —
                        # берём обычную запись и сдвигаем высоту/темп через ffmpeg
                        print("  edge-tts не дал звук с этими параметрами — сдвигаю обычную запись")
                        audio = _shift(_edge(text, 0, 0), tempo, semis)
                        note = " (высота и темп сдвинуты через ffmpeg)"
                else:
                    audio = _silero(text, voice, tempo, semis)
                _save_mp3(audio, out / name)
                dur = len(audio) / SAMPLE_RATE
                report.append(f"  {name:<24} {dur:4.1f} с   {desc}; {sdesc}{note}")
                ok += 1
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(limit=1)
                report.append(f"  {name:<24} НЕ ПОЛУЧИЛОСЬ: {type(e).__name__}: {str(e)[:150]}")
    total = len(VOICES) * len(STYLES)
    report += ["", f"Готово проб: {ok} из {total}.",
               "Модели Silero — лицензия CC BY-NC (некоммерческое использование)."]
    (out / "otchet.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if ok == 0:
        raise SystemExit("Ни одна проба не получилась")


if __name__ == "__main__":
    main()
