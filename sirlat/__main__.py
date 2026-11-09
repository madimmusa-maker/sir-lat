"""Командная строка: заметки -> сценарий -> озвучка -> видео.

Примеры:
    python -m sirlat --notes notes/example.txt --format long
    python -m sirlat --notes notes/example.txt --format short
    python -m sirlat --notes notes/example.txt --format both --minutes 15
    python -m sirlat --script output/example/script_long.txt --format long
    python -m sirlat --notes notes/example.txt --offline --preview 20   # быстрая проверка без сети
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .config import OUTPUT
from .render import LONG, SHORT, concat, part_duration, render_part, speak_part
from .script import Part, make_script, parse_script
from .stock import Stock

SHORT_LIMIT = 58.5   # шортс должен быть короче 59 секунд
LONG_MIN, LONG_MAX = 10 * 60, 30 * 60


def _fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def build(fmt: str, notes: str, out_dir: Path, args) -> Path:
    layout = LONG if fmt == "long" else SHORT
    script_file = out_dir / f"script_{fmt}.txt"
    if args.script:
        text = Path(args.script).read_text(encoding="utf-8")
    elif script_file.exists() and args.reuse_script:
        text = script_file.read_text(encoding="utf-8")
    else:
        text = make_script(notes, fmt, args.minutes, offline=args.offline)
    script_file.write_text(text, encoding="utf-8")
    script = parse_script(text)
    print(f"Сценарий «{script.title}»: {len(script.parts)} частей, {script.word_count()} слов → {script_file}")

    if fmt == "short":
        # один «кусок»: все фразы тизера
        phrases = [p for part in script.parts for p in part.phrases]
        script.parts = [Part(script.title, phrases)]

    print("Озвучка (Silero → edge-tts → gTTS)…")
    spoken_parts = [speak_part(p, args.offline) for p in script.parts]
    engines = [sp.engine for part in spoken_parts for _, sp in part]
    if len(set(engines)) > 1:
        # голос сменился посреди видео — переозвучиваем всё последним рабочим движком
        from collections import Counter
        final = engines[-1]
        print(f"Голос сменился по ходу озвучки ({dict(Counter(engines))}) — "
              f"переозвучиваю все фразы движком {final}")
        try:
            spoken_parts = [speak_part(p, args.offline, engine=final) for p in script.parts]
        except Exception as e:  # noqa: BLE001
            print(f"  не получилось ({type(e).__name__}: {e}) — оставляю смешанную озвучку")
    if engines:
        print(f"Озвучка: {engines[-1]}")

    if fmt == "short":
        sp = spoken_parts[0]
        while len(sp) > 2 and part_duration(sp) > SHORT_LIMIT:
            # убираем предпоследние фразы, сохраняя финальный призыв
            sp.pop(-2)
        spoken_parts[0] = sp
        script.parts[0].phrases = [ph for ph, _ in sp]

    total = sum(part_duration(s) for s in spoken_parts)
    print(f"Длительность по озвучке: {_fmt(total)}")
    if fmt == "long" and not args.preview:
        if total < LONG_MIN:
            print(f"ВНИМАНИЕ: видео короче 10 минут ({_fmt(total)}). "
                  "Увеличьте --minutes или добавьте больше заметок.")
        while total > LONG_MAX and len(spoken_parts) > 2:
            print("Видео длиннее 30 минут — убираю предпоследнюю часть.")
            spoken_parts.pop(-2)
            script.parts.pop(-2)
            total = sum(part_duration(s) for s in spoken_parts)

    stock = Stock(offline=args.offline)
    parts_dir = out_dir / f"parts_{fmt}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    files = []
    budget = args.preview
    for i, (part, spoken) in enumerate(zip(script.parts, spoken_parts)):
        if budget is not None and budget <= 0:
            break
        f = parts_dir / f"part_{i + 1:02d}.mp4"
        t0 = time.time()
        print(f"Рендер части {i + 1}/{len(script.parts)}: {part.title}")
        dur = render_part(part, spoken, layout, f, stock, seed=i + 1,
                          show_title=(fmt == "long"), max_seconds=budget)
        print(f"  готово: {_fmt(dur)} за {time.time() - t0:.0f} с")
        files.append(f)
        if budget is not None:
            budget -= dur
    final = out_dir / f"{out_dir.name}_{fmt}.mp4"
    if len(files) == 1:
        shutil.copy(files[0], final)
    else:
        concat(files, final)
    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"ГОТОВО: {final}")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description="Хроники сэра Лата — генератор видео")
    ap.add_argument("--notes", help="файл с заметками (notes/*.txt)")
    ap.add_argument("--script", help="готовый размеченный сценарий (пропустить Gemini)")
    ap.add_argument("--format", choices=["long", "short", "both"], default="long")
    ap.add_argument("--minutes", type=float, default=float(os.getenv("SIRLAT_MINUTES", 14)),
                    help="целевая длина длинного видео в минутах (10–30)")
    ap.add_argument("--offline", action="store_true",
                    help="без Gemini/edge-tts/Pixabay: тестовый сценарий и синтетический голос")
    ap.add_argument("--preview", type=float, default=None,
                    help="отрендерить только первые N секунд (для проверки)")
    ap.add_argument("--reuse-script", action="store_true",
                    help="взять уже сгенерированный сценарий из output/, если он есть")
    ap.add_argument("--keep-parts", action="store_true", help="не удалять отдельные части")
    ap.add_argument("--out", default=None, help="папка для результатов")
    args = ap.parse_args(argv)

    if not args.notes and not args.script:
        ap.error("укажите --notes или --script")
    args.minutes = max(10.0, min(30.0, args.minutes))
    src = Path(args.notes or args.script)
    if not src.exists():
        ap.error(f"файл не найден: {src}")
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    out_dir = Path(args.out) if args.out else OUTPUT / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = ["long", "short"] if args.format == "both" else [args.format]
    for fmt in formats:
        build(fmt, notes, out_dir, args)


if __name__ == "__main__":
    sys.exit(main())
